"""Recover a provider-generated image URL without another generation POST.

The observed gateway sometimes emits a Markdown CDN image instead of Gemini
inlineData. Require the actual complete response, one declared public CDN URL,
bounded unauthenticated download, and successful image decode. Never infer an
image from token usage or a textual success claim.
"""
import base64
import hashlib
import io
import re
import time
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler, ProxyHandler

from PIL import Image

from .core import bind, read, require, write, write_bytes

ALLOWED_IMAGE_HOSTS = {"media.deepwl.cn"}
MAX_BYTES = 32 * 1024 * 1024


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def response_image_url(response):
    from data_generation_methods.r2h_pipeline.providers import GeminiClient
    parts = GeminiClient._parts(response)  # actual STOP required
    require(not any(p.get("inlineData") for p in parts), "inline_image_is_not_url_recovery")
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    match = re.fullmatch(r"\s*!\[[^\]]*\]\((https://[^\s)]+)\)\s*", text)
    require(match is not None, "missing_single_markdown_image")
    url = match.group(1)
    parsed = urlsplit(url)
    require(parsed.hostname in ALLOWED_IMAGE_HOSTS and parsed.port in {None, 443}
            and parsed.username is None and parsed.password is None and not parsed.fragment,
            "undeclared_image_media_endpoint")
    return url


def download_image(url):
    # No API credential, cookies, redirects, or generation calls.
    request = Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "image/*"})
    with build_opener(ProxyHandler({}), NoRedirect()).open(request, timeout=60) as response:
        require(response.status == 200, "image_media_http_failure")
        raw = response.read(MAX_BYTES + 1)
    require(0 < len(raw) <= MAX_BYTES, "image_media_size_out_of_bounds")
    return raw


def decode_image(raw):
    require(0 < len(raw) <= MAX_BYTES, "image_media_size_out_of_bounds")
    with Image.open(io.BytesIO(raw)) as picture:
        require(picture.format in {"JPEG", "PNG", "WEBP"}, "unsupported_image_media_format")
        require(0 < picture.width * picture.height <= 20_000_000, "image_media_pixels_out_of_bounds")
        picture.load()
        return {"width": picture.width, "height": picture.height,
                "mime_type": Image.MIME[picture.format], "frames": getattr(picture, "n_frames", 1)}


def recover_image_result(stage, model):
    from data_generation_methods.r2h_pipeline.providers import GeminiClient
    stage = Path(stage)
    source = stage / "provider_response.json"
    response = read(source)
    url = response_image_url(response)
    media = stage / "recovered_provider_image.bin"
    receipt = stage / "image_url_recovery.json"
    if receipt.exists():
        prior = read(receipt)
        require(prior["provider_response"] == bind(source) and prior["generated_media"] == bind(media),
                "recovered_image_evidence_changed")
    started = time.time()
    raw = media.read_bytes() if media.exists() else download_image(url)
    metadata = decode_image(raw)
    require(metadata["frames"] == 1, "generated_anchor_must_be_single_image")
    write_bytes(media, raw)
    if not receipt.exists():
        write(receipt, {"schema": "bound_provider_image_url_recovery_v1",
            "provider_response": bind(source), "generated_media": bind(media),
            "url_sha256": hashlib.sha256(url.encode()).hexdigest(),
            "media_host": urlsplit(url).hostname, "image": metadata,
            "download_and_decode_seconds": time.time() - started,
            "generation_posts": 0, "original_call_count_unchanged": True,
            "credentials_sent_to_media_host": False, "redirects_allowed": False,
            "original_failure": bind(stage / "failure.json") if (stage / "failure.json").exists() else None,
            "implementation": bind(Path(__file__))})
    result = GeminiClient._result(response, model)
    result["images"] = [{"mime_type": metadata["mime_type"], "data": base64.b64encode(raw).decode("ascii")}]
    result["image_url_recovery"] = bind(receipt)
    return result
