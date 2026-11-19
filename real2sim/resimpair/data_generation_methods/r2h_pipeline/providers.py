"""Small, synchronous API adapters. No retries, background polling, or paid calls on import.

Injected transports implement ``(method, url, *, headers, json, timeout) -> dict``.
Only the caller may decide whether to retry. A POST transport failure can mean that
the provider already accepted a billable request; it is deliberately ambiguous.
"""
from __future__ import annotations

import base64
import binascii
import copy
import json
import math
import os
from pathlib import Path
import re
import socket
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class ProviderError(RuntimeError):
    """Sanitized error suitable for persistent job state (never a raw API error)."""

    def __init__(self, code: str, *, retryable: bool = False,
                 submission_unknown: bool = False, http_status: int | None = None):
        super().__init__(f"Provider request failed: {code}")
        self.code = code
        self.retryable = retryable
        self.submission_unknown = submission_unknown
        self.http_status = http_status


def _review_validator(response_schema: dict | None):
    """Compile the supplied schema locally before any billable request."""
    if response_schema is None:
        return None
    if not isinstance(response_schema, dict):
        raise ProviderError("invalid_response_schema")
    # Keep unrelated adapters usable without this review-only dependency.
    try:
        from jsonschema import validators
        from jsonschema.exceptions import SchemaError
        from referencing import Registry
    except ImportError:
        raise ProviderError("schema_validator_unavailable") from None
    try:
        # Schemas must themselves be JSON, including finite numeric values.
        _json_bytes(response_schema)
        validator_class = validators.validator_for(response_schema)
        validator_class.check_schema(response_schema)
        # An empty registry forbids implicit network retrieval of schema refs.
        return validator_class(response_schema, registry=Registry())
    except (SchemaError, TypeError, ValueError, RecursionError, ProviderError):
        raise ProviderError("invalid_response_schema") from None


def parse_review_json(text: str, response_schema: dict | None = None, *,
                      allow_json_fence: bool = False) -> dict:
    """Parse one JSON object and validate it without repairing its contents.

    The optional normalization removes only a single, complete `````json``
    outer fence, with its own opening and closing lines. Original text and the
    normalization label are returned on success. Malformed responses raise a
    sanitized ProviderError; transports remain responsible for raw-response
    retention. No retry or provider call is performed here.
    """
    if not isinstance(allow_json_fence, bool):
        raise ProviderError("invalid_review_parse_options")
    validator = _review_validator(response_schema)
    if not isinstance(text, str):
        raise ProviderError("invalid_review_json")
    payload = text.strip(" \t\r\n")
    normalization = "none"
    if allow_json_fence:
        fenced = re.fullmatch(r"```json\r?\n([\s\S]*)\r?\n```", payload)
        if fenced is not None:
            payload = fenced.group(1)
            normalization = "outer_json_fence"

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Nonfinite JSON number")
        return number

    def invalid_constant(value):
        raise ValueError("Nonstandard JSON constant")

    try:
        review = json.loads(payload, object_pairs_hook=unique_object,
                            parse_float=finite_float, parse_constant=invalid_constant)
    except (ValueError, UnicodeError, RecursionError):
        raise ProviderError("invalid_review_json") from None
    if not isinstance(review, dict):
        raise ProviderError("review_must_be_object")
    if validator is not None:
        try:
            # Use a real JSON Schema validator, not just provider-side hints.
            validator.validate(review)
        except Exception:
            # Schema resolution and validation failures both fail closed;
            # do not leak instance values, schema content, or reference URLs.
            raise ProviderError("review_schema_violation") from None
    return {"review": review, "raw_text": text, "normalization": normalization,
            "parse_status": "valid", "schema_validated": validator is not None}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, RecursionError):
        raise ProviderError("invalid_json_payload") from None


def _urllib_transport(method: str, url: str, *, headers: dict, json: dict | None,
                      timeout: float) -> dict:
    body = None if json is None else _json_bytes(json)
    request = Request(url, data=body, headers=headers, method=method)
    # urllib's default redirect handler can forward authentication headers.
    with build_opener(_NoRedirect()).open(request, timeout=timeout) as response:
        raw = response.read(128 * 1024 * 1024 + 1)
    if len(raw) > 128 * 1024 * 1024:
        raise ProviderError("response_too_large")
    try:
        decoded = __import__("json").loads(raw)
    except (ValueError, UnicodeError):
        raise ProviderError("invalid_json_response") from None
    if not isinstance(decoded, dict):
        raise ProviderError("invalid_json_response")
    return decoded


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}$")
_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,199}$")


def _identifier(value: Any) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ProviderError("explicit_model_required")
    return value


def _base_url(value: Any) -> str:
    if not isinstance(value, str):
        raise ProviderError("invalid_base_url")
    try:
        parts = urlsplit(value)
        hostname = parts.hostname
    except ValueError:
        raise ProviderError("invalid_base_url") from None
    if (parts.scheme != "https" or not hostname or parts.username or
            parts.password or parts.query or parts.fragment):
        raise ProviderError("invalid_base_url")
    return value.rstrip("/")


def _media_url(value: Any, kind: str) -> None:
    if not isinstance(value, str):
        raise ProviderError("invalid_media_source")
    if value.startswith("data:"):
        if kind == "video":
            raise ProviderError("video_requires_http_or_asset")
        match = re.fullmatch(r"data:(image|audio)/[a-zA-Z0-9.+-]+;base64,([A-Za-z0-9+/=\r\n]+)", value)
        if not match or match.group(1) != kind:
            raise ProviderError("invalid_media_source")
        try:
            base64.b64decode(match.group(2), validate=True)
        except (ValueError, binascii.Error):
            raise ProviderError("invalid_media_source") from None
        return
    try:
        parts = urlsplit(value)
    except ValueError:
        raise ProviderError("invalid_media_source") from None
    if parts.scheme == "asset" and parts.netloc and not parts.query and not parts.fragment:
        return
    if parts.scheme in {"http", "https"} and parts.hostname and not parts.username and not parts.password:
        return
    raise ProviderError("video_requires_http_or_asset" if kind == "video" else "invalid_media_source")


class _Client:
    def __init__(self, config: dict, transport: Callable | None = None):
        if not isinstance(config, dict):
            raise ProviderError("invalid_config")
        self.config = dict(config)
        self.model = _identifier(config.get("model"))
        self.base_url = _base_url(config.get("base_url", self.default_base_url))
        self.api_key_env = config.get("api_key_env", self.default_key_env)
        # Native Google defaults remain unchanged. Some explicitly configured
        # gateways expose generateContent with Bearer authentication instead.
        auth_scheme = config.get("auth_scheme", "native")
        if auth_scheme not in {"native", "bearer"}:
            raise ProviderError("invalid_auth_scheme")
        if auth_scheme == "bearer":
            self.key_header, self.key_prefix = "Authorization", "Bearer "
        if not isinstance(self.api_key_env, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env):
            raise ProviderError("invalid_key_environment_name")
        try:
            self.timeout = float(config.get("timeout", 120))
        except (TypeError, ValueError):
            raise ProviderError("invalid_timeout") from None
        if not 0 < self.timeout <= 600:
            raise ProviderError("invalid_timeout")
        self.transport = transport or _urllib_transport

    def _headers(self) -> dict:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ProviderError("credentials_missing")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        headers[self.key_header] = self.key_prefix + key
        return headers

    def _call(self, method: str, url: str, payload: dict | None = None) -> dict:
        headers = self._headers()
        try:
            response = self.transport(method, url, headers=headers, json=payload, timeout=self.timeout)
        except HTTPError as error:
            status = int(error.code)
            ambiguous = method == "POST" and status >= 500
            raise ProviderError("submission_unknown" if ambiguous else "http_error",
                                retryable=not ambiguous and (status == 429 or status >= 500),
                                submission_unknown=ambiguous, http_status=status) from None
        except ProviderError as error:
            # A malformed POST response can follow a successful submission.
            if method == "POST" and error.code in {"invalid_json_response", "response_too_large"}:
                raise ProviderError("submission_unknown", submission_unknown=True) from None
            raise
        except (TimeoutError, socket.timeout, URLError, OSError):
            ambiguous = method == "POST"
            raise ProviderError("submission_unknown" if ambiguous else "transport_error",
                                retryable=not ambiguous, submission_unknown=ambiguous) from None
        except Exception:
            # Custom transports must not expose their URLs, auth headers or bodies.
            ambiguous = method == "POST"
            raise ProviderError("submission_unknown" if ambiguous else "transport_error",
                                retryable=not ambiguous, submission_unknown=ambiguous) from None
        if not isinstance(response, dict):
            raise ProviderError("submission_unknown" if method == "POST" else "invalid_response",
                                submission_unknown=method == "POST")
        return response


class SeedanceClient(_Client):
    default_base_url = "https://ark.cn-beijing.volces.com/api/v3"
    default_key_env = "ARK_API_KEY"
    key_header = "Authorization"
    key_prefix = "Bearer "

    def prepare_submit(self, payload: dict) -> dict:
        """Validate and prepare a credential-free request, without network or env reads."""
        if not isinstance(payload, dict):
            raise ProviderError("invalid_payload")
        body = copy.deepcopy(payload)
        if body.get("model", self.model) != self.model:
            raise ProviderError("model_mismatch")
        body["model"] = self.model
        content = body.get("content")
        if not isinstance(content, list) or not content:
            raise ProviderError("content_required")
        roles = set()
        for item in content:
            if not isinstance(item, dict):
                raise ProviderError("invalid_content")
            kind = item.get("type")
            if kind == "text":
                if not isinstance(item.get("text"), str) or not item["text"].strip():
                    raise ProviderError("invalid_text_content")
                continue
            if kind not in {"image_url", "video_url", "audio_url"}:
                raise ProviderError("unsupported_content_type")
            media = item.get(kind)
            if not isinstance(media, dict):
                raise ProviderError("invalid_media_source")
            _media_url(media.get("url"), kind.split("_")[0])
            allowed = {"image_url": {"first_frame", "last_frame", "reference_image"},
                       "video_url": {"reference_video"}, "audio_url": {"reference_audio"}}
            role = item.get("role")
            if role not in allowed[kind]:
                raise ProviderError("invalid_media_role")
            roles.add(role)
        if "last_frame" in roles and "first_frame" not in roles:
            raise ProviderError("last_frame_requires_first_frame")
        if roles.intersection({"first_frame", "last_frame"}) and roles.intersection(
                {"reference_image", "reference_video", "reference_audio"}):
            raise ProviderError("mixed_frame_and_reference_modes")
        task_type = body.get("omni_reference_task_type", "auto")
        if task_type not in {"auto", "reference", "edit", "extend"}:
            raise ProviderError("invalid_reference_task_type")
        if task_type in {"edit", "extend"}:
            if "reference_video" not in roles:
                raise ProviderError("reference_video_required")
            body.setdefault("ratio", "adaptive")
            if body["ratio"] != "adaptive":
                raise ProviderError("edit_extend_requires_adaptive_ratio")
        if task_type == "edit":
            body.setdefault("duration", -1)
            if body["duration"] != -1 or "frames" in body:
                raise ProviderError("edit_requires_original_duration")
        if "first_frame" in roles and self.model.startswith("doubao-seedance-2-5"):
            body.setdefault("ratio", "adaptive")
            if body["ratio"] != "adaptive":
                raise ProviderError("first_frame_requires_adaptive_ratio")
        if len(_json_bytes(body)) > 64 * 1024 * 1024:
            raise ProviderError("request_too_large")
        return {"method": "POST", "url": self.base_url + "/contents/generations/tasks", "json": body}

    def submit(self, payload: dict) -> dict:
        request = self.prepare_submit(payload)
        response = self._call(request["method"], request["url"], request["json"])
        task_id = response.get("id")
        if not isinstance(task_id, str) or not _TASK_ID.fullmatch(task_id):
            raise ProviderError("submission_unknown", submission_unknown=True)
        return self._normalize(response, task_id, "submitted")

    def poll(self, task_id: str) -> dict:
        """Read task state once. This method does not sleep or retry."""
        if not isinstance(task_id, str) or not _TASK_ID.fullmatch(task_id):
            raise ProviderError("invalid_task_id")
        response = self._call("GET", self.base_url + "/contents/generations/tasks/" + task_id)
        if response.get("id", task_id) != task_id:
            raise ProviderError("task_id_mismatch")
        return self._normalize(response, task_id, "unknown")

    def _normalize(self, response: dict, task_id: str, default_status: str) -> dict:
        provider_status = response.get("status", default_status)
        # EchoFlow's task GET was observed returning "pending" while the
        # already-accepted generation waited. Keep the raw status and map only
        # this known nonterminal alias; unknown states still fail closed.
        status = "queued" if provider_status == "pending" else provider_status
        if not isinstance(status, str) or status not in {"submitted", "queued", "running", "succeeded", "failed", "expired", "cancelled", "unknown"}:
            raise ProviderError("invalid_task_status")
        content = response.get("content") or {}
        if not isinstance(content, dict):
            raise ProviderError("invalid_response")
        output_url = content.get("video_url")
        if output_url is not None:
            try:
                parts = urlsplit(output_url)
                valid = parts.scheme == "https" and bool(parts.hostname) and not parts.username and not parts.password
            except (TypeError, ValueError):
                valid = False
            if not valid:
                raise ProviderError("invalid_output_url")
        if status == "succeeded" and not output_url:
            raise ProviderError("missing_output_url")
        result = {"status": status, "task_id": task_id, "output_url": output_url,
                  "usage": response.get("usage") if isinstance(response.get("usage"), dict) else {},
                  "model": self.model}
        if provider_status != status:
            result["provider_status"] = provider_status
        if status in {"failed", "expired", "cancelled"}:
            # Provider's raw error message can contain submitted signed media URLs.
            result["error_code"] = "provider_task_" + status
        return result


class GeminiClient(_Client):
    default_base_url = "https://generativelanguage.googleapis.com/v1beta"
    default_key_env = "GEMINI_API_KEY"
    key_header = "x-goog-api-key"
    key_prefix = ""
    _VIDEO_MIMES = {".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm",
                    ".mpeg": "video/mpeg", ".mpg": "video/mpeg", ".avi": "video/x-msvideo"}
    _IMAGE_MIMES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}

    def _inline(self, path: Path, kind: str) -> dict:
        path = Path(path)
        mime = (self._VIDEO_MIMES if kind == "video" else self._IMAGE_MIMES).get(path.suffix.lower())
        if mime is None:
            raise ProviderError("unsupported_media_format")
        try:
            # Conservative total request cap, including Base64 and prompt/schema.
            if path.stat().st_size > 15 * 1024 * 1024:
                raise ProviderError("inline_limit_exceeded")
            with path.open("rb") as source:
                raw = source.read(15 * 1024 * 1024 + 1)
        except OSError:
            raise ProviderError("media_unreadable") from None
        if not raw:
            raise ProviderError("empty_media")
        if len(raw) > 15 * 1024 * 1024:
            raise ProviderError("inline_limit_exceeded")
        return {"inlineData": {"mimeType": mime, "data": base64.b64encode(raw).decode("ascii")}}

    def _generate(self, model: str, parts: list[dict], generation_config: dict) -> dict:
        _identifier(model)
        body = {"contents": [{"role": "user", "parts": parts}], "generationConfig": generation_config}
        if len(_json_bytes(body)) > 20 * 1024 * 1024:
            raise ProviderError("inline_limit_exceeded")
        return self._call("POST", self.base_url + "/models/" + model + ":generateContent", body)

    @staticmethod
    def _parts(response: dict) -> list[dict]:
        feedback = response.get("promptFeedback") or {}
        if isinstance(feedback, dict) and feedback.get("blockReason"):
            raise ProviderError("response_blocked")
        candidates = response.get("candidates")
        if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
            raise ProviderError("missing_candidate")
        candidate = candidates[0]
        if candidate.get("finishReason") != "STOP":
            raise ProviderError("incomplete_or_blocked_response")
        content = candidate.get("content") or {}
        parts = content.get("parts") if isinstance(content, dict) else None
        if not isinstance(parts, list) or not parts or any(not isinstance(part, dict) for part in parts):
            raise ProviderError("missing_response_parts")
        return parts

    @staticmethod
    def _result(response: dict, model: str) -> dict:
        return {"status": "succeeded", "task_id": None, "output_url": None,
                "usage": response.get("usageMetadata") if isinstance(response.get("usageMetadata"), dict) else {},
                "model": model,
                "provider_reported_model_version": response.get("modelVersion"),
                "provider_response_id": response.get("responseId")}

    @staticmethod
    def _review_config(prompt: str, response_schema: dict | None,
                       allow_json_fence: bool) -> dict:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ProviderError("prompt_required")
        if not isinstance(allow_json_fence, bool):
            raise ProviderError("invalid_review_parse_options")
        _review_validator(response_schema)
        generation_config = {"responseMimeType": "application/json"}
        if response_schema is not None:
            generation_config["responseJsonSchema"] = copy.deepcopy(response_schema)
        return generation_config

    def _review_result(self, response: dict, response_schema: dict | None,
                       allow_json_fence: bool) -> dict:
        parts = self._parts(response)
        text = "".join(part.get("text", "") for part in parts
                       if not part.get("thought") and isinstance(part.get("text"), str))
        parsed = parse_review_json(text, response_schema, allow_json_fence=allow_json_fence)
        result = self._result(response, self.model)
        result.update(parsed)
        return result

    def review(self, video_path: Path, prompt: str, response_schema: dict | None = None, *,
               allow_json_fence: bool = False) -> dict:
        """Review a video once; reject fenced JSON unless explicitly enabled."""
        generation_config = self._review_config(prompt, response_schema, allow_json_fence)
        response = self._generate(self.model, [self._inline(video_path, "video"), {"text": prompt}],
                                  generation_config)
        return self._review_result(response, response_schema, allow_json_fence)

    def review_frames(self, frames: list[dict], prompt: str,
                      response_schema: dict | None = None, *,
                      allow_json_fence: bool = True) -> dict:
        """Review explicit timestamped images once, without video subsampling.

        Each frame has exactly path/timestamp_seconds/frame_index. Times and
        indices must both strictly increase; input is never reordered. Only
        anonymous timestamps and indices accompany the inline image bytes.
        Frame review accepts a single outer JSON fence by default and records
        that normalization; every schema constraint still applies locally.
        """
        generation_config = self._review_config(prompt, response_schema, allow_json_fence)
        if not isinstance(frames, list) or not frames:
            raise ProviderError("invalid_frames")
        previous_timestamp = -1
        previous_index = -1
        for frame in frames:
            if (not isinstance(frame, dict) or
                    set(frame) != {"path", "timestamp_seconds", "frame_index"}):
                raise ProviderError("invalid_frames")
            timestamp = frame["timestamp_seconds"]
            index = frame["frame_index"]
            try:
                finite_timestamp = math.isfinite(timestamp)
            except (TypeError, ValueError, OverflowError):
                finite_timestamp = False
            if (not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool) or
                    not finite_timestamp or timestamp < 0 or
                    not isinstance(index, int) or isinstance(index, bool) or index < 0 or
                    not isinstance(frame["path"], (str, Path)) or not str(frame["path"])):
                raise ProviderError("invalid_frames")
            if timestamp <= previous_timestamp or index <= previous_index:
                raise ProviderError("invalid_frame_sequence")
            previous_timestamp, previous_index = timestamp, index

        parts = [{"text": prompt}]
        # Bound cumulative inline content while constructing the request, and
        # let _generate enforce the exact 20 MiB serialized request limit.
        serialized_size = len(_json_bytes(parts[0])) + len(_json_bytes(generation_config))
        for frame in frames:
            metadata = {"text": json.dumps({"frame_index": frame["frame_index"],
                                           "timestamp_seconds": frame["timestamp_seconds"]},
                                          ensure_ascii=False, allow_nan=False)}
            inline = self._inline(Path(frame["path"]), "image")
            serialized_size += len(_json_bytes(metadata)) + len(_json_bytes(inline))
            if serialized_size > 20 * 1024 * 1024:
                raise ProviderError("inline_limit_exceeded")
            parts.extend([metadata, inline])
        response = self._generate(self.model, parts, generation_config)
        return self._review_result(response, response_schema, allow_json_fence)

    def generate_image(self, image_path: Path, prompt: str,
                       model: str = "gemini-nano-banana-2.1") -> dict:
        """Generate/edit an anchor image; returns Base64, does not write files."""
        return self.generate_image_with_references(image_path, [], prompt, model=model)

    def generate_image_with_references(self, image_path: Path, reference_images: list[Path],
                                       prompt: str, model: str = "gemini-nano-banana-2.1") -> dict:
        """Edit the first image with explicitly labelled supporting images.

        References are separate image Parts, not a composited canvas. The
        caller must describe their roles and verify the actual edited result.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ProviderError("prompt_required")
        if not isinstance(reference_images, list) or len(reference_images) > 8:
            raise ProviderError("invalid_image_references")
        parts = [self._inline(image_path, "image")]
        for index, path in enumerate(reference_images, 1):
            parts.extend([{"text": f"SUPPORTING REFERENCE IMAGE {index}:"},
                          self._inline(Path(path), "image")])
        parts.append({"text": prompt})
        response = self._generate(model, parts,
                                  {"responseModalities": ["TEXT", "IMAGE"]})
        images = []
        for part in self._parts(response):
            if part.get("thought"):
                continue
            data = part.get("inlineData")
            if data is None:
                continue
            if not isinstance(data, dict) or not str(data.get("mimeType", "")).startswith("image/"):
                raise ProviderError("invalid_image_response")
            try:
                decoded = base64.b64decode(data.get("data", ""), validate=True)
            except (ValueError, TypeError, binascii.Error):
                raise ProviderError("invalid_image_response") from None
            if not decoded:
                raise ProviderError("empty_image_response")
            images.append({"mime_type": data["mimeType"], "data": data["data"]})
        if not images:
            raise ProviderError("missing_image_response")
        result = self._result(response, model)
        result["images"] = images
        return result
