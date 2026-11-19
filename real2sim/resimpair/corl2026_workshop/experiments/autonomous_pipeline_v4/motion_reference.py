"""Bounded motion-reference inputs, immutable media receipts and metered calls.

This candidate changes conditioning only. Existing native and visual acceptance
rules remain separate and unchanged. Provider media hosting is verified bytewise.
"""
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, ProxyHandler

from .billing import MeteredBudget, NoRedirect, snapshot
from .core import bind, checked, digest, read, require, write, write_bytes


class ComparisonBudget(MeteredBudget):
    def call(self, kind, request, operation):
        stage = self.directory(kind, request)
        if not (stage / "call_started.json").exists():
            before = read(self.root / "billing/campaign_before.json")
            current = snapshot(self.root, stage / "credit_admission.json")
            a = before.get("token_quota", {}).get("total_used")
            b = current.get("token_quota", {}).get("total_used")
            require(type(a) in (int, float) and type(b) in (int, float) and b >= a,
                    "observed_credit_required_for_comparison_admission")
            reserve = self.spec["reserve_quota_before_new_call"][kind]
            require(b-a+reserve <= self.spec["observed_quota_stop_limit"], "observed_credit_stop_limit_reached")
            write(stage / "credit_admission_decision.json", {"observed_spent": b-a,
                "reserve": reserve, "limit": self.spec["observed_quota_stop_limit"],
                "provider_final_price_not_guaranteed_by_reserve": True})
        return super().call(kind, request, operation)


def derivative(contract, output):
    """Retain the complete native interval, uniformly retimed to task duration."""
    from .media import packet
    output = Path(output)
    receipt = output.with_suffix(".json")
    if receipt.exists():
        old = read(receipt)
        checked(old["video"])
        require(old["native_video"] == contract["simulation_video"], "reference_native_video_changed")
        return old
    require(not output.exists(), "partial_reference_derivative_needs_diagnosis")
    native = checked(contract["simulation_video"])
    info = packet(native, output.parent / "native_packet", frame_budget=9)
    duration = contract["generation_information"]["duration_seconds"]
    count = int(round(duration*24))
    require(4 <= duration <= 30 and count > 1, "invalid_reference_duration")
    factor = ((count-1)/24) / info["all_native_pts_seconds"][-1]
    output.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-i", str(native),
        "-an", "-vf", f"setpts=(PTS-STARTPTS)*{factor:.15g},fps=24,scale=720:720,tpad=stop_mode=clone:stop_duration=0.1",
        "-frames:v", str(count), "-c:v", "libx264", "-threads", "2", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    start = time.time()
    subprocess.run(command, check=True, capture_output=True, timeout=120)
    decoded = packet(output, output.parent / "reference_packet", frame_budget=9)
    require(decoded["decoded_frames"] == count and abs(decoded["duration_seconds"]-duration) < .001,
            "reference_derivative_clock_mismatch")
    value = {"schema": "whole_native_motion_reference_v1", "video": bind(output),
        "native_video": contract["simulation_video"], "native_execution": contract["execution"],
        "native_packet": bind(output.parent / "native_packet/packet.json"),
        "reference_packet": bind(output.parent / "reference_packet/packet.json"),
        "transformation": "uniform full-interval retiming and resizing, no crop or motion editing",
        "source_pts_interval": [0, info["all_native_pts_seconds"][-1]], "time_scale": factor,
        "output_duration_seconds": duration, "output_frames": count, "native_states_unchanged": True,
        "elapsed_seconds": time.time()-start, "source_joint_paths_are_not_mandatory_human_paths": True}
    write(receipt, value)
    return value


def upload_reference(root, derived):
    """Single no-auth upload through the actual provider UI's public media route.

    Only the selected synthetic MP4 is sent. API credentials are never forwarded.
    No retry after an ambiguous upload; returned media must match local SHA256.
    """
    root = Path(root)
    source = checked(derived["video"])
    folder = root / "uploads" / derived["video"]["sha256"]
    final = folder / "receipt.json"
    if final.exists():
        saved = read(final);checked(saved["source"]);checked(saved["private_locator"])
        return saved
    require(not (folder / "upload_started.json").exists(), "previous_media_upload_unresolved_no_resubmit")
    policy_path = root / "human_runtime.json"
    policy_binding = None
    if policy_path.exists():
        policy = read(policy_path)
        require(policy['schema'] == 'resimpair_human_runtime_v1'
                and policy['budget'] == bind(root / 'budget.json'), 'human_runtime_policy_budget_mismatch')
        limit = policy['media_upload_limit']
        require(type(limit) is int and 0 <= limit <= read(root / 'budget.json')['new_call_limits']['video'],
                'invalid_bounded_media_upload_limit')
        public_js = checked(policy['provider_application'])
        policy_binding = bind(policy_path)
    else:
        limit = read(root / "experiment.json")["media_upload_limit"]
        public_js = root / "docs/provider_app.js"
    require(len(list((root / "uploads").glob("*/upload_started.json"))) < limit,
            "media_upload_limit_exhausted")
    endpoint = "https://imageproxy.zhongzhuan.chat/api/upload"
    require(endpoint in public_js.read_text(), "upload_route_not_in_actual_provider_application")
    body = source.read_bytes()
    require(0 < len(body) < 32 * 1024**2, "invalid_reference_upload_size")
    boundary = "resimpair" + derived["video"]["sha256"][:32]
    data = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="reference.mp4"\r\n'
            'Content-Type: video/mp4\r\n\r\n').encode() + body + f"\r\n--{boundary}--\r\n".encode()
    write(folder / "upload_request.json", {"source": bind(source), "endpoint": endpoint,
        "provider_application": bind(public_js), "content_type": "video/mp4", "api_credentials_sent": False,
        "runtime_policy": policy_binding})
    snapshot(root, folder / "credit_before.json")
    write(folder / "upload_started.json", {"started_unix": time.time(), "request": bind(folder / "upload_request.json"),
        "single_post": True, "automatic_retry": False})
    try:
        request = Request(endpoint, data=data, method="POST", headers={"User-Agent": "Mozilla/5.0",
            "Content-Type": "multipart/form-data; boundary="+boundary})
        with build_opener(ProxyHandler({}), NoRedirect()).open(request, timeout=90) as response:
            raw = response.read(1048577)
        require(len(raw) <= 1048576, "media_upload_response_too_large")
        value = json.loads(raw)
        url = value.get("url")
        parts = urlsplit(url or "")
        require(parts.scheme == "https" and parts.hostname and not parts.username and not parts.password,
                "media_host_did_not_return_https_url")
        # Locator may be a capability URL: save privately; logs contain only host/hash.
        private = root / ".remote_media"
        private.mkdir(mode=0o700, exist_ok=True);os.chmod(private, 0o700)
        locator = private / (derived["video"]["sha256"] + ".json")
        write(locator, {"url": url, "source": bind(source)});os.chmod(locator, 0o600)
        with build_opener(ProxyHandler({}), NoRedirect()).open(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60) as response:
            downloaded = response.read(32*1024**2+1)
        require(hashlib.sha256(downloaded).hexdigest() == derived["video"]["sha256"], "remote_reference_bytes_differ")
        saved = {"schema": "byte_verified_provider_media_upload_v1", "source": bind(source),
            "private_locator": bind(locator), "host": parts.hostname, "response_sha256": hashlib.sha256(raw).hexdigest(),
            "remote_sha256_matches_local": True, "credentials_sent_to_media_host": False,
            "completed_unix": time.time(), "automatic_retry": False}
        write(final, saved)
        return saved
    except Exception as exc:
        write(folder / "failure.json", {"exception_type": type(exc).__name__, "http_status": getattr(exc, "code", None),
            "raw_error_saved": False, "automatic_retry": False, "finished_unix": time.time()})
        raise RuntimeError("reference_media_upload_failed_inspect_safe_receipt") from None
    finally:
        snapshot(root, folder / "credit_after.json")


REFERENCE_INSTRUCTION = (
    "Reference routing: @image1 supplies the single human operator, appearance, initial object layout and fixed camera. "
    "@video1 is a robot simulation of the SAME task and supplies the object motion, contact order and functional path. "
    "Generate a new human demonstration beginning in @image1's state. Follow the declared task events and anatomical "
    "hand identities while using natural human reaches. The two robot arms correspond to the two arms of ONE person; "
    "no robot, mechanical gripper, additional person or extra arm may appear. The simulation is process evidence, "
    "not an instruction to copy robot joint angles or pauses. Preserve all task objects and the declared ending hand states.\n"
)


def reference_payload(config, request):
    from .models import image_uri
    require(request["method"] == "motion_reference", "wrong_reference_request_method")
    receipt = read(checked(request["remote_reference"]))
    require(receipt["remote_sha256_matches_local"] and receipt["source"] == request["reference_video"],
            "reference_upload_binding_mismatch")
    locator = read(checked(receipt["private_locator"]))
    require(locator["source"] == request["reference_video"], "remote_locator_source_mismatch")
    checked(request["reference_video"])
    require(urlsplit(locator["url"]).scheme == "https", "reference_video_requires_https_locator")
    return {**request["settings"], "model": config["seedance"]["model"], "omni_reference_task_type": "reference",
        "content": [{"type": "text", "text": request["prompt"]},
            {"type": "image_url", "role": "reference_image", "image_url": {"url": image_uri(checked(request["human_image"]))}},
            {"type": "video_url", "role": "reference_video", "video_url": {"url": locator["url"]}}]}


def submit_reference(root, budget, request_path):
    from data_generation_methods.r2h_pipeline.providers import SeedanceClient
    from corl2026_workshop.experiments.task_contract_v3.direct_transport import call
    from .models import configured, safe_error_diagnostic
    from urllib.error import HTTPError
    root = Path(root);request = read(request_path);config = configured(root)
    require(request["model_config"] == bind(root / "config.json"), "reference_model_configuration_changed")
    payload = reference_payload(config, request)
    SeedanceClient(config["seedance"]).prepare_submit(payload)
    def operation(stage):
        def audited(method, url, **kwargs):
            require(kwargs["json"] == payload, "reference_payload_changed")
            write(stage / "actual_input_audit.json", {"request": bind(request_path), "payload_sha256": digest(payload),
                "roles": [c.get("role") for c in payload["content"] if c["type"] != "text"],
                "human_image": request["human_image"], "native_reference": request["reference_video"],
                "remote_reference": request["remote_reference"], "payload_verified": True,
                "text_sha256": hashlib.sha256(request["prompt"].encode()).hexdigest(),
                "mode": "explicit_reference_image_plus_reference_video", "automatic_fallback": False})
            kwargs["headers"] = {**kwargs["headers"], "User-Agent": "Mozilla/5.0", "Connection": "close"}
            try:
                raw = call(method, url, **kwargs)
            except HTTPError as exc:
                write(stage / "provider_error_diagnostic.json", safe_error_diagnostic(exc.read(65537), exc.code))
                raise
            write(stage / "provider_response.json", raw)
            return raw
        client = SeedanceClient(dict(config["seedance"], timeout=budget.spec["timeout_seconds"]), transport=audited)
        return client.submit(payload)
    result, stage = budget.call("video", request_path, operation)
    return {"task_id": result["task_id"], "request": bind(request_path), "stage": str(stage),
            "submission": bind(stage / "result.json")}
