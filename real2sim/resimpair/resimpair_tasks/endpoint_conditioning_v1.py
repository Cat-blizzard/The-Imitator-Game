"""Candidate endpoint conditioning; existing default and acceptance stay unchanged.

The caller supplies one camera-scoped task prompt and independently checked
condition evidence. This module binds bytes and roles, not visual correctness.
It reuses the production upload receipts, provider client, durable budget and
video polling interface. No upload or generation occurs while building a request.
"""
from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import io
from pathlib import Path
from urllib.parse import urlsplit

from corl2026_workshop.experiments.autonomous_pipeline_v4.core import (
    bind, checked, digest, read, require, write,
)
from corl2026_workshop.experiments.autonomous_pipeline_v4.motion_reference import REFERENCE_INSTRUCTION


SCHEMA = "resimpair_endpoint_conditioning_v1"
METHODS = {"motion_reference", "endpoint_reference", "endpoint_frames"}
ENDPOINT_REFERENCE_ADDITION = (
    "Additional endpoint reference: @image2 shows the SAME person and scene at task completion. "
    "Finish matching @image2; do not merge initial and terminal object positions into duplicate objects.\n"
)
# A/B differ only in this endpoint-role explanation and the terminal reference
# image. Preserve every word of the original routing instruction for attribution.
ENDPOINT_REFERENCE_INSTRUCTION = REFERENCE_INSTRUCTION + ENDPOINT_REFERENCE_ADDITION
ENDPOINT_FRAMES_INSTRUCTION = (
    "Endpoint routing: the first-frame image is the INITIAL human task state and the last-frame image is the "
    "TERMINAL human task state. They depict the SAME single person, fixed camera, scene and objects at two times. "
    "Connect these endpoints with the complete declared task events in their required order. Follow the declared "
    "anatomical hand assignments. Preserve object identity, number and relative size; do not blend initial and final "
    "positions into duplicates or move objects without the required contact. Use natural human motion. "
    "No robot, mechanical gripper, extra person or extra limb may appear.\n"
)
INSTRUCTIONS = {
    "motion_reference": REFERENCE_INSTRUCTION,
    "endpoint_reference": ENDPOINT_REFERENCE_INSTRUCTION,
    "endpoint_frames": ENDPOINT_FRAMES_INSTRUCTION,
}
DEFAULT_SETTINGS = {
    "duration": 8, "resolution": "720p", "ratio": "adaptive",
    "generate_audio": False, "watermark": False,
}


def _binding(value):
    return deepcopy(value) if isinstance(value, dict) else bind(value)


def _evidence(value):
    require(isinstance(value, dict) and "anchor_verdict" in value,
            "endpoint_condition_evidence_requires_anchor_verdict_binding")
    answer = {name: _binding(item) for name, item in value.items()}
    for item in answer.values():
        checked(item)
    return answer


def _png_uri(binding):
    """Check and serialize the same bytes; do not silently relabel JPEG as PNG."""
    from PIL import Image
    path = checked(binding)
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == binding["sha256"], "endpoint_image_changed_during_read")
    require(0 < len(raw) < 30 * 1024**2, "endpoint_image_size_outside_provider_limit")
    with Image.open(io.BytesIO(raw)) as image:
        require(image.format == "PNG", "endpoint_candidate_requires_png_image")
        image.verify()
    return "data:image/png;base64," + base64.b64encode(raw).decode()


def _remote_url(request):
    receipt = read(checked(request["remote_reference"]))
    require(receipt.get("remote_sha256_matches_local") is True
            and receipt["source"] == request["reference_video"], "endpoint_upload_binding_mismatch")
    locator = read(checked(receipt["private_locator"]))
    require(locator["source"] == request["reference_video"], "endpoint_locator_source_mismatch")
    checked(request["reference_video"])
    parsed = urlsplit(locator["url"])
    require(parsed.scheme == "https" and parsed.hostname and not parsed.username and not parsed.password,
            "endpoint_video_requires_verified_https_locator")
    return locator["url"]


def implementation():
    """Bind the actual candidate and reused adapters; no frozen release is edited."""
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import models, motion_reference
    from data_generation_methods.r2h_pipeline import providers
    return {"endpoint_conditioning_v1.py": bind(__file__), "models.py": bind(models.__file__),
            "motion_reference.py": bind(motion_reference.__file__), "providers.py": bind(providers.__file__)}


def build_request(root, method, *, contract, task_prompt, initial_image, terminal_image=None,
                  reference_video=None, remote_reference=None, settings=None, condition_evidence=None):
    """Create a bound request for A/B/C. Caller writes it with immutable ``write``.

    ``condition_evidence`` must contain an ``anchor_verdict`` file binding. Its
    semantic verdict is owned by the independent evaluator, not this generator.
    Use the same ``task_prompt`` (including explicit hand mapping) across arms.
    ``reference_video`` and ``remote_reference`` use existing ``derivative`` and
    ``upload_reference`` outputs; this function never uploads or submits.
    """
    require(method in METHODS, "unknown_endpoint_method")
    require(isinstance(task_prompt, str) and task_prompt.strip(), "empty_endpoint_task_prompt")
    root = Path(root)
    request = {
        "schema": SCHEMA, "method": method, "contract": _binding(contract),
        "model_config": bind(root / "config.json"), "implementation": implementation(),
        "settings": {**DEFAULT_SETTINGS, **deepcopy(settings or {})},
        "task_prompt": task_prompt, "prompt": INSTRUCTIONS[method] + task_prompt,
        "prompt_components": {"routing": INSTRUCTIONS[method], "task": task_prompt},
        "human_image": _binding(initial_image),
        "condition_evidence": _evidence(condition_evidence),
        "endpoint_semantics": {"human_image": "task_initial_state"},
        "automatic_fallback": False, "candidate_not_promoted": True,
    }
    if terminal_image is not None:
        request["terminal_image"] = _binding(terminal_image)
        request["endpoint_semantics"]["terminal_image"] = "task_terminal_state"
    if reference_video is not None:
        request["reference_video"] = _binding(reference_video)
    if remote_reference is not None:
        request["remote_reference"] = _binding(remote_reference)
    payload(read(root / "config.json"), request)
    return request


def payload(config, request):
    """Serialize explicit roles, rejecting mixed modes and changed evidence."""
    require(request.get("schema") == SCHEMA and request.get("method") in METHODS,
            "invalid_endpoint_request_schema_or_method")
    method = request["method"]
    require(request.get("automatic_fallback") is False, "endpoint_fallback_not_allowed")
    require(request["prompt"] == INSTRUCTIONS[method] + request["task_prompt"]
            and request["prompt_components"] == {"routing": INSTRUCTIONS[method], "task": request["task_prompt"]},
            "endpoint_prompt_components_changed")
    checked(request["contract"])
    require(request["condition_evidence"] == _evidence(request["condition_evidence"]),
            "endpoint_condition_evidence_must_keep_sha_bindings")
    for item in request["implementation"].values():
        checked(item)
    require(request["implementation"] == implementation(), "endpoint_imported_implementation_changed")
    settings = request["settings"]
    require(not set(settings).intersection({"model", "content", "first_frame", "last_frame",
            "reference_image", "reference_video", "omni_reference_task_type"}), "endpoint_settings_override_roles")
    require(type(settings.get("duration")) is int and 4 <= settings["duration"] <= 30,
            "invalid_endpoint_duration")
    require(not any(k in request for k in ["first_frame", "last_frame", "content"]),
            "endpoint_mixed_request_roles")
    need_terminal = method != "motion_reference"
    require(("terminal_image" in request) == need_terminal, "endpoint_terminal_presence_mismatch")
    expected_semantics = {"human_image": "task_initial_state"}
    if need_terminal:
        expected_semantics["terminal_image"] = "task_terminal_state"
    require(request["endpoint_semantics"] == expected_semantics, "endpoint_frame_meaning_changed")
    content = [{"type": "text", "text": request["prompt"]},
               {"type": "image_url", "role": "first_frame" if method == "endpoint_frames" else "reference_image",
                "image_url": {"url": _png_uri(request["human_image"])}}]
    if need_terminal:
        content.append({"type": "image_url", "role": "last_frame" if method == "endpoint_frames" else "reference_image",
                        "image_url": {"url": _png_uri(request["terminal_image"])}})
    body = {**settings, "model": config["seedance"]["model"], "content": content}
    if method == "endpoint_frames":
        require("reference_video" not in request and "remote_reference" not in request,
                "endpoint_frames_cannot_include_reference_video")
        require(settings.get("ratio") == "adaptive", "endpoint_frames_require_adaptive_ratio")
    else:
        require("reference_video" in request and "remote_reference" in request,
                "endpoint_reference_requires_motion_receipt")
        content.append({"type": "video_url", "role": "reference_video", "video_url": {"url": _remote_url(request)}})
        body["omni_reference_task_type"] = "reference"
    return body


def submit(root, budget, request_path):
    """Metered single POST. Resume via the existing budget and ``poll_video``."""
    from urllib.error import HTTPError
    from data_generation_methods.r2h_pipeline.providers import SeedanceClient
    from corl2026_workshop.experiments.task_contract_v3.direct_transport import call
    from corl2026_workshop.experiments.autonomous_pipeline_v4.models import configured, safe_error_diagnostic
    root = Path(root)
    request = read(request_path)
    config = configured(root)
    require(request["model_config"] == bind(root / "config.json"), "endpoint_model_config_changed")
    body = payload(config, request)
    SeedanceClient(config["seedance"]).prepare_submit(body)

    def operation(stage):
        def audited(method, url, **kwargs):
            require(kwargs["json"] == body, "endpoint_payload_changed_before_post")
            write(stage / "actual_input_audit.json", {
                "request": bind(request_path), "payload_sha256": digest(body),
                "roles": [part.get("role") for part in body["content"] if part["type"] != "text"],
                "human_image": request["human_image"], "terminal_image": request.get("terminal_image"),
                "native_reference": request.get("reference_video"), "remote_reference": request.get("remote_reference"),
                "condition_evidence": request["condition_evidence"], "endpoint_semantics": request["endpoint_semantics"],
                "text_sha256": hashlib.sha256(request["prompt"].encode()).hexdigest(),
                "common_task_text_sha256": hashlib.sha256(request["task_prompt"].encode()).hexdigest(),
                "mode": request["method"], "payload_verified": True,
                "automatic_fallback": False, "visual_quality_inferred_by_payload_guard": False,
            })
            kwargs["headers"] = {**kwargs["headers"], "User-Agent": "Mozilla/5.0", "Connection": "close"}
            try:
                result = call(method, url, **kwargs)
            except HTTPError as exc:
                write(stage / "provider_error_diagnostic.json", safe_error_diagnostic(exc.read(65537), exc.code))
                raise
            write(stage / "provider_response.json", result)
            return result
        client = SeedanceClient(dict(config["seedance"], timeout=budget.spec["timeout_seconds"]), transport=audited)
        return client.submit(body)

    result, stage = budget.call("video", request_path, operation)
    return {"task_id": result["task_id"], "request": bind(request_path), "stage": str(stage),
            "submission": bind(stage / "result.json")}


def build_terminal_image_request(root, *, human_initial, simulation_initial, simulation_terminal,
                                 task_delta_prompt, condition_evidence):
    """Input for existing ``models.generate_image``; no generation here.

    Reference order is human initial, simulation initial, simulation terminal.
    ``task_delta_prompt`` must describe only task-relevant observed changes.
    The returned image needs a new independent endpoint review before video use.
    """
    require(isinstance(task_delta_prompt, str) and task_delta_prompt.strip(), "empty_terminal_delta_prompt")
    images = [_binding(p) for p in [human_initial, simulation_initial, simulation_terminal]]
    for item in images:
        _png_uri(item)
    prompt = (
        "Create only the TERMINAL human task image, by editing Image 1. Image 1 is the approved initial human scene. "
        "Image 2 is the corresponding INITIAL robot-simulation scene. Image 3 is the TERMINAL simulation scene. "
        "Use the state difference between Image 2 and Image 3 to make only the declared task changes to Image 1. "
        "Keep exactly the same single human operator, clothing, anatomical arm ownership, camera, background, "
        "lighting and unchanged objects as Image 1. Preserve every task object's identity, size and count. "
        "Translate robot actions into the declared human hand assignment; do not copy robot arms or add a second person. "
        "Do not reveal off-camera objects or invent furniture. Do not copy the simulation's robot appearance. "
        "The terminal contact and release state must follow the task description below. Output one image.\n"
        + task_delta_prompt
    )
    return {"schema": SCHEMA + "/terminal_image", "model_config": bind(Path(root) / "config.json"),
            "implementation": implementation(), "images": images, "prompt": prompt,
            "image_meanings": ["human_initial", "simulation_initial", "simulation_terminal"],
            "task_delta_prompt": task_delta_prompt, "condition_evidence": _evidence(condition_evidence),
            "automatic_fallback": False, "requires_independent_terminal_review_before_video": True}
