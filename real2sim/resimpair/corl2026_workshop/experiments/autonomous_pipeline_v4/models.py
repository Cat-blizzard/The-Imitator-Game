"""Actual provider adapters, media-byte receipts, and recoverable video IDs."""
from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path
import time

from .core import bind, checked, digest, read, replace_state, require, write, write_bytes
from .timing import span


def safe_error_diagnostic(raw, http_status):
    """Classify standard or gateway error envelopes without retaining their text."""
    import json
    import re
    try:
        value = json.loads(raw) if len(raw) <= 65536 else None
    except (ValueError, UnicodeDecodeError):
        value = None
    # Gateways also use a top-level message or an error string/list. Unknown
    # fields are not emitted; only fixed categories and a digest leave here.
    strings = []
    def visit(item, depth=0):
        if depth > 6:
            return
        if isinstance(item, str):
            strings.append(item)
        elif isinstance(item, dict):
            for child in list(item.values())[:32]:
                visit(child, depth + 1)
        elif isinstance(item, list):
            for row in item[:32]:
                visit(row, depth + 1)
    visit(value)
    message = " ".join(strings).lower()
    categories = {
        "schema": ("schema", "模式"), "complexity": ("too many", "complex", "复杂"),
        "invalid_json": ("invalid json", "json parse", "json格式"),
        "unsupported": ("unsupported", "not supported", "不支持"),
        "quota": ("quota", "余额", "配额", "insufficient balance", "credit", "payment"),
        "permission": ("permission", "forbidden", "无权限"),
        "credentials": ("api key", "api_key", "unauthorized", "认证", "令牌"),
        "rate_limit": ("rate limit", "too many requests", "限流"),
        "model_route": ("model", "channel", "upstream", "模型", "渠道", "上游"),
        "request_size": ("too large", "payload", "size limit", "请求过大"),
        "media": ("image", "video", "media", "图片", "视频"),
        "stream": ("streamgeneratecontent", "streaming", "流式"),
        "cardinality_limit": ("maximum number", "max number", "at most", "too many", "最多", "数量限制"),
    }
    status = value.get("error", {}).get("status") if isinstance(value, dict) and isinstance(value.get("error"), dict) else None
    return {"http_status": http_status,
            "provider_status": status if isinstance(status, str) and re.fullmatch(r"[A-Z_]{1,64}", status) else None,
            "reason_flags": {key: any(term in message for term in terms) for key, terms in categories.items()},
            "parameter_mentions": [name for name in ("responseJsonSchema", "responseSchema", "maxItems", "minItems",
                "uniqueItems", "max_output_tokens", "inlineData", "images", "frame_census") if name.lower() in message],
            "error_envelope": type(value).__name__, "recognized_message_fields": bool(strings),
            "error_body_bytes_read": len(raw), "error_body_sha256": hashlib.sha256(raw).hexdigest(),
            "raw_body_or_message_saved": False, "automatic_retry": False}


def configured(root):
    from data_generation_methods.r2h_pipeline.common import load_env_file
    config = read(Path(root) / "config.json")
    from resimpair_runtime import local_path
    credentials = local_path(config["credentials_file"])
    if credentials.is_file():          # optional: keys may come from the environment (ECHOFLOW_API_KEY, ...)
        load_env_file(credentials)
    return config


def diagnostic_client(config, stage, *, transport_kind="stream"):
    """Preserve safe error categories; never log credentials or raw error URLs."""
    from urllib.error import HTTPError
    from corl2026_workshop.experiments.task_contract_v3.stream_transport_direct import client
    require(transport_kind in {"stream", "generate"}, "Unknown declared provider transport")
    if transport_kind == "generate":
        from data_generation_methods.r2h_pipeline.providers import GeminiClient
        from corl2026_workshop.experiments.task_contract_v3.direct_transport import call
        def direct(method, url, **kwargs):
            kwargs["headers"] = {**kwargs["headers"], "User-Agent": "Mozilla/5.0", "Connection": "close"}
            started = time.time()
            result = call(method, url, **kwargs)
            write(stage / "provider_response.json", result)
            write(stage / "transport_receipt.json", {"mode": "generateContent", "single_post": True,
                  "elapsed_seconds": time.time() - started, "automatic_retry": False, "fallback": False})
            return result
        c = GeminiClient(config, transport=direct)
    else:
        c = client(config, stage)
    send = c.transport
    def transport(method, url, **kwargs):
        try:
            return send(method, url, **kwargs)
        except HTTPError as exc:
            try:
                raw = exc.read(65537)
            except Exception:
                raw = b""
            write(stage / "provider_error_diagnostic.json", safe_error_diagnostic(raw, exc.code))
            raise
    c.transport = transport
    return c


def reviewed_json(root, budget, request_path):
    from jsonschema import Draft202012Validator
    from .review_transport import generation_config, wire_schema
    config = configured(root)
    request = read(request_path)
    if "model_config" in request:
        require(request["model_config"] == bind(Path(root) / "config.json"), "Model configuration changed after request")
    for item in request["images"]:
        checked(item["image"])
    Draft202012Validator.check_schema(request["response_schema"])
    adapter = request.get("wire_schema_adapter", config['gemini'].get('wire_schema_adapter'))
    generation = generation_config(config, request["response_schema"], wire_schema_adapter=adapter)
    _, schema_encoding = wire_schema(request["response_schema"], adapter)
    def operation(stage):
        # The nonstreaming review billed successfully but its response was lost
        # in human_r2. New, unstarted calls use the existing durable SSE adapter;
        # no fallback POST, no schema/check reduction, and no lost-call retry.
        c = diagnostic_client(dict(config["gemini"], timeout=budget.spec["timeout_seconds"]), stage,
                              transport_kind="stream")
        parts = [{"text": request["prompt"]}]
        for item in request["images"]:
            parts += [{"text": item["label"]}, c._inline(checked(item["image"]), "image")]
        send = c.transport
        def audited_transport(method, url, **kwargs):
            body = kwargs["json"]
            require(body["generationConfig"] == generation
                    and body["contents"] == [{"role": "user", "parts": parts}],
                    "review_actual_payload_differs_from_bound_inputs")
            write(stage / "wire_config_audit.json", {"request": bind(request_path),
                "generation_config": generation, "payload_sha256": digest(body), "verified": True,
                "schema_encoding": schema_encoding,
                "automatic_protocol_fallback": False})
            return send(method, url, **kwargs)
        c.transport = audited_transport
        write(stage / "actual_input_audit.json", {
            "request": bind(request_path), "prompt_sha256": hashlib.sha256(request["prompt"].encode()).hexdigest(),
            "images": [bind(checked(i["image"])) for i in request["images"]],
            "image_bytes_checked": True, "response_schema": request["response_schema"],
            "wire_schema_adapter": adapter, "schema_encoding": schema_encoding,
            "automatic_visual_inference": True, "manual_observation_supplied": False,
            "transport_kind": "streamGenerateContent", "automatic_transport_fallback": False,
        })
        response = c._generate(c.model, parts, generation)
        return c._review_result(response, request["response_schema"], True)
    result, stage = budget.call("review", request_path, operation)
    return result["review"], {"request": bind(request_path), "result": bind(stage / "result.json"),
                              "raw_response": bind(stage / "provider_response.json"),
                              "actual_inputs": bind(stage / "actual_input_audit.json")}


def generate_image(root, budget, request_path, output):
    from PIL import Image
    request = read(request_path)
    config = configured(root)
    require(request["model_config"] == bind(Path(root) / "config.json"), "Model configuration changed after request")
    def operation(stage):
        c = diagnostic_client(dict(config["gemini"], timeout=budget.spec["timeout_seconds"]), stage)
        paths = [checked(item) for item in request["images"]]
        send = c.transport
        def audited(method, url, **kwargs):
            body = kwargs["json"]
            images = [hashlib.sha256(base64.b64decode(p["inlineData"]["data"], validate=True)).hexdigest()
                      for p in body["contents"][0]["parts"] if "inlineData" in p]
            require(images == [b["sha256"] for b in request["images"]], "Image inputs differ from request")
            write(stage / "actual_input_audit.json", {"request": bind(request_path), "image_sha256": images,
                  "prompt_sha256": hashlib.sha256(request["prompt"].encode()).hexdigest(), "bytes_verified": True})
            return send(method, url, **kwargs)
        c.transport = audited
        try:
            return c.generate_image_with_references(paths[0], paths[1:], request["prompt"], model=config["anchor_model"])
        except Exception as exc:
            if getattr(exc, "code", None) != "missing_image_response" or not (stage / "provider_response.json").exists():
                raise
            from .image_response import recover_image_result
            return recover_image_result(stage, config["anchor_model"])
    result, stage = budget.call("image", request_path, operation)
    with Image.open(io.BytesIO(base64.b64decode(result["images"][0]["data"], validate=True))) as image:
        image.load()
        raw = io.BytesIO()
        image.convert("RGB").save(raw, format="PNG")
    write_bytes(output, raw.getvalue())
    receipt = {"image": bind(output), "request": bind(request_path), "result": bind(stage / "result.json"),
               "actual_inputs": bind(stage / "actual_input_audit.json")}
    write(Path(output).with_suffix(".provenance.json"), receipt)
    return receipt


def image_uri(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def submit_video(root, budget, request_path):
    from data_generation_methods.r2h_pipeline.providers import SeedanceClient
    from corl2026_workshop.experiments.task_contract_v3.direct_transport import call
    config = configured(root)
    request = read(request_path)
    require(request["model_config"] == bind(Path(root) / "config.json"), "Model configuration changed after request")
    payload = {**request["settings"], "model": config["seedance"]["model"],
               "content": [{"type": "text", "text": request["prompt"]},
                           {"type": "image_url", "role": "first_frame",
                            "image_url": {"url": image_uri(checked(request["first_frame"]))}}]}
    def operation(stage):
        def audited(method, url, **kwargs):
            require(kwargs["json"] == payload, "Video inputs differ from compiled request")
            write(stage / "actual_input_audit.json", {"request": bind(request_path),
                  "payload_sha256": digest(payload), "first_frame": request["first_frame"],
                  "text_sha256": hashlib.sha256(request["prompt"].encode()).hexdigest(),
                  "payload_verified": True, "mode": "predeclared_first_frame_and_task_contract"})
            kwargs["headers"] = {**kwargs["headers"], "User-Agent": "Mozilla/5.0", "Connection": "close"}
            raw = call(method, url, **kwargs)
            # Persist provider task ID before normalization can fail.
            write(stage / "provider_response.json", raw)
            return raw
        c = SeedanceClient(dict(config["seedance"], timeout=budget.spec["timeout_seconds"]), transport=audited)
        return c.submit(payload)
    result, stage = budget.call("video", request_path, operation)
    return {"task_id": result["task_id"], "request": bind(request_path),
            "stage": str(stage), "submission": bind(stage / "result.json")}


def poll_video(root, submission, output):
    """One read-only poll per invocation; a polling timeout is never a new POST."""
    from data_generation_methods.r2h_pipeline.providers import SeedanceClient
    from corl2026_workshop.experiments.task_contract_v3.direct_transport import call
    from urllib.request import Request, build_opener, ProxyHandler
    from data_generation_methods.r2h_pipeline.providers import _NoRedirect
    config = configured(root)
    stage = Path(submission["stage"])
    saved = stage / "video_terminal.json"
    if saved.exists():
        state = read(saved)
    else:
        try:
            with span(stage, "video_poll"):
                state = SeedanceClient(config["seedance"], transport=call).poll(submission["task_id"])
        except Exception as exc:
            replace_state(stage / "poll_observation.json", {"status": "poll_transport_error",
                          "task_id": submission["task_id"], "code": getattr(exc, "code", "poll_error"),
                          "time": time.time(), "resubmit": False})
            return {"status": "poll_transport_error", "task_id": submission["task_id"]}
        replace_state(stage / "poll_state.json", state)
        if state["status"] in {"succeeded", "failed", "expired", "cancelled"}:
            write(saved, state)
            write(stage / "video_first_terminal_observation.json", {
                "observed_unix": time.time(), "status": state["status"],
                "scope": "first local terminal observation; provider queue/compute split unknown"})
    if state["status"] != "succeeded":
        return {"status": state["status"], "task_id": submission["task_id"]}
    path = Path(output)
    if not path.exists():
        with span(stage, "video_download") as timing:
            request = Request(state["output_url"], headers={"User-Agent": "Mozilla/5.0"})
            with build_opener(ProxyHandler({}), _NoRedirect()).open(request, timeout=180) as response:
                raw = response.read(128 * 1024 * 1024 + 1)
            require(0 < len(raw) <= 128 * 1024 * 1024, "Invalid downloaded video size")
            write_bytes(path, raw)
            timing["bytes"] = len(raw)
    write(path.with_suffix(".provenance.json"), {"video": bind(path), "submission": submission,
                                               "terminal_response": bind(saved)})
    return {"status": "downloaded", "video": bind(path), "task_id": submission["task_id"]}
