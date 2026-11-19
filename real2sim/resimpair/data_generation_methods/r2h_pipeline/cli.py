"""Explicit, resumable pipeline commands. Preparing work never contacts an API."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
import mimetypes
import os
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.parse
import urllib.request

from .common import file_sha256, json_sha256, load_env_file, probe_video, read_json, write_json
from .store import Store


DEFAULT_CONFIG = {
    "schema": "r2h_config_v1",
    "seedance": {"model": "doubao-seedance-2-5-260628", "base_url": "https://ark.cn-beijing.volces.com/api/v3",
                 "api_key_env": "ARK_API_KEY"},
    "gemini": {"model": "gemini-3.1-pro-preview", "base_url": "https://generativelanguage.googleapis.com/v1beta",
               "api_key_env": "GEMINI_API_KEY"},
    "anchor_model": "gemini-nano-banana-2.1",
    "max_generation_attempts": 2,
    "quality_thresholds_calibrated": False,
    "training_auto_start": False,
}


def emit(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def config_for(args):
    config = read_json(args.config) if args.config else DEFAULT_CONFIG
    if config.get("training_auto_start") is not False:
        raise ValueError("This pipeline does not start model training")
    return config


def require_credentials(provider_config):
    name = provider_config["api_key_env"]
    if not os.environ.get(name):
        raise ValueError("Missing environment variable: " + name)


def attempt_dir(row):
    return Path(row["request_path"]).parent


def doctor(args):
    cfg = config_for(args)
    result = {"schema": "r2h_doctor_v1", "python": sys.version.split()[0],
              "ffmpeg_available": shutil.which("ffmpeg") is not None,
              "ffprobe_available": shutil.which("ffprobe") is not None,
              "providers": {}, "network_calls_made": False,
              "training_auto_start": False}
    for name in ("seedance", "gemini"):
        entry = cfg[name]
        result["providers"][name] = {"model": entry["model"], "credential_variable": entry["api_key_env"],
                                     "credential_present": bool(os.environ.get(entry["api_key_env"]))}
    result["readiness"] = "configured_not_network_verified" if all(
        p["credential_present"] for p in result["providers"].values()) else "waiting_for_api_credentials"
    if args.output:
        write_json(args.output, result, overwrite=True)
    return result


def prepare(args):
    from .importer import prepare_catalog
    catalog = prepare_catalog(Path(args.source), Path(args.output), groups=args.groups, split=args.split)
    ledger = Store(args.db or str(Path(args.output) / "ledger.sqlite3"))
    try:
        ledger.register(catalog)
        return {"catalog": str(Path(args.output) / "catalog.json"), "jobs": len(catalog["jobs"]),
                "db": str(ledger.path), "status": ledger.status(), "network_calls_made": False}
    finally:
        ledger.close()


def load_store(args):
    if not args.db:
        raise ValueError("--db is required for this command")
    return Store(args.db)


def image_uri(path):
    path = Path(path).resolve(strict=True)
    mime = mimetypes.guess_type(path.name)[0]
    if mime not in ("image/png", "image/jpeg", "image/webp"):
        raise ValueError("Anchor must be PNG, JPEG or WebP")
    if path.stat().st_size > 10 * 1024**2:
        raise ValueError("Anchor too large for inline request")
    return "data:" + mime + ";base64," + base64.b64encode(path.read_bytes()).decode()


def plan(args, store, cfg):
    from .providers import SeedanceClient
    row = store.get(args.job)
    job = store.load_job(args.job)
    if row["status"] not in ("prepared", "generation_failed", "rejected", "needs_review"):
        raise ValueError("This job already has an unresolved request")
    if row["attempts"] >= cfg.get("max_generation_attempts", 2):
        raise ValueError("Generation attempt budget exhausted")
    prompt = Path(job["prompt_path"]).read_text()
    bindings = {}
    if args.mode == "semantic":
        anchor = store.anchor(row["group_id"])
        if not 4 <= args.duration <= 30:
            raise ValueError("Semantic request duration must be 4–30 seconds")
        content = [{"type": "text", "text": prompt},
                   {"type": "image_url", "image_url": {"url": image_uri(anchor["path"])}, "role": "first_frame"}]
        payload = {"model": cfg["seedance"]["model"], "content": content, "ratio": "adaptive",
                   "duration": args.duration, "generate_audio": False, "resolution": "720p"}
        bindings["shared_anchor"] = {"sha256": anchor["sha256"], "path": anchor["path"]}
    else:
        if not args.video_uri or not args.derived_media:
            raise ValueError("Edit requires --video-uri (HTTPS or asset://) and --derived-media manifest")
        derived = read_json(args.derived_media)
        if derived.get("schema") != "r2h_provider_media_v1":
            raise ValueError("Unrecognized derived media record")
        native = job["media"]["native_video"]
        if derived["source"]["sha256"] != native["sha256"]:
            raise ValueError("Edit input is not derived from this full source video")
        video = derived["derived"]
        if file_sha256(video["path"]) != video["sha256"] or not 4 <= video["duration_seconds"] <= 30:
            raise ValueError("Invalid edit derivative or duration")
        if not (args.video_uri.startswith("https://") or args.video_uri.startswith("asset://")):
            raise ValueError("Video input requires HTTPS or provider asset URI; not local paths/data URIs")
        content = [{"type": "text", "text": "Edit the supplied video. Replace the robot with human hands.\n" + prompt},
                   {"type": "video_url", "video_url": {"url": args.video_uri}, "role": "reference_video"}]
        payload = {"model": cfg["seedance"]["model"], "content": content, "ratio": "adaptive", "duration": -1,
                   "omni_reference_task_type": "edit", "generate_audio": False, "resolution": "720p"}
        bindings["derived_media"] = {"path": str(Path(args.derived_media).resolve()),
                                     "sha256": file_sha256(args.derived_media), "video_sha256": video["sha256"],
                                     "remote_upload_binding": "operator_supplied_uri; verify provider asset before large batches"}
    SeedanceClient(cfg["seedance"]).prepare_submit(payload)
    directory = Path(row["job_path"]).parent / "attempts" / f"{row['attempts'] + 1:03d}"
    path = directory / "request.json"
    write_json(path, {"schema": "r2h_generation_request_v1", "job_id": args.job, "mode": args.mode,
                     "payload": payload, "bindings": bindings,
                     "full_task_required": True, "human_wrist_gt_available": False,
                     "source_duration_seconds": job["media"]["native_video"]["duration_seconds"]})
    store.transition(args.job, "request_prepared", expected=row["status"], request_path=str(path),
                     request_sha256=file_sha256(path), detail={"mode": args.mode, "model": payload["model"]})
    return {"job_id": args.job, "status": "request_prepared", "request_path": str(path), "network_calls_made": False}


def submit(args, store, cfg):
    from .providers import SeedanceClient, ProviderError
    require_credentials(cfg["seedance"])
    job = store.load_job(args.job)
    row = store.get(args.job)
    request = read_json(row["request_path"])
    if request["payload"]["model"] != cfg["seedance"]["model"]:
        raise ValueError("Model differs from prepared request")
    for binding in request.get("bindings", {}).values():
        if file_sha256(binding["path"]) != binding["sha256"]:
            raise ValueError("Prepared input changed")
    row = store.claim_submission(args.job, cfg.get("max_generation_attempts", 2))
    try:
        result = SeedanceClient(cfg["seedance"]).submit(request["payload"])
        write_json(attempt_dir(row) / "submission.json", result)
        store.transition(args.job, "submitted", expected="submitting", task_id=result["task_id"],
                         detail={"model": cfg["seedance"]["model"]})
    except ProviderError as exc:
        status = "submission_unknown" if exc.submission_unknown else "generation_failed"
        store.transition(args.job, status, expected="submitting", detail={"code": exc.code})
        return {"job_id": args.job, "status": status, "error_code": exc.code}
    except Exception:
        # Even a local persistence failure after POST must never trigger a new POST.
        if store.get(args.job)["status"] == "submitting":
            store.transition(args.job, "submission_unknown", expected="submitting", detail={"code": "post_or_persistence_unknown"})
        raise
    return {"job_id": args.job, "status": "submitted", "task_id": result["task_id"]}


def download_video(url, path, *, task_id=None, request_sha256=None):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Generated media must use HTTPS")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = path.with_suffix(path.suffix + ".download.json")
    origin = {"task_id": task_id, "request_sha256": request_sha256,
              "output_url_sha256": json_sha256(url)}
    if path.exists():
        if not receipt_path.exists():
            raise ValueError("Unbound preexisting generated file has no download receipt")
        receipt = read_json(receipt_path)
        old = receipt.get("origin", {})
        if (old.get("task_id") != task_id or old.get("request_sha256") != request_sha256
                or (not task_id and old.get("output_url_sha256") != origin["output_url_sha256"])
                or file_sha256(path) != receipt["media"]["sha256"]):
            raise ValueError("Generated download origin or hash mismatch")
        return probe_video(path)
    fd, temporary = tempfile.mkstemp(prefix=".download-", suffix=".mp4", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as target, urllib.request.urlopen(url, timeout=60) as response:
            total = 0
            while True:
                data = response.read(1024 * 1024)
                if not data:
                    break
                total += len(data)
                if total > 1024**3:
                    raise ValueError("Generated media exceeds 1 GiB download budget")
                target.write(data)
        metadata = probe_video(temporary)
        os.replace(temporary, path)
        metadata["path"] = str(path.resolve())
        write_json(receipt_path, {"schema": "r2h_download_receipt_v1", "origin": origin, "media": metadata})
        return metadata
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def poll(args, store, cfg):
    from .providers import SeedanceClient
    require_credentials(cfg["seedance"])
    store.load_job(args.job)
    row = store.get(args.job)
    if row["status"] not in ("submitted", "generating"):
        raise ValueError("Only submitted/generating tasks can be polled")
    result = SeedanceClient(cfg["seedance"]).poll(row["task_id"])
    directory = attempt_dir(row)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    write_json(directory / ("poll_" + stamp + ".json"), result)
    status = result["status"]
    if status in ("succeeded", "success", "completed"):
        output = directory / "generated.mp4"
        metadata = download_video(result["output_url"], output, task_id=row["task_id"],
                                  request_sha256=row["request_sha256"])
        write_json(directory / "generated_media.json", metadata)
        store.transition(args.job, "generated", expected=row["status"], output_path=str(output),
                         detail={"output_sha256": metadata["sha256"]})
        status = "generated"
    elif status in ("failed", "expired", "cancelled", "canceled"):
        store.transition(args.job, "generation_failed", expected=row["status"], detail={"provider_status": status})
        status = "generation_failed"
    else:
        store.transition(args.job, "generating", expected=row["status"], detail={"provider_status": status})
        status = "generating"
    return {"job_id": args.job, "status": status}


def review(args, store, cfg):
    from .providers import GeminiClient, ProviderError
    from .review import evaluate_review, review_response_schema
    require_credentials(cfg["gemini"])
    row = store.get(args.job)
    job = store.load_job(args.job)
    if row["status"] not in ("generated", "review_failed", "needs_review"):
        raise ValueError("Review requires a decoded generated candidate")
    metadata = probe_video(row["output_path"])
    prompt = Path(job["blind_review_prompt_path"]).read_text()
    invocation = {"blind": True, "expected_goal_provided": False, "media_sha256": metadata["sha256"],
                  "prompt_sha256": file_sha256(job["blind_review_prompt_path"]), "origin": "actual_blind_vlm",
                  "model": cfg["gemini"]["model"], "response_schema_sha256": json_sha256(review_response_schema())}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    directory = attempt_dir(row) / ("review_" + stamp)
    write_json(directory / "invocation.json", invocation)
    write_json(directory / "media.json", metadata)
    store.transition(args.job, "reviewing", expected=row["status"])
    try:
        result = GeminiClient(cfg["gemini"]).review(Path(row["output_path"]), prompt, review_response_schema())
        review_path = directory / "response.json"
        write_json(review_path, result)
        job["review_request"] = invocation
        decision = evaluate_review(job, result["review"], metadata)
        decision_path = directory / "decision.json"
        decision["bindings"] = {"job_sha256": row["job_sha256"], "bundle_sha256": row["bundle_sha256"],
                                 "response_sha256": file_sha256(review_path), "media_sha256": metadata["sha256"],
                                 "invocation_sha256": file_sha256(directory / "invocation.json")}
        write_json(decision_path, decision)
        store.transition(args.job, decision["decision"], expected="reviewing", review_path=str(review_path),
                         decision_path=str(decision_path), decision_sha256=file_sha256(decision_path),
                         detail={"error_codes": decision.get("error_codes", [])})
    except ProviderError as exc:
        store.transition(args.job, "review_failed", expected="reviewing", detail={"code": exc.code})
        return {"job_id": args.job, "status": "review_failed", "error_code": exc.code,
                "automatic_retry": False}
    except Exception:
        if store.get(args.job)["status"] == "reviewing":
            store.transition(args.job, "review_failed", expected="reviewing", detail={"code": "review_or_persistence_failed"})
        raise
    return {"job_id": args.job, "status": decision["decision"], "decision_path": str(decision_path),
            "error_codes": decision.get("error_codes", []), "training_admitted": False}


def export_pairs(args, store):
    pairs = []
    all_rows = store.rows()
    for row in all_rows:
        if row["status"] != "accepted_for_pairing":
            continue
        reference = store.load_job(row["job_id"])
        decision = read_json(row["decision_path"])
        bindings = decision["bindings"]
        if (file_sha256(row["decision_path"]) != row["decision_sha256"]
                or decision["decision"] != "accepted_for_pairing" or decision.get("training_admitted") is not False
                or bindings["bundle_sha256"] != row["bundle_sha256"]
                or bindings["job_sha256"] != row["job_sha256"]
                or file_sha256(row["review_path"]) != bindings["response_sha256"]
                or file_sha256(row["output_path"]) != bindings["media_sha256"]
                or file_sha256(Path(row["decision_path"]).parent / "invocation.json") != bindings["invocation_sha256"]):
            raise ValueError("Review or source binding changed; cannot export")
        for query in all_rows:
            if (query["split"] != row["split"] or query["goal_id"] != row["goal_id"]
                    or query["group_id"] == row["group_id"]):
                continue
            query_job = store.load_job(query["job_id"])
            pairs.append({"pair_id": json_sha256([row["job_id"], query["job_id"]])[:24],
                          "goal_id": row["goal_id"], "split": row["split"],
                          "reference_job_id": row["job_id"], "reference_video": row["output_path"],
                          "reference_sha256": bindings["media_sha256"], "reference_source": "synthetic_human",
                          "query_job_id": query["job_id"], "query_source": query_job["source"],
                          "reference_group_id": row["group_id"], "query_group_id": query["group_id"],
                          "decision_path": row["decision_path"], "decision_sha256": file_sha256(row["decision_path"]),
                          "human_wrist_gt_available": False, "training_admitted": False})
    output = {"schema": "r2h_pair_candidates_v1", "pairs": pairs, "count": len(pairs),
              "training_admitted": False, "release_status": "requires_calibrated_quality_and_training_reader_validation"}
    write_json(args.output, output)
    return {"output": str(Path(args.output).resolve()), "pair_candidates": len(pairs), "training_admitted": False}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", help="SQLite job ledger")
    p.add_argument("--config", help="JSON config without credentials")
    p.add_argument("--env-file", help="Literal KEY=value credentials file; values are never displayed")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor").add_argument("--output")
    prep = sub.add_parser("prepare")
    prep.add_argument("--source", required=True); prep.add_argument("--output", required=True)
    prep.add_argument("--groups", type=int, default=2); prep.add_argument("--split", default="train", choices=("train", "validation", "test"))
    reg = sub.add_parser("register"); reg.add_argument("--catalog", required=True)
    sub.add_parser("status")
    sub.add_parser("list")
    anchor = sub.add_parser("bind-anchor"); anchor.add_argument("--group", required=True)
    anchor.add_argument("--image", required=True); anchor.add_argument("--provenance")
    for name in ("plan-anchor", "generate-anchor"):
        sub.add_parser(name).add_argument("--group", required=True)
    media = sub.add_parser("prepare-edit-media"); media.add_argument("--source", required=True); media.add_argument("--output", required=True)
    req = sub.add_parser("plan"); req.add_argument("--job", required=True)
    req.add_argument("--mode", choices=("semantic", "edit"), default="semantic")
    req.add_argument("--duration", type=int, default=30); req.add_argument("--video-uri"); req.add_argument("--derived-media")
    for name in ("submit", "poll", "review"):
        sub.add_parser(name).add_argument("--job", required=True)
    resume = sub.add_parser("attach-task")
    resume.add_argument("--job", required=True); resume.add_argument("--task-id", required=True)
    sub.add_parser("export-pairs").add_argument("--output", required=True)
    work = sub.add_parser("work", help="One bounded worker pass; rerun to resume polling")
    work.add_argument("--max-submissions", type=int, default=1); work.add_argument("--max-reviews", type=int, default=1)
    work.add_argument("--max-polls", type=int, default=8)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    store = None
    try:
        load_env_file(args.env_file)
        cfg = config_for(args)
        if args.command == "doctor":
            result = doctor(args)
        elif args.command == "prepare":
            result = prepare(args)
        elif args.command == "prepare-edit-media":
            from .media import prepare_seedance_video
            result = prepare_seedance_video(args.source, args.output)
        else:
            store = load_store(args)
            if args.command == "register":
                store.register(read_json(args.catalog)); result = {"status": store.status()}
            elif args.command == "status":
                result = {"status": store.status(), "training_auto_start": False}
            elif args.command == "list":
                result = [{k: r[k] for k in ("job_id", "group_id", "goal_id", "split", "status", "attempts")} for r in store.rows()]
            elif args.command == "bind-anchor":
                image_uri(args.image)
                result = store.bind_anchor(args.group, args.image, args.provenance)
            elif args.command in ("plan-anchor", "generate-anchor"):
                from .anchors import plan_anchor, generate_anchor
                result = {"plan-anchor": plan_anchor, "generate-anchor": generate_anchor}[args.command](store, args.group, cfg)
            elif args.command == "plan":
                result = plan(args, store, cfg)
            elif args.command in ("submit", "poll", "review"):
                result = {"submit": submit, "poll": poll, "review": review}[args.command](args, store, cfg)
            elif args.command == "attach-task":
                row = store.get(args.job)
                if row["status"] == "submitting":
                    store.transition(args.job, "submission_unknown", expected="submitting", detail={"code": "manual_recovery_after_interruption"})
                store.transition(args.job, "submitted", expected="submission_unknown", task_id=args.task_id,
                                 detail={"recovery": "provider_task_id_supplied_without_resubmission"})
                result = {"job_id": args.job, "status": "submitted"}
            elif args.command == "export-pairs":
                result = export_pairs(args, store)
            elif args.command == "work":
                if min(args.max_submissions, args.max_reviews, args.max_polls) < 0:
                    raise ValueError("Worker limits cannot be negative")
                limits = {"request_prepared": args.max_submissions, "generated": args.max_reviews,
                          "submitted": args.max_polls}
                results = []
                for row in store.rows():
                    status = row["status"]
                    category = "submitted" if status == "generating" else status
                    if category not in limits or limits[category] <= 0:
                        continue
                    args.job = row["job_id"]
                    function = {"request_prepared": submit, "generated": review, "submitted": poll}[category]
                    results.append(function(args, store, cfg)); limits[category] -= 1
                result = {"processed": results, "status": store.status(), "automatic_uncertain_resubmission": False}
            else:
                raise ValueError("Unsupported command")
        emit(result)
        return 0
    except Exception as exc:
        # Transport errors may contain URLs or keys. Only controlled ValueErrors
        # from our code may be shown; all other exceptions use a type/code only.
        from .providers import ProviderError
        if isinstance(exc, ProviderError):
            message = exc.code
        elif isinstance(exc, ValueError) and exc.__class__ is ValueError:
            message = str(exc)
            if any(token in message.lower() for token in ("api_key=", "key=", "bearer ", "authorization")):
                message = "Invalid configuration or operation"
        else:
            message = type(exc).__name__
        emit({"error": message, "command": args.command})
        return 2
    finally:
        if store is not None:
            store.close()


if __name__ == "__main__":
    raise SystemExit(main())
