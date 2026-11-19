"""CPU-only, hash-bound preparation of existing complete three-goal recordings.

No simulator, model, credentials or video generation backend is imported here.
The source is immutable.  Only anonymous event-covering PNGs and job metadata
are written, with no implicit admission to training.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import numpy as np

from .contracts import (
    GOAL_PREFIX, INITIAL_HASH_KEYS, PROGRAMS, blind_review_prompt,
    build_contract, generation_prompt,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2 ** 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _encoded(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _write_immutable(path: Path, content: str) -> None:
    if path.exists():
        _require(path.read_text(encoding="utf-8") == content, "Prepared artifact conflict: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    try:
        # A per-catalog exclusive lock serializes these atomic installs.
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _lock(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".prepare.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, str(os.getpid()).encode())
        os.close(descriptor)
        yield
    finally:
        lock.unlink(missing_ok=True)


def _binding(path: str | Path, expected: str, label: str) -> dict:
    source = Path(path).resolve(strict=True)
    _require(source.is_file(), "Not a regular source file: " + label)
    _require(isinstance(expected, str) and len(expected) == 64, "Missing declared SHA256: " + label)
    actual = _sha(source)
    _require(actual == expected, "Source hash mismatch: " + label + " " + str(source))
    return {"path": str(source), "sha256": actual, "bytes": source.stat().st_size}


def _select_rows(source: dict, groups: int, split: str) -> tuple[list[list[dict]], dict]:
    _require(isinstance(groups, int) and not isinstance(groups, bool) and groups > 0, "groups must be positive")
    _require(split in ("train", "validation", "test"), "Use an explicit existing split name")
    _require(source.get("formal_release_approved") is True, "Source manifest is not formally released")
    rows = source.get("episodes")
    _require(isinstance(rows, list) and rows, "Source manifest has no episodes")
    ids, by_group, group_splits = set(), defaultdict(list), defaultdict(set)
    for row in rows:
        identity, group = row.get("id"), row.get("group")
        _require(isinstance(identity, str) and identity not in ids, "Missing/duplicate source episode ID")
        _require(isinstance(group, str) and group, "Missing source group")
        ids.add(identity)
        group_splits[group].add(row.get("split"))
        by_group[group].append(row)
    _require(all(len(s) == 1 for s in group_splits.values()), "Source group crosses splits")
    complete = []
    for group in sorted(by_group):
        entries = by_group[group]
        if group_splits[group] != {split}:
            continue
        programs = [r.get("program") for r in entries]
        _require(len(programs) == len(set(programs)), "Duplicate goal within source group: " + group)
        if set(programs) != set(PROGRAMS):
            continue
        for row in entries:
            _require(row.get("task") == "long_block_access_restore", "Unsupported source task")
            _require(row.get("formal_release_approved") is True
                     and row.get("production_approved") is True
                     and row.get("qualification_only") is False, "Source row is not formally accepted")
            _require(row.get("goal_key") == GOAL_PREFIX + row["program"], "Source goal/program mismatch")
        complete.append(sorted(entries, key=lambda r: PROGRAMS.index(r["program"])))
    _require(len(complete) >= groups, f"Need {groups} complete {split} groups; found {len(complete)}")
    return complete[:groups], {
        "total_source_episodes": len(rows), "total_source_groups": len(by_group),
        "complete_groups_in_requested_split": len(complete), "requested_split": split,
        "selection": "lexicographic_group_then_fixed_three_program_order",
        "cross_split_group_conflicts": 0,
    }


def _source_row(row: dict) -> dict:
    receipt_file = _binding(row["collection_receipt"], row["receipt_sha256"], "receipt")
    receipt = _load(Path(receipt_file["path"]))
    for key in ("length", "fps", "split", "program", "group", "goal_key", "action_space", "label_kind"):
        _require(row.get(key) == receipt.get(key), "Row/receipt conflict: " + key)
    _require(receipt.get("production_approved") is True and receipt.get("qualification_only") is False,
             "Receipt is not a formal production result")
    payloads = receipt.get("sha256", {})
    arrays = _binding(row["arrays"], payloads.get("arrays.npz"), "arrays")
    _require(Path(receipt["arrays"]).resolve() == Path(arrays["path"]), "Array locator mismatch")
    native = _binding(row["reference_video"]["path"], row.get("reference_video_sha256"), "native RGB")
    _require(receipt["reference_video_sha256"] == native["sha256"]
             and Path(receipt["reference_video"]["path"]).resolve() == Path(native["path"]), "Native RGB locator/hash mismatch")
    policy = _binding(row["video"]["path"], row.get("source_video_sha256"), "policy RGB")
    _require(payloads.get("video.mp4") == policy["sha256"]
             and Path(receipt["video"]).resolve() == Path(policy["path"]), "Policy RGB locator/hash mismatch")
    extra = {}
    for field in ("qa_visibility", "qa_contact_forces"):
        if field in receipt:
            extra[field] = _binding(receipt[field], receipt.get(field + "_sha256"), field)
    with np.load(arrays["path"], allow_pickle=False) as source:
        data = {k: source[k] for k in source.files}
    n, fps = row["length"], row["fps"]
    for key in ("native_commands", "timestamps_seconds", "initial_physics_state", "stage", "object_states"):
        _require(key in data, "Missing recorded evidence: " + key)
    for key, value in data.items():
        if np.issubdtype(value.dtype, np.number):
            _require(np.isfinite(value).all(), "Nonfinite recorded array: " + key)
    commands, stamps = data["native_commands"], data["timestamps_seconds"]
    _require(commands.shape == (n, 16) and np.isfinite(commands).all(), "Invalid native commands")
    _require(stamps.shape == (n,) and np.isfinite(stamps).all(), "Invalid native clock")
    _require(np.allclose(stamps, np.arange(n) / fps, atol=1e-8, rtol=0), "Native clock mismatch")
    _require(data["stage"].shape == (n,), "Invalid stage timeline")
    _require(data["object_states"].shape == (n, 3, 13)
             and np.isfinite(data["object_states"]).all(), "Invalid object state evidence")
    initial_sha = hashlib.sha256(np.ascontiguousarray(data["initial_physics_state"]).tobytes()).hexdigest()
    _require(initial_sha == receipt.get("initial_physics_sha256"), "Initial physics snapshot hash mismatch")
    contract = build_contract(row, receipt)
    return {"row": row, "receipt": receipt, "contract": contract, "arrays_data": data,
            "files": {"receipt": receipt_file, "arrays": arrays, "native_video": native,
                      "policy_video": policy, **extra}}


def _probe(path: str, expected_frames: int, expected_fps: float, size: int) -> dict:
    command = ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
               "-show_frames", "-show_entries", "frame=best_effort_timestamp_time:stream=width,height,r_frame_rate,avg_frame_rate,nb_read_frames,duration:format=duration",
               "-of", "json", path]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=300)
    _require(not result.stderr.strip(), "Video decoder reported errors: " + result.stderr[:500])
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    _require(len(streams) == 1, "Need one primary video stream")
    video = streams[0]
    numerator, denominator = video["avg_frame_rate"].split("/")
    fps = int(numerator) / int(denominator)
    frames = int(video["nb_read_frames"])
    duration = float(video.get("duration", expected_frames / expected_fps))
    _require(video["width"] == video["height"] == size, "Unexpected RGB dimensions")
    _require(frames == expected_frames and abs(fps - expected_fps) < 1e-8, "RGB frame count/FPS mismatch")
    _require(abs(duration - expected_frames / expected_fps) <= 1 / expected_fps + 1e-6, "RGB duration mismatch")
    stamps = np.asarray([float(f["best_effort_timestamp_time"]) for f in payload["frames"]])
    _require(len(stamps) == expected_frames and np.allclose(stamps, np.arange(expected_frames) / expected_fps, atol=1e-5, rtol=0),
             "Decoded RGB/native-clock timestamp mismatch")
    return {"frames": frames, "fps": fps, "width": size, "height": size,
            "duration_seconds": duration, "full_frame_count_checked": True, "full_decode_passed": True,
            "decoded_timestamps_max_error_seconds": float(np.max(np.abs(stamps - np.arange(expected_frames) / expected_fps)))}


def _frame_selection(contract: dict, stages: np.ndarray, n: int) -> list[dict]:
    selected: dict[int, list[dict]] = defaultdict(list)
    selected[0].append({"reason": "initial_state", "event_id": None})
    selected[n - 1].append({"reason": "final_state", "event_id": None})
    for event in contract["required_events"]:
        start, end = event["first_frame"], event["end_frame_exclusive"]
        candidates = [i for i in range(start, end) if str(stages[i]).endswith(":lift")]
        middle = candidates[len(candidates) // 2] if candidates else (start + end) // 2
        for frame, reason in ((start, "phase_start"), (middle, "lift_stage_or_phase_midpoint"), (end - 1, "phase_end")):
            selected[frame].append({"reason": reason, "event_id": event["id"]})
    return [{"frame_index": frame, "coverage": selected[frame]} for frame in sorted(selected)]


def _frames(native: dict, selection: list[dict], directory: Path, fps: float) -> list[dict]:
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "frames.json"
    key = {"source_sha256": native["sha256"], "selection": selection}
    if manifest_path.exists():
        cached = _load(manifest_path)
        _require(cached["cache_key"] == key, "Frame cache input conflict")
        for entry in cached["frames"]:
            _require(_sha(Path(entry["path"])) == entry["sha256"], "Frame cache hash mismatch")
        return cached["frames"]
    _require(not list(directory.glob("frame_*.png")), "Uncommitted frame cache; use a fresh output directory")
    expression = "+".join(f"eq(n\\,{x['frame_index']})" for x in selection)
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "1",
               "-i", native["path"], "-vf", "select=" + expression,
               "-filter_threads", "1", "-vsync", "0", "-threads", "1", "-start_number", "1",
               str(directory / "frame_%06d.png")]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=300)
    paths = sorted(directory.glob("frame_*.png"))
    _require(len(paths) == len(selection), "Event frame extraction count mismatch")
    frames = [{"path": str(path), "sha256": _sha(path), "frame_index": item["frame_index"],
               "time_seconds": item["frame_index"] / fps, "coverage": item["coverage"]}
              for path, item in zip(paths, selection)]
    _write_immutable(manifest_path, _encoded({"cache_key": key, "frames": frames}))
    return frames


def _group_evidence(items: list[dict]) -> dict:
    hashes = {}
    for key in INITIAL_HASH_KEYS:
        values = {x["receipt"][key] for x in items}
        _require(len(values) == 1, "Goal-dependent initial state/RGB: " + key)
        hashes[key] = next(iter(values))
    differences = []
    for left, right in itertools.combinations(items, 2):
        a, b = left["arrays_data"]["native_commands"], right["arrays_data"]["native_commands"]
        n = min(len(a), len(b))
        different = np.flatnonzero(np.max(np.abs(a[:n] - b[:n]), axis=1) > 1e-5)
        frame = int(different[0]) if len(different) else None
        differences.append({"goals": [left["row"]["goal_key"], right["row"]["goal_key"]],
                            "first_native_command_difference_frame": frame,
                            "first_native_command_difference_seconds": frame / left["row"]["fps"] if frame is not None else None,
                            "compared_prefix_frames": n, "absolute_command_tolerance": 1e-5,
                            "lengths": [len(a), len(b)]})
    return {"group_id": items[0]["row"]["group"], "split": items[0]["row"]["split"],
            "initial_four_hashes": hashes, "initial_four_hashes_match": True,
            "command_divergences": differences,
            "interpretation": "Measured command differences, not an independently labeled semantic decision time or proof of learned reference use."}


def prepare_catalog(source_manifest: Path, output_dir: Path, groups: int = 2, split: str = "train") -> dict:
    """Prepare complete groups deterministically; identical repeated calls are safe."""
    manifest_path = Path(source_manifest).resolve(strict=True)
    output = Path(output_dir).resolve()
    manifest_sha = _sha(manifest_path)
    source = _load(manifest_path)
    selected, selection_summary = _select_rows(source, groups, split)
    request = {"source_manifest": str(manifest_path), "source_manifest_sha256": manifest_sha,
               "groups": groups, "split": split, "preparation_version": "retained319_three_goal_v2"}
    with _lock(output):
        catalog_path = output / "catalog.json"
        previous = _load(catalog_path) if catalog_path.exists() else None
        if previous:
            _require(previous.get("request") == request, "Catalog configuration/source conflict")
            for entry in previous["jobs"]:
                base = Path(entry["job_path"]).parent
                for name, digest in entry["artifact_sha256"].items():
                    _require(_sha(base / name) == digest, "Prepared artifact hash mismatch: " + str(base / name))
        all_items, group_proofs = [], []
        for entries in selected:
            items = [_source_row(row) for row in entries]
            proof = _group_evidence(items)
            group_proofs.append(proof)
            all_items.extend((item, proof) for item in items)
        jobs = []
        for item, proof in all_items:
            row, receipt, files, contract = item["row"], item["receipt"], item["files"], item["contract"]
            identity = hashlib.sha256(_encoded({"request": request, "episode": row["id"], "receipt_sha256": files["receipt"]["sha256"]}).encode()).hexdigest()[:24]
            job_id = "r2h_" + identity
            directory = output / "jobs" / job_id
            directory.mkdir(parents=True, exist_ok=True)
            existing = _load(directory / "job.json") if (directory / "job.json").exists() else None
            media = {}
            for field, size in (("native_video", 512), ("policy_video", 224)):
                if existing:
                    cached = existing["media"][field]
                    _require(cached["sha256"] == files[field]["sha256"], "Cached source media conflict")
                    media[field] = cached
                else:
                    media[field] = {**files[field], **_probe(files[field]["path"], row["length"], row["fps"], size)}
            selection = _frame_selection(contract, item["arrays_data"]["stage"], row["length"])
            media["event_frames"] = _frames(files["native_video"], selection, directory / "frames", row["fps"])
            evidence = {"schema": "r2h_source_evidence_v1", "training_admitted": False,
                        "source_row_unchanged": row, "source_files": files,
                        "source_receipt_flags_original": {k: v for k, v in receipt.items() if isinstance(v, bool)},
                        "group_proof": proof, "source_events_original": receipt["events"],
                        "source_phase_targets_original": receipt["phase_targets"],
                        "source_privileged_channels": sorted(item["arrays_data"]),
                        "source_camera": receipt.get("camera"), "source_asset_specs": receipt.get("asset_specs"),
                        "source_code_pins_declared": receipt.get("source_sha256", {}),
                        "checks_performed": ["all_declared_data_payload_hashes", "row_receipt_consistency", "finite_commands_and_object_states",
                                             "native_clock", "initial_physics_array_hash", "four_hash_group_equality",
                                             "full_RGB_frame_count_and_duration", "ordered_event_bounds", "event_covering_PNG_hashes"],
                        "physical_success_scope": "Existing formally accepted source evidence; this importer does not rerun the physical outcome oracle.",
                        "source_code_pin_execution_or_replay_performed": False,
                        "historical_source_code_pins_reverified": False,
                        "initial_RGB_hash_scope": "Raw RGB hashes are original receipt evidence compared across goals; lossy videos cannot reproduce raw RGB byte hashes.",
                        "source_depth_or_dense_mask_created": False, "source_snapshot_controller_state_portability_claimed": False}
            job = {"schema": "r2h_job_v1", "job_id": job_id, "status": "prepared",
                   "group_id": row["group"], "goal_id": row["goal_key"], "split": row["split"],
                   "training_admitted": False, "generated": False,
                   "source": {"manifest": {"path": str(manifest_path), "sha256": manifest_sha},
                              "episode_id": row["id"], "receipt": files["receipt"],
                              "group_id": row["group"], "goal_id": row["goal_key"], "split": row["split"]},
                   "contract": "contract.json", "evidence": "evidence.json",
                   "contract_path": str(directory / "contract.json"), "evidence_path": str(directory / "evidence.json"),
                   "prompt_path": str(directory / "prompt.txt"), "blind_review_prompt_path": str(directory / "blind_review_prompt.txt"),
                   "media": media,
                   "durations": {"native_seconds": media["native_video"]["duration_seconds"],
                                 "policy_seconds": media["policy_video"]["duration_seconds"],
                                 "requested_output_seconds": None, "full_task_preserved": True},
                   "outputs": {"generated_video": str(directory / "generated.mp4"), "review": str(directory / "review.json")}}
            _require(not Path(job["outputs"]["generated_video"]).exists(), "Importer does not own existing generated output")
            for name, doc in (("contract.json", contract), ("evidence.json", evidence), ("job.json", job)):
                _write_immutable(directory / name, _encoded(doc))
            _write_immutable(directory / "prompt.txt", generation_prompt(contract))
            _write_immutable(directory / "blind_review_prompt.txt", blind_review_prompt())
            artifact_names = ("job.json", "contract.json", "evidence.json", "prompt.txt", "blind_review_prompt.txt", "frames/frames.json")
            jobs.append({"job_id": job_id, "job_path": str(directory / "job.json"), "group_id": row["group"],
                         "goal_id": row["goal_key"], "split": row["split"],
                         "artifact_sha256": {name: _sha(directory / name) for name in artifact_names}})
        _require(_sha(manifest_path) == manifest_sha, "Source manifest changed during preparation")
        catalog = {"schema": "r2h_catalog_v1", "request": request, "training_admitted": False,
                   "generated_jobs": 0, "prepared_jobs": len(jobs), "prepared_groups": len(group_proofs),
                   "source_selection": selection_summary, "groups": group_proofs, "jobs": jobs,
                   "scope": "CPU inputs and task evidence only; no generated human media, no model training or new simulation."}
        _write_immutable(catalog_path, _encoded(catalog))
        return catalog
