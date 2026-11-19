#!/usr/bin/env python3
"""Compile available 20261020B outputs into immutable inspection-only cross-demo pairs.

No API calls, simulation, training, production manifest edits, or automatic
approval. Re-running after more videos arrive produces another content-addressed
snapshot. Missing outputs are reported separately; failed probes are not hidden.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from functools import lru_cache
import importlib.util
import json
import os
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[2]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from data_generation_methods.r2h_pipeline.common import file_sha256, json_sha256, write_json
from data_generation_methods.r2h_pipeline.training_pairs import (
    COLLECTION_SCHEMA, SCHEMA, read_pair_inputs, validate_collection, validate_pair,
)


QUERY_BY_SOURCE = {
    "simx_real_bowl_transfer_891911000_pick_first_to_right":
        "simx_real_bowl_transfer_891910000_pick_second_to_right",
    "simx_real_bowl_transfer_891910000_pick_second_to_right":
        "simx_real_bowl_transfer_891911000_pick_first_to_right",
    "simx_closed_book_to_open_box_891912000": "simx_closed_book_to_open_box_891913000",
    "simx_cabinet_right_door_cabinet_open_right_only_891914000":
        "simx_cabinet_right_door_cabinet_open_right_only_891914001",
}


def binding(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": file_sha256(path)}


def load_bound(value):
    actual = binding(value["path"])
    if actual["sha256"] != value["sha256"]:
        raise ValueError("Bound source changed: " + value["path"])
    return json.loads(Path(value["path"]).read_text())


def video_clock(path):
    import av
    import numpy as np
    stamps = []
    with av.open(str(path)) as movie:
        stream = movie.streams.video[0]
        stream.codec_context.thread_count = 1
        fps = float(stream.average_rate)
        tolerance = max(1e-8, float(stream.time_base) * .51)
        dimensions = [stream.width, stream.height]
        for frame in movie.decode(video=0):
            if frame.pts is None:
                raise ValueError("Generated output lacks PTS")
            stamps.append(float(frame.pts * frame.time_base))
    if len(stamps) < 2 or not np.all(np.diff(stamps) > 0):
        raise ValueError("Missing/nonmonotonic generated frames")
    error = float(np.max(np.abs(np.asarray(stamps) - np.arange(len(stamps)) / fps)))
    if error > tolerance + 1e-8:
        raise ValueError("Generated output is not on declared CFR clock")
    return {"fps": fps, "length": len(stamps), "start_seconds": 0.,
            "dimensions": dimensions, "PTS_max_error_seconds": error,
            "duration_seconds": len(stamps) / fps, "full_decode_passed": True}


def robot_roots(row):
    result = ["robot-group:" + row["group"]]
    if row.get("physical_fingerprint"):
        result.append("robot-rollout:" + row["physical_fingerprint"])
    return result


@lru_cache(maxsize=2)
def production_module(name):
    path = BASE / "PolicyGen/examples/baselines/hypericlzq" / (name + ".py")
    spec = importlib.util.spec_from_file_location("r2h_inspection_native_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_production_functions(pair, sample, query_index):
    """Compare isolated NumPy inputs with unchanged native media/EEF functions."""
    import numpy as np
    row, ref = pair["query"]["episode_row"], pair["reference"]
    manifest = load_bound(pair["query"]["source_manifest"])
    spec = manifest["action_spaces"][row["action_space"]]
    with np.load(row["arrays"], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    actions, state, mask = production_module("eef").eef_actions(row, arrays, query_index, 16)
    normalized = (actions - np.asarray(spec["action_mean"])) / np.asarray(spec["action_std"])
    expected_state = (state - np.asarray(spec["state_mean"])) / np.asarray(spec["state_std"])
    np.testing.assert_allclose(sample["actions"][:len(actions), :14], normalized, rtol=2e-6, atol=3e-6)
    np.testing.assert_allclose(sample["state"][:18], expected_state, rtol=2e-6, atol=3e-6)
    np.testing.assert_array_equal(sample["action_mask"][:len(actions), :14], mask)
    if sample["action_mask"][:, 14:].any() or sample["action_mask"][len(actions):].any():
        raise ValueError("Padding became supervision")
    indices = np.rint(np.linspace(0, ref["length"] - 1, 32)).astype(int)
    media = production_module("media")
    rgb = media.frames_at({"kind": "file", "path": ref["video"]["path"]},
                          ref["start_seconds"] + indices / ref["fps"], 224)
    rgb = rgb.transpose(0, 3, 1, 2).astype(np.float32) / 255
    obs = media.frames_at(row["video"], [row["start_seconds"] + query_index / row["fps"]], 224)
    obs = obs[0].transpose(2, 0, 1).astype(np.float32) / 255
    np.testing.assert_array_equal(sample["reference_video"], rgb)
    np.testing.assert_array_equal(sample["obs"], obs)
    return {"production_media_function_exercised": True, "production_eef_function_exercised": True,
            "full_legacy_CuratedDataset_exercised": False,
            "reference_RGB_exact_equal": True, "query_RGB_exact_equal": True,
            "native_action_state_mask_match": True,
            "normalized_action_max_abs_error": float(np.max(np.abs(
                sample["actions"][:len(actions), :14] - normalized))),
            "normalized_state_max_abs_error": float(np.max(np.abs(sample["state"][:18] - expected_state)))}


def build_pair(job, input_binding, path, pilot_root):
    source = job["robot"]["source_episode_row_original"]
    target_id = QUERY_BY_SOURCE[source["id"]]
    pilot_path = pilot_root / target_id / "input.json"
    pilot_binding = binding(pilot_path)
    pilot = load_bound(pilot_binding)
    row = deepcopy(pilot["source_episode_row_unchanged"])
    if row["id"] != target_id:
        raise ValueError("Cross-demo input identity differs")
    source_manifest = job["robot"]["source_policy_manifest"]
    load_bound(source_manifest)
    query_manifest = {"path": pilot["source_manifest"], "sha256": pilot["source_manifest_sha256"]}
    load_bound(query_manifest)
    spec = load_bound(job["source_spec"])
    task = next(t for t in spec["tasks"] if t["id"] == job["task_id"])
    human_roots = []
    for window in task["human_windows"]:
        if window["split"] == row["split"]:
            human_roots.extend(["real-video:" + window["video"]["sha256"],
                                "real-session:EgoDex:" + window["session"]])
    clock = video_clock(path)
    media = binding(path)
    reference_id = "synthetic_" + job["job_id"] + "_" + media["sha256"][:16]
    evidence = [input_binding, job["source_spec"], job["robot"]["source_input"],
                job["anchor"], pilot_binding]
    for filename in ("edit_anchor/result.json", "submit_video/result.json", "candidate_summary.json"):
        candidate = path.parent / filename
        if candidate.is_file():
            evidence.append(binding(candidate))
    ref = {
        "episode_id": reference_id, "source": "synthetic_human", "usage": "reference_only",
        "split": source["split"], "group": source["group"], "goal_key": source["goal_key"],
        "video": media, **{k: clock[k] for k in ("fps", "length", "start_seconds")},
        "action_space": None, "arrays": None, "human_wrist_3d_gt": False,
        "allowed_query_episode_ids": [target_id],
        "lineage_root_ids": sorted(set(human_roots + robot_roots(source)
                                      + ["source-anchor:" + job["anchor"]["sha256"]])),
        "quality_release_approved": False, "training_admitted": False,
        "reference_pool_eligible": False,
        "source_robot": {"episode_id": source["id"], "source_manifest": source_manifest},
        "source_human_window": deepcopy(job["source_window"]),
        "provenance_artifacts": evidence, "generated_media_inspection": clock,
    }
    pair = {
        "schema": SCHEMA, "pair_id": job["job_id"] + "__crossdemo__" + target_id,
        "split": row["split"], "pair_scope": "task_only",
        "cross_demo_require_distinct_group": True,
        "quality_release_approved": False, "training_admitted": False,
        "frame_alignment": False, "human_action_retargeting": False,
        "reference": ref,
        "query": {"episode_row": row, "source_manifest": query_manifest,
                  "lineage_root_ids": sorted(set(human_roots + robot_roots(row))),
                  "artifacts": {"video": binding(row["video"]["path"]),
                                "arrays": binding(row["arrays"]),
                                "receipt": binding(row["collection_receipt"])}},
    }
    if ref["goal_key"] != row["goal_key"]:
        pair["goal_relation"] = {
            "type": "explicit_task_only_equivalence", "reference_goal_key": ref["goal_key"],
            "query_goal_key": row["goal_key"], "training_approved": False,
            "evidence": [job["source_spec"]],
            "scope": "Explicit cross-asset closed-book-to-open-box inspection candidate; requires visual release.",
        }
    validate_pair(pair, mode="inspection")
    return pair


def compile_batch(input_manifest, run_dir=None, output_root=None):
    input_manifest = Path(input_manifest).resolve()
    run_dir = Path(run_dir).resolve() if run_dir else input_manifest.parent
    output_root = Path(output_root).resolve() if output_root else run_dir / "training_pair_inspection_snapshots"
    batch_binding = binding(input_manifest)
    batch = load_bound(batch_binding)
    pilot_root = BASE / "data_generation_methods/pilot_inputs"
    available, pending = [], []
    for job in batch["jobs"]:
        path = run_dir / "jobs" / job["job_id"] / "generated_human.mp4"
        if path.is_file():
            available.append((job, path, binding(path)))
        else:
            pending.append({"job_id": job["job_id"], "expected_video": str(path),
                            "status": "generated_video_not_yet_available", "training_admitted": False})
    code_bindings = [binding(__file__), binding(BASE / "data_generation_methods/r2h_pipeline/training_pairs.py")]
    code_bindings += [binding(BASE / "PolicyGen/examples/baselines/hypericlzq" / (name + ".py"))
                      for name in ("media", "eef")]
    snapshot_id = json_sha256({"input": batch_binding, "media": [x[2] for x in available],
                              "code": code_bindings})[:20]
    output = output_root / snapshot_id
    pairs, probes, rejected = [], [], []
    for job, path, _ in available:
        try:
            pair = build_pair(job, batch_binding, path, pilot_root)
            row = pair["query"]["episode_row"]
            checks = []
            for t in sorted({0, row["length"] // 2, row["length"] - 1}):
                sample = read_pair_inputs(pair, query_index=t)
                checks.append({**sample["report"], "unchanged_production_function_comparison":
                               audit_production_functions(pair, sample, t)})
            pairs.append(pair)
            probes.append({"pair_id": pair["pair_id"], "checks": checks,
                           "cannot_train": True, "quality_release_approved": False})
        except (ValueError, KeyError, OSError, StopIteration, AssertionError) as error:
            rejected.append({"job_id": job["job_id"], "error_type": type(error).__name__,
                             "error": str(error), "training_admitted": False})
    frozen_code = []
    for original in code_bindings:
        if binding(original["path"]) != original:
            raise ValueError("Inspection source code changed while compiling")
        destination = output / "source_snapshot" / (original["sha256"] + ".py")
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = Path(original["path"]).read_bytes()
        if destination.exists():
            if destination.read_bytes() != content:
                raise ValueError("Conflicting immutable source snapshot")
        else:
            with destination.open("xb") as stream:
                stream.write(content)
        frozen_code.append({**binding(destination), "original_path": original["path"]})
    collection = {"schema": COLLECTION_SCHEMA, "pairs": pairs, "training_admitted": False,
                  "quality_release_approved": False, "mode": "inspection",
                  "batch_input": batch_binding, "source_code": frozen_code,
                  "explicit_cross_demo_mapping": QUERY_BY_SOURCE,
                  "source_state_changed": False, "training_started": False}
    validation = validate_collection(collection) if pairs else {
        "passed": False, "reason": "No available candidates passed CPU inspection", "cannot_train": True}
    summary = {
        "schema": "r2h_batch_training_pair_inspection_v1",
        "output_directory": str(output), "pair_manifest": str(output / "pair_manifest.json"),
        "available_generated_videos": len(available), "pending_generated_videos": len(pending),
        "inspected_explicit_pairs": len(pairs), "fixed_sample_reads": sum(len(p["checks"]) for p in probes),
        "rejected_inspection_candidates": len(rejected), "pending": pending, "rejected": rejected,
        "validation": validation, "quality_release_approved": False, "training_admitted": False,
        "can_train": False, "cannot_train": True, "model_training_started": False,
        "old_production_reader_modified": False,
    }
    if binding(input_manifest) != batch_binding:
        raise ValueError("Batch manifest changed during compilation; rerun against completed preparation")
    write_json(output / "pair_manifest.json", collection)
    write_json(output / "fixed_sample_cpu_inspection.json", {"probes": probes, "cannot_train": True})
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    report = compile_batch(args.input_manifest, args.run_dir, args.output_root)
    print(json.dumps({key: report[key] for key in (
        "output_directory", "available_generated_videos", "pending_generated_videos",
        "inspected_explicit_pairs", "fixed_sample_reads", "rejected", "cannot_train")}, ensure_ascii=False))
    return 1 if report["rejected"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
