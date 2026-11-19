"""Read-only source/split inventory of existing strict apple execution records.

A shared task contract can enumerate several sources. Resolve the one selected
by the actual receipt, never count the whole contract as the execution's source.
This does not redo physical/visual acceptance or infer novel behavior from hashes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np

from .planning_contract import bind, check_binding, save_new, validate_contract
from ..r2h_pipeline.common import json_sha256

FIELDS = ("native_commands", "states", "object_states", "timestamps_seconds")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resolve_selected_source(metadata, receipt):
    """Semantic checks independent of file hashing; rehashed mismatches still fail."""
    proposal, contract = metadata["proposal"], metadata["contract"]
    require(receipt["randomization"].get("structured_proposal") == proposal,
            "Input/receipt selected proposal differs")
    matches = [w for w in contract["human_windows"] if w["id"] == proposal.get("human_window_id")]
    require(len(matches) == 1, "Selected human window must resolve exactly once")
    selected = matches[0]
    require(proposal.get("human_video") == selected["video"], "Selected human video differs")
    if "human_source_window" in proposal:
        require(proposal["human_source_window"] == selected, "Embedded source/session differs")
    require(receipt["split"] == proposal["split"] == selected["split"],
            "Source and execution dataset split differ")
    require(selected.get("session"), "Missing source session identity")
    return selected


def load_attempt(folder):
    folder = Path(folder).resolve(strict=True)
    result_path, input_path = folder / "result.json", folder / "input.json"
    result, metadata = [json.loads(p.read_text()) for p in (result_path, input_path)]
    check_binding(result["receipt"])
    receipt_path = Path(result["receipt"]["path"]).resolve(strict=True)
    require(receipt_path == folder / "episode/receipt.json", "Receipt from a different attempt")
    receipt = json.loads(receipt_path.read_text())
    verifier = result["native_verifier"]
    require(result.get("passed") is True and verifier.get("passed") is True,
            "Inventory of qualified attempts requires recorded native acceptance")
    require(verifier["receipt_sha256"] == result["receipt"]["sha256"],
            "Native verification binds a different receipt")
    validate_contract(metadata["contract"])
    selected = resolve_selected_source(metadata, receipt)
    arrays_path = Path(receipt["arrays"]).resolve(strict=True)
    require(arrays_path == receipt_path.parent / "arrays.npz", "Arrays from a different attempt")
    arrays_binding = {"path": str(arrays_path), "sha256": receipt["sha256"]["arrays.npz"]}
    check_binding(arrays_binding)
    fingerprint = hashlib.sha256()
    shapes = {}
    with np.load(arrays_path, allow_pickle=False) as arrays:
        for key in FIELDS:
            array = np.ascontiguousarray(arrays[key])
            require(len(array) == receipt["length"] and np.isfinite(array).all(),
                    "Nonfinite/misaligned recorded dynamics")
            fingerprint.update(array.tobytes())
            shapes[key] = {"shape": list(array.shape), "dtype": str(array.dtype)}
    execution_id, multiview = receipt.get("execution_id"), None
    if execution_id is not None:
        require(str(uuid.UUID(execution_id)) == metadata.get("execution_id"),
                "Execution UUID differs from input")
        check_binding(receipt["multiview"])
        multiview = json.loads(Path(receipt["multiview"]["path"]).read_text())
        require(multiview["execution_id"] == execution_id, "Multiview execution UUID differs")
        require(multiview["dynamic_trajectory_sha256"] == fingerprint.hexdigest(),
                "Multiview dynamics differ")
    return {
        "attempt": str(folder), "input": bind(input_path), "result": bind(result_path),
        "receipt": result["receipt"], "arrays": arrays_binding,
        "human_spec": metadata["contract"]["human_spec"], "selected_human_window": selected,
        "selection_evidence": "embedded_and_contract" if "human_source_window" in metadata["proposal"]
                              else "selected_id_and_video_resolved_from_frozen_contract",
        "dataset_split": receipt["split"], "execution_id": execution_id,
        "execution_identity_evidence": "shared_uuid" if execution_id else "legacy_attempt_record_no_uuid",
        "exact_dynamic_sha256": fingerprint.hexdigest(), "dynamic_array_signatures": shapes,
        "multiview_manifest": receipt.get("multiview"),
        "recorded_camera_streams": multiview["camera_streams"] if multiview else None,
        "recorded_viewpoints": multiview["distinct_viewpoints"] if multiview else None,
        "recorded_native_pass": True,
    }


def summarize(rows):
    """Enforce split isolation and avoid inflating counts by repeating an input."""
    seen_paths, seen_receipts, seen_uuids = set(), set(), set()
    assignments, sources, sessions, motions = {}, defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        path, receipt_sha, uid = row["attempt"], row["receipt"]["sha256"], row["execution_id"]
        require(path not in seen_paths and receipt_sha not in seen_receipts,
                "Duplicate attempt/receipt cannot count as an execution")
        if uid:
            require(uid not in seen_uuids, "Repeated execution UUID cannot count as a new execution")
            seen_uuids.add(uid)
        seen_paths.add(path)
        seen_receipts.add(receipt_sha)
        source = row["selected_human_window"]
        video_key = "human_video:" + source["video"]["sha256"]
        # Session names are scoped to the frozen source registry; they are not
        # global IDs or proof that an entire recording has been independently seen.
        session_key = "human_session:" + json_sha256([row["human_spec"]["sha256"], source["session"]])
        motion_key = "exact_motion:" + json_sha256([row["exact_dynamic_sha256"], row["dynamic_array_signatures"]])
        groups = [video_key, session_key, motion_key, "receipt:" + receipt_sha]
        if uid:
            groups.append("execution:" + uid)
        require(row["dataset_split"] in {"train", "val", "validation", "test"}, "Unknown dataset split")
        for group in groups:
            require(assignments.setdefault(group, row["dataset_split"]) == row["dataset_split"],
                    "Shared lineage crosses dataset splits: " + group)
        sources[video_key].append(path)
        sessions[session_key].append(path)
        motions[row["exact_dynamic_sha256"]].append(path)
    return {
        "qualified_attempt_records": len(rows), "explicit_execution_uuids": len(seen_uuids),
        "legacy_records_without_execution_uuid": len(rows) - len(seen_uuids),
        "selected_human_window_videos": len(sources), "registry_scoped_source_sessions": len(sessions),
        "exact_dynamic_byte_groups": len(motions), "source_groups": dict(sources),
        "session_groups": dict(sessions), "exact_dynamic_groups": dict(motions),
        "dataset_split_assignments": assignments,
        "new_behavior_coverage": "unknown_not_inferred_from_floating_point_or_hash_difference",
        "unseen_source_generalization": "unknown_requires_history_and_preregistered_holdout",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Use a new audit output; historical evidence is immutable")
    rows = [load_attempt(p) for p in args.attempt]
    report = {"schema": "strict_source_lineage_inventory_v1", "auditor": bind(__file__),
              "records": rows, "summary": summarize(rows), "dynamic_fields": list(FIELDS),
              "dynamic_hash_definition": "SHA256 concatenated C-contiguous bytes in listed order; no rounding",
              "scope": "Read-only provenance/counts; recorded physical acceptance, not a new physical replay",
              "source_sessions_scope": "session labels scoped by frozen human-spec hash, full recording hash unavailable",
              "api_calls": 0, "gpu_runs": 0, "training_admitted": False}
    save_new(args.output, report)
    print(json.dumps({"output": str(args.output), **{k: v for k, v in report["summary"].items()
                                                    if not isinstance(v, dict)}}))


if __name__ == "__main__":
    main()
