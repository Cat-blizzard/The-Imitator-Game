"""Replay immutable development evidence; never report annotation replay as detection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import file_sha256, read_json, write_json
from .single_actor_acceptance import RULE_VERSION, SCHEMA_PATH, evaluate


def verify_binding(binding):
    path = Path(binding["path"]).resolve(strict=True)
    if file_sha256(path) != binding["sha256"]:
        raise ValueError(f"Immutable regression artifact changed: {path}")
    return path


def verify_all_bindings(value):
    """Hash every path/SHA pair, including sampled frames and source lineage."""
    if isinstance(value, dict):
        count = 0
        if "path" in value and "sha256" in value:
            verify_binding(value)
            count += 1
        return count + sum(verify_all_bindings(v) for v in value.values())
    if isinstance(value, list):
        return sum(verify_all_bindings(v) for v in value)
    return 0


def validate_lineage_splits(cases):
    assignments = {}
    for case in cases:
        split = case["split"]
        if split not in {"development", "validation"}:
            raise ValueError("Explicit research split is required")
        groups = case.get("lineage_groups", [])
        if not groups:
            raise ValueError("Source/physical execution grouping is required")
        for group in groups:
            old = assignments.setdefault(group, split)
            if old != split:
                raise ValueError(f"Derivative lineage crosses research splits: {group}")
    return assignments


def run(manifest_path, output, *, run_boundary=False):
    manifest_path, output = Path(manifest_path), Path(output)
    if output.exists():
        raise ValueError("Regression runs require a new output directory")
    manifest = read_json(manifest_path)
    if (manifest.get("schema") != "r2h_failure_regression_v1"
            or manifest.get("actor_rule_version") != RULE_VERSION):
        raise ValueError("Regression requires the exact frozen rule version")
    bindings = verify_all_bindings(manifest)
    assignments = validate_lineage_splits(manifest["cases"])
    source = Path(__file__).with_name("single_actor_acceptance.py")
    if file_sha256(source) != manifest["actor_gate_source"]["sha256"]:
        raise ValueError("Actor gate changed: freeze a new regression version and compare both arms")
    if file_sha256(SCHEMA_PATH) != manifest["actor_gate_schema"]["sha256"]:
        raise ValueError("Actor schema changed: freeze a new regression version")
    actor_results, boundary_results = [], []
    for case in manifest["cases"]:
        if "actor_review" not in case:
            continue
        report = read_json(verify_binding(case["actor_review"]))
        result = evaluate(report, artifact_path=case["media"]["path"])
        expected = case["expected_actor_decision"]
        actor_results.append({
            "case_id": case["case_id"], "split": case["split"], "status": result["status"],
            "expected_status": expected, "matches": result["status"] == expected,
            "decision": result,
        })
    if run_boundary:
        import cv2
        from dataclasses import asdict
        from .boundary_discontinuity import Config, diagnose
        boundary = manifest["boundary_regression"]
        detector_source = Path(__file__).with_name("boundary_discontinuity.py")
        if file_sha256(detector_source) != boundary["detector_source"]["sha256"]:
            raise ValueError("Frozen boundary detector changed")
        config = Config(**boundary["config"])
        cv2.setNumThreads(2)
        for case in manifest["cases"]:
            if "expected_boundary_candidate_frames" not in case:
                continue
            report = diagnose(case["media"]["path"], output / "boundary" / case["case_id"], config)
            actual = [r["frame_index"] for r in report["candidates"]]
            boundary_results.append({
                "case_id": case["case_id"], "candidate_frames": actual,
                "expected": case["expected_boundary_candidate_frames"],
                "matches": actual == case["expected_boundary_candidate_frames"],
                "source_frames_decoded": report["source"]["frames"],
                "all_adjacent_transitions_evaluated": report["all_adjacent_transitions_evaluated"],
                "elapsed_cpu_seconds": report["elapsed_cpu_seconds"],
                "config": asdict(config),
            })
    result = {
        "schema": "r2h_failure_regression_run_v1",
        "manifest": {"path": str(manifest_path.resolve()), "sha256": file_sha256(manifest_path)},
        "rule_version": RULE_VERSION, "verified_file_bindings": bindings,
        "lineage_groups": assignments, "actor_annotation_replay": actor_results,
        "boundary_actual_cpu_runs": boundary_results,
        "all_expected_decisions_reproduced": all(r["matches"] for r in actor_results + boundary_results),
        "automatic_actor_visual_detection_rate": None,
        "automatic_hidden_identity_detection_rate": None,
        "heldout_samples_evaluated": 0,
        "new_api_calls": 0, "new_gpu_jobs": 0,
        "quality_release_approved": False, "training_admitted": False,
        "limitations": [
            "Actor observations were authored after actual visual inspection of known failures.",
            "Replaying those observations validates routing and binding, not autonomous vision accuracy.",
            "Boundary detector was developed on this same apple failure; this is a regression, not a blind test.",
            "Hidden-object identity observations are direct comparisons; no automated identity detector is claimed.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-boundary", action="store_true")
    args = parser.parse_args()
    result = run(args.manifest, args.output, run_boundary=args.run_boundary)
    print(json.dumps({k: result[k] for k in (
        "verified_file_bindings", "all_expected_decisions_reproduced",
        "heldout_samples_evaluated", "new_api_calls", "new_gpu_jobs")}))
    return 0 if result["all_expected_decisions_reproduced"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
