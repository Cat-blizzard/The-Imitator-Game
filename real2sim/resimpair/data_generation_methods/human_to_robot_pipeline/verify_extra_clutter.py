"""Independent CPU verification of experimental clutter contact sidecars.

This verifies recorded simulator forces and bindings. It does not assert exact
swept-collision completeness, render visibility, or general physical realism.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from .planning_contract import bind, check_binding, save_new


def verify(receipt_path, input_path):
    receipt = json.loads(Path(receipt_path).read_text())
    execution = json.loads(Path(input_path).read_text())
    proposal = execution["proposal"]
    if receipt["randomization"].get("structured_proposal") != proposal:
        raise ValueError("Recorded randomization differs from bound execution input")
    if receipt["layout_seed"] != proposal["seed"] or receipt["split"] != proposal["split"]:
        raise ValueError("Seed or split differs")
    for item in proposal["added_distractors"]:
        for path, digest in item["asset_files"].items():
            check_binding({"path": path, "sha256": digest})
    check_binding(receipt["extra_contact"])
    with np.load(receipt["extra_contact"]["path"], allow_pickle=False) as archive:
        force, timestamps = archive["force_N"], archive["timestamps_seconds"]
    if force.shape != (receipt["length"],) or timestamps.shape != force.shape:
        raise ValueError("Extra-contact timeline length differs")
    if not np.isfinite(force).all() or (force < 0).any() or not np.isfinite(timestamps).all():
        raise ValueError("Invalid accumulated contact magnitude or timestamps")
    np.testing.assert_allclose(timestamps, np.arange(len(force))/receipt["fps"], atol=1e-8, rtol=0)
    maximum = float(force.max(initial=0))
    if maximum >= .5:
        raise ValueError("Recorded robot/object–clutter contact exceeds fixed 0.5N threshold")
    if not np.isclose(receipt["max_extra_contact_force_N"], maximum, atol=1e-10, rtol=0):
        raise ValueError("Recorded summary force differs from raw sidecar")
    return {
        "passed": True, "receipt": bind(receipt_path), "execution_input": bind(input_path),
        "extra_contact": receipt["extra_contact"],
        "frames": len(force), "added_static_distractors": len(proposal["added_distractors"]),
        "max_robot_or_task_object_to_clutter_force_N": maximum, "fixed_threshold_N": .5,
        "scope": "Raw recorded force/time and source binding; static clutter only",
        "full_swept_collision_verified": False, "event_visibility_verified": False,
        "training_admitted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = verify(args.receipt, args.input)
    save_new(args.output, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
