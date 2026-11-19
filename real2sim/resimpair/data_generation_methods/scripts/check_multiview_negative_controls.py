"""Tamper copies of real evidence to exercise independent audit rejection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from data_generation_methods.human_to_robot_pipeline.audit_multiview import audit, bind


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def run(attempt, baseline):
    clean = audit(attempt, baseline)
    if not clean["synchronization_checks_passed"]:
        raise AssertionError("Real synchronization control must pass")
    mutations = (
        "shift_opposite_frame", "shift_timestamp", "skip_environment_step",
        "change_original_camera", "duplicate_opposite_camera", "wrong_execution",
        "stale_receipt_manifest_hash",
    )
    results = []
    for name in mutations:
        with tempfile.TemporaryDirectory(prefix="multiview_negative_") as temporary:
            target = Path(temporary)
            metadata = json.loads((attempt / "input.json").read_text())
            result = json.loads((attempt / "result.json").read_text())
            receipt = json.loads((attempt / "episode/receipt.json").read_text())
            manifest = json.loads((attempt / "episode/multiview/manifest.json").read_text())
            with np.load(manifest["timeline"]["path"], allow_pickle=False) as data:
                timeline = dict(data)
            if name == "shift_opposite_frame":
                timeline["camera_frame_index"][1:, 2] -= 1
            elif name == "shift_timestamp":
                timeline["timestamps_seconds"][10:] += 1 / manifest["fps"]
            elif name == "skip_environment_step":
                timeline["env_step_index"][10:] += 1
            elif name == "change_original_camera":
                manifest["cameras"]["native"]["camera_from_world"][0][3] += .1
            elif name == "duplicate_opposite_camera":
                manifest["cameras"]["opposite"]["camera_from_world"] = manifest["cameras"]["native"]["camera_from_world"]
            elif name == "wrong_execution":
                manifest["execution_id"] = "00000000-0000-0000-0000-000000000000"
            timeline_path = target / "timeline.npz"
            np.savez_compressed(timeline_path, **timeline)
            manifest["timeline"] = bind(timeline_path)
            manifest_path = target / "episode/multiview/manifest.json"
            write(manifest_path, manifest)
            receipt["multiview"] = bind(manifest_path)
            if name == "stale_receipt_manifest_hash":
                receipt["multiview"]["sha256"] = "0" * 64
            receipt_path = target / "episode/receipt.json"
            write(receipt_path, receipt)
            result["receipt"] = bind(receipt_path)
            write(target / "input.json", metadata)
            write(target / "result.json", result)
            # Read-only original visibility evidence; no historical file is edited.
            (target / "episode/qa_visibility.npz").write_bytes((attempt / "episode/qa_visibility.npz").read_bytes())
            try:
                audit(target, baseline)
            except (ValueError, AssertionError) as exc:
                results.append({"mutation": name, "rejected": True, "exception": type(exc).__name__,
                                "reason": str(exc).splitlines()[0] or "array comparison failed"})
            else:
                results.append({"mutation": name, "rejected": False})
    return {"schema": "real_multiview_negative_controls_v1", "attempt": bind(attempt / "result.json"),
            "auditor": clean["auditor"], "test_runner": bind(__file__),
            "clean_synchronization_passed": True,
            "clean_historical_bit_identity_passed": clean["historical_bit_identity_passed"],
            "cases": results, "all_mutations_rejected": all(row["rejected"] for row in results),
            "scope": "Real artifact integrity and temporal/calibration gate regression, not visual quality accuracy",
            "new_physical_executions": 0, "new_api_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.attempt, args.baseline)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"all_mutations_rejected": report["all_mutations_rejected"], "count": len(report["cases"])}))
    raise SystemExit(0 if report["all_mutations_rejected"] else 1)
