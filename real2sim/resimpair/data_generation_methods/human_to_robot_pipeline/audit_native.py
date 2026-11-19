"""Re-run independent native verifiers and summarize observed event visibility.

The event frame groups use controller phases only to locate inspection windows.
Actual object/support/grasp tests come from the separate native verifier.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from .planning_contract import WORKSPACE, bind, save_new


VERIFIERS = {
    "apple_bowl_transfer": "verify_bowl",
    "closed_book_to_open_box": "next_task_book_verify",
    "right_cabinet_open_only": "cabinet_open_atomic_production_verify_v1",
}


def audit(task, receipt_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(WORKSPACE / "PolicyGen"))
    started = time.time()
    receipt = json.loads(Path(receipt_path).read_text())
    module = importlib.import_module("examples.baselines.hypericl_data_factory."+VERIFIERS[task])
    physical = module.verify(receipt_path)
    with np.load(receipt["arrays"], allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    with np.load(receipt["qa_visibility"], allow_pickle=False) as archive:
        visibility = {k: archive[k] for k in archive.files}
    selected_name = {"apple_bowl_transfer": "apple", "closed_book_to_open_box": "closed_pocket_book",
                     "right_cabinet_open_only": "block"}[task]
    selected = next(i for i, asset in enumerate(receipt["asset_specs"]) if asset["name"] == selected_name)
    pixels = visibility["object_pixels_224"][:, selected]
    phases = arrays["stage"].astype(str)
    stages = []
    for stage in dict.fromkeys(phases):
        ids = np.flatnonzero(phases == stage)
        stages.append({
            "controller_phase": stage, "frame_range_inclusive": [int(ids.min()), int(ids.max())],
            "frames": len(ids), "selected_object_min_pixels_224": int(pixels[ids].min()),
            "selected_object_visible_ge16px_fraction": float((pixels[ids] >= 16).mean()),
            "phase_is_contact_gt": False,
        })
    return {
        "task": task, "receipt": bind(receipt_path), "verifier": bind(module.__file__),
        "raw_verification": physical, "frames": receipt["length"], "fps": receipt["fps"],
        "duration_seconds": receipt["length"]/receipt["fps"],
        "initial_selected_object_world_xyz_m": arrays["object_states"][0,selected,:3].tolist(),
        "final_selected_object_world_xyz_m": arrays["object_states"][-1,selected,:3].tolist(),
        "selected_object_name": selected_name,
        "selected_object_visible_ge16px_fraction": float((pixels >= 16).mean()),
        "controller_phase_inspection_windows": stages,
        "camera_randomization": receipt["randomization"]["camera"],
        "actual_object_count": len(receipt["asset_specs"]),
        "randomization": receipt["randomization"],
        "warning": ("Cabinet pixels refer to the internal block, not door or handle; use door_visibility "
                    "for door-specific QA." if task == "right_cabinet_open_only" else
                    "Phase averages may hide brief occlusions; physical success and visible evidence are separate."),
        "new_actual_video_review_completed": False, "training_admitted": False,
        "elapsed_seconds": time.time()-started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=list(VERIFIERS))
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.task, args.receipt)
    save_new(args.output, result)
    print(json.dumps({"output": str(args.output), "raw_passed": result["raw_verification"]["passed"],
                      "frames": result["frames"], "seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
