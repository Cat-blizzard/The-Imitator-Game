"""CPU-only geometric path checks and retrospective saved-command replay.

No planner or simulator is changed. Explicit desired end poses can be checked
by the library; old traces lack them, so retrospective endpoint checks stay
unknown. Controller stages select diagnostic rules, never contact labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np

from .planning_contract import WORKSPACE, bind, save_new


def validate_geometry(poses, *, workspace_min, workspace_max, desired_end_pose=None,
                      endpoint_tolerance_m=.01, endpoint_tolerance_rad=.10,
                      xy_center=None, xy_radius_m=None, maximum_z=None):
    poses = np.asarray(poses, dtype=float)
    if poses.ndim != 3 or poses.shape[1:] != (4,4) or not len(poses) or not np.isfinite(poses).all():
        raise ValueError("Expected finite nonempty [N,4,4] world TCP poses")
    xyz = poses[:,:3,3]
    low, high = np.asarray(workspace_min), np.asarray(workspace_max)
    if low.shape != (3,) or high.shape != (3,) or not np.all(low < high):
        raise ValueError("Invalid world workspace")
    failures = {}
    outside = np.any((xyz < low) | (xyz > high), axis=1)
    if outside.any():
        failures["outside_workspace"] = np.flatnonzero(outside).tolist()
    if xy_center is not None:
        if xy_radius_m is None or xy_radius_m <= 0:
            raise ValueError("XY corridor needs positive radius")
        distance = np.linalg.norm(xyz[:,:2]-np.asarray(xy_center), axis=1)
        if (distance > xy_radius_m).any():
            failures["outside_local_xy_corridor"] = np.flatnonzero(distance > xy_radius_m).tolist()
    if maximum_z is not None and (xyz[:,2] > maximum_z).any():
        failures["rises_above_descending_stage_envelope"] = np.flatnonzero(xyz[:,2] > maximum_z).tolist()
    endpoint = {"status": "unknown", "reason": "desired planner endpoint not supplied"}
    if desired_end_pose is not None:
        goal = np.asarray(desired_end_pose, dtype=float)
        if goal.shape != (4,4) or not np.isfinite(goal).all():
            raise ValueError("Invalid desired world endpoint")
        position = float(np.linalg.norm(xyz[-1]-goal[:3,3]))
        rotation = float(np.arccos(np.clip((np.trace(goal[:3,:3].T@poses[-1,:3,:3])-1)/2, -1, 1)))
        good = position <= endpoint_tolerance_m and rotation <= endpoint_tolerance_rad
        endpoint = {"status": "pass" if good else "fail", "position_error_m": position,
                    "rotation_error_rad": rotation}
        if not good:
            failures["endpoint_pose_error"] = [len(poses)-1]
    return {"geometry_proxy_passed": not failures, "failures": failures, "endpoint": endpoint,
            "path_length_m": float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()),
            "xyz_min_m": xyz.min(0).tolist(), "xyz_max_m": xyz.max(0).tolist(),
            "collision_verified": False, "live_execution_guard_installed": False}


def replay(case_dir):
    sys.path.insert(0, str(WORKSPACE/"PolicyGen"))
    from examples.baselines.hypericlzq.eef import panda_fk
    case_dir = Path(case_dir)
    proposal = json.loads((case_dir/"input.json").read_text())["proposal"]
    receipt_path = case_dir/"episode/receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        payload_path = Path(receipt["arrays"])
        with np.load(payload_path, allow_pickle=False) as archive:
            q = archive["states"].copy()
            commands = archive["native_commands"].copy()
            tcp = archive["tcp_world"].copy()
            stages = archive["stage"].copy()
            roots = archive["robot_roots_world"].copy()
        root_origin = "saved robot_roots_world"
    else:
        payload_path = case_dir/"failure/trace.npz"
        with np.load(payload_path, allow_pickle=False) as archive:
            q, commands, tcp, stages = (archive[k].copy() for k in ("states", "commands", "tcp", "stage"))
        initial_fk = panda_fk(np.stack([q[:1,:7], q[:1,9:16]],1))[0]
        roots = tcp[0]@np.linalg.inv(initial_fk)
        root_origin = "inferred from first measured TCP and independent joint FK; checked across full trace"
    measured_fk = roots[None]@panda_fk(np.stack([q[:,:7], q[:,9:16]],1))
    root_error = float(np.max(np.abs(measured_fk-tcp)))
    if root_error > 5e-6:
        raise ValueError("Root-frame reconstruction inconsistent with measured TCP")
    target_fk = roots[None]@panda_fk(np.stack([commands[:,:7], commands[:,8:15]],1))
    active = proposal["active_arm"]
    reports = []
    for stage in dict.fromkeys(stages):
        ids = np.flatnonzero(stages == stage)
        segment = target_fk[ids,active]
        settings = {"workspace_min": [-.35,-.55,-.03], "workspace_max": [.30,.55,.65]}
        if stage == "release_above_rim":
            settings.update(xy_center=proposal["roles"]["destination_container"]["center"][:2],
                            xy_radius_m=.12, maximum_z=float(segment[0,2,3]+.04))
        report = validate_geometry(segment, **settings)
        report["controller_phase"] = str(stage)
        report["source_frame_interval_inclusive"] = [int(ids.min()), int(ids.max())]
        report["rules"] = settings
        report["source_frame_failures"] = {k: ids[v].tolist() for k,v in report["failures"].items()}
        reports.append(report)
    return {
        "schema": "retrospective_command_geometry_v1", "input": bind(case_dir/"input.json"),
        "raw_payload": bind(payload_path), "robot_root_provenance": root_origin,
        "max_measured_fk_world_error": root_error, "active_arm": active,
        "stages": reports, "phases_with_proxy_failures": [r["controller_phase"] for r in reports if r["failures"]],
        "limits": {
            "scope": "retrospective diagnostic; provisional object-relative corridor and world work region",
            "desired_stage_endpoint_logged": False,
            "full_planned_unexecuted_path_available": False,
            "planner_mode_logged": False, "live_guard_installed": False,
            "collision_verified": False, "training_admitted": False,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = replay(args.case)
    save_new(args.output, report)
    print(json.dumps({"output": str(args.output), "phases_with_proxy_failures": report["phases_with_proxy_failures"],
                      "max_fk_error": report["max_measured_fk_world_error"]}))


if __name__ == "__main__":
    main()
