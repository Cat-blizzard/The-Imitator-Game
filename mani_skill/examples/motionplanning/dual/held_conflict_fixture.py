"""Fixed joint round trips for two already grasped and lifted payloads.

Contract: canonical left/right, controller-order radians[2,N,7], seconds[N],
world metres/WXYZ, constant CLOSED=-1. A seed-0-only bounded development
search may use private MPLib IK. Formal cases load its fixed middle joints;
only the measured source starts and the same analytic time law change.
No environment is constructed, reset, stepped, attached, detached or assigned.
All geometry gates remain sampled checks, not continuous safety certificates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
from typing import Any

import numpy as np

from .combined_collision import validate_schedule
from .conflict_coordination import search_start_delays
from .path_conflict_fixture import quintic_round_trip
from .trajectory_schedule import ArmTrajectory, build_hold_schedule

ROOT = Path(__file__).resolve().parents[7]
EVIDENCE = ROOT / "infrastructure/validation/dual-arm/held-path-conflict-20261116"
FIXTURE = EVIDENCE / "fixture-v3.npz"
METADATA = EVIDENCE / "fixture-v3.json"
OLD_FIXTURE = ROOT / "infrastructure/validation/dual-arm/path-conflict-20261116/fixture.npz"
OLD_METADATA = OLD_FIXTURE.with_suffix(".json")
VERSION = "fixed-held-middle-quintic-v3-separated-y"
DT, RESERVE_STEPS, MAX_DELAY_TICKS, DELAY_STRIDE = .05, 26, 40, 5
# Revision after the preserved 25-candidate v1 exhaustion: the banana stays
# in its measured orientation, with its TCP higher so its payload can share
# the lower arm's palm height while both palm geometries remain separated.
# Every candidate is fixed before this development attempt; no seed tuning.
GEOMETRY_GRID = tuple((-.15, 0., .22, gap, direction, 0., "x", dz, side_y)
                      for side_y in (.04, .06) for gap in (.14, .16, .18, .20)
                      for dz in (.08, .10) for direction in (-1., 1.))


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value).copy()
    if result.dtype.kind not in "biuf" or not np.isfinite(result).all():
        raise ValueError("Physical arrays must be finite numeric data")
    return result


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _flatten(value: Any, prefix: str = "") -> dict[str, np.ndarray]:
    if isinstance(value, dict):
        result: dict[str, np.ndarray] = {}
        for key, child in value.items():
            result.update(_flatten(child, prefix + "/" + str(key)))
        return result
    return {prefix: _array(value)}


def _context(solvers: Any, raw: Any, checker: Any) -> dict[str, Any]:
    if not isinstance(solvers, (tuple, list)) or len(solvers) != 2:
        raise ValueError("Two canonical solvers required")
    agents = tuple(raw.agent.agents)
    keys = tuple(raw.agent.agents_dict)
    if (len(agents) != 2 or len(keys) != 2
            or any(raw.agent.agents_dict[key] is not agent for key, agent in zip(keys, agents))
            or any(solver.env_agent is not agent or solver.base_env is not raw
                   for solver, agent in zip(solvers, agents))):
        raise ValueError("Physical left/right solver and action identities differ")
    if (not checker.attached or len(checker.attachments) != 2
            or len(checker._supports) != 2):
        raise ValueError("Both existing payload captures are required; do not attach again")
    if len(checker.targets) != 2:
        raise ValueError("Two real held targets required")
    for agent, target in zip(agents, checker.targets):
        grasping = _array(agent.is_grasping(target)).reshape(-1)
        if grasping.shape != (1,) or not np.isin(grasping, [0, 1]).all() or not bool(grasping[0]):
            raise ValueError("Both existing physical grasps must still be confirmed")
    if not np.isclose(float(raw.control_timestep), DT, rtol=0, atol=1e-12):
        raise ValueError("Expected 20 Hz control")
    starts, names, full_states, bases = [], [], [], []
    for agent in agents:
        actual = tuple(j.get_name() for j in agent.robot.get_active_joints())
        arm = tuple(agent.controller.controllers["arm"].config.joint_names)
        q = _array(agent.robot.get_qpos())
        if len(actual) != 9 or len(arm) != 7 or len(set(arm)) != 7 or q.shape != (1, 9):
            raise ValueError("Expected one nine-joint Panda and seven controller joints")
        arm_config = agent.controller.controllers["arm"].config
        if (agent.control_mode != "pd_joint_pos" or arm_config.normalize_action
                or arm_config.use_delta):
            raise ValueError("Absolute pd_joint_pos required")
        starts.append(q[0, [actual.index(name) for name in arm]])
        names.append(arm); full_states.append(q[0])
        bases.append(_array(agent.robot.pose.raw_pose).reshape(7))
    attachments = tuple(_array(a) for a in checker.attachments)
    if any(a.shape != (4, 4) for a in attachments):
        raise ValueError("Expected two captured homogeneous transforms")
    return {"starts": np.asarray(starts, dtype=float), "names": tuple(names), "keys": keys,
            "full_states": tuple(full_states), "base_poses": np.asarray(bases),
            "state": _flatten(raw.get_state_dict()), "elapsed": _array(raw.elapsed_steps),
            "attachments": attachments, "support_ids": tuple(id(s) for s in checker._supports),
            "permissions": tuple(checker.target_contacts_allowed)}


def _unchanged(context: dict[str, Any], raw: Any, checker: Any) -> None:
    after = _flatten(raw.get_state_dict())
    if (set(after) != set(context["state"])
            or any(not np.array_equal(after[k], context["state"][k]) for k in after)
            or not np.array_equal(_array(raw.elapsed_steps), context["elapsed"])):
        raise RuntimeError("Fixture geometry work changed physical state or elapsed steps")
    if (not checker.attached or tuple(id(s) for s in checker._supports) != context["support_ids"]
            or tuple(checker.target_contacts_allowed) != context["permissions"]
            or any(not np.array_equal(a, b) for a, b in zip(checker.attachments, context["attachments"]))):
        raise RuntimeError("Existing attachment capture or contact permissions changed")


def _budget(remaining_steps: int) -> int:
    if (isinstance(remaining_steps, bool) or not isinstance(remaining_steps, int)
            or not RESERVE_STEPS < remaining_steps <= 200):
        raise ValueError("Integer remaining budget must exceed the 26 reserved ticks and be <=200")
    return remaining_steps - RESERVE_STEPS


def _pair(context: dict[str, Any], middles: Any) -> tuple[ArmTrajectory, ArmTrajectory, dict[str, Any]]:
    goals = np.asarray(middles, dtype=float)
    if goals.shape != (2, 7) or not np.isfinite(goals).all():
        raise ValueError("Fixed middle joints must have finite shape[2,7]")
    times, positions, bounds = quintic_round_trip(context["starts"], goals)
    trajectories = tuple(ArmTrajectory(key, names, times, positions[i], -1.)
                         for i, (key, names) in enumerate(zip(context["keys"], context["names"])))
    kinematics = {key: bounds[key] for key in (
        "source", "outbound_duration_s", "round_trip_duration_s", "max_velocity_rad_s",
        "max_acceleration_rad_s2", "velocity_limit_rad_s", "acceleration_limit_rad_s2",
        "endpoint_velocity_and_acceleration", "source_dt_s", "motion_control_steps")}
    kinematics["source"] = "same analytic quintic joint round trip; fixed qmid and measured q0; not native planner output"
    return *trajectories, kinematics


def _yaw(quaternion: Any, angle: float) -> np.ndarray:
    q = np.asarray(quaternion, dtype=float).reshape(4)
    if not np.isfinite(q).all() or not np.isclose(np.linalg.norm(q), 1., atol=1e-5):
        raise ValueError("Normalized WXYZ orientation required")
    q = q / np.linalg.norm(q)
    c, s = np.cos(angle / 2), np.sin(angle / 2)
    w, x, y, z = q
    return np.array([c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w])


def _ik_middles(solvers: Any, raw: Any, context: dict[str, Any], geometry: Any) -> tuple[np.ndarray | None, dict[str, Any]]:
    x, y, z, gap, direction, right_yaw, axis, right_z_gain, side_y = geometry
    offset = np.array([direction * gap / 2, 0., 0.]) if axis == "x" else np.array([0., -direction * gap / 2, 0.])
    center = np.array([x, y, z])
    worlds = [np.r_[center + offset - [0.,side_y,right_z_gain/2], _array(raw.agent.agents[0].tcp.pose.q).reshape(4)],
              np.r_[center - offset + [0.,side_y,right_z_gain/2], _yaw(raw.agent.agents[1].tcp.pose.q, right_yaw)]]
    report: dict[str, Any] = {"world_tcp_targets_xyz_wxyz": [w.tolist() for w in worlds], "arms": []}
    middles = []
    for index, (solver, target) in enumerate(zip(solvers, worlds)):
        planner = solver.planner
        actual = tuple(j.get_name() for j in solver.robot.get_active_joints())
        planner_names = tuple(planner.user_joint_names)
        full = context["full_states"][index][[actual.index(n) for n in planner_names]].copy()
        arm_names = context["names"][index]
        arm_indices = [planner_names.index(n) for n in arm_names]
        mask = [name not in arm_names for name in planner_names]
        status, solutions = planner.IK(planner.transform_goal_to_wrt_base(target), full.copy(),
                                      mask=mask, n_init_qpos=4, threshold=1e-4)
        valid = [_array(q).reshape(9) for q in solutions]
        report["arms"].append({"status": status, "solution_count": len(valid),
                               "all_solutions_planner_order_rad": [q.tolist() for q in valid]})
        if status != "Success" or not valid:
            return None, report
        chosen = min(valid, key=lambda q: float(np.max(np.abs(q[arm_indices] - context["starts"][index]))))
        report["arms"][-1]["selected_solution_index"] = next(i for i, q in enumerate(valid) if q is chosen)
        middles.append(chosen[arm_indices].copy())
    return np.asarray(middles), report


def _cross_payload_witness(rows: Any, checker: Any) -> list[dict[str, Any]]:
    result = []
    groups = tuple(set(names) for names in checker._target_names)
    for row in rows:
        first, second = row.get("object_name1"), row.get("object_name2")
        if row.get("collision_type") == "payload_payload":
            result.append(row); continue
        for robot, other in ((first, second), (second, first)):
            if robot in ("left", "right"):
                index = 0 if robot == "left" else 1
                if other in groups[1 - index]:
                    result.append(row); break
    return result


class _RobotOnlyView:
    scope = "robot_self_and_interarm_subset_of_same_full_held_checker"

    def __init__(self, checker: Any) -> None:
        self.checker = checker

    def check(self, left7: np.ndarray, right7: np.ndarray) -> list[dict[str, Any]]:
        return [row for row in self.checker.check(left7, right7)
                if row.get("object_name1") in ("left", "right")
                and row.get("object_name2") in ("left", "right")]


def _qualify(left: ArmTrajectory, right: ArmTrajectory, checker: Any, max_steps: int) -> dict[str, Any]:
    starts = (left.positions_rad[0], right.positions_rad[0])
    holds = tuple(ArmTrajectory(t.action_key, t.joint_names, [0.], [q], -1.)
                  for t, q in zip((left, right), starts))
    report: dict[str, Any] = {"status": "rejected", "checks": {}, "cross_payload_witness": []}
    for name, pair in (("left_other_at_start", (left, holds[1])),
                       ("right_other_at_start", (holds[0], right)), ("synchronous", (left, right))):
        schedule = build_hold_schedule(*pair, control_timestep_s=DT, max_steps=max_steps)
        report["checks"][name] = validate_schedule(schedule, *starts, checker)
        if name != "synchronous" and report["checks"][name]["status"] != "passed":
            report["reason"] = name + "_collision"; return report
    synchronous = report["checks"]["synchronous"]
    conflict = synchronous["first_collision"]
    if not conflict or not 0 < conflict["time_s"] < max(left.duration_s, right.duration_s):
        report["reason"] = "no_strictly_midroute_conflict"; return report
    report["cross_payload_witness"] = _cross_payload_witness(conflict["collisions"], checker)
    if not report["cross_payload_witness"]:
        report["reason"] = "first_conflict_has_no_crossarm_payload_witness"; return report
    schedule = build_hold_schedule(left, right, control_timestep_s=DT, max_steps=max_steps)
    robot_only = validate_schedule(schedule, *starts, _RobotOnlyView(checker))
    report["checks"]["synchronous_robot_only"] = robot_only
    if robot_only["status"] != "passed":
        report["reason"] = "synchronous_robot_only_collision"; return report
    end_collisions = checker.check(left.positions_rad[-1], right.positions_rad[-1])
    report["end_collisions"] = end_collisions
    if end_collisions:
        report["reason"] = "common_endpoint_collision"; return report
    resolved, search = search_start_delays(left, right, checker, control_timestep_s=DT,
        max_delay_ticks=MAX_DELAY_TICKS, delay_step_ticks=DELAY_STRIDE, max_steps=max_steps)
    report["delay_search"] = search
    if resolved is None:
        report["reason"] = "bounded_start_wait_has_no_solution"; return report
    report.update(status="passed", reason=None, selected_delay_ticks=search["selected_delay_ticks"],
                  coordinated_schedule_steps=resolved.num_steps)
    return report


def _load_fixed(context: dict[str, Any], max_steps: int) -> tuple[ArmTrajectory, ArmTrajectory, dict[str, Any]]:
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    if (metadata.get("fixture_version") != VERSION
            or _sha(FIXTURE) != metadata.get("fixture_npz_sha256")
            or metadata.get("fixture_source_sha256") != _sha(Path(__file__))):
        raise ValueError("Frozen held fixture version or NPZ identity differs")
    with np.load(FIXTURE, allow_pickle=False) as data:
        middles, bases = data["middles"].copy(), data["base_poses"].copy()
    if (metadata["joint_names"] != [list(n) for n in context["names"]]
            or metadata["action_keys"] != list(context["keys"])
            or bases.shape != (2, 7)
            or not np.allclose(bases, context["base_poses"], rtol=0, atol=1e-6)):
        raise ValueError("Formal robot roots or joint/action identities differ from frozen recipe")
    left, right, kinematics = _pair(context, middles)
    schedule = build_hold_schedule(left, right, control_timestep_s=DT, max_steps=max_steps)
    return left, right, {"fixture_version": VERSION, "mode": "fixed_recipe_no_ik_no_geometry_search",
        "fixed_recipe_json": str(METADATA), "fixed_recipe_json_sha256": _sha(METADATA),
        "fixture_npz_sha256": _sha(FIXTURE), "fixed_middle_joints_rad": middles.tolist(),
        "measured_start_joints_rad": context["starts"].tolist(), "kinematics": kinematics,
        "uncoordinated_schedule_steps": schedule.num_steps, "max_schedule_steps": max_steps,
        "reserved_steps": RESERVE_STEPS, "gripper_commands": [-1., -1.],
        "per_case_full_collision_and_wait_validation_required": True,
        "development_reference_checks": metadata["checks"], "environment_steps": 0}


def build_held_fixture(solvers: Any, raw: Any, checker: Any, output_dir: Path | str,
                       remaining_steps: int, *, develop: bool = False
                       ) -> tuple[ArmTrajectory, ArmTrajectory, dict[str, Any]]:
    """Develop only at seed0, otherwise load qmid; never recapture attachments.

    Reserve 20 final common hover ticks, at most5 settling ticks, and one
    termination margin. Formal cases receive no IK or geometry search. The
    caller still checks the current held scene and selects initial waits.
    """
    if type(develop) is not bool:
        raise ValueError("develop must be an actual bool")
    maximum = _budget(remaining_steps)
    context = _context(solvers, raw, checker)
    try:
        elapsed = context["elapsed"].reshape(-1)
        if (elapsed.shape != (1,) or float(elapsed[0]) != int(elapsed[0])
                or int(elapsed[0]) < 0 or int(elapsed[0]) + remaining_steps > 200):
            raise ValueError("Remaining ticks must fit the original native 200-tick episode")
        if not develop:
            return _load_fixed(context, maximum)
        seed = _array(raw._episode_seed).reshape(-1)
        if seed.shape != (1,) or int(seed[0]) != 0:
            raise ValueError("Bounded held fixture development is restricted to seed0")
        if FIXTURE.exists() or METADATA.exists():
            raise FileExistsError("Preserve the existing fixed held recipe; development cannot overwrite it")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        log = output / "held-fixture-exploration.json"
        if log.exists():
            raise FileExistsError("Preserve the previous fixture exploration log")
        report: dict[str, Any] = {"fixture_version": VERSION, "development_seed": 0,
            "created_at": datetime.now(timezone.utc).isoformat(), "status": "searching",
            "remaining_steps": remaining_steps, "reserved_steps": RESERVE_STEPS,
            "max_schedule_steps": maximum, "max_delay_ticks": MAX_DELAY_TICKS,
            "delay_step_ticks": DELAY_STRIDE, "measured_start_joints_rad": context["starts"].tolist(),
            "captured_attachment_T_tcp_objects": [a.tolist() for a in context["attachments"]],
            "target_contacts_allowed_unchanged": list(context["permissions"]),
            "fixed_geometry_grid": [list(g) for g in GEOMETRY_GRID], "attempts": [],
            "fixed_geometry_grid_fields": ["center_x_m","center_y_m","center_z_m","gap_m","x_direction","right_world_yaw_rad","gap_axis","right_minus_left_z_m","own_side_y_offset_m"],
            "environment_steps": 0, "physical_capture_repeated": False}
        _save(log, report)
        initial = checker.check(*context["starts"])
        report["initial_collisions"] = initial
        if initial:
            report.update(status="initial_collision"); _save(log, report)
            raise ValueError("Lifted held fixture starts in a blocked configuration")
        old = json.loads(OLD_METADATA.read_text(encoding="utf-8"))
        if _sha(OLD_FIXTURE) != old["fixture_npz_sha256"]:
            raise ValueError("Read-only empty-hand reference identity differs")
        with np.load(OLD_FIXTURE, allow_pickle=False) as data:
            old_middle = data["middles"].copy()
        report["old_fixture_npz_sha256"] = _sha(OLD_FIXTURE)
        for index in range(1 + len(GEOMETRY_GRID)):
            attempt: dict[str, Any] = {"index": index, "status": "running",
                "kind": "old_fixed_qmid" if index == 0 else "fixed_geometry_ik",
                "geometry_index": None if index == 0 else index - 1}
            report["attempts"].append(attempt); _save(log, report)
            try:
                if index == 0:
                    middles = old_middle
                else:
                    middles, ik = _ik_middles(solvers, raw, context, GEOMETRY_GRID[index - 1])
                    attempt["ik"] = ik
                if middles is None:
                    attempt["status"] = "ik_failed"; _save(log, report); continue
                left, right, kinematics = _pair(context, middles)
                attempt.update(middle_joints_rad=middles.tolist(), kinematics=kinematics)
                if kinematics["motion_control_steps"] > maximum:
                    attempt["status"] = "source_exceeds_reserved_budget"; _save(log, report); continue
                qualification = _qualify(left, right, checker, maximum)
                attempt.update(qualification=qualification, status=qualification["status"])
                _unchanged(context, raw, checker); _save(log, report)
                if qualification["status"] != "passed":
                    continue
                EVIDENCE.mkdir(parents=True, exist_ok=True)
                if FIXTURE.exists() or METADATA.exists():
                    raise FileExistsError("Fixed held recipe appeared during development")
                with FIXTURE.open("xb") as stream:
                    np.savez(stream, middles=middles, starts=context["starts"], base_poses=context["base_poses"],
                             times_s=left.times_s, positions_rad=np.stack([left.positions_rad, right.positions_rad]),
                             attachment_T_tcp_objects=np.stack(context["attachments"]))
                report.update(status="selected", selected_attempt=index, fixed_recipe_json=str(METADATA))
                _save(log, report)
                metadata = {"fixture_version": VERSION, "development_seed": 0,
                    "created_at": datetime.now(timezone.utc).isoformat(), "selected_attempt": index,
                    "fixture_npz_sha256": _sha(FIXTURE), "fixture_source_sha256": _sha(Path(__file__)),
                    "joint_names": [list(n) for n in context["names"]], "action_keys": list(context["keys"]),
                    "gripper_commands": [-1., -1.], "fixed_middle_joints_rad": middles.tolist(),
                    "kinematics": kinematics, "checks": qualification, "reserved_steps": RESERVE_STEPS,
                    "exploration_log": str(log), "exploration_log_sha256": _sha(log),
                    "environment_steps": 0, "physical_capture_repeated": False,
                    "collision_scope": "sampled full held scene; robot-only full path must also pass; not CCD"}
                _save(METADATA, metadata)
                return _load_fixed(context, maximum)
            except Exception as exc:
                attempt.update(status="error", exception_type=type(exc).__name__, exception=str(exc))
                report.update(status="error"); _save(log, report)
                raise
        report["status"] = "bounded_grid_exhausted"; _save(log, report)
        raise RuntimeError("No held fixture met all fixed geometry, payload and timing gates")
    finally:
        _unchanged(context, raw, checker)


def self_test() -> dict[str, Any]:
    """NumPy-only contracts, payload classification, delay recovery and loader."""
    checks = []
    names = tuple(f"joint{i}" for i in range(7))
    context = {"starts": np.zeros((2, 7)), "keys": ("left", "right"), "names": (names, names),
               "base_poses": np.array([[0., -.35, 0., 1., 0., 0., 0.],
                                       [0., .35, 0., 1., 0., 0., 0.]])}
    middles = np.zeros((2, 7)); middles[:, 0] = .4
    left, right, bounds = _pair(context, middles)
    assert left.gripper_command == right.gripper_command == -1.
    assert np.array_equal(left.positions_rad[0], left.positions_rad[-1])
    assert np.allclose(left.positions_rad[len(left.times_s) // 2], middles[0])
    assert max(bounds["max_velocity_rad_s"]) <= 1.5 and max(bounds["max_acceleration_rad_s2"]) <= 2.6
    velocity = np.diff(left.positions_rad, axis=0) / np.diff(left.times_s)[:, None]
    assert np.max(np.abs(velocity)) <= 1.5
    assert np.max(np.abs(np.diff(velocity, axis=0) / .01)) <= 2.6
    assert _budget(99) == 73
    checks.append("closed quintic endpoint and derivative contracts; 26 ticks reserved")

    class Synthetic:
        _target_names = ({"apple"}, {"banana"})
        scope = "synthetic_cross_payload_no_native_geometry"

        def __init__(self, mode: str = "payload") -> None:
            self.mode = mode

        def check(self, ql: np.ndarray, qr: np.ndarray) -> list[dict[str, Any]]:
            if self.mode == "error":
                raise RuntimeError("synthetic_checker_error")
            if .17 < ql[0] < .23 and .17 < qr[0] < .23:
                if self.mode == "robot":
                    return [{"collision_type": "robot_robot", "object_name1": "left", "object_name2": "right"}]
                return [{"collision_type": "payload_robot", "object_name1": "left", "object_name2": "banana"}]
            return []

    result = _qualify(left, right, Synthetic(), 73)
    assert result["status"] == "passed" and result["cross_payload_witness"]
    assert result["checks"]["synchronous_robot_only"]["status"] == "passed"
    assert result["checks"]["left_other_at_start"]["status"] == "passed"
    assert result["checks"]["right_other_at_start"]["status"] == "passed"
    assert result["selected_delay_ticks"] != [0, 0] and result["coordinated_schedule_steps"] <= 73
    assert _qualify(left, right, Synthetic("robot"), 73)["status"] == "rejected"
    assert not _cross_payload_witness([{"object_name1": "left", "object_name2": "apple"}], Synthetic())
    assert not _cross_payload_witness([{"object_name1": "table", "object_name2": "banana"}], Synthetic())
    assert _cross_payload_witness([{"collision_type": "native", "object_name1": "right", "object_name2": "apple"}], Synthetic())
    assert _cross_payload_witness([{"collision_type": "payload_payload"}], Synthetic())
    checks.append("independent safe, midroute cross-payload conflict, robot-only safe, finite wait recovers")

    global FIXTURE, METADATA
    original = FIXTURE, METADATA
    try:
        with tempfile.TemporaryDirectory() as folder:
            FIXTURE, METADATA = Path(folder) / "fixture.npz", Path(folder) / "fixture.json"
            np.savez(FIXTURE, middles=middles, base_poses=context["base_poses"])
            _save(METADATA, {"fixture_version": VERSION, "fixture_npz_sha256": _sha(FIXTURE),
                "fixture_source_sha256": _sha(Path(__file__)),
                "joint_names": [list(names), list(names)], "action_keys": ["left", "right"], "checks": result})
            formal_left, formal_right, metadata = _load_fixed(context, 73)
            assert np.array_equal(formal_left.positions_rad, left.positions_rad)
            assert metadata["mode"] == "fixed_recipe_no_ik_no_geometry_search"
            changed = {**context, "starts": np.full((2, 7), .01)}
            changed_left, _, _ = _load_fixed(changed, 73)
            assert np.array_equal(changed_left.positions_rad[0], changed["starts"][0])
            assert np.allclose(changed_left.positions_rad[len(changed_left.times_s) // 2], middles[0])
            assert formal_right.gripper_command == -1.
            class Robot:
                def __init__(self, base: np.ndarray) -> None:
                    self.q = np.r_[np.zeros(7), .02, .02][None, :]
                    self.pose = NS(raw_pose=base[None, :])

                def get_qpos(self) -> np.ndarray:
                    return self.q.copy()

                def get_active_joints(self) -> list[Any]:
                    return [NS(get_name=lambda n=n: n) for n in (*names, "finger1", "finger2")]

            agents = tuple(NS(robot=Robot(base), control_mode="pd_joint_pos",
                controller=NS(controllers={"arm": NS(config=NS(joint_names=names, normalize_action=False, use_delta=False))}),
                is_grasping=lambda target: np.array([True])) for base in context["base_poses"])
            raw = NS(agent=NS(agents=agents, agents_dict=dict(zip(("left", "right"), agents))),
                     control_timestep=DT, elapsed_steps=np.array([101]), _episode_seed=np.array([31]))
            raw.get_state_dict = lambda: {"robot": np.stack([a.robot.q for a in agents]),
                                          "elapsed": raw.elapsed_steps.copy()}
            solvers = tuple(NS(env_agent=a, base_env=raw) for a in agents)
            checker = NS(attached=True, attachments=(np.eye(4), np.eye(4)), _supports=(object(), object()),
                         targets=(object(), object()), target_contacts_allowed=(False, False))
            full_context = _context(solvers, raw, checker)
            formal_left, _, formal = build_held_fixture(solvers, raw, checker, folder, 99)
            assert formal_left.gripper_command == -1. and formal["max_schedule_steps"] == 73
            _unchanged(full_context, raw, checker)
            for options, remaining in (({"develop": True}, 99), ({}, 100)):
                try:
                    build_held_fixture(solvers, raw, checker, folder, remaining, **options)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Formal seed development or enlarged episode budget accepted")
            checker.attachments[0][0, 3] = .1
            try:
                _unchanged(full_context, raw, checker)
            except RuntimeError:
                pass
            else:
                raise AssertionError("Attachment recapture/mutation escaped the invariant")
            checker.attachments[0][0, 3] = 0.
            checks.append("full formal API performs no IK or physical writes; seed restriction and attachment identity enforced")
            FIXTURE.write_bytes(FIXTURE.read_bytes() + b"changed")
            try:
                _load_fixed(context, 73)
            except ValueError:
                pass
            else:
                raise AssertionError("Changed fixture identity accepted")
    finally:
        FIXTURE, METADATA = original
    checks.append("formal fixed qmid loader uses measured q0, preserves CLOSED, rejects changed identity")
    rejected = 0
    for value in (True, 26, 0, 201, 99.):
        try:
            _budget(value)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError("Invalid remaining budget accepted")
    try:
        _qualify(left, right, Synthetic("error"), 73)
    except RuntimeError as exc:
        assert str(exc) == "synthetic_checker_error"
    else:
        raise AssertionError("Checker exception swallowed")
    assert np.allclose(_yaw([0., 1., 0., 0.], np.pi / 2), [0., np.sqrt(.5), np.sqrt(.5), 0.])
    checks.append("invalid budgets and checker exceptions fail closed; WXYZ world-yaw composition")
    return {"status": "passed", "checks": checks, "invalid_budget_checks": rejected,
            "source_shape": [2, len(left.times_s), 7], "source_control_steps": bounds["motion_control_steps"],
            "synthetic_selected_delay_ticks": result["selected_delay_ticks"],
            "environment_constructions": 0, "physics_steps": 0, "native_mplib_executed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(self_test(), indent=2, allow_nan=False))
