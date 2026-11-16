"""Native empty-hand path-conflict experiment using the existing paired executor.

The controlled fixture is an IK-defined joint excursion, not a basket-placement
benchmark. Both independent paths must pass with the other arm at its start;
their simultaneous combination must contain an actual native interarm conflict.
Only initial waiting may change. All unsuccessful candidates and executions are
retained. A 200-tick episode includes preparation and terminal joint settling.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import time
import traceback
from typing import Any

import numpy as np

from .trajectory_schedule import ArmTrajectory, build_hold_schedule
from .combined_collision import validate_schedule
from .conflict_coordination import search_start_delays
from .validate_grasp import TASK, json_value, recording_summary, save_report

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[6]
CONFIGURATION = {
    "task": TASK, "fixture_kind": "empty_hand_ik_joint_round_trip",
    "max_episode_steps": 200, "preparation_steps": 10,
    "settle_max_steps": 25, "stable_steps": 20,
    "maximum_schedule_steps": 165, "max_delay_ticks": 100,
    "delay_step_ticks": 10, "max_joint_step_rad": .02,
    "max_samples": 20000, "joint_vel_limit_rad_s": 1.5,
    "joint_acc_limit_rad_s2": 2.6, "gripper_commands": [1, 1],
    "control_freq_hz": 20, "physics_freq_hz": 100,
    "endpoint_limit_rad": .03, "stable_velocity_limit_rad_s": .05,
    "candidate_order": "zero, then (0,n),(n,0) for n=10,20,...,100",
    "continuous_collision_detection": False,
    "success_definition": "both empty hands finish their original round trips, settle together, no sampled collision or physical interarm contact",
}


def interarm_contact_rows(contacts: Any, identities: dict[Any, tuple[int, str]]) -> list[dict[str, Any]]:
    """Read all physical interarm contact manifolds, including zero impulses."""
    result = []
    for contact in contacts:
        sides = [identities.get(body.entity) for body in contact.bodies]
        if len(sides) != 2 or any(side is None for side in sides) or sides[0][0] == sides[1][0]:
            continue
        points = [{"position_m": np.asarray(p.position).tolist(),
                   "normal": np.asarray(p.normal).tolist(),
                   "impulse_Ns": np.asarray(p.impulse).tolist(),
                   "separation_m": float(p.separation)} for p in contact.points]
        result.append({"bodies": [{"robot_index": s[0], "link": s[1]} for s in sides], "points": points})
    return result


def _state_arrays(value: Any, prefix: str = "") -> dict[str, np.ndarray]:
    from .trajectory_execution import as_numpy
    if isinstance(value, dict):
        result: dict[str, np.ndarray] = {}
        for name, child in value.items():
            result.update(_state_arrays(child, prefix + "/" + str(name)))
        return result
    return {prefix: as_numpy(value).copy()}


def _assert_unchanged(before: dict[str, np.ndarray], raw: Any) -> None:
    after = _state_arrays(raw.get_state_dict())
    if set(before) != set(after) or any(not np.array_equal(before[k], after[k]) for k in before):
        raise RuntimeError("A zero-step check changed the physical state")


def _save_plan(output: Path, left: ArmTrajectory, right: ArmTrajectory, sync: Any,
               selected: Any | None = None) -> None:
    arrays = {"left_time": left.times_s, "left_position": left.positions_rad,
              "right_time": right.times_s, "right_position": right.positions_rad,
              "sync_target_times": sync.target_times_s, "sync_left_commands": sync.left_commands,
              "sync_right_commands": sync.right_commands}
    if selected is not None:
        arrays.update(selected_left_time=selected.trajectories[0].times_s,
                      selected_left_position=selected.trajectories[0].positions_rad,
                      selected_right_time=selected.trajectories[1].times_s,
                      selected_right_position=selected.trajectories[1].positions_rad,
                      selected_target_times=selected.target_times_s,
                      selected_left_commands=selected.left_commands,
                      selected_right_commands=selected.right_commands)
    np.savez(output / "trajectory-plan.npz", **arrays)


def outbound_only(trajectory: ArmTrajectory) -> ArmTrajectory:
    """Keep the first half of the symmetric fixture, holding its shared goal."""
    midpoint = trajectory.duration_s / 2
    indices = np.flatnonzero(np.isclose(trajectory.times_s, midpoint, rtol=0, atol=1e-10))
    if len(indices) != 1:
        raise ValueError("Round-trip fixture must contain its exact turnaround sample")
    stop = int(indices[0]) + 1
    return ArmTrajectory(trajectory.action_key, trajectory.joint_names,
                         trajectory.times_s[:stop], trajectory.positions_rad[:stop], trajectory.gripper_command)


def native_run(args: argparse.Namespace, output: Path, report: dict[str, Any]) -> None:
    import gymnasium as gym
    import imageio.v2 as imageio
    import mani_skill.envs
    from mani_skill.utils.wrappers.record import RecordEpisode
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
    from .trajectory_execution import as_numpy
    from .dual_arm_execution import execute_dual_reach
    from .dual_payload_collision import DualPayloadCollisionChecker
    from .path_conflict_fixture import build_fixture

    env, raw, original_hook = None, None, None
    frames: list[np.ndarray] = []
    ticks: list[dict[str, Any]] = []
    substeps: list[dict[str, Any]] = []
    keys: list[str] = []
    try:
        base = gym.make(TASK, obs_mode="none", reward_mode="none", control_mode="pd_joint_pos",
                        sim_backend="cpu", render_backend="gpu", render_mode="rgb_array", max_episode_steps=200)
        env = RecordEpisode(base, output_dir=str(output), trajectory_name="trajectory", save_video=False,
                            save_on_reset=False, clean_on_close=False, record_reward=False,
                            record_env_state=True, source_type="motionplanning",
                            source_desc="controlled empty-hand path conflict; finite initial-wait search")
        env.reset(seed=args.seed)
        raw = env.unwrapped
        agents, keys = tuple(raw.agent.agents), list(raw.agent.agents_dict)
        if not np.isclose(float(raw.control_timestep), .05) or not np.isclose(float(raw.sim_timestep), .01):
            raise RuntimeError("Unexpected simulation clock")
        identities = {obj.entity: (i, link.name) for i, agent in enumerate(agents)
                      for link in agent.robot.get_links() for obj in link._objs}
        report["backend"] = {"physics": raw.backend.sim_backend, "render": raw.backend.render_backend}

        class Capture:
            unwrapped = raw
            phase = "preparation"
            pending_control_step = 0
            pending_substep = 0

            def frame(self) -> None:
                value = as_numpy(env.render())
                if value.ndim == 4 and value.shape[0] == 1:
                    value = value[0]
                if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8:
                    raise ValueError("Expected HWC RGB uint8 render")
                frames.append(value.copy())

            def step(self, action: dict[str, Any]) -> tuple[Any, ...]:
                if set(action) != set(keys):
                    raise ValueError("Submit both physical arms on every control tick")
                self.pending_control_step = int(as_numpy(raw.elapsed_steps)[0]) + 1
                self.pending_substep = 0
                if self.pending_control_step > 200:
                    raise RuntimeError("Never advance beyond native tick 200")
                first_substep = len(substeps)
                result = env.step(action)
                _, _, done, truncated, info = result
                ticks.append({"phase": self.phase, "environment_step": int(as_numpy(raw.elapsed_steps)[0]),
                              "actions": json_value(action), "native_terminated": bool(as_numpy(done).item()),
                              "native_truncated": bool(as_numpy(truncated).item()), "official_evaluation": json_value(info),
                              "left_qpos": as_numpy(agents[0].robot.get_qpos())[0].tolist(),
                              "right_qpos": as_numpy(agents[1].robot.get_qpos())[0].tolist()})
                self.frame()
                if any(row["interarm_contacts"] for row in substeps[first_substep:]):
                    raise RuntimeError("Physical interarm contact observed; stop execution")
                return result

        captured = Capture()
        original_hook = raw._after_simulation_step

        def record_substep() -> None:
            original_hook()
            captured.pending_substep += 1
            substeps.append({"substep": len(substeps) + 1, "phase": captured.phase,
                             "substep_index": captured.pending_substep,
                             "control_step": captured.pending_control_step, "sim_timestep_s": float(raw.sim_timestep),
                             "left_qpos": as_numpy(agents[0].robot.get_qpos())[0].tolist(),
                             "right_qpos": as_numpy(agents[1].robot.get_qpos())[0].tolist(),
                             "left_qvel": as_numpy(agents[0].robot.get_qvel())[0].tolist(),
                             "right_qvel": as_numpy(agents[1].robot.get_qvel())[0].tolist(),
                             "interarm_contacts": interarm_contact_rows(raw.scene.px.get_contacts(), identities)})

        raw._after_simulation_step = record_substep
        captured.frame()
        hold = {k: np.r_[as_numpy(a.robot.get_qpos())[0, :7], 1].astype(np.float32)
                for k, a in raw.agent.agents_dict.items()}
        for _ in range(10):
            _, _, done, truncated, _ = captured.step(hold)
            if bool(as_numpy(done).item()) or bool(as_numpy(truncated).item()):
                raise RuntimeError("Native episode ended during preparation")
        solvers = tuple(PandaArmMotionPlanningSolver(env, base_pose=a.robot.pose, multi_robot_id=i,
                        joint_vel_limits=1.5, joint_acc_limits=2.6, debug=False, vis=False,
                        visualize_target_grasp_pose=False, print_env_info=False) for i, a in enumerate(agents))
        checker = DualPayloadCollisionChecker.from_solvers(raw, *solvers, targets=(raw.apple_out, raw.banana_out))
        checker.set_target_contacts_allowed((False, False))
        report["scene_coverage"] = checker.coverage
        before, before_steps = _state_arrays(raw.get_state_dict()), int(as_numpy(raw.elapsed_steps)[0])
        started = time.monotonic()
        left, right, metadata = build_fixture(solvers, raw, checker)
        report["fixture_metadata"] = metadata
        report["fixture_construction_wall_seconds"] = time.monotonic() - started
        _assert_unchanged(before, raw)
        sync = build_hold_schedule(left, right, control_timestep_s=.05, max_steps=165)
        _save_plan(output, left, right, sync)
        starts = (left.positions_rad[0], right.positions_rad[0])
        holds = tuple(ArmTrajectory(t.action_key, t.joint_names, [0.], [t.positions_rad[0]], 1) for t in (left, right))
        report["independent_validation"] = {}
        for name, pair in (("left", (left, holds[1])), ("right", (holds[0], right))):
            single = build_hold_schedule(*pair, control_timestep_s=.05, max_steps=165)
            report["independent_validation"][name] = validate_schedule(single, *starts, checker)
        report["synchronous_validation"] = validate_schedule(sync, *starts, checker)
        save_report(output / "report.json", report)
        if any(v["status"] != "passed" for v in report["independent_validation"].values()):
            raise RuntimeError("Fixture contains an unsafe individual trajectory")
        collision = report["synchronous_validation"].get("first_collision")
        if not collision or not any({c.get("object_name1"), c.get("object_name2")} == {"left", "right"}
                                    for c in collision["collisions"]):
            raise RuntimeError("Fixture does not expose a simultaneous native interarm conflict")
        captured.phase = "synchronous_rejected"
        report["blocked_execution"] = execute_dual_reach(captured, sync, checker, settle_max_steps=25, stable_steps=20)
        if report["blocked_execution"]["status"] != "rejected" or report["blocked_execution"].get("reason") != "candidate_collision":
            raise RuntimeError("Synchronous conflict was not rejected before execution")
        if int(as_numpy(raw.elapsed_steps)[0]) != before_steps or report["blocked_execution"]["environment_steps"] != 0:
            raise RuntimeError("Rejected path advanced the environment")
        _assert_unchanged(before, raw)
        rejected, negative = search_start_delays(left, right, checker, control_timestep_s=.05,
                        max_delay_ticks=0, delay_step_ticks=10, max_steps=165)
        report["zero_delay_negative"] = {**negative, "environment_steps": 0}
        if rejected is not None:
            raise RuntimeError("Zero-delay negative accepted a conflicting path")
        _assert_unchanged(before, raw)
        blocked_left, blocked_right = outbound_only(left), outbound_only(right)
        report["blocked_terminal_collisions"] = checker.check(blocked_left.positions_rad[-1], blocked_right.positions_rad[-1])
        if not any({c.get("object_name1"), c.get("object_name2")} == {"left", "right"}
                   for c in report["blocked_terminal_collisions"]):
            raise RuntimeError("Shared terminal goal must produce a native interarm conflict")
        blocked, terminal_negative = search_start_delays(blocked_left, blocked_right, checker,
                        control_timestep_s=.05, max_delay_ticks=100, delay_step_ticks=10, max_steps=165)
        report["blocked_goal_negative"] = {**terminal_negative, "environment_steps": 0,
                        "scope": "outbound halves end in the same occupied region; initial waiting cannot clear this terminal conflict"}
        if blocked is not None:
            raise RuntimeError("Waiting incorrectly accepted mutually colliding terminal goals")
        _assert_unchanged(before, raw)
        started = time.monotonic()
        selected, search = search_start_delays(left, right, checker, control_timestep_s=.05,
                        max_delay_ticks=100, delay_step_ticks=10, max_steps=165)
        report["coordination_search"] = search
        report["coordination_search_wall_seconds"] = time.monotonic() - started
        _assert_unchanged(before, raw)
        if selected is None:
            report.update(status="failed", reason="no_collision_free_start_delay")
            return
        _save_plan(output, left, right, sync, selected)
        captured.phase = "coordinated_reach"
        report["stage"] = "coordinated_execution"
        save_report(output / "report.json", report)
        report["execution"] = execute_dual_reach(captured, selected, checker, settle_max_steps=25, stable_steps=20)
        report["status"] = "passed" if report["execution"]["status"] == "passed" else "failed"
        report["scene_coverage_after_execution"] = checker.coverage
        report["stage"] = "finished"
    finally:
        errors = report.setdefault("artifact_errors", [])
        if raw is not None:
            report["total_environment_steps"] = int(as_numpy(raw.elapsed_steps)[0])
            if original_hook is not None:
                raw._after_simulation_step = original_hook
        for name, rows in (("ticks.jsonl", ticks), ("physics-substeps.jsonl", substeps)):
            with (output / name).open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(json_value(row), allow_nan=False) + "\n")
        report["physics_contact_audit"] = {"physical_substeps": len(substeps),
                     "interarm_contact_substeps": sum(bool(s["interarm_contacts"]) for s in substeps),
                     "expected_substeps": len(ticks) * 5, "all_control_ticks_have_five_substeps":
                     all(sum(s["control_step"] == t["environment_step"] for s in substeps) == 5 for t in ticks)}
        if env is not None:
            try:
                if getattr(env, "_trajectory_buffer", None) is not None:
                    env.flush_trajectory()
            except Exception as exc:
                errors.append(f"recording flush: {type(exc).__name__}: {exc}")
            finally:
                env.close()
            try:
                actions = np.asarray([np.concatenate([t["actions"][key] for key in keys]) for t in ticks])
                report["recording"] = recording_summary(output / "trajectory.h5", len(ticks), actions)
                report["recording"]["action_key_order"] = keys
            except Exception as exc:
                errors.append(f"recording validation: {type(exc).__name__}: {exc}")
        if frames:
            try:
                imageio.mimsave(output / "execution.mp4", [frames[0]] * 10 + frames + [frames[-1]] * 20,
                               fps=20, macro_block_size=1)
                imageio.imwrite(output / "preview.png", frames[-1])
            except Exception as exc:
                errors.append(f"video: {type(exc).__name__}: {exc}")
        if errors or any(s["interarm_contacts"] for s in substeps):
            report["status"] = "failed"


def self_test() -> dict[str, Any]:
    from types import SimpleNamespace as NS
    left, right, other = object(), object(), object()
    point = NS(position=np.zeros(3), normal=np.ones(3), impulse=np.zeros(3), separation=-.001)
    def contact(a: Any, b: Any) -> Any:
        return NS(bodies=(NS(entity=a), NS(entity=b)), points=[point])
    rows = interarm_contact_rows([contact(left, right), contact(right, left), contact(left, other), contact(left, left)],
                                {left: (0, "L"), right: (1, "R")})
    assert len(rows) == 2 and rows[0]["points"][0]["separation_m"] == -.001
    assert rows[0]["points"][0]["impulse_Ns"] == [0, 0, 0]
    assert CONFIGURATION["preparation_steps"] + CONFIGURATION["maximum_schedule_steps"] + CONFIGURATION["settle_max_steps"] == 200
    source = ArmTrajectory("left", tuple(f"j{i}" for i in range(7)), [0., 1., 2.],
                           np.array([[0.] * 7, [1.] * 7, [0.] * 7]), 1)
    first = outbound_only(source)
    assert np.array_equal(first.times_s, source.times_s[:2])
    assert np.array_equal(first.positions_rad, source.positions_rad[:2]) and first.gripper_command == 1
    assert ROOT.name == "ZeroWAM-for-TIG" and (ROOT / "infrastructure").is_dir()
    return {"status": "passed", "checks": ["physical interarm identity, both pair orders, zero impulse retained", "single-arm/world contacts excluded from interarm metric", "unchanged full episode budget", "terminal-blocked negative preserves source outbound prefix"]}


def main() -> int:
    faulthandler.enable()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2)); return 0
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not re.fullmatch(r"GPU-[0-9a-fA-F-]+", gpu):
        parser.error("Select one GPU UUID for rendering")
    output = args.output_dir or ROOT / "infrastructure/validation/dual-arm/path-conflict-20261116" / datetime.now(timezone.utc).strftime("case-%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    files = [p for p in HERE.glob("*.py") if p.name != "__init__.py"]
    report: dict[str, Any] = {"status": "running", "seed": args.seed, "configuration": CONFIGURATION,
                  "gpu_uuid": gpu, "started_at": datetime.now(timezone.utc).isoformat(),
                  "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                  "physical_pose_assignment": False, "continuous_collision_detection": False}
    started = time.monotonic()
    try:
        native_run(args, output, report)
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    report.update(completed_at=datetime.now(timezone.utc).isoformat(), wall_seconds=time.monotonic() - started)
    save_report(output / "report.json", report)
    print(json.dumps({"status": report["status"], "report": str(output / "report.json")}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
