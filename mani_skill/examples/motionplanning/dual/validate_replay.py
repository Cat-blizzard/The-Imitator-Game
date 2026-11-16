"""Validate saved states and initial-state-only action replay of the apple cycle.

This adapter supports the existing no-observation H5 recording and its flat
16-value action dataset. State playback never steps physics; action replay
restores frame zero only. Neither mode changes the input recording.
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
from typing import Any, Callable

import numpy as np

from .validate_grasp import TASK, json_value, save_report

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[7]
DEFAULT_INPUT = ROOT / "infrastructure/validation/dual-arm/place-positive-20261115T124829247240Z"
STATE_LIMITS = {"pose_translation_m": 1e-6, "pose_rotation_rad": 1e-5,
                "linear_velocity_m_s": 1e-5, "angular_velocity_rad_s": 1e-4,
                "arm_joint_max_rad": 1e-6, "finger_max_m": 1e-7, "joint_velocity_max": 1e-5}
ACTION_LIMITS = {"pose_translation_m": .005, "pose_rotation_rad": .05,
                 "arm_joint_max_rad": .01, "finger_max_m": .002}


def split_action(row: Any, keys: list[str]) -> dict[str, np.ndarray]:
    values = np.asarray(row, dtype=np.float32)
    if len(keys) != 2 or len(set(keys)) != 2 or values.shape != (16,) or not np.isfinite(values).all():
        raise ValueError("Expected two distinct ordered keys and one finite 16-value action")
    if not np.isin(values[[7, 15]], [-1, 1]).all():
        raise ValueError("This fixed-gripper recording requires OPEN/CLOSED commands")
    return {key: values[index * 8:(index + 1) * 8].copy() for index, key in enumerate(keys)}


def replay_sequence(recording: dict[str, Any], mode: str, *,
                    restore: Callable[[int, dict[str, Any]], None],
                    step: Callable[[dict[str, np.ndarray]], None],
                    observe: Callable[[int], None]) -> dict[str, int]:
    """Keep state-setting and physical stepping separate, including frame zero."""
    if mode not in ("state", "action"):
        raise ValueError("Unknown replay mode")
    states, actions = recording["states"], recording["actions"]
    if len(states) != len(actions) + 1 or not len(actions):
        raise ValueError("Replay requires T actions and T+1 states")
    resets, steps = 0, 0
    if mode == "state":
        for index, state in enumerate(states):
            restore(index, state)
            resets += 1
            observe(index)
    else:
        restore(0, states[0])
        resets += 1
        observe(0)
        for index, action in enumerate(actions, start=1):
            step(split_action(action, recording["action_keys"]))
            steps += 1
            observe(index)
    return {"state_restore_calls": resets, "physical_environment_steps": steps,
            "observed_frames": len(states)}


def within_limits(maxima: dict[str, Any], limits: dict[str, float]) -> bool:
    return all(name in maxima and np.isfinite(maxima[name]) and 0 <= maxima[name] <= limit
               for name, limit in limits.items())


def action_outcome(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate the already recorded phases without planning or future actions."""
    ticks = [row for row in rows if row["frame"] > 0]
    release = [row for row in ticks if row["phase"] == "release"]
    retreat = [row for row in ticks if row["phase"] == "retreat"]
    placement = [row for row in ticks if row["phase"] == "verify_placement"]
    checks = {"all_grasp_predicates_match": bool(ticks) and all(row["grasp_matches_reference"] for row in ticks),
              "release_open_ungrasped_last_three": len(release) >= 3 and all(
                  row["fingers_open"] and not row["is_grasping"] for row in release[-3:]),
              "recorded_twenty_tick_placement_window": len(placement) == 20,
              "placement_window_valid": len(placement) == 20 and all(row["placement_valid"] for row in placement)}
    rise = float(retreat[-1]["left_tcp_pose"][2] - release[-1]["left_tcp_pose"][2]) if retreat and release else 0.
    checks["retreat_rise_at_least_75mm"] = rise >= .075
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
            "actual_retreat_tcp_rise_m": rise, "release_steps": len(release), "placement_steps": len(placement),
            "grasp_mismatch_frames": [row["frame"] for row in ticks if not row["grasp_matches_reference"]],
            "invalid_placement_frames": [row["frame"] for row in placement if not row["placement_valid"]]}


def native_run(args: argparse.Namespace, recording: dict[str, Any], output: Path, report: dict[str, Any]) -> None:
    import gymnasium as gym
    import h5py
    import imageio.v2 as imageio
    import mani_skill.envs
    from PIL import Image, ImageDraw
    from .trajectory_execution import as_numpy
    from .replay_metrics import compare_states, pose_errors, summarize_errors
    from mani_skill.envs.tasks.tabletop.dual_tasks._008_pick_apple_banana_baskets import world_aabb_from_local_mesh

    env, raw = None, None
    frames: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    measured_states: list[dict[str, Any]] = []
    applied: list[np.ndarray] = []
    errors: list[dict[str, Any]] = []
    report["artifact_errors"] = []
    restore_events: list[int] = []
    report["state_restore_frames"] = restore_events

    def unbatch(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: unbatch(item) for key, item in value.items()}
        array = as_numpy(value)
        if array.ndim != 2 or array.shape[0] != 1 or not np.isfinite(array).all():
            raise ValueError("Expected one finite batched environment-state leaf")
        return array[0].copy()

    def batch(value: Any) -> Any:
        return {key: batch(item) for key, item in value.items()} if isinstance(value, dict) else np.asarray(value)[None].copy()

    try:
        metadata = recording["metadata"]
        kwargs = dict(metadata["env_info"]["env_kwargs"])
        kwargs["max_episode_steps"] = metadata["env_info"]["max_episode_steps"]
        env = gym.make(metadata["env_info"]["env_id"], **kwargs)
        env.reset(**metadata["episodes"][0]["reset_kwargs"])
        raw = env.unwrapped
        if list(raw.agent.agents_dict) != recording["action_keys"]:
            raise ValueError("Live arm action-key order differs from the recording")
        if not np.isclose(float(raw.control_timestep), .05, atol=1e-10, rtol=0):
            raise ValueError("Expected the original 20 Hz controller")
        report.update(backend={"physics": raw.backend.sim_backend, "render": raw.backend.render_backend},
                      control_timestep_s=float(raw.control_timestep), reset_calls=1)
        controller_configs: dict[str, Any] = {}
        for key, agent in raw.agent.agents_dict.items():
            if agent.control_mode != "pd_joint_pos":
                raise ValueError("This adapter replays absolute joint-position control only")
            controller_configs[key] = {}
            for name, controller in agent.controller.controllers.items():
                config = controller.config
                values = {field: bool(getattr(config, field)) for field in ("use_delta", "use_target", "interpolate", "normalize_action")}
                if any(values[field] for field in ("use_delta", "use_target", "interpolate")):
                    raise ValueError("Controller needs unrecorded cumulative/interpolated state")
                controller_configs[key][name] = values
        report["controller_configs"] = controller_configs
        initial_state = unbatch(raw.get_state_dict())
        report["fresh_reset_vs_source_initial"] = compare_states(recording["states"][0], initial_state)
        original_right = recording["states"][0]["articulations"]["panda_wristcam-agent-1"]
        placement_anchor: np.ndarray | None = None
        prior_apple: np.ndarray | None = None

        def restore(index: int, state: dict[str, Any]) -> None:
            if args.mode == "action" and (index != 0 or restore_events):
                raise RuntimeError("Action replay may restore only the initial frame")
            raw.set_state_dict(batch(state))
            restore_events.append(index)

        def step(action: dict[str, np.ndarray]) -> None:
            before = int(as_numpy(raw.elapsed_steps)[0])
            _, _, terminated, truncated, _ = env.step(action)
            applied.append(np.concatenate([action[key] for key in recording["action_keys"]]))
            if int(as_numpy(raw.elapsed_steps)[0]) != before + 1:
                raise RuntimeError("One submitted action did not advance exactly one control tick")
            if bool(as_numpy(terminated).reshape(-1)[0]) or bool(as_numpy(truncated).reshape(-1)[0]):
                raise RuntimeError("Replay reached an unexpected termination or time limit")

        def observe(index: int) -> None:
            nonlocal placement_anchor, prior_apple
            actual = unbatch(raw.get_state_dict())
            measured_states.append(actual)
            comparison = compare_states(recording["states"][index], actual)
            errors.append(comparison)
            phase = recording["ticks"][index - 1]["phase"] if index else "initial"
            row: dict[str, Any] = {"frame": index, "phase": phase,
                                   "environment_step": int(as_numpy(raw.elapsed_steps)[0]), "errors": comparison}
            tcp = as_numpy(raw.left_agent.tcp.pose.raw_pose)[0].copy()
            apple = as_numpy(raw.apple_out.pose.raw_pose)[0].copy()
            row.update(left_tcp_pose=tcp.tolist(), apple_pose=apple.tolist(),
                       left_finger_qpos_m=as_numpy(raw.left_agent.robot.get_qpos())[0, -2:].tolist(),
                       right_max_full_state_change=float(np.max(np.abs(
                           actual["articulations"]["panda_wristcam-agent-1"] - original_right))))
            if index:
                reference_tick = recording["ticks"][index - 1]
                row["left_tcp_errors"] = pose_errors(reference_tick["left_tcp_pose"], tcp)
            if args.mode == "action" and index:
                agent = raw.left_agent
                row["is_grasping"] = bool(as_numpy(agent.is_grasping(raw.apple_out)).reshape(-1)[0])
                row["reference_is_grasping"] = reference_tick["is_grasping"]
                row["grasp_matches_reference"] = row["is_grasping"] == row["reference_is_grasping"]
                forces = [as_numpy(raw.scene.get_pairwise_contact_forces(link, raw.apple_out))[0]
                          for link in (agent.finger1_link, agent.finger2_link)]
                row["finger_contact_force_norms_N"] = [float(np.linalg.norm(force)) for force in forces]
                row["fingers_open"] = all(np.allclose(as_numpy(arm.robot.get_qpos())[0, -2:],
                    as_numpy(arm.robot.get_qlimits())[0, -2:, 1], atol=.002, rtol=0) for arm in (raw.left_agent, raw.right_agent))
                row["official_evaluation"] = json_value(raw.evaluate())
                dt = float(raw.control_timestep)
                motion = pose_errors(prior_apple, apple)
                row["control_frame_linear_speed_m_s"] = motion["pose_translation_m"] / dt
                row["control_frame_angular_speed_rad_s"] = motion["pose_rotation_rad"] / dt
                row["apple_instant_linear_velocity_m_s"] = as_numpy(raw.apple_out.linear_velocity)[0].tolist()
                row["apple_instant_angular_velocity_rad_s"] = as_numpy(raw.apple_out.angular_velocity)[0].tolist()
                if phase == "verify_placement":
                    if placement_anchor is None:
                        placement_anchor = apple.copy()
                    drift = pose_errors(placement_anchor, apple)
                    bounds = as_numpy(world_aabb_from_local_mesh(raw.basket_apple, raw.device))[0]
                    center = as_numpy(raw._get_actor_geom_center(raw.apple_out))[0]
                    inside = bool(np.all(center[:2] >= bounds[0, :2]) and np.all(center[:2] <= bounds[1, :2])
                                  and bounds[0, 2] - .005 <= center[2] <= bounds[1, 2] + .01)
                    arm_speed = max(float(np.max(np.abs(as_numpy(arm.robot.get_qvel())[0, :7])))
                                    for arm in (raw.left_agent, raw.right_agent))
                    official_inside = bool(as_numpy(raw.evaluate()["apple_in_basket"]).reshape(-1)[0])
                    valid = (inside and official_inside and not row["is_grasping"] and row["fingers_open"]
                             and arm_speed <= .05 and row["control_frame_linear_speed_m_s"] <= .03
                             and row["control_frame_angular_speed_rad_s"] <= .2
                             and drift["pose_translation_m"] <= .003 and drift["pose_rotation_rad"] <= .03)
                    row.update(placement_valid=bool(valid), placement_world_drift=drift,
                               physical_region={"inside": inside, "object_geom_center_m": center.tolist(),
                                                "basket_aabb_m": bounds.tolist()}, max_arm_speed_rad_s=arm_speed)
            prior_apple = apple.copy()
            rows.append(row)
            frame = as_numpy(env.render())
            if frame.ndim == 4 and frame.shape[0] == 1:
                frame = frame[0]
            if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
                raise ValueError("Expected an HWC uint8 rendering")
            frames.append(frame.copy())

        report["sequence"] = replay_sequence(recording, args.mode, restore=restore, step=step, observe=observe)
        summary = summarize_errors(errors)
        report["trajectory_errors"] = summary
        limits = STATE_LIMITS if args.mode == "state" else ACTION_LIMITS
        checks = {"all_state_metrics_within_limits": within_limits(summary["global"], limits),
                  "all_frames_observed": len(rows) == len(recording["states"]),
                  "correct_setter_count": len(restore_events) == (len(rows) if args.mode == "state" else 1),
                  "correct_physics_steps": int(as_numpy(raw.elapsed_steps)[0]) == (0 if args.mode == "state" else len(recording["actions"]))}
        tcp_metrics = [row["left_tcp_errors"] for row in rows if "left_tcp_errors" in row]
        report["left_tcp_error_maxima"] = {key: max(row[key] for row in tcp_metrics) for key in ("pose_translation_m", "pose_rotation_rad")}
        checks["left_tcp_within_limits"] = within_limits(report["left_tcp_error_maxima"],
                                                        {key: limits[key] for key in report["left_tcp_error_maxima"]})
        if args.mode == "action":
            checks["actions_applied_exactly"] = np.array_equal(np.asarray(applied), recording["actions"])
            report["outcome"] = action_outcome(rows)
            checks["physical_outcome_reproduced"] = report["outcome"]["status"] == "passed"
        report["right_max_full_state_change"] = max(row["right_max_full_state_change"] for row in rows)
        checks["right_full_state_held"] = report["right_max_full_state_change"] <= 1e-6
        report["checks"] = checks
        report["status"] = "passed" if all(checks.values()) else "failed"
    finally:
        artifact_errors = report["artifact_errors"]
        if raw is not None:
            try:
                report["physical_environment_steps"] = int(as_numpy(raw.elapsed_steps)[0])
            except Exception as exc:
                artifact_errors.append(f"read final step count: {type(exc).__name__}: {exc}")
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                artifact_errors.append(f"environment close: {type(exc).__name__}: {exc}")
        try:
            with (output / "frames.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(json_value(row), allow_nan=False) + "\n")
            if measured_states:
                with h5py.File(output / "observed-states.h5", "w") as h:
                    h.attrs["validation_only"] = True
                    h.attrs["mode"] = args.mode
                    h.create_dataset("applied_actions", data=np.asarray(applied, dtype=np.float32).reshape(-1, 16))
                    for category in measured_states[0]:
                        for name in measured_states[0][category]:
                            h.create_dataset(f"env_states/{category}/{name}", data=np.stack(
                                [state[category][name] for state in measured_states]))
            report["recording"] = {"observed_state_frames": len(measured_states), "applied_action_steps": len(applied),
                                   "path": str(output / "observed-states.h5")}
        except Exception as exc:
            artifact_errors.append(f"measurement export: {type(exc).__name__}: {exc}")
        if frames:
            try:
                imageio.mimsave(output / "replay.mp4", [frames[0]] * 10 + frames + [frames[-1]] * 20,
                               fps=20, macro_block_size=1)
                imageio.imwrite(output / "preview.png", frames[-1])
                source = imageio.get_reader(Path(args.input_dir) / "execution.mp4")
                try:
                    with imageio.get_writer(output / "comparison.mp4", fps=20, macro_block_size=1) as writer:
                        for frame_index in [0] * 10 + list(range(len(frames))) + [len(frames) - 1] * 20:
                            original = source.get_data(frame_index + 10)
                            current = frames[frame_index]
                            if original.shape != current.shape:
                                raise ValueError("Original/replay video resolutions differ")
                            pair = Image.fromarray(np.concatenate([original, current], axis=1))
                            draw = ImageDraw.Draw(pair)
                            draw.rectangle((0, 0, pair.width, 27), fill="black")
                            draw.text((8, 7), f"Original | frame {frame_index}", fill="white")
                            draw.text((current.shape[1] + 8, 7), f"{args.mode.title()} replay", fill="white")
                            writer.append_data(np.asarray(pair))
                finally:
                    source.close()
                report["video"] = {"replay": str(output / "replay.mp4"), "comparison": str(output / "comparison.mp4"),
                                   "fps": 20, "padding_frames_before": 10, "padding_frames_after": 20,
                                   "source_video_padding_frames_before": 10}
            except Exception as exc:
                artifact_errors.append(f"video export: {type(exc).__name__}: {exc}")
        if artifact_errors:
            report["status"] = "failed"


def self_test() -> dict[str, Any]:
    keys = ["left", "right"]
    action = np.r_[np.zeros(7), -1, np.ones(7), 1].astype(np.float32)
    decoded = split_action(action, keys)
    assert list(decoded) == keys and decoded["left"].shape == decoded["right"].shape == (8,)
    decoded["left"][0] = .5
    assert action[0] == 0
    recording = {"actions": np.stack([action, action]), "states": [{"frame": i} for i in range(3)], "action_keys": keys}
    for mode in ("state", "action"):
        setters, steps, observed = [], [], []
        result = replay_sequence(recording, mode, restore=lambda i, state: setters.append(i),
                                 step=lambda value: steps.append(value), observe=lambda i: observed.append(i))
        assert setters == ([0, 1, 2] if mode == "state" else [0])
        assert len(steps) == (0 if mode == "state" else 2) and observed == [0, 1, 2]
        assert result["observed_frames"] == 3
    assert within_limits({"pose_translation_m": 0}, {"pose_translation_m": .001})
    assert not within_limits({"pose_translation_m": .01}, {"pose_translation_m": .001})
    assert not within_limits({"pose_translation_m": np.nan}, {"pose_translation_m": .001})
    assert not within_limits({}, {"pose_translation_m": .001})
    release = [{"frame": i, "phase": "release", "grasp_matches_reference": True,
                "fingers_open": True, "is_grasping": False, "left_tcp_pose": [0, 0, .1]} for i in range(1, 4)]
    retreat = [{"frame": 4, "phase": "retreat", "grasp_matches_reference": True, "left_tcp_pose": [0, 0, .18]}]
    placed = [{"frame": i, "phase": "verify_placement", "grasp_matches_reference": True,
               "placement_valid": True} for i in range(5, 25)]
    assert action_outcome(release + retreat + placed)["status"] == "passed"
    assert action_outcome(release + retreat + placed[:-1])["status"] == "failed"
    assert action_outcome([{**release[0], "is_grasping": True}] + release[1:] + retreat + placed)["status"] == "failed"
    assert action_outcome(release + retreat + [{**placed[0], "placement_valid": False}] + placed[1:])["status"] == "failed"
    assert (HERE.parents[3] / "mani_skill/envs/sapien_env.py").is_file()
    return {"status": "passed", "checks": ["ordered 16D action split and copy", "state playback performs zero physical steps",
                                             "action replay restores only frame zero", "T+1 frame alignment",
                                             "finite complete metric thresholds", "release/retreat/placement gate rejects failure"]}


def main() -> int:
    faulthandler.enable()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--mode", choices=("state", "action"), default="action")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return 0
    from .replay_metrics import load_recording
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not re.fullmatch(r"GPU-[0-9a-fA-F-]+", gpu):
        parser.error("Select one available GPU UUID for rendering")
    recording = load_recording(args.input_dir)
    if recording["report"].get("mode") != "positive" or recording["metadata"]["env_info"]["env_id"] != TASK:
        parser.error("This outcome validator requires the positive left-apple placement recording")
    output = args.output_dir or ROOT / "infrastructure/validation/dual-arm" / datetime.now(timezone.utc).strftime(
        f"replay-{args.mode}-%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    dev = HERE.parents[3]
    relative_sources = ["mani_skill/examples/motionplanning/dual/validate_replay.py",
                        "mani_skill/examples/motionplanning/dual/replay_metrics.py",
                        "mani_skill/examples/motionplanning/dual/validate_grasp.py",
                        "mani_skill/examples/motionplanning/dual/trajectory_execution.py",
                        "mani_skill/envs/sapien_env.py", "mani_skill/envs/scene.py",
                        "mani_skill/utils/structs/actor.py", "mani_skill/utils/structs/articulation.py",
                        "mani_skill/agents/controllers/pd_joint_pos.py",
                        "mani_skill/envs/tasks/tabletop/dual_tasks/_008_pick_apple_banana_baskets.py"]
    sources = {name: (dev / name).read_bytes() for name in relative_sources}
    for name, digest in recording["report"]["source_sha256"].items():
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Input recording's execution source changed: {name}")
    snapshot = output / "source"
    for name, contents in sources.items():
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents)
    input_hashes = {name: hashlib.sha256((args.input_dir / name).read_bytes()).hexdigest()
                    for name in ("trajectory.h5", "trajectory.json", "ticks.jsonl", "report.json")}
    report: dict[str, Any] = {"status": "running", "mode": args.mode, "started_at": datetime.now(timezone.utc).isoformat(),
                              "input_dir": str(args.input_dir.resolve()), "input_sha256": input_hashes,
                              "input_execution_source_sha256": recording["source_sha256"],
                              "task": TASK, "gpu_uuid": gpu, "limits": STATE_LIMITS if args.mode == "state" else ACTION_LIMITS,
                              "original_execution_sources_unchanged": True, "source_snapshot": str(snapshot),
                              "source_sha256": {name: hashlib.sha256(contents).hexdigest() for name, contents in sources.items()},
                              "scope": "recorded-state restoration" if args.mode == "state" else "initial-state-only action replay",
                              "max_episode_steps": 400, "official_default_max_episode_steps": 200,
                              "raw_velocity_differences": "reported; action replay pose/outcome gates do not assert instantaneous dynamic equivalence",
                              "contact_state_restored": False, "continuous_collision_detection": False}
    save_report(output / "report.json", report)
    started = time.monotonic()
    try:
        native_run(args, recording, output, report)
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    try:
        report["input_recording_unchanged"] = all(hashlib.sha256((args.input_dir / name).read_bytes()).hexdigest() == digest
                                                  for name, digest in input_hashes.items())
        if not report["input_recording_unchanged"]:
            report["status"] = "failed"
    except Exception as exc:
        report.update(status="failed", input_verification_error=f"{type(exc).__name__}: {exc}")
    report.update(completed_at=datetime.now(timezone.utc).isoformat(), wall_seconds=time.monotonic() - started)
    save_report(output / "report.json", report)
    print(json.dumps({"status": report["status"], "report": str(output / "report.json")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
