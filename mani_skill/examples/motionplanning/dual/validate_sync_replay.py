"""Replay synchronized dual-object states or actions without changing inputs.

State playback restores each recorded frame and renders with zero physics steps.
Action replay restores frame zero only and submits original paired actions.
Success-only native termination is retained and continued as in the source;
failure/truncation stop after preserving the triggering frame. No replanning,
per-tick correction, controller reset, warmup or collision revalidation is used.
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

from .validate_grasp import TASK, json_value, save_report
from .validate_replay import STATE_LIMITS, ACTION_LIMITS, replay_sequence, within_limits
from .validate_sync import continue_success_only, physical_region
from .replay_metrics import compare_states, pose_errors, summarize_errors

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[6]
DEFAULT_INPUT = ROOT / "infrastructure/validation/dual-arm/sync-positive-20261115T134421956872Z"
ACTORS = ("apple_out", "banana_out")


# Compatibility exports: implementations live in shared runtime modules.
from .replay_support import (unbatch_state, batch_state, replay_signal)







def native_run(args: argparse.Namespace, recording: dict[str, Any], output: Path, report: dict[str, Any]) -> None:
    import gymnasium as gym
    import h5py
    import imageio.v2 as imageio
    import mani_skill.envs
    from PIL import Image, ImageDraw
    from .trajectory_execution import as_numpy
    from .sync_replay_metrics import action_outcome, summarize_objects
    from mani_skill.envs.tasks.tabletop.dual_tasks._008_pick_apple_banana_baskets import world_aabb_from_local_mesh

    env, raw = None, None
    frames, rows, measured_states, applied, errors, signals, restore_events = [], [], [], [], [], [], []
    report["artifact_errors"] = []
    report["state_restore_frames"] = restore_events
    try:
        metadata = recording["metadata"]
        kwargs = dict(metadata["env_info"]["env_kwargs"])
        kwargs["max_episode_steps"] = metadata["env_info"]["max_episode_steps"]
        env = gym.make(metadata["env_info"]["env_id"], **kwargs)
        env.reset(**metadata["episodes"][0]["reset_kwargs"])
        raw = env.unwrapped
        agents, targets, baskets = (raw.left_agent, raw.right_agent), (raw.apple_out, raw.banana_out), (raw.basket_apple, raw.basket_banana)
        if list(raw.agent.agents_dict) != recording["action_keys"] or any(
                raw.agent.agents_dict[key] is not agent for key, agent in zip(recording["action_keys"], agents)):
            raise ValueError("Physical action-key identity differs from source")
        if not np.isclose(float(raw.control_timestep), .05, atol=1e-10, rtol=0):
            raise ValueError("Expected original20Hz control")
        report.update(backend={"physics": raw.backend.sim_backend, "render": raw.backend.render_backend},
                      control_timestep_s=float(raw.control_timestep), reset_calls=1)
        configs = {}
        for key, agent in zip(recording["action_keys"], agents):
            if agent.control_mode != "pd_joint_pos": raise ValueError("Expected absolute joint-position control")
            configs[key] = {}
            for name, controller in agent.controller.controllers.items():
                config = controller.config
                values = {k: bool(getattr(config, k)) for k in ("use_delta", "use_target", "interpolate", "normalize_action")}
                if any(values[k] for k in ("use_delta", "use_target", "interpolate")):
                    raise ValueError("Replay would require unrecorded cumulative controller state")
                configs[key][name] = values
        report["controller_configs"] = configs
        report["fresh_reset_vs_source_initial"] = compare_states(recording["states"][0], unbatch_state(raw.get_state_dict()))
        anchors: list[np.ndarray | None] = [None, None]
        previous: list[np.ndarray | None] = [None, None]

        def restore(index: int, state: dict[str, Any]) -> None:
            if args.mode == "action" and (index != 0 or restore_events):
                raise RuntimeError("Action replay may restore frame0 only once")
            raw.set_state_dict(batch_state(state))
            restore_events.append(index)

        def step(action: dict[str, np.ndarray]) -> None:
            before = int(as_numpy(raw.elapsed_steps)[0])
            _, _, term, trunc, info = env.step(action)
            applied.append(np.concatenate([action[key] for key in recording["action_keys"]]).copy())
            signal = replay_signal(term, trunc, info)
            signals.append(signal)
            index = len(applied)
            if int(as_numpy(raw.elapsed_steps)[0]) != before + 1:
                observe(index)
                raise RuntimeError("Paired action did not advance exactly one control tick")
            if signal["unexpected_end"]:
                observe(index)
                raise RuntimeError("Replay failure or unexpected termination/truncation; triggering frame retained")

        def observe(index: int) -> None:
            actual = unbatch_state(raw.get_state_dict())
            measured_states.append(actual)
            comparison = compare_states(recording["states"][index], actual)
            errors.append(comparison)
            phase = recording["ticks"][index - 1]["phase"] if index else "initial"
            reference = recording["ticks"][index - 1] if index else None
            row: dict[str, Any] = {"frame": index, "phase": phase, "environment_step": int(as_numpy(raw.elapsed_steps)[0]),
                                   "errors": comparison, "objects": []}
            signal = signals[index - 1] if args.mode == "action" and index else None
            row.update(native_success=signal["success"] if signal else None,
                       native_terminated=signal["terminated"] if signal else None,
                       native_truncated=signal["truncated"] if signal else None,
                       native_fail=signal["fail"] if signal else None,
                       success_only_termination_continued=signal["success_only_termination_continued"] if signal else False)
            evaluation = json_value(raw.evaluate()) if args.mode == "action" and index else None
            row["official_evaluation"] = evaluation
            raw_arm_speed = max(float(np.max(np.abs(as_numpy(a.robot.get_qvel())[0, :7]))) for a in agents)
            row["max_raw_arm_speed_rad_s"] = raw_arm_speed
            for i, (name, agent, actor, basket) in enumerate(zip(ACTORS, agents, targets, baskets)):
                pose, tcp = as_numpy(actor.pose.raw_pose)[0].copy(), as_numpy(agent.tcp.pose.raw_pose)[0].copy()
                obj: dict[str, Any] = {"actor": name, "pose": pose.tolist(), "tcp_pose": tcp.tolist(),
                    "pose_error": pose_errors(recording["states"][index]["actors"][name][:7], pose),
                    "tcp_pose_error": pose_errors(reference["tcp_poses"][i], tcp) if index else None,
                    "finger_qpos_m": as_numpy(agent.robot.get_qpos())[0, -2:].tolist(),
                    "instant_linear_velocity_m_s": as_numpy(actor.linear_velocity)[0].tolist(),
                    "instant_angular_velocity_rad_s": as_numpy(actor.angular_velocity)[0].tolist(),
                    "is_grasping": None, "grasp_matches_reference": None, "fingers_open": None,
                    "placement_valid": None, "physical_region": None}
                if args.mode == "action" and index:
                    grasp = bool(as_numpy(agent.is_grasping(actor)).reshape(-1)[0])
                    obj.update(is_grasping=grasp, reference_is_grasping=reference["objects"][i]["is_grasping"],
                               grasp_matches_reference=grasp == reference["objects"][i]["is_grasping"],
                               fingers_open=bool(np.allclose(as_numpy(agent.robot.get_qpos())[0, -2:],
                                   as_numpy(agent.robot.get_qlimits())[0, -2:, 1], atol=.002, rtol=0)))
                    forces = [as_numpy(raw.scene.get_pairwise_contact_forces(link, actor))[0] for link in (agent.finger1_link, agent.finger2_link)]
                    obj["finger_contact_force_norms_N"] = [float(np.linalg.norm(f)) for f in forces]
                    motion = pose_errors(previous[i], pose)
                    obj.update(control_frame_linear_speed_m_s=motion["pose_translation_m"] / float(raw.control_timestep),
                               control_frame_angular_speed_rad_s=motion["pose_rotation_rad"] / float(raw.control_timestep))
                    if phase == "verify_placement":
                        if anchors[i] is None: anchors[i] = pose.copy()
                        drift = pose_errors(anchors[i], pose)
                        region = physical_region(as_numpy(raw._get_actor_geom_center(actor))[0],
                            as_numpy(world_aabb_from_local_mesh(basket, raw.device))[0],
                            as_numpy(world_aabb_from_local_mesh(actor, raw.device))[0])
                        official = bool(np.asarray(evaluation[("apple_in_basket", "banana_in_basket")[i]]).all())
                        valid = bool(region["inside"] and official and obj["fingers_open"] and not grasp and raw_arm_speed <= .05
                            and obj["control_frame_linear_speed_m_s"] <= .03 and obj["control_frame_angular_speed_rad_s"] <= .2
                            and drift["pose_translation_m"] <= .003 and drift["pose_rotation_rad"] <= .03)
                        obj.update(placement_valid=valid, physical_region=region, official_region=official,
                                   window_position_drift_m=drift["pose_translation_m"], window_rotation_drift_rad=drift["pose_rotation_rad"])
                previous[i] = pose.copy()
                row["objects"].append(obj)
            rows.append(row)
            frame = as_numpy(env.render())
            if frame.ndim == 4 and frame.shape[0] == 1: frame = frame[0]
            if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
                raise ValueError("Expected HWC uint8 replay rendering")
            frames.append(frame.copy())

        report["sequence"] = replay_sequence(recording, args.mode, restore=restore, step=step, observe=observe)
        summary = summarize_errors(errors)
        report["trajectory_errors"] = summary
        report["object_errors"] = summarize_objects(rows)
        limits = STATE_LIMITS if args.mode == "state" else ACTION_LIMITS
        checks = {"all7_state_metrics_within_limits": within_limits(summary["global"], limits),
                  "all_frames_observed": len(rows) == len(recording["states"]),
                  "correct_restore_count": len(restore_events) == (len(recording["states"]) if args.mode == "state" else 1),
                  "correct_physical_control_steps": int(as_numpy(raw.elapsed_steps)[0]) == (0 if args.mode == "state" else len(recording["actions"]))}
        for i, obj in enumerate(report["object_errors"]["objects"]):
            checks[f"object{i}_tcp_within_limits"] = within_limits(obj["tcp_pose"], {k: limits[k] for k in obj["tcp_pose"]})
            checks[f"object{i}_pose_within_limits"] = within_limits(obj["object_pose"], {k: limits[k] for k in obj["object_pose"]})
        if args.mode == "action":
            checks["actions_applied_exactly"] = np.array_equal(np.asarray(applied), recording["actions"])
            checks["native_flags_match_source"] = all(np.array_equal(np.asarray([s[k] for s in signals]), recording["native_flags"][k])
                                                       for k in ("success", "terminated", "truncated"))
            report["outcome"] = action_outcome(rows)
            checks["both_physical_outcomes_reproduced"] = report["outcome"]["status"] == "passed"
        report["checks"] = checks
        report["status"] = "passed" if all(checks.values()) else "failed"
    finally:
        artifact_errors = report["artifact_errors"]
        if raw is not None:
            for key, read in (("physical_environment_steps", lambda: int(as_numpy(raw.elapsed_steps)[0])),
                              ("final_official_evaluation", lambda: json_value(raw.evaluate()) if args.mode == "action" else None)):
                try: report[key] = read()
                except Exception as exc: artifact_errors.append(f"final {key}: {type(exc).__name__}: {exc}")
        report["observed_state_frames"] = len(measured_states)
        report["applied_action_steps"] = len(applied)
        report["success_only_termination_continuation_ticks"] = sum(s["success_only_termination_continued"] for s in signals)
        if env is not None:
            try: env.close()
            except Exception as exc: artifact_errors.append(f"environment close: {type(exc).__name__}: {exc}")
        try:
            with (output / "frames.jsonl").open("w", encoding="utf-8") as f:
                for row in rows: f.write(json.dumps(json_value(row), allow_nan=False) + "\n")
            if measured_states:
                with h5py.File(output / "observed-states.h5", "w") as h:
                    h.attrs["validation_only"] = True; h.attrs["mode"] = args.mode
                    h.create_dataset("applied_actions", data=np.asarray(applied, dtype=np.float32).reshape(-1, 16))
                    for key in ("success", "terminated", "truncated", "fail"):
                        h.create_dataset(f"native_flags/{key}", data=np.asarray([s[key] for s in signals], dtype=bool))
                    for category in measured_states[0]:
                        for name in measured_states[0][category]:
                            h.create_dataset(f"env_states/{category}/{name}", data=np.stack([s[category][name] for s in measured_states]))
            report["recording"] = {"path": str(output / "observed-states.h5"), "observed_state_frames": len(measured_states), "applied_action_steps": len(applied)}
        except Exception as exc: artifact_errors.append(f"measurement export: {type(exc).__name__}: {exc}")
        if frames:
            try:
                imageio.mimsave(output / "replay.mp4", [frames[0]] * 10 + frames + [frames[-1]] * 20, fps=20, macro_block_size=1)
                imageio.imwrite(output / "preview.png", frames[-1])
                source = imageio.get_reader(Path(args.input_dir) / "execution.mp4")
                try:
                    with imageio.get_writer(output / "comparison.mp4", fps=20, macro_block_size=1) as writer:
                        for index in [0] * 10 + list(range(len(frames))) + [len(frames) - 1] * 20:
                            original, current = source.get_data(index + 10), frames[index]
                            if original.shape != current.shape: raise ValueError("Source/replay video resolutions differ")
                            pair = Image.fromarray(np.concatenate([original, current], axis=1)); draw = ImageDraw.Draw(pair)
                            draw.rectangle((0, 0, pair.width, 27), fill="black")
                            draw.text((8, 7), f"Original synced | frame {index}", fill="white")
                            draw.text((current.shape[1] + 8, 7), f"{args.mode.title()} replay", fill="white")
                            writer.append_data(np.asarray(pair))
                finally: source.close()
                report["video"] = {"replay": str(output / "replay.mp4"), "comparison": str(output / "comparison.mp4"), "fps": 20,
                                   "padding_frames_before": 10, "padding_frames_after": 20, "source_video_padding_frames_before": 10}
            except Exception as exc: artifact_errors.append(f"video export: {type(exc).__name__}: {exc}")
        if artifact_errors: report["status"] = "failed"


def self_test() -> dict[str, Any]:
    value = {"actors": {"a": np.arange(13, dtype=float)[None]}, "articulations": {"r": np.arange(31, dtype=float)[None]}}
    unpacked = unbatch_state(value); restored = batch_state(unpacked)
    assert all(np.array_equal(value[c][n], restored[c][n]) for c in value for n in value[c])
    unpacked["actors"]["a"][0] = -2; assert value["actors"]["a"][0, 0] == 0
    assert replay_signal(True, False, {"success": True})["success_only_termination_continued"]
    assert not replay_signal(True, False, {"success": True})["unexpected_end"]
    assert replay_signal(True, False, {"success": True, "fail": True})["unexpected_end"]
    assert replay_signal(True, True, {"success": True})["unexpected_end"]
    assert replay_signal(True, False, {})["unexpected_end"]
    assert (ROOT / "infrastructure").is_dir()
    return {"status": "passed", "checks": ["finite batched states and alias isolation", "success-only continuation with native flags",
        "failure, truncation and unknown termination reject", "established project root"]}


def main() -> int:
    faulthandler.enable(); resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--mode", choices=("state", "action"), default="action")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2)); return 0
    from .sync_replay_metrics import load_sync_recording
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not re.fullmatch(r"GPU-[0-9a-fA-F-]+", gpu): parser.error("Select one available GPU UUID for rendering")
    recording = load_sync_recording(args.input_dir)
    output = args.output_dir or ROOT / "infrastructure/validation/dual-arm" / datetime.now(timezone.utc).strftime(f"sync-replay-{args.mode}-%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    dev = HERE.parents[3]
    relative = [f"mani_skill/examples/motionplanning/dual/{n}" for n in set(recording["source_sha256"]) |
        {"validate_sync_replay.py", "sync_replay_metrics.py", "replay_metrics.py", "validate_replay.py"}]
    relative += ["mani_skill/envs/sapien_env.py", "mani_skill/envs/scene.py", "mani_skill/utils/structs/actor.py",
                 "mani_skill/utils/structs/articulation.py", "mani_skill/agents/controllers/pd_joint_pos.py",
                 "mani_skill/envs/tasks/tabletop/dual_tasks/_008_pick_apple_banana_baskets.py"]
    sources = {name: (dev / name).read_bytes() for name in sorted(relative)}
    for name, digest in recording["source_sha256"].items():
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Input execution source changed: {name}")
    snapshot = output / "source"
    for name, data in sources.items():
        target = snapshot / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
    input_names = ("trajectory.h5", "trajectory.json", "ticks.jsonl", "report.json", "physics-substeps.jsonl", "execution.mp4")
    input_hashes = {name: hashlib.sha256((args.input_dir / name).read_bytes()).hexdigest() for name in input_names}
    report: dict[str, Any] = {"status": "running", "mode": args.mode, "started_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(args.input_dir.resolve()), "input_sha256": input_hashes, "input_execution_source_sha256": recording["source_sha256"],
        "task": TASK, "gpu_uuid": gpu, "limits": STATE_LIMITS if args.mode == "state" else ACTION_LIMITS,
        "scope": "recorded7entity states restored and rendered without physics" if args.mode == "state" else "initial-state-only fullpairedaction replay andboth physicaloutcomes",
        "source_snapshot": str(snapshot), "source_sha256": {n: hashlib.sha256(b).hexdigest() for n, b in sources.items()},
        "original_execution_sources_unchanged": True, "max_episode_steps": 400, "official_default_max_episode_steps": 200,
        "contact_state_restored": False, "static_scene_reconstructed_from_seed": True, "collision_revalidated": False,
        "continuous_collision_detection": False, "replanning_performed": False, "controller_reset_after_state_restore": False,
        "input_receipt": recording["sync_receipt"], "raw_velocity_differences": "reported separately; action gates compare pose/joints/physicaloutcomes, not instantaneousdynamic equivalence"}
    save_report(output / "report.json", report); started = time.monotonic()
    try: native_run(args, recording, output, report)
    except Exception as exc: report.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    try:
        report["input_recording_unchanged"] = all(hashlib.sha256((args.input_dir / n).read_bytes()).hexdigest() == h for n, h in input_hashes.items())
        if not report["input_recording_unchanged"]: report["status"] = "failed"
    except Exception as exc: report.update(status="failed", input_verification_error=f"{type(exc).__name__}: {exc}")
    report.update(completed_at=datetime.now(timezone.utc).isoformat(), wall_seconds=time.monotonic() - started)
    save_report(output / "report.json", report)
    print(json.dumps({"status": report["status"], "report": str(output / "report.json")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
