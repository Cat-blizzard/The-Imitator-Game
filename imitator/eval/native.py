"""Single-environment native embodiment support for the public policy evaluator.

Public RGB and physical qpos are normalized using the selected training rows.
Policies return (1, Q, A) normalized chunks. Controller targets are split in the
recorded named order. Physical diagnostics are sampled after env.step and never
change actions, rewards, success or termination. No expert, state restoration,
contract settle or retreat is executed in this evaluation route.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .native_io import live_observation, split_action


def record_identity(record: Mapping[str, Any]) -> str:
    """Keep legitimate same-task datasets distinct in rollout paths/results."""
    identity = {name: record.get(name) for name in ("dataset_idx", "repo_id", "env_id", "native")}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return f"dataset-{record['dataset_idx']}-{digest[:16]}"


def evaluation_seeds(args: Any) -> list[int]:
    if args.seeds is not None and args.seeds_file is not None:
        raise ValueError("Choose --seeds or --seeds-file")
    if args.seeds_file:
        payload = json.loads(Path(args.seeds_file).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("seed_splits", payload.get("splits", payload))
            payload = payload[args.seed_split]
        seeds = payload
    else:
        seeds = args.seeds if args.seeds is not None else list(range(args.num_episodes))
    if (not isinstance(seeds, list) or not seeds or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("Evaluation seeds must be unique nonnegative integers")
    return seeds


def validate_native_contract(recorded: Mapping[str, Any], live: Mapping[str, Any]) -> None:
    """Reject physical-model drift instead of relabelling an old policy score."""
    if not recorded.get("model_fingerprints"):
        raise ValueError("Native evaluation needs recorded 2.0 model_fingerprints; "
                         "legacy weights can load but do not certify physical compatibility")
    for name in ("env_id", "robot_uids", "control_mode", "control_freq", "sim_freq",
                 "model_fingerprints", "mounts"):
        if recorded.get(name) != live.get(name):
            raise ValueError(f"Live native {name} differs from the training recording")
    source_order = recorded.get("action_order", recorded.get("action_layout"))
    target_order = live.get("action_order", live.get("action_layout"))
    if source_order != target_order:
        raise ValueError("Live native controller/state ordering differs from the training recording")


def task_events(raw: Any, info: Mapping[str, Any]) -> dict[str, Any]:
    """Read task facts after stepping, without introducing acceptance gates."""
    def scalar(value: Any) -> bool:
        return bool(torch.as_tensor(value).reshape(-1)[0].item())

    aliases = {"pill_a_in_region": "left_in_region", "pill_b_in_region": "right_in_region",
               "is_pill_a_grasped": "left_grasped", "is_pill_b_grasped": "right_grasped",
               "apple_in_basket": "left_in_region", "banana_in_basket": "right_in_region",
               "is_apple_grasped": "left_grasped", "is_banana_grasped": "right_grasped",
               "is_robot_static": "is_robot_static"}
    events: dict[str, Any] = {alias: scalar(info[name]) for name, alias in aliases.items() if name in info}
    events["objects"] = {}
    for name in ("pill_a", "pill_b", "apple_out", "banana_out"):
        if hasattr(raw, name):
            actor = getattr(raw, name)
            events["objects"][name] = {
                "position_m": torch.as_tensor(actor.pose.p).detach().cpu().reshape(-1).tolist(),
                "linear_speed_m_s": float(torch.linalg.vector_norm(torch.as_tensor(actor.get_linear_velocity()))),
                "angular_speed_rad_s": float(torch.linalg.vector_norm(torch.as_tensor(actor.get_angular_velocity())))}
    return events


def run_episode(policy: Any, env: Any, normalizer: Any, *, dataset_idx: int,
                config: Mapping[str, Any], seed: int, execute_steps: int,
                max_steps: int, demonstration: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """Run one budgeted episode. Diagnostics cannot intervene in execution."""
    if not 1 <= execute_steps <= config["pred_horizon"] or config["obs_horizon"] != 1:
        raise ValueError("Native evaluation requires obs_horizon=1 and 1 <= execute_steps <= pred_horizon")
    observation, _ = env.reset(seed=seed)
    raw = env.unwrapped
    camera = config["cameras"][0]
    order = config["native_contract"].get("action_order", config["native_contract"].get("action_layout"))
    keys, sizes = [item["key"] for item in order], [item["size"] for item in order]
    if sum(sizes) != config["action_dim"]:
        raise ValueError("Live action widths differ from the policy")

    def public_obs(value: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        sample = live_observation(raw, value, camera, normalizer, dataset_idx,
                                  config["image_size"], order)
        if sample["state"].shape != (config["state_dim"],):
            raise ValueError("Live public qpos width differs from the policy")
        return {"view_1": sample["rgb"].unsqueeze(0).unsqueeze(0).to(device),
                "states": sample["state"].unsqueeze(0).unsqueeze(0).to(device)}

    policy.eval()
    first = public_obs(observation)
    video = demonstration["human_video"].unsqueeze(0).to(device)
    policy.prepare_for_eval(video, first, None)
    events, actions, clipped_steps = [], None, 0
    first_success = None
    for step in range(max_steps):
        if step % execute_steps == 0:
            with torch.no_grad():
                actions = policy.get_action(first if step == 0 else public_obs(observation))
            if (actions.shape != (1, config["pred_horizon"], config["action_dim"])
                    or not torch.isfinite(actions).all()):
                raise ValueError("Policy must return finite normalized chunks (1,Q,A)")
        requested = normalizer.denormalize_action(actions[0, step % execute_steps], dataset_idx)
        action = split_action(requested, keys, sizes)
        clipped = False
        for key in keys:
            bounded = np.clip(action[key], raw.action_space[key].low, raw.action_space[key].high)
            clipped |= not np.array_equal(action[key], bounded)
            action[key] = bounded.astype(np.float32)
        clipped_steps += int(clipped)
        observation, _, terminated, truncated, info = env.step(action)
        success = bool(torch.as_tensor(info.get("success", False)).reshape(-1)[0])
        if success and first_success is None:
            first_success = step + 1
        # These measurements are reporting only; never branch on physical facts.
        events.append({"step": step + 1, "official_success": success, **task_events(raw, info)})
        if bool(torch.as_tensor(truncated).reshape(-1)[0]):
            break
    result = {"seed": seed, "control_steps": len(events), "success_once": first_success is not None,
              "success_at_end": bool(events and events[-1]["official_success"]),
              "first_success_step": first_success, "action_clip_steps": clipped_steps,
              "terminated": bool(torch.as_tensor(terminated).reshape(-1)[0]),
              "truncated": bool(torch.as_tensor(truncated).reshape(-1)[0]), "events": events,
              "human_demo_id": demonstration["human_demo_id"],
              "human_demonstration": demonstration["metadata"],
              "conditioning": getattr(policy, "conditioning", None)}
    policy.clear_cache()
    return result


def evaluate_native(args: Any, policy: Any, model_config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Dispatch saved native records through the public policy interface."""
    from imitator.dataset.native import NativeSingleTaskNormalizer
    from imitator.embodiments.native import make_native_env, native_contract, resolve_robot
    from mani_skill.utils.wrappers import RecordEpisode

    if args.num_envs != 1 or model_config["include_depth"] or len(model_config["cameras"]) != 1:
        raise ValueError("Native public evaluation currently supports one environment and one RGB camera")
    if args.sim_backend != "physx_cpu" or args.control_mode != "pd_joint_pos" or args.input_mode != "video_only":
        raise ValueError("Native public evaluation requires PhysX CPU, pd_joint_pos and video_only")
    if args.temporal_agg or args.light_temporal_agg:
        raise ValueError("Native public evaluation executes explicit chunks without temporal aggregation")
    if args.compute_dtw:
        raise ValueError("Native interface evaluation does not yet implement cross-embodiment TSS/nDTW")
    if model_config["state_type"] != "qpos" or model_config["single_arm"]:
        raise ValueError("Native public evaluation requires dual-arm physical qpos")
    saved = model_config.get("native_evaluation")
    if not saved:
        raise ValueError("Policy factory did not provide a saved native evaluation contract")
    if args.normalization_method not in (None, saved["normalization_method"]):
        raise ValueError("Native normalization must match the checkpoint's saved training method")
    if args.max_episode_steps not in (None, 200):
        raise ValueError("The adapted original tasks retain their 200-step budget")
    records = [item for item in saved["records"] if args.env_id is None or item["env_id"] == args.env_id]
    if not records:
        raise ValueError("No checkpoint dataset matches --env-id")
    normalizer = NativeSingleTaskNormalizer(saved["dataset_stats"], saved["normalization_method"],
                                           saved["normalization_min_range"])
    seeds = evaluation_seeds(args)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    results: list[dict[str, Any]] = []
    for record in records:
        native = record["native"]
        record_id = record_identity(record)
        robot = args.robot or native.get("robot")
        if robot is None:
            robot = resolve_robot(native["robot_uids"][0])
        env_id = record["env_id"]
        raw = make_native_env(env_id, robot=robot, obs_mode="rgb", render_mode="rgb_array",
                              max_episode_steps=200, control_mode="pd_joint_pos",
                              sensor_configs={"shader_pack": args.shader},
                              human_render_camera_configs={"shader_pack": args.shader})
        env = None
        try:
            live = native_contract(raw, cameras=tuple(model_config["cameras"]))
            validate_native_contract(native, live)
            directory = output / "rollouts" / env_id / record_id / stamp
            env = RecordEpisode(raw, output_dir=str(directory), save_trajectory=True,
                                save_video=args.save_video, save_on_reset=False,
                                source_type="policy", source_desc=model_config.get("name", "native_policy"))
            # Keep the actual native model/controller contract with the H5 pair.
            # These are recording facts, never policy inputs or success gates.
            env._json_data["native_data_contract"] = live
            env._json_data["evaluation_record"] = {"record_id": record_id,
                "dataset_idx": record["dataset_idx"], "repo_id": record["repo_id"]}
            config = dict(model_config, native_contract=native,
                          state_dim=saved["state_dim"], action_dim=saved["action_dim"])
            for seed in seeds:
                try:
                    row = run_episode(policy, env, normalizer, dataset_idx=record["dataset_idx"],
                                      config=config, seed=seed, execute_steps=args.execute_steps, max_steps=200,
                                      demonstration=saved["demonstrations"][env_id], device=torch.device(args.device))
                except Exception as exc:
                    row = {"seed": seed, "interface_error": f"{type(exc).__name__}: {exc}",
                           "success_at_end": False, "success_once": False}
                    policy.clear_cache()
                finally:
                    env.flush_trajectory(save=True)
                    if args.save_video:
                        env.flush_video(save=True)
                row.update(env_id=env_id, robot=robot, record_id=record_id,
                           dataset_idx=record["dataset_idx"], repo_id=record["repo_id"],
                           checkpoint=saved["checkpoint"],
                           native_contract=live, execute_steps=args.execute_steps,
                           trajectory_path=str(env._h5_file.filename))
                results.append(row)
                (output / f"native_{stamp}.json").write_text(
                    json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        finally:
            (env if env is not None else raw).close()
    print(json.dumps({"episodes": len(results),
                      "success_at_end": sum(bool(r["success_at_end"]) for r in results),
                      "interface_errors": sum("interface_error" in r for r in results)}))
    return results


if __name__ == "__main__":
    from types import SimpleNamespace

    assert evaluation_seeds(SimpleNamespace(seeds=[7, 8], seeds_file=None, num_episodes=1)) == [7, 8]
    c = {"env_id": "pill", "robot_uids": ["left", "right"], "model_fingerprints": ["m1", "m1"],
         "control_mode": "pd_joint_pos", "control_freq": 20, "sim_freq": 100,
         "mounts": [[0, 0, 0]], "action_order": [{"key": "left", "size": 7}]}
    validate_native_contract(c, dict(c))
    try:
        validate_native_contract(c, dict(c, model_fingerprints=["different"]))
    except ValueError:
        pass
    else:
        raise AssertionError("Changed physical model accepted")
    print({"seed_contract": True, "physical_model_mismatch_rejected": True})
