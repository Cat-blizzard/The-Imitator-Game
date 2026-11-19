"""Evaluate single-task ACT checkpoints on fixed, explicitly recorded seeds.

The policy receives the same selected RGB camera and physical qpos as training.
Only the policy's absolute joint/gripper targets control the environment.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import MISSING, asdict, fields
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from examples.baselines.act.act.single_task_eval_utils import (
    clear_l_level, extract_base_env_name, extract_level, set_l_level,
)
from examples.baselines.act.single_task_io import live_observation, split_action
from examples.baselines.act.train_act_single import ACTSingleTaskAgent, TrainingArgs
from examples.baselines.lerobot_dataset.normalizer import ActionNormalizer

RELEASE_OBSERVED_SEMANTICS = "legacy_ever_grasped_and_finally_ungrasped_includes_grasp_loss"


class SingleTaskNormalizer:
    """Use the exact training-row statistics embedded in the checkpoint."""

    def __init__(self, dataset_stats: list[dict[str, Any]], method: str = "bounds_q99",
                 min_range: float = 0.0):
        if not dataset_stats:
            raise ValueError("Checkpoint has no saved training statistics; retrain with train_act_single.py")
        self.normalizer = ActionNormalizer(min_range=min_range)
        self.method = method
        self.repo_to_idx = {}
        for record in dataset_stats:
            idx = int(record["dataset_idx"])
            self.normalizer.add_dataset_stats(
                idx, record["repo_id"], record["stats"], record["state_key"],
                record["action_key"], record.get("single_arm", False),
            )
            self.repo_to_idx[record["repo_id"]] = idx

    def resolve_idx(self, env_id: str) -> int:
        if env_id in self.repo_to_idx:
            return self.repo_to_idx[env_id]
        matching = [idx for repo, idx in self.repo_to_idx.items()
                    if extract_base_env_name(repo) == extract_base_env_name(env_id)]
        if len(matching) == 1:
            return matching[0]
        raise ValueError(f"No unambiguous checkpoint normalizer for {env_id}")

    def normalize_state(self, state: torch.Tensor, dataset_idx: int = 0) -> torch.Tensor:
        return self.normalizer.normalize_state(state, dataset_idx, self.method)

    def denormalize_action(self, action: torch.Tensor, dataset_idx: int = 0) -> torch.Tensor:
        return self.normalizer.denormalize_action(action, dataset_idx, self.method)


def load_checkpoint(ckpt_path: str, args_override: Any, device: torch.device):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_args = TrainingArgs()
    for key, value in checkpoint.get("args", {}).items():
        if hasattr(train_args, key):
            setattr(train_args, key, value)
    for key in ("state_type", "pred_horizon", "obs_horizon"):
        value = getattr(args_override, key, None)
        if value is not None and value != getattr(train_args, key):
            raise ValueError(f"Evaluation {key} must match the checkpoint")
    agent = ACTSingleTaskAgent(train_args, device)
    agent.load_state_dict(checkpoint["agent_state_dict"], strict=True)
    agent.eval()
    return agent, train_args


def action_coordinate_metadata(raw: Any, keys: list[str], sizes: list[int],
                               action_order: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Describe native action coordinates without changing controller settings."""
    coordinates = []
    for side, (key, size) in enumerate(zip(keys, sizes)):
        spec = action_order[side] if action_order else {}
        joints = spec.get("arm_joint_names", [])
        agent = raw.agent.agents[side]
        controllers = getattr(getattr(agent, "controller", None), "controllers", {})
        gripper = getattr(controllers.get("gripper"), "config", None)
        normalized = getattr(gripper, "normalize_action", None)
        for local_index in range(size):
            arm_joint = local_index < len(joints)
            known_gripper = bool(joints) and local_index == len(joints)
            unit = (spec.get("arm_coordinate_unit", "controller_action_unit") if arm_joint
                    else "normalized_controller_command" if known_gripper and normalized is True
                    else spec.get("gripper_coordinate_unit", "controller_action_unit")
                    if known_gripper and normalized is False else "controller_action_unit")
            coordinates.append({"robot_key": key, "side": "left" if side == 0 else "right",
                "name": joints[local_index] if arm_joint else "gripper" if known_gripper else f"coordinate_{local_index}",
                "kind": "arm_joint" if arm_joint else "gripper" if known_gripper else "controller_coordinate",
                "unit": unit})
    return coordinates


def evaluation_seeds(args: Any) -> list[int]:
    if args.seeds is not None and args.seeds_file is not None:
        raise ValueError("Choose --seeds or --seeds-file")
    if args.seeds_file:
        payload = json.loads(Path(args.seeds_file).read_text())
        if isinstance(payload, dict):
            payload = payload.get("seed_splits", payload.get("splits", payload))
            payload = payload[args.seed_split]
        seeds = payload
    elif args.seeds is not None:
        seeds = args.seeds
    else:
        seeds = list(range(args.seed, args.seed + args.num_episodes))
    if not isinstance(seeds, list) or not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Evaluation seeds must be a nonempty list of nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation seeds must be unique")
    return seeds


def evaluation_robot_uids(env_id: str, requested: str | None,
                          recorded: list[str] | None) -> tuple[str, ...] | None:
    """Resolve the original entry's Panda alias against the recorded robot pair."""
    if requested is None:
        return tuple(recorded) if recorded else None
    uid = "panda_wristcam" if requested == "panda" and env_id != "DualArmPickPlace-v1" else requested
    selected = (uid, uid)
    if recorded is not None and list(selected) != list(recorded):
        raise ValueError("Evaluation robot must match the training recording")
    return selected


def validate_recorded_profiles(contract: dict[str, Any], env_id: str,
                               robot_uids: tuple[str, ...] | None) -> dict[str, list[str]]:
    """Check saved physical configuration when present; keep legacy data usable."""
    if not any(name in contract for name in ("task_profile", "robot_profile")):
        return {}
    if robot_uids is None or len(robot_uids) != 2 or robot_uids[0] != robot_uids[1]:
        raise ValueError("Recorded embodiment profiles require their homogeneous robot pair")
    from mani_skill.agents.robots.embodiment_profiles import get_robot_profile
    from mani_skill.envs.tasks.tabletop.dual_arm_profiles import get_task_profile
    task_id = extract_base_env_name(env_id)
    profiles = {
        "task_profile": get_task_profile(task_id, robot_uids[0]),
        "robot_profile": get_robot_profile(robot_uids[0]),
    }

    def effective(value: Any) -> Any:
        # New optional None fields preserve behavior. Keep None list entries
        # (for example an inherited grasp quaternion) and all numeric settings.
        if isinstance(value, dict):
            return {key: effective(item) for key, item in value.items() if item is not None}
        if isinstance(value, list):
            return [effective(item) for item in value]
        return value

    task_environment_fields = {
        "TwoRobotPickPillToRegions-v1": {"mount_poses", "gripper_drive_force_limit"},
        "TwoRobotPickAppleBananaToBaskets-v1": {"mount_poses"},
        "DualArmPickPlace-v1": {"base_x", "arm_separation", "source_x", "destination_x",
                                 "base_height", "source_jitter_m"},
    }
    differences = {}
    for name, profile in profiles.items():
        if name not in contract:
            continue
        saved = dict(contract[name])
        # An omitted field predates its introduction. Its declared default is
        # the old behavior; using the selected profile's value could hide drift.
        for field in fields(profile):
            if field.name not in saved:
                if field.default is not MISSING:
                    saved[field.name] = field.default
                elif field.default_factory is not MISSING:
                    saved[field.name] = field.default_factory()
        saved = effective(json.loads(json.dumps(saved)))
        current = effective(json.loads(json.dumps(asdict(profile))))
        changed = {key for key in saved.keys() | current.keys() if saved.get(key) != current.get(key)}
        environment_fields = (task_environment_fields.get(task_id, saved.keys() | current.keys())
                              if name == "task_profile" else
                              (saved.keys() | current.keys()) - {"prefer_nearby_revolute_goals"})
        if changed & environment_fields:
            raise ValueError(f"Evaluation {name} differs from the recorded physical configuration")
        if changed:
            differences[name] = sorted(changed)
    return differences


def _bool(value: Any) -> bool:
    return bool(torch.as_tensor(value).reshape(-1)[0].item())


def _task_events(raw: Any, info: dict[str, Any]) -> dict[str, Any]:
    """Task facts for reporting only; these never alter policy actions or termination."""
    result: dict[str, Any] = {}
    if hasattr(raw, "targets"):
        objects = list(raw.targets)
        agents = raw.agent.agents
        region_keys = ("left_in_region", "right_in_region")
        for index, (agent, target) in enumerate(zip(agents, objects)):
            result[("left_grasped", "right_grasped")[index]] = _bool(agent.is_grasping(target))
            if region_keys[index] in info:
                result[region_keys[index]] = _bool(info[region_keys[index]])
        result["is_robot_static"] = all(_bool(agent.is_static(.05)) for agent in agents)
        named = list(zip(("left_target", "right_target"), objects))
    else:
        aliases = {"pill_a_in_region": "left_in_region", "pill_b_in_region": "right_in_region",
                   "is_pill_a_grasped": "left_grasped", "is_pill_b_grasped": "right_grasped",
                   "is_robot_static": "is_robot_static"}
        for name, alias in aliases.items():
            if name in info:
                result[alias] = _bool(info[name])
        named = [(name, getattr(raw, name)) for name in ("pill_a", "pill_b") if hasattr(raw, name)]
    result["objects"] = {
        name: {"position_m": torch.as_tensor(actor.pose.p).detach().cpu().reshape(-1).tolist(),
               "linear_speed_m_s": float(torch.linalg.vector_norm(torch.as_tensor(actor.get_linear_velocity()))),
               "angular_speed_rad_s": float(torch.linalg.vector_norm(torch.as_tensor(actor.get_angular_velocity())))}
        for name, actor in named
    }
    return result


def settling_measurements(events: list[dict[str, Any]], window_steps: int = 20) -> dict[str, Any]:
    """Report the trailing motion window without adding a policy acceptance gate."""
    window = events[-window_steps:]
    objects = {}
    names = set.intersection(*(set(row.get("objects", {})) for row in window)) if window else set()
    for name in sorted(names):
        positions = np.asarray([row["objects"][name]["position_m"] for row in window])
        objects[name] = {
            "max_displacement_from_window_start_m": float(np.linalg.norm(positions - positions[0], axis=-1).max()),
            "max_linear_speed_m_s": max(row["objects"][name]["linear_speed_m_s"] for row in window),
            "max_angular_speed_rad_s": max(row["objects"][name]["angular_speed_rad_s"] for row in window),
        }
    keys = ("left_in_region", "right_in_region", "left_grasped", "right_grasped")
    placement_observed = bool(window) and all(all(key in row for key in keys) for row in window)
    return {
        "window_steps": len(window), "objects": objects,
        "all_in_regions_and_not_grasped": (all(
            row["left_in_region"] and row["right_in_region"]
            and not row["left_grasped"] and not row["right_grasped"] for row in window
        ) if placement_observed else None),
        "robot_static_all": (all(row["is_robot_static"] for row in window)
                             if window and all("is_robot_static" in row for row in window) else None),
    }


def grasp_placement_measurements(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Report sampled predicates, without inferring intentional release or cause.

    A grasp-to-ungrasped transition can be normal release or loss of contact.
    A placement is the task's region predicate and no grasp in the same sample;
    it does not prove that a robot transported or deliberately released it.
    Missing predicates remain unknown, and gaps do not create transitions.
    """
    arms = {}
    for side in ("left", "right"):
        grasp_key, region_key = f"{side}_grasped", f"{side}_in_region"
        grasps = [row[grasp_key] for row in events if grasp_key in row]
        placed = [row.get(region_key) is True and row.get(grasp_key) is False
                  for row in events]
        transitions = [row["step"] for previous, row in zip(events, events[1:])
                       if previous.get(grasp_key) is True and row.get(grasp_key) is False]
        outside = [row["step"] for previous, row in zip(events, events[1:])
                   if previous.get(grasp_key) is True and row.get(grasp_key) is False
                   and row.get(region_key) is False]
        final = events[-1] if events else {}
        arms[side] = {
            "grasp_measurements_complete": bool(events) and len(grasps) == len(events),
            "region_measurements_complete": bool(events) and all(region_key in row for row in events),
            "ever_grasped": any(grasps) if grasps else None,
            "first_grasp_step": next((row["step"] for row in events
                                      if row.get(grasp_key) is True), None),
            "grasp_to_ungrasped_steps": transitions,
            "grasp_to_ungrasped_outside_region_steps": outside,
            "first_placed_and_ungrasped_step": next((row["step"] for row, valid in zip(events, placed)
                                                     if valid), None),
            "final_placed_and_ungrasped": (bool(final[region_key] and not final[grasp_key])
                if region_key in final and grasp_key in final else None),
        }
    final_values = [arm["final_placed_and_ungrasped"] for arm in arms.values()]
    return {
        "schema": "sampled-grasp-placement-v1",
        "semantics": "observed_predicates_only_not_controlled_release_or_loss_cause",
        "arms": arms,
        "both_final_placed_and_ungrasped": (all(final_values)
            if all(value is not None for value in final_values) else None),
    }


@torch.no_grad()
def evaluate_episode(agent: ACTSingleTaskAgent, env: Any, normalizer: SingleTaskNormalizer,
                     args: Any, train_args: TrainingArgs, seed: int, env_id: str,
                     device: torch.device) -> dict[str, Any]:
    """Run a bounded, seeded rollout without expert actions or state forcing."""
    observation, _ = env.reset(seed=seed)
    raw = env.unwrapped
    if raw.control_freq != args.fps:
        raise ValueError(f"Environment is {raw.control_freq} Hz, checkpoint/evaluation require {args.fps} Hz")
    if len(train_args.cameras) != 1 or train_args.state_type != "qpos" or train_args.single_arm:
        raise ValueError("This evaluation path expects one RGB camera and dual-arm native qpos")
    keys = list(raw.action_space.spaces)
    action_order = getattr(args, "native_contract", {}).get("action_order")
    sizes = [int(raw.action_space[key].shape[0]) for key in keys]
    if sum(sizes) != train_args.action_dim:
        raise ValueError("Live action dimensions differ from the checkpoint")
    if action_order is not None and [item["size"] for item in action_order] != sizes:
        raise ValueError("Live per-arm action widths differ from the checkpoint")
    action_diagnostics = None
    if getattr(args, "action_diagnostics", False):
        from examples.baselines.act.action_limit_diagnostics import ActionLimitDiagnostics
        action_diagnostics = ActionLimitDiagnostics(
            action_coordinate_metadata(raw, keys, sizes, action_order),
            np.concatenate([raw.action_space[key].low for key in keys]),
            np.concatenate([raw.action_space[key].high for key in keys]))
    dataset_idx = normalizer.resolve_idx(env_id)
    rgb_history: deque[torch.Tensor] = deque(maxlen=train_args.obs_horizon)
    state_history: deque[torch.Tensor] = deque(maxlen=train_args.obs_horizon)
    actions = None
    success_once = False
    first_success_step = None
    clip_steps = 0
    ever_grasped = [False, False]
    events = []
    terminated = truncated = False
    final_info: dict[str, Any] = {}
    for step in range(args.max_episode_steps):
        sample = live_observation(raw, observation, train_args.cameras[0], normalizer,
                                  dataset_idx, train_args.image_size, action_order)
        rgb_history.append(sample["rgb"])
        state_history.append(sample["state"])
        while len(rgb_history) < train_args.obs_horizon:
            rgb_history.appendleft(rgb_history[0])
            state_history.appendleft(state_history[0])
        if step % args.execute_steps == 0:
            rgb = torch.stack(list(rgb_history)).unsqueeze(0).to(device)
            state = torch.stack(list(state_history)).unsqueeze(0).to(device)
            actions = agent.get_action(rgb, state)
        prediction = normalizer.denormalize_action(actions[0, step % args.execute_steps], dataset_idx)
        action = split_action(prediction, keys, sizes)
        requested = (np.concatenate([action[key] for key in keys])
                     if action_diagnostics is not None else None)
        clipped = False
        for key in keys:
            bounded = np.clip(action[key], raw.action_space[key].low, raw.action_space[key].high)
            clipped |= not np.array_equal(action[key], bounded)
            action[key] = bounded.astype(np.float32)
        clip_steps += int(clipped)
        if action_diagnostics is not None:
            # Observe the existing prediction/clamp only. Never feed diagnostic
            # values back into actions, rewards, termination or official success.
            action_diagnostics.record(
                actions[0, step % args.execute_steps].detach().cpu().numpy(), requested,
                np.concatenate([action[key] for key in keys]))
        observation, _, term, trunc, final_info = env.step(action)
        terminated, truncated = _bool(term), _bool(trunc)
        success = _bool(final_info.get("success", False))
        success_once |= success
        if success and first_success_step is None:
            first_success_step = step + 1
        row = _task_events(raw, final_info)
        row.update(step=step + 1, official_success=success)
        events.append(row)
        ever_grasped[0] |= row.get("left_grasped", False)
        ever_grasped[1] |= row.get("right_grasped", False)
        # Success may precede physical release; keep observing through the budget.
        if truncated:
            break
    final = events[-1]
    final_success = _bool(final_info.get("success", False))
    grasp_keys = ("left_grasped", "right_grasped")
    release_observed = (all(key in final for key in grasp_keys)
                        and all(ever_grasped) and not any(final[key] for key in grasp_keys))
    if final_success:
        failure_stage = None
    elif not all(ever_grasped):
        failure_stage = "grasp_not_observed"
    elif any(final.get(key, False) for key in grasp_keys):
        failure_stage = "transport_or_release_incomplete"
    else:
        failure_stage = "placement_not_successful"
    result = {
        "seed": seed, "steps": len(events), "official_success_once": success_once,
        "official_success_at_end": final_success, "first_success_step": first_success_step,
        "terminated": terminated, "truncated": truncated,
        "timeout": len(events) >= args.max_episode_steps and not final_success,
        "failure_stage": failure_stage, "action_clipped_steps": clip_steps,
        "physical": {"ever_grasped": ever_grasped, "release_observed": release_observed,
                     "release_observed_semantics": RELEASE_OBSERVED_SEMANTICS,
                     "grasp_placement": grasp_placement_measurements(events),
                     "final": final, "settling": settling_measurements(events)}, "events": events,
    }
    if action_diagnostics is not None:
        directory = Path(args.output_dir) / "action_diagnostics"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{env_id}_seed_{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.npz"
        np.savez_compressed(path, **action_diagnostics.arrays())
        result["action_limit_diagnostics"] = {
            **action_diagnostics.summary(), "arrays_path": str(path.resolve())}
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sub-config", required=True)
    parser.add_argument("--env-id", help="Override dataset repo name when it is not a registered environment")
    parser.add_argument("--robot", help="Homogeneous pair; defaults to the recorded training robot")
    parser.add_argument("--sim-root", default="./data/sim", help="Retained for existing command compatibility")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--seeds-file")
    parser.add_argument("--seed-split", default="evaluation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--execute-steps", type=int, default=1,
                        help="Actions executed per prediction before observing and predicting again")
    parser.add_argument("--control-mode", default="pd_joint_pos")
    parser.add_argument("--sim-backend", default="physx_cpu", choices=["physx_cpu"])
    parser.add_argument("--state-type", default=None)
    parser.add_argument("--pred-horizon", type=int, default=None)
    parser.add_argument("--obs-horizon", type=int, default=None)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-trajectory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--action-diagnostics", action="store_true",
                        help="Record normalized/requested/executed actions and per-coordinate limit statistics; read-only")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def main(argv=None, *, checkpoint_loader=load_checkpoint, prepare_episode=None,
         report_prefix: str = "act_single"):
    """Run the shared native policy rollout with optional policy conditioning.

    ``prepare_episode`` receives the policy and immutable checkpoint/configuration
    before a seeded rollout. It returns reporting metadata; environment actions,
    reset, success and termination remain owned by this evaluation path.
    """
    args = parse_args(argv)
    if args.num_envs != 1 or args.max_episode_steps < 1 or args.control_mode != "pd_joint_pos":
        raise ValueError("This baseline uses one CPU environment, pd_joint_pos and a positive step budget")
    seeds = evaluation_seeds(args)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    agent, train_args = checkpoint_loader(args.checkpoint, args, device)
    contract = checkpoint.get("data_contract", {})
    args.native_contract = contract.get("native") or {}
    source_image_size = contract.get("source_image_size", train_args.image_size)
    if contract.get("fps") != args.fps:
        raise ValueError("Evaluation fps must equal the checkpoint's recorded training fps")
    if not 1 <= args.execute_steps <= train_args.pred_horizon:
        raise ValueError("execute_steps must be between 1 and the checkpoint prediction horizon")
    normalizer = SingleTaskNormalizer(checkpoint.get("dataset_stats", []),
                                      contract.get("normalization_method", train_args.normalization_method),
                                      contract.get("normalization_min_range", train_args.normalization_min_range))
    configs = json.loads(Path(args.sub_config).read_text())
    results = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    result_path = output / f"{report_prefix}_{stamp}.json"
    from mani_skill.utils.wrappers.record import RecordEpisode
    from mani_skill.envs.tasks.tabletop.utils import L0_L3_utils
    L0_L3_utils.set_lr_mirror_enabled(False)
    L0_L3_utils.set_lr_mirror_robot_pose_enabled(False)
    for config in configs:
        repo_id = config["repo_id"]
        env_id = args.env_id or config.get("env_id") or args.native_contract.get("env_id") or extract_base_env_name(repo_id)
        set_l_level(extract_level(env_id))
        env = None
        try:
            recorded_robots = args.native_contract.get("robot_uids")
            robot_kwargs = {}
            selected_robots = evaluation_robot_uids(env_id, args.robot, recorded_robots)
            if selected_robots:
                robot_kwargs["robot_uids"] = selected_robots
            profile_differences = validate_recorded_profiles(args.native_contract, env_id, selected_robots)
            if env_id == "DualArmPickPlace-v1" and args.max_episode_steps != 400:
                raise ValueError("Shared pick/place evaluation uses its 400-step task budget")
            rendering = args.native_contract.get("rendering", {})
            sensor_configs = dict(rendering.get("sensor_configs", {}))
            sensor_configs.update(width=source_image_size[1], height=source_image_size[0])
            render_kwargs = {}
            if rendering.get("shader_dir") is not None:
                render_kwargs["shader_dir"] = rendering["shader_dir"]
            env = gym.make(extract_base_env_name(env_id), num_envs=1, **robot_kwargs, **render_kwargs,
                           sim_backend=args.sim_backend, obs_mode="rgb", control_mode=args.control_mode,
                           reward_mode="none", render_mode="rgb_array", max_episode_steps=args.max_episode_steps,
                           sensor_configs=sensor_configs)
            if args.save_trajectory or args.save_video:
                env = RecordEpisode(env, output_dir=str(output / "rollouts" / env_id),
                                    trajectory_name=stamp, save_trajectory=args.save_trajectory,
                                    save_video=args.save_video, video_fps=args.fps,
                                    source_type="act", source_desc="single-task ACT seeded policy evaluation")
            for seed in seeds:
                episode_metadata = (prepare_episode(agent, checkpoint, config, args, seed,
                                                    env_id, device)
                                    if prepare_episode is not None else {})
                result = evaluate_episode(agent, env, normalizer, args, train_args, seed, repo_id, device)
                result.update(episode_metadata)
                result.update(env_id=env_id, robot_uids=list(env.unwrapped.robot_uids), checkpoint=str(Path(args.checkpoint).resolve()),
                              fps=args.fps, execute_steps=args.execute_steps,
                              profile_nonphysical_differences=profile_differences)
                if args.save_trajectory:
                    env.flush_trajectory()
                    env._h5_file.flush()
                    result["trajectory_path"] = str(Path(env._h5_file.filename).resolve())
                    result["episode_id"] = int(env._episode_id)
                results.append(result)
                result_path.write_text(json.dumps(results, indent=2) + "\n")
                print(f"seed={seed} success={result['official_success_at_end']} "
                      f"once={result['official_success_once']} timeout={result['timeout']} "
                      f"stage={result['failure_stage']}")
        finally:
            if env is not None:
                env.close()
            clear_l_level()
    summary = {
        "episodes": len(results), "seeds": seeds,
        "official_success_once": sum(result["official_success_once"] for result in results),
        "official_success_at_end": sum(result["official_success_at_end"] for result in results),
        "timeouts": sum(result["timeout"] for result in results),
        "release_observed": sum(result["physical"]["release_observed"] for result in results),
        "release_observed_semantics": RELEASE_OBSERVED_SEMANTICS,
        "placed_and_ungrasped_at_end": sum(
            result["physical"]["grasp_placement"]["both_final_placed_and_ungrasped"] is True
            for result in results),
        "results": str(result_path.resolve()),
    }
    (output / f"summary_{stamp}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
