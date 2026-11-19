"""Imitator evaluation protocol: policy-agnostic rollout loop, L0-L3 level handling, TSS/nDTW metrics.

Extracted unchanged from the 1.0 Diffusion Policy baseline (``examples/baselines/diffusion_policy/eval_dp_imitator.py``)
so that evaluation no longer depends on any model implementation. A policy only needs the interface in
:mod:`imitator.eval.policy`; the command-line entry point is :mod:`imitator.eval.run`.

- input modes: video_only, language_only, video_and_language
- the task (human video / description) is given to the policy ONCE at episode start (``prepare_for_eval``)
- optional temporal aggregation of action chunks
- incremental JSON logging, auto-skip of environments that already have enough episodes
- subprocess-safe L-level differentiation
"""

import gc
import json
import time
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional
import numpy as np
import torch
import gymnasium
from tqdm import tqdm

from imitator.dataset.lerobot_paired_dataset import InputMode
from imitator.eval.trajectory_metrics import (
    TrajectoryMetrics,
    GTTrajectoryProvider,
    EpisodeActionBuffer,
)
from imitator.eval.make_env import make_eval_envs
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.envs.tasks.tabletop.utils import L0_L3_utils
from mani_skill.utils import common

L0_L3_utils.set_lr_mirror_robot_pose_enabled(False)


# =============================================================================
# JSON-based skip check
# =============================================================================

def load_existing_results(output_dir: Path, input_mode: str) -> Dict[str, Dict]:
    existing: Dict[str, Dict] = {}
    for json_file in output_dir.glob(f"*_{input_mode}_*.json"):
        try:
            with open(json_file) as f:
                results = json.load(f)
            for r in results:
                env_id = r.get("env_id")
                if env_id and r.get("status") == "success":
                    prev = existing.get(env_id)
                    if prev is None or r.get("timestamp", "") >= prev.get("timestamp", ""):
                        existing[env_id] = r
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return existing


def is_env_already_evaluated(
    env_id: str,
    existing_results: Dict[str, Dict],
    num_episodes: int,
) -> bool:
    prev = existing_results.get(env_id)
    if prev is None:
        return False
    if prev.get("num_episodes", 0) >= num_episodes:
        print(f"⏭️  Skipping {env_id}: already evaluated "
              f"({prev['num_episodes']} episodes, "
              f"success_once={prev.get('success_once_mean', 'N/A')})")
        return True
    print(f"⚠️  Partial eval for {env_id}: previous run had "
          f"{prev.get('num_episodes', '?')}/{num_episodes} episodes — re-evaluating")
    return False


# =============================================================================
# Subprocess-safe L-level differentiation
# =============================================================================

_L_ENV_VARS = {
    "L1": "MANI_SKILL_L1",
    "L2": "MANI_SKILL_L2",
    "L3": "MANI_SKILL_L3",
}


def set_l_level(level: str):
    for env_var in _L_ENV_VARS.values():
        os.environ.pop(env_var, None)
    if level in _L_ENV_VARS:
        os.environ[_L_ENV_VARS[level]] = "1"
    L0_L3_utils.set_l1_enabled(False)
    L0_L3_utils.set_l2_enabled(False)
    L0_L3_utils.set_l3_enabled(False)
    if level == "L1":
        L0_L3_utils.set_l1_enabled(True)
    elif level == "L2":
        L0_L3_utils.set_l2_enabled(True)
    elif level == "L3":
        L0_L3_utils.set_l3_enabled(True)


def clear_l_level():
    for env_var in _L_ENV_VARS.values():
        os.environ.pop(env_var, None)
    L0_L3_utils.set_l1_enabled(False)
    L0_L3_utils.set_l2_enabled(False)
    L0_L3_utils.set_l3_enabled(False)


def make_eval_envs_with_level(
    base_env_name: str,
    level: str,
    num_envs: int,
    sim_backend: str,
    env_kwargs: dict,
    other_kwargs: dict,
    video_dir: str,
    wrappers: list,
):
    set_l_level(level)
    clean_env_kwargs = {k: v for k, v in env_kwargs.items() if k != "l_level"}
    envs = make_eval_envs(
        base_env_name, num_envs, sim_backend, clean_env_kwargs, other_kwargs,
        video_dir=video_dir,
        wrappers=wrappers,
        l_level=level,
    )
    return envs


# =============================================================================
# Core evaluation function
# =============================================================================

def evaluate_with_task_encoder(
    n: int,
    agent,
    eval_envs,
    eval_kwargs: Dict,
    evaluate_processor,
    input_mode: InputMode,
    progress_bar: bool = True,
    # ── DTW arguments ────────────────────────────────────────────────────────
    dtw_provider=None,
    traj_metrics=None,
    dtw_band_ratio: float = 0.15,
) -> Dict[str, np.ndarray]:
    """
    Roll out a policy with the Imitator protocol + optional DTW trajectory metrics.

    - action_seq shape: (num_envs, pred_horizon, action_dim)
    - Correct temporal slice: actions_to_take[:, ts % query_frequency]
    - Env step format: {"panda_wristcam-0": (num_envs, 8), "panda_wristcam-1": (num_envs, 8)}
    """
    env_id             = eval_kwargs.get("env_id")
    delta_control      = eval_kwargs.get("delta_control")
    pred_horizon       = eval_kwargs.get("pred_horizon")
    temporal_agg       = eval_kwargs.get("temporal_agg")
    light_temporal_agg = eval_kwargs.get("light_temporal_agg", False)
    tagg_window        = eval_kwargs.get("tagg_window", 4)
    max_timesteps      = eval_kwargs.get("max_timesteps")
    device             = eval_kwargs.get("device")
    sim_backend        = eval_kwargs.get("sim_backend")

    # light_temporal_agg takes priority over temporal_agg
    if light_temporal_agg:
        temporal_agg = False

    # ── Action dim from env (dual arm: 8 + 8 = 16) ───────────────────────────
    if isinstance(eval_envs.single_observation_space, gymnasium.spaces.Box):
        action_dim = eval_envs.action_space.shape[-1]
    else:
        action_dim = (eval_envs.action_space["panda_wristcam-0"].shape[-1] +
                      eval_envs.action_space["panda_wristcam-1"].shape[-1])

    num_envs = eval_envs.num_envs

    if light_temporal_agg:
        # ── Lightweight sliding-window temporal aggregation ────────────────
        # Query every tagg_window steps; keep last tagg_window predictions
        # and exponentially-weight them. Much faster than full temp agg (1
        # query per step) while still smoothing chunk-boundary transitions.
        query_frequency = tagg_window
        action_window   = deque(maxlen=tagg_window)
    elif temporal_agg:
        query_frequency  = 1
        all_time_actions = torch.zeros(
            [num_envs, max_timesteps, max_timesteps + pred_horizon, action_dim],
            device=device,
        )
    else:
        query_frequency  = pred_horizon
        actions_to_take  = torch.zeros([num_envs, pred_horizon, action_dim], device=device)

    agent.eval()

    # ── Prepare task inputs ───────────────────────────────────────────────────
    human_video  = None
    human_desc   = None

    if input_mode in [InputMode.VIDEO_ONLY, InputMode.VIDEO_AND_LANGUAGE]:
        human_video = evaluate_processor.get_video(env_id, num_envs).to(device)

    if input_mode in [InputMode.LANGUAGE_ONLY, InputMode.VIDEO_AND_LANGUAGE]:
        human_desc = [evaluate_processor.get_task_description(env_id)] * num_envs

    # ── DTW helpers ───────────────────────────────────────────────────────────
    compute_dtw = (dtw_provider is not None)
    if compute_dtw:
        if traj_metrics is None:
            traj_metrics = TrajectoryMetrics(
                band_ratio=dtw_band_ratio
            )
        gt_traj = dtw_provider.sample_gt_trajectory(env_id, seed=None, normalize=False)
        if gt_traj is None:
            print(f"⚠️  No GT trajectory for {env_id}; TSS skipped.")
            compute_dtw = False
        action_buf = EpisodeActionBuffer(num_envs=num_envs)
        # Track first success step per env (-1 = not yet succeeded)
        success_steps = torch.full((num_envs,), -1, dtype=torch.long, device=device)

    with torch.no_grad():
        eval_metrics  = defaultdict(list)
        obs, info     = eval_envs.reset()
        ts, eps_count = 0, 0

        pbar = tqdm(
            total=n,
            desc=f"Eval [{input_mode.value}]{'[+DTW]' if compute_dtw else ''}",
            disable=not progress_bar,
            unit="ep",
        )

        while eps_count < n:
            obs = {k: common.to_tensor(v, device) for k, v in obs.items()}
            if not delta_control:
                obs["state"], obs["rgb"] = evaluate_processor.normalize_state_rgb(
                    obs["state"], obs["rgb"], env_id
                )

            # ── Encode task ONCE at episode start ─────────────────────────────
            if ts == 0:
                robot_obs_for_task = {
                    "states": obs.get("state", obs.get("states")),
                    "view_1": obs.get("rgb",   obs.get("view_1")),
                }
                agent.prepare_for_eval(
                    human_video=human_video,
                    robot_obs=robot_obs_for_task,
                    human_desc=human_desc,
                )

            # ── Query policy every query_frequency steps ──────────────────────
            if ts % query_frequency == 0:
                action_seq = agent.get_action(obs)
                if action_seq.dim() == 2:
                    action_seq = action_seq.unsqueeze(0)
                if light_temporal_agg:
                    action_window.append(action_seq)

            # ── Temporal aggregation or direct slice ──────────────────────────
            if light_temporal_agg:
                # Sliding-window weighted average of the last tagg_window predictions.
                # For each past prediction, find which index in that chunk corresponds
                # to the current timestep:
                #   age=0 → newest (queried this step), index = step_in_chunk
                #   age=1 → queried tagg_window steps ago, index = step_in_chunk + tagg_window
                #   ...
                # Skip any prediction whose index falls outside pred_horizon.
                step_in_chunk = ts % query_frequency
                acts_list, ages = [], []
                for age, past_pred in enumerate(reversed(action_window)):
                    idx = step_in_chunk + age * query_frequency
                    if idx < pred_horizon:
                        acts_list.append(past_pred[:, idx])  # (num_envs, action_dim)
                        ages.append(age)
                if acts_list:
                    w = torch.exp(-0.1 * torch.tensor(ages, dtype=torch.float32, device=device))
                    w = w / w.sum()
                    stacked = torch.stack(acts_list, dim=1)  # (num_envs, valid, action_dim)
                    raw_action = (stacked * w.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
                else:
                    # Fallback (shouldn't happen in practice)
                    raw_action = action_window[-1][:, step_in_chunk]
            elif temporal_agg:
                all_time_actions[:, ts, ts:ts + pred_horizon] = action_seq
                actions_for_curr_step = all_time_actions[:, :, ts]
                populated = torch.zeros(max_timesteps, dtype=torch.bool, device=device)
                populated[max(0, ts + 1 - pred_horizon):ts + 1] = True
                actions_for_curr_step = actions_for_curr_step[:, populated]
                k = 0.01
                exp_w = torch.exp(
                    -k * torch.arange(actions_for_curr_step.shape[1], device=device)
                )
                exp_w = (exp_w / exp_w.sum()).unsqueeze(0).unsqueeze(-1).expand(num_envs, -1, -1)
                raw_action = (actions_for_curr_step * exp_w).sum(dim=1)
            else:
                if ts % query_frequency == 0:
                    actions_to_take = action_seq
                raw_action = actions_to_take[:, ts % query_frequency]

            # ── Denormalize & step ────────────────────────────────────────────
            _action = (
                evaluate_processor.denormalize_action(raw_action, env_id)
                if not delta_control else raw_action
            )

            if sim_backend == "physx_cpu":
                _action = _action.cpu().numpy()

            # ── Collect action for DTW ────────────────────────────────────────
            if compute_dtw:
                action_buf.append(_action)

            action = {
                "panda_wristcam-0": _action[:, :8],
                "panda_wristcam-1": _action[:, 8:16],
            }

            obs, rew, terminated, truncated, info = eval_envs.step(action)
            ts += 1

            # Track first success per env for TSS (valid while not truncated)
            if compute_dtw and not truncated.any():
                _cur_s = info.get("success")
                if _cur_s is not None:
                    _cur_s_t = torch.as_tensor(_cur_s, dtype=torch.bool, device=device)
                    _new_s = (success_steps == -1) & _cur_s_t
                    success_steps[_new_s] = ts

            if truncated.any():
                if isinstance(info["final_info"], dict):
                    for k, v in info["final_info"]["episode"].items():
                        eval_metrics[k].append(v.float().cpu().numpy())
                else:
                    for final_info in info["final_info"]:
                        for k, v in final_info["episode"].items():
                            eval_metrics[k].append(v)

                # ── TSS: split by success / fail ─────────────────────────────────────
                if compute_dtw:
                    _fi = info.get("final_info", {})
                    _ls = (_fi.get("success") if isinstance(_fi, dict) else None)
                    if _ls is not None:
                        _ls_t = torch.as_tensor(_ls, dtype=torch.bool, device=device)
                        success_steps[(success_steps == -1) & _ls_t] = ts
                    pred_trajs = action_buf.get_and_reset()
                    ss_np      = success_steps.cpu().numpy()
                    success_steps[:] = -1
                    T_gt = len(gt_traj)   # GT length for fair fail comparison
                    for env_i, pred_traj in enumerate(pred_trajs):
                        ss = int(ss_np[env_i])
                        if ss >= 2:
                            # Successful: trim to first-success step
                            try:
                                m = traj_metrics.compute(pred_traj[:ss], gt_traj)
                                eval_metrics["tss_success"].append(np.array(m["tss"],  dtype=np.float32))
                                eval_metrics["ndtw_success"].append(np.array(m["ndtw"], dtype=np.float32))
                            except Exception as exc:
                                print(f"⚠️  TSS(success) env {env_i}: {exc}")
                        else:
                            # Failed: trim to GT length for fair shape comparison
                            pred_trimmed = pred_traj[:T_gt]
                            if len(pred_trimmed) >= 2:
                                try:
                                    m = traj_metrics.compute(pred_trimmed, gt_traj)
                                    eval_metrics["tss_fail"].append(np.array(m["tss"],  dtype=np.float32))
                                    eval_metrics["ndtw_fail"].append(np.array(m["ndtw"], dtype=np.float32))
                                except Exception as exc:
                                    print(f"⚠️  TSS(fail) env {env_i}: {exc}")
                pbar.update(num_envs)
                eps_count += num_envs
                ts = 0

                if light_temporal_agg:
                    action_window.clear()
                elif temporal_agg:
                    all_time_actions = torch.zeros(
                        [num_envs, max_timesteps, max_timesteps + pred_horizon, action_dim],
                        device=device,
                    )

                agent.clear_cache()
                obs, info = eval_envs.reset()

                if compute_dtw:
                    new_gt = dtw_provider.sample_gt_trajectory(env_id, seed=None)
                    if new_gt is not None:
                        gt_traj = new_gt

        pbar.close()

    agent.train()

    for k in eval_metrics:
        eval_metrics[k] = np.stack(eval_metrics[k])

    return eval_metrics



# =============================================================================
# Config helpers
# =============================================================================

def load_eval_config(config_path: str) -> Dict[str, Any]:
    config_path = Path(config_path)
    if config_path.suffix != ".txt":
        raise ValueError(f"Only .txt config files are supported. Got: {config_path.suffix}")
    with open(config_path, "r") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return {
        "eval_name":    config_path.stem,
        "description":  f"Evaluation from {config_path.name}",
        "environments": [{"env_id": env_id} for env_id in lines],
    }


def extract_base_env_name(env_id: str) -> str:
    if env_id.startswith("L") and "_" in env_id:
        parts = env_id.split("_", 1)
        if len(parts) == 2 and parts[0] in ["L0", "L1", "L2", "L3"]:
            level, base = parts[0], parts[1]
            if level == "L3":
                if "-v" in base:
                    name_part, version_part = base.rsplit("-v", 1)
                    return f"{name_part}L3-v{version_part}"
                return f"{base}L3"
            return base
    return env_id


def extract_level(env_id: str) -> str:
    if env_id.startswith("L") and "_" in env_id:
        parts = env_id.split("_", 1)
        if len(parts) == 2 and parts[0] in ["L0", "L1", "L2", "L3"]:
            return parts[0]
    return "L0"


def save_results_to_json(results: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)



# =============================================================================
# Evaluate a single environment
# =============================================================================

def evaluate_single_env(
    agent,
    env_id: str,
    args,
    evaluate_processor,
    output_dir: Path,
    model_config: Dict,
    input_mode: InputMode,
    dtw_provider=None,
    traj_metrics=None,
) -> Dict:
    base_env_name = extract_base_env_name(env_id)
    level         = extract_level(env_id)

    print(f"\n{'='*60}")
    print(f"🎯 Evaluating: {env_id}")
    print(f"   Base env: {base_env_name}, Level: {level}")
    print(f"{'='*60}")

    obs_mode = "rgbd" if model_config["include_depth"] else "rgb"
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="dense",
        obs_mode=obs_mode,
        render_mode="rgb_array",
        max_episode_steps=args.max_episode_steps, 
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
    )
    other_kwargs = dict(obs_horizon=model_config["obs_horizon"])

    # NOTE: initialise to None so the finally block can safely guard
    # eval_envs.close() even if env creation itself raises (e.g. SAPIEN
    # Vulkan resource exhaustion after several sequential env creations).
    eval_envs = None
    try:
        eval_envs = make_eval_envs_with_level(
            base_env_name=base_env_name,
            level=level,
            num_envs=args.num_envs,
            sim_backend=args.sim_backend,
            env_kwargs=env_kwargs,
            other_kwargs=other_kwargs,
            video_dir=str(output_dir / "videos" / env_id),
            wrappers=[FlattenRGBDObservationWrapper],
        )
        eval_kwargs = dict(
            env_id=env_id,
            delta_control=("delta" in args.control_mode),
            pred_horizon=model_config["pred_horizon"],
            temporal_agg=args.temporal_agg,
            light_temporal_agg=args.light_temporal_agg,
            tagg_window=args.tagg_window,
            max_timesteps=args.max_episode_steps,
            device=args.device,
            sim_backend=args.sim_backend,
        )

        eval_metrics = evaluate_with_task_encoder(
            n=args.num_episodes,
            agent=agent,
            eval_envs=eval_envs,
            eval_kwargs=eval_kwargs,
            evaluate_processor=evaluate_processor,
            input_mode=input_mode,
            progress_bar=True,
            dtw_provider=dtw_provider,
            traj_metrics=traj_metrics,
            dtw_band_ratio=args.dtw_band_ratio,
        )

        results = {
            "env_id":      env_id,
            "level":       level,
            "input_mode":  input_mode.value,
            "policy":      model_config.get("name", type(agent).__name__),
            "num_episodes": args.num_episodes,
            "timestamp":   datetime.now().strftime("%Y%m%d_%H%M%S"),
            "status":      "success",
        }
        for k, v in eval_metrics.items():
            results[f"{k}_mean"] = float(v.mean())
            results[f"{k}_std"]  = float(v.std())

        print(f"\n📊 Results [{input_mode.value}]:")
        for k in ["success_once", "success_at_end", "return"]:
            if f"{k}_mean" in results:
                print(f"   {k}: {results[f'{k}_mean']:.4f} ± {results[f'{k}_std']:.4f}")
        # Ensure TSS fields always present; compute wTSS
        if args.compute_dtw:
            for _k in ["tss_success", "ndtw_success", "tss_fail", "ndtw_fail"]:
                if f"{_k}_mean" not in results:
                    results[f"{_k}_mean"] = None
                    results[f"{_k}_std"]  = None
            _sr  = results.get("success_once_mean") or 0.0
            _tss = results.get("tss_success_mean")
            results["wtss_mean"] = round(_sr * _tss, 6) if _tss is not None else 0.0

        if args.compute_dtw:
            print("\n📏 Trajectory Similarity:")
            print(f"   wTSS        : {results['wtss_mean']:.4f}  (= SR × TSS_success)")
            if results.get("tss_success_mean") is not None:
                print(f"   TSS_success : {results['tss_success_mean']:.4f} ± {results['tss_success_std']:.4f}  (higher=better)")
                print(f"   nDTW_success: {results['ndtw_success_mean']:.4f} ± {results['ndtw_success_std']:.4f}")
            else:
                print("   TSS_success : N/A  (no successful episodes)")
            if results.get("tss_fail_mean") is not None:
                print(f"   TSS_fail    : {results['tss_fail_mean']:.4f} ± {results['tss_fail_std']:.4f}")
                print(f"   nDTW_fail   : {results['ndtw_fail_mean']:.4f} ± {results['ndtw_fail_std']:.4f}")
        return results

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "env_id":    env_id,
            "level":     level,
            "input_mode": input_mode.value,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "status":    "error",
            "error":     str(e),
        }

    finally:
        # Close the env and explicitly delete the reference so Python's GC can
        # immediately trigger SAPIEN's destructor.  Without del + gc.collect(),
        # Vulkan/CUDA render contexts accumulate across sequential env creations
        # and exhaust GPU render resources (typically crashes on the 5th env
        # when a large model occupies ~10 GB of VRAM).
        # time.sleep(1) gives the GPU driver time to fully release Vulkan
        # handles before the next RenderSystem is initialised.
        if eval_envs is not None:
            try:
                eval_envs.close()
            except Exception:
                pass
            del eval_envs
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)
        clear_l_level()
