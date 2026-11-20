"""Evaluate a HyperICL checkpoint in the simulator (reuses the Diffusion Policy eval pipeline exactly
like eval_template_imitator.py). Extra switches:

    --ttt-steps K            latent TTT steps on the human video before acting (0 = one-shot)
    --state-update-every k   closed-loop recurrence: update the skill state every k get_action calls
                             (0 = frozen one-shot parameters for the whole episode)

Example
    python -m examples.baselines.hypericl.eval_hypericl_imitator --eval-config <env_list.txt> \
        --checkpoint runs/s_E/checkpoints/final_model.pt --output-dir results/s_E \
        --human-config ... --sim-config ... --task-mapping ... --human-task-desc ... --sim-task-desc ... \
        --ttt-steps 2 --state-update-every 1
"""
from __future__ import annotations

import argparse
import gc
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from tqdm import tqdm

from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from imitator.dataset.lerobot_paired_dataset import InputMode
from imitator.eval.processor import (
    HumanVideoSimEvaluateProcessor, HumanVideoSimEvaluateProcessorConfig)
from examples.baselines.diffusion_policy.eval_dp_imitator import (
    evaluate_with_task_encoder, load_eval_config, load_existing_results, is_env_already_evaluated,
    extract_base_env_name, extract_level, make_eval_envs_with_level, clear_l_level, save_results_to_json)

from .agent import HyperICLAgent
from .config import HyperICLArgs
from .train_hypericl_imitator import load_into


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--human-root", default="demos/demo_data")
    p.add_argument("--sim-root", default="demos/imitator_data")
    p.add_argument("--human-config", required=True)
    p.add_argument("--sim-config", required=True)
    p.add_argument("--task-mapping", required=True)
    p.add_argument("--human-task-desc", required=True)
    p.add_argument("--sim-task-desc", required=True)
    p.add_argument("--input-mode", default="video_only")
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--max-episode-steps", type=int, default=500)
    p.add_argument("--sim-backend", default="physx_cpu")
    p.add_argument("--control-mode", default="pd_joint_pos")
    p.add_argument("--shader", default="rt-fast")
    p.add_argument("--device", default="cuda")
    p.add_argument("--ttt-steps", type=int, default=0)
    p.add_argument("--state-update-every", type=int, default=1)
    p.add_argument("--temporal-agg", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--light-temporal-agg", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--tagg-window", type=int, default=4)
    return p.parse_args()


def load_agent(checkpoint_path: str, device: torch.device, ttt_steps: int, state_update_every: int):
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_args = HyperICLArgs()
    for key, value in ck.get("args", {}).items():
        if hasattr(train_args, key):
            setattr(train_args, key, value)
    train_args.stage = ck.get("stage", train_args.stage)
    train_args.ttt_steps_eval = ttt_steps
    train_args.state_update_every = state_update_every
    agent = HyperICLAgent(train_args, device)
    load_into(agent, checkpoint_path, device)
    agent.eval()
    model_config = {"cameras": train_args.cameras, "include_depth": train_args.include_depth,
                    "image_size": train_args.image_size, "num_video_frames": train_args.frozen_backbone_num_frames,
                    "state_type": train_args.state_type, "single_arm": train_args.single_arm,
                    "pred_horizon": train_args.pred_horizon, "obs_horizon": train_args.obs_horizon}
    return agent, model_config


def build_evaluate_processor(args, model_config: Dict):
    return HumanVideoSimEvaluateProcessor(HumanVideoSimEvaluateProcessorConfig(
        human_root=args.human_root, human_split="train", human_dataset_file=args.human_config,
        human_task_description_file=args.human_task_desc, human_cameras=model_config["cameras"],
        human_include_depth=model_config["include_depth"], human_num_frames=model_config["num_video_frames"],
        human_image_size=model_config["image_size"], human_video_backend="torchcodec", human_fps=30,
        sim_root=args.sim_root, sim_split="train", sim_dataset_file=args.sim_config,
        sim_task_description_file=args.sim_task_desc, sim_state_type=model_config["state_type"],
        sim_single_arm=model_config["single_arm"], normalization_method="bounds_q99",
        task_mapping_file=args.task_mapping))


def evaluate_one_env(env_id, agent, args, model_config, evaluate_processor, input_mode: InputMode, output_dir: Path) -> Dict:
    base_env_name, level = extract_base_env_name(env_id), extract_level(env_id)
    obs_mode = "rgbd" if model_config["include_depth"] else "rgb"
    env_kwargs = dict(control_mode=args.control_mode, reward_mode="dense", obs_mode=obs_mode, render_mode="rgb_array",
                      max_episode_steps=args.max_episode_steps, sensor_configs=dict(shader_pack=args.shader),
                      human_render_camera_configs=dict(shader_pack=args.shader))
    eval_envs = None
    try:
        eval_envs = make_eval_envs_with_level(
            base_env_name=base_env_name, level=level, num_envs=args.num_envs, sim_backend=args.sim_backend,
            env_kwargs=env_kwargs, other_kwargs=dict(obs_horizon=model_config["obs_horizon"]),
            video_dir=str(output_dir / "videos" / env_id), wrappers=[FlattenRGBDObservationWrapper])
        metrics = evaluate_with_task_encoder(
            n=args.num_episodes, agent=agent, eval_envs=eval_envs,
            eval_kwargs=dict(env_id=env_id, delta_control=("delta" in args.control_mode),
                pred_horizon=model_config["pred_horizon"], temporal_agg=args.temporal_agg,
                light_temporal_agg=args.light_temporal_agg, tagg_window=args.tagg_window,
                max_timesteps=args.max_episode_steps, device=args.device, sim_backend=args.sim_backend,
            ),
            evaluate_processor=evaluate_processor, input_mode=input_mode, progress_bar=True,
            dtw_provider=None, traj_metrics=None)
        result = {"env_id": env_id, "base_env_name": base_env_name, "level": level, "status": "success",
                  "num_episodes": args.num_episodes, "ttt_steps": args.ttt_steps,
                  "state_update_every": args.state_update_every,
                  "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")}
        for key, value in metrics.items():
            result[f"{key}_mean"], result[f"{key}_std"] = float(value.mean()), float(value.std())
        return result
    except Exception as exc:
        return {"env_id": env_id, "base_env_name": base_env_name, "level": level, "status": "error",
                "error": str(exc), "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")}
    finally:
        if eval_envs is not None:
            eval_envs.close()
        del eval_envs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(1)
        clear_l_level()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    input_mode = InputMode(args.input_mode)
    device = torch.device(args.device)
    eval_config = load_eval_config(args.eval_config)
    eval_name = eval_config.get("eval_name", "hypericl_eval")
    tag = f"{eval_name}_ttt{args.ttt_steps}_upd{args.state_update_every}"
    existing = load_existing_results(output_dir, input_mode.value)
    pending = [c["env_id"] for c in eval_config["environments"]
               if not is_env_already_evaluated(c["env_id"], existing, args.num_episodes)]
    if not pending:
        print("All environments already evaluated."); return
    agent, model_config = load_agent(args.checkpoint, device, args.ttt_steps, args.state_update_every)
    evaluate_processor = build_evaluate_processor(args, model_config)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"{tag}_{input_mode.value}_{ts}.json"
    csv_path = output_dir / f"{tag}_{input_mode.value}_{ts}.csv"
    all_results: List[Dict] = []
    for env_id in tqdm(pending, desc="Evaluating"):
        result = evaluate_one_env(env_id, agent, args, model_config, evaluate_processor, input_mode, output_dir)
        all_results.append(result)
        save_results_to_json(all_results, json_path)
        if result["status"] == "success":
            print(f"{env_id}: success_once={result.get('success_once_mean', 0.0):.4f}")
        else:
            print(f"{env_id}: ERROR: {result['error']}")
    pd.DataFrame(all_results).to_csv(csv_path, index=False)
    print(f"Saved JSON: {json_path}\nSaved CSV : {csv_path}")


if __name__ == "__main__":
    main()
