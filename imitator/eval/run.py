"""Evaluate any policy on The Imitator Game (L0-L3, human-video / language conditioning).

    python -m imitator.eval.run --policy examples.policy_template.policy:make_policy \
        --eval-config imitator/eval/lists/debug_eval.txt --output-dir results/my_policy \
        --input-mode video_only --human-root demos --sim-root demos

``--policy`` is ``module:factory``; the factory is called as ``factory(checkpoint=..., device=..., **policy_kwargs)``
and returns ``(policy, model_config)`` (see :mod:`imitator.eval.policy`). Ported from the 1.0 Diffusion Policy
evaluation script; only the model-specific parts (agent construction, DDIM injection) were replaced by the factory.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

from imitator.dataset.lerobot_paired_dataset import InputMode
from imitator.eval.policy import build_policy
from imitator.eval.processor import HumanVideoSimEvaluateProcessor, HumanVideoSimEvaluateProcessorConfig
from imitator.eval.rollout import (
    evaluate_single_env,
    is_env_already_evaluated,
    load_eval_config,
    load_existing_results,
    save_results_to_json,
)
from imitator.eval.trajectory_metrics import GTTrajectoryProvider, TrajectoryMetrics

_D = "imitator/dataset"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--policy", type=str, required=True, help="module:factory returning (policy, model_config)")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--policy-kwargs", type=str, default="{}", help="JSON dict passed to the factory")
    p.add_argument("--eval-config", type=str, help=".txt file, one env id per line (e.g. L1_TwoRobotStirSpoon-v1)")
    p.add_argument("--input-mode", type=str, default="video_only", choices=[m.value for m in InputMode])
    # data
    p.add_argument("--sim-config", type=str, default=f"{_D}/config/sim_config_all.json")
    p.add_argument("--human-config", type=str, default=f"{_D}/config/human_config_all.json")
    p.add_argument("--task-mapping", type=str, default=f"{_D}/task_mapping.json")
    p.add_argument("--human-task-desc", type=str, default=f"{_D}/task_desc/human_desc.json")
    p.add_argument("--sim-task-desc", type=str, default=f"{_D}/task_desc/sim_desc.json")
    p.add_argument("--human-root", type=str, default="demos")
    p.add_argument("--sim-root", type=str, default="demos")
    # evaluation
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--max-episode-steps", type=int, default=None,
                   help="Legacy evaluator: 500; native adapted original tasks: 200")
    p.add_argument("--sim-backend", type=str, default="physx_cpu")
    p.add_argument("--control-mode", type=str, default="pd_joint_pos")
    p.add_argument("--shader", type=str, default="rt-fast")
    p.add_argument("--temporal-agg", action="store_true", default=False)
    p.add_argument("--light-temporal-agg", action="store_true", default=False)
    p.add_argument("--tagg-window", type=int, default=4)
    p.add_argument("--compute-dtw", action=argparse.BooleanOptionalAction, default=None,
                   help="Legacy default: enabled; native interface smoke default: disabled")
    p.add_argument("--dtw-band-ratio", type=float, default=0.15)
    p.add_argument("--normalization-method", type=str, default=None,
                   help="Legacy default: bounds_q99; native uses the checkpoint's saved method")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--native", action="store_true", help="Use the policy factory's saved native embodiment contract")
    p.add_argument("--env-id", help="Select one native task recorded by the checkpoint")
    p.add_argument("--robot", help="Explicit native embodiment; must match the recording")
    p.add_argument("--execute-steps", type=int, default=8, help="Native actions executed before the next model query")
    p.add_argument("--seeds", type=int, nargs="+", help="Explicit native environment seeds")
    p.add_argument("--seeds-file", help="Native seed list or split mapping JSON")
    p.add_argument("--seed-split", default="evaluation")
    p.add_argument("--save-video", action="store_true")
    args = p.parse_args(argv)
    if args.max_episode_steps is None and not args.native:
        args.max_episode_steps = 500
    if args.compute_dtw is None:
        args.compute_dtw = not args.native
    if args.normalization_method is None and not args.native:
        args.normalization_method = "bounds_q99"
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.native:
        from imitator.eval.native import evaluate_native

        if args.eval_config is not None:
            raise ValueError("Native tasks come from the checkpoint; select one with --env-id")
        policy, model_config = build_policy(args.policy, args.checkpoint, torch.device(args.device),
                                             **json.loads(args.policy_kwargs))
        return evaluate_native(args, policy, model_config)
    if args.eval_config is None:
        raise ValueError("The legacy evaluator requires --eval-config")
    if args.max_episode_steps is None:
        args.max_episode_steps = 500
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_mode = InputMode(args.input_mode)

    eval_config = load_eval_config(args.eval_config)
    eval_name = eval_config.get("eval_name", "unnamed_eval")
    existing = load_existing_results(output_dir, input_mode.value)
    pending = [e for e in eval_config.get("environments", [])
               if not is_env_already_evaluated(e["env_id"], existing, args.num_episodes)]
    if not pending:
        print("Nothing to evaluate: every environment already has enough episodes.")
        return []

    device = torch.device(args.device)
    policy, model_config = build_policy(args.policy, args.checkpoint, device, **json.loads(args.policy_kwargs))

    processor = HumanVideoSimEvaluateProcessor(HumanVideoSimEvaluateProcessorConfig(
        human_root=args.human_root,
        human_split="train",
        human_dataset_file=args.human_config,
        human_task_description_file=args.human_task_desc,
        human_cameras=model_config["cameras"],
        human_include_depth=model_config["include_depth"],
        human_num_frames=model_config["num_video_frames"],
        human_image_size=tuple(model_config["image_size"]),
        human_video_backend="torchcodec",
        human_fps=30,
        sim_root=args.sim_root,
        sim_split="train",
        sim_dataset_file=args.sim_config,
        sim_task_description_file=args.sim_task_desc,
        sim_state_type=model_config["state_type"],
        sim_single_arm=model_config["single_arm"],
        normalization_method=args.normalization_method,
        task_mapping_file=args.task_mapping,
    ))

    dtw_provider = traj_metrics = None
    if args.compute_dtw:
        try:
            action_key = ("action.qpos_gripper_actions" if model_config["state_type"] in ("qpos", "mixpos")
                          else "action.eepos_gripper_actions")
            dtw_provider = GTTrajectoryProvider(sim_dataset_file=args.sim_config, sim_root=args.sim_root,
                                                action_key=action_key, normalizer=processor.normalizer,
                                                normalization_method=args.normalization_method)
            traj_metrics = TrajectoryMetrics(band_ratio=args.dtw_band_ratio)
        except Exception as exc:  # no ground-truth trajectories available
            print(f"TSS/nDTW disabled: {exc}")
            args.compute_dtw = False

    results_json = output_dir / f"{eval_name}_{input_mode.value}_{timestamp}.json"
    all_results = []
    for env_config in tqdm(pending, desc="Evaluating"):
        r = evaluate_single_env(agent=policy, env_id=env_config["env_id"], args=args, evaluate_processor=processor,
                                output_dir=output_dir, model_config=model_config, input_mode=input_mode,
                                dtw_provider=dtw_provider, traj_metrics=traj_metrics)
        all_results.append(r)
        save_results_to_json(all_results, results_json)
    try:
        import pandas as pd
        pd.DataFrame(all_results).to_csv(results_json.with_suffix(".csv"), index=False)
    except ImportError:
        pass
    print(f"Results: {results_json}")
    return all_results


if __name__ == "__main__":
    main()
