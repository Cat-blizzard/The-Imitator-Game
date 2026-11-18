# ACT Baseline

This directory contains the ACT-based vision-action baseline used in the Imitator Game experiments. The policy receives a robot observation stream together with a task representation extracted from the human demonstration video, and predicts a chunk of dual-arm actions with a DETR-style transformer decoder.

ACT is the Action Chunking with Transformers policy originally proposed for bimanual imitation learning (see [Citation](#citation)).

## Entrypoints

- Training: `train_act_imitator.py`, `train_act_single.py`
- Evaluation: `eval_act_imitator.py`, `parallel_eval_act.py`, `eval_act_single.py`
- Shared dependencies: `examples/baselines/lerobot_dataset` and `examples/baselines/encoders`

## Expected execution context

Run from the repository root (see top-level [`README.md`](../../../README.md) for environment setup):

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

## Reference training command

The training script uses `tyro` over a dataclass configuration. The command below shows the frozen task-video (video-only) configuration used in the reported experiments:

```bash
TAG=<15|30|45>
BACKBONE=<dinov2_vitl14|siglip2_so400m|videomae_large>

python -m examples.baselines.act.train_act_imitator \
  --human-root demos/demo_data \
  --sim-root demos/imitator_data \
  --human-dataset-file examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_${TAG}.json \
  --sim-dataset-file examples/baselines/lerobot_dataset/config/exp_configs/sim_train_config_${TAG}.json \
  --task-mapping-file examples/baselines/lerobot_dataset/task_mapping.json \
  --human-task-description-file examples/baselines/lerobot_dataset/task_desc/human_desc.json \
  --sim-task-description-file examples/baselines/lerobot_dataset/task_desc/sim_desc.json \
  --input-mode video_only \
  --task-encoder-type frozen_backbone \
  --frozen-backbone-type ${BACKBONE} \
  --frozen-backbone-num-frames 10 \
  --frozen-backbone-adapter-layers 1 \
  --frozen-backbone-seq-patches 32 \
  --pred-horizon 24 \
  --batch-size 128 \
  --total-epochs 10 \
  --warmup-epochs 1 \
  --lr 1e-4 \
  --control-mode pd_joint_pos \
  --save-epoch-freq 5 \
  --num-eval-envs 1 \
  --num-eval-episodes 1 \
  --env-id TwoRobotPourCup-v1 \
  --max-episode-steps 500 \
  --no-include-depth
```

Additional model hyperparameters used in the reported ablation runs: `--hidden-dim 1024 --dim-feedforward 4096 --nheads 16 --enc-layers 12 --dec-layers 12`. Checkpoints are written to `runs/<run_name>/checkpoints/`, including `final_model.pt`.

## Reference evaluation command

`eval_config.txt` should contain one environment ID per line, for example `L0_TwoRobotStirSpoon-v1` or `L3_TwoRobotPourCup-v1`.

```bash
BACKBONE=<dinov2_vitl14|siglip2_so400m|videomae_large>

python -m examples.baselines.act.eval_act_imitator \
  --eval-config path/to/eval_config.txt \
  --checkpoint path/to/final_model.pt \
  --human-root demos/demo_data \
  --sim-root demos/imitator_data \
  --human-config path/to/human_eval_config.json \
  --sim-config path/to/sim_eval_config.json \
  --task-mapping examples/baselines/lerobot_dataset/task_mapping.json \
  --human-task-desc examples/baselines/lerobot_dataset/task_desc/human_desc.json \
  --sim-task-desc examples/baselines/lerobot_dataset/task_desc/sim_desc.json \
  --input-mode video_only \
  --task-encoder-type frozen_backbone \
  --frozen-backbone-type ${BACKBONE} \
  --frozen-backbone-num-frames 10 \
  --pred-horizon 24 \
  --obs-horizon 1 \
  --action-dim 16 \
  --state-dim 18 \
  --num-episodes 10 \
  --num-envs 1 \
  --max-episode-steps 500 \
  --control-mode pd_joint_pos \
  --obs-mode rgb \
  --sim-backend physx_cpu \
  --shader rt-fast
```

## Experiment scripts

The reported ACT experiments are reproduced by the launchers under [`../exp_scripts/`](../exp_scripts/):

- [`exp_scripts/act/run_exp_act.sh`](../exp_scripts/act/run_exp_act.sh) — parallel 45/30/15-task frozen-backbone training on GPUs 0/1/2.
- [`exp_scripts/act/run_eval_act.sh`](../exp_scripts/act/run_eval_act.sh) — parallel evaluation on the seen/unseen benchmark tasks. Point `RUN_DIR_{45,30,15}` at the training run directories.

## Single-task simulation data

`train_act_single` trains from robot demonstrations without a human-video task
condition; `eval_act_single` runs the trained policy in simulation. Datasets can
come from either MPLib or cuRobo trajectories. For generation and conversion of
the synchronous Panda pill data, see [synchronous dual-arm generation](../../../docs/dual-arm-sync.md#replay-and-dataset-conversion).

The following small interface check assumes a converted 20 Hz dataset named
`TwoRobotPickPillToRegions-v1` under `imitator_data`. Save this selection as
`pill-datasets.json`:

```json
[{"repo_id": "TwoRobotPickPillToRegions-v1", "root": "TwoRobotPickPillToRegions-v1", "train": "[0]"}]
```

Selecting episode 0 explicitly avoids an empty training split when the dataset
contains only one episode. Train and reload the checkpoint for a simulation
rollout from a different initial seed:

```bash
python -m examples.baselines.act.train_act_single \
  --sub-config pill-datasets.json --sim-root imitator_data \
  --cameras zed2i --state-dim 18 --action-dim 16 \
  --obs-horizon 1 --pred-horizon 16 --batch-size 8 --num-workers 0 \
  --total-epochs 1 --save-freq-epochs 1 --no-enable-augmentation \
  --output-dir act-pill

python -m examples.baselines.act.eval_act_single \
  --checkpoint act-pill/checkpoint_epoch_0001.pt \
  --sub-config pill-datasets.json --sim-root imitator_data \
  --camera zed2i --seed 402 --num-envs 1 --num-episodes 1 \
  --max-episode-steps 200 --sim-backend physx_cpu --no-compute-dtw \
  --output-dir act-pill-eval
```

Single-task training reads the sampling frequency from the selected datasets'
`meta/info.json` and rejects mixed rates. Evaluation with `--camera zed2i`
preserves training's RGB range, checkpoint normalization statistics, 18 physical
joint coordinates and 16 action coordinates; it also checks the environment's
control frequency against the recorded training frequency. One episode and one
epoch exercise these interfaces, rather than measure a learned-policy success
rate. Use separate training, development and evaluation initial states for a
learning experiment.

## Notes

- The ACT baseline depends on the shared task encoder package under `examples/baselines/encoders`.
- Training and evaluation must use compatible task encoder settings, horizons, and action dimensions.
- L1 and L2 evaluation are handled by the benchmark environment wrappers rather than separate policy settings.

## Citation

```bibtex
@article{zhao2023learning,
  title={Learning fine-grained bimanual manipulation with low-cost hardware},
  author={Zhao, Tony Z and Kumar, Vikash and Levine, Sergey and Finn, Chelsea},
  journal={arXiv preprint arXiv:2304.13705},
  year={2023}
}
```
