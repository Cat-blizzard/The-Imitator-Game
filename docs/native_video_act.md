# Human video to a native robot policy

The native-video ACT entry pairs human demonstrations with the selected robot's
RGB, physical joint positions and native action labels. It reuses the official
video-conditioned ACT network and frozen DINOv2 task encoder, together with the
native data converter, normalization and seeded rollout used by the single-task
baseline. One invocation can pair multiple tasks from the same embodiment with
their human demonstrations. A separate policy is trained for each embodiment.
The original single-task manifest and checkpoints remain supported; the
[two-task workflow](#7-train-one-model-on-two-tasks) uses manifest version 2.

The initial **L0 engineering pilot** uses ARX X5 on
`TwoRobotPickPillToRegions-v1`: twenty robot training trajectories, three robot
development trajectories, four human training demonstrations and separate human
development/evaluation episodes. The optimizer budget is fixed at 6,000 updates.
This checks a real video-conditioned training/checkpoint/rollout interface. It
does not cover the full L0–L3 evaluation, reproduce the official main table,
establish cross-task generalization, or transfer a Panda policy to ARX without
ARX training data. Results and their
source versions are recorded in the [validation report](native_video_act_validation.md).

## 1. Prepare the environment and data

Follow the repository [installation instructions](../README.md#installation)
and [robot asset preparation](multi-embodiment.md#prepare-assets), including the
Vulkan rendering setup. Use the same physics and embodiment asset versions for
data collection and policy evaluation.

The robot input is a converted LeRobot dataset with its original episode split
and `native_data_contract`. The human input is the corresponding TIG human task
dataset; the pill task maps to `human_H31`. Human videos provide the task
condition, and robot trajectories provide action supervision. Their frames are
not aligned in time, and human hand joints are not retargeted to robot joints.

If robot trajectories have not been collected and converted, use
[the original-task collection recipe](multi-embodiment.md#collect-and-learn-an-original-task).
Reserve evaluation seeds before collection. Keep partial and failed collection
attempts with their attempt log. The commands below assume the twenty selected
training episodes have indices `0..19` and the three development episodes have
indices `20..22`; preserve a different dataset's actual selection explicitly.

Run from the checkout with the native-video entries. Set paths for the new
server; the examples do not depend on the original development directory:

The fork's implementation branch is `feat/native-video-act`, based on
`integration/current-features`. Use that branch when preparing the checkout.

```bash
export REPO_ROOT=/path/to/The-Imitator-Game
export DATA_ROOT=/path/to/converted-native-datasets
export NATIVE_DATASET=arx_x5-pill-v1
export HUMAN_ROOT=/path/to/IG-10K/imitator_human_v1/human_H31
export RUN_ROOT=/path/to/outputs/arx-native-video-v1
export HF_HOME=/path/to/model-cache/huggingface
export TORCH_HOME=/path/to/model-cache/torch
export MS_ASSET_DIR=/path/to/maniskill-assets

cd "$REPO_ROOT"
mkdir -p "$RUN_ROOT/config"
python -m examples.baselines.act.train_act_native_video --help
python -m examples.baselines.act.eval_act_native_video --help
```

`MS_ASSET_DIR` contains the `data/robots/` and task assets directories. Training
uses pretrained observation-network and DINOv2 weights; arrange their downloads
or copy the model caches to the new server. The model is eagerly constructed
before strict checkpoint reload, so evaluation also requires the corresponding
model configuration and local cache, or network access to obtain them.

## 2. Freeze the manifests and configuration

Human manifest version 1 records a task ID and a list of demonstrations. Each
record names the human repository, episode, camera, video file, role and sampled
frames. Roles are `train`, `development` and `evaluation`. Paths may be absolute
or relative to the manifest JSON file.

LeRobot v3 can store multiple episodes in one MP4. Set
`episode_frame_range: [start, end_exclusive]` to the physical frame bounds of
that episode in the file. `frame_indices` are global file indices, not indices
relative to the episode. Alternatively, `num_frames` samples uniformly within
the declared episode bounds, including its endpoints. Do not sample uniformly
across a shared file without those bounds.

This configuration script reads the actual LeRobot episode metadata. It creates
the human manifest, native dataset selection, model configuration and fixed
environment seeds used by the small protocol:

```bash
python - <<'PY'
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

config = Path(os.environ["RUN_ROOT"]) / "config"
human = Path(os.environ["HUMAN_ROOT"]).resolve()
data = Path(os.environ["DATA_ROOT"]) / os.environ["NATIVE_DATASET"]
robot_info = json.loads((data / "meta/info.json").read_text())
assert robot_info["total_episodes"] >= 23
assert robot_info["fps"] == 20
human_info = json.loads((human / "meta/info.json").read_text())
camera = "zed2i"
video_key = f"observation.images.{camera}"
fps = human_info["features"][video_key]["info"]["video.fps"]
rows = {}
for path in sorted((human / "meta/episodes").rglob("*.parquet")):
    for row in pq.read_table(path).to_pylist():
        rows[int(row["episode_index"])] = row
demos = []
for episode, role in [(0, "train"), (1, "train"), (2, "train"),
                      (3, "train"), (4, "development"), (5, "evaluation")]:
    row = rows[episode]
    key = f"videos/{video_key}"
    start = round(row[f"{key}/from_timestamp"] * fps)
    end = round(row[f"{key}/to_timestamp"] * fps)
    assert end - start == int(row["length"])
    relative_video = human_info["video_path"].format(
        video_key=video_key, chunk_index=int(row[f"{key}/chunk_index"]),
        file_index=int(row[f"{key}/file_index"]))
    video = human / relative_video
    assert video.is_file()
    demos.append({"id": f"human_H31_ep{episode}_{role}", "repo_id": "human_H31",
                  "episode_id": episode, "camera": camera, "path": str(video),
                  "role": role, "episode_frame_range": [start, end],
                  "frame_indices": np.linspace(start, end - 1, 10, dtype=int).tolist()})
payloads = {
    "human-manifest.json": {"schema_version": 1,
        "task_id": "TwoRobotPickPillToRegions-v1", "demos": demos},
    "datasets.json": [{"repo_id": os.environ["NATIVE_DATASET"],
        "root": os.environ["NATIVE_DATASET"],
        "env_id": "TwoRobotPickPillToRegions-v1", "train": list(range(20)),
        "validation": [20, 21, 22]}],
    "model.json": {"task_encoder_type": "frozen_backbone",
        "frozen_backbone_type": "dinov2_vitl14", "frozen_backbone_lora_rank": 0,
        "frozen_backbone_num_frames": 4, "task_latent_dim": 256,
        "task_num_frames": 10, "hf_cache_dir": str(Path(os.environ["HF_HOME"]) / "hub")},
    "seeds.json": {"train": list(range(6000, 6020)),
        "development": [7000, 7001, 7002], "evaluation": list(range(8000, 8005))},
}
for name, payload in payloads.items():
    with (config / name).open("x") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
PY
```

The human manifest loads ten sampled RGB frames per clip. The configured
DINOv2 backbone uses its four-frame sampling setting. Both settings are saved
with the checkpoint. A manifest's roles are the split chosen for this engineering
run, rather than an official TIG split declaration. The training loader only
uses human `train` records; the normal evaluation entry selects `evaluation`.
Do not relabel clips already used in training as held out.

The above seed file reproduces the small protocol's allocation. When using an
existing robot dataset, check its collection manifest against these training and
development seeds. If its seeds differ, freeze a matching file with disjoint
training, development and evaluation allocations before training. Robot episode
indices and environment seeds are separate identifiers.

The loader records the video SHA256 and the actual selected frame indices.
The effective demonstration identity includes content, episode, camera, sampling
and resize settings. Different episodes in the same MP4 therefore have different
identities. Raw frozen features use this identity for caching; the trainable
adapter is recomputed during training. This prevents replacing a demonstration
from silently reusing a task-level feature cached under its repository name.

## 3. Train a native video-conditioned policy

For the ARX X5 engineering run:

```bash
python -m examples.baselines.act.train_act_native_video \
  --sub-config "$RUN_ROOT/config/datasets.json" \
  --sim-root "$DATA_ROOT" \
  --human-manifest "$RUN_ROOT/config/human-manifest.json" \
  --model-config "$RUN_ROOT/config/model.json" \
  --cameras zed2i --state-type qpos --state-dim 16 --action-dim 14 \
  --obs-horizon 1 --pred-horizon 16 \
  --normalization-method bounds_q99 --normalization-min-range 0.01 \
  --batch-size 16 --num-workers 2 --max-updates 6000 \
  --save-freq-updates 3000 --seed 1 \
  --output-dir "$RUN_ROOT/train"
```

The human DINOv2 backbone is frozen without LoRA. The human task adapter, task
normalization and ACT policy are trained; the robot ResNet18 observation backbone
is trainable with its separate `lr_backbone`. Native action chunks use the robot
dataset's explicit padding mask. Normalization statistics are computed only from
the selected robot training episodes and embedded in checkpoints.

This command retains the native trainer's default ACT architecture and learning
rates. Model JSON selects the video encoder; the native training arguments own
state/action sizes, ACT architecture, horizons and robot camera settings. The
final checkpoint is `train/checkpoint_step_006000.pt`. A checkpoint at 3,000
updates is also retained. Configuration, optimization metrics and the final
training summary are written beside the weights. Use a new output directory for
another run; the metrics log is created without overwriting an existing run.

Six thousand updates is the fixed budget for this interface experiment. It is
not evidence of convergence. The initial protocol selects the final checkpoint
in advance; independent evaluation outcomes are not used to pick a checkpoint.

Native dimensions depend on the recorded embodiment:

| Robot selection | Dual-arm qpos width | Dual-arm action width |
| --- | ---: | ---: |
| `panda` (pill wrist-camera pair) | 18 | 16 |
| `piper` | 16 | 14 |
| `arx_x5` | 16 | 14 |
| `ur5_wsg` | 16 | 14 |
| `xarm6_robotiq` | 24 | 14 |

For another supported pair, use that pair's own dataset, configuration, output
directory and dimensions. The ARX setting `normalization_min_range=0.01` is
explicitly inherited from its native baseline; the other existing native
baselines use zero. Record a different choice as a separate configuration.
Changing `--robot` during evaluation does not convert one body's checkpoint into
another body's policy.

## 4. Evaluate the checkpoint on fixed initial states

```bash
export CHECKPOINT="$RUN_ROOT/train/checkpoint_step_006000.pt"

python -m examples.baselines.act.eval_act_native_video \
  --checkpoint "$CHECKPOINT" \
  --hf-cache-dir "$HF_HOME/hub" \
  --sub-config "$RUN_ROOT/config/datasets.json" \
  --human-manifest "$RUN_ROOT/config/human-manifest.json" \
  --seeds-file "$RUN_ROOT/config/seeds.json" --seed-split evaluation \
  --max-episode-steps 200 --fps 20 --execute-steps 8 \
  --save-video --output-dir "$RUN_ROOT/evaluation"
```

The entry reloads the exact native normalizer, camera and action order and
strictly reloads model weights. The recorded robot pair is reconstructed by
default. `--robot arx_x5` may be supplied explicitly and must agree with the
recording. The policy observes public robot RGB/qpos and predicts native joint
targets. Evaluation does not call expert planning, replay expert actions or
restore intermediate world states.

The L0 pill task keeps its 200-step budget, 20 Hz control and 100 Hz physics.
A prediction contains sixteen actions; eight are executed before the next policy query.
Official success/timeout and sampled physical facts are reported independently.
Physical observations do not change actions, rewards, success or termination.
The shared cube task, if separately collected and paired with an appropriate
human demonstration, retains its distinct 400-step task budget.

Output includes `act_native_video_*.json`, `summary_*.json`, and recorded policy
rollouts/videos. Each episode identifies its environment seed, checkpoint,
human video SHA/frame indices, effective demonstration ID and conditioning
vector hash/norm. Review final success, success at any point, failure stage and
failure videos together. A low training loss alone does not establish task
success.

Checkpoints contain the resolved manifest from the original machine. On another
server, `--human-manifest` above supplies the relocated video paths and verifies
their content again. `--hf-cache-dir` relocates the pretrained model cache saved
in the model arguments. Preserve the actual video content and sampling when
claiming the same evaluation configuration.

## 5. Check that a changed demonstration reaches the policy

The same task's development demonstration can be selected explicitly while
holding the checkpoint and environment seeds fixed:

```bash
python -m examples.baselines.act.eval_act_native_video \
  --checkpoint "$CHECKPOINT" \
  --hf-cache-dir "$HF_HOME/hub" \
  --sub-config "$RUN_ROOT/config/datasets.json" \
  --human-manifest "$RUN_ROOT/config/human-manifest.json" \
  --human-role development --human-demo-id human_H31_ep4_development \
  --seeds-file "$RUN_ROOT/config/seeds.json" --seed-split evaluation \
  --max-episode-steps 200 --fps 20 --execute-steps 8 \
  --save-video --output-dir "$RUN_ROOT/replacement-diagnostic"
```

Compare the effective demonstration IDs and `conditioning.task_z_sha256`, as
well as the predicted behavior. A different clip from the same task checks
conditioning plumbing; it is not a wrong-task semantic test. An intentionally
different task video requires its own manifest and explicit `--human-demo-id`;
the report marks the task mismatch. Such replacement diagnostics do not replace
the ordinary evaluation results.

Single-task success can occur even if the policy largely ignores the human
condition. A changed latent or changed actions demonstrates that an intervention
reached the policy; demonstrating task understanding needs discriminable tasks
or goals, controlled scenes and a separate experimental protocol. Five fixed
initial states are an engineering sample, not a broad success-rate estimate.

CPU interface checks can be run without a simulator GPU:

```bash
python -m pytest -q tests/baselines/test_native_video_data.py \
  tests/baselines/test_native_video_model.py \
  tests/baselines/test_native_video_eval.py \
  tests/baselines/test_policy_eval_boundary.py
```

## 6. Inspect action limits during a fixed regression

`--action-diagnostics` is an optional, read-only measurement in the shared
native evaluator. It records the selected normalized prediction, denormalized
requested action, and existing bounded action. It changes no action, reward,
termination or success decision and defaults to off.

Each episode adds `action_limit_diagnostics` to its result JSON, including
per-coordinate lower/upper exceedance counts and conditional magnitude
statistics. An NPZ beside the rollout contains the three `(steps, action_dim)`
arrays and actual controller bounds. Magnitudes keep each coordinate's unit:
arm targets are radians, while normalized gripper commands are dimensionless.
The dataset's gripper physical-coordinate unit does not turn a normalized
gripper command into metres.

The following reproduces the twenty seeds reserved for the 2026-11-19
engineering regression. For a new independent batch, freeze another allocation
after checking the retained history.

```bash
python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["RUN_ROOT"]) / "config/fresh-seeds.json"
with path.open("x", encoding="utf-8") as stream:
    json.dump({"evaluation": list(range(18000, 18020))}, stream, indent=2)
    stream.write("\n")
PY

python -m examples.baselines.act.eval_act_native_video \
  --checkpoint "$CHECKPOINT" --hf-cache-dir "$HF_HOME/hub" \
  --sub-config "$RUN_ROOT/config/datasets.json" \
  --human-manifest "$RUN_ROOT/config/human-manifest.json" \
  --human-role evaluation \
  --seeds-file "$RUN_ROOT/config/fresh-seeds.json" --seed-split evaluation \
  --max-episode-steps 200 --fps 20 --execute-steps 8 \
  --action-diagnostics --save-video \
  --output-dir "$RUN_ROOT/fresh-seed-regression"
```

Retain every allocated seed, including a completed task failure or an interface
error. A clipping step means at least one coordinate crossed a controller bound;
its count alone does not measure severity. Review the affected coordinates and
magnitudes before considering any training or execution change.

## 7. Train one model on two tasks

The completed Piper pilot and matched zero-condition run are documented in
[the two-task validation report](native_video_act_multitask_validation.md).
The subsequent matched runs use `--max-updates 12000` with new output directories
and the same 3,000-update snapshot interval. Their development comparison selects
the 9,000-update checkpoints. Both now complete twenty fresh robot seeds per
task with the same development human clips: pill 18/20, baskets 20/20, macro 95%,
without interface errors. This independent scene regression keeps eight-step
execution and official scoring; it does not test new human clips. See the
report for all outcomes, failures and exact checkpoint identities.
The subsequent paired failure diagnosis reads the existing trajectories and
reconstructs saved-observation chunks without new rollouts. It distinguishes
unstable approach contact from commanded dwell. A subsequent twelve-episode
development comparison executes sixteen predicted actions before replanning:
all succeed, including the
previous constant pill7001 failure. A subsequent frozen comparison on new robot
seeds 25000–25019 adds 160 episodes across both models, tasks and windows.
Video achieves 20/20 per task under either window; constant pill improves
18/20→20/20 and constant basket stays 20/20. No final success regresses, and
all paired initial states, first eight actions and action records pass audit.
For these selected 9k pilot models, prefer `--execute-steps 16` in further
development. Constant basket's median first success is six steps later, so
sixteen is not uniformly faster. Generic defaults, the historical commands
above and the earlier 80-episode record retain eight-action execution. No weights
are retrained. See the report for paired timings, retained failures and limits.

Piper supports two original L0 tasks through the shared expert and learning
entries:

| Robot dataset task | Human task | Native qpos / action widths |
| --- | --- | --- |
| `TwoRobotPickPillToRegions-v1` | `human_H31` | 16 / 14 |
| `TwoRobotPickAppleBananaToBaskets-v1` | `human_H8` | 16 / 14 |

Prepare a converted dataset for each task, including explicit train/development
episodes. The dataset configuration is a list of the same records used in
section 2, with a distinct `repo_id`, `root` and correct `env_id` for each task.
Human manifest version 2 has `schema_version: 2` and a `tasks` list. Each task
record has `task_id` and `demos`; each demonstration uses the same fields, roles
and physical MP4 episode bounds as version 1. Demo IDs are unique across tasks.

The wrapper selects videos from each robot sample's actual dataset/task, even
after shuffling. Within that task it cycles through the training clips. The
checkpoint stores every dataset's native physical contract, source image size
and training-only normalization statistics. The evaluator restores the entry
matching each configuration's `repo_id`. In particular, the two Piper tasks
have different robot mounting poses; the first task's profile cannot represent
both. Task IDs route data, configuration and statistics, without becoming an
extra policy input. Shared-policy training currently requires matching native
embodiment, joint ordering, state/action widths, camera shapes and control FPS.

For legacy converted data lacking profiles, a dataset record can include
`recorded_profiles` with the original collection's `robot_profile`,
`task_profile` and `provenance` (source file/hash and field mapping). The trainer
copies these into the checkpoint while preserving the original native metadata;
it rejects a supplement that conflicts with an already recorded profile. Use
actual saved collection settings for this supplement.

Set `MULTITASK_ROOT` to the prepared manifest/configuration directory, and
`DATA_ROOT` to the parent of the converted datasets. A dataset record may also
use an absolute `root`. Use the same model JSON encoder settings as section 2.
Run the video model and its zero-condition ablation with matching budgets:

```bash
for mode in video constant; do
  python -m examples.baselines.act.train_act_native_video \
    --sub-config "$MULTITASK_ROOT/datasets.json" --sim-root "$DATA_ROOT" \
    --human-manifest "$MULTITASK_ROOT/human-manifest.json" \
    --model-config "$MULTITASK_ROOT/model-config.json" \
    --conditioning-mode "$mode" --sampling-policy equal_task \
    --cameras zed2i --state-type qpos --state-dim 16 --action-dim 14 \
    --obs-horizon 1 --pred-horizon 16 \
    --normalization-method bounds_q99 --normalization-min-range 0 \
    --batch-size 16 --num-workers 2 --max-updates 6000 \
    --save-freq-updates 3000 --seed 1 \
    --output-dir "$MULTITASK_ROOT/training-$mode"
done
```

`equal_task` samples windows with replacement and equal expected task mass,
independent of trajectory length. Its dedicated seeded sampler, sampling policy
and per-dataset statistics are saved. The default `uniform_window` samples
without replacement, retaining the single-task workflow's sampling semantics.
The `constant` ablation uses the same visual ACT architecture and robot data,
replacing the final human condition with a zero vector. Human pixels and clip
identity then have no effect on policy predictions. It is an ablation of this
video policy, rather than the separately implemented single-task ACT model.

With a seeds JSON containing the frozen `evaluation` allocation, evaluate both
final checkpoints through the same task list and original 200-step budget:

```bash
for mode in video constant; do
  python -m examples.baselines.act.eval_act_native_video \
    --checkpoint "$MULTITASK_ROOT/training-$mode/checkpoint_step_006000.pt" \
    --sub-config "$MULTITASK_ROOT/datasets.json" --hf-cache-dir "$HF_HOME/hub" \
    --human-manifest "$MULTITASK_ROOT/human-manifest.json" \
    --seeds-file "$MULTITASK_ROOT/evaluation-seeds.json" \
    --max-episode-steps 200 --fps 20 --execute-steps 8 \
    --save-video --action-diagnostics \
    --output-dir "$MULTITASK_ROOT/evaluation-$mode"
done
```

The ordinary evaluation chooses that task's evaluation demonstration. Reports
include per-task and macro-average final official success. Retain every allocated
attempt and report interface failures separately from completed policy failures.
Six thousand updates is a fixed engineering budget, not a convergence claim.
The two scenes expose task identity through the robot observation; comparing
the two conditioning modes measures this pilot's outcome but does not by itself
establish semantic goal-following from video. That question needs the same valid
robot initial state with multiple executable demonstrated goals.
