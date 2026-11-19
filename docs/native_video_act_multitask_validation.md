# Piper two-task native video ACT pilot

Recorded on 2026-11-19 with runtime source
`376c5ecce1b6cf4cc8c8ef1ba8dfe449acc9f904`. One shared video-conditioned policy
was trained on two original L0 tasks and compared with a separately trained
zero-condition policy. Both training runs and all forty allocated policy
episodes completed. The applicable CPU regression suite passed **603 tests**.

| Condition | Pill final success | Apple/banana baskets final success | Macro final success |
| --- | ---: | ---: | ---: |
| Human video | 2/10 | 10/10 | 60% |
| Constant zero condition | 0/10 | 10/10 | 50% |

These results establish the two-task data, training and native evaluation
workflow. Pill performance remains limited. One training seed and ten evaluation
seeds per task do not establish a reliable video benefit, convergence or a
complete official TIG result. The scenes also expose task identity, so the
comparison alone does not establish demonstration-goal understanding.

## Data and task routing

| Original environment | Human source | Robot train episodes / frames | Robot development episodes / frames |
| --- | --- | ---: | ---: |
| `TwoRobotPickPillToRegions-v1` | `human_H31` | 20 / 3,867 | 3 / 582 |
| `TwoRobotPickAppleBananaToBaskets-v1` | `human_H8` | 20 / 3,806 | 3 / 569 |
| Total | | 40 / 7,673 | 6 / 1,151 |

Pill reuses the recorded Piper dataset: training seeds 6000–6004 and 6006–6020,
development seeds 7000–7002. New basket training seeds are 21011–21030 and
development seeds 22000–22002. Each task uses human episodes 0–3 for training,
4 for development and 5 for evaluation, with ten sampled zed2i frames per clip.
The split allocation and final 6,000-update checkpoint were fixed before training.

Basket collection retained all startup attempts: ten asset-path failures, one
interrupted worker and one earlier import failure before worker launch. The
23 completed native expert attempts passed official success, execution, quality
and original-action replay. Failed launches were retained without seed reuse or
threshold changes. All 46 converted episodes preserve their source state and
action rows bitwise at 20 Hz; terminal observations are excluded from training.

The version-2 manifest pairs each robot sample with its own task's videos after
sampling. Both Piper tasks have 16 state and 14 action coordinates, but different
mounting poses. Each dataset therefore retains its own native physical profile,
image dimensions and training-only normalizer. A documented profile supplement
for legacy pill metadata comes from the original saved collection configuration;
the original dataset metadata remains unchanged. Independently recomputed
training statistics and actual decoded robot/human batches passed the preflight.
Task IDs select data, configuration and reports; they are not policy inputs.

## Matched training and evaluation

Both runs use seed 1, batch 16, 6,000 optimizer updates, equal-task window
sampling, AdamW at `1e-4`, weight decay `1e-4`, gradient norm clipping at 1 and
bf16. There is no augmentation or learning-rate decay. The shared ACT uses
hidden width 256, four encoder and seven decoder layers, eight attention heads,
feedforward width 2048 and dropout 0.1. Its objective is masked L1 plus ten times
the official KL sum. The pretrained robot ResNet18 is trainable at `1e-5`;
DINOv2 ViT-L14 is frozen and its task adapter is trainable in the video run.
The ablation supplies a constant zero task condition; its human adapter receives
no gradient. Both restore their own fixed final checkpoint, without evaluation
selection. See [the workflow](native_video_act.md#7-train-one-model-on-two-tasks)
for commands and configuration contracts.

The following matched quantities were verified:

- Identical initial policy weights/buffers outside `task_encoder.*`, SHA256:
  `744c85140d8e2ba1ac9e482d7ce9bce0fc82c0c8b27badc2560ecb435f218d4a`.
  This hash excludes the human encoder and its adapter.
- Identical actual window-index sequence SHA256:
  `201111ec16fe87e7ee39948f031f2701aae5db971264739741c5d986056a7a7a`,
  comprising 47,869 pill and 48,047 basket windows.
- Bitwise-equal saved dataset statistics, task/native routing and configuration
  except the condition mode and output directory.
- All twenty paired seeded initial physical states and initial public RGB/qpos
  arrays match bitwise across conditions. Evaluation uses seeded resets, without
  injecting saved world states.

Each condition executes seeds **23000–23009 once per task**, selected after a
local-history audit. Evaluation retains the original 200-step horizon, 20 Hz
control and 100 Hz physics. The policy receives public RGB/qpos and its selected
human condition, predicts sixteen actions and executes eight before replanning.
It uses the recorded native contract and ordinary action clamp. Expert planning,
expert action replay and expert quality rejection do not control policy rollout.
Physical diagnostics are read-only. Final official success is the reported
metric; success at any point happens to give the same counts in this batch.

Both twenty-episode evaluation processes exited successfully, with **zero
interface errors**. All failures and timeouts remain in the denominator. Pill
video successes are seeds 23000 and 23009; all eight other video pill episodes
and all ten constant-condition pill episodes time out. Both basket policies
succeed on all ten seeds. Each final checkpoint and each episode is identified
in [the machine-readable record](native_video_act_multitask_validation.json).

## Failure and condition diagnostics

Read-only stage labels classify the eight video pill failures as three
transport/release-incomplete cases, three placement failures and two cases with
no observed right-arm grasp. In the constant-condition pill run, neither arm is
observed grasping in any episode. These observations identify useful development
targets; they do not prove a specific optimization or representation cause.

Across all **8,000 control steps**, normalized, requested and executed actions
are finite. Executed diagnostic actions match recorded H5 actions bitwise, and
the original clamp applied to requested actions reproduces them bitwise.
All twelve arm coordinates have zero limit exceedances. Clipping only affects
grippers; video maximum excess is 0.062699 in pill and 0.049678 in baskets,
in normalized gripper-command units. No action range or clipping behavior was
changed for this experiment.

An offline intervention holds the saved first public observation of development
pill7000 or basket22000 fixed and swaps the two tasks' evaluation videos. The
video model's mean absolute normalized action-chunk difference is respectively
**0.0204604** and **0.0153848**. Both constant-condition differences are zero,
with bitwise-equal outputs. Switching back restores the original output bitwise
for both models, checking per-task cache restoration. This used no additional
simulator episodes. A cross-task swap measures input sensitivity; it does not
show that the alternate goal is executable in that scene or establish a causal
effect on success.

The next development step is to diagnose pill failures and compare checkpoints
on fixed development cases before setting a larger training budget. The forty
reported episodes are now consumed evaluation cases; further choices should use
development evidence, followed by a separately frozen fresh evaluation batch.
A demonstration-following experiment additionally needs multiple valid goals
under the same initial scene. Full task and L0–L3 coverage remain open.

## Retained evidence

Raw evidence stays outside Git under
`$WORKSPACE_ROOT/infrastructure/validation/native-video-act-multitask-20261119/`:
the frozen protocol, source/input hashes, collection attempts, original H5 and
videos, replay and conversion receipts, training logs and both checkpoints,
all forty rollout records, and the paired-state, normalizer, action-recording
and offline-intervention audits. The compact JSON beside this report preserves
portable identities and per-seed outcomes; datasets and weights are not included
in the fork. Original ARX single-task results remain in
[their separate validation report](native_video_act_validation.md).
