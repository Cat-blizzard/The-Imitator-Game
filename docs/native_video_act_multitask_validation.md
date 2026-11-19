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

The subsequent [12,000-update experiment](#completed-12000-update-budget)
completes the planned development curve. Its selected 9,000-update checkpoints
now complete an [independent scene-seed regression](#independent-9k-scene-seed-regression):
both achieve pill **18/20**, baskets **20/20**, macro **95%**, with zero interface
errors. This checks new scenes with the same development human clips. The table
above remains the original 6,000-update final batch; its human clips and robot
seeds differ, so its scores are not a controlled training-budget comparison.

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

The forty reported episodes are now consumed evaluation cases. The development
comparison below uses the separately reserved development robot/human splits.
A demonstration-following experiment additionally needs multiple valid goals
under the same initial scene. Full task and L0–L3 coverage remain open.

## Development diagnosis and checkpoint comparison

The follow-up uses source `e4bd5212c577b751499d6e36443e2354b4dac766`, whose
runtime is unchanged from `376c5ec` (the intervening commit changes documentation).
Both conditions' existing 3,000/6,000-update checkpoints run on pill seeds
7000–7002 and basket seeds 22000–22002, with human development episode 4.
All **24 allocated episodes / 4,800 steps** complete, with zero interface errors.
For each task/seed, all four initial physical states and initial public RGB/qpos
match bitwise (six scene/seed cases, eighteen comparisons). The prediction horizon,
normalizer, 200-step budget and eight executed actions remain unchanged.

| Condition | Updates | Pill development final success | Basket development final success |
| --- | ---: | ---: | ---: |
| Human video | 3,000 | 0/3 | 3/3 |
| Human video | 6,000 | 0/3 | 3/3 |
| Constant zero condition | 3,000 | 0/3 | 2/3 |
| Constant zero condition | 6,000 | 0/3 | 3/3 |

Video pill behavior progresses despite unchanged final success: at 3,000
updates neither arm grasps in any development episode. At 6,000, seed7001
grasps with both arms and places the right object; seed7002 briefly grasps with
the left arm, while seed7000 never satisfies either grasp predicate. The left
object loses its grasp predicate outside the region at step134 in seed7001 and
steps106/110 in seed7002, while its gripper command still requests closure.
These measurements distinguish a transient grasp from sustained possession;
they do not identify the mechanical cause of contact loss.

The previous consumed-test records show the same distinction: six of the eight
video pill failures lose a left grasp outside the region, and two never observe
a right grasp. The constant-condition test policies remain near their initial
poses, moving each TCP by only about 10–14 mm. Thus a uniform shortage of the
200-step budget does not explain all failures. The actual 23 source expert
episodes reach official success at action steps138–141. Their H5 actions/qpos
were checked against the converted rows when establishing this timing.

### Starting motion and the executed query window

Both tasks' forty training trajectories begin with ten identical rest actions
and eleven identical qpos observations. Pill does not have an unusually large
static-data share: repeated adjacent actions are 30.47% in pill and 31.75% in
baskets; future padding is 3.879% and 3.941%, correctly excluded from L1.
The selected training statistics, padding and routing audit finds no interface
error requiring a retraining fix.

Offline inference on the six real development trajectories uses saved public
RGB/qpos and the actual zero-CVAE-latent `get_action`; future expert actions are
comparison targets only. Across pill initial observations, video6000 predicts
maximum arm deviations from rest of **0.0222–0.0230 rad in queries1–8**, versus
**0.2148–0.2215 rad in queries9–16**. Constant6000 shows the same tendency
(about 0.0284 versus 0.165 rad). Repeatedly executing the first half can therefore
omit much of the predicted starting motion. Similar observations during the
waiting prefix produce similar chunks; the single-frame policy has no elapsed
time input. Baskets also show this pattern and can succeed, so the prefix alone
does not establish the cause of failure.

A subsequent, separately frozen development intervention changes only
`execute_steps` from 8 to 16 for both 6,000-update pill models on the same three
seeds. All **six episodes / 1,200 steps** complete, retaining the 200-step limit.
Both final scores remain **0/3**. The constant model now observes right grasps
and completed right placements in all three cases, plus left grasps in two;
with eight executed steps it grasps neither side. Its start-state stagnation is
therefore sensitive to the executed window. Left-object loss or missed grasp
still prevents success. The video model also retains its grasp/placement
failures. This intervention supports a control-schedule contribution to the
constant model's stagnation, without demonstrating that sixteen steps is the
best setting or that the static prefix caused it. The default remains eight;
the forty original evaluation outcomes are unchanged.

Each paired window comparison uses the same initial physical/public state,
checkpoint and human condition; the first eight actual actions are bitwise
equal. With sixteen executed actions, both constant-model TCPs move more than
5 cm from their initial poses by observation16 in all three cases; they never
reach that displacement with eight executed actions. These motion summaries
are descriptive measurements, not additional physical acceptance gates.

Changing the six H31 clips at one fixed development initial observation does
not produce large starting differences: the video6000 chunks differ by at most
0.000554 rad in arm targets. This does not support the hypothesis that the
development clip alone causes the observed initial stagnation. Episode5 in
this descriptive check is already consumed test material and is not used for
checkpoint selection.

### Training-budget decision

Both training loss and actual inference error on real development observations
improve from 3,000 to 6,000 updates. Pill mean absolute arm prediction error
over the first eight queries is:

| Condition | Updates | Starting observations, frames0/5/9/10/15/20 | Later observations, frames80/90/100/120/140 |
| --- | ---: | ---: | ---: |
| Human video | 3,000 | 0.03266 rad | 0.02797 rad |
| Human video | 6,000 | 0.01339 rad | 0.01952 rad |
| Constant zero condition | 3,000 | 0.02620 rad | 0.02051 rad |
| Constant zero condition | 6,000 | 0.02027 rad | 0.01342 rad |

These are comparisons at expert observations, not closed-loop success or
posterior-conditioned training loss. The constant model's low later error also
shows why offline error alone cannot choose the successful controller.

This diagnosis motivated a bounded plan of **12,000 total updates per condition**,
with development checks at 9,000 and 12,000, keeping the robot/human data,
architecture, sampling, loss and default eight-step execution fixed. The budget
doubling tested whether fitting and grasp-stage progress continued; it was not
a convergence estimate. The experiment below executes that plan. Because the
trainer has no exact-resume entry or complete mid-epoch/RNG recovery record,
it uses fresh matched seed1 runs and checks their actual 6,000-update prefixes.

## Completed 12,000-update budget

Both runs complete at source `ac52479fa57aaea90f61190ac6783893b2357022`, with
the same runtime as the pilot. The two original training commands change only
`max_updates=12000` and the output directory. Full saved configuration checks
confirm the data, human clips, native profiles, normalization and model settings.

Each new 3,000/6,000-update checkpoint exactly reproduces its corresponding
pilot checkpoint: all **794 state-dict tensors**, optimizer state/parameter
groups, sampler RNG, actual window sequence and training metrics excluding
elapsed time match bitwise. This includes the frozen human backbone and its
trainable adapter, beyond the narrower initial-policy hash. Thus the previous
6,000-update development results remain the matched baseline for this budget
comparison, without repeating those rollouts.

Both models finish all 12,000 updates and exit zero. Each draws **191,825
windows**, including 95,908 pill and 95,917 basket windows, over 25 sampler
epochs with replacement. Each epoch's final batch has nine rows, so the actual
count is 175 below `12000 * 16`. The two actual sequences and saved sampler RNGs
match an independent CPU reconstruction. Their final sequence SHA256 is
`560be2cd8db2347395a7be1ec0d59c97c5c1946e2675972b6b3ee6f3bf80fb2c`.
All 24,000 training metric rows are finite and contiguous. Process wall times
are 1,644.15 seconds for video and 1,567.67 seconds for constant conditioning,
on separate RTX 5880 Ada GPUs; these are run costs, not a controlled speed test.

### Fixed development results

The preallocated 9k/12k comparison uses the same three robot development seeds
per task and human episode4, with the original 200-step horizon and eight
executed actions. All **24 episodes / 4,800 control steps** complete once, with
zero interface errors. Their initial physical/public states match across all
four groups and the existing 6k development recordings. Actions are finite,
the ordinary clamp reproduces execution bitwise, and execution matches H5.
No arm action clips; maximum gripper excess is 0.034800 in normalized command
units. No policy, quality threshold, task limit or final-test seed is changed.

| Condition | Updates | Pill development final success | Basket development final success | Macro final success |
| --- | ---: | ---: | ---: | ---: |
| Human video | 6,000 baseline | 0/3 | 3/3 | 50% |
| Human video | 9,000 | 3/3 | 3/3 | 100% |
| Human video | 12,000 | 2/3 | 3/3 | 83.3% |
| Constant zero condition | 6,000 baseline | 0/3 | 3/3 | 50% |
| Constant zero condition | 9,000 | 2/3 | 3/3 | 83.3% |
| Constant zero condition | 12,000 | 2/3 | 1/3 | 50% |

The 9k video policy succeeds on pill seeds7000/7001/7002 at steps141/132/149;
both objects are finally placed and ungrasped, with no observed grasp loss
outside the region. The 9k constant policy succeeds on seeds7000/7002; seed7001
grasps both objects but remains holding them at the horizon. Starting stagnation
from the 6k constant policy is absent in all three 9k development cases under
the unchanged eight-step execution rule.

At 12k, video pill7002 loses its left grasp outside the region at step93 and
still holds the right object at the horizon. Constant pill7000 only grasps at
steps150/169, then loses the right grasp outside the region at step193. Its
two failed basket seeds22000/22001 move the TCPs about 0.50 m while leaving
grippers almost fully open; neither arm satisfies the grasp predicate. These
failures are retained rather than rerun or grouped as uniform starting failures.

Training fit continues to improve while these development scores decline:
the last-500-update mean loss decreases from 0.034719 at 9k to 0.029195 at
12k for video, and from 0.034134 to 0.028879 for constant conditioning. This
demonstrates that lower training loss and the latest checkpoint do not reliably
select the better controller in this experiment. Three development seeds and
one training seed do not identify overfitting or establish a stable video gain.

### Candidate checkpoints and next evaluation

The recorded macro-development selection criterion favors **9,000 updates for
both conditions**. Preserve both candidates at the same update budget:

- Video: `budget-12000/training-video/checkpoint_step_009000.pt`, SHA256
  `b8cdd0a08bf4a21b57ba4a8c28bdce52b39901fb1d9b6fbb0d2a223f1d51a390`.
- Constant: `budget-12000/training-constant/checkpoint_step_009000.pt`, SHA256
  `61d9cf3782f7fc405bdb135994b3f72a9cca132ff2975eb4ceb140ba33542eef`.

The local `budget-12000/selected-checkpoints.json` records those identities and
their selection basis. Every 3k/6k/9k/12k checkpoint remains retained. Its
`not_run_for_selected_candidates` field preserves the status at selection;
the separate scene-seed regression below follows that frozen selection.
No new final-test episodes were run within the 12k budget experiment.

## Independent 9k scene-seed regression

At source `61740faebc87ee204f3113f733f17c2a3bf4c3bf`, both previously selected
9k checkpoints execute **seeds 24000–24019 once per task**: 20 seeds, two tasks,
two conditions, **80 episodes / 16,000 control steps**. The retained readable
local history contains no seed/command/filename conflict; deleted records,
remote hosts, denied shell history and untraversed directory symlinks remain
outside that audit's scope. Both actual checkpoint files were rehashed before
launch and match their recorded selection identities.

Only robot reset seeds change from the 9k development evaluation. Human clips
remain development episode 4, H31 for pill and H8 for baskets, and their complete
recorded identities match development. Prediction horizon 16, execution window 8, the 200-step
limit, public RGB/qpos, native profiles, training-only normalizers, ordinary
clamp and official scoring remain fixed. There is no retraining, candidate
reselection, failed-case retry, seed replacement or extra quality rejection.
Diagnostics only read state and do not change actions, rewards or termination.

| Condition | Pill final success (Wilson 95% CI) | Baskets final success (Wilson 95% CI) | Macro final success |
| --- | ---: | ---: | ---: |
| Human video, 9k | 18/20, 90% (69.9–97.2%) | 20/20, 100% (83.9–100%) | 95% |
| Constant zero condition, 9k | 18/20, 90% (69.9–97.2%) | 20/20, 100% (83.9–100%) | 95% |

Each process completes all forty allocations and exits zero, in 260.63 seconds
for video and 263.77 seconds for constant conditioning on separate RTX 5880 Ada
GPUs. These are run costs, not a controlled speed comparison. Final and
once-success counts agree: **38/40 per condition**, with two task timeouts and
zero interface errors each. All failures remain in the denominator. All forty
paired initial physical states and public RGB/qpos arrays match bitwise. All
16,000 executed action rows are finite; recomputed ordinary clamp, recorded
execution and H5 actions match bitwise. Arm action clipping is zero; maximum
gripper excess is 0.045280 in normalized command units. No range is changed.

| Task | Both succeed | Video only | Constant only | Both fail |
| --- | ---: | ---: | ---: | ---: |
| Pill | 16 | 2 | 2 | 0 |
| Baskets | 20 | 0 | 0 | 0 |

The four pill failures show different behavior:

- **Video, seeds 24012/24014:** the left grasp predicate is transient and lost outside
  the target region; the left object remains outside at step 200. The right
  object is placed and ungrasped. Closed-gripper losses occur at steps 83/85
  and 90 respectively; seed 24014 also has an earlier loss at step 82 while its
  command is about +0.595. These records do not establish one mechanical cause.
- **Constant, seed 24003:** both sides grasp at step 81 and retain the objects to 200,
  lifted about 0.13 m near the starting XY position. The final twenty steps move
  the objects only about 5 mm; neither enters its region.
- **Constant, seed 24015:** both sides also retain their grasps from 81 to 200, but move
  the objects about 0.23/0.21 m in the final twenty steps. Both are still held
  outside their regions at the horizon. This case is still progressing late;
  the four failures cannot all be attributed to insufficient time.

Pill success on twenty new scenes is stronger evidence than the previous
three-seed development result, while **two video failures remain**. The equal
success rates and symmetric paired differences show no aggregate video gain in
this batch; they also do not prove that conditioning has no effect. The Wilson
intervals describe scene sampling for each fixed trained model, not training-seed
variation. One training seed, two L0 tasks and already-used development clips
do not establish unseen-video generalization or complete official TIG performance.
The original 6k batch used human episode 5 and seeds 23000–23009, so its score
difference cannot be attributed solely to another 3k training updates.

Keep these weights and results fixed. The next learning experiment should test
different executable demonstration goals in the same valid scene, with a
separate development/test allocation. The four retained failures can support
development diagnosis, but this batch must not become fresh test evidence again
after tuning. Longer training is not justified merely by the remaining failures.

## Retained evidence

Raw evidence stays outside Git under
`$WORKSPACE_ROOT/infrastructure/validation/native-video-act-multitask-20261119/`:
the frozen protocol, source/input hashes, collection attempts, original H5 and
videos, replay and conversion receipts, training logs and both checkpoints,
all forty rollout records, and the paired-state, normalizer, action-recording
and offline-intervention audits. The `development-diagnosis/` subdirectory
retains both development protocols, all thirty development rollouts and videos,
paired-state and motion audits, data/loss statistics, and offline chunk arrays.
The initial offline-loader metadata error and its successful corrected run are
retained separately. The `budget-12000/` subdirectory retains the frozen
12k protocol, both complete training runs and snapshots, 24 development records
and videos, full prefix and final-training audits, candidate identities and
action/initial-state audits. A tuple/list comparison mismatch in the standalone
audit was corrected with its original log retained; it was not a training or
rollout failure. The `fresh-9k-20261119/` subdirectory retains the pre-run
protocol, seed-history and checkpoint audits, all eighty trajectories/videos,
launch receipts, external launcher/auditor scripts and `compact-results.json`.
The compact JSON beside this report preserves
portable identities and per-seed outcomes; datasets and weights are not included
in the fork. Original ARX single-task results remain in
[their separate validation report](native_video_act_validation.md).
