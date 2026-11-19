# Combined development branch

`integration/current-features` combines the fork's published synchronous execution,
cuRobo backend, multi-embodiment workflows, and three standalone dataset fixes.
It is an integration target for development and regression. The feature/fix
branches remain separate for upstream review, and `main` is unchanged.

Subsequent native human-video ACT work is on `feat/native-video-act`, based on
this integration target. It has not yet been merged into
`integration/current-features`. The status and priorities below cover both.

## Included sources

| Source branch | Frozen head |
| --- | --- |
| `contrib/dual-arm-sync` | `e0e6a842aa601801f896c353a66af25bb643fde6` |
| `feat/curobo-backend` | `1a516903a7862c4ed58a7aadb15779b16586709a` |
| `feat/multi-embodiment` | `a2b5db93058287833c45a96e5f00ef9da70523f7` |
| `fix/image-statistics` | `fd824b0c1dca8fad23011cff10541d8f8ef6caa8` |
| `fix/video-frame-alignment` | `ca1c245ff43d515db653cb95c4b279a9a32cf644` |
| `fix/act-dataset-fps` | `88896fe8cf46cc4dd335f3cdf7ac006d50cf09c2` |

Local joint-14 planning and planner-stress experiments are separate prototypes;
their existence does not make them supported routes in this branch.

## Supported combinations

| Task | Robot pair | Planner |
| --- | --- | --- |
| `TwoRobotPickPillToRegions-v1` | Panda wristcam | MPLib or cuRobo 0.9.0 |
| `TwoRobotPickPillToRegions-v1` | Piper, ARX X5, UR5+WSG, xArm6+Robotiq | MPLib |
| `TwoRobotPickAppleBananaToBaskets-v1` | Piper | MPLib |
| `DualArmPickPlace-v1` shared example | Five supported robot pairs | MPLib |

The original serial route remains the CLI default. Synchronous original tasks
use one CPU physics environment, 20 Hz absolute joint-position control,
100 Hz physics, and the original 200-step horizon. The separate shared
pick/place example retains its documented 400-step horizon.
cuRobo planning requires an NVIDIA GPU; action replay does not require cuRobo.
Unsupported task/robot/backend combinations are rejected before environment or
worker creation.

```bash
# Original-task synchronous generation (replace robot as needed).
python -m mani_skill.examples.motionplanning.dual.two_robot_run \
  -e TwoRobotPickPillToRegions-v1 --execution-mode sync --robot piper \
  --seed 6000 -n 1 -b cpu -o rgbd --save-video \
  --record-dir demos --traj-name piper-pill-6000

# Panda cuRobo: a continuous collection process reuses its planner session.
python -m mani_skill.examples.motionplanning.dual.two_robot_run \
  -e TwoRobotPickPillToRegions-v1 --execution-mode sync \
  --planner-backend curobo --curobo-python /path/to/curobo/bin/python \
  --seed 541 -n 2 -b cpu -o rgbd --save-video \
  --record-dir demos --traj-name curobo-pill-541-542

python -m mani_skill.examples.motionplanning.dual.sync.replay \
  --traj-path demos/TwoRobotPickPillToRegions-v1/motionplanning/piper-pill-6000.h5
```

See [generation and conversion](dual-arm-sync.md),
[multi-embodiment setup and native ACT](multi-embodiment.md), and
[cuRobo installation and reuse](curobo-backend.md) for the full workflows.

## Integration decisions

- Shared execution retains native joint counts, physical left/right identities,
  named finger and mounting permissions, controller units, action conversion,
  and complete recording metadata. Contact decoding and fixed support
  fingerprints are shared once.
- Initial pill support uses convex-geometry acceptance version
  `pill_initial_tabletop_geometry_v2`. Actor-origin motion remains diagnostic.
  Both previous diagnostic field spellings are emitted for existing consumers;
  penetration, depth-growth and finite-table limits are unchanged.
- Panda pill generation uses the cuRobo branch's measured quiet preparation
  (at most ten control steps) and one-way release-contact separation with
  either planner. Other robot/task combinations retain ten native open steps
  and their existing retreat behavior. Each new episode records these choices
  in `dual_sync.execution_features` and the actual preparation step count.
  This intentionally updates the older Panda MPLib execution behavior.
- Replay reads the recorded preparation length, falling back to ten only for
  old archives. New explicit release-separation metadata takes precedence;
  old Panda archives with recorded release-separation physics retain that rule
  even when their backend was MPLib, with the old cuRobo backend as an additional
  fallback. Old sync/multi archives without it do not acquire an additional gate.
- Single-task ACT uses the multi-embodiment seeded native CPU rollout, with
  saved camera/statistics/state/action contracts. State projection, padding
  masks, exact video frame access, floating-point image statistics and recorded
  dataset FPS fixes are preserved. Augmentation defaults to off and can be
  enabled explicitly. The native human-video ACT entry builds on this rollout
  and adds the official frozen video encoder and trainable task adapter; see
  [the native video-conditioned workflow](native_video_act.md).
- Official success, physical events, expert execution/quality and recording
  facts remain separate. Optional policy-evaluation observation is read-only.

## Replay-state repair and current validation

Replay initialization was repaired in `c17c461`: when the seeded reset already
matches every recorded S0 value, replay preserves it instead of writing the same
state through the simulator again. An actual mismatch retains the previous
one-time restoration. The redundant write was the cause of the cuRobo541
generation/replay discrepancy below. Actions and acceptance thresholds did not
change, and the applicable CPU suite now passes **528 tests**.

All eight existing integration episodes now pass production replay with every
action and T+1 state array exactly equal to generation. A predeclared new batch
attempted seeds **6100/6101 once per supported original-task route**, fourteen
episodes in total. All passed official success and execution; thirteen passed
quality and production replay with exact state arrays. xArm6101 already fails
generation quality (9/20 stable steps at action 199). Its normal replay rejects
failed phase metadata before execution; a separate diagnostic reproduces all
199 actions/200 states exactly and leaves its quality failure intact.

The known Piper450 collision negative remains rejected. The early shared Piper
example passes functional replay but crosses a known physical-model change and
is excluded from same-version exact-state claims. Detailed evidence, limits and
fixed batch results are in [the replay repair report](replay-state-repair.md)
and [its machine-readable record](replay-state-validation.json).

## Initial integration validation (before replay-state repair)

The combined runtime was exercised at `cc60d7f`; the legacy MPLib archive
compatibility fix is `6d0e8b7`. At that point the applicable CPU suite passed
**511 tests**.

| Fresh generation | Seed | Control actions | Official / execution / quality | Action replay |
| --- | --- | ---: | --- | --- |
| Panda pill, MPLib | 6000 | 162 | pass / pass / pass | pass |
| Piper pill, MPLib | 6000 | 193 | pass / pass / pass | pass |
| ARX X5 pill, MPLib | 6000 | 194 | pass / pass / pass | pass |
| UR5+WSG pill, MPLib | 6000 | 184 | pass / pass / pass | pass |
| xArm6+Robotiq pill, MPLib | 9300 | 190 | pass / pass / pass | pass |
| Piper apple/banana baskets, MPLib | 450 | 191 | pass / pass / pass | pass |
| Panda pill, shared cuRobo session | 541 | 171 | pass / pass / pass | official/execution pass; quality fails |
| Panda pill, same cuRobo session | 542 | 183 | pass / pass / pass | pass |

**Historical discrepancy, resolved above:** the new cuRobo541 recording passes
generation's twenty stable placement observations. Replaying its exact 171 recorded actions reaches
official success with no forbidden contacts, but counts 18 final stable steps,
so the unchanged quality rule rejects it. This pre-fix result is retained.
A control with the original
cuRobo branch `1a51690` produces the same 18-step result and identical execution,
physics and placement reports. That control only converts the new metadata's
`robot_uids` list to the same Panda tuple at environment construction; files,
actions and thresholds are unchanged. The initial control without this metadata
compatibility stopped before physics and is retained separately. No action was
appended and no stability or collision threshold was changed.

Four historical archives (sync Panda542, Piper6000, cuRobo541 and the old cuRobo
branch's MPLib542 route) also replay successfully. The last archive's recorded
release-separation facts are preserved by the compatibility fix. The known
Piper450 negative still produces the same forbidden robot contacts at physics
substeps 749/750; its complete diagnostic result equals the historical result.

A fresh Piper recording converts to 193 real 20 Hz rows with 16 state and 14 action
coordinates. Numeric rows and timestamps agree with the source H5; RGB/depth
frames are actually decoded on CPU, and all exported statistics are finite
with nonnegative standard deviations. This one-episode product is a format
smoke check, not a training dataset; use an explicit episode manifest for
training splits.

These are fixed development cases and interface checks, not benchmark success
rates, complete task coverage, a new policy training run or a speed comparison.
See [the compact validation record](integration-validation.json) for source
identities and machine-readable results.

## TODO status as of 2026-11-19

- [x] Shared synchronous execution, concentrated robot/task configuration,
  portable runtime, and distinct planning, execution, expert-quality and policy
  evaluation responsibilities for the supported combinations above.
- [x] Five robot pairs complete the original pill generation, replay, conversion,
  separate single-task ACT training and native rollout workflow. Those earlier
  ACT models have no human video condition.
- [x] Repair generation/replay initialization: all eight existing integration
  archives replay with exact actions and T+1 state arrays. The new fourteen-case
  batch retains one generation-quality failure, xArm6101.
- [x] Complete the ARX X5 human-video ACT L0 pilot: real human_H31 clips,
  6,000 training updates, strict reload, 3/3 development and 5/5 initial evaluation
  success. A subsequent frozen batch of twenty new local-history seeds achieves
  20/20 final success with the same checkpoint.
- [x] Diagnose action clipping without changing scoring or execution. Across
  4,000 new rollout steps the twelve arm coordinates never exceed bounds;
  gripper excess corresponds to at most 0.247/0.511 mm of per-finger target
  correction. Denormalization and executed H5 actions agree exactly. The
  applicable CPU suite at that point passed 582 tests. Details and limits are in
  [the validation report](native_video_act_validation.md).
- [x] Complete the Piper two-task pilot: each task contributes twenty training
  and three development trajectories, paired with human_H31/H8. One shared
  video model and a matched zero-condition model each complete 6,000 updates.
  Across forty fixed new-seed episodes, final success is pill 2/10 versus 0/10,
  baskets 10/10 for both (macro 60% versus 50%), with zero interface errors.
  Initial states, initial policy weights/buffers outside `task_encoder.*`, actual
  sampling, task routing and normalizers match across conditions. Offline
  condition swapping changes video-model output and preserves constant-model
  output. The latest applicable CPU suite
  passes 603 tests. See [the two-task report](native_video_act_multitask_validation.md).
- [x] Compare existing 3,000/6,000-update checkpoints on both tasks' reserved
  development seeds, using development human clips: twenty-four episodes, zero
  interface errors, all four initial-state groups equal. Pill final success
  stays 0/3, while video grasp/transport stages and offline inference error
  improve. A separate six-case 8-to-16 execution-window intervention lets the
  constant policy start and place right objects, but both pill models remain
  0/3. The default stays eight steps. Start-state stagnation and left grasp loss
  remain distinct development targets; see the report's development diagnosis.
- [x] Complete matched 12,000-update training and the preallocated 9k/12k
  development curve. All four new/old 3k/6k training prefixes match bitwise,
  including full model and optimizer state. At 9k video achieves pill/baskets
  3/3 and 3/3, constant 2/3 and 3/3; at 12k they reach 2/3 and 3/3, versus
  2/3 and 1/3. All twenty-four new development episodes complete without
  interface errors. Both 9k checkpoints are selected as candidates for an
  independent fresh-seed comparison, with all weights and failures retained.
- [x] Run both selected 9k checkpoints once on twenty new robot seeds per task:
  eighty episodes and 16,000 control steps. Both reach pill 18/20 and baskets
  20/20 (macro 95%), with zero interface errors and all forty paired initial
  physical/public states bitwise equal. Human clips stay development episode 4;
  this is new-scene regression, not new-video generalization. Video's two pill
  failures lose the left grasp; constant's two failures retain the objects
  outside their regions. All allocations remain recorded, without retries.
- [x] Diagnose the four pill failures against the other model's same-seed
  success using eight saved trajectories. Video approach disturbs the left
  bottle before stable lifted possession; constant commands dwell after lift.
  Sixteen offline predictions exactly reproduce the archived first eight
  actions and expose larger changes in the unexecuted later half at selected
  holding observations. No controller/interface error is confirmed; runtime and
  all original scores remain fixed. See the report's paired diagnosis.
- [x] Complete the 9k execute8/16 development comparison, reusing twelve old
  references and adding twelve execute16 episodes. All new cases succeed;
  constant pill improves 2/3 to 3/3 and the other groups remain 3/3. The prior
  constant7001 failure succeeds at step143 with earlier lift and transport.
  Initial states and first eight actions match in all twelve pairs. No prior
  success regresses; execute16 remains a development candidate rather than a
  replacement for the recorded execute8 scene-regression result.
- [x] Extract and push four standalone fixes to the fork:
  `fix/image-statistics`, `fix/video-frame-alignment`, `fix/act-dataset-fps` and
  `fix/act-single-interfaces`. The ACT interface fix is based directly on the
  official source; preparing that branch is complete, upstream submission is
  separate.

## Next implementation priorities

1. **P0: improve and validate the two-task learning result.** The Piper pilot
   now connects pill and apple/banana baskets to one model using human_H31/H8.

   - [x] Freeze splits and collect/convert basket data. The combined robot set
     has forty training episodes (7,673 frames) and six development episodes
     (1,151 frames); collection failures and all replays are retained.
   - [x] Implement task-paired video manifests, equal-task sampling, saved native
     profiles and per-task training-only normalizers, with strict shared-model
     reload. Task IDs route configuration and reports without extra policy input.
   - [x] Complete fixed-budget training and new-seed evaluation for the video
     model and matched zero-condition baseline; retain all forty outcomes.
   - [x] Verify offline video-condition sensitivity and A-to-B-to-A cache
     restoration. This intervention does not establish goal semantics.
   - [x] Diagnose fixed development pill cases, compare existing checkpoints and
     test the executed query window separately. Confirm static-prefix behavior,
     left grasp loss and continuing inference-error improvement, without finding
     an interface error requiring a data/model repair.
   - [x] Complete the 12,000-update budget with unchanged data/model/execution
     and checks at 9,000/12,000. Fresh seed1 runs reproduce the old 6k prefixes
     exactly. The two 9k candidates score better than 12k on this development
     set even though training loss keeps falling.
   - [x] Freeze seeds 24000–24019 and compare the selected 9k video/constant
     candidates once with unchanged clips, scoring and eight-step execution.
     Both achieve 18/20 pill and 20/20 baskets. Pill paired outcomes are sixteen
     shared successes, two video-only and two constant-only successes. Keep
     both candidates, every failure and the original forty final outcomes.
     The new batch shows no aggregate video success gain; it must not be reused
     as fresh test evidence after tuning.
   - [x] Compare all four pill failures with their same-seed successes, check
     the controller contract, and reconstruct full chunks at saved holding
     observations. The diagnosis uses no new simulator episode or training.
   - [x] Test the execution-window interaction on development cases: reuse
     the existing 9k execute8 records and add twelve execute16 episodes with
     unchanged clips, horizon and scoring. All new cases succeed; constant7001
     advances through lift and transport instead of timing out. Seven previous
     successes occur earlier, three later by one or two steps, one unchanged.
   - [ ] Freeze execute16 as a development candidate and compare execute8/16
     on a new scene-seed allocation, keeping both9k weights and human identity
     fixed. Preserve the earlier80 outcomes. Do not claim the unrerun video
     approach failures are fixed; use separate development evidence if approach
     robustness needs changes.
   - [ ] Test demonstration following with the same valid initial scene and
     multiple executable goals. Different task scenes and cross-task video swaps
     do not establish a reliable video benefit or semantic necessity.

2. **P1: maintain supported routes and investigate xArm6101.** Its placement
   reaches only 9/20 stable observations before the original action budget ends,
   and the generation trace replays exactly. Determine the cause before changing
   task procedures or quality settings. Broader regression or a speed comparison
   should use a bounded, separately frozen batch; compare MPLib/cuRobo with the
   same source, task and initial state, reporting cold/continuous costs and
   qualified trajectories per hour separately.
3. **P2: integrate and prepare the fork's deliverable.** Merge the validated
   native-video workflow into the integration target, verify startup from an
   independent checkout, and update supported commands and the capability
   matrix. Retain the separate feature/fix branches for review. Official upstream
   submission follows the user's explicit instruction.
4. **Then expand tasks and L0–L3 pairing before adding another capability.**
   Choose joint planning when a task requires both spatial paths to change, or
   audit one arm/hand for a rigid grasp/carry/replay loop. Historical handover
   and recovery prototypes have bounded engineering evidence but are not part
   of the current clean supported routes. USD/Real2Sim, closed-chain and
   soft-body work remain later work packages.

The ARX result establishes a single-task learning loop; the Piper pilot adds a
shared two-task model and a matched condition ablation. Reliable demonstration
following, full L0–L3 coverage, cross-task transfer and a complete official
benchmark result remain open.
