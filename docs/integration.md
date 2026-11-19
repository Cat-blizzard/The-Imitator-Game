# Combined development branch

`integration/current-features` combines the fork's published synchronous execution,
cuRobo backend, multi-embodiment workflows, and three standalone dataset fixes.
It is an integration target for development and regression. The feature/fix
branches remain separate for upstream review, and `main` is unchanged.

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
  [the ARX video-conditioned workflow](native_video_act.md).
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

## Next implementation priorities

The older A–I implementation plan contains dated status notes. The following
priorities distinguish implemented interfaces from remaining benchmark work.

1. **Use the repaired baseline to characterize remaining task quality failures.**
   The fixed new-seed batch retains xArm6101's placement quality failure; it is
   already present in generation and its state trace reproduces exactly.
   Investigate its physical cause before changing the task procedure or quality
   settings. Broader coverage and matched timing experiments remain necessary
   before general success-rate or cuRobo speed claims.
2. **Extend the native video-conditioned pilot to benchmark coverage.** The
   `feat/native-video-act` branch connects human_H31 videos to an ARX X5 policy
   through native robot data, training, strict checkpoint reload and L0 policy
   rollouts. Its fixed protocol and measured results are in the
   [validation report](native_video_act_validation.md). Full L0–L3 pairing,
   multiple tasks and cross-task evaluation remain separate work. The earlier
   five-body single-task ACT checks have no human video condition.
3. **Review the minimal ACT interface fix separately.**
   `fix/act-single-interfaces` has been extracted from the current official base
   and pushed to the fork. It contains constructor/import, condition projection
   and evaluation entry fixes. It has not been submitted upstream; training
   choices and this broader learning workflow remain separate.
4. **Expand task coverage before increasing robot count.** Add a representative
   supported task or bounded new-seed set through the same entry, recording,
   replay and conversion paths. Backend comparisons require the same task,
   source version, policy and initialization; shared executor support alone
   does not make every planner/robot combination available.
5. **Choose one further capability after the baseline is stable.** For dexterous
   hands, first audit one arm/hand's assets and controls and close a simple rigid
   grasp/carry/replay loop. For joint planning, use a task where changing both
   spatial paths is necessary and build on the separate joint-14 prototype.
   Handover, closed-chain manipulation, automatic recovery, USD/Real2Sim and
   soft-body tasks each need additional execution/data contracts.

These priorities do not represent completed policy training or additional
supported planner/task/robot combinations.
