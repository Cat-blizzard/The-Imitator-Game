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
  enabled explicitly. The human-video ACT evaluator remains separate.
- Official success, physical events, expert execution/quality and recording
  facts remain separate. Optional policy-evaluation observation is read-only.

## Validation

Validation results for this combined tree are recorded below after execution.
Historical results from the source branches do not certify the merged tree.

## Next implementation priorities

The older A–I implementation plan contains dated status notes. The following
priorities distinguish implemented interfaces from remaining benchmark work.

1. **Finish a releasable integrated baseline.** Freeze dependencies and source,
   retain known positive/negative inputs, and run a small fixed cross-seed
   regression. Report official success, quality pass, failure categories, cold
   startup and continuous throughput separately. Existing development seeds
   do not establish general success rates or a cuRobo speed advantage.
2. **Connect a human demonstration to a new embodiment.** Select one supported
   task and one added robot; fix human demo IDs, L0–L3 pairing, robot-data
   provenance and train/dev/test roles. Run an existing video-conditioned
   baseline through training and policy evaluation. The current five-body
   single-task ACT checks use robot images and joint state and do not establish
   this complete TIG imitation setting.
3. **Prepare the minimal ACT interface fix for upstream review.** Extract
   constructor/import, condition projection and evaluation entry fixes from the
   broader learning workflow. Keep training choices and unrelated compatibility
   changes reviewable separately. This can proceed alongside priority 2.
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

The current task creates the combined branch and assesses these priorities;
it does not start new policy training or implement the remaining capabilities.
