# Synchronous dual-arm generation

`two_robot_run` retains the original serial solutions by default. The optional
`--execution-mode sync` supports these homogeneous robot pairs:

| Original task | Supported sync pair |
| --- | --- |
| `TwoRobotPickPillToRegions-v1` | Panda wrist-camera (default), Piper, ARX X5, UR5+WSG, xArm6+Robotiq |
| `TwoRobotPickAppleBananaToBaskets-v1` | Piper |

The implementation uses one environment, CPU physics and absolute `pd_joint_pos`
commands. Rendering still requires the normal TIG rendering setup.
Other tasks reject explicit sync selection before creating an environment.

## Generate

Use the normal TIG installation and assets, then run from this repository:

```bash
python -m mani_skill.examples.motionplanning.dual.two_robot_run \
  -e TwoRobotPickPillToRegions-v1 --execution-mode sync \
  --seed 401 -n 1 -b cpu -o rgbd --save-video \
  --record-dir demos --traj-name sync-pill-401
```

Omit `--execution-mode sync` to use the original serial solution. Original
solution scripts and the Panda planner are unchanged. The default planning
backend is MPLib; that route does not import cuRobo or start a worker process.
For the optional GPU planner, add `--planner-backend curobo` and select an
interpreter with cuRobo 0.9.0 installed; see [cuRobo backend](curobo-backend.md).
cuRobo selection requires sync mode and supports the same Panda pill task.

Add `--robot piper`, `--robot arx_x5`, `--robot ur5_wsg` or
`--robot xarm6_robotiq` to the command above to use that pair on
the same pill objects and regions. `--robot panda` is an alias for `panda_wristcam` on
this entry. Selecting another robot with the serial mode, or selecting an
unsupported task/robot combination, reports an error before creating a simulator.
Piper uses raised mounts; UR5 uses a task-specific right-base orientation to
avoid unnecessary initial arm rotation. Object geometry, masses, scene
sampling, region tests, external cameras and the official 200-step horizon are
unchanged. Piper and UR5 controllers have seven actions per arm. They have no
wrist cameras; use the original external `zed2i` view.
Initial validation covers the default scenes; L1--L3 scene variants
need their own task/robot validation.

For the original apple/banana task, select its task ID and Piper:

```bash
python -m mani_skill.examples.motionplanning.dual.two_robot_run \
  -e TwoRobotPickAppleBananaToBaskets-v1 --execution-mode sync --robot piper \
  --seed 401 -n 1 -b cpu -o rgb --shader default --save-video \
  --record-dir demos --traj-name sync-baskets-401
```

This adapter computes grasp and release goals from measured geometry and
reuses the same executor and replay. Original basket contents remain present;
normal contact with the corresponding fruit is allowed during placement when
both geometric centers and the measured contact lie in that basket under the
original task's region/height semantics. Other-object and transport contacts
retain their collision rules.

The synchronous solution uses `solve(env, seed=None, debug=False, vis=False)`
and operates on the supplied environment. It neither creates a second simulator
nor closes the caller's recorder. The caller's standard `RecordEpisode` saves
H5, JSON, and optional video under the normal task/motionplanning directory.
Choose a new output name/directory for each experiment.

## Planning and execution

MPLib plans each arm without advancing physics. Actual forward kinematics
checks each endpoint, with bounded native IK correction when needed. Its native
geometric paths are retimed against the selected phase velocity/acceleration
limits. The optional cuRobo backend supplies timestamped trajectories for the
same phase goals. Both routes use the common 20 Hz control clock and shared
executor, recording, replay, and conversion entries. Both arms start
together; a shorter path holds its endpoint and gripper state while its peer
finishes.

The shared executor owns the combined candidate check and submits one paired
action per control tick. The collision layer checks both robots, actual scene
geometry, and measured held-object geometry. Initial table support, intended
finger contact, and destination support have specific phase permissions.
Measured poses refresh held-object prediction after physics advances.
During open-hand retreat, existing own-finger/target contact may only separate:
original mesh/box geometry must not deepen the intersection, and cleared pairs
cannot re-enter. Candidate scans use independent contact state; execution and
replay require both geometric and physical contact to clear before proceeding.
Native contacts and held-object slip are sampled at 100 Hz. These are sampled
checks, not continuous collision detection or joint 14-DoF planning.

Both task adapters use preparation, pregrasp, descend, confirmed grasp,
lift, transport to measured release targets, release, optional retreat, and
optional final stability. There is no automatic recovery or spatial replanning.
For the Panda pill route, preparation holds the initial arm targets and opens
both hands for up to ten control ticks. It ends early after the selected
reach-stability window meets the existing hand-opening and measured arm/object
speed thresholds. Other robot/task routes retain ten native open-control ticks.
Replay uses the recorded preparation length; older records without that field
retain their ten-tick interpretation. The integration branch also records
release-separation behavior explicitly; see [integration decisions](integration.md#integration-decisions).

## Check boundaries

The checks above belong to expert generation and action replay. Policy
benchmarks step the environment directly; reporting task/contact facts must
leave policy actions, reward, native termination and official success unchanged.
A policy can lose an object, recover it and still satisfy the official task.

Within each physics substep, native contact bodies and point values are decoded
once. Contact classification still runs before and after the source-departure
latch update; these two passes have different support permissions. Fixed support
geometry fingerprints are captured at registration. Live scene, geometry,
finger, attachment and measured contact checks retain their existing guards;
mutable report evidence is copied for each snapshot. No contact permission,
quality threshold, sampling frequency or task horizon changes in this cleanup.

## Configuration and results

Defaults retain the validated pill settings: five grasp observations, 5 mm
position slip, 0.15 rad rotation slip, 8 cm measured retreat, and 20 final
stability observations. Quality selections can be changed with a JSON file:

```json
{"require_retreat": false, "require_final_stability": false}
```

Pass its path with `--sync-config policy.json`. The resolved settings are saved
in episode metadata. Disabling these quality phases still requires observing
actual release and the task result. The official 200-step horizon remains
fixed, with completion before step 200. Phase budgets use the planned duration
and selected future windows; there is no fixed extra three-tick margin veto.

Each standard episode JSON includes `dual_sync` with independent facts:

- `official`: environment success, elapsed steps, horizon, native done signals.
- `execution`: planning, geometry, budget, physical, control, or runtime failure reason.
- `quality`: selected retreat/stability outcome and evaluated flags.
- `physics`: actual forbidden contacts, source departure, and measured slip.
- `recording`: whether the caller saved the standard trajectory.

An official success is preserved when a later quality check fails. A proxy
collision refusal is a planning/execution refusal, not an observed physical
collision. Normal generation keeps compact phase summaries, actions, states,
seed, and resolved configuration; it does not archive source trees or require
per-file source hashes. Failures remain recordable; `--only-count-success`
continues to filter on the original environment success rule.

Geometric refusals use the `geometry` execution category. Historical reason
names such as `measured_collision` mean a geometry query at measured poses;
native forbidden contacts and observed grasp/drop/slip events remain physical
failures. The terminal summary reports execution categories, official success
and quality failures separately. Initial pill/table support is decided by all
actual convex pieces, finite tabletop containment and the existing penetration
and depth-growth limits. Actor-origin displacement and rotation are diagnostic;
measured departure permanently closes the initial-support permission.

## Replay and dataset conversion

For action replay with sampled physical and final outcome checks:

```bash
python -m mani_skill.examples.motionplanning.dual.sync.replay \
  --traj-path demos/TwoRobotPickPillToRegions-v1/motionplanning/sync-pill-401.h5
```

Replay restores the recorded initial simulator state once, then executes the
stored actions. It does not force recorded states during playback.

H5 stores `T` actions and `T+1` observations/states. Arm order is left then right:
each action has seven arm radians plus one normalized gripper command (16
values total); observed qpos has nine joints per arm (18 values total).
For Piper, ARX and UR5, each arm has six arm radians plus one normalized gripper command
(14 actions total), and eight physical joint coordinates (16 states total).
For xArm, each arm has six arm radians and one native angular gripper target
(0=open, 0.81=closed), with twelve physical state coordinates including passive
gripper joints. Replay submits saved native actions directly; it applies no
second open/close mapping. Rotary grasp stability uses the same measured
TCP-local finger-link distances as generation.
Native action keys and sizes are saved explicitly. New replay reads those
sizes and continues to accept the original Panda archives.
New recordings also save the measured controller joint order, physical state
joint order and native gripper endpoints in `motion_planning.native_action_layout`.
The converter reads this description for the selected robot pair. Earlier Piper pill
archives that lack this description remain replayable; conversion requires a
recording with an explicit native layout instead of inferring joint semantics
from its action width. Earlier Panda archives retain their original fallback.
Action `t` pairs with observation `t` and advances it to observation `t+1`.
Sampling is 20 Hz, or 0.05 s per control step; conversion and training must use
that recorded frequency.

```bash
python -m examples.baselines.lerobot_dataset.h5_to_lerobot \
  --input demos/TwoRobotPickPillToRegions-v1/motionplanning/sync-pill-401.h5 \
  --output-dir imitator_data --fps 20 --drop-terminal-frame --no-gpu --n-jobs 1
```

`--drop-terminal-frame` produces `T` supervised rows and preserves the terminal
observation in the source H5. Without the flag, the converter retains its old
`T+1` rows with the final action duplicated. To train and evaluate a policy from
the converted data, follow the [single-task ACT guide](../examples/baselines/act/README.md#single-task-simulation-data),
which covers dataset selection, timing and checkpoint reuse.

## Validation

```bash
python -m pytest -q tests/motionplanning/dual_sync
```

Tests cover scheduling/order, endpoint correction, retiming, collision/support
boundaries, failed grasp/drop/slip/release, policy selection, entry dispatch,
and conversion timing. Historical source-branch UR5 pill regressions passed at 183/184/184
steps, and Panda/Piper cases completed in 171/194 steps. Combined-branch
validation is reported in [the integration guide](integration.md#validation). These are small task integration checks; broader task/seed coverage
is needed before claiming general success rates.
The corrected Piper basket landing proposal chooses free space beside the
pre-existing fruit instead of always preferring the basket center. Six fixed
initial states complete generation and original-action replay in
191/189/188/187/187/188 steps. Earlier three-seed results (three official
successes, two complete generations, one accepted generation-plus-replay pair)
remain separate. Diagnostic replay identified a residual finger contact at the
release-to-retreat boundary in one old archive; the old trajectory is not
relabelled as passing.

Pill task profiles also describe robot-specific grasp/pregrasp offsets, positive
release clearance, optional gripper effort, held-motion acceleration and retreat
offsets. Panda profile values are unchanged; preparation and retreat follow the
explicit integration decisions above. xArm uses a side
grasp, a task-specific 0.5 Nm gripper drive limit and 2 rad/s² held acceleration;
its global agent/cube drive default remains 0.1 Nm. These are model settings,
not a hardware calibration or an equal-speed planner comparison.

Initial tabletop contact is checked against all actual convex pieces, the
finite table, existing depth/growth limits and permanent measured source
departure. The former fixed vertical-pick displacement/rotation window is
diagnostic; exceeding it alone does not prove a forbidden collision. Actual
forbidden contacts remain failures. Failure reports retain the collision pair
already found, and native forbidden events preserve their existing contact
points and substep index without adding another scan.

For bounded original-task collection, conversion and separate per-robot ACT
training/evaluation, see [the multi-embodiment guide](multi-embodiment.md#collect-and-learn-an-original-task).
Tested runtime: Python 3.12, SAPIEN 3.0.3, MPLib 0.1.1, NumPy 1.26.4.
