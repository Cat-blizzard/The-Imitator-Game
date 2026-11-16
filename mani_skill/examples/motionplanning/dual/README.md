# Dual-Arm Motion Planning (Imitator Game tasks)

This folder contains the scripted **motion-planning solutions** that generate
the dual-panda demonstration dataset for the Imitator Game benchmark.

```
mani_skill/examples/motionplanning/dual/
├── two_robot_run.py               # runner CLI (the entry point)
├── utils.py                       # shared helpers
├── solutions/                     # one solve() per base (L0/L1/L2) task
│   ├── two_robot_pick_cube_ycb.py # simple reference solution (start here)
│   ├── _001_stir_spoon.py         # full task solution example
│   └── ...
├── solutions_l3/                  # one solve() per L3 task
└── README.md
```

## Run one solution

```bash
# visualize the scene and data-collection process
python -m mani_skill.examples.motionplanning.dual.two_robot_run \
  -e TwoRobotPickCubeYCB-v1 -n 1 --vis --save-video

# collect 1 demo + its video (saved under demos/)
python -m mani_skill.examples.motionplanning.dual.two_robot_run \
  -e TwoRobotPickCubeYCB-v1 -n 1 --only-count-success --save-video \
  --traj-name L0_TwoRobotPickCubeYCB-v1 --record-dir demos
```

The command:
1. opens the scene (and the viewer with `--vis`);
2. runs the registered `solve()` until `-n` successful trajectories are recorded;
3. saves `demos/{env_id}/motionplanning/{traj_name}.h5` (actions + states +
   RGB-D obs) and the companion `.json`; `--save-video` additionally writes
   an MP4 of the collection run.

## How to add a solution for a new task

Every released task is four components: base env (`...-v1`), L3 env
(`...L3-v1`), base solution, and L3 solution. Only the solution is described
here; see `mani_skill/envs/tasks/_template/README.md` for the environments.

### Step 1: create the solution file

Copy the reference solution:

```bash
cp mani_skill/examples/motionplanning/dual/solutions/two_robot_pick_cube_ycb.py \
   mani_skill/examples/motionplanning/dual/solutions/_NNN_your_task.py
```

### Step 2: implement `solve()`

The runner calls `solve(env, seed=None, debug=False, vis=False)` and expects the
tuple `(left_res, right_res)` (each result from the last planner action, or
`-1` on failure).

Minimal skeleton:

```python
from mani_skill.envs.tasks import YourTaskEnv
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)


def solve(env: YourTaskEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)

    left_planner = PandaArmMotionPlanningSolver(
        env, debug=debug, vis=vis,
        base_pose=env.unwrapped.agent.agents[0].robot.pose,
        visualize_target_grasp_pose=vis, print_env_info=False,
        multi_robot_id=0,                      # left robot
    )
    right_planner = PandaArmMotionPlanningSolver(
        env, debug=debug, vis=vis,
        base_pose=env.unwrapped.agent.agents[1].robot.pose,
        visualize_target_grasp_pose=vis, print_env_info=False,
        multi_robot_id=1,                      # right robot
    )
    env = env.unwrapped

    # Preserve the other robot's gripper command while this robot moves.
    left_planner.move_to_pose_with_screw(
        my_pose, other_gripper_state=right_planner.gripper_state)
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)
    ...
    left_res = left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)
    return left_res, left_res
```

Key planner API (`PandaArmMotionPlanningSolver`, see `../panda/motionplanner.py`):

| Method | Purpose |
| --- | --- |
| `move_to_pose_with_screw(pose, other_gripper_state=...)` | plan + execute a screw-motion path to `pose` |
| `close_gripper(...)` / `open_gripper(...)` | gripper commands (returns `(obs, reward, terminated, truncated, info)`) |
| `follow_path(result)` | execute a returned plan |
| `gripper_state` | current gripper state (share it with the other planner) |
| `move_to_pose_with_screw(..., dry_run=True)` | plan without executing |

Useful helpers: `get_actor_obb(actor)`, `compute_grasp_info_by_obb(...)` from
`../panda/utils.py` to build grasp poses from an object's oriented bounding box.

### Step 3: register the solution

In `solutions/__init__.py`:

```python
from ._NNN_your_task import solve as solveYourTask
```

In `two_robot_run.py`, add to `MP_SOLUTIONS`:

```python
MP_SOLUTIONS = {
    ...,
    "TwoRobotYourTask-v1": solveYourTask,
}
```

If the task is L0/L1/L2, its L3 variant goes into `solutions_l3/` and is
registered with the `...L3-v1` key.

### Step 4: verify

```bash
python -m mani_skill.examples.motionplanning.dual.two_robot_run \
  -e TwoRobotYourTask-v1 -n 1 --only-count-success --record-dir demos
```

## Tips

- **Level hooks**: the base task environment (L0/L1/L2) is the same class; use
  `is_l2_enabled()` from `envs/tasks/tabletop/utils/L0_L3_utils.py` inside the
  solution to adapt waypoints when needed (see `_001_stir_spoon.py`).
- **L3 solutions** are separate files because the object affordances and
  waypoints differ structurally from the base task.
- **Determinism**: `env.reset(seed=seed)` at the start keeps episodes
  reproducible; the runner increments `seed` per trajectory.
- Pass `other_gripper_state` to preserve the other robot's gripper command.
  This parameter does not add inter-arm collision checking.

## Dual-arm trajectory scheduling (development)

`trajectory_schedule.py` implements simultaneous starts with independent
completion: each arm keeps its source timing, and the shorter trajectory holds
its final joint target and phase gripper command until the longer one finishes.
`plan_pose()` uses the existing solver's `dry_run=True` interface and checks the
planner's joint order against the caller-supplied arm-controller joint names.

`build_hold_schedule()` samples both trajectories by their actual timestamps
onto one control clock. Each yielded dictionary contains both arms' targets
for the end of the same control interval. Gripper commands are normalized
values in `[-1, 1]`, held constant within a phase. A zero-duration phase emits
no actions; opening or closing with a dwell requires a positive duration.

Run the CPU self-test from this checkout, with its dependencies activated:

```bash
python -m mani_skill.examples.motionplanning.dual.trajectory_schedule --self-test
```

The scheduler generates commands only. The separate development executor below
now connects those commands to the simulator; the existing collection runner
continues to use its original solution functions.

### Plan, check, then execute

The development path consists of four additional modules:

| Module | Role |
| --- | --- |
| `trajectory_execution.py` | `plan_pair()` calls both planners without stepping; `execute_schedule()` submits both arms through one `env.step()` per control interval. |
| `combined_collision.py` | Builds an independent MPLib world containing both robots and checks their combined configurations, including synchronous interpolation between commands. |
| `scene_collision.py` | Exports actual scene collision shapes, including rigid bodies absent from the actor registry; provides a conservative ground halfspace check. |
| `validate_execution.py` | Runs a reproducible small reaching phase, saves native/resampled trajectories, per-step measurements, collision results, source hashes and video. |

The executor requires a single environment with absolute `pd_joint_pos` arm
targets in radians. It checks the actual start state, controller joint order,
joint limits, control clock and every action before the first motion step.
Each arm's gripper must already be stable at the commanded OPEN or CLOSED
endpoint. During execution it checks measured configurations, tracking,
fingers, scene geometry and environment termination. The current validation
harness uses empty, OPEN grippers and aborts if snapshot geometry moves; moving
grippers, attached objects and contact cooperation require further work.

Run each module's CPU self-test with this checkout on `PYTHONPATH`:

```bash
cd /data/jxc/ld/ZeroWAM-for-TIG
export PYTHONPATH="$PWD/infrastructure/development/tig-dual-arm"
export MS_ASSET_DIR="$PWD/infrastructure/maniskill"
for module in trajectory_schedule combined_collision scene_collision trajectory_execution validate_execution; do
  CUDA_VISIBLE_DEVICES='' infrastructure/envs/tig-official/bin/python \
    -m "mani_skill.examples.motionplanning.dual.$module" --self-test
done
```

To reproduce the native validation, select an available GPU UUID for SAPIEN
rendering. Physics and MPLib run on CPU. The successful run used the UUID below;
check availability before reusing it. Without `--execute`, the harness only
plans and checks the path. Every invocation creates a new output directory.

```bash
CUDA_VISIBLE_DEVICES=GPU-1f8a72b4-0821-9275-e36f-49ab74f13572 \
  timeout 300s infrastructure/envs/tig-official/bin/python \
  -m mani_skill.examples.motionplanning.dual.validate_execution --seed 0 --execute
```

On 2026-11-15, `TwoRobotPickRemoteControl-v1`, seed 0, passed one empty-hand
reaching phase. The left and right native paths took 0.43696 s and 0.72724 s.
At 20 Hz the common horizon was 0.75 s (15 paired motion steps); the left
target stayed at its endpoint from the 0.45 s command onward. The maximum
final joint error was 0.00568 rad. Ten earlier hold steps settled passive
objects, bringing the complete invocation to 25 environment steps. Planning
itself used zero environment steps. This is a reaching check, not completion
of the remote-control manipulation task or a multi-seed benchmark.

The scene export covered 38 shapes, including the wall. In pinned MPLib 0.1.1,
Panda collision geometry against an FCL Plane caused a native crash. The
checker therefore rejects Plane objects in its native world and the harness
checks ground with conservative bounding-sphere/halfspace tests instead.
Only the two fixed `panda_link0`/tabletop mounting contacts are allowed;
robot-base poses are guarded. Moving-link, object and inter-arm contacts stay
enabled. A negative control with coincident cloned robot bases detected
inter-arm collisions. Path checks sample combined joint space at at most
0.02 rad per joint, and execution checks measured control-tick states;
neither is continuous collision detection.

Evidence, relative to the infrastructure root:

- `validation/dual-arm/trajectory-execution-selftests.json`: CPU checks and native Plane rejection regression.
- `validation/dual-arm/native-20261115T115348326385Z/report.json`: successful native run and source hashes.
- The same native directory contains `trajectory.npz` and `execution.mp4`; video includes one-second still padding on each side without additional physics steps.
- `validation/dual-arm/trajectory-execution.json`: consolidated evidence index, limits and retained failure references.

### Contact-confirmed grasp stage (development)

`grasp_execution.py` adds an open-hand left reach with a stationary right arm,
explicit bounded settling, and a separate `close_and_confirm()` stage. Closing
requests `-1`; the physical finger positions remain measured values. A grasp
passes after five consecutive control ticks with the native contact-based
`is_grasping()` predicate true and finger changes at most 0.2 mm per tick.
The predicate uses the Panda defaults of 0.5 N minimum force on each finger
and 85 degrees maximum force angle. A timeout does not advance to another
stage. Aborting stops simulation stepping; it does not implement braking,
retreat, or automatic release.

`grasp_collision.py` refreshes both robots' physical finger positions and all
exported scene shape poses. It reuses geometry and rebuilds the native world
so that broad-phase collision checks see moved objects. Only the active
robot's two finger links may contact the specified target during descent and
closing. Other links, the other robot and other objects remain checked; the
two fixed base/table mounting pairs and mandatory ground halfspace checks
retain the reaching harness's treatment. These are sampled snapshots, not a
prediction of moving objects or continuous collision detection.

Run the new CPU checks with the development `PYTHONPATH` and asset directory
set as above:

```bash
for module in grasp_collision grasp_execution validate_grasp; do
  CUDA_VISIBLE_DEVICES='' infrastructure/envs/tig-official/bin/python \
    -m "mani_skill.examples.motionplanning.dual.$module" --self-test
done
```

Native commands below execute a bounded phase; each creates a new directory
and records submitted actions, initial and per-step environment states, contact
forces, phase traces and video. Select an available GPU UUID for rendering.

```bash
CUDA_VISIBLE_DEVICES=GPU-1f8a72b4-0821-9275-e36f-49ab74f13572 \
  timeout 300s infrastructure/envs/tig-official/bin/python \
  -m mani_skill.examples.motionplanning.dual.validate_grasp --mode positive --seed 0

CUDA_VISIBLE_DEVICES=GPU-1f8a72b4-0821-9275-e36f-49ab74f13572 \
  timeout 300s infrastructure/envs/tig-official/bin/python \
  -m mani_skill.examples.motionplanning.dual.validate_grasp --mode empty-grasp --seed 0
```

On 2026-11-15, seed 0 of `TwoRobotPickAppleBananaToBaskets-v1` passed the
left-apple contact grasp stage: 10 preparation steps, 34 pregrasp steps,
26 descent steps and 7 closing/confirmation steps (77 total, 3.85 s at 20 Hz).
The final five closing ticks met the grasp/stability predicate. Actual finger
positions were 0.024286 m and 0.024259 m, with contact force magnitudes about
33.95 N and 33.91 N. The right arm's measured joints stayed at their initial
values, and checked states had no unallowed robot collisions.

The empty-grasp control returned `grasp_timeout` after 40 closing ticks:
finger positions and contact forces were zero, and no later stage ran.
Its overall validation status is `passed` because timeout was the expected
result. Both runs exported 69 scene shapes (68 native objects plus one ground
plane), and recorded H5 `[T,16]` actions matched every submitted action in
left/right order; environment state arrays contained `T+1` rows.

Evidence: `validation/dual-arm/grasp-selftests.json` and the consolidated
`validation/dual-arm/grasp-stage.json`; successful native directories are
`grasp-positive-20261115T121630792226Z` and
`grasp-empty-grasp-20261115T121621327172Z`. Each contains `report.json`,
`ticks.jsonl`, `trajectory.h5`, companion metadata and `execution.mp4`.
The first native attempt's evidence is retained with its artifact-check
failure; that cleanup-order bug was corrected before the two passing runs.

This stage confirms contact while the apple remains on the table. Lifting,
attached collision geometry, load-bearing grasp stability, carrying and
state playback remain separate validation steps. The official two-object
task success predicate remained false, as expected for this partial stage.

### Physical lift and hover (development)

`held_collision.py` records the measured TCP/object transform after grasp
confirmation. For each candidate arm configuration, world-frame FK predicts
the payload's pose and replaces all three apple shapes in a private collision
world. It checks the payload against both robots and every other scene object,
including object/object pairs that MPLib's normal-object world query omits.
Measured checks separately use the real physics actor. No physical weld,
object-pose assignment or collision-mask change makes the apple follow the hand.

Initial apple/table contact has a narrow upward-separation allowance: only
initially observed tabletop support pairs, within 3 mm rise, 1 mm horizontal
motion, 0.01 rad rotation, and initial penetration plus 0.2 mm. Once the measured
object separates, the allowance cannot be re-enabled. All later tabletop
collisions and mandatory ground halfspace checks remain active.

`lift_execution.py` keeps the left hand CLOSED and the right arm/hand at their
holding/open commands, predicts the complete candidate path, then monitors
measured contact, relative translation/rotation, object height, joint tracking
and collisions. Position slip above 5 mm, rotation slip above 0.15 rad, a drop
above 1 cm or loss of grasp aborts the phase without automatic release.
After reaching, it requires 20 consecutive valid hover ticks at 20 Hz.

Hover stability uses adjacent control-frame pose differences, with unchanged
speed limits of 0.03 m/s and 0.2 rad/s, plus cumulative world-pose drift limits
of 3 mm and 0.03 rad. Raw PhysX instantaneous velocities are retained. The
first native attempt timed out on an instantaneous-angular-speed gate despite
small pose changes; the CPU getter and H5 agreed, so this was not a discovered
Python channel-mapping error. That failed report remains unchanged. New runs
also record every 100 Hz physics substep and audit relative-pose envelopes;
this verifies sampled pose holding, not instantaneous dynamic rest or CCD.
The underlying instantaneous velocity/pose discrepancy remains diagnostic data.

Run CPU checks with the development environment variables set as above:

```bash
for module in held_collision lift_execution validate_lift; do
  CUDA_VISIBLE_DEVICES='' infrastructure/envs/tig-official/bin/python \
    -m "mani_skill.examples.motionplanning.dual.$module" --self-test
done
```

Native positive and deliberate-release controls each start a new recorded
episode. Select an available rendering GPU UUID:

```bash
CUDA_VISIBLE_DEVICES=GPU-1f8a72b4-0821-9275-e36f-49ab74f13572 \
  timeout 300s infrastructure/envs/tig-official/bin/python \
  -m mani_skill.examples.motionplanning.dual.validate_lift --mode positive --seed 0

CUDA_VISIBLE_DEVICES=GPU-1f8a72b4-0821-9275-e36f-49ab74f13572 \
  timeout 300s infrastructure/envs/tig-official/bin/python \
  -m mani_skill.examples.motionplanning.dual.validate_lift --mode release-control --seed 0
```

On 2026-11-15, seed 0 passed: the apple physically rose 0.118833 m, followed
by 20 hover ticks (1 s). This invocation used 124 total environment steps:
77 preparation/reach/grasp steps, 26 lift targets, one explicit settling tick
and 20 hover ticks. The right arm's measured joints did not change. Hover
world-pose drift was at most 0.299 mm and 0.001102 rad. All 235 recorded lift
physics substeps stayed within the attachment envelope: maximum relative
translation change 0.0191 mm and rotation change 0.00350 rad. Raw instantaneous
angular speed reached 1.840 rad/s across those substeps and remains recorded.

The release control deliberately applied OPEN to the left hand after a measured
0.100648 m rise, preserving the requested CLOSED command separately in its
trace. The next measured tick exceeded the slip limit (9.610 mm) and aborted
with `position_slip`, with no later motion. Its validation status is `passed`
because the applied fault and expected physical failure were verified;
arbitrary checker, measurement or step errors cannot pass that control.

Evidence index: `validation/dual-arm/lift-stage.json`; CPU checks:
`validation/dual-arm/lift-selftests.json`. Native directories
`lift-positive-20261115T123458991524Z` and
`lift-release-control-20261115T123513614128Z` contain reports, per-control-tick
and physics-substep JSONL, H5 actions/states, video and exact source snapshots.
The `[124,16]`/125-state positive recording and `[97,16]`/98-state control
matched submitted actions; state playback has not yet been tested.
The lift-only evidence above covers one scene and seed. The next stage below
adds single-arm transport and release; simultaneous carrying and full task
success remain pending.

## Physical carry, release and basket placement

`place_execution.py` adds three left-active/right-held phases. `execute_carry()`
keeps the apple grasped, checks the complete sampled candidate path with the
attached-object checker, and monitors real geometry, grasp, slip, tracking and
endpoint settling. Planned descent is permitted: drop is measured against the
current TCP-predicted object height rather than a world-height maximum.
`release_and_confirm()` holds arm joints while opening both hands and requires
actual finger positions near the open limits and no grasp for three ticks.
The fingertip/target contact allowance ends when release returns.

After release, `validate_place.py` switches to the live scene checker and plans
an 8 cm open-hand retreat. `verify_placement()` then requires 20 consecutive
ticks with both hands open, no grasp, static arms, the official apple-in-basket
flag, and an explicit physical region predicate. The harness requires the
apple geometric center inside the basket XY bounds, between basket Z-min minus
5 mm and rim plus 10 mm. This additional height condition excludes the hovering
release point and supplies the official predicate's missing lower bound. Placement stability uses
control-frame pose differences (0.03 m/s, 0.2 rad/s) and window drift (3 mm,
0.03 rad); raw instantaneous velocities remain separate recorded fields.
Only left-active operation is supported; reversed arm order rejects before motion.

The base task's official candidate release TCP offset is basket position plus
`[.03,.03,.12]`. The harness first rises to release height plus 8 cm, carries
above the basket, then lowers to release. Existing basket fruit stays in place.
No physical weld, actor pose assignment or collision-mask change is used.

Run with the development environment variables shown above:

```bash
for module in place_execution validate_place; do
  CUDA_VISIBLE_DEVICES='' infrastructure/envs/tig-official/bin/python \
    -m "mani_skill.examples.motionplanning.dual.$module" --self-test
done

CUDA_VISIBLE_DEVICES=GPU-1f8a72b4-0821-9275-e36f-49ab74f13572 \
  timeout 300s infrastructure/envs/tig-official/bin/python \
  -m mani_skill.examples.motionplanning.dual.validate_place --mode positive --seed 0

CUDA_VISIBLE_DEVICES=GPU-1f8a72b4-0821-9275-e36f-49ab74f13572 \
  timeout 300s infrastructure/envs/tig-official/bin/python \
  -m mani_skill.examples.motionplanning.dual.validate_place --mode retain-grip --seed 0
```

This is an **extended-budget engineering validation**: this harness explicitly
uses 400 steps, while the official task's default remains 200. The seed-0
positive run used 259 steps: 124 for preparation/grasp/lift/hover, 22 rising,
48 carrying above the basket, 19 lowering, 7 releasing, 19 retreating and 20
verifying. Release occurred at steps 214–220, beyond the official default.
Timing must be compressed and revalidated before claiming official-budget success.

On 2026-11-15 both new modules' CPU checks and both native controls passed.
In the positive run, the final release finger positions were about 39.549 mm,
actual TCP retreat was 80.447 mm, and the apple remained ungrasped in the basket
for 20 ticks (1 s). Its final geometric center was `[.146286,-.193408,.043689]`
metres, below the 71.553 mm basket rim. The right arm's complete 31-value state
did not change across all 260 recorded frames. Sampled collision checks found
no unallowed contacts; all 680 held physics substeps stayed within 5 mm/0.15 rad
attachment limits. This is finite sampling, not continuous collision detection.

The retain-grip control requested OPEN but applied CLOSED for 30 release ticks.
Measured finger joint positions remained about 24.3 mm and the grasp persisted;
the phase returned `release_timeout` without any retreat or verification tick.
Its official apple-in-basket flag was already true while still held, which is
why that flag alone cannot establish successful placement. This control passes
only for the expected timeout with verified applied actions, not checker errors.

Evidence index: `validation/dual-arm/place-stage.json`; CPU checks:
`validation/dual-arm/place-selftests.json`. Final native directories are
`place-positive-20261115T124829247240Z` and
`place-retain-grip-20261115T124851311744Z`. Each contains exact snapshots of all
12 imported source modules, video, plans, per-control and per-physics-step
measurements, plus H5 actions/states. The positive `[259,16]`/260-state and
control `[243,16]`/244-state recordings exactly match actual submitted actions.
An earlier control had correct physical timeout but incomplete JSONL export
from a NumPy boolean; its failed report is retained unchanged. The final Python
boolean conversion and error-isolated cleanup preserve the complete control.
Previous lift/grasp/execution source files remain byte-identical.

Only the left apple cycle is validated at one seed. Banana placement and full
task success are pending, as are simultaneous dual-arm manipulation, recovery,
multiple seeds. The recording audits above describe the original placement
runs; subsequent state/action replay validation is documented below.

## State playback and initial-state-only action replay

`replay_metrics.py` validates the input H5, metadata, actual submitted actions,
frame order and source snapshots, then measures all seven actor/articulation
state leaves. It handles no-observation recordings and compares quaternion
orientations independently of sign. `validate_replay.py` provides two modes:

- `state`: restore each of the 260 saved frames and render it, without stepping
  physics. This checks recorded pose/velocity restoration. Contact caches,
  forces, grasp predicates, RNG and reward state are not reconstructed.
- `action`: restore frame zero once, then submit all 259 original paired action
  dictionaries. There is no per-step state correction, replanning or control
  conversion. It measures the whole trajectory and checks actual release,
  retreat and the same 20-tick placement window.

The original CPU replay script assumes `obs/extra` is present and treats a flat
action dataset as a single-robot action. The adapter supports this recording's
empty observations and explicitly maps `[T,16]` to two ordered `[8]` actions.
Official core code and the earlier execution modules remain unchanged.

Run with the development environment variables shown above:

```bash
for module in replay_metrics validate_replay; do
  CUDA_VISIBLE_DEVICES='' infrastructure/envs/tig-official/bin/python \
    -m "mani_skill.examples.motionplanning.dual.$module" --self-test
done

CUDA_VISIBLE_DEVICES=GPU-1f8a72b4-0821-9275-e36f-49ab74f13572 \
  timeout 300s infrastructure/envs/tig-official/bin/python \
  -m mani_skill.examples.motionplanning.dual.validate_replay --mode state

CUDA_VISIBLE_DEVICES=GPU-1f8a72b4-0821-9275-e36f-49ab74f13572 \
  timeout 300s infrastructure/envs/tig-official/bin/python \
  -m mani_skill.examples.motionplanning.dual.validate_replay --mode action
```

Both CPU self-tests and both native runs passed on 2026-11-15. State playback
made 260 state-setter calls and zero physics steps; apple poses, arm joints and
finger positions matched exactly. The largest state difference was a robot
base-position round trip of 1.49e-8 m; TCP reconstruction differed by at most
1.19e-7 m and 3.14e-9 rad.

Action replay made one initial state-setter call and 259 physics steps. All
applied actions exactly matched the source H5. Maximum target-apple translation
error was 0.017269 mm and orientation error 0.002731 rad; maximum arm-joint error
was 7.153e-7 rad, finger-position error 1.863e-8 m, and TCP translation error
0.000493 mm. All 259 recorded grasp predicates matched. It again released the
apple in seven ticks, retreated 80.447 mm and passed the full 20-tick placement
window. Right-arm states did not change relative to the replay's first frame.

Limits were fixed before running. State restoration checks pose, joint and
velocity errors; action replay checks pose/joint errors and physical outcome.
All raw velocity differences are still reported: the basket's existing apple
had a maximum angular-velocity difference of 0.327373 rad/s despite small pose
differences. This validation establishes trajectory/outcome reproduction on
this backend and source episode, not bitwise dynamic equivalence.

Evidence index: `validation/dual-arm/replay-stage.json`; CPU checks:
`validation/dual-arm/replay-selftests.json`. Native directories are
`replay-state-20261115T125859213887Z` and
`replay-action-20261115T125910005070Z`. Each contains a report, 260 per-frame
error records, measured H5 states, source snapshots, replay video and a labeled
side-by-side comparison with the original video. All input file hashes remained
unchanged. The action run retains the source's 400-step engineering budget;
259 steps still exceeds the official default 200, and full-task success remains
false because banana placement is pending.

## Right-banana physical cycle, with the left arm held

`single_arm_execution.py` adds active-arm selection while keeping action keys,
planning schedules and collision calls in physical `(left, right)` order.
`validate_banana.py` exercises the right arm on the same seed-0 basket task:
pregrasp, descend, contact-confirmed close, lift and 20-tick hover, raise,
rotate, carry, lower, release, 8 cm retreat and 20-tick placement verification.
The earlier left-apple modules and official task/core files remain unchanged.

The OBB grasp follows the official solution. The selected release pose adds
`[.03,0,.12]` m to the banana basket position and rotates the grasp by 90 degrees
about world Z. The banana's long axis then fits the basket's long axis; the
official unrotated `[.03,.03,.12]` candidate projects beyond its short edge.
Actual sampled collision checks and placement measurements remain required.
Existing fruit stays in the basket. No physical attachment, pose assignment or
collision-mask change is used in either native acceptance run.

Two measured issues are addressed only in these new modules:

- `table_support_collision.py` recognizes a bounded numerical top touch when
  FCL reports a zero normal. It requires depth and tabletop-distance within
  1 micrometre, actual scaled convex vertices on the tabletop and their full
  horizontal projection inside it. The real contact had depth 87.207 nm and
  the lowest vertex was 22.570 nm above the tabletop. Raw contact data is
  retained. Standard support rules and strict checks after separation remain.
- Held-arm stability uses control-frame joint-position differences, still
  limited to 0.05 rad/s, plus whole-hover joint drift at most 0.003 rad. An
  independent 100 Hz joint-position audit must pass before carrying. A fresh
  action-prefix diagnostic measured raw velocity about 0.05984 rad/s while
  100 Hz joint-position speed was at most 0.0000477 rad/s. Raw velocities stay
  recorded; the underlying discrepancy is unresolved. Open-arm reach and
  placement retain the original raw joint-velocity criterion.

Run with the development environment variables shown above:

```bash
for module in single_arm_execution table_support_collision validate_banana; do
  CUDA_VISIBLE_DEVICES='' infrastructure/envs/tig-official/bin/python \
    -m "mani_skill.examples.motionplanning.dual.$module" --self-test
done

CUDA_VISIBLE_DEVICES=GPU-1f8a72b4-0821-9275-e36f-49ab74f13572 \
  timeout 300s infrastructure/envs/tig-official/bin/python \
  -m mani_skill.examples.motionplanning.dual.validate_banana --mode positive --seed 0

CUDA_VISIBLE_DEVICES=GPU-1f8a72b4-0821-9275-e36f-49ab74f13572 \
  timeout 300s infrastructure/envs/tig-official/bin/python \
  -m mani_skill.examples.motionplanning.dual.validate_banana --mode retain-grip --seed 0
```

On 2026-11-15 the three CPU self-tests and both native controls passed. The
positive used 319 control ticks: 10 preparation, 37 pregrasp, 25 descend,
7 close, 48 lift/hover, 21 raise, 59 rotate, 47 carry, 19 lower, 7 release,
19 retreat and 20 placement verification. Scheduled arm motion accounts for
235 ticks, additional endpoint settling for 20, lifted hover for 20, contact
close/release for 14, placement verification for 20 and preparation for 10.
This is a 400-step engineering run; the official default remains 200 steps.
Rotation is the largest single motion phase and timing still needs work.

The banana rose 118.360 mm; measured TCP retreat was 80.397 mm. Final release
finger positions were 39.411 mm, with three consecutive open/ungrasped ticks.
The banana then stayed ungrasped in the basket for 20 ticks (1 s). Its complete
XY bounds fit the basket; geometric-center height was 52.085 mm against a
71.553 mm rim. The conservative world AABB top was 90.334 mm; independently
transformed original collision-mesh vertices reached 78.644 mm, 7.091 mm above
the rim. The entire mesh is not below the rim; this is reported separately
from the placement gate. The passive left arm's complete 31-value state did
not change across any recorded frame in either run.
All 970 held physics samples met the 5 mm/0.15 rad attachment envelope; maxima
were 0.406 mm and 0.017002 rad. The lifted hover's 100 physical samples met
0.05 rad/s joint-position speed and 0.003 rad cumulative drift, with maxima
0.010610 rad/s and 0.001127 rad. Sampled checks found no unallowed collisions.

The retain-grip control requested right OPEN but applied right CLOSED for
30 release ticks, retaining both arm joint commands and the passive left
action. The grasp persisted with finger joint positions about 20.3 mm;
`release_timeout` stopped the run before retreat or verification. Unrelated
errors cannot pass this control. Both H5 recordings retain paired `[T,16]`
actions and `T+1` states (319/320 and 303/304), source snapshots and videos.
CPU zero-normal contact fixtures use synthetic contacts with actual geometry;
real FCL zero-normal evidence is in the native coverage and support diagnostic.

Evidence: `validation/dual-arm/banana-stage.json`,
`banana-selftests.json` and `banana-recording-audit.json`. Native directories
are `banana-positive-20261115T132429292748Z` and
`banana-retain-grip-20261115T132458195759Z`. Both earlier failed attempts and
both diagnostics remain available through the evidence index.

Left-apple and right-banana cycles have now each passed independently at seed
0. This banana episode leaves the outside apple untouched, so full task success
is false. Banana state/action replay, simultaneous manipulation, the official
200-step full-task budget, recovery and multiple seeds remain pending. The next
step is to schedule both physical arms together and recheck their combined
paths and held-object geometry, retaining these separate cycles as regressions.

## Synchronized physical apple and banana manipulation

`dual_arm_execution.py` plans both arms at the same measured scene state and
executes one paired action dictionary per control tick. Each phase starts both
trajectories together without time stretching. An arm that finishes early
holds its final joint and phase gripper targets; both arms must satisfy the
phase's physical conditions before the next phase. The first integration uses
phase barriers rather than independently advancing each arm's task stages.

`dual_payload_collision.py` replaces both carried objects in one private
collision world using their own measured TCP/object relations. It checks robot
self/interarm/world collisions, each payload against the environment and both
robots, and payload against payload. Only an owner's fingers may contact that
owner's target during permitted phases. Both initial table support policies
and strict ground halfspaces remain. The predicted relations do not create
physical attachments. At release both targets return to measured live geometry.

`validate_sync.py` runs the full paired cycle, including a shared 20-tick lifted
hover and final 20-tick placement window. The native task terminates as soon as
both geometric centers satisfy its basket predicate, which can happen while
the objects are still grasped. For this engineering validation, the capture
adapter continues success-only termination signals to confirm actual release,
retreat and settling. Native termination/success flags remain intact in H5 and
ticks; failure and the 400-step truncation still stop execution. No task/core
file, physical pose, collision mask, or earlier execution module is changed.

Run with the development environment variables shown above:

```bash
for module in dual_payload_collision dual_arm_execution validate_sync; do
  CUDA_VISIBLE_DEVICES='' infrastructure/envs/tig-official/bin/python \
    -m "mani_skill.examples.motionplanning.dual.$module" --self-test
done

CUDA_VISIBLE_DEVICES=GPU-1f8a72b4-0821-9275-e36f-49ab74f13572 \
  timeout 300s infrastructure/envs/tig-official/bin/python \
  -m mani_skill.examples.motionplanning.dual.validate_sync --mode positive --seed 0

CUDA_VISIBLE_DEVICES=GPU-1f8a72b4-0821-9275-e36f-49ab74f13572 \
  timeout 300s infrastructure/envs/tig-official/bin/python \
  -m mani_skill.examples.motionplanning.dual.validate_sync --mode retain-right --seed 0
```

On 2026-11-15 all three CPU self-tests passed (17 collision checks, 14 executor
groups and 6 harness groups), followed by both native controls. Real CPU/FCL
fixtures include full robot interarm overlap and clear/overlap/clear queries
with the official apple/banana assets, each cooked into three convex shapes.
The official-asset overlap produces three payload/payload collision records.
These CPU fixtures have zero physics steps and are distinct from task runs.

The native positive used 323 ticks: 10 preparation, 37 pregrasp, 26 descend,
7 close, 48 lift/hover, 22 raise, 59 banana rotation with apple held, 49 paired
carry, 19 lower, 7 release, 19 paired retreat and 20 common placement checks.
This splits into 244 scheduled arm-target ticks, 15 additional endpoint-settling
ticks, 20 lifted-hover ticks, 14 close/release ticks, 20 final-placement ticks
and 10 preparation ticks. Planning advances zero physics steps. For example,
the carry source durations are 2.3708 s left and 2.1914 s right on a 2.4 s
shared command horizon; the right endpoint holds while the left finishes.

Apple and banana rise were 118.834 mm and 118.363 mm. Actual retreats were
80.447 mm and 80.397 mm. Both hands physically opened and neither object was
grasped for the required three release ticks and full final 20-tick window.
Final center heights were 43.693 mm and 51.949 mm against 71.553 mm basket
rims; both complete XY bounding boxes were contained. The banana's conservative
AABB still extends above the rim; independently transformed original asset
vertices reach 78.433 mm, 6.880 mm above it. These are recorded separately
from the placement gate.
Official final apple/banana predicates, static-robot flag and full-task success
are all true. This remains a 400-step engineering run, beyond the official
200-step default, and does not establish official-budget benchmark success.

All 985 held physical samples per object passed the 5 mm/0.15 rad relative-pose
envelope. Maxima were 0.0194 mm/0.005730 rad for apple and
0.4042 mm/0.017001 rad for banana. The shared hover's 100 physical samples
passed each arm's 0.05 rad/s position-difference speed and 0.003 rad drift
limits. Raw instantaneous velocities remain separate; open reach and final
placement still use raw arm-velocity gating. Collision checks are sampled at
planned configurations and control frames, not every physical substep or CCD.

The `retain-right` control changes only the right release gripper from requested
OPEN to applied CLOSED. The left releases while the right remains grasped;
after 30 ticks the release barrier times out, without retreat or verification.
This 307-tick control has official success true despite the retained banana,
demonstrating why the physical release checks are required. Native success-only
termination was continued on 58 positive and 42 control ticks, retained in H5.

Evidence: `validation/dual-arm/sync-stage.json`, `sync-selftests.json` and
`sync-recording-audit.json`. Native directories are
`sync-positive-20261115T134421956872Z` and
`sync-retain-right-20261115T134457550731Z`; they contain video, plans, 18 exact
source snapshots, actual paired actions, states and physical-substep logs.
H5 shapes are `[323,16]`/324 states and `[307,16]`/308 states. The earlier
15 execution modules and both standalone regression recordings are preserved.
Independent H5 audit confirms all actions, states, flags and source hashes;
179 scheduled ticks change both arms' joint targets together. During banana
rotation the left target holds exactly, with actual arm-joint drift at most
0.000516 rad. The final 20 recorded object poses are constant for each fruit.

Next work is full-task timing reduction and acceptance under the unchanged
official 200-step limit. Subsequent synchronized replay is documented below;
multiple seeds, conflict recovery and handoff remain pending. A phase abort stops advancing
the environment; it does not implement physical braking or automatic recovery.

## Synchronized state playback and action replay

`sync_replay_metrics.py` validates the frozen synchronized recording and both
physical outcomes. `validate_sync_replay.py` reuses the earlier state-error and
action-sequence helpers, with explicit physical left/apple and right/banana
pairing. Previous execution and replay sources remain unchanged. Both new
modules' CPU checks passed (6 metric groups and 4 harness groups).

Run with the development environment variables shown above:

```bash
for module in sync_replay_metrics validate_sync_replay; do
  CUDA_VISIBLE_DEVICES='' infrastructure/envs/tig-official/bin/python \
    -m "mani_skill.examples.motionplanning.dual.$module" --self-test
done

CUDA_VISIBLE_DEVICES=GPU-1f8a72b4-0821-9275-e36f-49ab74f13572 \
  timeout 300s infrastructure/envs/tig-official/bin/python \
  -m mani_skill.examples.motionplanning.dual.validate_sync_replay --mode state

CUDA_VISIBLE_DEVICES=GPU-1f8a72b4-0821-9275-e36f-49ab74f13572 \
  timeout 300s infrastructure/envs/tig-official/bin/python \
  -m mani_skill.examples.motionplanning.dual.validate_sync_replay --mode action
```

On 2026-11-15 state playback passed: 324 state-setter calls, zero physics
steps and all seven recorded entity leaves within the original limits.
Object poses, arm joints, finger positions and velocities matched exactly;
robot-base position round-trip error was at most 1.49e-8 m. Both TCP errors
were within 1.20e-7 m and 3.37e-8 rad. This restores the recorded pose/velocity
fields; static objects are rebuilt by seed and contact caches, sleep/solver
state, RNG and controller internals are not restored. No grasp/contact outcome
is asserted for zero-step state playback.

Action replay restored frame zero once, executed the exact 323 paired actions,
and observed all 324 states without replanning or state correction. It
reproduced both target outcomes: both grasp sequences match all 323 ticks,
release has three consecutive physically open/ungrasped ticks, both TCPs
retreat about 80 mm, and both objects pass the common 20-tick placement window.
Maximum apple/banana translation errors were 9.794/24.791 micrometres and
rotation errors 0.000196/0.000616 rad; TCP translation errors were
0.477/0.645 micrometres. All native success/termination/truncation flags match,
including the 58 continued success-only terminations at steps 266–323.

The action run's **overall strict verdict remains failed**. The only failing
check is the seven-entity pose-error gate: the basket's existing `apple_in`
has rotation error 0.07517195 rad at frame 284 (the final release tick), above
the fixed 0.05 rad limit. Its maximum translation error is 1.945 mm, within
the 5 mm limit. This is the only violating frame; its final error is
0.361 mm/0.014065 rad. No threshold or original failed report was changed.
The target-outcome verdict is separate and passed.

One additional reset-only diagnostic starts from a same-seed reset, first
asserts all seven visible initial states equal the source exactly, then sends
the same 323 actions with zero state-setter calls. All seven recorded state
arrays and native flags match the original exactly in that diagnostic. It is
not the restore-once protocol and does not replace its failed verdict. It has
no per-frame rendering/contact observer; the exact internal source of the
restore-path difference is not isolated. The observed discrepancy begins
during release; its specific contact cause is not recorded.

Evidence: `validation/dual-arm/sync-replay-stage.json`,
`sync-replay-selftests.json` and `sync-replay-audit.json`. Native directories
are `sync-replay-state-20261115T140012409088Z` and
`sync-replay-action-20261115T140023455173Z`; each has 28 source snapshots,
observed-state H5, 324 per-frame error rows, replay video and a labeled
side-by-side comparison. Diagnostic directory:
`sync-replay-reset-diagnostic-20261115T140441142120Z`.
All six input evidence files and the earlier 18 execution sources are intact.
This remains the original 400-step engineering episode; 323 exceeds 200.
Replay did not rerun collision checks or establish instantaneous dynamic
equivalence. The subsequent controlled diagnosis and guarded restoration are
documented below; the original full-restoration failed verdict remains intact.

## Fixed-root restoration diagnosis and compact 200-step candidate

`diagnose_sync_restore.py` uses one runner to separate state setters from the
original full observer. No-restore/minimal and no-restore/full observation both
reproduce all seven arrays exactly. Full restore/minimal and full restore/full
produce identical arrays to each other and reproduce the original frame-284
failure. Actors-only restore is exact; articulations-only matches full restore.
Restoring articulation velocities and joints while omitting root-pose writes is
exact. This isolates the trigger to the root-pose write for this episode; it
does not establish the underlying SAPIEN numerical/contact mechanism.

Eight calls are retained: seven complete 323-action diagnostics and one
zero-action full-observer dependency error, subsequently fixed and self-tested.
See `validation/dual-arm/restore-diagnostic-stage-20261115T153225446218Z/report.json`
and `restore-audit-20261115.json`. All original source and input evidence remains.

`fixed_root_restore.py` is a guarded development adapter: it first requires a
single CPU environment, fixed Panda roots and **exactly equal requested/current
root poses**. It then restores all actors, root velocities, joint positions and
joint velocities, skipping only the redundant root-pose writes. Unsupported
root poses fail before mutation. This is not an arbitrary physics checkpoint.
The complete frozen 323-action replay using this adapter passed, with all seven
324-frame state arrays matching the input bytes and both physical outcomes
reproduced. See `fixed-root-replay-audit-20261115.json` and
`compact-replay-action-20261115T153314438043Z/report.json`.

`validate_sync_compact.py` adds an explicit execution profile. Its accepted
seed-0 candidate uses the native 200-step TimeLimit, unchanged 20Hz control and
100Hz physics, and counts all 10 preparation ticks. It merges lift/raise and
banana orientation/transport; it omits the independent airborne 20-tick
diagnostic hover. `--execution-profile diagnostic-hover` retains that separate
diagnostic, and the old hover regression remains preserved. The final common
20-tick placement window and original collision/slip/release thresholds remain.

Nominal planner limits are 0.9rad/s and 0.9rad/s² for pregrasp/descend, and
1.5rad/s and 2.6rad/s² for held moves and retreat. The solver receives 99% of
each value; source derivatives are checked against the nominal limits. That
margin avoids a measured native acceleration interpolation overshoot without
relaxing its acceptance gate. Actual MPLib TOPP limit arrays are updated per
phase; source timestamps are never scaled. The velocity limit is below all
seven asset-URDF limits; no hardware acceleration certification is claimed.

| Physical stage | Original ticks | Compact ticks |
|---|---:|---:|
| Preparation, approach, descend, grasp | 80 | 80 |
| Lift, independent hover, raise | 70 | 24 |
| Banana orientation and transport | 108 | 35 |
| Lower and release | 26 | 26 |
| Retreat | 19 | 14 |
| Common placement window | 20 | 20 |
| Total | 323 | 199 |

The measured physical cycle is 9.95s instead of 16.15s. Native success first
occurs at tick145, while physical release, retreat and common settling finish
at tick199, with no native truncation. Both objects physically rise at least
0.10m; actual retreats are 80.835mm and 80.707mm. All 390 held 100Hz samples per
object meet the original 5mm/0.15rad slip envelope. This combines route changes,
timing changes and omission of a separate diagnostic hover; stage savings are
not an isolated scheduling or planner-backend comparison.

The identical-config retain-right control passes at tick188 by observing the
expected 30-tick release timeout, with the left released and the right still
grasping. It never enters retreat or placement. Official geometric success is
already true, which remains insufficient to establish physical release.

With the development environment variables set as above, reproduce the accepted
configuration explicitly (the CLI defaults retain conservative nominal limits):

```bash
infrastructure/envs/tig-official/bin/python \
  -m mani_skill.examples.motionplanning.dual.validate_sync_compact \
  --mode positive --route merge-lift --execution-profile execution --seed 0 \
  --approach-vel-limit .9 --approach-acc-limit .9 \
  --joint-vel-limit 1.5 --joint-acc-limit 2.6
```

Change only `--mode retain-right` for the failure control. New variable-length
recordings use `compact_replay_metrics.py` and `validate_compact_replay.py`;
the fixed 323-step loaders and original replay source remain unchanged. For
action replay select `--restore-policy fixed-root-exact`; state playback uses
the original full policy and performs zero physics steps. The six new modules
have CPU self-tests, including four positive and eight negative loader fixtures.

For the frozen 199-action candidate, both subsequent replay modes passed.
State playback restores 200 frames and takes zero physical steps (maximum
root-pose round-trip difference 1.49e-8m). Guarded action replay restores once,
executes all 199 actual actions, and reproduces all seven 200-frame state arrays
exactly, along with both physical outcomes and all native flags. It uses the
unchanged 200-step budget. Recordings are
`compact-replay-state-20261115T154218604821Z` and
`compact-replay-action-20261115T154327760030Z`; action replay does not rerun
collision or 100Hz slip checks, which belong to the original execution audit.

Evidence index: `validation/dual-arm/sync-compact-stage.json`. Five rejected
candidates remain recorded, including the native numerical overshoot, two
support-contact failures and two genuine 200th-tick truncations. Independent
positive and negative audits preserve these failures. Results cover one seed
and one backend, with only one control tick of budget margin; multiple seeds,
recovery, handoff, continuous collision checking and BPP policy evaluation
remain separate work.

## Fixed-parameter seeds 1–10 (2026-11-15)

`validate_compact_seeds.py` freezes the accepted compact configuration before
launching independent native subprocesses in ascending seed order. Seeds 1–10
each receive exactly one attempt; seed0 is excluded. No retries, per-seed
adjustments or source/configuration changes are allowed. The native entry,
19 execution sources and 11 task/core/controller/URDF sources are checked
against frozen identities. Administrative `status=completed` means all attempts
finished; it does not mean all trajectories passed.

From the project root, with the development environment variables set as above:

```bash
infrastructure/envs/tig-official/bin/python \
  -m mani_skill.examples.motionplanning.dual.validate_compact_seeds --self-test
infrastructure/envs/tig-official/bin/python \
  -m mani_skill.examples.motionplanning.dual.validate_compact_seeds --prepare-only
```

Preparation prints a new batch directory containing `protocol.json`. Launch
that prepared batch once with `--batch-dir <printed-absolute-directory>`;
an existing started batch cannot be resumed or rerun. Each seed retains its
invocation receipt, log, native report, H5 actions/states, tick/substep traces,
plans, video and source snapshots. The runner self-test actually exited0.

The completed frozen batch is
`infrastructure/validation/dual-arm/sync-compact-seeds-20261115T155130470922Z`.
Full physical closure and observed native success are both **5/10**: seeds
3/5/7/8/10 finish in 194/195/195/195/198 ticks. Seeds1/9 reject initial banana
support normals; seeds2/4/6 abort on measured banana/table contact at the first
lift tick, with only another convex piece registered as initial support.
These are support-initialization or geometric-check outcomes, not established
inter-arm collisions or environment configuration faults. All10 dual-grasp
confirmations pass. No native truncation or subprocess timeout is observed;
early safety stops leave the later stages of those five episodes untested.

The independent `audit.json` passes action/flag/T+1 seven-entity state checks,
frozen identities, execution order and recorded physical gates, including the
successful episodes' 100Hz slip and final common20 window. It does not rerun
physics or collision geometry. New per-seed state/action replay was not run.
Pure planning wall time remains uncollected. See batch `report.md`,
`summary.json`, `protocol.json` and `failure-analysis.json`. Current priority is
support-contact diagnosis and convex-piece transition handling; seed0–10
should become regression cases after any uniform revision, with separately
predeclared fresh seeds for subsequent generalization checks.

## Support-contact diagnosis (2026-11-16)

`diagnose_support_geometry.py` reads the archived attach and first-lift actor
states for failed seeds1/2/4/6/9 plus control seed3. It reconstructs current
asset geometry in private FCL objects, writes full convex vertex arrays and
compares direct SAPIEN poses with the original held-object recomposition path.
Environment and scene step methods are guarded; the completed run took zero
physics steps, actions or recorded physical-state restores. Its CPU self-test
and diagnostic run both actually exited0.

The banana has three convex pieces. Seeds1/9 reproduce near-coplanar support
rejections with unit-length but nonvertical FCL normals; independent vertex
penetrations are only about0.135/0.035 micrometres. Seeds2/4/6 switch support
from piece1 to piece0 with real15–33 micrometre penetration; old measured
contacts are reproduced exactly. The original exact-piece support lookup and
permanent-release rule both need attention. No runtime fix or gate change was
applied, and the archived5/10 result is unchanged.

Results, limitations and proposed repair bounds are in
`infrastructure/validation/dual-arm/support-geometry-20261115T160722994022Z/diagnosis.md`.
Its `audit.json` independently checks input/source identity and geometry
arithmetic. Original cooked meshes and the two rejected raw normals were not
recorded, so the new cooking remains a diagnostic reconstruction. The report's
reset counter covers one explicit reset; construction also resets once, as
derived from the frozen BaseEnv source. Both occur under the step guards.

## Whole-object support correction and regression (2026-11-16)

The correction changes only `held_collision.py` and
`table_support_collision.py` within the original19 execution files. Near-plane
support retains the1µm independent vertex/depth/position/footprint bounds and
preserves raw normals. All convex pieces share initial-object support, with
the original motion limits and initial deepest penetration plus0.2mm. Only
measured refresh can permanently release support, once all pieces separate or
the original3mm height band is exceeded. Candidate vertices use the supplied
object pose and captured shape-local transform; they do not read the actor's
current pose or change measured support state. Non-ConvexMesh primitives and
missing-table scenes explicitly retain the base exact-pair policy.

Initial cooked geometry and compact contact/state histories are recorded in
coverage. The support module's27 real-geometry tests, base11 tests, original
dual-payload17 tests and execution/replay compatibility tests actually exit0.
Depth and footprint negatives include cases inside the motion bounds.

`validate_support_regression.py` freezes positives0..10, one seed0 retain-right
control, and state/action replay of its new seed0 positive, each once. It
checks the other17 execution sources, core11,8 assets/builders and protected
historical files before and after every case. The old
`validate_compact_seeds.py` intentionally expects the old19 source hashes;
use archived sources for that historical batch. For the new version, from the
project root with the development environment variables set as above:

```bash
infrastructure/envs/tig-official/bin/python \
  -m mani_skill.examples.motionplanning.dual.validate_support_regression --self-test
infrastructure/envs/tig-official/bin/python \
  -m mani_skill.examples.motionplanning.dual.validate_support_regression --prepare-only
```

Launch the printed prepared directory once using `--batch-dir`; no resume or
retry is allowed. The completed batch is
`infrastructure/validation/dual-arm/support-regression-20261115T163705585638Z`.
Matched seeds1..10 achieve full physical closure6/10 and observed native
success8/10. All five former support stops are passed. Seed4 newly completes;
seeds1/6 stop before carry on native acceleration-limit violations, and
seeds2/9 reach200 ticks with18/20 and10/20 common stable samples. All original
successes remain; seed0 is separately passed at199 ticks and excluded.

The negative correctly detects retained right grasp at188 ticks. State replay
passes its existing tolerances with200 frames/zero physics; guarded action
replay executes199 actions and reproduces all seven200-frame state arrays
exactly. Old successful full recordings and old failed prefixes are byte-equal
for actions, native flags and all seven state leaves. All14 raw recording
checks pass. See batch `report.md`, `audit.json` and `support-fix.patch`.
This is a regression on already observed seeds; fresh-seed generalization and
BPP policy evaluation remain untested here.

## Source timing, release geometry and shorter clearance (2026-11-16)

The current `validate_sync_compact.py` uses
`analytic-phase-retiming-v1`, `geometry-informed-v1` release poses and
`compact-clearance-v2` transport geometry. Its frozen source SHA256 is
`ecc2088e12947921d829056d02cf11041f048a6da682497ed2a5871ee6758409`.
All18 shared execution sources, core11, assets/builders, clocks and physical
acceptance gates retain their reviewed identities.

Native TOPPRA constraints hold on a finite path grid and can miss an interior
acceleration peak. The entry first stores the untouched native result, then
reconstructs the same TOPPRA curve and requires exact equality of returned
samples. It partitions cubic path and constant path-acceleration intervals,
checks analytic stationary points and both boundary limits, and applies one
common phase scale `s=max(1,vmax/V,sqrt(amax/A))`. It preserves q and scales
`t/duration *= s`, `v /= s`, `a /= s**2`, then rebuilds the shared schedule and
uses the original combined collision and physical execution gates. Both
native and retimed arrays and full spline/grid evidence remain in the NPZs.
Option A still holds the shorter arm's endpoint; scaling does not equalize arm
durations. These are numerical source-curve bounds, with the existing1e-6
output tolerance, rather than physical actuator or continuous collision
certification.

The apple release TCP offset relative to its basket is now `[.03,.08,.10]m`;
the banana remains `[.03,0,.12]m`. Cooked geometry across all11 known seeds
checks the held objects and hands at the candidate endpoints. Transport height
is `max(current_z+.12, release_z+.06)` instead of `release_z+.08` in that
maximum; the original120mm commanded lift floor and actual100mm gate remain.
Release endpoints, nominal1.5rad/s and2.6rad/s² global limits, .9/.9 approach
limits,99pct solver margin,20Hz/100Hz clocks, native200 budget, actual80mm
retreat and final20 common stable ticks remain fixed.

The first timing/release version completed9/10 matched seeds. Seed6 reached
200 ticks with19/20 stable samples and remains a failure in
`timing-regression-20261115T170213786951Z`; it is not replaced by a retry.
The next separately frozen route version is
`clearance-regression-20261115T171457407702Z`. It runs each positive0..10,
retain-right negative0, state replay and action replay once. Matched1..10
complete10/10 in189–195 ticks (mean190.3); development0 is191 and excluded.
Seed6's final20 are environment steps176..195. The retain-right control
correctly stops at180 without retreat or placement verification. State replay
restores192 frames with zero physics and passes the original tolerances;
action replay restores once, executes191 actions and reproduces all seven
192-frame state arrays, actions and native flags exactly.

Region acceptance retains the original full XY bounding-box tolerance,
center XY/height and floor bounds. `object_fully_below_rim`, computed from the
conservative bounding box, remains descriptive. Sampling, fixed-base CPU replay
and this one L0 task define the
validated scope; unseen seeds and BPP policy performance remain separate.

Use the established development environment variables and the latest runner:

```bash
infrastructure/envs/tig-official/bin/python \
  -m mani_skill.examples.motionplanning.dual.validate_clearance_regression --self-test
infrastructure/envs/tig-official/bin/python \
  -m mani_skill.examples.motionplanning.dual.validate_clearance_regression --prepare-only
```

Launch the printed new directory exactly once with `--batch-dir`; archived
runners intentionally reject changed source identities. The latest protocol
protects2860 existing files, including the complete preceding failure batch.
Evidence, independent audit, source review, raw source/retimed derivatives,
phase comparison, videos and replay records are linked from the latest
batch's `report.md`. The matched subprocess total is284.923s, yielding2.106
accepted trajectories/minute; pure planning wall time remains uncollected.
These observed costs include startup, simulation, rendering and recording.

## Frozen test on new seeds11..20 (2026-11-16)

`validate_unseen_seeds.py` freezes the same ECC execution source, all19
execution files, complete configuration, core11,8 assets/builders and package
versions. It runs seeds11..20 once each in ascending order, with no geometry
screening, retries, replacement seeds or parameter changes. All3,551 protected
historical files retain their hashes. Its actual self-test exits0.

The completed batch is
`infrastructure/validation/dual-arm/unseen-seeds-20261115T173800Z`.
New random initializations of this one L0 task achieve9/10 full physical
closure and9/10 observed native success. The successful episodes take188–193
ticks (mean190.8889, sample standard deviation1.36423); each completes actual
release,80mm retreat and the final20 common stable ticks before native
truncation. The matched known-seed1..10 result remains separately10/10.

Seed12 stops after80 executed ticks: the first lift candidate sample at t=0
reports left apple convex/table-box contact depth0.0924µm with a zero raw
normal. Initial whole-object support was not registered; initial recorded
vertex clearance is+0.1213µm. The lift executes zero steps. These receipts
require a saved-evidence check of support initialization and candidate
geometry before assigning the cause to physical contact or numerical error.
The failure remains in the denominator; the execution method is unchanged.

Independent raw H5/actions/native-flags/T+1-state,100Hz, retiming, route and
physical-gate audit exits0 with `evidence_integrity=passed`, including the
valid failed prefix. There are no native truncations, subprocess timeouts or
artifact failures. New episode replay was not run; earlier seed0 controls and
replays remain separate regression evidence. All ten subprocesses total
272.593s, yielding1.981 accepted trajectories/minute under that timing scope;
pure planning wall time remains uncollected. See batch `report.md`,
`protocol.json`, `summary.json`, `audit.json` and `failure-analysis.json`.

Any later method revision retains this original9/10 result and uses its own
frozen protocol. Cross-task, additional-robot and BPP policy evaluation remain
separate work.

## Seed12 support-initialization diagnosis (2026-11-16)

The saved frame80 was reconstructed with standalone native FK and FCL, with
zero environment/scene creation, physics, actions or planning. All three cooked
scaled-vertex arrays exactly match the archive. Original table quaternion and
query settings reproduce the archived candidate depth/normal/position exactly.

Measured shape and measured TCP/attachment recomposition report no contact.
Native FK differs from measured TCP by -0.112331µm in z, reducing the apple's
minimum gap from0.121327µm to0.009536µm; its candidate reports0.092410µm
FCL depth with zero normal. The unchanged1µm numerical top-geometry test
passes on this candidate. Initialization requires actual FCL touching before
registering support, while registered support remains active within1µm
separation. The candidate is rejected at the no-initial-support gate before
geometry validation. This identifies inconsistent initialization/state rules
triggered by the measured/FK residual.

Execution files and thresholds are unchanged; the original9/10 outcome and
all4,051 protected input/history files remain intact. Original triangle indices
were not exported; reconstructed faces/hashes and exact archived-vertex checks
are retained. See
`infrastructure/validation/dual-arm/seed12-support-consistency-20261116/diagnosis.md`,
`result.json`, `geometry-arrays.npz` and `run-receipt.json` for the actual
zero-step diagnosis/self-test. A repair and native regression remain next.

## Near-top initialization repair and regression (2026-11-16)

`table_support_collision.py` now registers independent initial geometric
support when a nonintersecting convex piece satisfies the existing1µm band,
complete tabletop footprint and vertex-height proof. It exports
`initial_near_top_support_receipts` without inventing FCL contacts or pairs.
Whole-object checks, candidate motion/depth limits, measured permanent
release, all other18 execution sources and the full configuration remain
unchanged. The module's33 self-tests and archived seed12 zero-step fixture
actually exit0. Frozen support SHA256:
`4e67b24ca89ba1043e655200cddca12b4ab630204efb2a1030a3892802ee6add`.

`validate_near_support_regression.py` runs positive0..20, retain-right0 and
state/action replay0 once each. Completed batch:
`infrastructure/validation/dual-arm/near-support-regression-20261115T181122Z`.
Known seeds1..20 achieve20/20 full closure in188–195 ticks (mean190.65,
sample standard deviation1.53125); development0 is191 and excluded. Seed12
completes192 ticks, permits the exact archived t=0 candidate under the original
gates, permanently releases tabletop support at measured step81, and completes
common stable steps173..192. Original unseen9/10 evidence is retained;
this new result is known-seed repair regression.

The negative detects retained right grasp at180 ticks. State replay restores
192 frames with zero physics and passes the original tolerances; action replay
restores once and executes191 actions, reproducing all seven192-frame state
arrays, actions and native flags exactly. All nineteen previous successful
scored episodes plus development0 are byte-equal for those raw arrays; seed12's
old80-action/81-frame prefix is also byte-equal. Raw recording, physical gates,
source timing, initial geometric receipts and source/history identities pass
independent audit, actual exit0. All4,069 protected historical files remain
unchanged. Twenty scored subprocesses total562.398s, or2.134 accepted
trajectories/minute under that timing scope; pure planning time is uncollected.
See batch `report.md`, `protocol.json`, `summary.json` and `audit.json`, with
the reviewed patch and zero-step evidence under
`infrastructure/validation/dual-arm/near-support-fix-20261116`.


## Empty-hand path conflict and initial-wait coordination (2026-11-16)

`conflict_coordination.py` adds `delay_trajectory()` and bounded
`search_start_delays()`. It inserts a stationary source prefix and preserves
original positions, segment durations and phase gripper commands. Candidates
are checked in a deterministic order with the existing full combined checker;
checker errors propagate, and infeasible candidates are retained.

`path_conflict_fixture.py` retains one development IK middle and generates the
same analytic quintic round-trip recipe from each case's measured start.
`validate_path_conflict.py` checks each arm with its peer at the start, rejects
the colliding simultaneous combination before stepping, searches initial waits,
and invokes the unchanged `execute_dual_reach()`. A terminal-overlap negative
keeps both outbound goals occupied: all 21 delay candidates must be rejected.
No root, asset, task predicate or collision permission is changed.

Seeds21..25 have complete 115-tick records with a 30-tick right-arm wait,
84 scheduled ticks and 21 settling ticks after the 10-tick preparation.
The last20 ticks satisfy the original .03rad endpoint/.05rad/s speed bounds.
Each record contains575 physical substeps with no observed interarm contacts.
The original batch is four successes among five allocations plus one
infrastructure interruption (outer exit143); seed25 recovered in one separate
invocation with the identical method. The original incomplete artifacts remain,
and recovery is reported separately rather than rewriting the original batch.
Development seed0 is excluded. All19 earlier execution modules remain unchanged.

Protocol, sources, H5 actions/T+1 states, sampled collision records, 100Hz contact
telemetry, videos, raw audit and interruption/recovery receipts are in
`infrastructure/validation/dual-arm/path-conflict-20261116/` from the project root.
Use its frozen protocol for environment and command arguments; new executions
require new output directories. This stage covers an empty-hand controlled
route; carrying crossings, spatial replanning and handoff remain later stages.

## Actual held-object conflict and initial-wait coordination (2026-11-16)

`held_conflict_fixture.py` and `validate_held_path_conflict.py` extend the
controlled test to actual dual grasps and lifted apple/banana payloads.
Each grasp transform is captured once for collision prediction; no physical
weld or actor-pose assignment is created. The existing reach, lift and held
executors remain unchanged. Each individual carried route must be safe with
its peer held at the start, and the complete robot-only simultaneous route
must pass. Adding the captured payloads must introduce a crossarm conflict,
which the held executor rejects with zero environment steps. Bounded initial
waits are checked with the complete held-scene checker before execution.

The v3 middle joints were selected on development seed0. Formal seeds31..35
load that fixed recipe without traffic-route IK or geometry selection, use
their measured lifted q0, and preserve the source geometry, quintic time law
and CLOSED grippers while waiting. All five pass in186/185/184/182/184 ticks,
mean184.2. Each selects a15-tick right-arm wait, then executes57..61 motion
ticks,4 settling ticks and20 common airborne hover ticks within the native200
budget. These are held-traffic fixture outcomes; all official basket-placement
success flags remain false.

The100Hz raw audit covers4,605 substeps: zero actual interarm contacts and
zero forbidden physical payload contacts, with35 separated zero-impulse
interarm proximity manifolds retained. Maximum left/right position slip is
0.3064/0.0872mm; minimum carried height above the original capture is
13.665/16.206cm. The original5mm/.15rad and10cm bounds remain unchanged.

All five development attempts remain archived and excluded from the score.
Attempt03 caused a real banana-camera impulse and stopped at144 ticks;
attempt04 stopped at135 because the old monitor treated positive-gap,
zero-impulse interarm proximity as collision. The new entrypoint versions
monitoring semantics consistently for robots and objects: separation<=0 OR
any nonzero impulse means physical contact; positive-gap all-zero impulse
records remain proximity telemetry. This changes the old any-manifold
interarm abort rule. It does not modify the old monitor, FCL gates, collision
permissions, PhysX contact offsets or grasp tolerances. Attempt05 passes184
ticks using the unchanged v3 recipe and the new monitor semantics.

The two new modules, read-only diagnostic v2 and evidence auditor actually
exit0 in CPU self-tests. The formal audit also exits0, verifies all actions,
seven T+1 state groups, analytic sources, collision/search records, physical
substeps and all development failures, and preserves5,428 historical files
plus the previous41 source modules. Official checkout remains clean.
Evidence, source identities, videos and exact commands are under
`infrastructure/validation/dual-arm/held-path-conflict-20261116/`:
`report.md`, `protocol.json`, `summary.json`, `audit.json`,
`audit-run-receipt.json` and `geometry-diagnosis.json`.

This historical result covers one L0 dual-Panda controlled fixture. The final
section below reports the later three-task and spatial/handoff/recovery studies;
their results and limitations do not change this earlier frozen fixture.

## Three-task M1 and separate capability assessment (final evidence, 2026-11-16)

Evidence root: `infrastructure/validation/dual-arm/m1-multitask-20261116`.
All120 fixed M1 allocations have terminal receipts. The immutable main summary
accepts44 complete trajectories,75 physical/quality failures and one integrity
failure. A separate mathematical world-hull proof accepts that same remaining
raw episode: primary44 plus supplementary1, total45. Original summary bytes and
its `formal_evaluation_complete=false` stay unchanged. Sync results are18/20
apple-banana,16/20 pill, primary10 plus1 fruit (11/20); serial is0/20 per task
under the added common safety/closure gates. This is an end-to-end route
comparison, not unmodified official TIG success rates or isolated scheduler gain.

The120 protocol freezes seeds100–119, both methods once,669 identities and
actual initial arrays. All60 initial pairs and, with the supplement, all60
outcomes are audited. Every failure contributes to cost. Outer native-batch
wall is3781.137377s;45 accepted trajectories correspond algebraically to
42.844251/h on that original clock, excluding post-hoc supplementary audit CPU.
Ordinary task closure requires steps<200; native TimeLimit at200 wins.

The separate35-case batch also finished once per case. Independent integrity
is35/35 and source freeze passed; `capability-summary-v1.json` records
`formal_evaluation_complete=true`. The fixed per-mode results are:

| Capability/mode | Seeds | Qualified/fixed5 | Budget |
| --- | --- | --- | --- |
| Spatial detour positive |300–304|4/5|200|
| Spatial endpoint-occupied safe refusal |300–304|5/5|200|
| Handoff complete positive |310–314|0/5|400|
| Handoff receiver-failure safe hold |310–314|1/5|400|
| Grasp-failure full-task recovery |320–324|5/5|400|
| Transport-blockage full-task recovery |320–324|5/5|400|
| Unrecoverable transport safe refusal |320–324|5/5|400|

In the original35-case batch, complete handoff independent acceptance did not pass. Eight handoff episodes
stop before receiver approach at the captured-support rotation bound, while
still genuinely grasping; this is not a deep-penetration or lost-grasp claim.
Positive312 proves five receiver confirmations before giver release, one-owner
transfer and85.48mm giver retreat, then fails the20-tick final hold at238 on
pose finite-difference angular speed0.254>0.2rad/s (only2 stable ticks).
Negative312 safely closes empty at checked pregrasp and retains the giver;
other negative prefix failures do not count as successful refusals. Development
seed0 successes never replace scored failures.

Totals are14 qualified positives,11 qualified safe refusals and10 physical or
quality failures. Capability outer wall is1905.319291s; all failed costs are
retained. Controlled completion, negative refusal and full original-task
recovery are different outcomes. All35 count0 official200 task completions;
400 is an engineering capability budget. A complete evaluation does not mean
all capabilities passed. See `CAPABILITY-FORMAL-FAILURES-v1.md` and
`handoff-final-ten-diagnosis-v1.md` for the preserved limitations.

`multitask_adapters.py` supplies three-task actors/grasp/regions/predicates.
The shared public executors and `multitask_recording.py` support both M1 methods.
The original43 source modules/core remain byte-identical;13 new dual runtime
files and extra parent/audit/tools are separately identified. Only
`validate_handoff.py` differs between the two657 runtime/asset manifests.
Source support v06 uses a bounded geometry-proved measured minimum Z reference;
its1um numeric allowance is unchanged, but the reference semantics changed.
Candidates cannot update it, and permanent departure cannot reopen support.
Only sync pill/fruit use the actual-grasp geometric release formula; serial
keeps the official fixed-height route. Hidden native IK RNG is not proved
completely env-reset-seed controlled; exact initial pairs are not bitwise plans.

`M1-AND-CAPABILITIES-REPORT-v1.md` is the unified evidence report, with original
M1 report/supplement, all costs,31 current source-bound CPU receipts, raw/video
examples and actual argv. `requirements-evidence-final-v1.md` is the final
46-item delivery map. The8 older scratch analysis sources remain read-only
historical evidence, outside the current runtime/protocol/README entrypoints;
no built-in self-test is claimed for them.

Reproduction parents are evidence-root `run_case.py`, `run_formal_batch.py`,
`run_capability_case.py`, `run_capability_batch.py`. Receipts bind actual argv,
source/input snapshots, package identities and process exits. Do not resume or
rerun the already consumed allocations. Read-only re-summary uses the frozen
`summarize_capability.py`, exact bundle/batch/stage-freeze hashes and a new unused
output; the report preserves the actual executed command and execution receipt.


## Handoff repair v14 (independent assessment, 2026-11-16)

The new fixed350–354 cohort ran both modes once: complete handoff5/5 and
receiver-failure safe hold5/5. All10 independent raw/source/command-continuity
audits pass, with20 consecutive final hold ticks each. Positive steps are
319/304/319/317/317; negative steps283/267/280/279/279. This is the400-step,
L0 dual-Panda single-banana capability fixture, with no official200 basket-task
completion or cross-object/role/scene generalization claim.

The repair inherits the last issued hold commands, aligns receiver pregrasp
with its actual approach axis, and uses gentle liftoff with a fixed grasp recipe.
Transfer-only permanent source departure now requires measured height>3mm,
all convex gaps>1um, no FCL intersection and no actual native source contact.
This extends the bounded initial-support phase and changes its lifecycle
semantics; original motion/depth/contact thresholds and5/3/20 confirmation,
release and final-hold rules remain. Original43/core3 and official checkout
stay unchanged. All earlier failures remain historical.

The v12 independent cohort is4/5 per mode. The interrupted v13 cohort retains
2/5 positives,1/5 negatives, one integrity failure and six unstarted allocations;
its orphaned native success has no terminal parent receipt and is not counted.
No consumed allocation was retried or replaced. v14 changes only cohort/version
metadata from v13; new actual calibration and a new fixed protocol were used.
Each v14 parent ran through a separate tool call. Total parent wall is
768.327173s and native subprocess wall716.323008s; analysis costs are separate.

Evidence root: `infrastructure/validation/dual-arm/handoff-repair-round3-20261116`.
See [repair report](/data/jxc/ld/ZeroWAM-for-TIG/infrastructure/validation/dual-arm/handoff-repair-round3-20261116/HANDOFF-REPAIR-REPORT-v14.md),
`formal-summary-v14.json`, `protocol-v14/handoff-repair-protocol-v1.json`,
`formal-batch-receipt-v14.json`, `departure-proof-summary-v14.json` and
`preservation-and-current-cpu-v14.json`. The first fixed positive video is
`capability_acceptance/handoff-repair-s350-positive/execution.mp4`.

## cuRoboV2-Fast / TIG initial integration (2026-11-16)

Official cuRobo v0.9.0 (`42498cf4e7d0d6fbd4ef9fbcec1243c00c888a83`)
now generates trajectories for the existing TIG dual-Panda executor. The
dedicated `infrastructure/envs/curobo-tig` environment and
`infrastructure/backends/curobo` source are independent of RoboTwin.
New modules are `curobo_scene.py`, `curobo_backend.py`, `validate_curobo.py`
and the task-owned `infrastructure/curobo_tig_worker.py` process.

One unloaded paired reach in the apple/banana scene, seed401, passes full
sampled scene/paired checks and actual execution:29 total control ticks,
including10 preparation and3 settling ticks,145 physics observations,
no forbidden native physical contact. The same initial state and targets
pass the MPLib reference on the same GPU in28 ticks. This is an integration
check, not complete apple/banana placement or a backend performance benchmark.
Original public scheduling/execution/collision modules are unchanged.

The model includes actual v3 camera and finger collision shapes. A declared
derived URDF removes only the second finger's mimic relation and absolutizes
mesh paths to represent SAPIEN's two independent physical joints. Original
TIG assets stay unchanged. Full native9-DOF trajectories are retained before
selecting7 controller columns by name. TimeOpt remains enabled; CUDA graphs
are disabled for this debugging run, and torque/PhysX consistency is untested.

See the [integration report](/data/jxc/ld/ZeroWAM-for-TIG/infrastructure/validation/dual-arm/curobo-integration-20261116/CUROBO-TIG-INTEGRATION.md)
for evidence, retained development failures, scope and reproduction.
The persistent planner/state step below is now complete. Next is the
complete200-step grasp/transport/place loop with target contact permissions,
real grasp attachments, release and standard recording/replay. General
ManiSkill extraction follows cross-task TIG validation.

## Persistent cuRobo sessions (2026-11-16)

`curobo_sessions.py` and `infrastructure/curobo_tig_session_worker.py` retain
four native planners: physical left/right times fixed approach0.9/0.9 and
transport1.5/2.6 velocity/acceleration profiles. Each is initialized/warmed
once. Existing obstacle poses and independent measured finger joints update
in place; profile limits, geometry and topology cannot silently change.
Queries require the current physical snapshot version. Failed updates block
planning until a complete successful update.

`validate_curobo_sessions.py` passes one real seed401 development run:
approach/transport/approach through the unchanged shared executor, plus an
empty close/open cycle,69/200 control ticks and345 physical observations.
Four planner identities and native storage remain unchanged, with zero
forbidden physical contacts and zero simulator steps during planning.
All87 obstacles per arm are read back after updates. Independent URDF FK
checks the actual native finger collision spheres, including the closed state.
Client and native stale-snapshot rejection both pass. All three new modules'
CPU self-tests exit0.

This is unloaded state/phase integration. It does not establish actual grasp,
payload, official task completion or standard recording/replay. Native solve
RPCs take about0.47–0.51s, but per-arm model updates still take about12s due
to temporary KinematicsCfg generation; this remains a performance bottleneck.
Official sources, environments and earlier execution/collision code are
unchanged. See the
[step report](/data/jxc/ld/ZeroWAM-for-TIG/infrastructure/validation/dual-arm/curobo-sessions-20261116/CUROBO-SESSIONS.md)
and `session-summary.json` in that evidence directory for actual timings,
source identities, snapshots, native trajectories and reproduction.

## cuRobo real grasp and payload (2026-11-16)

`curobo_payload.py`, `infrastructure/curobo_tig_payload_worker.py` and
`validate_curobo_payload.py` extend the persistent sessions through physical
grasp, lift, payload transport, lowering and release. Existing shared
schedulers, executors and scene/payload collision checkers are unchanged.
The actual PhysX target convex solids become native collision attachments only
after both arms pass the existing five-tick actual grasp confirmation. Physics
continues to hold objects through finger contacts; no weld or physical collision
mask is added. Only the owning fingers receive target-contact permission.
The bounded initial tabletop departure uses the existing exact support checker;
payload/table permission is revoked after departure. Release removes native
payload spheres and restores the target's world collision geometry.

The accepted seed401 development run, `s401-dev03 --stop-after-release`, passes
in179/200 control ticks and895 physical observations, including430 held-object
observations. Both objects lift at least10cm. Maximum position slip is0.017mm
for the apple and0.741mm for the banana, within the unchanged5mm/0.15rad
slip bounds. No forbidden actual robot/target contact or interarm physical
contact is detected. Four planner identities and native storage stay fixed;
planning and state updates advance no simulation steps. The three new modules'
CPU self-tests pass, including contact ownership, stale states and solid-coverage
negative/positive cases.

The standard RecordEpisode H5 contains179 matched actions of16 dimensions and
180 actor/articulation states. The official predicate is true at release.
The run stops after release/native detach; retreat of at least8cm,20 consecutive
placement-stability ticks and standard replay remain untested. This is one
development seed with two retained earlier planning failures, not a formal
success-rate estimate or complete200-step placement acceptance.

Coarse collision spheres initially rejected valid finger/table and hand/target
geometry. The adapter now uses5mm finger coverage and tight convex-solid/cell
intersection bounds for hand and payload spheres. These remain complete solid
outer covers, not surface samples or relaxed collision permissions. Official
sources and physical assets stay unchanged. Payload mass/COM/inertia are
exported for audit but are not applied to native TimeOpt dynamics. CUDA graphs
remain off. State refresh still takes about21–24s per arm; native solve RPCs
sum to6.24s while the full cold run takes523.87s, so no overall speed gain is
established. See the
[payload report](/data/jxc/ld/ZeroWAM-for-TIG/infrastructure/validation/dual-arm/curobo-payload-20261116/CUROBO-PAYLOAD.md),
`payload-summary.json`, `cpu-self-tests.json`, `delivery-preservation.json`
and the accepted run's `execution.mp4` in that evidence directory.

### Complete cuRobo placement and actual action replay (2026-11-16)

The new `validate_curobo_complete` entry passes apple/banana seed401 in198/200
control ticks, including actual release,81.904/81.379mm retreat and20 final
stable ticks. The198 recorded actions and199 states also pass native action
replay with exactly one initial-state restore; all subsequent states result
from the original physical actions. Both source and replay videos have229
frames, including30 presentation-only padding frames. Original grasp, slip,
support and full-scene contact thresholds remain unchanged; all development
failures are retained. The MPLib reference and its action replay pass184 ticks.
These development runs used different GPUs and do not establish a paired
backend performance result.

`curobo_tig_fast_payload_worker.py` caches fixed kinematic structure while
updating measured independent fingers, world and payload permissions. Six
native snapshot comparisons preserve full readback and storage identities:
20.96–22.69s old updates versus0.118–0.166s cached updates. The complete Task
run includes extra source auditing:14 updates total3.415s, with245.586s total
native wall time,97.028s initialization and50.165s warmup reported separately.
The new conditional fitted-seed worker keeps native TrajOpt/graph/TimeOpt
fallback and validation. Payload inertia is still not applied to TimeOpt.

The fixed five-seed paired cohort is complete: MPLib4/5 and cuRobo3/5 under
unchanged qualification, with all failures retained and complete initial scenes
matched. Native held-object conflict tests also pass (cuRobo176/MPLib164 ticks),
including zero-step unsafe-pair refusal and actual safe delayed execution. Native
outbound paths use derived reversed return paths and are rechecked in full.
Pill regions and fruits on a shared plate remain in engineering validation.
See the
[complete placement report](/data/jxc/ld/ZeroWAM-for-TIG/infrastructure/validation/dual-arm/curobo-complete-20261116/CUROBO-COMPLETE.md)
and `execution-plan.json` in the same evidence directory.

### Task extension development v1/v2 (2026-11-16)

The explicit pill-only `--transport-recipe pill-direct-release-v1` keeps the
original post-lift source-departure barrier, release goal, common executor and
physical tail. cuRobo passes197 ticks and actual197-action replay. The paired
MPLib direct-route attempt stops on real pill/table contact at131 ticks; its
original native screw endpoint consumes about10mm of the requested clearance.
Raw path, retimed path and schedule endpoints agree. The older standard MPLib
pill route passes193 ticks and actual replay; these versions are kept separate.

`--payload-refinement fixed-task-v2` keeps v1 intact and changes only the right
fruit's complete solid cover from121 to603 spheres. This resolves the original
start/table false rejection: actual grasp, lift and transport reach139 ticks,
but right-arm lowering returns native NO_SOLUTION, so full cuRobo fruit
placement is still unproven. The paired MPLib fruit route and actual181-action
replay pass. New CPU evidence identifies a coarse wrist/peer-camera overlap at
the lowering start despite positive full physical-mesh clearance; native IK
diagnosis and a separate wrist refinement are pending. Source protocols,
full-process timings and all failures are preserved in the complete report.

`task_replay_alias.py` binds semantic target roles through hash-checked saved
geometry to actual native H5 actor keys. The native keys, frame0-only restore,
actions and physical gates stay unchanged. Source actions/states/videos and
the accepted pill/fruit reference replays retain exact temporal alignment.

### Fruit complete development v4 (2026-11-17)

The explicit `--payload-refinement fixed-task-v4` reuses the actual PhysX
geometry exporter and gives both fruit arms a fixed tight25mm full-solid
link7 cover (64 spheres each). Existing v2 payloads (194/603), all other links,
world geometry, masks, limits, four resident planners and the common physical
executor remain unchanged. The preceding349-sphere v3 CUDA OOM and all earlier
failed attempts are preserved under their original source/protocol versions.

The new80-source paired protocol passes MPLib181 and cuRobo196 control ticks.
cuRobo completes lowering154, release161, retreat176 and final stability196,
with actual retreat81.482/81.444mm and all980 physical rows checked. Its actual
196-action replay restores only frame0; all197 recorded native state frames
match, physical acceptance passes and227 video frames align. Whole-process
times are60.49s/435.77s; these development runs do not establish a cuRobo speed
advantage or a heldout success rate. See the complete report and final item5
audit under `infrastructure/validation/dual-arm/curobo-complete-20261116`.
