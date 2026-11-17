# Fixed-base robot integration

This example adds Piper, ARX X5 and UR5 with WSG to the existing ManiSkill
agent interface. A shared pick/place task runs on dual Piper and dual Panda
with the same MPLib planning adapter, synchronous executor and recording format.

## Current capabilities

| Agent UID | Arm joints | Native action size | Loading, reset, motion, gripper, TCP and RGB | Shared pick/place and action replay |
| --- | ---: | ---: | --- | --- |
| `piper` | 6 | 7 | Verified | Verified with two instances |
| `arx_x5` | 6 | 7 | Verified | Pending |
| `ur5_wsg` | 6 | 7 | Verified | Pending |
| `xarm6_robotiq` | 6 | 7 | Verified using the existing agent | Pending |
| `panda` | 7 | 8 | Existing agent | Verified with two instances |

These checks establish loading, control and representative task execution.
Broader task compatibility, training and policy evaluation remain separate work.

## Prepare assets

Use the embodiment assets distributed by
[RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin). Point the import tool
at a local checkout containing `assets/embodiments/`:

```bash
python scripts/prepare_embodiment_assets.py --robotwin-root /path/to/RoboTwin
python -m mani_skill.utils.download_asset xarm6 --non-interactive
```

Prepared robots live under `$MS_ASSET_DIR/data/robots/`, or
`~/.maniskill/data/robots/` by default. The importer preserves source files and
keeps meshes outside this repository. It makes resource paths portable, adds a
fixed grasp TCP, prepares MPLib convex meshes, and retains source attribution.
Follow the asset providers' redistribution terms when distributing prepared assets.

The UR5 import reverses the left finger's coordinate and axis together, preserving
physical motion while letting both fingers use one opening command. Invalid SRDF
references to a different gripper are removed. Piper starts from an unfolded rest
configuration; its source zero configuration has a physical self-contact.

## Check loading and control

```bash
python -m mani_skill.examples.embodiments.smoke \
  --robot-uids piper arx_x5 ur5_wsg xarm6_robotiq \
  --output-dir outputs/embodiments
```

This reuses `Empty-v1`, with one CPU physics environment and RGB rendering. Each
attempt saves a small result, images, video and control trace in its own directory.
The checks exercise named arm joints, measured finger opening, TCP displacement,
holding and reset after motion. They do not require a motion planner.

## Generate and replay a shared task

```bash
python -m mani_skill.examples.motionplanning.dual.sync.pick_place \
  --robot piper --seed 401 --output-dir outputs/piper-401

python -m mani_skill.examples.motionplanning.dual.sync.pick_place \
  --replay outputs/piper-401/trajectory.h5 --output-dir outputs/piper-401-replay
```

Use `--robot panda` to run the same task logic on Panda. `--reach-only` stops
after synchronized reaching; such an archive does not satisfy full-task replay
acceptance. Output directories must be new so earlier attempts are preserved.

Both robots start each motion phase together; the shorter trajectory holds its
endpoint and gripper command. A single executor submits the paired action once
per 20 Hz control tick. Physics runs at 100 Hz. Planning is separate from stepping,
uses named joints and verifies actual TCP endpoints. A native IK/RRTConnect path
is available when a straight screw path cannot connect a rest pose to its goal.
Combined checks include both robots, the measured scene and carried objects.

The robot profiles in `mani_skill/envs/tasks/tabletop/dual_pick_place.py` specify
mounting, reachable object positions, grasp orientation and approach distances.
Piper uses raised mounts and a grasp-height offset for its finger geometry.
The task's object definitions, success conditions and expert phases are shared.
Its integration horizon is **400 steps**. Existing TIG tasks keep their own
horizons; the original pill task remains at 200 steps.

`trajectory.h5` and its JSON metadata use `RecordEpisode`: T native actions,
T+1 observations and states, with images and video at the actual 20 Hz rate.
Joint names and per-robot action widths are recorded explicitly. `result.json`
keeps task success, execution completion and recording status separate.

Replay restores the recorded initial state once and executes exactly the saved
actions. It checks real grasps, lift, release, retreat, final stability and measured
collisions using the recorded policy. Monitoring is sampled at control ticks;
state differences are reported separately from task acceptance. It does not
replan, add settling actions or impose recorded states on later frames.

## Regression checks

```bash
python -m pytest -q tests/agents/test_parallel_gripper.py \
  tests/test_embodiment_smoke.py tests/motionplanning/dual_sync

python -m mani_skill.examples.embodiments.check_sync_collision \
  --robot piper --output outputs/piper-collision-check.json
```

The collision negative test overlaps only private planning models, requires
inter-robot collisions, restores the models, and verifies that the physical
environment did not move. The existing synchronous pill generation and action
replay remain available through the [original entry](dual-arm-sync.md).
