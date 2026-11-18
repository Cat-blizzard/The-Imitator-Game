# Fixed-base robot integration

Five homogeneous robot pairs use the same MPLib planning adapter, synchronous
executor and recording format. Each pair contains two instances of the selected
robot. All five support the original pill-sorting task and a shared cube
pick/place task.

## Current capabilities

| Agent UID | Native action / physical qpos per arm | Basic control and RGB | Expert task | Recording and action replay | LeRobot conversion | ACT policy evaluation |
| --- | ---: | --- | --- | --- | --- | --- |
| `panda` | 8 / 9 | Verified | Original pill | Verified | Verified | Pill: 5/5 |
| `piper` | 7 / 8 | Verified | Original pill; baskets | Verified | Verified | Pill: 4/5 |
| `arx_x5` | 7 / 8 | Verified | Original pill | Verified | Verified | Pill: 5/5 |
| `ur5_wsg` | 7 / 8 | Verified | Original pill | Verified | Verified | Pill: 4/5 |
| `xarm6_robotiq` | 7 / 12 | Verified | Original pill | Verified | Verified | Pill: 5/5 |

The table reports original-task coverage. Each pill policy uses twenty training
episodes and five reserved independent evaluation initial states. These small
results demonstrate the interfaces; they are not general success-rate estimates
or an embodiment comparison. Cube-task usage is described below.
The original pill task supports all five pairs through
the [original entry](dual-arm-sync.md). Piper also supports the original
apple/banana basket task. Realman and mixed-model pairs are outside this
integration's current scope. Corrected Piper basket generation and original-action
replay pass on six fixed initial states; its earlier failures remain documented
separately. This is task-specific coverage, not automatic compatibility with
every TIG task.

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
physical motion while letting both fingers use one opening command. Its source
relation `q_right = -q_left` becomes `q_right = q_left` in the prepared coordinates;
a finite native tendon represents this single opening coordinate. Invalid SRDF
references to a different gripper are removed. Piper starts from an unfolded rest
configuration; its source zero configuration has a physical self-contact.
Piper also uses a finite-stiffness physical tendon for its single symmetric
opening coordinate (`joint7 = joint8` in the imported coordinates). Sending the
same drive target to two independent fingers does not impose this relation.
The tendon uses ManiSkill's existing sibling-mimic construction; finger drive
force limits, friction and task acceptance settings retain their prior values.
The upstream [current gripper description](https://github.com/agilexrobotics/agx_arm_urdf/blob/main/piper/urdf/piper_with_gripper_description.xacro)
describes two fingers driven by one opening coordinate. Its meshes and coordinate
signs differ from the imported RoboTwin asset; the import does not replace that
asset with the newer geometry.

These are simulation models, not calibrated hardware twins. The distributed
[xArm asset](https://github.com/haosulab/ManiSkill-XArm6/blob/v0.1.1/xarm6_robotiq.urdf)
inherits link2/link3 inertias whose principal values are positive
but violate the rigid-body triangle inequality. Its finger masses/inertias are
computed from collision geometry by the loader. This integration preserves that
upstream model; successful task runs do not validate hardware dynamics, torque
ratings or those source inertias.

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
  --task DualArmPickPlace-v1 --robot piper --seed 401 --output-dir outputs/piper-401

python -m mani_skill.examples.motionplanning.dual.sync.pick_place \
  --replay outputs/piper-401/trajectory.h5 --output-dir outputs/piper-401-replay
```

Choose `--robot panda`, `piper`, `arx_x5`, `ur5_wsg` or `xarm6_robotiq` to
replan the same task on that pair. `--reach-only` stops
after synchronized reaching; such an archive does not satisfy full-task replay
acceptance. Output directories must be new so earlier attempts are preserved.

Both robots start each motion phase together; the shorter trajectory holds its
endpoint and gripper command. A single executor submits the paired action once
per 20 Hz control tick. Physics runs at 100 Hz. Planning is separate from stepping,
uses named joints and verifies actual TCP endpoints. A native IK/RRTConnect path
is available when a straight screw path cannot connect a rest pose to its goal.
Combined checks include both robots, the measured scene and carried objects.

Robot profiles in `mani_skill/agents/robots/embodiment_profiles.py` provide joint
names, TCP, native gripper mapping, finger links, units and rest poses to the agents.
Task profiles in `mani_skill/envs/tasks/tabletop/dual_arm_profiles.py` specify
mounting, sampled object workspaces, grasp orientation and approach distances.
`get_task_profile(task_id, robot_uid)` resolves declared combinations; unsupported
combinations report their supported choices. `dual_pick_place.PROFILES` remains
available for existing callers. The sampling rectangles describe the configured
task arrangement, not a complete kinematic workspace certificate.
Piper and ARX use raised mounts; robot-specific grasp-height offsets account
for actual finger geometry. Piper's equivalent top-down grasp avoids placing
its finite-range wrist near the +/-pi branch boundary. xArm opts into choosing
limit-valid 2*pi-equivalent revolute goals, verified by FK before a fresh native
path is planned and retimed. Other agents retain their existing planner behavior.
The task's object definitions, success conditions and expert phases are shared.
Its integration horizon is **400 steps**. Existing TIG tasks keep their own
horizons; the original pill task remains at 200 steps.

`trajectory.h5` and its JSON metadata use `RecordEpisode`: T native actions,
T+1 observations and states, with images and video at the actual 20 Hz rate.
Joint names and per-robot action widths are recorded explicitly. `result.json`
keeps task success, execution completion and recording status separate.
Planning refusals do not establish physical infeasibility. Inspect the failed
phase and reason separately from the environment result. Unsupported task or
robot selections are rejected at the entrypoint, before planning.

The executor maps its canonical open/close command to each native controller.
xArm retains its original 0=open, 0.81=closed action and all six physical gripper
joints, including four passive joints. Its angular gripper coordinates are
labelled in radians; grasp stability uses measured finger-link translation in
the TCP frame. Saved native actions are replayed directly without remapping.

Replay restores the recorded initial state once and executes exactly the saved
actions. It checks real grasps, lift, release, retreat, final stability and measured
collisions using the recorded policy. Monitoring is sampled at control ticks;
state differences are reported separately from task acceptance. It does not
replan, add settling actions or impose recorded states on later frames.

## Regression checks

```bash
python -m pytest -q tests/agents/test_parallel_gripper.py \
  tests/agents/test_parallel_gripper_dynamics.py \
  tests/agents/test_embodiment_profiles.py tests/test_embodiment_smoke.py \
  tests/agents/test_robotiq_outer_coupling.py \
  tests/motionplanning/dual_sync tests/baselines/test_act_single_pipeline.py \
  tests/baselines/test_act_physical_events.py \
  tests/baselines/test_pill_native_conversion.py \
  tests/baselines/test_h5_episode_manifest.py \
  tests/baselines/test_pick_place_collection.py tests/baselines/test_image_statistics.py

python -m mani_skill.examples.embodiments.check_sync_collision \
  --robot piper --output outputs/piper-collision-check.json
```

The collision negative test overlaps only private planning models, requires
inter-robot collisions, restores the models, and verifies that the physical
environment did not move. Synchronous pill generation and action replay support
all five supported pairs through the [original entry](dual-arm-sync.md),
with the original 200-step task horizon. The cube integration remains at 400 steps.
The original apple/banana basket task supports a Piper pair with the same
200-step horizon. The original Panda serial solutions remain available.

## Collect and learn an original task

The bounded collector also uses the original synchronous task entry. Select the
task explicitly; omitting `--tasks` retains the shared cube-task default. Each
robot has a separate data version and policy. The original pill task uses
`zed2i` RGB, 20 Hz control and its original 200-step horizon.

Create a seed file with `schema_version: 1` and disjoint `seed_splits` named
`train`, `development` and `evaluation`. Declare enough training candidates
for the attempt limit, and keep evaluation seeds out of expert collection.

```bash
python scripts/collect_data.py --robot piper \
  --tasks TwoRobotPickPillToRegions-v1 --seed-splits path/to/seed-splits.json \
  --split train --target-episodes 20 --max-attempts 50 \
  --demos-dir outputs/piper-pill-raw-v1

python scripts/collect_data.py --robot piper \
  --tasks TwoRobotPickPillToRegions-v1 --seed-splits path/to/seed-splits.json \
  --split development --target-episodes 3 --max-attempts 5 \
  --demos-dir outputs/piper-pill-raw-v1

python -m examples.baselines.lerobot_dataset.h5_to_lerobot \
  --episode-manifest outputs/piper-pill-raw-v1/episode_manifest.json \
  --output-dir outputs/datasets --name piper-pill-v1 \
  --fps 20 --drop-terminal-frame --no-gpu --n-jobs 1
```

Collection resumes from the same attempt log and frozen configuration. Each
launched seed counts toward the attempt limit, including interrupted workers.
All failed attempts remain in their own directories. The collector reports
official outcome, execution, selected quality and recording facts separately;
exhausting the pool returns a partial-data result rather than replacing failures.
Omitting `--robot` retains the original collection script's behavior.

Create `piper-pill-datasets.json`:

```json
[{"repo_id": "piper-pill-v1", "root": "piper-pill-v1", "env_id": "TwoRobotPickPillToRegions-v1"}]
```

```bash
python -m examples.baselines.act.train_act_single \
  --sub-config piper-pill-datasets.json --sim-root outputs/datasets \
  --cameras zed2i --state-dim 16 --action-dim 14 \
  --batch-size 16 --num-workers 2 --max-updates 6000 \
  --output-dir outputs/piper-pill-act-v1

python -m examples.baselines.act.eval_act_single \
  --checkpoint path/to/checkpoint_epoch_XXXX.pt \
  --sub-config piper-pill-datasets.json --robot piper \
  --seeds-file path/to/seed-splits.json --seed-split evaluation \
  --max-episode-steps 200 --fps 20 --execute-steps 8 \
  --save-video --output-dir outputs/piper-pill-evaluation-v1
```

Use training/development runs to choose the update budget and executed chunk
length, then fix the model and settings before independent evaluation. Short
chunks can repeatedly predict an early action prefix; a completed training run
alone does not establish that a policy can finish the task. Only public camera
and recorded-order joint observations enter the policy. Training statistics are
computed from training rows and saved in the checkpoint; evaluation reloads
those exact statistics and model weights.

The validated small pill baselines use twenty training episodes per pair,
ResNet18, batch size 16, a sixteen-action prediction horizon and eight executed
actions per query. Panda/UR5/xArm use 3,000 updates; Piper/ARX use 6,000. Each
model is selected using development states before five independent initial
states are evaluated within the original 200-step horizon.

For near-constant numeric features, ACT offers an explicit
`--normalization-min-range` in each feature's native units. The default is zero
and retains existing scaling. A positive value expands narrow bounds about their
midpoint for both normalization and inversion; raw training statistics stay
unchanged. The chosen value is saved and reloaded with the checkpoint. The ARX
pill baseline uses `--normalization-min-range 0.01`; the other four use zero.

Panda pill recordings use wrist-camera Panda agents (18 states, 16 actions).
Piper, ARX and UR5 use 16 states and 14 actions. xArm retains all passive gripper
joints (24 states, 14 actions). `--robot panda` resolves to the original pill
task's Panda wrist-camera pair; evaluation otherwise reconstructs the recorded
pair automatically. This does not transfer one policy's weights across bodies.

New original-task recordings preserve task and robot profiles through
`native_data_contract` into checkpoints, including task-specific effort and
physical coupling settings. Evaluation checks settings that affect the live
environment when present; changing only expert grasp or retreat proposals does
not block policy evaluation in otherwise identical physics. Older datasets
without these fields remain readable. Source versions
with different physical settings require separate collection directories.

Policy reports preserve official success and timeout. They also report sampled
grasp transitions and whether both objects finish in their regions ungrasped.
The legacy `release_observed` field means only that both objects were once
grasped and finish ungrasped; a dropped object can satisfy it. These observations
do not establish a controlled release or continuous collision safety.

## Shared cube-task baseline

`DualArmPickPlace-v1` remains a control and executor regression task. Use the
same collection, conversion and learning entries above, with these differences:

- Omit `--tasks` or select `--tasks DualArmPickPlace-v1` when collecting.
- Use a separate raw directory, dataset name and model for this task.
- Select `--cameras base_camera` for training instead of `zed2i`.
- Evaluate with `--env-id DualArmPickPlace-v1 --max-episode-steps 400`.

Native state/action widths stay the same for each robot. Its 400-step integration
budget does not change the original TIG tasks' 200-step horizons.
Earlier cube learning runs used different physics versions, data and initial
states; their outcomes are retained with the historical artifacts. They do not
replace the original pill results in the capability table.

Images use division by 255 in both training and evaluation. Exported image
statistics use floating-point second moments to avoid uint8 square overflow.
Their aggregation covers all converted episodes; this ACT image path does not
consume them as train-only image normalization.
