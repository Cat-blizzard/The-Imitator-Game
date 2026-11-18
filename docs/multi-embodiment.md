# Fixed-base robot integration

This example adds Piper, ARX X5 and UR5 with WSG to the existing ManiSkill
agent interface. A shared pick/place task runs on five homogeneous robot pairs
with the same MPLib planning adapter, synchronous executor and recording format.
Each pair contains two instances of the selected robot.

## Current capabilities

| Agent UID | Native action / physical qpos per arm | Basic control and RGB | Expert task | Recording and action replay | LeRobot conversion | ACT policy evaluation |
| --- | ---: | --- | --- | --- | --- | --- |
| `panda` | 8 / 9 | Verified | Verified | Verified | Verified | Evaluated: 4/5 |
| `piper` | 7 / 8 | Verified | Verified | Verified | Verified | Frozen-policy regression: 1/5 after coupling correction |
| `arx_x5` | 7 / 8 | Verified | Verified | Verified | Not tested | Not tested |
| `ur5_wsg` | 7 / 8 | Verified | Verified | Verified | Not tested | Not tested |
| `xarm6_robotiq` | 7 / 12 | Verified | Verified | Verified | Not tested | Not tested |

Task verification refers to `DualArmPickPlace-v1`, a shared cube pick/place
integration task. It does not establish compatibility with all TIG tasks. The
original pill task supports Panda wrist-camera and Piper pairs through the
[original entry](dual-arm-sync.md), selecting `--robot piper` for Piper.
ARX, UR5 and xArm have not been adapted to that task. Realman
and mixed-model pairs are outside this integration's current scope.

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
Piper also uses a finite-stiffness physical tendon for its single symmetric
opening coordinate (`joint7 = joint8` in the imported coordinates). Sending the
same drive target to two independent fingers does not impose this relation.
The tendon uses ManiSkill's existing sibling-mimic construction; finger drive
force limits, friction and task acceptance settings retain their prior values.
The upstream [current gripper description](https://github.com/agilexrobotics/agx_arm_urdf/blob/main/piper/urdf/piper_with_gripper_description.xacro)
describes two fingers driven by one opening coordinate. Its meshes and coordinate
signs differ from the imported RoboTwin asset; the import does not replace that
asset with the newer geometry.

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
  tests/motionplanning/dual_sync tests/baselines/test_act_single_pipeline.py \
  tests/baselines/test_pill_native_conversion.py

python -m mani_skill.examples.embodiments.check_sync_collision \
  --robot piper --output outputs/piper-collision-check.json
```

The collision negative test overlaps only private planning models, requires
inter-robot collisions, restores the models, and verifies that the physical
environment did not move. Synchronous pill generation and action replay support
Panda wrist-camera or Piper pairs through the [original entry](dual-arm-sync.md),
with the original 200-step task horizon. The cube integration remains at 400 steps.

## Dataset conversion and policy baseline

Use a separate dataset and model for each robot pair. The shared task records
native joint-position targets, both arms' physical joint states and `base_camera`
RGB at 20 Hz. Dual Piper has 14 actions and 16 joint states; dual Panda has 16
and 18. Controller keys, joint order and rendering settings remain in the converted
dataset's `native_data_contract` and the trained checkpoint.

Create a seed file before collection with `schema_version: 1` and disjoint
`seed_splits` named `train`, `development` and `evaluation`. For example, use
1000–1029 for a bounded training pool, 2000–2004 for development and 3000–3004
for independent policy evaluation. Do not collect evaluation seeds into the
training dataset.

```bash
python scripts/collect_data.py --robot piper \
  --seed-splits path/to/seed-splits.json --split train \
  --target-episodes 20 --max-attempts 30 --demos-dir outputs/piper-raw

python scripts/collect_data.py --robot piper \
  --seed-splits path/to/seed-splits.json --split development \
  --target-episodes 3 --max-attempts 5 --demos-dir outputs/piper-raw

python -m examples.baselines.lerobot_dataset.h5_to_lerobot \
  --episode-manifest outputs/piper-raw/episode_manifest.json \
  --output-dir outputs/datasets --name dual-piper-v1 \
  --fps 20 --drop-terminal-frame --no-gpu --n-jobs 1
```

Collection retains failed attempts and resumes from its attempt log. The manifest
selects usable source episodes without moving seeds between pools. Conversion
writes `train` and `validation` episode indices to `meta/info.json` and preserves
source mapping in `meta/source_episodes.json`. Use a new dataset name for a new
version. T actions and T+1 observations become T supervised rows; the terminal
observation has no action label. Video decoding uses exact frame seeking.

Create `piper-datasets.json`:

```json
[{"repo_id": "dual-piper-v1", "root": "dual-piper-v1", "env_id": "DualArmPickPlace-v1"}]
```

Then use the existing single-task ACT entries:

```bash
python -m examples.baselines.act.train_act_single \
  --sub-config piper-datasets.json --sim-root outputs/datasets \
  --cameras base_camera --state-dim 16 --action-dim 14 \
  --batch-size 16 --num-workers 2 --max-updates 1000 \
  --output-dir outputs/piper-act

python -m examples.baselines.act.eval_act_single \
  --checkpoint path/to/checkpoint_epoch_XXXX.pt \
  --sub-config piper-datasets.json --env-id DualArmPickPlace-v1 --robot piper \
  --seeds-file outputs/piper-raw/episode_manifest.json --seed-split evaluation \
  --max-episode-steps 400 --fps 20 --execute-steps 8 \
  --save-video --output-dir outputs/piper-policy-eval
```

`training_summary.json` identifies the final checkpoint. To check a few episodes
can be fitted, use a separate dataset configuration with `train: [0]` or
`train: [0, 1]`, a finite update budget and a separate output directory. A small
fit check does not establish performance on unseen states.

Normalization is computed from the selected training rows, including when loading
validation. Evaluation uses the exact saved training statistics and strict model
reload. It renders with the expert's shader and source image resolution, then
uses the same image resize as training. Only predicted native actions drive the
robot; no expert planner or intermediate state restoration is used. Results
separate task success, timeout, grasp/release observations and failure stage.
Predictions are clipped to native controller bounds; clipped-step counts are
reported separately from execution failures.
For Panda, change the robot, dataset configuration and model dimensions to 18
state values and 16 actions. This baseline does not imply that one policy's
weights transfer between embodiments.

The initial baseline, before the Piper finger-coupling correction, used
20 training and 3 development episodes per embodiment,
with five reserved evaluation seeds (3000–3004). Each separate ACT model used
1,000 updates, the default ResNet18 and transformer configuration, a 16-action
prediction horizon, and 8 predicted actions per query. Chunk execution was
selected on development seeds; querying every control tick caused the Panda
policy to stall. Independent evaluation was run after this choice was fixed.

| Pair | Training / development rows | Independent final successes | Unsuccessful horizon runs | Failure stage |
| --- | ---: | ---: | ---: | --- |
| Panda | 4475 / 672 | 4/5 | 1/5 | Placement incomplete |
| Piper | 4402 / 658 | 4/5 | 1/5 | Right grasp not observed |

These are small-sample results on this integration task. They establish an
actual generation-to-policy-control baseline, not general success rates or
transfer to other TIG tasks. Both successful and failed policy runs were
retained. Source H5 arrays, converted rows, frame timing and train-only
normalization were checked independently; detailed runs, datasets and models
remain outside the source repository.

After correcting Piper's symmetric finger dynamics, the unchanged initial
Piper checkpoint was run on the same five evaluation seeds with the same
8-action execution setting. It succeeded in 1/5 episodes, with four timeouts.
This is a cross-version regression on previously evaluated seeds, not a fresh
independent benchmark. The earlier dataset, checkpoint and 4/5 result remain
separate historical artifacts. Regenerating Piper training data and training
a baseline for the corrected dynamics remain future work; the original pill
integration is currently verified with expert generation and action replay.
