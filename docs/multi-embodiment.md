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
  --max-episode-steps 400 --fps 20 --execute-steps 1 \
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
For Panda, change the robot, dataset configuration and model dimensions to 18
state values and 16 actions. This baseline does not imply that one policy's
weights transfer between embodiments.
