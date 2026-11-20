# Original TIG tasks on native embodiments

The optional native bridge reuses the original task objects, scene sampling,
cameras, success predicates and 200-step horizon. Robot definitions come from
`imitator.embodiments.registry`; task-local mounts, grasps and drive settings
live in `imitator.envs.original_task_profiles`. Omitting native selection keeps
the original Panda serial runner.

The first bridge uses one CPU PhysX environment, 20 Hz absolute joint-position
control and 100 Hz physics. A SAPIEN renderer is needed by the original table
builder, including when only state observations are requested. It does not
require Newton or the soft-body extras.

| Original task | Configured native pairs | Planner |
|---|---|---|
| `TwoRobotPickPillToRegions-v1` | Panda, Piper, ARX X5, UR5+WSG, xArm6+Robotiq | MPLib; optional cuRobo for Panda |
| `TwoRobotPickAppleBananaToBaskets-v1` | Piper | MPLib |

Each pair contains two instances of the selected arm. Allegro and other tasks
are outside this bridge. These combinations do not imply full L0-L3 benchmark
coverage or transfer of a checkpoint between physical models.

## Generate and replay

Run from this checkout with the normal TIG dependencies and task assets:

```bash
python -m imitator.planning.original_tasks \
  -e TwoRobotPickPillToRegions-v1 --robot piper \
  --seed 31000 -n 1 -b cpu -o rgbd --shader default \
  --record-dir outputs/piper --traj-name pill

python -m imitator.planning.original_tasks replay \
  --traj-path outputs/piper/TwoRobotPickPillToRegions-v1/motionplanning/pill.h5 \
  --episode-id 0 --output outputs/piper/replay.json
```

The original runner accepts the same options through `--execution-mode sync`.
The shorter arm path holds its endpoint while the other finishes; one paired
action is submitted per control tick. Planning itself does not advance physics.
This is independent arm planning with synchronous execution.

Use the public environment entry for a learned policy:

```python
import imitator
env = imitator.make("TwoRobotPickPillToRegions-v1", "dual_piper",
                    sim="cpu", render="sapien_gpu", obs="rgb")
```

Experts record official task success, execution failures, physical events,
optional retreat/stability quality and recording separately. Expert checks do
not control learned-policy actions or termination. The `--sync-config` JSON
can select the extra expert quality phases; it does not redefine task success.
`--only-count-success` retains its original success-filtering semantics.

Replay uses executed actions. It preserves an exact seeded reset instead of
rewriting the same S0, and otherwise restores the recorded initial state once.
The native model/controller/mount identity must match. Failed or partial expert
archives remain data records, but the production expert-quality replay requires
complete stage boundaries. It is not a recovery-data replay tool.

## Export native robot data

```bash
python -m imitator.dataset.native_export \
  --input outputs/piper/TwoRobotPickPillToRegions-v1/motionplanning/pill.h5 \
  --output-dir outputs/lerobot --name piper-pill
```

The exporter retains the recorded FPS and writes T training rows from T actions
and T+1 observations. It selects public robot qpos and RGB; the simulator's
object state is never used as policy proprioception. Training normalizers can
use the explicitly selected training episodes rather than whole-dataset stats.
Exporting an attempt does not certify it as a successful expert demonstration.
Use `--episodes` and an explicit training split to select the intended examples
from the separately recorded task, execution and quality outcomes.

For certified perturbation data, `actions_clean` supplies clean expert labels
and `actions` remains the executed sequence for replay. The exporter preserves
the original certification and perturbation metadata. Historical flat-state
archives without a controller/model contract need an explicit recorded
`--contract` supplement; present-day URDFs are not used to guess past physics.
New supported single-environment engine runs write that contract themselves.

## Human-video ACT

The native training entry reuses the official ACT and frozen video encoder:

```bash
python -m examples.baselines.act.train_act_native_video --help

python -m imitator.eval.run --native \
  --policy examples.baselines.act.native_video_policy:make_policy \
  --checkpoint outputs/train/checkpoint_step_006000.pt \
  --policy-kwargs '{"human_manifest":"config/human.json","hf_cache_dir":"/path/to/hf/hub"}' \
  --env-id TwoRobotPickPillToRegions-v1 --robot piper \
  --seeds 71001 --execute-steps 8 --max-episode-steps 200 \
  --output-dir outputs/evaluation --save-video
```

Training pairs each robot dataset with its task's human-video manifest and
stores camera sampling, train-only normalization, padding masks and the actual
native contract in the checkpoint. The first public adapter supports one
environment and `video_only` input. Different repositories for the same task
receive distinct rollout directories.

`prepare_for_eval`, `get_action` and `clear_cache` use the common policy
interface. Native evaluation checks model compatibility, consumes public
RGB/qpos, executes normal controller actions and records diagnostics after
stepping. It neither calls the expert nor applies the separate contract
evaluator's settle/retreat actions. Loss of grasp may be followed by recovery
within the original budget. Existing Panda evaluation remains the default when
`--native` is omitted.

Earlier policies trained with another TCP, gripper sign, inertia or controller
must retain their original physical identity. Matching network dimensions does
not make their old evaluation scores results for these native models.

## Optional cuRobo

Select the optional interpreter on the Panda pill route:

```bash
python -m imitator.planning.original_tasks \
  -e TwoRobotPickPillToRegions-v1 --robot panda \
  --planner-backend curobo --curobo-python /path/to/curobo/bin/python \
  --seed 32000 -n 2 -b cpu -o rgbd --shader default \
  --record-dir outputs/curobo --traj-name pill
```

The client uses the existing `veritask.data.curobo_server` entry with its native
JSONL profile. The legacy data-corrector socket profile is preserved. Actual
robot and peer geometry, independent finger positions and measured payloads
refresh before queries. Continuous collection reuses at most two compatible
approach/transport planners and invalidates episode state on reset. Candidate
acceptance and physics monitoring remain with the expert executor.

The existing 2.0 Robotiq agent also retains the single motor's physical coupling
between its two outer knuckles, using a finite tendon alongside ManiSkill's
within-finger four-bar constraints. The coupling identity is recorded with the
model; equal drive targets alone do not impose that mechanical relationship.

The xArm asset generator also corrects the identified ManiSkill-XArm6 v0.1.1
inertia-product convention in derived simulation and planning URDFs. Source
assets and custom model descriptions remain unchanged. Old physical-model
results must be kept separate.
