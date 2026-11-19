# Optional cuRobo planning backend

The synchronous dual-Panda pill workflow can select MPLib or cuRobo while
keeping the same environment, executor, quality configuration, recording,
action replay, and 20 Hz dataset converter. MPLib remains the default. The
cuRobo route currently supports `TwoRobotPickPillToRegions-v1`, one environment,
CPU physics, and absolute `pd_joint_pos` arm control. Planning uses an NVIDIA
CUDA GPU. The ordinary TIG rendering requirements still apply.

## Install the optional planner

Keep the normal TIG installation for simulation. The planner can run in a
separate Python environment, selected with `--curobo-python`. This adapter
targets **cuRobo 0.9.0** and its public `MotionPlanner` API. For the required
driver and CUDA packages, follow the [versioned cuRobo installation guide](https://github.com/NVlabs/curobo/blob/v0.9.0/docs/getting-started/installation.rst).

For example, prepare a separate cuRobo environment with the CUDA 12 extras:

```bash
git clone --branch v0.9.0 --depth 1 https://github.com/NVlabs/curobo.git
cd curobo
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install '.[cu12-torch]'
python -c "import curobo; assert curobo.__version__ == '0.9.0'"
```

Use the upstream CUDA 13 extras instead when appropriate for the installed
driver. Return to TIG and its simulation environment before collecting data.
The selected planner interpreter must retain its cuRobo dependencies.

## Select the backend

From TIG, with its usual robot assets installed:

```bash
python -m mani_skill.examples.motionplanning.dual.two_robot_run \
  -e TwoRobotPickPillToRegions-v1 --execution-mode sync \
  --planner-backend curobo \
  --curobo-python /path/to/curobo/.venv/bin/python \
  --curobo-device cuda:0 \
  --seed 401 -n 1 -b cpu -o rgbd --save-video \
  --record-dir demos --traj-name curobo-sync-pill-401
```

| Option | Meaning |
| --- | --- |
| `--planner-backend mplib` | Default synchronous MPLib planner; needs no cuRobo installation. |
| `--planner-backend curobo` | Optional cuRobo planner; requires `--execution-mode sync`. |
| `--curobo-python PYTHON` | Planner interpreter. Resolution is this flag, then `TIG_CUROBO_PYTHON`, then the current interpreter. |
| `--curobo-device DEVICE` | CUDA device inside the worker; defaults to `cuda:0`, relative to `CUDA_VISIBLE_DEVICES` when set. |
| `--curobo-cache-dir DIR` | Optional writable directory for prepared robot assets and compiled CUDA kernels. |

Explicit `--curobo-*` arguments require the cuRobo backend. A cuRobo environment
variable alone does not change the default MPLib route. Invalid mode/backend
combinations are rejected before an environment is created. Backend failures
are reported through the selected route; they do not silently select MPLib.

## Planner lifetime and reuse

With `-n 1`, each arm reuses one approach planner for pregrasp/descend, then
retires it and constructs a transport planner with measured held-object
geometry. At most two planners remain resident. A direct call to `solve` uses
the same lifetime unless the caller supplies a shared `CuroboSession`.

For `-n 2` or more, or `--only-count-success`, the collection process shares one
session across attempts. It retains at most two planners: one approach profile
and one transport profile. The two Panda arms use each profile sequentially
while generating their candidate paths. Before each arm's query, its complete
base-frame world, both measured fingers, attached geometry, and contact
permissions replace the previous query's state. Both resulting paths still
execute together through the common executor. Compatible profiles reuse GPU
allocations and warmup. The complete covers of both held objects determine
reserved capacity; topology or capacity changes rebuild the affected profile.

The environment seed increments normally. A shared session keeps the first
episode's planner seed and resets native samplers before each arm query.
Metadata records `episode_seed`, `planner_seed`, and
`shared_per_query_seed_reset`; this sampling sequence differs from independent
one-episode arm planners. Reuse preserves candidate counts, motion limits and
collision acceptance. The session closes its worker on completion or errors.

When measuring reuse, compare the same planner seed and report cold startup,
scene updates, planning and total collection time separately. Warmup and
construction are paid again by each separate collection command.

## Record and reuse trajectories

The normal `RecordEpisode` output records the selected `planner_backend` and
its source description. State, action and image sampling remains 20 Hz, with
`T` actions and `T+1` observations. The same action-replay entry consumes the
recorded actions; cuRobo is not required for action replay.

```bash
python -m mani_skill.examples.motionplanning.dual.sync.replay \
  --traj-path demos/TwoRobotPickPillToRegions-v1/motionplanning/curobo-sync-pill-401.h5

python -m examples.baselines.lerobot_dataset.h5_to_lerobot \
  --input demos/TwoRobotPickPillToRegions-v1/motionplanning/curobo-sync-pill-401.h5 \
  --output-dir imitator_data --name TwoRobotPickPillToRegions-v1 \
  --fps 20 --drop-terminal-frame --no-gpu --n-jobs 1
```

See [synchronous generation](dual-arm-sync.md) for the action/observation
contract, collision and quality settings, and independent result fields.
The converted data can use the [single-task ACT training and rollout guide](../examples/baselines/act/README.md#single-task-simulation-data).

## Validation and current scope

CPU tests cover backend dispatch, interpreter selection, independent finger
updates, held-object geometry, shared-session reset and cleanup, failure
diagnostics, recording metadata, and
importing the default routes without cuRobo or CUDA initialization:

```bash
python -m pytest -q tests/motionplanning/dual_sync tests/baselines
```

The arms are planned separately and executed together, without a combined
14-DoF search or automatic recovery. The original task success rule and
200-step horizon apply to either backend.

An earlier native validation of the pill workflow at seed 401 completed in 194 of 200
control steps, including physical release, retreat and final stability. Action
replay from the recorded initial state completed the same task in 194 steps.
The default MPLib route retained its 171-step action sequence in the regression
case. These functional checks do not establish comparative success rates or
throughput advantages.

The same recording converted to 194 real training samples at 20 Hz. A one-epoch
ACT check reproduced fixed-input predictions after strict checkpoint reload,
but the policy timed out at seed 402 after 200 steps. This verifies the data,
training and rollout interfaces; a successful learned policy remains to be
demonstrated.
