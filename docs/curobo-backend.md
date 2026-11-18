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

The worker starts and warms the GPU planner before handling its planning
requests. Record startup, scene updates and planning separately when comparing
costs. Selecting the backend does not establish a performance advantage.
Each arm reuses its planner within a phase profile. The approach planners retire
after grasp; transport planners then use the measured attached-object geometry.
Reuse across episodes is outside this adapter's current scope.

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

The original task success rule and 200-step horizon apply to either backend.
Collision, grasp, release and selected quality checks belong to the shared
execution workflow described in [synchronous generation](dual-arm-sync.md).
The two arms are planned separately and executed together; this is not a
combined 14-DoF search.

## Use the single-task ACT baseline

The converted dataset uses the existing baseline entries. For a small interface
check, save the following as `pill-datasets.json`:

```json
[{"repo_id": "TwoRobotPickPillToRegions-v1", "root": "TwoRobotPickPillToRegions-v1", "train": "[0]"}]
```

Then train and reload the checkpoint for an actual simulation rollout:

```bash
python -m examples.baselines.act.train_act_single \
  --sub-config pill-datasets.json --sim-root imitator_data \
  --cameras zed2i --state-dim 18 --action-dim 16 \
  --obs-horizon 1 --pred-horizon 16 --batch-size 8 --num-workers 0 \
  --total-epochs 1 --save-freq-epochs 1 --no-enable-augmentation \
  --output-dir act-pill

python -m examples.baselines.act.eval_act_single \
  --checkpoint act-pill/checkpoint_epoch_0001.pt \
  --sub-config pill-datasets.json --sim-root imitator_data \
  --camera zed2i --seed 402 --num-envs 1 --num-episodes 1 \
  --max-episode-steps 200 --sim-backend physx_cpu --no-compute-dtw \
  --output-dir act-pill-eval
```

The camera selection preserves training's RGB range, saved training statistics,
18 physical joint coordinates and 16 action coordinates. Evaluation checks the
recorded frequency against the environment. One episode and one training epoch
check the interface; they do not establish a learned-policy success rate.

## Validation

CPU tests cover backend dispatch, interpreter selection, independent finger
updates, held-object geometry, failure diagnostics, recording metadata, and
importing the default routes without cuRobo or CUDA initialization:

```bash
python -m pytest -q tests/motionplanning/dual_sync tests/baselines
```

Native validation of the pill workflow at seed 401 completed in 194 of 200
control steps, including physical release, retreat and final stability. Action
replay from the recorded initial state completed the same task in 194 steps.
The default MPLib route retained its 171-step action sequence in the regression
case. These are functional checks of one initial seed, not a comparative
success-rate or throughput benchmark.

The same recording converted to 194 real training samples at 20 Hz. A one-epoch
ACT interface check performed 25 optimizer updates and reproduced fixed-input
predictions exactly after strict checkpoint reload. The resulting policy ran
for 200 steps at seed 402 but timed out without completing the task. Dataset,
training and rollout interfaces are connected; policy learning is unproven by
this small check.
