# Synchronous dual-arm generation

`two_robot_run` retains the original serial solutions by default. The optional
`--execution-mode sync` currently supports **TwoRobotPickPillToRegions-v1**,
two Panda wrist-camera arms, one environment, CPU physics, and absolute
`pd_joint_pos` commands. Rendering still requires the normal TIG rendering setup.
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
Native contacts and held-object slip are sampled at 100 Hz. These are sampled
checks, not continuous collision detection or joint 14-DoF planning.

The pill workflow includes preparation, pregrasp, descend, confirmed grasp,
lift, transport to measured release targets, release, optional retreat, and
optional final stability. There is no automatic recovery or spatial replanning.

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
- `execution`: planning, budget, physical, control, or runtime failure reason.
- `quality`: selected retreat/stability outcome and evaluated flags.
- `physics`: actual forbidden contacts, source departure, and measured slip.
- `recording`: whether the caller saved the standard trajectory.

An official success is preserved when a later quality check fails. A proxy
collision refusal is a planning/execution refusal, not an observed physical
collision. Normal generation keeps compact phase summaries, actions, states,
seed, and resolved configuration; it does not archive source trees or require
per-file source hashes. Failures remain recordable; `--only-count-success`
continues to filter on the original environment success rule.

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
Action `t` pairs with observation `t` and advances it to observation `t+1`.
Sampling is 20 Hz, or 0.05 s per control step.

```bash
python -m examples.baselines.lerobot_dataset.h5_to_lerobot \
  --input demos/TwoRobotPickPillToRegions-v1/motionplanning/sync-pill-401.h5 \
  --output-dir imitator_data --fps 20 --drop-terminal-frame --no-gpu --n-jobs 1
```

`--drop-terminal-frame` produces `T` supervised rows and preserves the terminal
observation in the source H5. Without the flag, the converter retains its old
`T+1` rows with the final action duplicated. Single-task ACT now reads sampling
frequency from selected datasets' `meta/info.json` and rejects mixed rates.
The converter's original train/test split may leave a one-episode dataset's
training split empty; select episode 0 explicitly for a smoke test. Conversion
compatibility does not by itself establish policy learning performance.

## Validation

```bash
python -m pytest -q tests/motionplanning/dual_sync
```

Tests cover scheduling/order, endpoint correction, retiming, collision/support
boundaries, failed grasp/drop/slip/release, policy selection, entry dispatch,
and conversion timing. The initial integration check uses pill seed 401;
broader task/seed coverage is needed before claiming general success rates.
Tested runtime: Python 3.12, SAPIEN 3.0.3, MPLib 0.1.1, NumPy 1.26.4.
