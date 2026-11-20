# Imitator 2.0 platform layer (`imitator` package)

This is the platform layer added in Imitator 2.0 on top of the Imitator fork of ManiSkill (`mani_skill/`):

- dual-arm embodiments: ARX X5, UR5 + WSG-50, AgileX PiPER, xArm6 + Robotiq 2F-85, xArm6 + Allegro hands
  (`imitator/embodiments`, assets in `imitator/assets/embodiments`, sources vendored under `_src/` with their licenses);
- tasks whose success is judged by physical **contracts** (`veritask`, protocol v3 / v3.1 / v3.2) instead of a
  single-step flag: release, settle and retreat are run by the evaluator before the verdict (`imitator/verify`);
- soft bodies through a Newton (VBD) cloth bridge (`imitator/soft`), a Newton ray-cast camera (`imitator/render`),
  tactile pads (`imitator/sensors`);
- a data engine: contract-certified datasets, composable perturbations (DART / directed DART, physical uncertainty),
  certification of external HDF5 datasets, dataset cards (`imitator/data`);
- one entry point (`imitator.make`), one evaluation harness (`imitator.evaluate`) and a CLI (`python -m imitator`).

It was developed as `imsim` on ManiSkill 3.0.1 and ported unchanged in behaviour to the fork; the port was checked
episode by episode (see `docs/imitator2/MIGRATION.md`). Paths such as `research/results/...` that appear in docstrings
and contracts refer to the internal research archive where the experiments behind each component are recorded; they
are provenance notes and are not needed to run anything.

Environment variables: `IMITATOR_OUT` (outputs, default `./results`), `IMITATOR_SAPIEN_HOME` (optional HOME for
SAPIEN's PhysX GPU library download), `MS_ASSET_DIR` (ManiSkill assets).

## Make an env

```python
import imitator
imitator.list_tasks()        # every task: see 'Task list (E29b)' below
imitator.list_embodiments()  # ['dual_arx_x5', 'dual_piper', 'dual_ur5_wsg', 'dual_xarm6_allegro', 'dual_xarm6_robotiq', 'panda_cloth']

env = imitator.make("DualHandover-v1", embodiment="dual_ur5_wsg", num_envs=1, sim="cpu", render=None, obs="state")
obs, info = env.reset(seed=0)
u = env.unwrapped                       # the ManiSkill env; u.agent.agents = [left, right]
acts = {k: ag.home_action() for k, ag in zip(u.single_action_space.spaces, u.agent.agents)}   # pd_joint_pos
obs, rew, term, trunc, info = env.step(acts)

env = imitator.make("DualCubePlace-v1", "dual_piper", num_envs=64, sim="gpu")   # PhysX GPU, 500 Hz physics for PiPER
env = imitator.make("DualFoldTowel-v1", "dual_arx_x5", num_envs=16, sim="gpu", obs="rgb")   # cloth + Newton camera
```

- `sim="cpu"` runs one env per process. `sim="gpu"` runs N envs. The GPU physics rate comes from the embodiment (E19
  stress test): ARX X5 200 Hz, PiPER 500 Hz, the others 100 Hz.
- `render=None` picks the backend automatically:
  - `"newton"` (Newton Warp ray-cast camera, CUDA only) when the task has cloth or no Vulkan device is visible;
  - with GPU physics: `"sapien_gpu"` if a CUDA-visible Vulkan device exists, else `"newton"`;
  - with CPU physics: `"sapien_cpu"` (SAPIEN on lavapipe).

  You can also force `"none" | "sapien_cpu" | "sapien_gpu" | "newton"`.
- `obs="rgb"` adds a camera image: ManiSkill's `front_camera` for SAPIEN backends, or
  `obs["sensor_data"]["newton_camera"]["rgb"]` for Newton.

## Evaluate (contract + benchmark flag + evidence)

```python
res = imitator.evaluate("DualHandover-v1", "dual_ur5_wsg", "handover", seeds=range(8), card=True)
res = imitator.evaluate("DualCubePlace-v1", "dual_xarm6_robotiq", "hold", n_episodes=8)       # v3.1 and v3 on the same rollout
res = imitator.evaluate("DualCubePlace-v1", "dual_ur5_wsg", "expert", sim="gpu", num_envs=16)  # batched, per-env verdicts
res = imitator.evaluate("DualFoldTowel-v1", "dual_piper", "all", num_envs=64)                  # 16 worlds per behaviour
res = imitator.evaluate("DualHandover-v1", "dual_ur5_wsg", my_policy, seeds=[0])               # closed loop: action = my_policy(obs)
```

For every episode, `evaluate`:

1. runs the policy or script;
2. records the per-step traces the contract needs (substep-accumulated contact impulses over every robot link,
   enclosure, object pose, robot speed, ...);
3. runs the protocol-v3 **hold A settle**: the robot is held where it stopped until the scene is quiescent, then a
   1 s judged window;
4. runs the v3.1 **retreat aftermath** when the contract has that term;
5. scores the contract and the benchmark-style `info["success"]`;
6. writes these outputs:
   - `episodes.jsonl`;
   - `evidence_<task_id>.jsonl` (veritask `EvidenceRow`; the role is taken from the contract's counterexamples);
   - `traces/*.npz` (verdicts can be recomputed from these with `imitator.verify.v31.judge_traces`);
   - optionally `card_<task_id>.md` (veritask credibility card).

Scripted behaviours (names = the contract's counterexamples; every task added in E29b: see 'Task list (E29b)' below):

| task | behaviours | contract |
|---|---|---|
| DualCubePlace-v1 | scripted_place (`expert`), hold_without_release (`hold`), cage_no_contact (`cage`), no_op | dual_cube_place_v3.1 (+ v3) |
| DualHandover-v1 | handover (`expert`), no_handover, both_hold, drop_and_push, table_transfer, no_op | imsim_dual_handover_v2 (E29b; v1 kept) |
| DualFoldTowel-v1 | expert, no_grip, drop_midway, drag_on_ground (`all` = split over the worlds) | dual_towel_fold_v2 (E29b; v1 kept) |
| FoldTowel-v1 (E20, Panda) | expert, no_grip, drop_midway, drag_on_ground | towel_fold_diagonal_v3 |

CLI:

```bash
python -m imitator list
python -m imitator run DualHandover-v1 -e dual_piper -s handover --seed 1 --frames handover.png    # keyframes (SAPIEN on CPU)
python -m imitator eval DualHandover-v1 -e dual_arx_x5 -s no_handover --seeds 0-7 --card
python -m imitator eval DualCubePlace-v1 -e dual_ur5_wsg -s expert --sim gpu --num-envs 16
python -m imitator card dual_handover_v1 <out>/evidence_imsim_dual_handover_v1.jsonl --out card.md
```

## Task list (E29b)

This list describes the contract-registered platform tasks. Selected original
TIG scenes also work through `imitator.make` and the public `imitator.eval.run
--native` path; their original success rules remain separate from these
contracts. See [the original-task workflow](ORIGINAL_TASKS.md) for the native
embodiment, recording, replay and video-policy interfaces.

`imitator.list_tasks()` returns every task below. Each task has a frozen physical outcome contract (SHA-256 in
`tests/imitator/data/e29b_contract_manifest.json`, checked by the tests), a phase-annotated scripted expert
(`imitator.data.phase`: FREE / CONTACT per arm), designed counterexample behaviours (names = the contract's
counterexamples) and a credibility card (`research/results/e29_platform/tasks/cards/`). Expert acceptance,
counterexample rejection, the benchmark predicate's behaviour and the card level per task are in
`research/results/e29_platform/tasks/task_table.md` (E29B_REPORT.md next to it explains them).

Contracts written in E29b use the **v3.2** signals of `imitator/verify/v32.py` (protocol v3 judging, unchanged in
veritask): hold A where the robot stopped, quiescence then a 1 s window inside a **5 s** settle cap, object speeds from
**positions** (never PhysX's reported velocity), release by contact + all-link enclosure, a **retreat aftermath**
(robot backs off 3 cm with its last gripper command; the objects must not move) and **commanded vs actual gripper
opening** (`grip_cmd_gap_m`, free-space calibration in `imitator/verify/grip_calib.json`).

**Single-arm Panda tasks.** The ten tasks wrapping ManiSkill's stock tabletop tasks (PickCube ... PullCubeTool) are
registered only when the installed `mani_skill` provides those stock tasks (ManiSkill 3.0.1). The Imitator fork keeps
only the Imitator tasks, so on this branch they are not listed by `imitator.list_tasks()`. The topological tasks
(RingOnPeg-v1, MugOnHook-v1) and the Panda towel fold (FoldTowel-v1) work on the fork.

| task | skill | kind | embodiments | contract | behaviours (first = positive control) | level (task_table.md) |
|---|---|---|---|---|---|---|
| DualClothCoverBox-v1 | bimanual cloth covering of an object (Newton VBD) | soft | UR5+WSG, ARX X5, PiPER | `dual_cloth_cover_box_v2` | expert, short_fold, drag_on_ground, hold_cloth, no_grip | L0 |
| DualCubePlace-v1 | pick-and-place (either arm) | rigid | UR5+WSG, ARX X5, PiPER, xArm6+Robotiq, xArm6+Allegro | `dual_cube_place_v3.1` | scripted_place, hold_without_release, cage_no_contact, no_op | - |
| DualDrawer-v1 | bimanual stabilise-and-open (articulated drawer) | rigid | UR5+WSG, ARX X5, PiPER | `imsim_dual_drawer_v1` | expert, pull_without_stabilising, open_small, hold_handle, no_op | L1 |
| DualFoldTowel-v1 | bimanual cloth fold (Newton VBD) | soft | UR5+WSG, ARX X5, PiPER | `dual_towel_fold_v2` | expert, no_grip, drop_midway, drag_on_ground | L1 |
| DualHandover-v1 | bimanual mid-air handover | rigid | UR5+WSG, ARX X5, PiPER, xArm6+Robotiq | `imsim_dual_handover_v2` | handover, no_handover, both_hold, drop_and_push, table_transfer, no_op | L1 |
| DualLiftBox-v1 | bimanual cooperative lift (held, level) | rigid | UR5+WSG, ARX X5, PiPER, xArm6+Robotiq | `imsim_dual_lift_box_v1` | expert, single_arm_center, one_end_lift, low_lift, drop_at_end, no_op | L1 |
| DualPegInHole-v1 | bimanual in-air insertion (hold + insert) | rigid | UR5+WSG, ARX X5, xArm6+Robotiq | `imsim_dual_peg_in_hole_v1` | expert, hold_peg, peg_at_entrance, block_on_table, no_op | L1 |
| DualStack-v1 | bimanual sequential stacking (one cube per arm) | rigid | UR5+WSG, ARX X5, PiPER, xArm6+Robotiq | `imsim_dual_stack_v1` | expert, one_arm_does_both, hold_top, side_by_side, cage, no_op | L1 |
| FoldTowel-v1 | cloth diagonal fold (Newton VBD) | soft | Panda (cloth) | `towel_fold_diagonal_v3` | expert, no_grip, drop_midway, drag_on_ground | - |
| LiftPegUpright-v1 | reorient (stand upright) | rigid | Panda | `imsim_lift_peg_upright_v1` | expert, hold_upright, tilted_release, lift_flat, no_op | L1 |
| PegInsertionSide-v1 | insertion (side hole, 3 mm clearance) | rigid | Panda | `imsim_peg_insertion_side_v1` | expert, hold_inserted, cage_inserted, partial_insert, release_at_entrance, no_op | L1 |
| PickCube-v1 | pick and hold at a 3-D goal | rigid | Panda | `imsim_pick_cube_v1` | expert, drop_at_goal, place_under_goal, near_miss, no_op | L1 |
| PlaceSphere-v1 | place into a container | rigid | Panda | `imsim_place_sphere_v1` | expert, hold_in_bin, cage, beside, no_op | L1 |
| PlugCharger-v1 | insertion (tight, 0.5 mm) | rigid | Panda | `imsim_plug_charger_v1` | expert, hold_plugged, release_short, cage_plugged, no_op | L0 |
| PullCube-v1 | non-prehensile pull | rigid | Panda | `imsim_pull_cube_v1` | expert, carry, hold_above, pull_and_press, short, no_op | L1 |
| PullCubeTool-v1 | tool use (hook and pull) | rigid | Panda | `imsim_pull_cube_tool_v1` | expert, hold_tool, cage_tool, pull_short, pull_by_gripper, no_op | L0 |
| PushCube-v1 | non-prehensile push | rigid | Panda | `imsim_push_cube_v1` | expert, carry, push_and_press, short, no_op | L1 |
| StackCube-v1 | stack (place on object) | rigid | Panda | `imsim_stack_cube_v1` | expert, hold_on_top, cage, press_on_top, beside, no_op | L1 |
| StackPyramid-v1 | multi-object assembly (3-cube pyramid) | rigid | Panda | `imsim_stack_pyramid_v1` | expert, hold_top, c_on_a_only, pair_only, cage_top, no_op | L1 |

Not done: DrawTriangle-v1 (the drawing is not physical — kinematic dots spawned under the TCP — and the env cannot
run past 300 control steps, so no settle / outcome contract); DualClothFlattenCorner (attempted: the folded flap lies
2-3 mm above the base layer in this cloth model and cannot be taken alone; env kept unregistered,
`imitator/envs/dual_cloth_flatten.py`). PegInsertionSide-v1 and PullCubeTool-v1 (E27's held-out tasks) were added after
E27 finished; E27's protocol-v2 contracts do not run unchanged in imsim's v3 judge, so they have v3.2 contracts that keep
E27's geometry terms and record the E27 contract (path + prereg hash) in `source.reference`. L0 tasks are registered but do not count as done:
PlugCharger-v1 (expert 17/20), PullCubeTool-v1 (expert 17/20) and DualClothCoverBox-v1 (a designed counterexample is
accepted on 2 of 3 configs).

Run a task:

```bash
python -m imitator eval StackCube-v1 -e panda -s expert --seeds 0-4 --card          # ManiSkill Panda tasks: embodiment 'panda'
python -m imitator eval StackCube-v1 -e panda -s cage --seeds 0-4                   # a designed counterexample
```

## Add an embodiment

Follow E19_REPORT.md "How to add a new embodiment". In short:

1. Add an `ArmRecord` to `imitator/embodiments/registry.py`: URDF, joints, TCP (z = approach, y = closing axis), pads,
   gripper commands, limits, home pose.
2. Build the derived assets: `python -c "from imitator.embodiments import registry; print(registry.build_assets(['<name>'], force=True))"`
   (writes `imitator/assets/embodiments/<name>/` with mesh paths relative to the URDF; put new source files under
   `imitator/assets/embodiments/_src/` and list them in its NOTICE.md).
3. The generic agent is registered automatically.
4. Add a `DualConfig` to `imitator/embodiments/dual.py`, with `gpu_sim_freq` from the E19 stress test
   (`e19_load_tests.py <cfg> gpu 16`).
5. Add the configuration to `TaskSpec.embodiments` in `imitator/tasks.py`.

The embodiment-agnostic verifier then works without further code:

- contact over all robot links;
- v3.1 enclosure over all gripper links below the flange;
- the retreat aftermath along the TCP approach axis;
- cloth colliders and pads from the URDF (`imitator/soft/embodiment_colliders.py`, prismatic grippers).

## Add a task with a contract

1. **Env.** Subclass `DualTabletopEnv` (`imitator/envs/dual_tabletop.py`; e.g. `envs/dual_handover.py`) or write a ManiSkill
   `BaseEnv`. Register it with `@register_env`, and expose the object (`self.<obj>`, half extents) and
   `evaluate()["success"]` (the benchmark-style flag).
2. **Signals.** Write a recorder factory returning `veritask.adapters.maniskill_v3.TraceRecorderV3(env, probes, pairs)`
   (see `verify/handover.py`). Probes return floats per step. `PairV3` gives substep-accumulated contact impulses.
   Use `v31.EnclosureAll(obj, half_attr)` for the release rule. Process terms can be derived from the traces by a pure
   function (`derive`), like `handover_done`.
3. **Contract.** Write a JSON contract (schema 0.3, protocol v3; copy `contracts/dual_handover_v1.json`):
   - end-state terms on the judged window (`trace.window_max/min/disp`);
   - event terms (`trace.max`);
   - optional `aftermath.retreat_object_disp`;
   - a release block (contact + enclosure signals);
   - a settle block;
   - counterexamples (names = your scripted behaviours; `expected` true for positive controls).

   **Validate it and freeze it (sha256) before any evaluation run.** Never edit a frozen contract: write a new version.
4. **Scripts.** Each scripted behaviour is a function `f(env, **kw) -> dict(status=...)` that drives `env.step`. The
   mplib helpers are in `planning/scripted.py` and `planning/handover.py`.
5. **Register.** Add a `TaskSpec` to `imitator/tasks.py`: env id, module, embodiments, contracts, scripts, recorder,
   object attributes, optional `derive` / `reset_options` / `ground_truth`.
6. **Test.** Add synthetic-trace tests for the contract (accept / reject / unknown) to `imitator/tests/` and run the
   counterexamples: `python -m imitator eval <task> -s <counterexample> --seeds 0-7 --card`.

## Data engine (E29a): certified datasets, perturbations, deviation models, certification

Report: `research/results/e29_platform/data_engine/E29A_REPORT.md`. Code: `imitator/data/` (`engine.py`, `perturb.py`,
`devmodel.py`, `store.py`, `certify.py`, `card.py`, `policy.py`, `phase.py`).

```python
import imitator
from imsim import PerturbSpec, PhysSpec
s = imitator.generate("DualCubePlace-v1", "dual_ur5_wsg", seeds=range(50), perturb="dart", workers=8, out=DIR)
s = imitator.generate("DualCubePlace-v1", seeds=range(50), workers=8, out=DIR,
                   perturb=PerturbSpec(noise="ar1", cov="iso", sd=0.01, rho=0.9, gating="all", contact_scale=0.25,
                                       phys=PhysSpec(pose_xy_m=0.01, yaw_deg=10, friction=(0.7, 1.3), mass=(0.7, 1.3))))
s = imitator.generate("FoldTowel-v1", seeds=range(50), perturb="phys", num_envs=25, out=DIR)   # cloth: batched worlds (CUDA)
dm = imitator.fit_deviation_model("DualCubePlace-v1", "dual_ur5_wsg", policy="ckpt_final.pt", expert=MP_DIR, n=20,
                               workers=8, out="dev.json")                                     # E26 deviation model
r = imitator.calibrate_scale("DualCubePlace-v1", "dual_ur5_wsg", devmodel="dev.json", workers=8)  # E26 s_t rule
s = imitator.generate("DualCubePlace-v1", "dual_ur5_wsg", seeds=range(50), perturb=imitator.preset("dart_dir", devmodel="dev.json"))
rep = imitator.certify("trajectory.h5", "DualCubePlace-v1", workers=8)   # replay + judge any ManiSkill3 trajectory file
imitator.dataset_card(DIR)                                                # card.md + card.json
```

CLI: `python -m imitator generate TASK -e EMB --preset dart --seeds 0-49 --workers 8 --out DIR`,
`python -m imitator certify DATASET TASK --workers 8`, `python -m imitator devmodel TASK -e EMB --policy CKPT --expert DIR
--out dev.json --calibrate --workers 8`, `python -m imitator card DIR`.

**Perturbations** (`PerturbSpec`, all composable): action noise `none | white | ar1` (with `rho`, or the deviation model's
per-phase rho, clipped at 0.95); covariance `iso` (`sd` rad) or `devmodel` (per arm and phase); phase gating
`free_only` or `all` with `contact_scale` (variance factor in CONTACT); global std scale `scale` (None = the deviation
model's calibrated s_t); reset physical uncertainty `PhysSpec` (pose +-dx / +-yaw, friction and mass ranges) applied to
every dynamic non-robot actor of the scene (`TaskSpec.phys_exclude` to opt actors out; mass / friction on CPU PhysX only).
Cloth: `PhysSpec.cloth = "pose"` (default; rigid offset of each world's towel, only on tasks whose expert reads the cloth
pose, `TaskSpec.cloth_pose_aware`; otherwise skipped and logged), `"pose_mass"` (+ particle mass), or `"skip"`; cloth
friction is never perturbed (Newton's body-particle friction is global).
Presets: `mp`, `dart` (E12: white N(0, 0.03^2) rad on FREE phases only), `dart_dir` (E26: AR(1), devmodel covariance x
s_t^2, CONTACT variance x 0.25, + physical uncertainty), `phys`, and the E27 ablations `dart_hi_phys`, `dart_dir_nophys`,
`dart_dir_white`, `dart_dir_freeonly` (aliases `mp_fresh`, `dart_hi`, `phys_only`).
Noise goes only on the ARM joint targets (pd_joint_pos), never on the gripper / hand; the labels are the clean expert
actions; the evaluator's settle / aftermath steps are never part of an episode.

**Phases.** Experts annotate stages with `imitator.data.phase.set_phase(env, FREE | CONTACT, arm=i)` before the env.step
calls they describe (unannotated arms count as CONTACT). Annotated: `planning/scripted.pick_place` (DualCubePlace-v1) and
`soft/expert.run_towel` (FoldTowel-v1); see each file's docstring for the stage mapping. The phase of every step and arm
is stored with the data.

**Certification.** Each episode is judged like `imitator.evaluate` (TaskSpec.judge, else `v31.judge_rec`: settle, retreat
aftermath, verdict) with the default contract and every extra contract; kept only if all accept and the expert reported
`status == "ok"`. Rejection cause = the first failing check (`plan:<stage>`, `<contract>:<binding term>`,
`<contract>:not_released`, `<contract>:unknown:<window status>`, `error:<type>`).

**Storage** (one format; see `imitator/data/store.py`): `trajectory.h5` + `trajectory.json` in the ManiSkill3 trajectory
layout (`traj_<id>/actions` = executed actions, flat in action-space key order, i.e. `FlattenActionSpaceWrapper`;
`obs` (T+1) state; `env_states` (T+1); `success / terminated / truncated / rewards`) plus `actions_clean` (labels),
`phase` (T, n_arms; 0 FREE, 1 CONTACT), `arm_qpos` and optional `rgb` (the env's render path: SAPIEN on lavapipe for CPU
rigid tasks, the Newton camera for cloth); per-episode metadata under `episodes[i]["imsim"]` (verdicts, normalised
margins, rejection cause, benchmark flag, physical draw, noise statistics, expert log); `attempts.jsonl` (every attempt,
rejected ones included), `dataset.json`, `card.md / card.json`, optional `rejected.h5`. Read labels with
`imitator.data.store.load_episodes(DIR)` (obs before each action, clean labels). Replay: `store.reset_to_episode` (seed +
reset options, mass / friction draw, `env_states[0]`) then step `actions`; on CPU PhysX this replays exactly.

**Execution.** Rigid tasks: CPU PhysX, one env per worker process (`workers=k`, spawn). Cloth tasks: Newton needs CUDA, so
a worker runs `num_envs` worlds at once (PhysX GPU, or CPU PhysX with `num_envs=1`), each world one episode. GPU batched
generation for rigid tasks is not implemented (the batched DualCubePlace script has no phase annotation).

## Expert parameters and tuning (A2)

The scripted experts of PlugCharger, PullCubeTool, StackCube, PushCube (Panda), DualCubePlace (gripper and hand spaces),
DualDrawer, DualFoldTowel and FoldTowel read 8-15 continuous parameters (with bounds) from
`imitator/planning/params.py`. The defaults are the frozen values: with no tuned set an expert runs exactly as before A2.

```python
from imitator.planning import params as P
s = P.space_for("PlugCharger-v1", "panda"); s.describe()             # names, defaults, bounds, units
P.expert_callable("PlugCharger-v1", "panda", {"pre_wait": 10})(env)   # one expert episode with overrides
P.resolve("StackCube-v1", None, "panda")                               # defaults <- adopted tuned set of this embodiment
```

Tuned sets are stored in `imitator/planning/tuned_params.json` as `"<space>@<embodiment>": {params, adopted, source}`. An
adopted set replaces the defaults for that embodiment only, and only for the positive control. Counterexample
variants always keep the frozen parameters, so tuning cannot change a designed counterexample.

Set `IMSIM_EXPERT_PARAMS=frozen` to force the frozen defaults, or `IMSIM_EXPERT_PARAMS=tuned` to use non-adopted sets
as well. Tuning, validation and results are in `research/scripts/a2_*.py` and `research/results/a2_expert_tuning/`
(`A2_REPORT.md`).

Adopted sets (2026-11-17):
- Panda: StackCube, PushCube, PullCubeTool, PlugCharger.
- DualDrawer and DualFoldTowel on `dual_piper`.
- DualCubePlace on `dual_ur5_wsg` and `dual_xarm6_allegro`.

Two structural revisions (A2) apply to everyone:
- StackCube v1.2 verifies its grasp and retries.
- The Panda planner treats degenerate zero-length moves as done instead of crashing.

## Low-GPU settings

The full numbers are in `平台低GPU配置说明.md` (project root, E19 / E20 / E23 sections).

- **Hardware.** Only CUDA compute is needed: no RT cores, no Vulkan / graphics. With `render=None` the platform never
  asks for Vulkan on the GPU.
- **Memory.** Rigid dual-arm tasks take about 1.0-1.1 GB of GPU memory at 16-64 envs (PhysX GPU). Cloth (Newton VBD)
  adds about 0.2-0.4 GB per 64 towels.
- **Rendering on GPUs without graphics.** Use the Newton camera (`render="newton"`): a few ms per frame for all envs.
  SAPIEN on lavapipe runs on the CPU at roughly 50-120 ms per 640x480 frame (E23 measurements).
- **Without a GPU.** Rigid tasks run on CPU PhysX (1 env per process) with CPU rendering. Contracts, planning and
  evaluation are all CPU. Cloth needs CUDA.
- **Light arms on GPU physics** need a higher physics rate (applied automatically).
- **Known GPU limitation.** Robotiq passive finger joints and Allegro fingers can leave their joint limits on GPU
  physics under stress (E19).

## Tests

```bash
python -B -m pytest -q -p no:cacheprovider imitator/tests/                  # fast (E19 + E20 + platform + data engine)
IMSIM_SLOW=1 python -B -m pytest -q -p no:cacheprovider imitator/tests/     # + scripted episodes, GPU, Newton camera
```

`imitator/tests/test_a2_params.py` covers the A2 parameter spaces, the tuned-set hook, and (slow) default identity.
