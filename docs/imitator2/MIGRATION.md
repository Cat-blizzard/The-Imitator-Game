# From `main` (1.0) to `imitator2.0`

## What moved

| 1.0 location | 2.0 location |
|---|---|
| `examples/baselines/lerobot_dataset/` (IG-10K LeRobot conversion, loaders, configs, task mapping, task descriptions, mask generation) | `imitator/dataset/` |
| `examples/baselines/lerobot_dataset/evaluate_processor.py` | `imitator/eval/processor.py` |
| `examples/baselines/lerobot_dataset/trajectory_metrics.py` (TSS / nDTW) | `imitator/eval/trajectory_metrics.py` |
| `examples/baselines/lerobot_dataset/eval/*.txt` (evaluation lists) | `imitator/eval/lists/` |
| `examples/baselines/diffusion_policy/diffusion_policy/make_env.py` | `imitator/eval/make_env.py` |
| model-independent parts of `examples/baselines/diffusion_policy/eval_dp_imitator.py` (rollout loop, L0–L3 handling, result bookkeeping) | `imitator/eval/rollout.py` (logic unchanged) + `imitator/eval/run.py` (CLI with `--policy module:factory`) + `imitator/eval/policy.py` (interface) |
| — | `examples/policy_template/`: minimal evaluation interface for any policy (the baseline template stays in `examples/baselines/_template/`) |
| `python -m examples.baselines.lerobot_dataset.h5_to_lerobot` | `python -m imitator.dataset.h5_to_lerobot` |

## What was removed

Only `examples/tutorials/` (the stock ManiSkill quick-start notebook with saved outputs, 17 MB; it is upstream
ManiSkill material) and `mani_skill/examples/motionplanning/widowxai/test_simple_motion.py` (a stale test of a stock
robot that puts `/home/ManiSkill` on `sys.path`).

The policy baselines (`examples/baselines/`: ACT, Diffusion Policy, VQ-BeT, UniSkill, XSkill, GR00T, RDT, π0.5,
OpenVLA-OFT, HyperICL, shared encoders, launch scripts, HaMeR / WiLoR hand estimation), their loaders
(`imitator/dataset/agibot/`, `lerobot_xskill_task_dataset.py`, `lerobot_self_prompt_dataset.py`), the GR00T and
real-robot perception scripts in `scripts/`, and their dependencies (extra `baselines`) are part of this branch. Their
code is unchanged from `main` except the import paths of the moved dataset / evaluation modules (and the template
baseline's import, which pointed at a non-existent `examples.baselines.template`).

## What was added

- `imitator/` (the 2.0 platform layer, formerly `imsim`): dual-arm embodiments, contract-judged tasks, data engine, verification, cloth bridge, tactile pads, Newton camera. See `PLATFORM.md`.
- `veritask/` (success contracts and their audit tools). See `VERITASK.md`.
- `tests/` (`tests/imitator`, `tests/veritask`).

## Changes to the fork itself (`mani_skill/`)

`mani_skill/envs/sapien_env.py` (and `sim2real_env.py`) read `controller.sets_target_qpos/qvel`, which
the fork's controllers never define (they were added in ManiSkill 3.0.1). Single-robot GPU simulation therefore raised
AttributeError; it now defaults to applying both targets. Multi-robot envs (all 1.0 tasks) never took this branch, so
their behaviour is unchanged.

The optional original-task bridge also adds explicit embodiment selection to
the pill and apple/banana environments and synchronous dispatch to the original
motion-planning runner. Omitted selection keeps the Panda serial defaults.
Original reward and success predicates are preserved. Robot definitions remain
in the 2.0 registry; the new module contains task-specific mounts and grasps.
Native recording, export, replay and human-video policy evaluation are described
in [ORIGINAL_TASKS.md](ORIGINAL_TASKS.md).

## How `imsim` was ported to the fork (ManiSkill 3.0.0b21 instead of 3.0.1)

| difference | handling |
|---|---|
| `parse_backend_device_id` does not exist in the fork | the render-backend patch is skipped (`imitator/compat.py`) |
| the fork's `PDJointPosMimicControllerConfig` has no `mimic` map | ManiSkill 3.0.1's mimic controller is vendored (`imitator/embodiments/controllers.py`) |
| the fork's `TableSceneBuilder` is the RoboTwin-style table (top at 0.725 m, needs a render system) | the single-arm Panda tasks use ManiSkill 3.0.1's stock builder (`imitator/envs/stock_table.py`) |
| the fork's `PandaArmMotionPlanningSolver` has a different API, `base_motionplanner` is missing | ManiSkill 3.0.1's planner stack is vendored (`imitator/planning/ms_planner/`) |
| the fork's Panda has no `tcp_pose` | added to the cloth Panda agent |
| the fork's Panda `rest` keyframe has joint 7 = -π/4 (ManiSkill 3.0.1: +π/4; fingers rotated 90° about the vertical axis) | the cloth Panda agent (`panda_cloth`) keeps the 3.0.1 rest pose; the topological tasks set it through the stock table builder |
| `import mani_skill.utils.x as y` resolves `mani_skill.utils` to the tabletop `utils` package in the fork | `from ... import` forms are used instead |
| the fork has no stock tabletop tasks (PickCube, ...) | the ten Panda wrappers of those tasks register only if the stock tasks exist; their tests skip |
| robot meshes were absolute paths into a local research folder | the needed source files are vendored in `imitator/assets/embodiments/_src/` (licenses in `NOTICE.md`), URDF paths are relative |
| hard-coded project paths | `IMITATOR_OUT`, `IMITATOR_SAPIEN_HOME`, `MS_ASSET_DIR` (`imitator/paths.py`) |

Parity check (same scripted behaviours and seeds, contract verdicts compared episode by episode, ManiSkill 3.0.1 +
imsim vs this branch): DualCubePlace on all four gripper embodiments, DualCubePlace `hold`, DualHandover, DualStack,
DualDrawer, DualLiftBox, DualPegInHole, RingOnPeg (`expert`, `held`), MugOnHook (`expert`, `on_tip`) on CPU; FoldTowel
and DualFoldTowel (Newton cloth, GPU) and batched GPU DualCubePlace (4 envs x 4 seeds). All verdicts identical
(47 episode verdicts). GPU cloth is not run-to-run deterministic on either platform, so FoldTowel was also compared as
a distribution: 4 seeds x 2 repeats x 4 worlds, expert accepted 28/32 (ManiSkill 3.0.1) vs 31/32 (fork).

## real2sim (ReSimPair)

`real2sim/resimpair/` vendors the ReSimPair pipeline as an optional feature (`python -m imitator.real2sim ...`,
`pip install -e ".[real2sim]"`); see `real2sim/README.md` for what was copied, what was changed and what was run.
