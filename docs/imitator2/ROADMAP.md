# Imitator 2.0 roadmap

Goal (like RoboTwin 1.0 → 2.0): more embodiments, more tasks, better rendering, agent-facing tooling, a
real-to-sim-to-human pipeline (ReSimPair), a streamlined workflow and a larger benchmark.

## Status of this branch (first step: one codebase)

| area | in this branch | next |
|---|---|---|
| code base | one repository and one install for simulation (fork), benchmark tasks, evaluation, dataset tools, platform layer (`imitator`), contracts (`veritask`); model code removed (kept on `main`) | publish `pip install` extras; CI running `pytest tests` |
| embodiments | 1.0: dual Panda (`panda_wristcam`), Realman tasks. 2.0 layer: dual ARX X5, AgileX PiPER, UR5 + WSG-50, xArm6 + Robotiq 2F-85, xArm6 + Allegro (L/R hands), with actuator calibration cards from teleoperation logs (`imitator/embodiments/calibration`) | run the 1.0 tasks on the new embodiments (the tasks currently hard-code `panda_wristcam`); wire the fork's `ur10e_shadow` / `realman_inspire`; Cobot Magic (aloha-agilex) |
| tasks | 1.0: 50 base tasks (L0–L2) + 50 L3 tasks with motion-planning solutions. 2.0 layer: 8 dual-arm tasks (pick-place, handover, cooperative lift, stacking, in-air peg insertion, drawer, cloth fold, cloth cover), 2 topological tasks (ring on peg, mug on hook), Panda towel fold — each with a frozen contract, scripted expert and counterexamples | contracts for the 1.0 tasks (three exist: press_juicer, place_cup_plate, pour_kettle); task generation through the agent tooling |
| success criteria | contracts judged after the evaluator releases, settles and retreats; metamorphic tests; static checks of success code | add "persistent success" (continue / release / walk-away) next to `success_once` in `imitator.eval` |
| data | IG-10K conversion and loaders; data engine with certified datasets, DART / directed DART, physical uncertainty, certification of external HDF5 | connect the data engine to the 1.0 tasks; LeRobot export of certified datasets |
| rendering | SAPIEN (incl. `rt-fast`), Newton ray-cast camera for cloth / no-Vulkan machines | quality tier (e.g. Filament through MuJoCo 4) evaluated on the Imitator scenes |
| soft bodies | Newton VBD cloth bridge (route A), MuJoCo-Warp route B | deterministic replay, two-way finger coupling, static-friction anchors, rope / bag / sponge |
| agent tooling | `veritask.ai` (model drafts contracts and counterexamples, simulation decides), CLI for every step | white-box / black-box tool layer, clause-level diagnostics, disturbance library |
| real2sim2human | ReSimPair vendored as an optional feature (`real2sim/`, `python -m imitator.real2sim`); its tests and the pick-and-place sample run on the fork (see `real2sim/README.md`) | publish its asset packs; GPU-rendered end-to-end runs; judge accepted scenes with the benchmark contracts; port the newest chain (pp2) inputs |

## ReSimPair (real2sim2human) — what the port had to handle

Findings from a read-only inventory of `fufumeichi/ReSimPair` (2026-11-19):

1. **Credentials.** `private/.env.local` is tracked in that repository's git history. Rotate the key before anything
   is copied; the port reads keys from environment variables only (`ECHOFLOW_API_KEY`, `ARK_API_KEY`) and copies files,
   not history.
2. **Size and coupling.** 2.7 GB tracked (asset packs, sqlite indices, reports, validation logs), circular imports
   between its packages, frozen-file hash checks (`doctor`, 619 hashes) that break when files move.
3. **Its own ManiSkill copy** (`PolicyGen/mani_skill`, same 3.0.0b21 base as this fork, 146 files differ, 87 one-sided;
   e.g. its dual tasks use `panda` with one zed2i camera instead of `panda_wristcam`). The port must run on this fork.
4. **The newest pipeline (pp2)** reads earlier run outputs under `runs/` (gitignored, not in the repository) and
   `assets/appearance_v2` (ignored, empty); they must be exported or regenerated.
5. **Paid services** (human video, image edit, VLM review) go through providers in `configs/providers.json`; the
   local stages (asset qualification, grasps, scene and task factories, physics, acceptance, rendering, dataset
   export) need no API.

Done (first port, 2026-11-20): `real2sim/resimpair/` (see `real2sim/README.md`). Originally proposed: a `real2sim/` subpackage with the minimal core (`resimpair_{grasp, pickplace, tasks (pp2 versions),
assets, tabletop, scene, validation, pp2}` and the `autonomous_pipeline_v4 / task_contract_v3 / state_anchor_v2`
experiments it imports), code and data roots set by environment variables, asset packs hosted on Hugging Face next to
IG-10K, and accepted scenes judged by the same contracts as the rest of the benchmark.

## Known issues

- The DualPegInHole scripted expert is not run-to-run deterministic (seed 0 failed in one of three repeats, on
  ManiSkill 3.0.1 and on the fork alike); report its acceptance rate over many seeds.
- The ten single-arm Panda wrappers of ManiSkill stock tasks need ManiSkill 3.0.1 (not in the fork).
- In the fork, `import mani_skill.utils.<x> as y` resolves `mani_skill.utils` to the tabletop `utils` package; use
  `from mani_skill.utils.<x> import ...`.
- `python -m imitator.eval.run` needs the IG-10K data; on this branch its loop is tested with stubs
  (`tests/imitator/test_eval_interface.py`), the logic is unchanged from 1.0.
