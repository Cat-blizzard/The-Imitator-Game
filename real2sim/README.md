# real2sim: ReSimPair real → sim → human pipeline (optional, experimental)

ReSimPair turns real scenes and assets into simulated Imitator-style tasks, solves them with a robot, and produces
paired synthetic human reference videos:

asset ingest and qualification → grasp generation → scene and task construction → robot execution and acceptance →
rendering → dataset export → (paid) human-video generation and VLM review.

This folder vendors the pipeline code from the ReSimPair repository so that it ships with Imitator 2.0 as an
**optional** feature. It is a first port: the code runs on this repository's ManiSkill fork, but it has not yet been
re-validated end to end here (see *Status*).

## Layout

`real2sim/resimpair/` keeps ReSimPair's own relative layout, because its modules import each other by top-level name
and its provenance checks are path based:

| folder | content |
|---|---|
| `resimpair.py`, `resimpair_runtime.py` | CLI and path setup (`ROOT` = this folder) |
| `resimpair_{assets, grasp, pickplace, tabletop, tasks, scene, validation, pp2}/` | pipeline packages (`pp2` = newest end-to-end chain) |
| `corl2026_workshop/experiments/{autonomous_pipeline_v4, task_contract_v3, state_anchor_v2}`, `corl2026_workshop/tools/audit_batch_generation_20261025.py` | core of the earlier mechanism pipeline and real-video intake (only the modules the pipeline imports) |
| `data_generation_methods/` | frozen older pipeline code (Python sources of its hash manifest) |
| `PolicyGen/examples/baselines/hypericl*` | the few data-factory helpers the pipeline imports (no policy code) |
| `PolicyGen/mani_skill` | **link to `../../../mani_skill`** (this repository's fork) instead of ReSimPair's own ManiSkill copy |
| `configs/`, `scripts/`, `third_party/` (grasp_sampler, MIT; cuDex patch), `docs/`, `tests/` | as in ReSimPair |
| `provenance/` | the small run-time provenance files (path aliases, lineages, frozen-code manifest) |

Selected with a static import closure from the CLI entry points (513 modules) plus the complete `resimpair_*`
packages. Not copied: `assets/` (1.8 GB), `validation/`, `reports/`, run outputs, the paper and figure tools, the
grasp-method benchmark, policy-training code, ReSimPair's ManiSkill copy, `private/` and the git history.

## Setup

```bash
pip install -e ".[real2sim]"        # extra deps: jsonschema, referencing, av, coacd, requests, pyyaml
python -m imitator.real2sim doctor  # environment check (also needs ffmpeg, ffprobe, nvidia-smi on PATH)
```

Data folders live inside `real2sim/resimpair/` and are git-ignored: `assets/` (asset packs: copy or link the
`assets/` folder of a ReSimPair checkout, or download the packs once they are published next to IG-10K), `runs/`
(run outputs), `runtime/` (campaign registry, relocated catalogs) and `private/` (optional credentials file).

Credentials are read from the environment (`ECHOFLOW_API_KEY` for the Gemini / Seedance relay in
`configs/providers.json`, `ARK_API_KEY` for direct Volcengine Seedance). Only the paid stages need them; asset
qualification, grasps, scene and task factories, physics, acceptance, rendering and export are local.

## Usage

All ReSimPair commands go through `python -m imitator.real2sim <command> ...` (same arguments as ReSimPair's
`scripts/run.sh`), e.g. the small pick-and-place sample of `docs/GENERAL_PICKPLACE.md`:

```bash
python -m imitator.real2sim pickplace compose --root runs/pnp_pair_demo \
  --index assets/general_pickplace_demo_v1/cache_index.json \
  --containers assets/general_pickplace_demo_v1/container_catalog.json --seed 2026102690
python -m imitator.real2sim pickplace run --root runs/pnp_pair_demo --case <case id> --limit 1
python -m imitator.real2sim pickplace audit --root runs/pnp_pair_demo
```

The newest chain: `cd real2sim/resimpair && python -m resimpair_pp2.pipeline --root runs/<name> [--config ...]`
(see `docs/PICKPLACE_V2_SPEC.md`; it reads catalog runs that must be produced first, see *Status*).

## Changes against ReSimPair

- `PolicyGen/mani_skill` points at this repository's fork. Every ManiSkill module the pipeline imports exists in the
  fork; the modules it uses are identical to ReSimPair's copy except comments, an optional `wrist_sensor` argument of
  `MultiAgent`, a smaller robot registry (the pipeline uses `panda_wristcam`) and the single-agent GPU fix.
- `corl2026_workshop/experiments/state_anchor_v2/bootstrap.py`: the frozen-code check verifies every vendored file
  byte for byte against the manifest and fails on a missing `.py`; the frozen tree's non-code files (images, logs,
  results) are not vendored and are reported as `not_vendored_non_code_files`.
- Credentials: environment variables first; the credentials file is optional (`autonomous_pipeline_v4/models.py`,
  `resimpair_pp2/seedance.py`).
- `pyproject.toml` test paths: `tests/` only; `tests/conftest.py` reports tests whose ReSimPair data (asset packs,
  run outputs, validation evidence) is absent as *skipped* with the missing path; two tests updated to the new
  policies (frozen-code count, credentials never in the repository).
- Rendering device and shader are configurable: `RESIMPAIR_RENDER_BACKEND` (e.g. `pci:0` for a CPU Vulkan renderer on
  machines whose GPU has no graphics capability) and `RESIMPAIR_SHADER` (default `rt-fast`; `default` = rasteriser,
  needed without RT cores) in the pick-and-place runners.
- `resimpair_grasp/common.py::binding()` records files behind the `PolicyGen/mani_skill` link under their
  ReSimPair-relative names, so cached grasps bound to the Panda gripper files stay valid (the files are byte-identical).
- `configs/general_pickplace_v1.json`: `"gpu": 0` (was the original host's device 2).
- `resimpair_benchmark/` (grasp-method comparison) is included because five asset modules import it lazily.

## Status (2026-11-20, this repository's fork, one RTX 4090 without graphics capability, CPU Vulkan renderer)

| check | result |
|---|---|
| ReSimPair test suite on the port | 970 passed, 205 skipped (data not in the repository), 4 failed. The same 4 also fail when the original ReSimPair checkout (its own ManiSkill copy) is tested on this machine (there: 990 passed, 78 failed, 61 errors, 50 skipped, mostly missing local data or GPU rendering). No failure is new in the port |
| pick-and-place sample (`docs/GENERAL_PICKPLACE.md`) | `init`: 6 tasks registered, 5 eligible; `compose`: 5 same-scene tasks |
| one task executed and audited | `bulk_6e4ad25c3ebb1be8c80ea523_general_L0_container_0` (move an object into a container among 5 distractors): executed in 413 s, ReSimPair's own audit **passed** (goal position / support / stable / clear, no robot collision, object held during transport, kinematics, geometry, two-camera synchronisation, guarded paths, payload inside the container floor) |
| not run here | the paid stages (human-video generation, VLM review), the newest chain `resimpair_pp2` (needs earlier catalog runs), grasp generation for new assets (GraspGenX / cuDex), full-library asset ingest |

Practical notes from the run:
- `ffmpeg` and `ffprobe` must be on `PATH` (the audit's video check calls `ffprobe`).
- With a CPU renderer a task takes ~7 min; the sample config's 360 s per-attempt limit is too short there (two runs
  timed out while placing). Use a local copy of the config with a larger `per_attempt_timeout_seconds` (`init
  --config ...`) or a GPU with graphics capability.
- Run outputs must not be edited: the pipeline binds code and asset hashes into every campaign; changing a vendored
  file invalidates existing campaigns (re-create them).
