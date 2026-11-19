# veritask: verifiable success for physical tasks

`veritask` answers one question: did the physical manipulation task really succeed, and how far can that verdict be
trusted? It has no simulator dependency (numpy + scipy); simulator bindings are imported lazily.

| module | content |
|---|---|
| `spec/contract.py`, `spec/contract.schema.json` | **task contract**: the physical outcomes that must hold (outcome terms + thresholds + how to measure them on a real robot + counterexample set). `verdict()` returns the decision and normalised margins; a missing signal gives `unknown`, never a silent False. Schema validation, `to_dict` / `save`. |
| `spec/semantics.py` | scoring semantics: `any` (any step, as LIBERO / RoboTwin score), `final`, `window:K`, `persist:K`; one trajectory can be rescored under each, with Wilson intervals |
| `outcomes/metrics.py`, `outcomes/topology.py` | outcome-metric registry (cloth, liquid / granular, rigid, trajectory signals) and topological primitives (winding number, disk crossings, wrap angle) |
| `adapters/base.py`, `adapters/protocol_v2.py`, `adapters/protocol_v3.py` | simulator-independent judging: measurements from traces, "released", the evaluator-run protocol (hold, settle, judged window, retreat aftermath), trace files (`.npz`) |
| `adapters/maniskill*.py` | ManiSkill / SAPIEN trace recorders (v1 / v2 / v3) and task bindings, incl. Imitator tasks (PressJuicer, PlaceCupPlate, PourKettle) |
| `metamorphic/` | metamorphic tests M0–M7 on recorded episodes (hold / release / retreat / nudge / replay), runner, statistics; adapter for the Imitator tasks (`metamorphic/adapters/imitator.py`) |
| `audit/static_lint.py`, `audit/primitives.py` | static checks of success code (conditions computed but not used, latching, no rest check) and a library of benchmark-independent "did not do the task" behaviours |
| `audit/evidence.py`, `audit/card.py`, `audit/credibility.py` | evidence ledger (one row per executed episode), credibility cards (L0 declared / L1 counterexample-tested / L2 cross-engine / L3 real robot), false-accept / false-reject rates, AUROC, cross-engine agreement |
| `calib/agreement.py` | sim-to-real agreement: Kendall τ, Spearman ρ, MMRV, PPI intervals |
| `judges/` | evaluate VLM / reward-model judges against physical ground truth; calibration (Platt), SFT export |
| `ai/` | "AI proposes, physics disposes": model backends (local / OpenAI / Claude / agent queue; paid APIs are interfaces only), predicate review, contract drafting, red-team counterexamples — every proposal is executed in simulation before it counts (design note: `AI_NATIVE.zh.md`) |
| `data/` | data engine v0 for the Imitator tasks: snapshots, perturbers, correctors (replay / mplib re-plan / cuRobo client), contract certifier |
| `tasks/*.json` | contracts of the Imitator tasks used so far (press_juicer v1–v3, place_cup_plate v1–v3, pour_kettle, towel_fold_diagonal) and the ManiSkill poke_cube references |
| `cli.py` | `python -m veritask {validate, lint, score, rescore, credibility, card, primitives, metrics, metamorphic}` |

```bash
python -m veritask validate veritask/tasks
python -m veritask lint mani_skill/envs/tasks/tabletop/dual_tasks          # static checks of the 1.0 success functions
python -m veritask score veritask/tasks/press_juicer_v3.json run.npz       # contract verdict vs the benchmark flag
python -m veritask rescore episodes.jsonl --group-by model,suite           # rescoring under several semantics
python -m veritask card veritask/tasks/press_juicer_v1.json evidence.jsonl --out card.md
```

Removed for 2.0 (kept in the research archive): the LIBERO / RoboTwin / Meta-World adapters and executors, the 326
generated LIBERO contracts and the RoboTwin contracts, the research statistics modules (`audit/trajstats`,
`datasetstats`, `humanstats`), trajectory-generation experiments (`data/fieldgen`, `humanref*`, `osg`) and the small
diffusion policy in `train/` (model code).
