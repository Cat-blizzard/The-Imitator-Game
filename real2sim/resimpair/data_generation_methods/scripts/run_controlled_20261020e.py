"""Prepare or advance one budgeted real→sim→human A/B case. No implicit paid loops."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_generation_methods.r2h_pipeline.batch_pipeline import DEFAULT_CONFIG, binding
from data_generation_methods.r2h_pipeline.common import read_json, write_json
from data_generation_methods.r2h_pipeline.controlled_experiment import (
    create_experiment, plan_next, execute_next, summarize,
)

BASE = Path("/mnt/sdg/datasets/real_sim_human_pipeline")
DEFAULT_ROOT = BASE/"controlled_20261020e"


def historical_source_cases():
    """Explicitly reuse known sources; overrides should bind repaired simulation evidence."""
    specs = [
        ("apple", "apple_bowl_transfer_891910000", 2,
         ["Move the apple from the initially occupied bowl to the initially empty bowl.",
          "Both bowls and the distractor pear stay in their initial places."]),
        ("book", "closed_book_to_open_box_891912000", 2,
         ["The book stays closed.", "The open box and distractor stay in their initial places."]),
        ("cabinet", "right_cabinet_open_only_891914000", 1,
         ["The image-left door and cabinet remain unchanged.",
          "The original multicolored cube remains inside the upper compartment and is revealed when the right door opens.",
          "The distractor pear remains unchanged."]),
    ]
    cases = []
    for short, job_id, hands, invariants in specs:
        job_dir = BASE/"batch_20261020b/jobs"/job_id
        source = job_dir/"input.json"
        job = read_json(source)
        first = (BASE/"state_conditioned_20261020c/cabinet_case/human_initial_v1.png"
                 if short == "cabinet" else job_dir/"human_initial.png")
        case = {
            "id": short, "source_record": binding(source),
            "human_source": binding(job["source_window"]["video"]["path"]),
            "human_window_frames": job["source_window"]["window_frames"],
            "human_source_fps": job["source_window"]["fps"],
            "human_window_interval": "start_inclusive_end_exclusive",
            "robot_video": binding(job["robot"]["video"]["path"]),
            "robot_trajectory": binding(job["robot"]["trajectory_path"]),
            "sim_initial": binding(job["robot"]["initial_frame"]["path"]),
            "sim_terminal": binding(job["robot"]["final_frame"]["path"]),
            "human_initial": binding(first), "maximum_visible_hands": hands,
            "task": {"initial": job["initial"], "necessary_steps": job["necessary_steps"],
                     "goal": job["final"], "invariants": invariants},
            "source_acceptance_scope": "Existing reviewed source binding, not a new physics revalidation.",
            "common_anchor_sunk_cost_provenance": (
                binding(BASE/"state_conditioned_20261020c/cabinet_case/edit_initial_v1/request_metadata.json")
                if short == "cabinet" else binding(job_dir/"edit_anchor/request_metadata.json")),
        }
        cases.append(case)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("export-sources", "prepare", "plan", "advance", "summary"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--sources", type=Path,
                        help="JSON list with immutable bindings; override to use repaired robot sources")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--case", choices=("apple", "book", "cabinet"))
    parser.add_argument("--arm", choices=("A", "B"))
    parser.add_argument("--attempt", type=int, choices=(1,2), default=1)
    parser.add_argument("--execute", action="store_true",
                        help="Required for advance; one paid stage or read-only poll at most")
    args = parser.parse_args()
    if args.operation == "export-sources":
        target = args.sources or args.root/"proposed_historical_sources.json"
        write_json(target, historical_source_cases())
        result = {"source_proposal": str(target), "new_physics_validation": False}
    elif args.operation == "prepare":
        cases = read_json(args.sources) if args.sources else historical_source_cases()
        result = create_experiment(args.root, cases, read_json(args.config))
        result = {"root": str(args.root), "cases": [c["id"] for c in cases],
                  "planned_initial_candidates": len(cases)*2, "provider_called": False,
                  "budgets": result["per_case_per_group_call_budget"]}
    elif args.operation == "summary":
        result = summarize(args.root)
        write_json(args.root/f"summary_{__import__('time').time_ns()}.json", result)
    else:
        if not args.case or not args.arm:
            parser.error("plan/advance require --case and --arm")
        if args.operation == "advance" and not args.execute:
            parser.error("advance requires --execute; use plan for no paid call")
        result = (execute_next(args.root, args.case, args.arm, attempt=args.attempt)
                  if args.operation == "advance"
                  else plan_next(args.root, args.case, args.arm, attempt=args.attempt))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
