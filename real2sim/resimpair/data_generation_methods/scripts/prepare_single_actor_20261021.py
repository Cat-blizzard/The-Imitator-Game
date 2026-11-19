"""Prepare three known development sources locally; no API calls or held-out claims."""
from __future__ import annotations

import argparse
from pathlib import Path

from data_generation_methods.r2h_pipeline.batch_pipeline import binding, contract_for
from data_generation_methods.r2h_pipeline.common import read_json, write_json
from data_generation_methods.r2h_pipeline.single_actor_candidate import create_plan, prepare_anchor


def prepare(manifest_path, config_path, output):
    output = Path(output)
    config, jobs = read_json(config_path), read_json(manifest_path)["jobs"]
    records, seen = [], set()
    for job in jobs:
        family = job["task_family"]
        if family in seen:
            continue
        if family not in {"bowl", "book", "cab", "cabinet"}:
            continue
        seen.add(family)
        directory = output / job["job_id"]
        contract = contract_for(job)
        write_json(directory / "input.json", job)
        write_json(directory / "task_contract.json", contract)
        cabinet = family in {"cab", "cabinet"}
        # These placements are a proposed design after inspecting source images.
        # No pose detector, reachability solver or generated anchor has validated them.
        position = (
            {"table_side": "right", "image_region": "right edge, lower half; torso at table side",
             "facing": "towards the front of the cabinet, diagonally left into the image"}
            if cabinet else
            {"table_side": "far", "image_region": "upper center, behind the table far edge",
             "facing": "towards the tabletop and camera"}
        )
        hands = {}
        for side in ("left", "right"):
            hands[side] = {
                "owner_id": "operator_1", "anatomical_side": side,
                "initial_entry_region": (
                    f"from the visible {side} shoulder of the same right-side torso"
                    if cabinet else
                    f"from the visible {side} shoulder of the same upper-center torso"),
                "initial_role": "rest clear of every task object, visibly connected to this operator",
            }
        placement = {
            "actor": {"id": "operator_1", "body_position": position, "hands": hands},
            "event_assignments": [
                {"event_id": event["id"], "actor_id": "operator_1",
                 "hands": ["right"] if cabinet or family == "bowl" else ["left", "right"],
                 "role": event["verb"] + " " + ", ".join(event["args"])}
                for event in contract["required_events"]],
        }
        plan = create_plan(job["anchor"]["path"], directory / "task_contract.json", placement, {
            "human_source_id": job["source_window"]["id"],
            "physical_execution_id": job["robot"]["episode_id"],
            "split": job["source_window"]["split"],
        })
        plan["placement_basis"] = "explicit design after local source-image inspection; not pose or reachability GT"
        plan["research_partition"] = "development_known_historical_failures_not_holdout"
        write_json(directory / "actor_plan.json", plan)
        prepare_anchor(directory / "actor_plan.json", directory, config)
        records.append({"job_id": job["job_id"], "plan": binding(directory / "actor_plan.json"),
                        "request": binding(directory / "anchor_request.json"),
                        "anchor_generated": False})
    budget = {"schema": "r2h_single_actor_budget_v1",
              "cumulative_call_limits": {"image": 0, "video": 0, "review": 0},
              "max_concurrent": 1, "timeout_seconds": 600}
    write_json(output / "budget.json", budget)
    result = {
        "schema": "r2h_single_actor_local_preparation_v1",
        "source_manifest": binding(manifest_path), "config": binding(config_path),
        "implementation": binding(Path(__file__)),
        "records": records, "new_paid_calls": 0,
        "visual_effectiveness": "unvalidated_no_new_generated_anchors",
        "research_partition": "development_only",
        "heldout_requirement": "different human source and layout, fixed v2 rules, before promotion",
        "training_admitted": False, "quality_release_approved": False,
    }
    write_json(output / "preparation.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.manifest, args.config, args.output)
    print(f"Prepared {len(result['records'])} development cases; new paid calls: 0")
