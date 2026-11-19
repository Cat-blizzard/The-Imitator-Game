#!/usr/bin/env python3
"""Bridge independently qualified mesh executions into the retained human pipeline.

Creates a zero-call budget and performs the real executor's dry run. A new
explicitly budgeted generation campaign is required to submit paid requests.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resimpair_runtime import ROOT, configure
configure()
from resimpair import historical_lineage, run_frozen
from resimpair_grasp import evaluator
from resimpair_grasp.runner import verify
from corl2026_workshop.experiments.autonomous_pipeline_v4 import core, experiment, intake
from corl2026_workshop.experiments.autonomous_pipeline_v4.budget import Budget
from corl2026_workshop.experiments.autonomous_pipeline_v4.compiler import completed_contract
from corl2026_workshop.experiments.autonomous_pipeline_v4.generic_compiler import compile_bundle


def prepare(source, output, case_ids):
    source, output = source.resolve(), output.resolve()
    verify(source)
    core.require(not output.exists(), "New immutable handoff campaign required")
    cohort = core.read(source / "cohort.json")["cases"]
    selected = [c for c in cohort if c["execution_id"] in case_ids]
    core.require(len(selected) == len(set(case_ids)), "Unknown or duplicate execution")
    for case in selected:
        directory = source / "executions" / case["execution_id"]
        core.require(evaluator.evaluate(directory)["passed"], "Independent native audit must pass")
    lineage, evidence = historical_lineage()
    output.mkdir(parents=True)
    core.write(output / "experiment.json", {"schema": "resimpair_generic_mesh_human_handoff_v1", "sources": [], "cases": [],
        "api_generation_started": False, "source_coverage": "synthetic_only", "formal_pair_count": 0})
    core.write(output / "config.json", core.read(ROOT / "configs/providers.json"))
    core.write(output / "budget.json", {"schema": "resimpair_cumulative_budget_v4", "campaign_id": output.name,
        "ledger_path": str(output / "campaign_ledger"), "new_call_limits": {"image": 0, "video": 0, "review": 0},
        "ancestor_started_calls": lineage, "lineage_cumulative_limits": lineage, "ancestor_evidence": evidence,
        "max_concurrent": 1, "timeout_seconds": 600, "automatic_retry": False,
        "authorization": "Local pipeline integration only; zero paid-call budget"})
    Budget(output / "budget.json")
    experiment.freeze(output, "portable_r1")
    records = []
    for case in selected:
        directory = source / "executions" / case["execution_id"]
        bundle = directory / "execution_bundle.json"
        intake.register_planning(output, case["id"], bundle, make_reference=True)
        mapping = {"mode": "synthetic_only", "sources": [], "source_status": "missing",
                   "formal_pair_qualified": False, "exact_source_episode_equivalence": False,
                   "manual_mapping_supplied": False}
        contract = compile_bundle(bundle, mapping)
        contract["independent_mesh_qualification"] = core.bind(directory / "audit.json")
        contract["physical_execution_provenance"] = {
            "reservation": core.bind(directory / "reservation.json"),
            "experiment_and_extension_code": core.bind(source / "experiment.json"),
            "controller_snapshot_directory": str(source / "release"),
            "scope": "binds the generic controller and asset implementation in addition to the retained exporter"}
        dest = output / "compiled_by_release/portable_r1" / case["id"] / "external_export"
        core.write(dest / "contract.json", contract)
        core.write(dest / "media_validation.json", {"contract": core.bind(dest / "contract.json"),
            "cameras": {camera: core.bind(directory / "audit_frames" / camera / "packet.json")
                        for camera in ["robot_side", "opposite_high"]}, "both_fully_decoded": True})
        completed_contract(dest / "contract.json")
        # Exercise the existing frozen executor, including media validation,
        # hand correspondence, source lineage and budget checks. No mock output.
        run_frozen(output, "autonomous_pipeline_v4", ["run", "--root", output, "--case", case["id"]])
        records.append({"case": case["id"], "execution": core.bind(bundle), "contract": core.bind(dest / "contract.json"),
            "native_audit": core.bind(directory / "audit.json"), "source_coverage": "synthetic_only"})
    result = {"status": "existing_human_executor_prepared", "records": records,
              "api_calls": Budget(output / "budget.json").usage()["new_started"], "new_human_videos": 0}
    core.write(output / "handoff_result.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--case", action="append", required=True)
    a = p.parse_args(); print(json.dumps(prepare(a.source, a.output, a.case), ensure_ascii=False, indent=2))
