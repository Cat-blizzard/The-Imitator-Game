"""Geometry wrapper revision: rejected rigs never leave an executable unified package.

The frozen v1 geometry compiler and numerical protocol are unchanged. The v1
wrapper remains historical evidence; this version produces diagnostic request
specifications for blocked rigs, which the old provider entry cannot dispatch.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path

from . import actor_geometry_candidate as geometry
from .batch_pipeline import binding, checked_binding
from .common import read_json, write_json

SCHEMA = "r2h_geometry_preparation_v2"


def prepare(report_path, output, config):
    report = geometry.validate_geometry(report_path)
    output = Path(output).resolve()
    geometry._require(not output.exists(), "Use a new immutable preparation directory")
    output.mkdir(parents=True)
    source = report["bindings"]["source_package"]["path"]
    data = geometry.unified.validate_package(source)
    data["plan"] = deepcopy(data["plan"])
    rig = read_json(report["bindings"]["rig"]["path"])
    data["plan"]["actor"]["synthetic_geometry"] = {
        "rig": rig, "geometry_preflight_report": binding(report_path),
        "is_anthropometric_or_visual_evidence": False,
        "geometry_preflight_required_before_provider": True}
    intended = geometry.unified._initial_request(source, data, config)
    diagnostic = {"schema": "r2h_non_dispatchable_geometry_request_spec_v2",
                  "purpose": "Review intended initial-only conditioning; not a provider request/package",
                  "source_image": intended["source_image"], "model": intended["model"],
                  "initial_conditions": intended["initial_conditions"], "prompt": intended["prompt"],
                  "geometry_report": binding(report_path), "dispatch_allowed": False,
                  "blocking_reasons": report["reject_reasons"] + report["projection_blocks"]}
    write_json(output / "diagnostic_request_spec.json", diagnostic)
    wrapper = {"schema": SCHEMA, "geometry": binding(report_path),
               "diagnostic_request_spec": binding(output / "diagnostic_request_spec.json"),
               "implementation": binding(__file__), "status": "blocked_geometry_preflight",
               "executable_unified_package": None, "new_paid_calls": 0,
               "can_generate_video": False, "quality_release_approved": False}
    if report["clears_geometry_preflight"]:
        prepared = geometry.prepare_with_geometry(report_path, output / "qualified_preparation", config)
        wrapper["status"] = "prepared_unaccepted_anchor_candidate"
        wrapper["executable_unified_package"] = prepared["package"]
        wrapper["qualified_wrapper"] = binding(output / "qualified_preparation" / "wrapper.json")
    write_json(output / "preparation.json", wrapper)
    return wrapper


def execute(preparation_path, config, budget_path, *, credentials_path=None):
    state = read_json(preparation_path)
    geometry._require(state.get("schema") == SCHEMA, "Unknown preparation schema")
    for key in ("geometry", "diagnostic_request_spec", "implementation"):
        checked_binding(state[key])
    geometry._require(state["implementation"] == binding(__file__), "Preparation implementation changed")
    report = geometry.validate_geometry(state["geometry"]["path"])
    geometry._require(report["clears_geometry_preflight"],
                      "Geometry preflight blocked; no executable unified package was prepared")
    geometry._require(state.get("qualified_wrapper") is not None,
                      "No qualified provider wrapper")
    checked_binding(state["qualified_wrapper"])
    return geometry.execute_with_geometry(state["qualified_wrapper"]["path"], config, budget_path,
                                           credentials_path=credentials_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--report", required=True); p.add_argument("--output", required=True)
    p.add_argument("--config", required=True)
    p = sub.add_parser("execute")
    p.add_argument("--preparation", required=True); p.add_argument("--config", required=True)
    p.add_argument("--budget", required=True); p.add_argument("--credentials")
    a = parser.parse_args()
    if a.command == "prepare":
        result = prepare(a.report, a.output, read_json(a.config))
    else:
        result = execute(a.preparation, read_json(a.config), a.budget, credentials_path=a.credentials)
    print(json.dumps(result, indent=2))
