"""Bind the reviewable geometry preparation to the exact qualified request.

This revision repairs v2 outer/inner source splicing and diagnostic/request
divergence. Frozen geometry, provider budgets and independent image gates are
unchanged. A rejected rig still leaves no executable unified package.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path

from . import actor_geometry_candidate as geometry
from .batch_pipeline import binding, checked_binding
from .common import read_json, write_json

SCHEMA = "r2h_geometry_preparation_v3"
REQUEST_FIELDS = ("source_image", "model", "initial_conditions", "prompt")


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _diagnostic(report_path, report, config):
    source = report["bindings"]["source_package"]["path"]
    data = geometry.unified.validate_package(source)
    data["plan"] = deepcopy(data["plan"])
    data["plan"]["actor"]["synthetic_geometry"] = {
        "rig": read_json(report["bindings"]["rig"]["path"]),
        "geometry_preflight_report": binding(report_path),
        "is_anthropometric_or_visual_evidence": False,
        "geometry_preflight_required_before_provider": True}
    request = geometry.unified._initial_request(source, data, config)
    return {"schema": "r2h_non_dispatchable_geometry_request_spec_v3",
            "purpose": "Review intended initial-only conditioning; not a provider request/package",
            **{key: request[key] for key in REQUEST_FIELDS},
            "geometry_report": binding(report_path), "dispatch_allowed": False,
            "blocking_reasons": report["reject_reasons"] + report["projection_blocks"]}


def prepare(report_path, output, config):
    report = geometry.validate_geometry(report_path)
    output = Path(output).resolve()
    _require(not output.exists(), "Use a new immutable preparation directory")
    output.mkdir(parents=True)
    diagnostic = _diagnostic(report_path, report, config)
    write_json(output / "diagnostic_request_spec.json", diagnostic)
    state = {"schema": SCHEMA, "geometry": binding(report_path),
             "diagnostic_request_spec": binding(output / "diagnostic_request_spec.json"),
             "implementation": binding(__file__), "status": "blocked_geometry_preflight",
             "executable_unified_package": None, "actual_initial_request": None,
             "qualified_wrapper": None, "new_paid_calls": 0,
             "can_generate_video": False, "quality_release_approved": False}
    if report["clears_geometry_preflight"]:
        prepared = geometry.prepare_with_geometry(report_path, output / "qualified_preparation", config)
        state.update(status="prepared_unaccepted_anchor_candidate",
                     executable_unified_package=prepared["package"],
                     actual_initial_request=prepared["request"],
                     qualified_wrapper=binding(output / "qualified_preparation" / "wrapper.json"))
        request = read_json(prepared["request"]["path"])
        _require(all(diagnostic[key] == request[key] for key in REQUEST_FIELDS),
                 "Diagnostic does not show the exact prepared initial request")
    write_json(output / "preparation.json", state)
    return state


def validate_preparation(preparation_path, config):
    """Cheap binding checks precede fresh source/geometry recomputation and budget."""
    state = read_json(preparation_path)
    _require(set(state) == {"schema", "geometry", "diagnostic_request_spec", "implementation",
             "status", "executable_unified_package", "actual_initial_request", "qualified_wrapper",
             "new_paid_calls", "can_generate_video", "quality_release_approved"}
             and state["schema"] == SCHEMA, "Unknown or incomplete preparation schema")
    _require(type(state["new_paid_calls"]) is int and state["new_paid_calls"] == 0
             and state["can_generate_video"] is False and state["quality_release_approved"] is False,
             "A preparation cannot claim generated-image/video qualification")
    for key in ("geometry", "diagnostic_request_spec", "implementation"):
        checked_binding(state[key])
    _require(state["implementation"] == binding(__file__), "Preparation implementation changed")
    diagnostic = read_json(state["diagnostic_request_spec"]["path"])
    _require(diagnostic.get("geometry_report") == state["geometry"],
             "Diagnostic geometry differs from outer preparation")
    inner = None
    if state["qualified_wrapper"] is not None:
        for key in ("qualified_wrapper", "executable_unified_package", "actual_initial_request"):
            checked_binding(state[key])
        inner = read_json(state["qualified_wrapper"]["path"])
        _require(inner.get("schema") == "r2h_geometry_unified_wrapper_v1",
                 "Unknown qualified geometry wrapper")
        for key in ("geometry", "package", "request", "implementation"):
            checked_binding(inner[key])
        _require(inner["implementation"] == binding(geometry.__file__),
                 "Qualified geometry implementation changed")
        _require(inner["geometry"] == state["geometry"],
                 "Outer and inner geometry differ (source or rig splice)")
        _require(inner["package"] == state["executable_unified_package"],
                 "Outer executable package differs from inner package")
        _require(inner["request"] == state["actual_initial_request"],
                 "Outer actual request differs from inner request")
        _require(inner["request"] == binding(Path(inner["package"]["path"]).parent / "initial_request.json"),
                 "Actual request does not belong to the qualified package")
        request = read_json(inner["request"]["path"])
        _require(request.get("package") == inner["package"],
                 "Actual request binds another package")
        _require(all(diagnostic.get(key) == request.get(key) for key in REQUEST_FIELDS),
                 "Diagnostic source_image/model/initial_conditions/prompt differ from actual request")
    else:
        _require(state["executable_unified_package"] is None and state["actual_initial_request"] is None,
                 "Blocked preparation must not bind an executable package/request")
    report = geometry.validate_geometry(state["geometry"]["path"])
    _require(diagnostic == _diagnostic(state["geometry"]["path"], report, config),
             "Diagnostic differs from recomputed bound source, geometry, or model")
    if report["clears_geometry_preflight"]:
        _require(inner is not None and state["status"] == "prepared_unaccepted_anchor_candidate"
                 and inner["status"] == "prepared_unaccepted_anchor_candidate",
                 "Positive geometry has no matching qualified preparation")
    else:
        _require(inner is None and state["status"] == "blocked_geometry_preflight",
                 "Rejected geometry must not retain an executable package")
    return {"state": state, "report": report, "inner": inner}


def execute(preparation_path, config, budget_path, *, credentials_path=None):
    validated = validate_preparation(preparation_path, config)
    state, report = validated["state"], validated["report"]
    _require(report["clears_geometry_preflight"],
             "Geometry preflight blocked; no executable unified package was prepared")
    return geometry.execute_with_geometry(state["qualified_wrapper"]["path"], config, budget_path,
                                           credentials_path=credentials_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--report", required=True); p.add_argument("--output", required=True)
    p.add_argument("--config", required=True)
    for command in ("validate", "execute"):
        p = sub.add_parser(command)
        p.add_argument("--preparation", required=True); p.add_argument("--config", required=True)
        if command == "execute":
            p.add_argument("--budget", required=True); p.add_argument("--credentials")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.report, args.output, read_json(args.config))
    elif args.command == "execute":
        result = execute(args.preparation, read_json(args.config), args.budget, credentials_path=args.credentials)
    else:
        validated = validate_preparation(args.preparation, read_json(args.config))
        result = {"status": validated["state"]["status"], "bindings_recomputed": True, "new_paid_calls": 0}
    print(json.dumps(result, indent=2))
