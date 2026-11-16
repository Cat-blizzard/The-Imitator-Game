"""Report-level run results, separate from legacy acceptance decisions.

The core summary consumes JSON reports and tick rows without evaluating an
environment, reading artifacts or changing thresholds. The publication helper
also reads tick JSONL and preserves the final report if that input is damaged.
Physics describes explicit monitoring/incident facts; quality separately applies
recorded policy to measured receipts. Unknown observations remain null/unknown.
Only acceptance retains the legacy complete-run decision, including its existing
engineering and recording failure policy. No section certifies continuous safety.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
import json
import math
from pathlib import Path
from typing import Any

SCHEMA = "tig-run-results-v1"


def _flag(value: Any) -> bool | None:
    # Recorded single-env flags are booleans, 0/1, or singleton JSON arrays.
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if type(value) is bool:
        return value
    if type(value) in (int, float) and math.isfinite(value) and value in (0, 1):
        return bool(value)
    return None


def _integer(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def official_completion(evaluation: Mapping[str, Any] | None, elapsed: Any,
                        horizon: Any) -> bool | None:
    """Observed final success before the declared horizon, or unknown."""
    success = _flag(evaluation.get('success')) if isinstance(evaluation, Mapping) else None
    step, limit = _integer(elapsed), _integer(horizon)
    if success is None or step is None or limit is None or limit == 0:
        return None
    return bool(success and step < limit)


def _status(value: bool | None) -> str:
    return "unknown" if value is None else "passed" if value else "failed"


def _all(values: Sequence[bool | None]) -> bool | None:
    if any(value is False for value in values):
        return False
    return True if values and all(value is True for value in values) else None


def _any(values: Sequence[bool | None]) -> bool | None:
    if any(value is True for value in values):
        return True
    return False if values and all(value is False for value in values) else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[Mapping[str, Any]]:
    return [row for row in value if isinstance(row, Mapping)] if isinstance(value, (list, tuple)) else []


def _terminal_check(value: Any) -> bool | None:
    return True if value == "passed" else False if value in ("failed", "rejected", "aborted") else None


def _count_at_least(value: Any, minimum: int) -> bool | None:
    if isinstance(value, (list, tuple)):
        return _all([_count_at_least(item, minimum) for item in value])
    number = _integer(value)
    return None if number is None else number >= minimum


def _receipt(value: bool | None, source: str) -> dict[str, Any]:
    return {"value": value, "status": _status(value), "source": source}


def _official(report: Mapping[str, Any], ticks: list[Mapping[str, Any]]) -> dict[str, Any]:
    observations = []
    invalid_ticks = []
    for index, tick in enumerate(ticks):
        success = _flag(_mapping(tick.get("official_evaluation")).get("success"))
        step = _integer(tick.get("environment_step"))
        if success is None or step is None:
            invalid_ticks.append(index)
        observations.append({"step": step, "success": success,
                             "source": f"ticks[{index}].official_evaluation.success"})
    total = _integer(report.get("total_environment_steps"))
    horizon = _integer(report.get("max_episode_steps"))
    if horizon == 0:
        horizon = None
    raw = _mapping(report.get("official_task_evaluation"))
    final = _flag(raw.get("success"))
    final_source = "report.official_task_evaluation.success" if final is not None else None
    final_evaluation = copy.deepcopy(dict(raw)) if raw else None
    if final is None and ticks and total is not None and observations[-1]["step"] == total:
        fallback = observations[-1]
        if fallback["success"] is not None:
            final, final_source = fallback["success"], fallback["source"]
            final_evaluation = copy.deepcopy(dict(_mapping(ticks[-1].get("official_evaluation"))))
    observed = list(observations)
    if final_source == "report.official_task_evaluation.success":
        observed.append({"step": total, "success": final, "source": final_source})
    successes = [row for row in observed if row["success"] is True]
    known_steps = [row["step"] for row in successes if row["step"] is not None]
    first = min(known_steps) if known_steps else None
    prefix_complete = first is not None and len(ticks) >= first and all(
        row["step"] == index + 1 and row["success"] is not None
        for index, row in enumerate(observations[:first]))
    transitions = []
    previous = None
    for row in observations:
        current = row["success"]
        if previous is not None and current is not None and current != previous:
            transitions.append({"step": row["step"], "from": previous, "to": current})
        previous = current
    before = None if total is None or horizon is None else total < horizon
    mismatch = bool(final is not None and observations and observations[-1]["step"] == total
                    and observations[-1]["success"] is not None
                    and observations[-1]["success"] != final)
    return {"status": _status(final), "final_success": final,
            "final_evaluation": final_evaluation, "final_source": final_source,
            "final_step": total, "horizon_steps": horizon,
            "final_step_before_horizon": before,
            "final_success_before_horizon": _all([final, before]),
            "ever_observed_success": _any([row["success"] for row in observed]),
            "first_observed_success_step": first,
            "first_success_step_prefix_complete": prefix_complete,
            "success_transitions": transitions,
            "tick_observations": len(observations), "invalid_or_missing_tick_observations": invalid_ticks,
            "tick_steps_complete_to_final": total is not None and total == len(ticks)
                and not invalid_ticks and all(row["step"] == i + 1 for i, row in enumerate(observations)),
            "final_evaluation_conflicts_with_last_tick": mismatch,
            "native_truncation_observed": _any([_flag(row.get("native_truncated")) for row in ticks]),
            "native_termination_observed": _any([_flag(row.get("native_terminated")) for row in ticks]),
            "native_fail_observed": _any([_flag(_mapping(row.get("official_evaluation")).get("fail"))
                                          for row in ticks]),
            "legacy_official_200_step_complete": _flag(report.get("official_200_step_complete")),
            "legacy_flag_is_official_observation": False}


def _events(report: Mapping[str, Any], ticks: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    seen = set()

    def add(kind: str, source: str, *, phase: Any = None, step: Any = None,
            message: Any = None) -> None:
        event = {"kind": kind, "source": source}
        if isinstance(phase, str):
            event["phase"] = phase
        if _integer(step) is not None:
            event["step"] = step
        if isinstance(message, str):
            event["message"] = message
        key = json.dumps(event, sort_keys=True)
        if key not in seen:
            seen.add(key)
            events.append(event)

    def error_kind(message: str) -> str:
        text = message.lower()
        if "forbidden physical contact" in text or "forbidden actual robot/world contact" in text:
            return "physical_contact_aborted"
        if "measured_collision" in text:
            return "measured_geometry_collision_gate"
        if "candidate_collision" in text:
            return "planned_collision_rejection"
        if "initial_collision" in text:
            return "initial_geometry_collision_gate"
        if "pregrasp recipe lacks" in text and "clearance" in text:
            return "pregrasp_clearance_rejection"
        if ("budget" in text and any(word in text for word in ("insufficient", "cannot fit", "exhausted"))) \
                or "placement window does not fit" in text:
            return "budget_insufficient"
        if any(word in text for word in ("planning failed", "planner", "topp", "ik failed", "screw plan failed")):
            return "planning_failure"
        if "endpoint exceeds contract" in text:
            return "planning_endpoint_rejection"
        return "execution_error"

    if isinstance(report.get("error"), str):
        add(error_kind(report["error"]), "report.error", phase=report.get("stage"),
            message=report["error"])
    for index, row in enumerate(_rows(report.get("planning"))):
        validation = _mapping(row.get("collision_validation"))
        if validation.get("status") == "rejected":
            add("planned_collision_rejection", f"report.planning[{index}].collision_validation",
                phase=row.get("phase"), step=row.get("before_control_step"))
    for index, stage in enumerate(_rows(report.get("stages"))):
        source = f"report.stages[{index}]"
        reason, state = stage.get("reason"), stage.get("status")
        kind = {"candidate_collision": "planned_collision_rejection",
                "measured_collision": "measured_geometry_collision_gate",
                "initial_collision": "initial_geometry_collision_gate",
                "position_slip": "quality_slip_gate_failure", "rotation_slip": "quality_slip_gate_failure",
                "object_dropped": "object_dropped", "grasp_lost": "grasp_lost",
                "initial_grasp_or_attachment_invalid": "held_precondition_rejection",
                "hold_not_stable": "quality_stability_gate_failure"}.get(reason)
        if kind:
            add(kind, source + ".reason", phase=stage.get("name"), message=reason)
        if reason == "initial_grasp_or_attachment_invalid":
            # This shared reason also covers a quality slip limit. Only a
            # measured missing grasp establishes an incident on its own.
            initial_objects = _rows(_mapping(stage.get("initial_observation")).get("objects"))
            for side, obj in enumerate(initial_objects):
                if _flag(obj.get("is_grasping")) is False:
                    add("grasp_lost", source + f".initial_observation.objects[{side}].is_grasping",
                        phase=stage.get("name"))
        if state in ("grasp_timeout", "release_timeout", "held_move_timeout", "reach_timeout", "placement_timeout"):
            add("quality_completion_window_failure", source + ".status", phase=stage.get("name"), message=state)
        if isinstance(stage.get("error"), str):
            add(error_kind(stage["error"]), source + ".error", phase=stage.get("name"), message=stage["error"])
    errors = report.get("artifact_errors")
    if isinstance(errors, list):
        for index, error in enumerate(errors):
            if not isinstance(error, str):
                continue
            kind = "official_observation_error" if error.startswith("final official_task_evaluation:") else (
                "cleanup_error" if error.startswith(("worker close:", "close:")) else "recording_error")
            add(kind, f"report.artifact_errors[{index}]", message=error)
    for index, tick in enumerate(ticks):
        if _flag(tick.get("native_truncated")) is True:
            add("native_time_limit", f"ticks[{index}].native_truncated", step=tick.get("environment_step"))
    if _flag(report.get("timed_out")) is True:
        add("subprocess_timeout", "report.timed_out")
    if _flag(report.get("input_evidence_unchanged")) is False:
        add("source_evidence_changed", "report.input_evidence_unchanged")
    return events


def _quality_policy(report: Mapping[str, Any]) -> dict[str, Any]:
    # These are fallback values for legacy reports only; the executor owns
    # policy validation and records the complete resolved runtime policy.
    legacy = {"grasp_stable_steps": 5, "release_stable_steps": 3,
              "position_slip_m": .005, "rotation_slip_rad": .15,
              "minimum_retreat_m": .08, "final_stable_steps": 20,
              "min_lift_m": .10, "require_retreat": True,
              "require_final_stability": True}
    if "runtime_policy" not in report:
        return {"source": "legacy fallback: runtime_policy absent", "resolved": legacy}
    record = _mapping(report.get("runtime_policy"))
    resolved = _mapping(record.get("resolved"))
    values = {}
    for name in legacy:
        value = resolved.get(name)
        valid = type(value) is bool if name.startswith("require_") else (
            type(value) is int and value > 0 if name.endswith("_steps") else
            type(value) in (int, float) and math.isfinite(value) and value >= 0)
        values[name] = value if valid else None
    return {"source": "report.runtime_policy.resolved", "resolved": values,
            "recorded_sha256": record.get("sha256") if isinstance(record.get("sha256"), str) else None,
            "policy_validation": "executor-owned; only required scalar types checked here"}


def _quality(report: Mapping[str, Any]) -> dict[str, Any]:
    policy = _quality_policy(report)
    limits = policy["resolved"]
    checks: dict[str, dict[str, Any]] = {}
    stages = {row.get("name"): row for row in _rows(report.get("stages"))}

    def add(name: str, value: bool | None, source: str, *, required: bool | None = True) -> None:
        receipt = _receipt(value, source)
        receipt["required"] = required
        if required is False:
            receipt.update(value=None, status="not_required")
        checks[name] = receipt

    def count(name: str, field: str, threshold: Any) -> bool | None:
        if _integer(threshold) is None:
            return None
        return _count_at_least(_mapping(stages.get(name)).get(field), threshold)

    def measured_pair(values: Any, limit: Any, *, minimum: bool) -> bool | None:
        if type(limit) not in (int, float) or not math.isfinite(limit):
            return None
        if not isinstance(values, (list, tuple)) or len(values) != 2 or not all(
                type(value) in (int, float) and math.isfinite(value) for value in values):
            return None
        return all(value >= limit if minimum else value <= limit for value in values)

    replay = report.get("mode") == "action-replay"
    if replay:
        outcome = _mapping(report.get("outcome"))
        outcome_checks = _mapping(outcome.get("checks"))
        grasp = _mapping(report.get("attachment_capture")).get("grasp_stable_steps_observed")
        add("grasp_stability", None if _integer(limits["grasp_stable_steps"]) is None else
            _count_at_least(grasp, limits["grasp_stable_steps"]),
            "report.attachment_capture.grasp_stable_steps_observed")
        release = _flag(outcome_checks.get("both_physically_released_last_three")) \
            if limits["release_stable_steps"] == 3 else None
        add("release_stability", release, "report.outcome.checks.both_physically_released_last_three (3-tick receipt)")
        add("minimum_lift", None, "no replay lift-height receipt available in legacy report")
        rise = measured_pair(outcome.get("actual_retreat_tcp_rise_m"), limits["minimum_retreat_m"], minimum=True)
        add("minimum_retreat", rise, "report.outcome.actual_retreat_tcp_rise_m", required=limits["require_retreat"])
        window = _flag(outcome_checks.get("common_final20_window")) if limits["final_stable_steps"] == 20 else None
        placement = _flag(outcome_checks.get("both_final_placements_valid"))
        add("final_stability", _all([window, placement]),
            "report.outcome.checks.common_final20_window + both_final_placements_valid (20-tick receipt)",
            required=limits["require_final_stability"])
    else:
        for name, key in (("close_and_confirm", "grasp_stable_steps"), ("release", "release_stable_steps")):
            add("grasp_stability" if name == "close_and_confirm" else "release_stability",
                count(name, "stable_steps_observed", limits[key]),
                f"report.stages[name={name}].stable_steps_observed")
        lift_trace = _rows(_mapping(stages.get("lift_and_hold")).get("trace"))
        objects = _rows(lift_trace[-1].get("objects")) if lift_trace else []
        gains = [row.get("object_height_gain_m") for row in objects]
        add("minimum_lift", measured_pair(gains, limits["min_lift_m"], minimum=True),
            "report.stages[name=lift_and_hold].trace[-1].objects[*].object_height_gain_m")
        retreat = report.get("actual_retreat_displacement_m")
        rises = [row[2] for row in retreat] if isinstance(retreat, (list, tuple)) and len(retreat) == 2 \
            and all(isinstance(row, (list, tuple)) and len(row) == 3 for row in retreat) else None
        add("minimum_retreat", measured_pair(rises, limits["minimum_retreat_m"], minimum=True),
            "report.actual_retreat_displacement_m[*][2]", required=limits["require_retreat"])
        placement = _mapping(stages.get("verify_placement"))
        measured_window = count("verify_placement", "stable_steps_observed", limits["final_stable_steps"])
        # The passed placement phase includes the existing regional and velocity
        # predicates. A recorded timeout proves the selected window was not met;
        # it is a quality failure, never a physical accident.
        placement_ok = True if placement.get("status") == "passed" else False \
            if placement.get("status") == "placement_timeout" else None
        add("final_stability", _all([measured_window, placement_ok]),
            "report.stages[name=verify_placement].stable_steps_observed + status",
            required=limits["require_final_stability"])
    slip_rows = _rows(_mapping(report.get("held_physics_audit")).get("objects"))
    add("held_position_slip", measured_pair([row.get("max_position_slip_m") for row in slip_rows],
        limits["position_slip_m"], minimum=False), "report.held_physics_audit.objects[*].max_position_slip_m")
    add("held_rotation_slip", measured_pair([row.get("max_rotation_slip_rad") for row in slip_rows],
        limits["rotation_slip_rad"], minimum=False), "report.held_physics_audit.objects[*].max_rotation_slip_rad")
    required = [row["value"] if row["required"] is True else None
                for row in checks.values() if row["required"] is not False]
    return {"status": _status(_all(required)), "policy": policy, "checks": checks,
            "evidence_scope": "policy-selected quality from stage and measured audit receipts only",
            "official_acceptance_or_artifact_errors_used": False}


def _physics(report: Mapping[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    monitoring = _mapping(report.get("physical_monitoring"))
    observed = _integer(monitoring.get("observed_substeps"))
    expected = _integer(monitoring.get("expected_substeps"))
    forbidden = _integer(monitoring.get("forbidden_contact_count"))
    interarm = _integer(monitoring.get("interarm_forbidden_count"))
    robot_world = _integer(monitoring.get("robot_world_forbidden_count"))
    robot_observed = _integer(monitoring.get("robot_world_observed_substeps"))
    robot_complete = _flag(monitoring.get("robot_world_coverage_complete"))
    complete = _flag(monitoring.get("coverage_complete"))
    coverage = (complete is True and observed == expected and observed > 0) \
        if complete is not None and observed is not None and expected is not None else None
    robot_coverage = (robot_complete is True and robot_observed == expected and robot_observed > 0) \
        if robot_complete is not None and robot_observed is not None and expected is not None else None
    positive_contacts = any(value is not None and value > 0 for value in (forbidden, interarm, robot_world))
    contact_free = False if positive_contacts else True \
        if coverage is True and robot_coverage is True and forbidden == interarm == robot_world == 0 else None
    facts = [copy.deepcopy(row) for row in events if row["kind"] in (
        "physical_contact_aborted", "grasp_lost", "object_dropped", "held_model_invalid", "release_not_detached")]
    # Release timeout by itself is only an incomplete window. Actual measured
    # continued grasp at its final sample establishes the not-released fact.
    for index, stage in enumerate(_rows(report.get("stages"))):
        if stage.get("name") != "release" or stage.get("status") != "release_timeout":
            continue
        trace = _rows(stage.get("trace"))
        objects = _rows(trace[-1].get("objects")) if trace else []
        if any(_flag(row.get("is_grasping")) is True for row in objects):
            facts.append({"kind": "release_not_detached", "source":
                f"report.stages[{index}].trace[-1].objects[*].is_grasping", "phase": "release"})
    if any(row["kind"] == "physical_contact_aborted" for row in facts):
        contact_free = False
    value = False if facts or positive_contacts else contact_free
    return {"status": _status(value), "facts": facts,
            "contact_free": _receipt(contact_free, "report.physical_monitoring or explicit physical-contact abort"),
            "monitoring": {"observed_substeps": observed, "expected_substeps": expected,
                "forbidden_contact_count": forbidden, "interarm_forbidden_count": interarm,
                "robot_world_forbidden_count": robot_world,
                "robot_world_observed_substeps": robot_observed,
                "robot_world_coverage_complete": robot_complete,
                "positive_complete_robot_world_coverage": robot_coverage,
                "coverage_complete": complete, "positive_complete_coverage": coverage},
            "legacy_closure": _flag(report.get("project_complete_physical_closure")),
            "legacy_replay_contact_check": _flag(_mapping(report.get("checks")).get("no_forbidden100Hz_contacts")),
            "evidence_scope": "explicit physical-monitoring counters and incident receipts; no raw stream reread",
            "legacy_status_quality_or_recording_used_to_infer_physics": False}


def _recording(report: Mapping[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    recording = _mapping(report.get("recording"))
    steps = _integer(report.get("total_environment_steps"))
    count = _integer(recording.get("recorded_steps"))
    matched = _flag(recording.get("submitted_actions_matched"))
    shape = recording.get("actions_shape")
    shape_ok = None if steps is None or shape is None else shape == [steps, 16]
    lengths = recording.get("state_lengths")
    lengths_ok = None
    if steps is not None and isinstance(lengths, Mapping) and lengths:
        values = [_integer(value) for value in lengths.values()]
        lengths_ok = all(value == steps + 1 for value in values) if all(value is not None for value in values) else None
    checks = {"submitted_actions_match": _receipt(matched, "report.recording.submitted_actions_matched"),
              "action_shape": _receipt(shape_ok, "report.recording.actions_shape"),
              "T_plus_one_states": _receipt(lengths_ok, "report.recording.state_lengths"),
              "recorded_step_count": _receipt(None if count is None or steps is None else count == steps,
                                                "report.recording.recorded_steps + report.total_environment_steps")}
    for field in ("recording_actions_and_Tplus1_verified", "input_evidence_unchanged"):
        if field in report:
            checks[field] = _receipt(_flag(report.get(field)), f"report.{field}")
    for field in ("standard_recording_timing", "video_timing"):
        row = _mapping(report.get(field))
        if row:
            checks[field] = _receipt(_terminal_check(row.get("status")), f"report.{field}.status")
    errors = [event for event in events if event["kind"] == "recording_error"]
    value = False if errors else _all([check["value"] for check in checks.values()])
    return {"status": _status(value), "checks": checks, "errors": copy.deepcopy(errors),
            "video_reported_written": isinstance(report.get("video"), Mapping),
            "evidence_scope": "existing report receipts only; H5/video/files not read"}


def summarize_run(report: Mapping[str, Any], ticks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return independent JSON sections without changing either input.

    Legacy status remains the phase-1 input in ``acceptance``. It never
    supplies an official observation or a physics conclusion. This function
    does not backfill/re-score a historical report or inspect external files.
    """
    if not isinstance(report, Mapping) or not isinstance(ticks, (list, tuple)) \
            or any(not isinstance(row, Mapping) for row in ticks):
        raise TypeError("Expected a report mapping and JSON tick mappings")
    tick_rows = list(ticks)
    official = _official(report, tick_rows)
    events = _events(report, tick_rows)
    if official["final_evaluation_conflicts_with_last_tick"]:
        events.append({"kind": "official_observation_conflict", "source": "report.official_task_evaluation vs final tick"})
    legacy_status = report.get("status")
    acceptance = True if legacy_status == "passed" else False if legacy_status == "failed" else None
    results = {"schema": SCHEMA, "official": official,
        "execution": {"status": "running" if legacy_status == "running" else "terminal" if acceptance is not None else "unknown",
                      "legacy_status": legacy_status if isinstance(legacy_status, str) else None,
                      "last_phase": report.get("stage") if isinstance(report.get("stage"), str) else None,
                      "environment_steps": _integer(report.get("total_environment_steps")),
                      "native_truncation_observed": official["native_truncation_observed"],
                      "subprocess_timeout_observed": _flag(report.get("timed_out")),
                      "failure_categories": list(dict.fromkeys(row["kind"] for row in events)),
                      "error_type": report.get("error_type") if isinstance(report.get("error_type"), str) else None},
        "physics": _physics(report, events), "quality": _quality(report),
        "acceptance": {"status": _status(acceptance), "policy": "phase1-legacy-acceptance-unchanged",
                       "legacy_acceptance": _receipt(acceptance, "report.status"),
                       "legacy_exit_code_if_finalized": 0 if acceptance is True else 1 if acceptance is False else None,
                       "observed_exit_code": report.get("exit_code") if type(report.get("exit_code")) is int else None,
                       "legacy_official_200_step_complete": _flag(report.get("official_200_step_complete")),
                       "legacy_project_complete_physical_closure": _flag(report.get("project_complete_physical_closure")),
                       "scope": "legacy decision; not recomputed from new result sections",
                       "legacy_report_fields_modified": False},
        "recording": _recording(report, events), "events": events}
    # A rejected input is preferable to a silently non-JSON result.
    json.dumps(results, allow_nan=False)
    return results


def update_run_results(report: dict[str, Any], ticks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Replace only report['results']; return that newly computed value."""
    if not isinstance(report, dict):
        raise TypeError("Expected a mutable report dictionary")
    result = summarize_run(report, ticks)
    report["results"] = result
    return result


def update_run_results_from_file(report: dict[str, Any], tick_path: Path) -> dict[str, Any]:
    """Publish available observations even when recording left a damaged JSONL.

    The input file is never repaired or overwritten. A read failure marks
    recording/legacy acceptance failed; official and measured results retain
    their own evidence, and unavailable tick observations remain unknown.
    """
    try:
        ticks = [json.loads(line) for line in tick_path.read_text().splitlines() if line.strip()]
        if any(not isinstance(row, Mapping) for row in ticks):
            raise ValueError("Tick rows must be JSON mappings")
    except (OSError, UnicodeError, ValueError) as exc:
        report.setdefault("artifact_errors", []).append(
            f"ticks.jsonl summary input: {type(exc).__name__}: {exc}")
        report["status"] = "failed"
        ticks = []
    return update_run_results(report, ticks)


def self_test() -> dict[str, Any]:
    """Synthetic JSON evidence checks; no environment, NumPy or GPU."""
    checks = []
    assert official_completion({'success': [True]}, 171, 200) is True
    assert official_completion({'success': [True]}, 200, 200) is False
    assert official_completion({'success': []}, 171, 200) is None
    assert official_completion(None, 171, 200) is None
    checks.append('raw official completion and unknown values are independent of artifacts')
    values = [False, True, False, True] + [True] * 46
    ticks = [{"environment_step": i + 1, "official_evaluation": {"success": [value]},
              "native_terminated": [value], "native_truncated": False}
             for i, value in enumerate(values)]
    report = {"status": "failed", "stage": "verify_placement", "total_environment_steps": 50,
        "max_episode_steps": 200, "official_task_evaluation": {"success": [True]},
        "official_200_step_complete": False, "project_complete_physical_closure": False,
        "stages": [{"name": "close_and_confirm", "status": "passed", "stable_steps_observed": [5, 5]},
                   {"name": "lift_and_hold", "status": "passed", "trace": [{"objects": [
                       {"object_height_gain_m": .11}, {"object_height_gain_m": .12}]}]},
                   {"name": "carry_lower", "status": "passed"},
                   {"name": "release", "status": "passed", "stable_steps_observed": 3},
                   {"name": "retreat", "status": "passed"},
                   {"name": "verify_placement", "status": "passed", "stable_steps_required": 20,
                    "stable_steps_observed": 20}],
        "actual_retreat_displacement_m": [[0., 0., .081], [0., 0., .082]],
        "held_physics_audit": {"status": "passed", "objects": [
            {"max_position_slip_m": .001, "max_rotation_slip_rad": .01},
            {"max_position_slip_m": .002, "max_rotation_slip_rad": .02}]},
        "artifact_errors": ["video: OSError: disk full"],
        "recording": {"recorded_steps": 50, "actions_shape": [50, 16], "submitted_actions_matched": True,
                      "state_lengths": {"env_states/actors/object": 51}}}
    before = copy.deepcopy(report)
    result = summarize_run(report, ticks)
    assert result["official"]["final_success"] is True and result["official"]["final_success_before_horizon"] is True
    assert result["quality"]["status"] == "passed" and result["recording"]["status"] == "failed"
    assert result["acceptance"]["status"] == "failed" and result["physics"]["status"] == "unknown"
    assert report == before
    monitored = copy.deepcopy(report)
    monitored["physical_monitoring"] = {"observed_substeps": 250, "expected_substeps": 250,
        "forbidden_contact_count": 0, "interarm_forbidden_count": 0, "coverage_complete": True,
        "robot_world_forbidden_count": 0, "robot_world_observed_substeps": 250,
        "robot_world_coverage_complete": True}
    assert summarize_run(monitored, ticks)["physics"]["status"] == "passed"
    robot_collision = copy.deepcopy(monitored)
    robot_collision["physical_monitoring"]["robot_world_forbidden_count"] = 1
    assert summarize_run(robot_collision, ticks)["physics"]["status"] == "failed"
    robot_collision = copy.deepcopy(monitored)
    robot_collision["error"] = "Forbidden actual robot/world contact at100Hz"
    assert summarize_run(robot_collision, ticks)["physics"]["status"] == "failed"
    incomplete_robot = copy.deepcopy(monitored)
    del incomplete_robot["physical_monitoring"]["robot_world_coverage_complete"]
    assert summarize_run(incomplete_robot, ticks)["physics"]["status"] == "unknown"
    checks.append("actual robot/world collisions fail physics; incomplete robot/world coverage stays unknown")
    checks.append("recording failure preserves official true and quality pass; physics needs explicit monitoring")
    assert result["official"]["first_observed_success_step"] == 2
    assert result["official"]["first_success_step_prefix_complete"] is True
    assert result["official"]["success_transitions"] == [
        {"step": 2, "from": False, "to": True}, {"step": 3, "from": True, "to": False},
        {"step": 4, "from": False, "to": True}]
    changed = copy.deepcopy(report); changed["official_task_evaluation"] = {"success": False}
    changes = summarize_run(changed, ticks)
    assert changes["official"]["ever_observed_success"] is True and changes["official"]["final_success"] is False
    assert changes["official"]["final_evaluation_conflicts_with_last_tick"] is True
    checks.append("multiple success changes, first observed success, final failure and conflict stay separate")
    empty = summarize_run({"status": "failed", "official_200_step_complete": False}, [])
    assert empty["official"]["final_success"] is None and empty["official"]["ever_observed_success"] is None
    assert empty["physics"]["status"] == empty["recording"]["status"] == empty["quality"]["status"] == "unknown"
    assert empty["official"]["native_truncation_observed"] is None
    no_proof = summarize_run({"status": "passed", "project_complete_physical_closure": True}, [])
    assert no_proof["physics"]["status"] == no_proof["quality"]["status"] == "unknown"
    checks.append("missing evidence and legacy acceptance never invent official, quality or physical observations")
    partial = summarize_run({"total_environment_steps": 51}, ticks)
    assert partial["official"]["final_success"] is None and partial["official"]["ever_observed_success"] is True
    final_tick = summarize_run({"total_environment_steps": 50, "max_episode_steps": 200}, ticks)
    assert final_tick["official"]["final_success"] is True
    late = copy.deepcopy(report); late["total_environment_steps"] = 200
    late_ticks = [{**ticks[-1], "environment_step": 200, "native_truncated": True}]
    late_result = summarize_run(late, late_ticks)
    assert late_result["official"]["final_success"] is True and late_result["official"]["final_success_before_horizon"] is False
    assert late_result["official"]["native_truncation_observed"] is True
    checks.append("final-tick fallback requires matching step; success and horizon/truncation stay independent")
    plan = {"status": "failed", "error": "Remaining official200 budget cannot fit schedule", "planning": [
        {"phase": "carry_lower", "collision_validation": {"status": "rejected"}}]}
    plan_result = summarize_run(plan, [])
    assert set(plan_result["execution"]["failure_categories"]) == {"budget_insufficient", "planned_collision_rejection"}
    assert plan_result["physics"]["status"] == "unknown"
    fcl = {"status": "failed", "stages": [{"name": "pregrasp", "status": "aborted", "reason": "measured_collision"}]}
    assert summarize_run(fcl, [])["physics"]["status"] == "unknown"
    contact = copy.deepcopy(fcl)
    contact["stages"][0].update(reason="step_or_measurement_failed", error="Forbidden physical contact at100Hz; stop both arms")
    assert summarize_run(contact, [])["physics"]["status"] == "failed"
    for reason in ("grasp_lost", "object_dropped"):
        incident = {"stages": [{"name": "carry_lower", "status": "aborted", "reason": reason}]}
        assert summarize_run(incident, [])["physics"]["status"] == "failed"
    initial_gate = {"stages": [{"name": "carry_lower", "status": "rejected",
        "reason": "initial_grasp_or_attachment_invalid", "initial_observation": {
            "objects": [{"is_grasping": True, "position_slip_m": .006}, {"is_grasping": True}]}}]}
    assert summarize_run(initial_gate, [])["physics"]["status"] == "unknown"
    initial_gate["stages"][0]["initial_observation"]["objects"][1]["is_grasping"] = False
    assert summarize_run(initial_gate, [])["physics"]["status"] == "failed"
    release = {"stages": [{"name": "release", "status": "release_timeout", "stable_steps_observed": 2}]}
    assert summarize_run(release, [])["physics"]["status"] == "unknown"
    release["stages"][0]["trace"] = [{"objects": [{"is_grasping": True}]}]
    assert summarize_run(release, [])["physics"]["status"] == "failed"
    checks.append("planning/FCL/budget differ from actual contact, lost grasp, drop and measured unreleased object")
    short_window = copy.deepcopy(monitored)
    short_window["stages"][-1].update(status="placement_timeout", stable_steps_observed=19)
    window_result = summarize_run(short_window, ticks)
    assert window_result["quality"]["status"] == "failed" and window_result["physics"]["status"] == "passed"
    short_retreat = copy.deepcopy(monitored); short_retreat["actual_retreat_displacement_m"][0][2] = .079
    assert summarize_run(short_retreat, ticks)["quality"]["status"] == "failed"
    assert summarize_run(short_retreat, ticks)["physics"]["status"] == "passed"
    slipped = copy.deepcopy(monitored); slipped["held_physics_audit"]["objects"][0]["max_position_slip_m"] = .006
    assert summarize_run(slipped, ticks)["quality"]["status"] == "failed"
    assert summarize_run(slipped, ticks)["physics"]["status"] == "passed"
    checks.append("window insufficiency, retreat distance and slip quality limits are not physical accidents")
    policy = copy.deepcopy(result["quality"]["policy"]["resolved"])
    policy.update(minimum_retreat_m=.05, require_final_stability=False)
    relaxed = copy.deepcopy(short_retreat); relaxed["runtime_policy"] = {"resolved": policy}
    relaxed["stages"] = relaxed["stages"][:-1]
    assert summarize_run(relaxed, ticks)["quality"]["status"] == "passed"
    relaxed["runtime_policy"] = {"resolved": {}}
    assert summarize_run(relaxed, ticks)["quality"]["status"] == "unknown"
    checks.append("quality reads serialized resolved policy; fallback applies only when policy is absent")
    bad_record = copy.deepcopy(monitored); bad_record["artifact_errors"] = []
    bad_record["recording"]["state_lengths"]["env_states/actors/object"] = 50
    bad = summarize_run(bad_record, ticks)
    assert bad["recording"]["status"] == "failed" and bad["official"]["final_success"] is True
    assert bad["quality"]["status"] == bad["physics"]["status"] == "passed"
    changed_input = copy.deepcopy(report); changed_input["artifact_errors"] = []
    changed_input["input_evidence_unchanged"] = False
    assert summarize_run(changed_input, ticks)["recording"]["status"] == "failed"
    checks.append("T+1 mismatch and changed input evidence affect recording, not official or measured quality")
    replay = {"status": "passed", "mode": "action-replay", "total_environment_steps": 50, "max_episode_steps": 200,
        "official_task_evaluation": {"success": True}, "official_200_step_complete": False,
        "checks": {"no_forbidden100Hz_contacts": True}, "project_complete_physical_closure": True}
    replay_result = summarize_run(replay, ticks)
    assert replay_result["official"]["final_success"] is True
    assert replay_result["physics"]["status"] == replay_result["quality"]["status"] == "unknown"
    checks.append("legacy replay metadata and old contact claims cannot replace missing monitoring or measurements")
    updated = copy.deepcopy(report); updated["results"] = {"old": True}
    original = copy.deepcopy(updated); value = update_run_results(updated, ticks)
    assert updated["results"] is value and {k:v for k,v in updated.items() if k != "results"} == {
        k:v for k,v in original.items() if k != "results"}
    assert summarize_run(updated, ticks) == value
    checks.append("update replaces only results and the JSON summary is deterministic")
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "ticks.jsonl"
        corrupt = '{"environment_step":'
        path.write_text(corrupt)
        interrupted = copy.deepcopy(monitored)
        interrupted.update(status="passed", artifact_errors=[])
        summary = update_run_results_from_file(interrupted, path)
        assert path.read_text() == corrupt and interrupted["status"] == "failed"
        assert summary["official"]["final_success"] is True
        assert summary["physics"]["status"] == summary["quality"]["status"] == "passed"
        assert summary["recording"]["status"] == "failed"
        assert summary["official"]["tick_observations"] == 0
        assert summary["official"]["tick_steps_complete_to_final"] is False
        missing = copy.deepcopy(monitored)
        missing["artifact_errors"] = []
        assert update_run_results_from_file(missing, Path(directory)/"missing.jsonl")["recording"]["status"] == "failed"
    checks.append("damaged or missing tick input preserves final observations and records a recording failure")
    json.dumps([result, empty, no_proof, partial, late_result, replay_result], allow_nan=False)
    return {"status": "passed", "checks": checks, "check_count": len(checks),
            "environment_steps": 0, "gpu_used": False, "stdlib_only": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(self_test(), indent=2, allow_nan=False))
