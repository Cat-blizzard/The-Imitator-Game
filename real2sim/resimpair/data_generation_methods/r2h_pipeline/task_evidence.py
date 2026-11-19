"""Task-level evidence gate, independent of generators and provider protocols.

This gate checks hash-bound observations and explicit numeric measurements. It
does not turn a VLM's description into a pixel measurement, run a tracker, or
certify human 3D/force labels. ``motion_verified`` always reports a limited scope
and requires configured, non-VLM measurements of the actual evaluated media.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


SOURCE_TYPES = {"sim_measured", "vlm_observed", "tracker_measured", "manual_verified"}
STATES = {"observed", "unknown", "not_observed"}
MEASURED_SOURCES = {"sim_measured", "tracker_measured"}


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _binding(value, *, verify_files):
    _require(isinstance(value, dict) and _text(value.get("path")) and _sha(value.get("sha256")),
             "Missing path/SHA256 artifact binding")
    if verify_files:
        _require(file_sha256(value["path"]) == value["sha256"], "Artifact SHA256 mismatch")


def _signature(item):
    return item["verb"], tuple(item["args"])


def validate_contract(contract, *, verify_files=True):
    """Validate a generic contract; source checks can be disabled only for inspection."""
    _require(isinstance(contract, dict), "Contract must be an object")
    _require(contract.get("schema") == "r2h_task_evidence_contract_v1", "Unknown contract schema")
    _require(contract.get("training_admitted") is False, "A task contract cannot admit training")
    _require(_text(contract.get("task_id")), "Missing task ID")
    entities = contract.get("entities")
    _require(isinstance(entities, list) and entities and all(_text(x) for x in entities)
             and len(entities) == len(set(entities)), "Invalid entity IDs")
    source = contract.get("source_window", {})
    _binding(source.get("video"), verify_files=verify_files)
    interval = source.get("window_frames")
    _require(isinstance(interval, list) and len(interval) == 2
             and all(type(x) is int for x in interval) and 0 <= interval[0] < interval[1]
             and source.get("interval") == "start_inclusive_end_exclusive"
             and _number(source.get("fps")) and source["fps"] > 0, "Invalid source window clock")
    _require(source.get("origin") == "real_human" and _text(source.get("split")),
             "Require explicitly bound real-human source and split")
    _binding(contract.get("source_spec"), verify_files=verify_files)
    for item in source.get("review_evidence", []):
        _binding(item, verify_files=verify_files)
    relation_ids = []
    for phase in ("initial", "final"):
        relations = contract.get(phase + "_relations")
        _require(isinstance(relations, list) and relations, "Require explicit initial and final relations")
        for item in relations:
            _require(isinstance(item, dict) and _text(item.get("id")) and _text(item.get("predicate"))
                     and isinstance(item.get("args"), list) and item["args"]
                     and all(x in entities for x in item["args"])
                     and type(item.get("expected")) is bool, "Invalid relation")
            relation_ids.append(item["id"])
    _require(len(relation_ids) == len(set(relation_ids)), "Duplicate relation IDs")
    events = contract.get("required_events")
    _require(isinstance(events, list) and 1 <= len(events) <= 32, "Require 1..32 necessary events")
    for key in ("required_events", "forbidden_events", "allowed_additional_events"):
        _require(isinstance(contract.get(key, []), list), "Invalid event list")
        for item in contract.get(key, []):
            _require(isinstance(item, dict) and _text(item.get("id")) and _text(item.get("verb"))
                     and isinstance(item.get("args"), list) and item["args"]
                     and all(x in entities for x in item["args"]), "Invalid event signature")
    ids = [x["id"] for x in events]
    _require(len(ids) == len(set(ids)), "Duplicate necessary event IDs")
    forbidden = contract.get("forbidden_events", [])
    _require(len({x["id"] for x in forbidden}) == len(forbidden), "Duplicate forbidden event IDs")
    _require(not ({_signature(x) for x in events} & {_signature(x) for x in forbidden}),
             "An event cannot be required and forbidden")
    edges = contract.get("precedence")
    _require(isinstance(edges, list) and all(isinstance(x, list) and len(x) == 2
             and x[0] in ids and x[1] in ids and x[0] != x[1] for x in edges), "Invalid precedence")
    reach = set(map(tuple, edges))
    for mid in ids:
        reach |= {(a, b) for a in ids for b in ids if (a, mid) in reach and (mid, b) in reach}
    _require(not any((x, x) in reach for x in ids), "Cyclic precedence")
    variations = contract.get("allowed_variations")
    _require(isinstance(variations, list) and all(_text(x) for x in variations),
             "Require explicit allowed variations")
    return contract


def validate_policy(policy):
    """No default confidence or motion thresholds; all temporal gates are explicit."""
    _require(isinstance(policy, dict), "Policy must be an object")
    for name in ("time_tolerance_seconds", "initial_window_seconds", "final_window_seconds"):
        _require(_number(policy.get(name)) and policy[name] >= 0, "Missing/invalid explicit threshold: " + name)
    _require(policy["initial_window_seconds"] > 0 and policy["final_window_seconds"] > 0,
             "Boundary windows must be positive")
    measurements = policy.get("motion_requirements", [])
    _require(isinstance(measurements, list), "Invalid motion requirements")
    ids = []
    for item in measurements:
        _require(isinstance(item, dict) and all(_text(item.get(x)) for x in ("id", "metric", "unit", "scope")),
                 "Motion requirement needs ID, metric, unit and scope")
        _require(item.get("comparison") in ("<=", ">=") and _number(item.get("threshold")),
                 "Motion metric needs an explicit finite threshold and comparison")
        _require(isinstance(item.get("allowed_sources"), list) and item["allowed_sources"]
                 and all(x in MEASURED_SOURCES for x in item["allowed_sources"]),
                 "Only sim/tracker measurements can verify motion")
        _require(isinstance(item.get("allowed_methods"), list) and item["allowed_methods"]
                 and all(_text(x) for x in item["allowed_methods"]), "Explicit measured methods required")
        ids.append(item["id"])
    _require(len(ids) == len(set(ids)), "Duplicate motion requirement IDs")


def _measure_paired_tracks(item, subject, *, verify_files):
    """Recompute a small supported measurement; never trust a reported pass/value.

    Input tracks must already share pixel coordinates and have explicit time
    pairing. This verifies residuals/coverage of those visible tracks only,
    not the correctness of tracking, camera calibration, 3D geometry or contact.
    """
    _require(item["method"] == "paired_2d_tracks_v1", "No implemented measurement backend for this method")
    _require(item["source_type"] == "tracker_measured", "2D residual backend requires tracker output")
    _require("value" not in item, "Supply raw tracks, not a self-reported metric value")
    raw = json.loads(Path(item["artifact"]["path"]).read_text())
    _require(isinstance(raw, dict) and raw.get("schema") == "r2h_paired_2d_tracks_v1", "Unknown raw-track schema")
    _require(raw.get("coordinate_system") == "shared_pixel_coordinates"
             and _text(raw.get("alignment_method")), "Require explicit common coordinates and time alignment")
    _binding(raw.get("candidate_media"), verify_files=verify_files)
    _binding(raw.get("reference_media"), verify_files=verify_files)
    _require(raw["candidate_media"]["sha256"] == subject["media"]["sha256"], "Raw tracks bind different media")
    _require(raw["reference_media"]["sha256"] != subject["media"]["sha256"], "Cannot compare candidate with itself")
    _require(isinstance(raw.get("tracker"), dict) and _text(raw["tracker"].get("name"))
             and _text(raw["tracker"].get("version")), "Missing tracker identity")
    stamps = raw["candidate_timestamps_seconds"]
    reference_stamps = raw["reference_timestamps_seconds"]
    predicted, reference, valid = raw["candidate_xy"], raw["reference_xy"], raw["valid_pairs"]
    _require(isinstance(stamps, list) and len(stamps) >= 2
             and all(_number(x) and 0 <= x <= subject["duration_seconds"] for x in stamps)
             and all(a < b for a, b in zip(stamps, stamps[1:])), "Invalid candidate track timestamps")
    _require(isinstance(reference_stamps, list) and len(reference_stamps) == len(stamps)
             and all(_number(x) and x >= 0 for x in reference_stamps)
             and all(a < b for a, b in zip(reference_stamps, reference_stamps[1:])),
             "Invalid explicit reference time pairing")
    _require(all(isinstance(x, list) and len(x) == len(stamps) for x in (predicted, reference, valid)),
             "Track/time dimensions differ")
    n_tracks = len(valid[0])
    _require(n_tracks > 0, "No paired object tracks")
    errors, count = [], 0
    for actual_row, target_row, mask in zip(predicted, reference, valid):
        _require(all(isinstance(x, list) and len(x) == n_tracks for x in (actual_row, target_row, mask)),
                 "Track dimensions differ")
        for actual, target, available in zip(actual_row, target_row, mask):
            _require(type(available) is bool, "Validity mask must be Boolean")
            count += 1
            if available:
                _require(all(isinstance(x, list) and len(x) == 2 and all(_number(v) for v in x)
                             for x in (actual, target)), "Invalid visible 2D coordinate")
                errors.append(sum((a - b) ** 2 for a, b in zip(actual, target)))
    _require(errors, "No visible paired coordinates")
    _require(item["scope"] == "visible_aligned_2d_tracks_only", "Track backend cannot verify broader motion scope")
    if item["metric"] == "visible_track_rmse":
        _require(item["unit"] == "pixels", "2D RMSE must use pixels")
        value = math.sqrt(sum(errors) / len(errors))
    elif item["metric"] == "visible_pair_coverage":
        _require(item["unit"] == "fraction", "Coverage must use fraction")
        value = len(errors) / count
    else:
        raise ValueError("Unsupported raw-track metric")
    return {"value": value, "visible_pairs": len(errors), "total_pairs": count,
            "raw_artifact_sha256": item["artifact"]["sha256"], "recomputed_from_raw_tracks": True}


def evaluate_task_evidence(contract, evidence, *, policy, verify_files=True):
    """Compare observations with the contract; never use ``observed_goal`` as a pass.

    Every observation has a state, source_type, media_sha256, time interval and
    an evidence artifact. ``not_observed`` means absence was looked for, not a
    physically proven negative. For required facts it causes abstention.
    Numeric confidence is optional metadata and never an acceptance probability.
    """
    validate_contract(contract, verify_files=verify_files)
    validate_policy(policy)
    problems, motion_problems, used_sources, measured_values = [], [], Counter(), []

    def issue(code, detail, *, reject=False):
        problems.append({"code": code, "detail": detail, "reject": reject})

    result = {
        "schema": "r2h_task_evidence_decision_v1", "contract_sha256": canonical_sha256(contract),
        "policy_sha256": canonical_sha256(policy), "training_admitted": False,
        "decision": "needs_review", "task_equivalent_candidate": False, "motion_verified": False,
        "motion_verification_scope": [], "human_3d_ground_truth": False,
        "confidence_used_as_probability": False, "evidence_files_verified": verify_files,
        "quality_release_approved": False, "media_decoded_by_this_gate": False,
    }
    try:
        _require(isinstance(evidence, dict), "Evidence must be an object")
        _require(evidence.get("schema") == "r2h_task_observations_v1", "Unknown evidence schema")
        _require(evidence.get("contract_sha256") == result["contract_sha256"], "Contract evidence hash mismatch")
        subject = evidence["subject"]
        _binding(subject["media"], verify_files=verify_files)
        _require(subject.get("origin") in ("real_human", "synthetic_human", "robot_sim"),
                 "Unknown evaluated media origin")
        duration = subject["duration_seconds"]
        _require(_number(duration) and duration > 0, "Invalid evaluated media duration")
        media_sha = subject["media"]["sha256"]
        tolerance = policy["time_tolerance_seconds"]
        relations, events = evidence["relation_observations"], evidence["event_observations"]
        _require(isinstance(relations, list) and isinstance(events, list), "Missing observation lists")

        def observation(item):
            _require(isinstance(item, dict) and item.get("state") in STATES and item.get("source_type") in SOURCE_TYPES,
                     "Invalid observation state/source")
            _require(item.get("media_sha256") == media_sha, "Observation refers to different media")
            _binding(item.get("artifact"), verify_files=verify_files)
            _require(_text(item.get("method")) and _text(item.get("description")), "Missing method/evidence text")
            start, end = item.get("t_start"), item.get("t_end")
            _require(_number(start) and _number(end) and 0 <= start <= end <= duration + tolerance,
                     "Observation interval outside evaluated media")
            if "confidence" in item:
                _require(_number(item["confidence"]) and 0 <= item["confidence"] <= 1,
                         "Confidence metadata must be finite in [0,1]")
            if subject["origin"] != "robot_sim":
                _require(item["source_type"] != "sim_measured",
                         "Sim measurements cannot certify pixels in a human video")
            used_sources[item["source_type"]] += 1

        relation_lookup = {(phase, x["id"]): x for phase in ("initial", "final")
                           for x in contract[phase + "_relations"]}
        seen_relations = Counter()
        for item in relations:
            observation(item)
            key = item.get("phase"), item.get("requirement_id")
            _require(key in relation_lookup, "Unknown relation requirement")
            expected = relation_lookup[key]
            _require(item.get("predicate") == expected["predicate"] and item.get("args") == expected["args"],
                     "Relation signature differs from named requirement")
            seen_relations[key] += 1
            phase = key[0]
            if phase == "initial" and item["t_end"] > policy["initial_window_seconds"] + tolerance:
                issue("INITIAL_EVIDENCE_OUTSIDE_BOUNDARY", expected["id"])
            if phase == "final" and item["t_start"] < duration - policy["final_window_seconds"] - tolerance:
                issue("FINAL_EVIDENCE_OUTSIDE_BOUNDARY", expected["id"])
            if item["state"] != "observed":
                _require(item.get("value") is None, "Unknown/unobserved relation cannot assert a value")
                issue("RELATION_NOT_ESTABLISHED", expected["id"] + ":" + item["state"])
            else:
                _require(type(item.get("value")) is bool, "Observed relation requires an explicit Boolean value")
                if item["value"] != expected["expected"]:
                    issue("RELATION_CONTRADICTION", expected["id"], reject=True)
        for key in relation_lookup:
            if not seen_relations[key]:
                issue("RELATION_EVIDENCE_MISSING", ":".join(key))

        for item in events:
            observation(item)
            _require(_text(item.get("verb")) and isinstance(item.get("args"), list) and item["args"]
                     and all(x in contract["entities"] for x in item["args"]), "Invalid observed event")
            if item["state"] == "observed":
                # A visible milestone is a witnessed state at a frame timestamp,
                # not the duration of a grasp/hold or a reconstructed onset.
                _require(item["t_end"] > item["t_start"]
                         or item.get("temporal_semantics") == "visible_milestone",
                         "An observed event needs a nonzero interval or explicit visible milestone")
        signatures = {_signature(x) for x in contract["required_events"]}
        signatures |= {_signature(x) for x in contract.get("allowed_additional_events", [])}
        forbidden_signatures = {_signature(x) for x in contract.get("forbidden_events", [])}
        for item in events:
            if item["state"] == "observed" and _signature(item) not in signatures | forbidden_signatures:
                issue("UNEXPLAINED_OBSERVED_EVENT", item["verb"])
        for expected in contract.get("forbidden_events", []):
            observations = [x for x in events if _signature(x) == _signature(expected)]
            if any(x["state"] == "observed" for x in observations):
                issue("FORBIDDEN_EVENT_OBSERVED", expected["id"], reject=True)
            elif not any(x["state"] == "not_observed" and x["t_start"] <= tolerance
                         and x["t_end"] >= duration - tolerance for x in observations):
                issue("FORBIDDEN_EVENT_COVERAGE_MISSING", expected["id"])

        # Two reviewers describing the same signature/time interval are
        # corroboration, not two separate physical occurrences.
        unique_observed = {}
        for i, item in enumerate(events):
            if item["state"] == "observed":
                unique_observed.setdefault((_signature(item), item["t_start"], item["t_end"]), i)
        required = contract["required_events"]
        choices = {x["id"]: [i for (signature, _, _), i in unique_observed.items()
                            if _signature(x) == signature] for x in required}
        for name, candidates in choices.items():
            if not candidates:
                issue("REQUIRED_EVENT_NOT_ESTABLISHED", name)
        assignment = {}

        def match(index):
            if index == len(required):
                return True
            name = required[index]["id"]
            for candidate in choices[name]:
                if candidate in assignment.values():
                    continue
                assignment[name] = candidate
                if all(a not in assignment or b not in assignment
                       or events[assignment[a]]["t_end"] <= events[assignment[b]]["t_start"] + tolerance
                       for a, b in contract["precedence"]) and match(index + 1):
                    return True
                assignment.pop(name)
            return False

        matched = all(choices.values()) and match(0)
        if all(choices.values()) and not matched:
            issue("EVENT_PRECEDENCE_VIOLATION", "No one-to-one matching satisfies required order", reject=True)
        result["matched_event_indices"] = dict(assignment) if matched else {}
        if not verify_files:
            issue("ARTIFACT_FILES_UNVERIFIED", "Inspection-only mode cannot approve evidence")

        measurements = evidence.get("motion_measurements", [])
        _require(isinstance(measurements, list) and all(isinstance(x, dict) for x in measurements),
                 "Invalid measurement list")
        for requirement in policy.get("motion_requirements", []):
            matches = [x for x in measurements if x.get("requirement_id") == requirement["id"]]
            if len(matches) != 1:
                motion_problems.append({"code": "MOTION_MEASUREMENT_MISSING_OR_DUPLICATED", "id": requirement["id"]})
                continue
            item = matches[0]
            try:
                _require(item.get("state") == "observed", "Motion measurement unknown/not observed")
                _require(item.get("source_type") in requirement["allowed_sources"]
                         and item["source_type"] in MEASURED_SOURCES, "Unmeasured/VLM motion claim")
                _require(subject["origin"] == "robot_sim" or item["source_type"] != "sim_measured",
                         "Simulation measurements do not certify generated human motion")
                _require(item.get("media_sha256") == media_sha, "Motion measurement has wrong media hash")
                _binding(item.get("artifact"), verify_files=verify_files)
                _require(item.get("method") in requirement["allowed_methods"], "Unconfigured measurement method")
                _require(all(item.get(k) == requirement[k] for k in ("metric", "unit", "scope")),
                         "Motion measurement metric/unit/scope mismatch")
                measurement = _measure_paired_tracks(item, subject, verify_files=verify_files)
                value, threshold = measurement["value"], requirement["threshold"]
                _require(_number(value), "Computed motion metric is not finite")
                measured_values.append({"id": requirement["id"], "metric": requirement["metric"],
                                        "unit": requirement["unit"], **measurement})
                passed = value <= threshold if requirement["comparison"] == "<=" else value >= threshold
                _require(passed, "Motion measurement exceeds configured threshold")
            except (KeyError, TypeError, ValueError, OSError, ArithmeticError) as error:
                motion_problems.append({"code": "MOTION_CHECK_NOT_VERIFIED", "id": requirement["id"], "detail": str(error)})
        result["task_equivalent_candidate"] = not problems
        result["motion_verified"] = bool(not problems and policy.get("motion_requirements") and not motion_problems)
        if result["motion_verified"]:
            result["motion_verification_scope"] = sorted({x["scope"] for x in policy["motion_requirements"]})
        result["decision"] = ("rejected" if any(x["reject"] for x in problems) else
                              "needs_review" if problems else "motion_verified" if result["motion_verified"]
                              else "task_equivalent_candidate")
    except (KeyError, TypeError, ValueError, OSError) as error:
        issue("EVIDENCE_INVALID", str(error), reject=True)
        result["decision"] = "rejected"
    result["issues"] = problems
    result["motion_issues"] = motion_problems
    result["motion_measured_values"] = measured_values
    result["observation_source_counts"] = dict(used_sources)
    result["verification_limit"] = (
        "Checks bound observation/measurement reports; no tracker or physical reconstruction is run. "
        "Task equivalence does not imply human 3D/action labels or framewise alignment.")
    return result


def build_apple_contract(spec_path, window_id):
    """Translate the existing reviewed apple task; hash-check sources, never approve data."""
    spec_path = Path(spec_path).resolve()
    spec = json.loads(spec_path.read_text())
    task = next(x for x in spec["tasks"] if x["id"] == "apple_bowl_transfer")
    window = next(x for x in task["human_windows"] if x["id"] == window_id)
    source = {**deepcopy(window), "origin": "real_human",
              "review_evidence": [deepcopy(window["review"]), deepcopy(window["actual_review_sheet"])]}
    def relation(name, predicate, *args):
        return {"id": name, "predicate": predicate, "args": list(args), "expected": True}
    def event(name, verb, *args):
        return {"id": name, "verb": verb, "args": list(args)}
    events = [event("grasp_apple", "grasp", "actor", "apple"),
              event("lift_from_source", "lift_clear_of", "apple", "bowl_source"),
              event("transport_to_target", "transport_to", "apple", "bowl_target"),
              event("release_in_target", "release_in", "apple", "bowl_target"),
              event("withdraw_hand", "withdraw_from", "actor", "bowl_target")]
    contract = {
        "schema": "r2h_task_evidence_contract_v1", "task_id": task["canonical_goal"],
        "source_spec": {"path": str(spec_path), "sha256": file_sha256(spec_path)},
        "source_window": source, "entities": ["actor", "apple", "bowl_source", "bowl_target"],
        "entity_roles": {"bowl_source": "the initially apple-occupied bowl",
                         "bowl_target": "the unique initially empty destination bowl",
                         "actor": "the acting human hand or robot gripper; embodiment may differ"},
        "initial_relations": [relation("initial_apple_in_source", "inside", "apple", "bowl_source"),
                              relation("initial_target_empty", "empty", "bowl_target")],
        "final_relations": [relation("final_apple_in_target", "inside", "apple", "bowl_target"),
                            relation("final_apple_supported", "stably_supported_by", "apple", "bowl_target"),
                            relation("final_apple_released", "released_from", "apple", "actor"),
                            relation("final_hand_withdrawn", "withdrawn_from", "actor", "bowl_target")],
        "required_events": events, "precedence": [[a["id"], b["id"]] for a, b in zip(events, events[1:])],
        "forbidden_events": [event("move_source_bowl", "transport", "bowl_source"),
                             event("move_target_bowl", "transport", "bowl_target"),
                             event("pour_from_source", "pour_from", "bowl_source")],
        "allowed_additional_events": [], "allowed_variations": deepcopy(task["allowed_differences"]),
        "source_initial_description": window["initial"], "source_final_description": window["final"],
        "source_necessary_steps": deepcopy(window["steps"]),
        "contract_status": "prepared_from_existing_review_not_new_quality_approval",
        "training_admitted": False, "human_wrist_3d_ground_truth_created": False,
    }
    validate_contract(contract)
    return contract
