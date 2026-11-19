"""Directed, frame-grounded VLM observations for the generic task evidence gate.

This module makes no provider calls. A valid response schema is only a parsing
result, never task correctness or permission to release training data.

The prompt hides expected truth values, goal IDs, event obligations and order,
but revealing the predicates still directs the review and can bias a reviewer.
Use an independently prompted blind description as a separate cross-check.

Sparse frames can show a visible milestone (for example an object already clear
of a support). They cannot establish continuous motion, contact force, or the
absence of an event between frames. A milestone timestamp is the time that state
is visibly witnessed, not a reconstructed action onset or a long held-state
duration. Missing transition frames are returned as refinement requests.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

from .providers import ProviderError, parse_review_json
from .task_evidence import canonical_sha256, file_sha256, validate_contract


_PREDICATES = {
    "inside": "The named object is visibly inside the named container.",
    "empty": "The visible container interior has no object; occlusion is unknown.",
    "stably_supported_by": "Visible support and unchanged resting placement across frames; "
                           "a single image cannot establish temporal stability.",
    "released": "The object is visibly no longer held; hidden contact is unknown.",
    "released_from": "The object is visibly separated from the named actor's hold.",
    "withdrawn_from": "The actor is visibly clear of the named object or region.",
    "closed": "The named door, book, or container is visibly in its closed configuration.",
    "open": "The named door, book, or container is visibly in its open configuration.",
    "supported_by": "The named object visibly rests on the named support. Do not infer force.",
    "unchanged_from_initial": "The named object's visible pose/configuration matches its "
                              "initial appearance. Cite both the initial frame and the "
                              "current frame; this does not establish no movement in between.",
}
_EVENTS = {
    "grasp": "Milestone: first unambiguous visible hold by the actor. A hand merely near "
             "an object does not show a grasp; no contact force is inferred.",
    "lift_clear_of": "Milestone: object visibly clear of its former support, rather than "
                     "the duration for which it remains lifted.",
    "transport_to": "Milestone: object visibly reaches the named destination region. "
                    "This alone does not certify the intervening trajectory or release.",
    "release_in": "Milestone: object visibly rests inside the container with the actor's "
                  "hold visibly removed. Occluded contact remains unknown.",
    "withdraw_from": "Milestone: actor visibly clears the named object or region.",
    "transport": "A visible change in position of the named object. Report only the "
                 "visible transition or a clearly witnessed arrival milestone.",
    "release": "Milestone: the actor's hold on the named object is visibly removed, "
               "with visible separation. Do not infer hidden contact or force.",
    "move": "A visible change in the named object's position.",
    "pour_from": "Visible contents leaving the named tilted container.",
    "place_inside": "Milestone: the named object visibly reaches the interior of the "
                    "named container. Placement alone does not establish release.",
    "open": "Milestone: the named door, book, or container is visibly in an open "
            "configuration. This does not certify the entire opening trajectory.",
    "remove_from_cabinet": "Milestone: the named contents are visibly outside the cabinet "
                           "after having been visibly inside. Do not infer an unseen removal.",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _queries(contract):
    validate_contract(contract, verify_files=False)
    relations = [(phase, item) for phase in ("initial", "final")
                 for item in contract[phase + "_relations"]]
    relations.sort(key=lambda pair: (pair[0], pair[1]["predicate"], pair[1]["args"], pair[1]["id"]))
    relation_queries = {f"relation_{i:03d}": (phase, item)
                        for i, (phase, item) in enumerate(relations)}
    # Deduplicate signatures, not physical occurrences. Repeated occurrences are
    # returned in each query's list and the gate does one-to-one event matching.
    signatures = {(item["verb"], tuple(item["args"]))
                  for field in ("required_events", "forbidden_events", "allowed_additional_events")
                  for item in contract.get(field, [])}
    event_queries = {f"event_{i:03d}": {"verb": verb, "args": list(args)}
                     for i, (verb, args) in enumerate(sorted(signatures))}
    return relation_queries, event_queries


def _object(properties):
    return {"type": "object", "additionalProperties": False,
            "properties": properties, "required": list(properties)}


def build_review_schema(contract):
    """Return strict JSON Schema containing no expected answers or private IDs."""
    relations, events = _queries(contract)
    text = {"type": "string", "minLength": 1}
    observed = {"anyOf": [{"type": "boolean"}, {"type": "string", "enum": ["unknown"]}]}
    frame = _object({
        "frame_index": {"type": "integer", "minimum": 0},
        "timestamp_seconds": {"type": "number", "minimum": 0},
        "description": text,
    })
    evidence = {"type": "array", "maxItems": 512, "items": frame}
    relation = _object({"observed": observed, "evidence": evidence, "description": text})
    event = _object({
        "observed": observed,
        "temporal_semantics": {"type": "string",
                               "enum": ["visible_milestone", "visible_transition", "unknown"]},
        "t_start": {"type": ["number", "null"], "minimum": 0},
        "t_end": {"type": ["number", "null"], "minimum": 0},
        "evidence": evidence,
        "description": text,
    })
    additional_event = deepcopy(event)
    additional_event["properties"].update({
        "verb": text,
        "args": {"type": "array", "minItems": 1,
                 "items": {"type": "string", "enum": list(contract["entities"])}},
    })
    additional_event["required"].extend(["verb", "args"])
    return deepcopy(_object({
        "relations": _object({key: relation for key in relations}),
        "events": _object({key: {"type": "array", "minItems": 1, "maxItems": 128, "items": event}
                           for key in events}),
        "additional_events": {"type": "array", "maxItems": 128, "items": additional_event},
        "limitations": {"type": "array", "maxItems": 128, "items": text},
    }))


def build_review_prompt(contract):
    """Render a directed observation request, not a target-answer checklist.

    Optional entity_roles must describe identities, not intended actions. Only
    these identity definitions and the queried signatures enter the prompt;
    arbitrary contract descriptions, source metadata and private IDs do not.
    """
    relations, events = _queries(contract)
    roles = contract.get("entity_roles", {})
    _require(isinstance(roles, dict), "entity_roles must be an object")
    _require(all(isinstance(roles.get(entity, entity), str) and roles.get(entity, entity).strip()
                 for entity in contract["entities"]), "Invalid entity role description")
    public = {
        "entities": [{"id": entity, "definition": roles.get(entity, entity)}
                     for entity in contract["entities"]],
        "relation_queries": {
            key: {"phase": phase, "predicate": item["predicate"], "args": item["args"],
                  "definition": _PREDICATES.get(item["predicate"],
                      "Use the literal visible relation; return unknown if its meaning or evidence is unclear.")}
            for key, (phase, item) in relations.items()},
        "event_queries": {
            key: {**item, "definition": _EVENTS.get(item["verb"],
                      "Use a literal visible change or clearly witnessed milestone; "
                      "return unknown if its meaning or evidence is unclear.")}
            for key, item in events.items()},
        "unknown_observation_examples": {
            "relation": {"observed": "unknown", "evidence": [],
                         "description": "The supplied frames do not establish this relation."},
            "event": {"observed": "unknown", "temporal_semantics": "unknown",
                      "t_start": None, "t_end": None, "evidence": [],
                      "description": "The supplied frames do not establish this event."},
        },
        # Some gateways/models may not enforce responseJsonSchema. The exact
        # same schema is also supplied as text, without relaxing local checks.
        "response_json_schema": build_review_schema(contract),
    }
    return (
        "Inspect ONLY the supplied timestamped frames. These are observation queries, not "
        "instructions for what should happen. No target truth values or target goal are supplied. "
        "Object roles name identities; keep identities fixed even when objects move. "
        "For each relation, observed is the actual visible predicate truth (true or false), "
        "or the string unknown when evidence is insufficient. Inspect the beginning for initial "
        "relations and the end for final relations. Cite exact supplied frame_index and "
        "timestamp_seconds with a literal visual description. Never invent frames or timestamps. "
        "Return an array for every event query, listing each distinct witnessed occurrence; "
        "otherwise return one unknown/false observation. Include other visible actions in "
        "additional_events, even if they are not among the queries. "
        "For a visible_milestone use t_start=t_end at the supplied frame that visibly witnesses "
        "the milestone. This records when a state is seen, not the hidden time it began. "
        "For a visible_transition give the earliest/latest supplied evidence frames showing "
        "that transition, not the duration of a subsequent held state. A grasp interval must "
        "not extend through transporting and releasing. Do not infer grasp onset from an "
        "already lifted object, or release from an occluded hand. "
        "Sparse images do not show a complete grasp/lift/transport/release trajectory. "
        "A transition across unsupplied original frames remains unknown; cite its bracketing "
        "frames so denser review can be requested. A clear lifted-object or withdrawn-hand "
        "milestone can be reported at one frame without claiming continuous motion. "
        "Event false means only that it was not visible in the inspected frames, never that "
        "it was absent throughout the video. Use null event bounds if no supplied frames "
        "can bound the uncertainty. For unknown observations, evidence can be empty. "
        "Never infer contact, force, stability, physical validity, success or 3D motion from "
        "proximity, captions, filenames or a plausible intended action. Contact hidden by "
        "occlusion is unknown; a single image does not prove stable support over time. "
        "Describe limitations honestly. The relation_queries and event_queries below are "
        "QUERY DEFINITIONS, NOT the output template; do not return those container names. "
        "The output object MUST have exactly these four keys: relations, events, "
        "additional_events, limitations. Include every listed anonymous relation/event ID "
        "under relations/events respectively. Each relation MUST include observed, evidence, "
        "description. EVERY event entry, including false or unknown, MUST include observed, "
        "temporal_semantics, t_start, t_end, evidence, description. Event bounds are JSON "
        "numbers or null, never quoted strings; use t_start/t_end, never start/end. "
        "The neutral unknown_observation_examples illustrate all mandatory fields, without "
        "providing answers. Always include additional_events and limitations, even when empty. "
        "The complete response_json_schema below is authoritative and is identical to the "
        "schema supplied in the request configuration. Return only the JSON response object "
        "matching it, with no Markdown fences and no omitted fields.\n"
        + json.dumps(public, ensure_ascii=False, sort_keys=True)
    )


def _binding(value):
    _require(isinstance(value, dict), "Missing artifact binding")
    path, digest = value.get("path"), value.get("sha256")
    _require(isinstance(path, (str, Path)) and str(path).strip(), "Missing artifact path")
    _require(isinstance(digest, str) and len(digest) == 64
             and all(char in "0123456789abcdef" for char in digest), "Invalid artifact SHA256")
    _require(file_sha256(path) == digest, "Artifact SHA256 mismatch")
    return {"path": str(Path(path).resolve()), "sha256": digest}


def _review_value(result):
    _require(isinstance(result, dict), "Review result must be an object")
    return result["review"] if "review" in result else result


def normalize_review(contract, result, *, media, review_artifact, supplied_frames,
                     origin="synthetic_human"):
    """Bind actual reviewer output to gate observations, with conservative scope.

    ``media`` supplies path, sha256 and duration_seconds; ``review_artifact``
    supplies path and sha256 of the saved raw review or Gemini result envelope.
    ``supplied_frames`` is the actual review_frames input (path, frame_index,
    timestamp_seconds), optionally with a sha256 that is checked. Files are
    always hash-checked; no inspection-only acceptance mode exists.

    Citation clocks must exactly match caller-supplied decoded-frame clocks.
    This checks bindings, not that an image was correctly decoded from a video;
    caller extraction provenance remains necessary. Boolean event absence is
    downgraded to unknown because frame packets cannot certify whole-clip absence.
    No confidence values, measured motion, task approval or training admission
    are manufactured.
    """
    validate_contract(contract)
    schema = build_review_schema(contract)
    review = _review_value(result)
    try:
        parsed = parse_review_json(json.dumps(review, allow_nan=False), schema)
    except (ProviderError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("Review schema validation failed") from error
    review = parsed["review"]
    media_binding = _binding(media)
    artifact_binding = _binding(review_artifact)
    duration = media.get("duration_seconds")
    _require(_number(duration) and duration > 0, "Invalid clip duration")
    _require(origin in ("real_human", "synthetic_human", "robot_sim"), "Invalid media origin")
    # A hash alone must not let a different response borrow a real artifact.
    saved = parse_review_json(Path(artifact_binding["path"]).read_text())["review"]
    _require(canonical_sha256(_review_value(saved)) == canonical_sha256(review),
             "Review artifact contains a different review")
    _require(isinstance(supplied_frames, list) and supplied_frames, "No supplied frames")
    frames, by_index = [], {}
    for frame in supplied_frames:
        _require(isinstance(frame, dict), "Invalid supplied frame")
        index, timestamp = frame.get("frame_index"), frame.get("timestamp_seconds")
        _require(type(index) is int and index >= 0 and _number(timestamp)
                 and 0 <= timestamp <= duration, "Invalid supplied frame clock")
        _require(not frames or (index > frames[-1]["frame_index"]
                 and timestamp > frames[-1]["timestamp_seconds"]), "Unordered/duplicate supplied frames")
        path = frame.get("path")
        _require(isinstance(path, (str, Path)) and str(path).strip(), "Missing supplied frame path")
        digest = file_sha256(path)
        _require("sha256" not in frame or frame["sha256"] == digest, "Supplied frame SHA256 mismatch")
        bound = {"frame_index": index, "timestamp_seconds": timestamp,
                 "artifact": {"path": str(Path(path).resolve()), "sha256": digest}}
        frames.append(bound)
        by_index[index] = bound
    available_times = {frame["timestamp_seconds"] for frame in frames}
    relation_queries, event_queries = _queries(contract)
    refinements, relations, events = [], [], []

    def citations(item):
        points = item["evidence"]
        seen = set()
        for point in points:
            index, timestamp = point["frame_index"], point["timestamp_seconds"]
            _require(index in by_index and by_index[index]["timestamp_seconds"] == timestamp,
                     "Evidence citation is not an exact supplied frame")
            _require(index not in seen, "Duplicate evidence frame")
            _require(point["description"].strip(), "Empty frame evidence description")
            seen.add(index)
        return deepcopy(points)

    def refine(query, kind, points, reason, bounds=None):
        if bounds is None and points:
            low = min(point["frame_index"] for point in points)
            high = max(point["frame_index"] for point in points)
            # For a single ambiguous image request neighboring supplied-frame
            # brackets, without inventing a missing video's total frame count.
            before = [frame for frame in frames if frame["frame_index"] < low]
            after = [frame for frame in frames if frame["frame_index"] > high]
            if low == high:
                low = before[-1]["frame_index"] if before else low
                high = after[0]["frame_index"] if after else high
            bounds = [low, high]
        if bounds is None:
            bounds = [frames[0]["frame_index"], frames[-1]["frame_index"]]
        refinements.append({"query_id": query, "kind": kind, "frame_interval": bounds,
                            "interval": "start_inclusive_end_inclusive",
                            "time_interval_seconds": [by_index[bounds[0]]["timestamp_seconds"],
                                                      by_index[bounds[1]]["timestamp_seconds"]],
                            "reason": reason})

    def common(item, points, start, end, state):
        _require(item["description"].strip(), "Empty observation description")
        return {"source_type": "vlm_observed", "media_sha256": media_binding["sha256"],
                "artifact": deepcopy(artifact_binding), "method": "directed_frame_task_review_v1",
                "description": item["description"], "state": state,
                "t_start": start, "t_end": end, "frame_evidence": points}

    for query, (phase, requirement) in relation_queries.items():
        item = review["relations"][query]
        points = citations(item)
        times = [point["timestamp_seconds"] for point in points]
        established = type(item["observed"]) is bool and bool(points)
        # "Stable" is a temporal predicate; a single witness cannot establish it.
        if (requirement["predicate"] == "stably_supported_by"
                and item["observed"] is True and len(points) < 2):
            established = False
        comparison_points = []
        if requirement["predicate"] == "unchanged_from_initial":
            # Initial evidence is a comparison reference, not a claim that the
            # final relation was observed throughout the entire intervening clip.
            comparison_points = [point for point in points if point["frame_index"] == 0]
            established = established and bool(comparison_points) and len(points) >= 2
        state = "observed" if established else "unknown"
        start = min(times) if times else 0
        end = max(times) if times else 0
        if comparison_points:
            start = end
        relation = common(item, points, start, end, state)
        if comparison_points:
            relation["comparison_frame_evidence"] = deepcopy(comparison_points)
        relation.update(phase=phase, requirement_id=requirement["id"],
                        predicate=requirement["predicate"], args=deepcopy(requirement["args"]),
                        value=item["observed"] if established else None)
        relations.append(relation)
        if not established:
            refine(query, "relation", points, "Relation truth is not established by supplied evidence.")

    def normalize_event(query, signature, item):
        points = citations(item)
        start, end = item["t_start"], item["t_end"]
        _require((start is None) == (end is None), "Both event bounds must be supplied or null")
        if start is not None:
            _require(_number(start) and _number(end) and 0 <= start <= end <= duration,
                     "Event interval outside clip or reversed")
            _require(start in available_times and end in available_times,
                     "Event endpoints are not supplied frame timestamps")
            _require(all(start <= point["timestamp_seconds"] <= end for point in points),
                     "Evidence citation outside event interval")
        semantics = item["temporal_semantics"]
        established = item["observed"] is True and bool(points) and start is not None
        reason = "Event occurrence is unknown; sampled non-detection cannot prove clip-wide absence."
        if established and semantics == "visible_milestone":
            _require(start == end, "A visible milestone must be a point, not a held-state interval")
            established = any(point["timestamp_seconds"] == start for point in points)
        elif established and semantics == "visible_transition":
            _require(start < end, "A visible transition must have a positive interval")
            cited_times = {point["timestamp_seconds"] for point in points}
            established = start in cited_times and end in cited_times
            covered = [frame for frame in frames if start <= frame["timestamp_seconds"] <= end]
            if any(right["frame_index"] != left["frame_index"] + 1
                   for left, right in zip(covered, covered[1:])):
                established = False
                reason = "Transition crosses unsupplied original frames; continuous evidence is unknown."
            elif not established:
                reason = "Transition lacks cited evidence at both supplied-frame endpoints."
        else:
            established = False
        times = [point["timestamp_seconds"] for point in points]
        if start is None:
            start, end = (min(times), max(times)) if times else (0, 0)
        observation = common(item, points, start, end, "observed" if established else "unknown")
        observation.update(deepcopy(signature), temporal_semantics=semantics)
        events.append(observation)
        if not established:
            bounds = None
            if item["t_start"] is not None:
                bounds = [next(frame["frame_index"] for frame in frames
                               if frame["timestamp_seconds"] == item[key])
                          for key in ("t_start", "t_end")]
            refine(query, "event", points, reason, bounds)

    for query, signature in event_queries.items():
        for item in review["events"][query]:
            normalize_event(query, signature, item)
    for i, item in enumerate(review["additional_events"]):
        normalize_event(f"additional_{i:03d}", {"verb": item["verb"], "args": item["args"]}, item)
    return {
        "schema": "r2h_task_observations_v1", "contract_sha256": canonical_sha256(contract),
        "subject": {"media": media_binding, "duration_seconds": duration, "origin": origin},
        "relation_observations": relations, "event_observations": events,
        "motion_measurements": [], "training_admitted": False, "quality_release_approved": False,
        "review_metadata": {
            "schema_validated": True, "task_correctness": "not_evaluated",
            "review_artifact": artifact_binding, "supplied_frames": frames,
            "directed_predicate_review": True, "expected_truth_values_provided": False,
            "blind_review": False, "continuous_motion_verified": False,
            "contact_force_verified": False, "frame_extraction_verified_by_this_module": False,
            "whole_clip_absence_established": False, "limitations": deepcopy(review["limitations"]),
            "bias_note": "Predicate queries direct attention and can bias observations; "
                         "compare a separate blind description.",
        },
        "refinement_requests": refinements,
    }
