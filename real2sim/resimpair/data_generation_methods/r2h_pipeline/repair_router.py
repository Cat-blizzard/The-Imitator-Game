"""Deterministic repair planning; no provider calls, image edits or approval.

``plan_repairs`` consumes explicit, hash-bound observations. Unknown is never a
pass. Returned actions must be executed by an orchestrator, whose new evidence
and updated counters are supplied on the next call. See REPAIR_ROUTER.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .anchor_body_review import decision as body_decision
from .refinement import plan_refinement
from .task_evidence import canonical_sha256, file_sha256


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _binding(value, verify_files):
    _require(isinstance(value, dict), "Artifact binding must be an object")
    path, digest = value.get("path"), value.get("sha256")
    _require(isinstance(path, str) and path.strip(), "Artifact path is required")
    _require(isinstance(digest, str) and len(digest) == 64
             and all(c in "0123456789abcdef" for c in digest), "Invalid artifact SHA256")
    if verify_files:
        _require(file_sha256(path) == digest, "Artifact SHA256 mismatch: " + path)
    return {"path": path, "sha256": digest}


def _count(mapping, key, minimum=0):
    value = mapping.get(key)
    _require(type(value) is int and value >= minimum, "Invalid explicit count: " + key)
    return value


def plan_repairs(facts, *, verify_files=True):
    """Return the next bounded actions, without executing or accepting data.

    ``verify_files=False`` is for fixture/dry-run inspection only and disables
    submission eligibility. Budget counters are supplied by the durable caller;
    returned reservations are not actual expenditures.
    """
    _require(isinstance(facts, dict)
             and facts.get("schema") == "r2h_repair_facts_v1", "Unknown facts schema")
    reference = _binding(facts.get("reference"), verify_files)
    anchor = _binding(facts.get("anchor"), verify_files)
    bindings = {"reference": reference, "anchor": anchor}
    if facts.get("video") is not None:
        bindings["video"] = _binding(facts["video"], verify_files)
    budget, attempts = facts.get("budget", {}), facts.get("attempts", {})
    paid = _count(budget, "paid_calls_remaining")
    maxima = {key: _count(budget, key) for key in
              ("max_body_reviews", "max_anchor_repairs", "max_contact_reviews")}
    max_frames = _count(budget, "max_review_frames", 1)
    counts = {key: _count(attempts, key) for key in
              ("body_reviews", "anchor_repairs", "contact_reviews")}
    result = {
        "schema": "r2h_repair_plan_v1", "facts_sha256": canonical_sha256(facts),
        "evidence_bindings": bindings, "state": "needs_review", "actions": [],
        "blockers": [], "paid_calls_reserved": 0, "files_verified": verify_files,
        "allowed": {"video_submission": False, "first_frame_only_submission": False,
                    "training_handoff": False},
        "quality_release_approved": False, "training_admitted": False,
        "actions_executed": False,
    }

    def block(reason):
        if reason not in result["blockers"]:
            result["blockers"].append(reason)

    def action(kind, *, params=None, cost=0):
        if cost > paid - result["paid_calls_reserved"]:
            block("paid_call_budget_exhausted")
            return False
        result["actions"].append({
            "kind": kind, "execution_status": "planned",
            "paid_calls_reserved": cost, "parameters": params or {},
            "input_facts_sha256": result["facts_sha256"],
        })
        result["paid_calls_reserved"] += cost
        return True

    # Protocol recovery precedes any generation or semantic repair. Never retry
    # an unresolved paid POST merely because no usable output has been recorded.
    protocol = facts.get("protocol", {})
    status = protocol.get("state")
    _require(status in {"ok", "poll_pending", "submission_unknown", "runtime_error",
                        "schema_error"}, "Unknown protocol state")
    if status != "ok":
        block("protocol_" + status)
        if status == "submission_unknown":
            result["state"] = "submission_reconciliation"
            action("reconcile_existing_submission", params={"allow_paid_resubmit": False})
        elif status == "poll_pending":
            _require(isinstance(protocol.get("task_id"), str) and protocol["task_id"],
                     "Pending polling requires an existing task ID")
            result["state"] = "protocol_recovery"
            action("poll_existing_task", params={"task_id": protocol["task_id"],
                                                "allow_paid_resubmit": False})
        elif status == "schema_error":
            _require(type(protocol.get("lossless_shape_adapter_available")) is bool,
                     "Declare whether a lossless shape adapter is available")
            result["state"] = "protocol_recovery"
            if protocol["lossless_shape_adapter_available"]:
                bindings["raw_response"] = _binding(protocol.get("raw_response"), verify_files)
                action("apply_lossless_shape_adapter", params={
                    "preserve_raw": True, "allow_field_invention": False,
                    "allow_semantic_changes": False, "revalidate_original_schema": True})
            else:
                action("inspect_schema_failure", params={"allow_semantic_repair": False})
        else:
            result["state"] = "protocol_recovery"
            action("inspect_runtime_failure", params={"allow_paid_resubmit": False})
        return result

    hidden = facts.get("hidden_state", {})
    _require(type(hidden.get("required")) is bool
             and type(hidden.get("reveal_condition_present")) is bool
             and type(hidden.get("condition_transport_verified")) is bool,
             "Explicit hidden-state condition facts are required")
    if hidden["required"]:
        if not hidden["reveal_condition_present"]:
            block("required_hidden_state_not_conditioned")
            result["state"] = "condition_preparation"
            action("extract_reveal_or_terminal_anchor", params={
                "must_show_required_hidden_entities": True,
                "derive_from_bound_source": True, "do_not_invent_hidden_contents": True})
            return result
        bindings["reveal_condition"] = _binding(hidden.get("condition_artifact"), verify_files)
        if not hidden["condition_transport_verified"]:
            block("condition_transport_not_verified")
            result["state"] = "condition_preparation"
            action("verify_multi_anchor_condition_transport", params={
                "must_be_in_actual_provider_payload": True,
                "text_description_is_not_visual_condition": True,
                "paid_request_authorized_by_this_action": False})
            return result
        bindings["condition_transport_evidence"] = _binding(
            hidden.get("transport_evidence"), verify_files)

    inventory = facts.get("actor_inventory", {})
    maximum_hands = _count(inventory, "maximum_visible_hands", 1)
    scope = inventory.get("review_scope")
    _require(scope in {"none", "global", "localized"}, "Unknown body-review scope")
    review = inventory.get("review")
    if review is None:
        _require(scope == "none", "A review scope without an inventory is invalid")
        state = "not_checked"
    else:
        _require(scope != "none", "An inventory requires its actual review scope")
        _require(inventory.get("media_sha256") == anchor["sha256"],
                 "Actor inventory binds a different anchor")
        bindings["actor_review"] = _binding(inventory.get("evidence_artifact"), verify_files)
        if verify_files:
            saved_review = json.loads(Path(bindings["actor_review"]["path"]).read_text())
            if isinstance(saved_review, dict) and "review" in saved_review:
                saved_review = saved_review["review"]
            _require(saved_review == review, "Actor inventory differs from bound review artifact")
        state = body_decision(review, maximum_visible_hands=maximum_hands)["status"]
    if state != "inventory_within_budget":
        block("actor_inventory_" + state)
        if state == "rendering_plan_mismatch" and scope == "localized":
            result["state"] = "anchor_repair"
            if counts["anchor_repairs"] >= maxima["max_anchor_repairs"]:
                block("anchor_repair_attempt_limit")
            else:
                action("repair_anchor_body", cost=1, params={
                    "maximum_visible_hands": maximum_hands, "local_edit_only": True,
                    "preserve_task_objects": True, "preserve_original_artifacts": True,
                    "new_anchor_requires_new_body_and_task_review": True})
        else:
            result["state"] = "anchor_review"
            can_add_evidence = inventory.get("localized_review_can_add_evidence", False)
            _require(type(can_add_evidence) is bool, "Evidence availability must be Boolean")
            if counts["body_reviews"] >= maxima["max_body_reviews"]:
                block("body_review_attempt_limit")
            elif scope == "localized" and not can_add_evidence:
                block("no_new_body_evidence")
                action("request_manual_body_review", params={"unknown_remains_blocked": True})
            else:
                action("review_anchor_body_global" if scope == "none"
                       else "review_anchor_body_localized", cost=1, params={
                    "include_full_image_and_original_resolution_crops": scope != "none",
                    "maximum_visible_hands": maximum_hands,
                    "do_not_infer_hidden_limbs": True})
        return result

    contact = facts.get("contact")
    if contact is not None:
        _require("video" in bindings, "Contact review requires actual candidate media")
        _require(contact.get("media_sha256") == bindings["video"]["sha256"],
                 "Contact observations bind a different video")
        bindings["contact_review"] = _binding(contact.get("evidence_artifact"), verify_files)
        _require(isinstance(contact.get("unknown_event_ids"), list)
                 and all(isinstance(x, str) and x for x in contact["unknown_event_ids"]),
                 "Unknown contact events must be explicit")
        requests = contact.get("requests")
        _require(isinstance(requests, list), "Contact refinement requests must be a list")
        known_unknowns = set(contact["unknown_event_ids"])
        _require(all(request.get("query_id") in known_unknowns for request in requests),
                 "Refinement request does not belong to an unknown event")
        if known_unknowns:
            block("contact_events_unknown")
            result["state"] = "contact_review"
            if not requests:
                block("no_localizable_contact_interval")
                action("request_contact_localization", params={"paid_request_authorized": False})
            elif counts["contact_reviews"] >= maxima["max_contact_reviews"]:
                block("contact_review_attempt_limit")
            else:
                refinement = plan_refinement(
                    contact.get("frame_count"), contact.get("base_frame_indices", []),
                    requests, max_frames=max_frames)
                result["contact_refinement_plan"] = refinement
                if refinement["added_frame_count"] == 0:
                    block("no_additional_contact_frames")
                else:
                    action("review_dense_contact_frames", cost=1, params={
                        "refinement_plan": refinement, "preserve_base_coverage": True,
                        "use_actual_media_timestamps": True,
                        "selection_is_not_contact_verification": True,
                        "preserve_previous_reviews": True})
            return result

    pair = facts.get("pair_validation", {})
    _require(pair.get("state") in {"not_run", "passed", "failed"},
             "Explicit pair-validation state is required")
    if pair["state"] != "not_run":
        bindings["pair_validation"] = _binding(pair.get("evidence_artifact"), verify_files)
    if pair["state"] == "failed":
        result["state"] = "pair_repair"
        block("pair_validation_failed")
        action("repair_explicit_pair_manifest", params={
            "do_not_change_ground_truth_to_force_pass": True,
            "preserve_query_observation_action_identity": True,
            "revalidate_lineage_split_and_timebase": True})
    elif "video" not in bindings:
        result["state"] = "ready_for_submission" if verify_files else "inspection_only"
        result["allowed"]["video_submission"] = verify_files
        result["allowed"]["first_frame_only_submission"] = verify_files and not hidden["required"]
        if not verify_files:
            block("file_bindings_not_verified")
    else:
        result["state"] = "candidate_inspection"
        action("inspect_task_evidence_and_training_pair", params={
            "pair_already_validated": pair["state"] == "passed",
            "training_auto_start": False, "requires_independent_quality_release": True})
    # Passing checks only enables the next operation. It never grants training.
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("facts", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-verify-files", action="store_true")
    args = parser.parse_args(argv)
    result = plan_repairs(json.loads(args.facts.read_text()),
                          verify_files=not args.no_verify_files)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
