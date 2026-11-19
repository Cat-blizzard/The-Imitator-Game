"""Frozen conditioning comparison with one development interface qualification.

Baseline and candidate use the same qualified native episode, shared human
image, task text, output duration and unchanged visual rules. Image-role changes
are an explicit confound: this evaluates a conditioning package, not video alone.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np

from .core import activate, bind, checked, implementation, read, replace_state, require, write
from .pipeline import release_id
from .motion_reference import ComparisonBudget, REFERENCE_INSTRUCTION, derivative, upload_reference, submit_reference


def unusable_review(budget, request_path):
    """Terminalize a complete but invalid response; never synthesize a review.

    Ambiguous transport failures and partial responses still stop the campaign.
    This helper only isolates the schema failure already recorded by the frozen
    model adapter, and independently verifies it against the original schema.
    """
    from jsonschema import Draft202012Validator
    request_path = Path(request_path)
    if not request_path.exists(): return None
    stage = budget.directory("review", request_path)
    failure_path = stage / "failure.json"
    if not failure_path.exists(): return None
    failure = read(failure_path)
    require(failure.get("code") == "review_schema_violation"
            and failure.get("provider_outcome") == "definite_failure"
            and failure.get("provider_response_complete") is True
            and not (stage / "result.json").exists(), "review_failure_requires_separate_recovery")
    metadata = read(stage / "request_metadata.json")
    started = read(stage / "call_started.json")
    require(metadata["kind"] == started["kind"] == "review"
            and metadata["request"] == started["request"] == bind(request_path),
            "failed_review_request_binding_mismatch")
    raw = read(stage / "provider_response.json")
    candidates = raw.get("candidates", [])
    require(len(candidates) == 1 and candidates[0].get("finishReason") == "STOP",
            "failed_review_must_have_complete_provider_finish")
    body = "".join(part.get("text", "") for part in candidates[0].get("content", {}).get("parts", [])
                   if not part.get("thought")).strip()
    if body.startswith("```"): body = "\n".join(body.splitlines()[1:-1])
    value = json.loads(body)
    errors = list(Draft202012Validator(read(request_path)["response_schema"]).iter_errors(value))
    require(errors, "schema_failure_not_reproduced")
    return {"status": "review_output_unusable", "quality": "unknown", "passed": False,
        "request": bind(request_path), "failure": bind(failure_path),
        "raw_response": bind(stage / "provider_response.json"),
        "schema_errors": [{"path": list(e.absolute_path), "validator": e.validator} for e in errors],
        "automatic_retry": False, "model_response_modified": False,
        "acceptance_rule_changed": False, "call_and_charge_preserved": True}


def _local_review_error(exc):
    """Stable validation identity without copying the entire model response."""
    from jsonschema import ValidationError
    from .core import digest
    if isinstance(exc, ValidationError):
        return {"type": "ValidationError", "validator": exc.validator,
                "path": list(exc.absolute_path), "schema_path": list(exc.absolute_schema_path),
                "message_sha256": digest(exc.message)}
    if type(exc) is ValueError:
        return {"type": "ValueError", "code": str(exc).split(":", 1)[0],
                "message_sha256": digest(str(exc))}
    return None


def _local_review_context(exc, request):
    """Only actual frozen visual-validator frames can supply replay arguments.

    Do not convert an arbitrary local exception into a content-quality result.
    Captured arguments are checked against the request's acceptance schema and
    saved with the failure so restart does not need another provider operation.
    """
    from . import review
    if _local_review_error(exc) is None:
        return None
    trace = exc.__traceback__
    while trace is not None:
        frame = trace.tb_frame
        if frame.f_code in {review.normalize_wire.__code__, review.validate.__code__}:
            local = frame.f_locals
            context = {k: local.get(k) for k in ["checks", "census_targets", "hand_requirements"]}
            if frame.f_code is review.validate.__code__:
                context.update(supplied_frames=list(local["supplied_frames"]), edges=list(local["edges"]))
            else:
                if request.get("schema") == "resimpair_anchor_review_request_v4":
                    frames = [0]
                elif "frames" in request:
                    frames = [f["frame_index"] for f in read(checked(request["frames"]))["frames"]]
                else:
                    return None
                context.update(supplied_frames=frames, edges=[])
            return json.loads(json.dumps(context, allow_nan=False))
        trace = trace.tb_next
    return None


def unusable_local_review(budget, request_path, error=None):
    """Retire one complete, billed, locally invalid visual review as unknown.

    This does not alter the provider result or any validator. A normal negative
    gate stays a negative gate. Missing/partial responses, corrupt bindings and
    non-validation errors still stop instead of being silently retired.
    """
    from jsonschema import Draft202012Validator, ValidationError
    from . import review, hand_review
    request_path = Path(request_path)
    if not request_path.exists():
        return None
    stage = budget.directory("review", request_path)
    receipt_path = stage / "local_validation_failure.json"
    prior = read(receipt_path) if receipt_path.exists() else None
    if error is None and prior is None:
        return None
    request = read(request_path)
    context = prior["validation_context"] if prior else _local_review_context(error, request)
    if context is None:
        return None
    require((stage / "result.json").exists() and not (stage / "failure.json").exists(),
            "local_review_requires_completed_paid_result")
    metadata = read(stage / "request_metadata.json")
    started = read(stage / "call_started.json")
    require(metadata["kind"] == started["kind"] == "review"
            and metadata["request"] == started["request"] == bind(request_path),
            "local_review_request_binding_mismatch")
    validators = {Path(m.__file__).name: bind(Path(m.__file__)) for m in [review, hand_review]}
    for name, bound in validators.items():
        require(request["implementation"][name]["sha256"] == bound["sha256"],
                "local_review_validator_changed:" + name)
        checked(request["implementation"][name])
    raw = read(stage / "provider_response.json")
    candidates = raw.get("candidates", [])
    require(len(candidates) == 1 and candidates[0].get("finishReason") == "STOP",
            "local_review_must_have_complete_provider_finish")
    body = "".join(part.get("text", "") for part in candidates[0].get("content", {}).get("parts", [])
                   if not part.get("thought")).strip()
    if body.startswith("```"):
        body = "\n".join(body.splitlines()[1:-1])
    observations = json.loads(body)
    result = read(stage / "result.json")
    require(result.get("status") == "succeeded" and result.get("review") == observations,
            "local_review_raw_and_paid_result_differ")
    Draft202012Validator(request["response_schema"]).validate(observations)
    targets, hands = context["census_targets"], context["hand_requirements"]
    supplied = context["supplied_frames"]
    require(request["response_schema"] == review.wire_schema(targets is not None, bool(hands))
            and request["acceptance_schema"] == review.schema(context["checks"],
                supplied if targets is not None else None, targets or (), hands),
            "local_review_context_differs_from_bound_schema")
    expected_frames = ([0] if request.get("schema") == "resimpair_anchor_review_request_v4" else
                       [f["frame_index"] for f in read(checked(request["frames"]))["frames"]])
    require(supplied == expected_frames, "local_review_supplied_frames_changed")
    try:
        normalized = review.normalize_wire(observations, context["checks"], targets, hands)
        review.validate(normalized, context["checks"], supplied, context["edges"], targets, hands)
    except (ValueError, ValidationError) as reproduced:
        signature = _local_review_error(reproduced)
        require(signature is not None, "local_review_unrecognized_validation_error")
    else:
        raise ValueError("local_review_validation_failure_not_reproduced")
    if error is not None:
        require(signature == _local_review_error(error), "local_review_validation_error_did_not_match")
    value = {"status": "review_output_unusable", "quality": "unknown", "passed": False,
        "reason": "complete_paid_response_failed_local_visual_validation",
        "request": bind(request_path), "result": bind(stage / "result.json"),
        "raw_response": bind(stage / "provider_response.json"),
        "call_started": bind(stage / "call_started.json"), "request_metadata": bind(stage / "request_metadata.json"),
        "validation_context": context, "validation_error": signature, "validators": validators,
        "automatic_retry": False, "model_response_modified": False,
        "acceptance_rule_changed": False, "call_and_charge_preserved": True}
    if prior is not None:
        require(prior == value, "local_review_failure_receipt_changed")
    else:
        write(receipt_path, value)
    return {**value, "local_failure": bind(receipt_path)}


def review_or_unknown(budget, request_path, operation):
    prior = unusable_review(budget, request_path)
    if prior: return None, prior
    prior = unusable_local_review(budget, request_path)
    if prior: return None, prior
    try:
        return operation(), None
    except Exception as exc:
        if getattr(exc, "code", None) == "review_schema_violation":
            failure = unusable_review(budget, request_path)
        else:
            failure = unusable_local_review(budget, request_path, exc)
        if failure is None: raise
        return None, failure


def prepared_contract(root, selection, key):
    from .mechanism_conditions import compile_candidate
    from .media import packet
    from .compiler import completed_contract
    folder = Path(root) / "compiled" / release_id() / key
    contract = compile_candidate(selection)
    write(folder / "contract.json", contract)
    write(folder / "evidence_plan.json", contract["human_evidence_plan"])
    require(contract["human_evidence_plan"]["declaration_coverage_complete"], "unsupported_task_evidence")
    bundle = read(checked(selection["execution"]))
    with np.load(checked(bundle["measurements"]), allow_pickle=False) as archive:
        times = archive["timestamps_seconds"].copy()
    cameras = {}
    for name, camera in bundle["cameras"].items():
        dest = folder / (name + "_frames")
        decoded = packet(checked(camera["video"]), dest, frame_budget=9)
        require(decoded["decoded_frames"] == camera["frame_count"] == len(times)
                and np.allclose(decoded["all_native_pts_seconds"], times, atol=1e-5, rtol=0), "native_timeline_mismatch")
        cameras[name] = bind(dest / "packet.json")
    write(folder / "media_validation.json", {"contract": bind(folder / "contract.json"),
        "cameras": cameras, "both_fully_decoded": True})
    completed_contract(folder / "contract.json")
    return bind(folder / "contract.json")


def compile_inputs(root):
    from .paid_pilot import compact_video_prompt
    from .prompts import anchor_prompt
    root = Path(root);exp = read(root / "experiment.json")
    for binding in exp["immutable_evaluators"].values(): checked(binding)
    ancestor = Path(checked(exp["development_qualification"]["ancestor"])).parent
    old = read(ancestor / "experiment.json")
    selected = next(s for s in old["selections"] if s["case"]["id"] == exp["development_qualification"]["case_id"])
    old_record = next(r for r in read(ancestor / "compiled/human_r4/index.json")["records"] if r["case"] == selected["case"]["id"])
    anchor = ancestor / "anchors" / old_record["anchor_id"]
    require(read(anchor / "review.json")["gate"]["passed_in_sampled_visual_scope"], "development_anchor_not_qualified")
    contract_ref = prepared_contract(root, selected, "qualification_hook")
    contract = read(checked(contract_ref))
    write(root / "compiled" / release_id() / "qualification_record.json", {"key": "qualification_hook", "phase": "development",
        "case": selected["case"]["id"], "contract": contract_ref, "human_image": bind(anchor / "human.png"),
        "anchor_review": bind(anchor / "review.json"), "base_prompt": compact_video_prompt(contract),
        "method": "motion_reference", "old_generation_prompt_identical": compact_video_prompt(contract) == read(checked(old_record["video_prompt"]))["prompt"]})
    audit_ref = bind(root / "audits/holdout_acceptance_v3.json")
    audit = read(checked(audit_ref))
    require(audit["acceptance_version"] == "novel_mechanism_acceptance_v3", "unexpected_native_acceptance")
    require(audit["counts"]["negative_controls_rejected"] == audit["counts"]["negative_controls"], "native_negative_control_failed")
    records = []
    for row in exp["rows"]:
        cid = row["case"]["id"]
        accepted = next(x for x in audit["cases"] if x["case"] == cid)
        if not accepted["independent_qualified"]:
            records.append({**row, "native_qualified": False, "independent_native_audit": audit_ref})
            continue
        result = read(checked(accepted["result"]))
        selection = {**row, "execution": accepted["execution"], "physical_contract": result["physics_contract"],
            "rule": audit["acceptance_version"], "independent_audit": audit_ref}
        contract_ref = prepared_contract(root, selection, cid)
        contract = read(checked(contract_ref))
        anchor = root / "anchors" / cid
        write(anchor / "image_request.json", {"schema": "resimpair_initial_request_v4",
            "prompt": anchor_prompt(contract["initial_scene_information"]), "model_config": bind(root / "config.json"),
            "images": [contract["initial_image"]], "initial_scene": contract["initial_scene_information"],
            "implementation": implementation(), "task_specific_prompt_override": None})
        records.append({**row, "native_qualified": True, "contract": contract_ref,
            "anchor_directory": str(anchor), "base_prompt": compact_video_prompt(contract)})
    write(root / "compiled" / release_id() / "index.json", {"records": records, "frozen_denominator": 4,
        "same_evaluator_both_methods": True, "video_only_causal_effect_claimed": False})
    return {"prepared_native_cases": sum(x["native_qualified"] for x in records), "qualification_prepared": True}


def run_video(root, budget, record, method, human_image, anchor_review, *, poll_deadline_seconds=None,
              wire_schema_adapter=None):
    from .compiler import completed_contract
    from .models import submit_video, poll_video
    from . import mechanism_review
    from .review_transport import WIRE_SCHEMA_ADAPTER
    require(wire_schema_adapter in {None, WIRE_SCHEMA_ADAPTER}, 'unsupported_wire_schema_adapter')
    root = Path(root)
    poll_seconds = (read(root / "experiment.json")["per_video_poll_deadline_seconds"]
                    if poll_deadline_seconds is None else poll_deadline_seconds)
    require(type(poll_seconds) in (int, float) and 0 < poll_seconds <= 900, "invalid_explicit_video_poll_deadline")
    key = record.get("key", record["case"]["id"] if isinstance(record["case"], dict) else record["case"])
    directory = root / "runs" / key / method / release_id()
    output = directory / "outcome.json"
    if output.exists(): return read(output)
    previous = list(directory.parent.glob("*/outcome.json"))
    require(len(previous) <= 1, "multiple_terminal_comparison_results")
    if previous:
        old = read(previous[0])
        require(old["contract"] == record["contract"] and old["human_image"] == human_image
                and old["method"] == method, "prior_comparison_inputs_changed")
        return old
    require(not [p for p in directory.parent.glob("*/video_request.json") if p.parent != directory],
            "unfinished_method_requires_its_original_release")
    contract = completed_contract(checked(record["contract"]))
    request_path = directory / "video_request.json"
    settings = {"duration": contract["generation_information"]["duration_seconds"], "resolution": "720p",
        "ratio": "adaptive", "generate_audio": False}
    request = {"schema": "motion_conditioning_comparison_request_v1", "method": method,
        "contract": record["contract"], "model_config": bind(root / "config.json"),
        "implementation": implementation(), "settings": settings, "first_frame_review": anchor_review,
        "automatic_fallback": False, "generation_scope": contract["generation_scope"]}
    if wire_schema_adapter is not None:
        request['review_wire_schema_adapter'] = wire_schema_adapter
    if method == "baseline":
        request.update(prompt=record["base_prompt"], first_frame=human_image)
        submit = submit_video
    else:
        require(method == "motion_reference", "unknown_comparison_method")
        derived = derivative(contract, root / "motion_inputs" / key / "reference.mp4")
        remote = upload_reference(root, derived)
        request.update(prompt=REFERENCE_INSTRUCTION+record["base_prompt"], human_image=human_image,
            reference_video=derived["video"], remote_reference=bind(root / "uploads" / derived["video"]["sha256"] / "receipt.json"))
        submit = submit_reference
    write(request_path, request)
    submission = submit(root, budget, request_path)
    write(directory / "submission.json", submission)
    deadline = time.monotonic()+poll_seconds
    while True:
        state = poll_video(root, submission, directory / "human.mp4")
        replace_state(directory / "progress.json", {"status": state["status"], "task_id": submission["task_id"],
            "updated_unix": time.time(), "resubmit": False})
        if state["status"] in {"downloaded", "failed", "expired", "cancelled"}:
            budget.settle_video(submission["stage"])
            break
        if time.monotonic() > deadline:
            return {"status": "waiting_existing_video", "key": key, "method": method, "submission": submission}
        time.sleep(8)
    result = {"key": key, "method": method, "contract": record["contract"], "human_image": human_image,
        "anchor_review": anchor_review, "submission": submission, "source_coverage": "unverified",
        "formal_real_sim_human_pair": False, "training_admitted": False, "manual_semantic_overrides": 0}
    if state["status"] != "downloaded":
        result.update(status="video_provider_failed", provider_status=state["status"], video=None)
    else:
        result["video"] = state["video"]
        review_options = {} if wire_schema_adapter is None else {'wire_schema_adapter': wire_schema_adapter}
        review, unavailable = review_or_unknown(budget, directory / "review_request.json",
            lambda: mechanism_review.video(root, budget, directory, contract, **review_options))
        if unavailable:
            write(directory / "unusable_review.json", unavailable)
            result.update(status="generated_video_review_unknown", gate=None,
                review_failure=bind(directory / "unusable_review.json"))
        else:
            result.update(status="sampled_visual_pass" if review["gate"]["passed_in_sampled_visual_scope"] else "generated_video_rejected",
                gate=review["gate"], video_review=bind(directory / "review.json"))
    write(output, result)
    return result


def run(root, phase, input_release=None):
    from .billing import snapshot, report
    from .models import generate_image
    from . import review
    root = Path(root);release = release_id()
    input_release = input_release or release
    from .core import identifier
    identifier(input_release)
    manifest = read(root / "releases" / input_release / "release.json")
    require(manifest["release"] == input_release, "input_release_manifest_mismatch")
    for bound in manifest["files"]: checked(bound)
    if input_release != release:
        for binding in read(root / "experiment.json")["immutable_evaluators"].values(): checked(binding)
        write(root / "resume" / (release + ".json"), {"input_release": input_release,
            "frozen_inputs": bind(root / "compiled" / input_release / "index.json"),
            "executor_release": bind(root / "releases" / release / "release.json"),
            "reason": "isolate complete schema-invalid review as unknown and continue other frozen cases",
            "resubmit_failed_review": False, "change_acceptance_or_generation_prompts": False})
    with (root / ".paid_batch.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget = ComparisonBudget(root)
        snapshot(root, root / "billing/campaign_before.json")
        try:
            replace_state(root / "progress.json", {"status": "running", "phase": phase, "pid": os.getpid(), "release": release})
            if phase == "qualification":
                record = read(root / "compiled" / input_release / "qualification_record.json")
                result = run_video(root, budget, record, "motion_reference", record["human_image"], record["anchor_review"])
                if result["status"] == "waiting_existing_video":
                    replace_state(root / "progress.json", {"status": "waiting_existing_video", "phase": phase,
                        "pid": None, "release": release, "resubmit": False, "updated_unix": time.time()})
                    return result
                write(root / "qualification_outcome.json", result)
                report(root, root / "billing/qualification_usage.json")
            else:
                require(phase == "comparison", "invalid_comparison_phase")
                decision = read(root / "expansion_decision.json")
                require(decision["qualification_outcome"] == bind(root / "qualification_outcome.json")
                    and decision["proceed_to_frozen_comparison"] is True, "reference_route_not_qualified_for_expansion")
                rows = []
                for record in read(root / "compiled" / input_release / "index.json")["records"]:
                    if not record["native_qualified"]:
                        rows.append({"case": record["case"]["id"], "status": "native_rejected_both_methods_unattempted"})
                        continue
                    contract = read(checked(record["contract"]))
                    anchor = Path(record["anchor_directory"])
                    receipt = generate_image(root, budget, anchor / "image_request.json", anchor / "human.png")
                    initial, unavailable = review_or_unknown(budget, anchor / "review_request.json",
                        lambda: review.anchor(root, budget, anchor, receipt,
                            contract["initial_scene_information"], contract["initial_image"]))
                    if unavailable:
                        write(anchor / "unusable_review.json", unavailable)
                        rejected = {"case": record["case"]["id"],
                            "status": "initial_image_review_unknown_both_methods_unattempted",
                            "anchor": receipt["image"], "review_failure": bind(anchor / "unusable_review.json")}
                        write(anchor / "paired_rejection.json", rejected);rows.append(rejected)
                        print(json.dumps({"case": record["case"]["id"], "status": rejected["status"],
                            "usage": budget.usage()}), flush=True)
                        continue
                    if not initial["gate"]["passed_in_sampled_visual_scope"]:
                        rejected = {"case": record["case"]["id"], "status": "initial_image_rejected_both_methods_unattempted",
                            "anchor_review": bind(anchor / "review.json")}
                        write(anchor / "paired_rejection.json", rejected);rows.append(rejected)
                        continue
                    for method in record["method_order"]:
                        value = run_video(root, budget, record, method, receipt["image"], bind(anchor / "review.json"))
                        rows.append(value)
                        print(json.dumps({"case": record["case"]["id"], "method": method, "status": value["status"], "usage": budget.usage()}), flush=True)
                        require(value["status"] != "waiting_existing_video", "waiting_existing_video_resume_no_new_post")
                result = {"schema": "cross_mechanism_comparison_results_v1", "rows": rows,
                    "frozen_native_cases": 4, "planned_method_attempts": 8, "formal_pairs": 0, "training_admitted": False}
                write(root / "comparison_results.json", result)
                snapshot(root, root / "billing/campaign_after.json")
                report(root, root / "billing/usage_report.json")
            replace_state(root / "progress.json", {"status": "phase_complete", "phase": phase, "pid": None,
                "release": release, "updated_unix": time.time()})
            return {"phase": phase, "status": result.get("status", "complete"), "usage": budget.usage()}
        except Exception as exc:
            replace_state(root / "progress.json", {"status": "stopped_on_exception", "phase": phase, "pid": None,
                "release": release, "exception_type": type(exc).__name__, "code": getattr(exc, "code", "operation_failed"),
                "updated_unix": time.time(), "automatic_retry": False})
            raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["compile", "qualification", "comparison"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--input-release", help="Reuse immutable compiled inputs after an orchestration-only recovery")
    a = p.parse_args();activate()
    print(compile_inputs(a.root) if a.command == "compile" else run(a.root, a.command, a.input_release))
