"""Connect scene proposals to local diagnostics and quality screening.

This runner performs no paid calls or video generation. Geometry/ROI correction
is an upstream stage. It records unresolved work rather than inventing ROIs,
hand budgets or contact labels. Existing generation/review evidence stays intact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from .common import file_sha256, json_sha256, write_json
from .interaction_triage import triage_interaction
from .quality_cascade import inspect_video
from .scene_proposals import apply_correction, binding, checked_path


def check_actor_budget(plan_binding, video_hash, budget):
    from .actor_plan_guard import validate_plan
    path = checked_path(plan_binding)
    plan = json.loads(path.read_text())
    if validate_plan(plan, video_hash) != budget:
        raise ValueError("Requested hand budget differs from its bound actor plan")
    return path


def choose_rois(proposal, patch=None):
    """Use uniquely identified interaction/background roles, never list order."""
    if proposal.get("target") is None:
        return None, ["target_video_missing"]
    candidates = (apply_correction(proposal, patch)["roi_candidates"] if patch
                  else proposal["target"]["roi_candidates"])
    roles, pending = {}, []
    for role in ("interaction_target", "background_reference"):
        matches = [r for r in candidates if r.get("role") == role
                   and r.get("status") in {"candidate", "vlm_reviewed_candidate"}
                   and r.get("box_xyxy_normalized") is not None]
        if len(matches) != 1:
            pending.append(role + "_requires_unambiguous_localization")
        else:
            roles[role] = matches[0]
    return (roles if not pending else None), pending


def condition_handoff(proposal):
    """Source-frame conditions, explicitly not generated-human timestamps."""
    frames = {f["frame_index_0based"]: f for f in proposal["source_keyframes"]}
    anchors = []
    for obj in proposal["hidden_state_proposals"]:
        if not obj["hidden_then_revealed"]:
            continue
        i = obj["maximum_visibility_frame"]
        anchors.append({
            "object_index": obj["object_index"],
            "condition_kind": "revealed_state_visual_anchor",
            "source_frame": frames.get(i),
            "source_event": obj,
            "status": "candidate" if i in frames else "needs_source_frame_extraction",
            "generated_human_frame_index": None,
        })
    return {
        "schema": "r2h_automatic_condition_handoff_v1",
        "revealed_state_conditions": anchors,
        "keyframes": proposal["source_keyframes"],
        "actor_plan": proposal["actor_plan_proposal"],
        "manual_roi_used": False,
        "generation_executed": False,
        "warning": "These are simulator-time candidates. Do not copy frame indices to a different video.",
    }


def boundary_handoff(report, video_hash):
    """Route local discontinuities to inspection, never to an automatic verdict."""
    if report["source"]["sha256"] != video_hash:
        raise ValueError("Boundary diagnostic targets different media")
    n = report["source"]["frames"]
    requests = []
    for event in sorted(report["candidates"],
                        key=lambda x: x["peak_robust_z"], reverse=True)[:2]:
        start, end = event["previous_frame"], event["frame_index"]
        if not (type(start) is int and type(end) is int and 0 <= start < end < n):
            raise ValueError("Invalid boundary evidence frame interval")
        requests.append({
            "reason": "local_boundary_discontinuity_candidate",
            "frame_range_inclusive": [max(0, start-2), min(n-1, end+2)],
            "candidate_transition": [start, end],
            "regions": event["regions"],
            "peak_robust_z": event["peak_robust_z"],
            "next_action": "Inspect native consecutive frames and compare source task states; "
                           "do not regenerate or reject solely from this heuristic.",
            "automatic_task_verdict": None,
        })
    return requests


def run(config_path):
    config_path = Path(config_path).resolve(strict=True)
    config = json.loads(config_path.read_text())
    output = Path(config["output"]).resolve()
    if output.exists():
        raise FileExistsError("Choose a new output; prior diagnostics are immutable")
    proposal_path = checked_path(config["proposal"]).resolve(strict=True)
    proposal = json.loads(proposal_path.read_text())
    if proposal.get("schema") != "r2h_scene_proposals_v1":
        raise ValueError("Unsupported proposal schema")
    for artifact in proposal["bindings"].values():
        checked_path(artifact)
    for frame in proposal["source_keyframes"]:
        checked_path(frame)
    if not proposal.get("target"):
        raise ValueError("A generated target video is required for this runner")
    video = checked_path(proposal["target"]["video"]).resolve(strict=True)
    checked_path(proposal["target"]["frame"])
    patch = None
    patch_path = None
    if config.get("correction_patch"):
        patch_path = checked_path(config["correction_patch"]).resolve(strict=True)
        patch = json.loads(patch_path.read_text())
    rois, pending = choose_rois(proposal, patch)
    camera_mode = config.get("camera_mode", "unknown")
    if camera_mode not in {"fixed", "egocentric", "unknown"}:
        raise ValueError("Invalid camera mode")
    budget = config.get("maximum_visible_hands")
    if budget is not None:
        if type(budget) is not int or budget < 1:
            raise ValueError("Invalid explicit visible-hand budget")
        if not config.get("actor_budget_source"):
            raise ValueError("Visible-hand budget requires evidence/provenance")
        check_actor_budget(config["actor_budget_source"], file_sha256(video), budget)
    evidence_paths = [checked_path(x) for x in config.get("quality_evidence", [])]
    output.mkdir(parents=True)
    record = {
        "schema": "r2h_automated_review_execution_v1",
        "config": binding(config_path), "proposal": binding(proposal_path),
        "correction_patch": binding(patch_path) if patch_path else None,
        "video": binding(video), "upstream_manual_roi_used": proposal["manual_roi_used"],
        "paid_calls": 0, "training_admitted": False,
        "quality_release_approved": False, "pending": pending,
    }
    handoff_path = output / "condition_handoff.json"
    write_json(handoff_path, condition_handoff(proposal))
    record["condition_handoff"] = binding(handoff_path)
    provenance = json.dumps({
        "proposal": binding(proposal_path),
        "patch": binding(patch_path) if patch_path else None,
        "scope": "automatically proposed 2D locations; not contact ground truth",
    }, sort_keys=True)
    background = None
    if camera_mode == "fixed" and rois:
        background = rois["background_reference"]["box_xyxy_normalized"]
    quality_dir = output / "quality"
    quality = inspect_video(
        video, quality_dir, origin="synthetic_human", camera_mode=camera_mode,
        background_roi=background, roi_provenance=provenance if background else None,
        evidence_paths=evidence_paths,
    )
    record["quality"] = binding(quality_dir / "quality.json")
    record["quality_decision"] = quality["decision"]
    pending.extend("quality_" + name + "_unresolved"
                   for name in quality["decision"]["unknown_dimensions"])
    # Runs independently of semantic ROI localization and hand-model availability.
    # A missing ROI must not disable detection of a pasted/abrupt terminal frame.
    from .boundary_discontinuity import diagnose
    boundary_dir = output / "boundary"
    try:
        boundary = diagnose(video, boundary_dir)
    except (ValueError, RuntimeError, subprocess.SubprocessError) as error:
        write_json(output / "boundary_error.json",
                   {"error_type": type(error).__name__, "error": str(error),
                    "task_outcome": "unknown"})
        record["boundary_error"] = binding(output / "boundary_error.json")
        pending.append("boundary_diagnostic_unavailable")
    else:
        record["boundary_diagnostic"] = binding(boundary_dir / "report.json")
        record["boundary_refinement_requests"] = boundary_handoff(boundary, file_sha256(video))
        if record["boundary_refinement_requests"]:
            pending.append("boundary_event_windows_need_visual_review")
    if camera_mode != "fixed":
        pending.append("fixed_camera_interaction_diagnostic_not_applicable")
    elif not rois:
        pending.append("local_diagnostic_waiting_for_ROI_correction")
    elif not config.get("hand_python") or not config.get("hand_model"):
        pending.append("local_hand_detector_not_configured")
    else:
        model = checked_path(config["hand_model"]).resolve(strict=True)
        diagnostic_dir = output / "interaction"
        # Preserve a venv's executable symlink: resolving it changes sys.prefix.
        interpreter = Path(config["hand_python"]).absolute()
        if not interpreter.is_file():
            raise FileNotFoundError(interpreter)
        command = [
            str(interpreter), "-m",
            "data_generation_methods.r2h_pipeline.contact_diagnostics",
            "--video", str(video), "--model", str(model), "--output", str(diagnostic_dir),
            "--handle-roi", *map(str, rois["interaction_target"]["box_xyxy_normalized"]),
            "--background-roi", *map(str, background),
            "--roi-provenance", provenance,
        ]
        with (output / "interaction.log").open("w") as log:
            result = subprocess.run(command, cwd=Path(__file__).resolve().parents[2],
                                    stdout=log, stderr=subprocess.STDOUT, timeout=300)
        if result.returncode:
            pending.append("interaction_detector_failed_see_bound_log")
            record["interaction_log"] = binding(output / "interaction.log")
        else:
            diagnostic_path = diagnostic_dir / "diagnostic.json"
            diagnostic = json.loads(diagnostic_path.read_text())
            if diagnostic["video"]["sha256"] != file_sha256(video):
                raise ValueError("Diagnostic targets different media")
            record["interaction"] = binding(diagnostic_path)
            record["interaction_summary"] = diagnostic["summary"]
            if budget is None:
                pending.append("actor_budget_unknown_no_extra_hand_decision")
            else:
                triage = triage_interaction(diagnostic_path, maximum_visible_hands=budget)
                triage_path = output / "interaction_triage.json"
                write_json(triage_path, triage)
                record["triage"] = binding(triage_path)
                record["refinement_requests"] = triage["refinement_requests"]
                if triage["refinement_requests"]:
                    pending.append("bounded_event_windows_need_visual_review")
    record["state"] = ("rejected_by_existing_evidence" if quality["decision"]["status"] == "reject"
                       else "awaiting_independent_semantic_pair_and_event_review")
    record["record_content_sha256"] = json_sha256(record)
    write_json(output / "STATUS.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config)
    print(json.dumps({"state": result["state"], "pending": result["pending"],
                      "paid_calls": 0, "quality_release_approved": False}))


if __name__ == "__main__":
    main()
