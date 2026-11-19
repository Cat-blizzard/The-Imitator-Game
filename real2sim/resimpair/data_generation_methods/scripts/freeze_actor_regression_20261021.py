"""Freeze the actual 2026-10-21 assistant visual audit, NOT run a vision model.

Observations below were authored after viewing the original two anchors and
seven sampled frames from each of eight archived videos. Do not reuse these
labels for different media. All source SHA256 values are verified against the
previous immutable experiment manifest/packet before freezing.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import shutil

from data_generation_methods.r2h_pipeline.common import file_sha256, read_json, write_json
from data_generation_methods.r2h_pipeline.single_actor_acceptance import CHECKS, RULES, RULE_VERSION, evaluate

REPO = Path(__file__).resolve().parents[2]
OLD = Path("/mnt/sdg/datasets/real_sim_human_pipeline/controlled_20261020e")
DEFAULT_OUTPUT = Path("/mnt/sdg/datasets/real_sim_human_pipeline/harness_20261021_v1/regression")


def binding(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": file_sha256(path)}


def author_review(media, two_people, scope, frames):
    """Serialize already-completed visual observations; no new inference."""
    if two_people:
        observation = (
            "Two separate seated torso/sleeve origins are visible at opposite image edges; "
            "one forearm and hand extends from each. The open space behind the table "
            "does not connect them into one body. This is direct two-operator evidence, "
            "not an inference from seeing two hands."
        )
        notes = {
            "person_count": ("fail", "Two separate operators are visibly represented."),
            "body_position": ("fail", "Independent torso/arm origins occupy opposite table sides."),
            "left_hand_ownership": ("unknown", "Anatomical left-hand ownership of one operator is not established."),
            "right_hand_ownership": ("unknown", "Anatomical right-hand ownership of that same operator is not established."),
            "body_structure": ("unknown", "Two visible people fail the task plan; this does not prove either person's anatomy impossible."),
            "task_roles": ("unknown", "No predeclared single-actor hand-role plan existed for this historical media."),
        }
    else:
        observation = (
            "Only one foreground hand and its sleeved forearm are visible in sampled cabinet frames. "
            "The shoulder, torso and other hand are outside the image; the view does not independently "
            "establish a complete single-operator ownership map."
        )
        notes = {name: ("unknown", "One visible hand is insufficient to verify this full actor/plan dimension.")
                 for name in CHECKS}
    return {
        "schema": "r2h_single_actor_review_v2", "rule_version": RULE_VERSION,
        "artifact": media, "plan_sha256": None, "scope": scope,
        "reviewer": {"id": "independent-audit-agent-20261021", "kind": "assistant_visual"},
        "evidence": [{
            "id": "actual_visual_observation", "kind": "direct_visual",
            "artifact_sha256": media["sha256"], "frame_indices": frames,
            "observation": observation,
        }],
        "checks": {name: {"status": status, "evidence_ids": ["actual_visual_observation"], "note": note}
                   for name, (status, note) in notes.items()},
        "limitations": [
            "Assistant directly inspected known development samples; no blind human study or automatic body detector ran.",
            "Video observations cover only the named frames, not every frame or physical contact truth.",
            "Historical media predates the single-actor body/hand-role plan; no plan is retroactively fabricated.",
        ],
    }


def freeze(output):
    output = Path(output)
    if (output / "manifest.json").exists():
        raise ValueError("Frozen manifest already exists; do not overwrite")
    packets_path = output / "sample_packets.json"
    packets = read_json(packets_path)["records"]
    experiment = read_json(OLD / "experiment.json")
    sources = {x["id"]: x for x in experiment["cases"]}
    initial = {x["case"] + "_" + x["arm"]: x
               for x in read_json(OLD / "audit_initial/INITIAL_AUDIT.json")["records"]}
    posthoc = {x["case"]: x for x in read_json(OLD / "audit_posthoc/POSTHOC_AUDIT.json")["records"]}
    snapshot_dir = output / "frozen_sources"
    snapshot_dir.mkdir(parents=True, exist_ok=False)
    source_names = (
        "r2h_pipeline/single_actor_acceptance.py",
        "r2h_pipeline/failure_regression.py",
        "r2h_pipeline/anchor_body_review.py",
        "r2h_pipeline/actor_plan_guard.py",
        "r2h_pipeline/boundary_discontinuity.py",
        "r2h_pipeline/schemas/single_actor_review_v2.schema.json",
        "PIPELINE_REPAIR_AND_CONTROLLED_20261020E.md",
        "CONTROLLED_INITIAL_RESULTS_20261020E.md",
    )
    snapshots = {}
    for relative in source_names:
        src = REPO / "data_generation_methods" / relative
        dest = snapshot_dir / src.name
        shutil.copyfile(src, dest)
        snapshots[relative] = binding(dest)
    write_json(output / "actor_rule_semantics.json", {
        "rule_version": RULE_VERSION, "checks": RULES,
        "unknown_policy": "Never authorizes video generation.",
        "observation_policy": "Direct visual/human evidence required; VLM yes alone is insufficient.",
        "scope": "This is an evidence gate, not an automatic visual detector.",
    })
    cases = []
    for task in ("apple", "book"):
        media = sources[task]["human_initial"]
        if binding(media["path"]) != media:
            raise ValueError("Original anchor changed")
        report = author_review(media, True, "anchor", [0])
        target = output / "actor_reviews" / f"{task}_common_anchor.json"
        write_json(target, report)
        cases.append({
            "case_id": f"{task}_common_anchor", "split": "development", "media": media,
            "lineage_groups": ["human:" + sources[task]["human_source"]["sha256"],
                               "physical:" + sources[task]["robot_trajectory"]["sha256"]],
            "actor_review": binding(target), "expected_actor_decision": "reject",
            "expected_combined_decision": "reject",
            "failure_types": ["multiple_operators"],
            "legacy_dimensions": {
                "maximum_visible_hands": 2, "observed_visible_hands_in_this_audit": 2,
                "person_count_requirement": "missing_from_old_protocol",
                "hand_inventory_is_not_person_acceptance": True,
            },
        })
    for packet in packets:
        case_id = packet["case_id"]
        task = case_id.split("_")[0]
        previous = (posthoc[task] if case_id.endswith(("_repair", "_redraw")) else initial[case_id])
        if previous["video"] != packet["media"]:
            raise ValueError("Sample packet media differs from immutable old audit")
        report = author_review(packet["media"], task != "cabinet", "sampled_video",
                               [x["frame_index"] for x in packet["samples"]])
        target = output / "actor_reviews" / f"{case_id}.json"
        write_json(target, report)
        actor_status = "reject" if task != "cabinet" else "needs_review"
        bad_identity = case_id in {"cabinet_A", "cabinet_A_redraw"}
        observation_dimensions = {
            "single_actor": actor_status,
            "task_goal_visible": "fail_terminal" if case_id == "apple_B" else "supported_in_sampled_scope",
            "object_identity": "fail" if bad_identity else "unknown_except_reviewed_task_object",
            "relative_size": "fail" if bad_identity else "unknown_except_reviewed_task_object",
            "critical_contact": "unknown_not_inferred_from_sparse_sampling",
            "full_clip_continuity": "unknown",
            "real_3d_physics": "unavailable_not_required_from_synthetic_RGB",
        }
        case = {
            "case_id": case_id, "split": "development", "media": packet["media"],
            "lineage_groups": ["human:" + sources[task]["human_source"]["sha256"],
                               "physical:" + sources[task]["robot_trajectory"]["sha256"]],
            "actor_review": binding(target), "expected_actor_decision": actor_status,
            "expected_combined_decision": "reject" if task != "cabinet" or bad_identity else "needs_review",
            "source_human": sources[task]["human_source"],
            "source_robot_trajectory": sources[task]["robot_trajectory"],
            "common_initial_anchor": sources[task]["human_initial"],
            "sampled_frame_evidence": packet["samples"],
            "contact_sheet": binding(packet["contact_sheet"]),
            "legacy_dimensions": {
                "review_status": previous.get("review_status", previous.get("vlm_status")),
                "quality_decision": previous["quality_decision"],
                "single_person_dimension": "not_verified_by_old_hand_budget",
            },
            "new_separate_dimensions": observation_dimensions,
            "failure_types": ["multiple_operators"] if task != "cabinet" else (
                ["hidden_object_identity", "hidden_object_relative_size"] if bad_identity else []),
            "trial_kind": "posthoc_repair" if case_id.endswith("_repair") else (
                "posthoc_same_request_redraw" if case_id.endswith("_redraw") else "initial"),
        }
        if task == "cabinet":
            case["object_reference"] = sources[task]["sim_terminal"]
            case["hidden_object_observation"] = (
                "At frames 144,191,192 a tall colorful rectangular grid occupies much of the upper "
                "compartment; source sim terminal shows a small cube with ample free space around it."
                if bad_identity else
                "At frames 144,191,192 a small multicolored object remains in the upper compartment, "
                "consistent with source scale in these views; exact 3D identity is not claimed."
            )
        if case_id in {"apple_B", "apple_B_repair"}:
            case["expected_boundary_candidate_frames"] = [192] if case_id == "apple_B" else []
            case["terminal_condition"] = binding(OLD / (
                "apple/B/human_terminal_01.png" if case_id == "apple_B"
                else "apple/B/posthoc_remaining_budget_01/human_terminal.png"))
            case["boundary_observation"] = (
                "Frame 191 apple is in image-right bowl; frame 192 it returns to image-left bowl while "
                "hands are withdrawn. The terminal condition also has apple in image-left bowl."
                if case_id == "apple_B" else
                "Frames 191 and 192 both keep apple in image-right bowl, as does corrected terminal "
                "condition. Two operators remain. Single random redraw does not isolate causality."
            )
        cases.append(case)
    # Reserve by metadata only. Historical source reviews are disclosed, not relabeled blind.
    reservation_source = REPO / "data_generation_methods/human_to_robot_runs/20261020_v1/single_cup_future_qualification_spec.json"
    reserved = read_json(reservation_source)["tasks"][0]["source_windows"][0]
    holdout = {
        "status": "prospective_source_reservation_only_no_generated_validation_samples",
        "source_spec": binding(reservation_source),
        "human_id": reserved["human_id"], "session": reserved["session"],
        "window": reserved["window"], "original_dataset_split": reserved["split"],
        "research_role": "reserved_validation_candidate",
        "source_visuals_opened_in_this_actor_development": False,
        "previous_task_semantic_review_exists": True,
        "globally_blind_or_unseen_claimed": False,
        "robot_layout": None,
        "remaining_requirements": [
            "Freeze a compatible cup/lid asset and previously unused physical layout before evaluation.",
            "Exclude the entire source session and any derived physical execution from development.",
            "Obtain a new independent source/layout for apples or books if cup physics is not yet qualified.",
        ],
    }
    manifest = {
        "schema": "r2h_failure_regression_v1", "actor_rule_version": RULE_VERSION,
        "created_date": "2026-10-21",
        "actor_gate_source": snapshots["r2h_pipeline/single_actor_acceptance.py"],
        "actor_gate_schema": snapshots["r2h_pipeline/schemas/single_actor_review_v2.schema.json"],
        "frozen_sources": snapshots, "rule_semantics": binding(output / "actor_rule_semantics.json"),
        "source_experiment": binding(OLD / "experiment.json"),
        "old_initial_audit": binding(OLD / "audit_initial/INITIAL_AUDIT.json"),
        "old_posthoc_audit": binding(OLD / "audit_posthoc/POSTHOC_AUDIT.json"),
        "sample_packets": binding(packets_path), "cases": cases,
        "boundary_regression": {
            "detector_source": snapshots["r2h_pipeline/boundary_discontinuity.py"],
            "config": read_json(OLD / "boundary_diagnostics/development_protocol.json")["config"],
            "development_protocol": binding(OLD / "boundary_diagnostics/development_protocol.json"),
        },
        "reserved_validation": holdout,
        "candidate_comparison": {
            "rule": "Baseline and candidate must use this exact same rule revision; do not compare old-rule acceptance rates.",
            "new_candidate_media": 0,
            "existing_apple_posthoc_repair_under_same_rule": "reject_two_operators",
            "holdout_evaluated": 0,
        },
        "statistics_scope": {
            "two_anchors_directly_inspected": 2, "archived_videos_sampled": 8,
            "sampled_video_frames_viewed": 56, "actor_visual_detector_run": False,
            "actor_annotation_replay_is_not_detection_rate": True,
            "all_cases_known_development": True,
            "physical_trajectories_new": 0, "paid_calls_new": 0,
        },
    }
    for case in cases:
        if evaluate(read_json(case["actor_review"]["path"]))["status"] != case["expected_actor_decision"]:
            raise AssertionError("Authored expectation contradicts independent gate")
    write_json(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print("Frozen cases:", len(freeze(args.output)["cases"]))
