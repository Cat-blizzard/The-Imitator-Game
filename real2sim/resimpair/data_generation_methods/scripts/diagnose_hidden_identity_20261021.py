"""Known-cabinet-failure study with explicit reviewed ROI inputs; no paid calls."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_generation_methods.r2h_pipeline import hidden_identity_extent as candidate
from data_generation_methods.r2h_pipeline.common import write_json
from data_generation_methods.r2h_pipeline.scene_proposals import binding, checked_path, frame_at


BASE = Path("/mnt/sdg/datasets/real_sim_human_pipeline")


def run(root):
    root = Path(root)
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    run_dir = root / "run_01"
    run_dir.mkdir(exist_ok=False)
    proposal_path = BASE / "automation_20261020d/scene_proposals/cabinet_new_v2/proposal.json"
    proposal = json.loads(proposal_path.read_text())
    sim_video = checked_path(proposal["bindings"]["video"])
    qa_path = checked_path(proposal["bindings"]["visibility"])
    _, sim_image = frame_at(sim_video, 498, run_dir / "frames/sim_000498.png")
    sim_image.update(video=binding(sim_video), timestamp_seconds=24.9)
    with np.load(qa_path, allow_pickle=False) as qa:
        sim_box = (qa["projected_bounds_224"][498, 0] / 224 * 512).tolist()
    source = {
        "image": sim_image, "object_box_xyxy_px": sim_box,
        "object_visibility": "fully_visible",
        "contexts_xyxy_px": {"upper_compartment": [257, 183, 316, 228],
                              "fixed_cabinet_body": [157, 98, 355, 291]},
        "geometry_evidence": {"visibility": binding(qa_path), "frame_index": 498, "object_index": 0},
        "object_roi_provenance": "Automatically read bound simulator projected collision hull; source visible instance pixels = 376",
        "context_roi_provenance": "Codex manually localized actual compartment/body borders after direct view_image inspection; not automatic detection",
    }
    # These bounds were manually localized from the actual three frames of
    # each clip, after the coarse thresholds and uncertainty were frozen.
    # The colored extent is a conservative visible lower bound for A, because
    # the opening door occludes part of the changed content.
    specs = {
        "cabinet_A_original": {
            "folder": "A/attempt_01", "object": [499, 341, 593, 463],
            "upper": [486, 339, 627, 469], "visibility": "visible_extent_lower_bound",
            "observation": "Visible colored grid occupies much of the upper cavity; right edge partly behind opening door.",
        },
        "cabinet_A_redraw": {
            "folder": "A/posthoc_remaining_budget_01", "object": [502, 341, 578, 438],
            "upper": [486, 339, 617, 443], "visibility": "visible_extent_lower_bound",
            "observation": "Visible tall grid remains in the same upper cavity; right edge partly behind opening door.",
        },
        "cabinet_B": {
            "folder": "B/attempt_01", "object": [495, 375, 533, 414],
            "upper": [482, 340, 598, 430], "visibility": "fully_visible",
            "observation": "Small cube-like content with dark surrounding space is visible; this observation alone does not prove identity.",
        },
    }
    rows, annotations = [], []
    started = time.perf_counter()
    for case, spec in specs.items():
        video = BASE / "controlled_20261020e/cabinet" / spec["folder"] / "candidate.mp4"
        for index in protocol["target_frames"]:
            _, image = frame_at(video, index, run_dir / "frames" / f"{case}_{index:06d}.png")
            image.update(video=binding(video), timestamp_seconds=index / 24.)
            annotation = {
                "schema": "reviewed_hidden_content_correspondence_v1", "sample_id": f"{case}_{index}",
                "source": source,
                "target": {
                    "image": image, "object_box_xyxy_px": spec["object"],
                    "contexts_xyxy_px": {"upper_compartment": spec["upper"],
                                          "fixed_cabinet_body": [300, 173, 674, 548]},
                    "object_visibility": spec["visibility"],
                    "roi_provenance": "Codex manual localization after direct image inspection; bounds intentionally reused only where visible extents remain within the declared two-pixel uncertainty",
                },
                "correspondence": {
                    "status": "reviewed_semantic_slot", "reviewer": "Codex multiview agent, method-aware posthoc visual review",
                    "source_image_sha256": sim_image["sha256"], "target_image_sha256": image["sha256"],
                    "source_slot": "contents_of_cabinet_upper_right_compartment",
                    "target_slot": "contents_of_cabinet_upper_right_compartment",
                    "same_context_instance": True, "comparable_view": True,
                    "evidence": "Cabinet body, fixed marker panel, cavity borders and table arrangement correspond in direct full-image comparison. " + spec["observation"],
                    "same_object_identity": "unknown; this diagnostic tests contradiction rather than assuming identity",
                },
                "known_failure_study": True, "not_independent_sample_within_same_video": True,
            }
            annotations.append(annotation)
            write_json(run_dir / "annotations" / f"{annotation['sample_id']}.json", annotation)
            result = candidate.diagnose_pair(annotation, protocol, run_dir / "diagnostics" / annotation["sample_id"])
            rows.append({"case": case, "frame_index": index, "status": result["status"],
                         "strong_flags": result.get("strong_flags", []),
                         "result": binding(run_dir / "diagnostics" / annotation["sample_id"] / "diagnostic.json")})
    negatives = {}
    for name in protocol["negative_controls"]:
        annotation = copy.deepcopy(annotations[0])
        annotation["sample_id"] = "negative_" + name
        if name == "object_context_ROI_swapped":
            a = annotation["target"]
            a["object_box_xyxy_px"], a["contexts_xyxy_px"]["upper_compartment"] = (
                a["contexts_xyxy_px"]["upper_compartment"], a["object_box_xyxy_px"])
        elif name == "incorrect_semantic_slot_correspondence":
            annotation["correspondence"]["target_slot"] = "pear_on_table"
        elif name == "reversed_coordinates":
            annotation["target"]["object_box_xyxy_px"] = [593, 341, 499, 463]
        elif name == "fully_occluded_object":
            annotation["target"]["object_visibility"] = "fully_occluded"
        elif name == "unreviewed_correspondence":
            annotation["correspondence"]["status"] = "automatic_candidate_only"
        elif name == "changed_media_binding":
            annotation["target"]["image"]["sha256"] = "0" * 64
        result = candidate.diagnose_pair(annotation, protocol, run_dir / "negative_controls" / name)
        negatives[name] = {"status": result["status"], "reason": result.get("reason"),
                           "conservative_unknown": result["status"].startswith("unknown"),
                           "identity_pass": result["identity_pass"]}
    report = {
        "schema": "hidden_identity_extent_known_failure_study_v1", "protocol": binding(protocol_path),
        "source_proposal": binding(proposal_path), "candidate_code": binding(candidate.__file__),
        "study_code": binding(__file__), "cases": rows, "negative_controls": negatives,
        "seconds": time.perf_counter() - started, "paid_calls": 0, "gpu_used": False,
        "independent_source_executions": 1, "generated_videos": 3, "frame_observations": 9,
        "automatically_verified_object_correspondences": 0,
        "manual_target_roi_and_slot_reviews": 9,
        "identity_passes": 0,
        "claim": "Semi-automatic scale contradiction diagnostic on known examples; no general detection-rate claim.",
        "next_validation": "Freeze protocol before new sources; independently annotate object/context uncertainty and audit false flags, misses and correspondence unknowns.",
    }
    write_json(run_dir / "report.json", report)
    frozen = run_dir / "frozen_sources"
    frozen.mkdir()
    for path in (Path(candidate.__file__), Path(__file__)):
        (frozen / path.name).write_bytes(path.read_bytes())
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    report = run(args.root)
    print(json.dumps({"cases": [(r["case"], r["frame_index"], r["status"]) for r in report["cases"]],
                      "all_negatives_unknown": all(r["conservative_unknown"] for r in report["negative_controls"].values()),
                      "seconds": report["seconds"]}, indent=2))
