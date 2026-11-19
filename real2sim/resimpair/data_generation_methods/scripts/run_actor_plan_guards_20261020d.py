"""Join existing blinded VLM reviews and local hand detections, without new API."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_generation_methods.r2h_pipeline.actor_plan_guard import run_guard
from data_generation_methods.r2h_pipeline.common import file_sha256, read_json, write_json

ROOT = Path("/mnt/sdg/datasets/real_sim_human_pipeline")


def binding(path):
    return {"path": str(path), "sha256": file_sha256(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"automation_20261020d/quality/actor_guard_v1")
    args = parser.parse_args()
    reviews = ROOT/"automation_20261020d/quality/blind_reviews"
    old_source = ROOT/"batch_20261020b/jobs/right_cabinet_open_only_891914000/edit_anchor/request_metadata.json"
    new_source = ROOT/"state_conditioned_20261020c/cabinet_case/source_manifest.json"
    settings = [
        ("candidate_a", 2, old_source, "old_cabinet_contact",
         "Retrospective numeric formalization of the original prompt's right hand and resting left hand; not falsely claimed to be an original numeric parameter."),
        ("candidate_b", read_json(new_source)["rendering_plan"]["maximum_visible_hands"],
         new_source, "new_cabinet_contact_v1",
         "Explicit maximum_visible_hands in the generation-time source_manifest rendering_plan."),
    ]
    records = []
    for candidate, budget, source, diagnostic_dir, scope in settings:
        review = reviews/candidate/"quality_review.json"
        local = ROOT/"state_conditioned_20261020c/tools"/diagnostic_dir
        output = args.output/candidate
        plan = {
            "schema": "r2h_actor_plan_v1",
            "media_sha256": read_json(review)["media"]["sha256"],
            "maximum_visible_hands": budget,
            "maximum_visible_people": None,
            "active_actor_plan": "One active hand operates the right handle.",
            "people_budget_note": "Original evidence does not give a validated visible-person count; do not infer people from hands.",
            "declaration_scope": scope, "sources": [binding(source)],
        }
        plan_path = output/"actor_plan.json"
        write_json(plan_path, plan)
        result = run_guard(plan_path, review, local/"diagnostic.json", output,
                           point_evidence_path=local/"point_evidence.json")
        row = {
            "candidate": candidate, "maximum_visible_hands": budget,
            "status": result["decision"]["status"],
            "vlm_excess_observations": len(result["decision"]["vlm_budget_excess_observations"]),
            "detector_excess_frames": len(result["decision"]["detector_budget_excess_frames"]),
            "count_disagreements": len(result["decision"]["count_disagreements"]),
            "dense_frame_requests": len(result["decision"]["review_requests"]),
            "exported_native_frames": len(result["packets"]),
            "report": binding(output/"actor_plan_guard.json"),
        }
        records.append(row)
        print(row, flush=True)
    write_json(args.output/"SUMMARY.json", {
        "schema": "r2h_actor_guard_examples_v1", "records": records,
        "new_paid_calls": 0, "new_detector_inference": False,
        "new_review_performed": False, "training_admitted": False})


if __name__ == "__main__":
    main()
