"""Replay existing, source-bound reviews through CPU quality screening. No API."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_generation_methods.r2h_pipeline.common import file_sha256, read_json, write_json
from data_generation_methods.r2h_pipeline.quality_cascade import inspect_video

BASE = Path("/mnt/sdg/datasets/real_sim_human_pipeline")
DEFAULT_OUTPUT = BASE/"automation_20261020d/quality"


def binding(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": file_sha256(path)}


def check(dimension, result, description, sources, scope):
    return {"dimension": dimension, "result": result, "description": description,
            "sources": [binding(p) for p in sources], "scope": scope}


def bind_media(record, key, video):
    if record[key]["sha256"] != file_sha256(video):
        raise ValueError("Existing evidence is not bound to this media")


def examples(output):
    batch = BASE/"batch_20261020b"
    independent_path = batch/"independent_visual_inspection.json"
    independent = read_json(independent_path)
    candidates = []
    for job in sorted((batch/"jobs").iterdir()):
        if not job.is_dir():
            continue
        video = job/"generated_human.mp4"
        checks = []
        records = [x for x in independent["records"] if x["job_id"] == job.name]
        if records:
            record = records[0]
            bind_media(record, "generated_video", video)
            for failure in record.get("visual_failures", []):
                checks.append(check("embodiment", "fail", failure["observation"],
                                    [independent_path], failure["scope"]))
            checks.append(check(
                "task_semantics", "pass_in_scope",
                record["task_only_readability"]["reason"], [independent_path],
                "Source-aware inspection of listed sampled frames; no all-frame absence guarantee"))
            for unknown in record.get("unresolved_visual_constraints", []):
                checks.append(check("task_semantics", "unknown", unknown["observation"],
                                    [independent_path], unknown["scope"]))
        review_dir = (job/"task_review_protocol_shape_adapter_v1"
                      if job.name == "apple_bowl_transfer_891911000" else job/"task_review")
        if job.name == "closed_book_to_open_box_891912000":
            refined = sorted(job.glob("*refin*"))
            refined = [p for p in refined if (p/"gate_decision.json").exists()]
            if refined:
                review_dir = refined[-1]
        gate_path = review_dir/"gate_decision.json"
        evidence_path = review_dir/"normalized_evidence.json"
        if gate_path.exists():
            gate, evidence = read_json(gate_path), read_json(evidence_path)
            if evidence["subject"]["media"]["sha256"] != file_sha256(video):
                raise ValueError("Task gate media binding mismatch")
            for issue in gate.get("issues", []):
                checks.append(check("task_semantics", "fail" if issue.get("reject") else "unknown",
                                    issue["code"] + ": " + issue.get("detail", ""),
                                    [gate_path, evidence_path], "Existing task gate; unchanged decision"))
        if job.name == "apple_bowl_transfer_891911000":
            repair_path = BASE/"repair_study_20261020_v1/repair_quality_observation.json"
            repair = read_json(repair_path)
            bind_media(repair, "media", video)
            checks.append(check("task_semantics", "pass_in_scope",
                                repair["decision"], [repair_path], "Previously inspected sampled frames"))
        candidates.append((job.name, video, "synthetic_human", "fixed", checks))
    # Known bad clip remains rejected even if its pixels are sharp.
    old_path = BASE/"pilot_20261020/quality_rejection.json"
    old = read_json(old_path)
    old_video = Path(old["media"]["path"])
    bind_media(old, "media", old_video)
    candidates.append(("known_bad_apple_teleport", old_video, "synthetic_human", "fixed", [
        check("task_semantics", "fail", "; ".join(old["violations"]),
              [old_path], old["reviewer_type"])]))
    current = BASE/"state_conditioned_20261020c/cabinet_case"
    review_path = current/"video_review_v1/visible_evidence_summary.json"
    review = read_json(review_path)
    new_video = current/"state_conditioned_human_v1.mp4"
    bind_media(review, "video", new_video)
    checks = []
    for name, observation in review["observations"].items():
        dimension = ("embodiment" if name == "one_hand_visible" else
                     "interaction" if name in {"hand_handle_interaction_visible",
                                               "door_motion_after_hand_arrival",
                                               "hand_released_and_withdrawn"} else "task_semantics")
        checks.append(check(
            dimension, {"yes": "pass_in_scope", "no": "fail", "unknown": "unknown"}[observation["value"]],
            name + ": " + observation["description"], [review_path],
            "VLM visual observation on cited frames only; not 3D or force/contact ground truth"))
    checks.append(check("task_semantics", "unknown", "Full-clip forbidden-event absence is not verified.",
                        [review_path], "Unobserved intervals remain unknown"))
    candidates.append(("state_conditioned_cabinet", new_video, "synthetic_human", "fixed", checks))
    source_path = batch/"jobs/apple_bowl_transfer_891910000/input.json"
    source = read_json(source_path)
    human = source["source_window"]
    human_path = Path(human["video"]["path"])
    bind_media(human, "video", human_path)
    candidates.append(("real_ego_apple", human_path, "real_human", "egocentric", [
        check("task_semantics", "pass_in_scope", human["initial"] + " → " + human["final"],
              [source_path, human["review"]["path"]], human["visual_review_scope"])]))
    robot = source["robot"]
    robot_path = Path(robot["video"]["path"])
    bind_media(robot, "video", robot_path)
    raw_path = Path(robot["prior_native_verification"]["path"])
    raw = read_json(raw_path)
    if file_sha256(raw_path) != robot["prior_native_verification"]["sha256"]:
        raise ValueError("Native physics verifier changed")
    result = "pass_in_scope" if raw["raw_verifier"]["passed"] else "fail"
    candidates.append(("sim_robot_apple", robot_path, "robot_sim", "fixed", [
        check("sim_physics", result, json.dumps(raw["raw_verifier"]["physical_checks"]),
              [source_path, raw_path, robot["receipt"]["path"]], raw["raw_verifier"]["scope"]),
        check("task_semantics", result, "Native bowl task verifier; human-equivalence review remains separate.",
              [source_path, raw_path], "Native simulator execution only")]))
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name, video, origin, camera, checks in examples(args.output):
        directory = args.output/name
        ledger_path = directory/"evidence_ledger.json"
        write_json(ledger_path, {
            "schema": "r2h_quality_evidence_v1", "media_sha256": file_sha256(video),
            "adapter": "run_quality_cascade_examples_20261020d.py",
            "new_review_performed": False, "checks": checks,
        })
        report_path = directory/"quality.json"
        if report_path.exists():
            report = read_json(report_path)
            if report["media"]["sha256"] != file_sha256(video):
                raise ValueError("Existing quality output has wrong media")
            implementation = Path(inspect_video.__code__.co_filename)
            if report.get("implementation", {}).get("sha256") != file_sha256(implementation):
                raise ValueError("Quality implementation changed; use a new output directory")
        else:
            report = inspect_video(video, directory, origin=origin, camera_mode=camera,
                                   evidence_paths=[ledger_path])
        summary = {"id": name, "media": report["media"], "report": binding(report_path),
                   "frame_count": report["media_integrity"]["frame_count"],
                   "media_integrity": report["media_integrity"]["status"],
                   "decision": report["decision"],
                   "quality_vector": report["perceptual"]["quality_vector"],
                   "runtime_seconds": report["runtime_seconds"]}
        summaries.append(summary)
        print(name, summary["frame_count"], summary["decision"]["status"],
              round(report["runtime_seconds"], 2), flush=True)
    summary = {
        "schema": "r2h_quality_examples_v1", "count": len(summaries),
        "quality_release_approved": False, "paid_api_calls": 0,
        "learned_quality_models_run": [], "records": summaries,
        "known_failure_retention_is_not_new_detector_recall": True,
    }
    write_json(args.output/"SUMMARY.json", summary)


if __name__ == "__main__":
    main()
