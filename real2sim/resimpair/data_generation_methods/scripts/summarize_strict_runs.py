"""Read-only aggregation of every strict-planner attempt, including failures."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def summarize(root: Path) -> dict:
    rows = []
    for path in sorted(root.glob("*/input.json")):
        case = path.parent
        request = json.loads(path.read_text())
        result_path = case / "result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        proposal = request["proposal"]
        plans = []
        for decision_path in sorted((case / "plans").glob("*/decision.json")):
            decision = json.loads(decision_path.read_text())
            plan_request = json.loads((decision_path.parent / "request.json").read_text())
            guard = decision.get("guard", {})
            geometry = guard.get("geometry", {})
            plans.append({
                "mode": plan_request["mode"],
                "stage": plan_request["stage"],
                "accepted": decision.get("accepted", False),
                "planner_status": decision.get("planner_status"),
                "error": decision.get("error"),
                "joint_failures": guard.get("joint", {}).get("failures", []),
                "geometry_failures": list(geometry.get("failures", {})),
                "endpoint": geometry.get("endpoint"),
                "collision_failures": guard.get("collision_failures", []),
                "dense_samples": guard.get("dense_samples"),
                "all_dense_collision_samples_checked": guard.get("all_dense_collision_samples_checked"),
                "attached_object_enabled": guard.get("attached_object_enabled"),
                "evidence": str(decision_path),
            })
        accepted = [p for p in plans if p["accepted"]]
        rows.append({
            "attempt": case.name, "seed": proposal["seed"],
            "case_id": proposal["case_id"], "active_arm": proposal["active_arm"],
            "human_window_id": proposal.get("human_window_id"),
            "distractors": len(proposal.get("added_distractors", [])),
            "asset_categories": sorted({x["category"] for x in proposal.get("added_distractors", [])}),
            "camera": proposal["camera"],
            "status": "passed_native" if result.get("passed") else ("failed" if result else "running_or_interrupted"),
            "frames": result.get("frames"),
            "failure": result.get("error"), "failure_stage": result.get("controller_phase"),
            "source_sha256": request.get("source_sha256"),
            "accepted_paths": len(accepted), "rejected_paths": len(plans) - len(accepted),
            "max_accepted_endpoint_m": max(
                (p["endpoint"]["position_error_m"] for p in accepted if p["endpoint"]), default=None),
            "accepted_attached_paths": sum(bool(p["attached_object_enabled"]) for p in accepted),
            "all_accepted_paths_collision_checked": all(
                p["all_dense_collision_samples_checked"] is True for p in accepted) if accepted else None,
            "plans": plans, "result": str(result_path) if result_path.exists() else None,
            "native_video": str(case / "episode/video_native512.mp4")
                if (case / "episode/video_native512.mp4").exists() else None,
            "qualification_caveats": result.get("still_pending", []),
        })
    return {
        "schema": "strict_attempt_summary_v1", "root": str(root),
        "all_attempts": len(rows), "unique_seeds": len({x["seed"] for x in rows}),
        "status_counts": dict(Counter(x["status"] for x in rows)),
        "interpretation": "Adaptive development attempts; never a fixed-distribution generalization rate.",
        "formal_continuous_collision_claimed": False, "training_started": False,
        "attempts": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = summarize(args.root)
    rendered = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered)
