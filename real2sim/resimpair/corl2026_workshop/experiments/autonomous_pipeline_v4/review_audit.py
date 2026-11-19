"""A versioned review of retained candidates; it never changes their outcomes."""
from __future__ import annotations

from pathlib import Path

from .core import bind, checked, implementation, read, require, write
from .budget import Budget
from .pipeline import release_id
from . import review


def run(root, *, execute=False):
    root = Path(root)
    release = release_id()
    declaration = read(root / "review_audits" / release / "declaration.json")
    require(declaration["release"] == release and declaration["automatic_retry"] is False, "Invalid fixed audit declaration")
    summary_path = root / "review_audits" / release / "summary.json"
    if summary_path.exists():
        saved = read(summary_path)
        require(saved["declaration"] == bind(root / "review_audits" / release / "declaration.json"), "Audit declaration changed")
        return {"results": [{"case": r["case"], "status": r["status"]} for r in saved["results"]],
                "usage": saved["usage"], "restored_terminal_audit": True,
                "not_attempted_cases": saved.get("not_attempted_cases", [])}
    budget = Budget(root / "budget.json")
    missing = [c for c in declaration["cases"] if not (root / "review_audits" / release / c["id"] / "outcome.json").exists()]
    if execute:
        require(len(missing) <= declaration["additional_paid_reviews"], "This audit declaration does not permit more review calls")
        require(len(missing) <= budget.usage()["new_remaining"]["review"], "Registered review budget is unavailable")
    results = []
    for case in declaration["cases"]:
        checked(case["outcome"])
        checked(case["contract"])
        video = checked(case["video"])
        directory = root / "review_audits" / release / case["id"]
        output = directory / "outcome.json"
        if output.exists():
            saved = read(output)
            results.append(saved)
            if saved.get("stop_shared_batch"):
                break
            continue
        directory.mkdir(parents=True, exist_ok=True)
        candidate = directory / "human.mp4"
        if not candidate.exists():
            candidate.symlink_to(video)
        require(bind(candidate) == case["video"], "Review-only candidate changed")
        if not execute:
            prepared = review.video(root, budget, directory, read(case["contract"]["path"]), execute=False)
            results.append({"case": case["id"], **prepared})
            continue
        systemic = False
        try:
            result = review.video(root, budget, directory, read(case["contract"]["path"]))
            row = {"case": case["id"], "status": "sampled_visual_pass" if result["gate"]["passed_in_sampled_visual_scope"] else "sampled_visual_rejected",
                   "gate": result["gate"], "review": bind(directory / "review.json")}
        except Exception as exc:
            http_status = getattr(exc, "http_status", None)
            systemic = http_status in {400, 401, 403, 413, 429}
            row = {"case": case["id"], "status": "review_stopped", "exception_type": type(exc).__name__,
                   "code": getattr(exc, "code", "review_schema_or_runtime_failed"), "http_status": http_status,
                   "stop_shared_batch": systemic, "automatic_retry": False}
        row.update(original_outcome=case["outcome"], original_outcome_changed=False, video=case["video"],
                   new_video_generated=False, human_observations_sent_to_reviewer=False, implementation=implementation())
        write(output, row)
        results.append(row)
        if systemic:
            break
    if execute:
        write(root / "review_audits" / release / "summary.json", {
            "declaration": bind(root / "review_audits" / release / "declaration.json"), "results": results,
            "not_attempted_cases": [c["id"] for c in declaration["cases"] if c["id"] not in {r["case"] for r in results}],
            "usage": budget.usage(), "new_videos": 0, "scope": "development review audit; no original run is reopened or repaired"})
    return {"results": [{"case": r["case"], "status": r["status"], "failed_checks": r.get("gate", {}).get("failed_checks", []),
                         "unknown_checks": r.get("gate", {}).get("unknown_checks", []),
                         "prepared_request": r.get("request")} for r in results], "usage": budget.usage()}
