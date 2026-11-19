"""Read-only replay of retained workbench hand-mapping failures.

Exports diagnostic previews only. Never submits a model call, runs physics,
changes an old contract, or registers the reconstructed protocol as a new run.
"""
from pathlib import Path

import numpy as np

from .core import bind, checked, implementation, read, require, write
from .event_protocol import compile_events
from .generic_compiler import generation_information
from .hand_correspondence import compile_correspondence, review_requirements
from .prompts import anchor_prompt, video_prompt
from .review import anchor_checks, video_checks
from .workbench import protocol as planned_protocol


def replay(root, output):
    root, output = Path(root), Path(output)
    require(root.resolve() != output.resolve() and root.resolve() not in output.resolve().parents,
            "hand_regression_output_must_be_outside_retained_campaign")
    rows = []
    for path in sorted((root / "executions").glob("*/*/execution_bundle.json")):
        bundle = read(path)
        if (bundle.get("schema") != "resimpair_execution_bundle_v5" or bundle.get("planning_failure") is not None
                or bundle.get("scene", {}).get("schema") != "resimpair_mechanism_workbench_v1"):
            continue
        old = read(checked(bundle["event_protocol"]))
        with np.load(checked(bundle["measurements"]), allow_pickle=False) as archive:
            arrays = dict(archive)
        camera = bundle["cameras"]["opposite_high"]
        measured = compile_events(old, arrays)
        require(measured["qualification"]["passed"], "retained_execution_fails_original_physics")
        try:
            compile_correspondence(old, arrays, camera)
            original_mapping = {"accepted": True, "reason": None}
        except ValueError as exc:
            original_mapping = {"accepted": False, "reason": str(exc)}
        # Run the shared future planning builder on the retained scene/task.
        # This is deliberately labelled retrospective, never a new declaration.
        planned = planned_protocol(bundle["scene"], bundle["task_intent"])
        require(compile_events(planned, arrays) == measured, "hand_fix_changed_physical_qualification")
        correspondence = compile_correspondence(planned, arrays, camera)
        generation, initial = generation_information(planned, measured, arrays, camera)
        contract = {"schema": "resimpair_hand_mapping_replay_preview_v1",
                    "generation_information": generation, "initial_scene_information": initial, "measured": measured}
        destination = output / bundle["case"]["id"] / path.parent.name
        preview = {"contract": contract, "hand_correspondence_evidence": correspondence,
                   "anchor_prompt": anchor_prompt(initial), "video_prompt": video_prompt(contract),
                   "anchor_checks": anchor_checks(initial), "video_checks": video_checks(contract),
                   "hand_review_requirements": review_requirements(generation),
                   "retrospective_measurement_replay_only": True, "submitted_to_provider": False,
                   "new_physical_execution": False, "training_admitted": False}
        write(destination / "preview.json", preview)
        source_files = [path, checked(bundle["event_protocol"]), checked(bundle["measurements"]),
                        checked(camera["video"]), checked(bundle["cameras"]["robot_side"]["video"])]
        for run in sorted((root / "runs" / bundle["case"]["id"]).glob("*/human.mp4")):
            source_files += [run, run.parent / "video_request.json", run.parent / "review.json", run.parent / "outcome.json"]
        rows.append({"case": bundle["case"]["id"], "attempt": path.parent.name,
                     "retained_inputs": [bind(p) for p in source_files if p.exists()],
                     "original_mapping_under_new_geometric_rule": original_mapping,
                     "original_declared_hands": {k: v["human_hand"] for k, v in old["effectors"].items()},
                     "candidate_measured_hands": {k: v["human_hand"] for k, v in correspondence["effectors"].items()},
                     "physics_results_identical": True, "native_frames": len(arrays["timestamps_seconds"]),
                     "events": len(measured["events"]), "preview": bind(destination / "preview.json")})
    require(rows, "no_retained_successful_workbench_exports_found")
    result = {"schema": "resimpair_hand_mapping_regression_v1", "implementation": implementation(), "cases": rows,
              "new_physical_executions": 0, "new_provider_calls": 0, "videos_automatically_rejudged": 0,
              "manual_visual_labels_consumed_by_compiler": False, "quality_release_approved": False,
              "scope": "original and candidate role bindings use the same new geometric rule; original videos and all historical verdicts remain immutable"}
    write(output / "audit.json", result)
    return {"audit": str(output / "audit.json"), "retained_executions": len(rows),
            "old_mapping_conflicts": sum(not r["original_mapping_under_new_geometric_rule"]["accepted"] for r in rows),
            "new_provider_calls": 0, "new_simulations": 0}
