"""A bounded new experiment with old usage retained and explicit held-out cases."""
from __future__ import annotations

import shutil
from pathlib import Path

from .core import WORKSPACE, bind, checked, read, require, write
from .library import make_scene, instantiate, task_semantic_key, TASK_TEMPLATES


def prepare(root):
    root = Path(root).resolve()
    require(not (root / "experiment.json").exists(), "Campaign already prepared; do not reset")
    require(shutil.disk_usage(root.parent).free > 10 * 2**30, "Insufficient free disk space")
    old = Path("/mnt/sdg/datasets/real_sim_human_pipeline/five_task_probe_20261022_v1")
    ancestors = []
    prior = {"image": 0, "video": 0, "review": 0}
    for folder in ("human_generation", "human_generation_v3"):
        budget_path = old / folder / "budget.json"
        old_budget = read(budget_path)
        ledger = Path(old_budget["ledger_path"])
        calls = []
        for marker in sorted(ledger.glob("stages/*/call_started.json")):
            metadata = read(marker.parent / "request_metadata.json")
            kind = metadata["experiment_call_kind"]
            prior[kind] += 1
            calls.append({"kind": kind, "started": bind(marker), "metadata": bind(marker.parent / "request_metadata.json")})
        ancestors.append({"budget": bind(budget_path), "calls": calls})
    require(prior == {"image": 18, "video": 7, "review": 0}, "Prior campaign usage changed; re-audit lineage")
    write(root / "ancestor_usage.json", {"ancestors": ancestors, "started_calls": prior,
                                        "historical_unknown_outcomes_remain_unknown": True})
    limits = {"image": 4, "video": 4, "review": 12}
    write(root / "budget.json", {
        "schema": "resimpair_cumulative_budget_v4", "campaign_id": "autonomous-multitask-20261023-v4",
        "ledger_path": str(root / "campaign_ledger"), "new_call_limits": limits,
        "ancestor_started_calls": prior, "ancestor_evidence": [bind(root / "ancestor_usage.json")],
        "lineage_cumulative_limits": {k: prior[k] + limits[k] for k in prior},
        "max_concurrent": 1, "timeout_seconds": 600,
        "authorization": "User explicitly requests autonomous pipeline, new scenes/tasks, randomized simulation and generated human references; bounded first validation campaign.",
        "currency_budget": None, "automatic_retry": False,
    })
    public = read(old / "r2h/public_config.json")
    write(root / "config.json", {"gemini": public["gemini"], "seedance": public["seedance"],
                                  "anchor_model": public["anchor_model"],
                                  "credentials_file": str(WORKSPACE / "corl2026_workshop/private/.env.local")})
    spec_path = Path("/mnt/sdf/datasets/hypericl_human_robot_task_library_20261019_v1/generation/prepared_v1/human_grounded_task_spec.json")
    spec = read(spec_path)
    source = next(w for t in spec["tasks"] if t["id"] == "apple_bowl_transfer"
                  for w in t["human_windows"] if w["id"].endswith("8006"))
    sources = [{"id": "source_01", "video": source["video"], "window_frames": source["window_frames"],
                "session": source["session"], "split": source["split"], "lineage": bind(spec_path)}]
    jenga = old / "human_source_probes/ego_part1_assemble_jenga_1"
    provenance = read(jenga / "provenance.json")
    sources.append({"id": "source_02", "video": bind(jenga / "native.mp4"), "window_frames": [0, 109],
                    "session": provenance["session"], "split": provenance["source_split"],
                    "lineage": bind(jenga / "provenance.json")})
    for s in sources:
        checked(s["video"])
    scenes = [make_scene("scene_A", 2026102304),
              make_scene("scene_A_nuisance", 2026102305, nuisance=True),
              make_scene("scene_B_layout", 2026102306, nuisance=True, layout=True)]
    cases = []
    for scene in scenes:
        write(root / "scenes" / (scene["id"] + ".json"), scene)
        for family, objects, supports in [
            ("pack", ["red_block", "yellow_block"], ["left_tray"]),
            ("stack", ["red_block", "blue_block", "yellow_block"], ["platform"]),
            ("arrange", ["red_block", "blue_block", "yellow_block"], []),
        ]:
            cid = scene["id"] + "__" + family
            task = instantiate(family, cid, scene, objects, supports)
            write(root / "tasks" / (cid + ".json"), task)
            cases.append({"id": cid, "scene": bind(root / "scenes" / (scene["id"] + ".json")),
                          "task": bind(root / "tasks" / (cid + ".json")), "semantic_key": task_semantic_key(task),
                          "partition": "development" if scene["id"] == "scene_A" else "held_out_after_release_freeze",
                          "request_human_reference": scene["id"] == "scene_A" or cid == "scene_B_layout__pack",
                          "reference_policy": "one_or_two_per_task_semantics_not_per_nuisance_variant"})
    manifest = {"schema": "resimpair_autonomy_experiment_v4", "sources": sources, "cases": cases,
                "source_frame_budget": 24, "review_frame_budget": 49, "simulation_limit": 12,
                "planned_physical_executions": 9, "simulation_timeout_seconds": 900,
                "max_video_poll_seconds_per_invocation": 45, "gpu": 2,
                "gpu_uuid": "GPU-807bd44d-36ea-31d0-427c-50e264428c67", "protected_gpus": [1, 3],
                "task_templates": TASK_TEMPLATES,
                "generation_mode": "first_frame_plus_compiled_contract_declared_before_generation",
                "run_failure_policy": "terminal_diagnosis_no_semantic_retry_no_manual_annotations",
                "development_policy": "version_shared_prompts_or_code_after_aggregating_failures",
                "prior_five_task_results_reused_without_regeneration": True,
                "no_training": True, "source_model_is_not_human_ground_truth": True,
                "hypotheses": [
                    "A fixed source schema plus a complete execution export removes post-planner per-task annotations.",
                    "Shared scene/primitive definitions support different tasks and nuisance variants without task-specific compiler branches.",
                    "A fixed opposite elevated camera and one initial actor protocol suffice to attempt human synthesis; model failures remain terminal evidence.",
                ],
                "autonomy_success_criterion": "After planning, unmodified release runs to generated/reviewed or diagnosed terminal result without manual semantic material.",
                "generation_quality_is_a_separate_metric": True}
    write(root / "experiment.json", manifest)
    return {"root": str(root), "cases": len(cases), "sources": len(sources), "new_call_limits": limits,
            "lineage_cumulative_limits": {k: prior[k] + limits[k] for k in prior}}


def freeze(root, release):
    """Execute copies, not a mutable worktree that merely claims a snapshot hash."""
    from .core import identifier
    source = Path(__file__).parent
    dest = Path(root) / "releases" / identifier(release) / "autonomous_pipeline_v4"
    require(not dest.exists(), "Release snapshots are immutable")
    dest.mkdir(parents=True)
    files = []
    for path in sorted(source.glob("*.py")):
        shutil.copyfile(path, dest / path.name)
        files.append(bind(dest / path.name))
    write(dest.parent / "release.json", {"schema": "resimpair_release_v4", "release": release,
                                         "files": files, "experiment": bind(Path(root) / "experiment.json")})
    return {"release": str(dest.parent), "files": len(files)}
