"""Register three bounded physical trials; old capability probes stay unchanged."""
from copy import deepcopy
from pathlib import Path
import shutil

from .core import activate, bind, checked, read, require, write
from .active_assets import descriptor, object_spec, prepare_cooked
from .benchmark import ASSETS
from .budget import counts
from .intake import register_case, sources
from .library import make_scene, instantiate


def prepare(root, ancestor):
    activate()
    root, ancestor = Path(root).resolve(), Path(ancestor).resolve()
    require(not (root / "experiment.json").exists(), "active_asset_campaign_already_registered")
    require(shutil.disk_usage(root.parent).free > 100 * 2**30, "insufficient_asset_trial_disk")
    old = read(ancestor / "budget.json")
    prior = counts(old["ledger_path"])
    lineage = {k: old["ancestor_started_calls"][k] + prior[k] for k in prior}
    source_rows = sources(ancestor)
    for source in source_rows:
        original = ancestor / "sources" / source["id"] / "source_contract.json"
        checked(read(original)["input"]["video"])
        folder = root / "sources" / source["id"]
        folder.mkdir(parents=True)
        (folder / "source_contract.json").symlink_to(original.resolve())
    write(root / "ancestor_usage.json", {"budget": bind(ancestor / "budget.json"), "started": prior,
          "lineage_started": lineage, "old_physical_attempts_not_reopened": True})
    write(root / "budget.json", {"schema": "resimpair_cumulative_budget_v4",
        "campaign_id": "active-mesh-local-20261023-v1", "ledger_path": str(root / "campaign_ledger"),
        "new_call_limits": {k: 0 for k in lineage}, "ancestor_started_calls": lineage,
        "ancestor_evidence": [bind(root / "ancestor_usage.json")], "lineage_cumulative_limits": lineage,
        "max_concurrent": 1, "timeout_seconds": 600, "currency_budget": 0, "automatic_retry": False,
        "authorization": "User requested continued pipeline optimization; local bounded active-asset qualification, no paid allocation."})
    write(root / "config.json", read(ancestor / "config.json"))
    write(root / "experiment.json", {"schema": "resimpair_active_mesh_experiment_v1", "sources": source_rows,
        "cases": [], "simulation_limit": 3, "planned_physical_executions": 3, "simulation_timeout_seconds": 180,
        "source_frame_budget": 24, "review_frame_budget": 49, "gpu": 2,
        "gpu_uuid": "GPU-807bd44d-36ea-31d0-427c-50e264428c67", "protected_gpus": [0, 1, 3],
        "no_training": True, "automatic_retry": False, "new_api_call_limits": {k: 0 for k in lineage},
        "validation_plan": "box baseline, mesh development, then new layout under the SAME frozen code; no failures replaced",
        "scope": "rigid milk carton transfer only; neither package opening nor real mass/scale qualification",
        "validation_rules_frozen_before_execution": True, "source_split": "train",
        "source_independent_holdout_claimed": False, "ancestor": bind(ancestor / "experiment.json")})
    cooked_binding = prepare_cooked(ASSETS, "milk_carton", mass_kg=.15, output=root / "assets/milk_carton_cooked")
    probes = []
    for category in ["milk_carton", "soda_can", "stapler"]:
        try:
            value = descriptor(ASSETS, category, mass_kg=.15)
            probes.append({"category": category, "status": "eligible_for_bounded_trial", "qualified": False})
        except ValueError as exc:
            probes.append({"category": category, "status": "needs_different_contact_adapter", "reason": str(exc),
                           "qualified": False})
    write(root / "asset_intake.json", {"probes": probes, "catalog": bind(ASSETS), "original_catalog_unchanged": True})
    cohort = []
    for case_id, seed, mode, xy, yaw in [
        ("active_box_baseline", 2026102351, "box", [-.24, -.34], 0.),
        ("active_milk_development", 2026102351, "mesh", [-.24, -.34], 0.),
        ("active_milk_layout_validation", 2026102352, "mesh", [-.22, -.30], .18),
    ]:
        scene = make_scene(case_id + "_scene", seed, nuisance=True, layout=False)
        if mode == "mesh":
            scene["objects"][0] = object_spec(cooked_binding, "milk_carton", xy, yaw)
        oid = scene["objects"][0]["id"]
        task = instantiate("relocate", case_id, scene, [oid], ["left_tray"])
        scene_path, task_path = root / "scenes" / (case_id + ".json"), root / "tasks" / (case_id + ".json")
        write(scene_path, scene)
        write(task_path, task)
        row = register_case(root, case_id, scene_path, task_path, make_reference=True)
        cohort.append({"case_id": case_id, "mode": mode, "scene": row["scene"], "task": row["task"],
                       "development_or_validation": "validation_after_freeze" if "validation" in case_id else "development",
                       "source_split": "train", "max_attempts": 1})
    write(root / "cohort.json", {"cases": cohort, "replacement": False, "automatic_retry": False,
          "two_mesh_layouts_do_not_equal_two_new_asset_types": True})
    return {"root": str(root), "cohort": cohort, "asset_probes": probes, "new_api_limit": 0, "physical_limit": 3}
