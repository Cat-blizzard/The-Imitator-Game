"""CPU layout repair using an actual same-seed baseline's untouched-object mesh.

No dynamics are guessed from stage labels or pre-settling nominal poses.
Old proposals and simulator attempts are immutable. Only required XY changes
are made, with task role positions, camera, yaw, asset geometry and seed fixed.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from .planning_contract import bind,check_binding,save_new
from .structured_randomization import screen_layout
from ..r2h_pipeline.common import json_sha256


def world_vertices(spec):
    for path,sha in spec["asset_files"].items():
        check_binding({"path":path,"sha256":sha})
    mesh=trimesh.load(spec["collision"],force="mesh",process=False)
    rotation=Rotation.from_euler("z",spec["yaw_rad"]).as_matrix()
    return (mesh.vertices-mesh.bounds.mean(0))@rotation.T+np.asarray(spec["center"])


def bbox_margin_conflict(a,b,margin=.025):
    overlap=np.minimum(a[1],b[1])-np.maximum(a[0],b[0])
    return bool((overlap>-margin).all())


def build_obstacle(baseline,expected):
    baseline=Path(baseline)
    original=json.loads((baseline/"input.json").read_text())["proposal"]
    result=json.loads((baseline/"result.json").read_text())
    if result.get("passed") is not True:
        raise ValueError("Repair baseline must have a completed independent native verification")
    for key in ("seed","roles","active_arm","split"):
        if json_sha256(original[key])!=json_sha256(expected[key]):
            raise ValueError("Baseline native state differs: "+key)
    if original["added_distractors"]:
        raise ValueError("Use a baseline without added clutter for unperturbed initial geometry")
    receipt_path=baseline/"episode/receipt.json"
    check_binding(result["receipt"])
    if Path(result["receipt"]["path"]).resolve()!=receipt_path.resolve():
        raise ValueError("Baseline result receipt points to another episode")
    receipt=json.loads(receipt_path.read_text())
    mesh_path=baseline/"plans/unselected_object_world.obj"
    mesh=trimesh.load(mesh_path,force="mesh",process=False)
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("Nonfinite actual unselected object mesh")
    return {
        "id":"native_unselected_dynamic_object",
        "evidence_kind":"actual_runtime_world_collision_mesh_after_initial_settling",
        "baseline_input":bind(baseline/"input.json"),"baseline_result":bind(baseline/"result.json"),
        "baseline_receipt":bind(receipt_path),"world_collision_mesh":bind(mesh_path),
        "bounds_world_m":mesh.bounds.tolist(),
        "baseline_seed":original["seed"],"active_arm":original["active_arm"],
        "unselected_index":1-original["active_arm"],
        "initial_measured_pose":receipt["initial_measured_object_states"][1-original["active_arm"]][:7],
        "reuse_scope":"same seed/task role geometry and initial native asset ordering; new clutter must be collision-free",
        "not_guaranteed_identical_until_new_runtime_initial_check":True,
    }


def repair_group(case,obstacle,spacing=.005):
    bounds=np.asarray(obstacle["bounds_world_m"])
    output=deepcopy(case)
    local={}
    for spec in output["added_distractors"]:
        local[spec["instance_id"]]=world_vertices(spec)-np.asarray(spec["center"])
    moves=[]
    for index,spec in enumerate(output["added_distractors"]):
        shape=local[spec["instance_id"]]
        before=np.asarray(spec["center"],float)
        world=shape+before
        if not bbox_margin_conflict(np.stack([world.min(0),world.max(0)]),bounds):
            continue
        radius=spec["radius_m"]
        xs=np.unique(np.r_[np.arange(-.30+radius,.25-radius,spacing),before[0]])
        ys=np.unique(np.r_[np.arange(-.46+radius,.46-radius,spacing),before[1]])
        xx,yy=np.meshgrid(xs,ys)
        candidates=np.column_stack([xx.ravel(),yy.ravel(),np.full(xx.size,before[2])])
        distances=np.linalg.norm(candidates-before,axis=1)
        found=None
        for candidate in candidates[np.argsort(distances,kind="stable")]:
            vertices=shape+candidate
            if bbox_margin_conflict(np.stack([vertices.min(0),vertices.max(0)]),bounds):
                continue
            trial=deepcopy(output)
            trial["added_distractors"][index]["center"]=candidate.tolist()
            if screen_layout(trial)["coarse_screen_passed"]:
                found=candidate;break
        if found is None:
            raise RuntimeError("No valid translation-only repair; do not relax geometry gates")
        spec["center"]=found.tolist()
        moves.append({
            "instance_id":spec["instance_id"],"category":spec["category"],
            "from_center_m":before.tolist(),"to_center_m":found.tolist(),
            "delta_xyz_m":(found-before).tolist(),
            "translation_norm_m":float(np.linalg.norm(found-before)),
            "reason":"native_unselected_dynamic_object_AABB_clearance",
            "search_grid_spacing_m":spacing,
            "minimal_among_tested_fixed_yaw_XY_grid":True,
        })
    for spec in output["added_distractors"]:
        vertices=local[spec["instance_id"]]+np.asarray(spec["center"])
        if bbox_margin_conflict(np.stack([vertices.min(0),vertices.max(0)]),bounds):
            raise RuntimeError("Unresolved native initial collision")
    if not screen_layout(output)["coarse_screen_passed"]:
        raise RuntimeError("Repair invalidated existing coarse constraints")
    return output["added_distractors"],moves


def repair(proposals,baselines,output):
    proposals=Path(proposals);output=Path(output)
    source=json.loads(proposals.read_text())
    groups={}
    for case in source["cases"]:
        groups.setdefault(case["controlled_group_id"],[]).append(case)
    cases=[];reports=[]
    for group,group_cases in groups.items():
        richest=max(group_cases,key=lambda c:len(c["added_distractors"]))
        baseline=baselines[str(richest["seed"])]
        obstacle=build_obstacle(baseline,richest)
        repaired,moves=repair_group(richest,obstacle)
        for original in group_cases:
            case=deepcopy(original)
            case["case_id"]=original["case_id"]+"_nativeclear_v3"
            case["added_distractors"]=deepcopy(repaired[:len(original["added_distractors"])])
            relevant=[m for m in moves if m["instance_id"] in {x["instance_id"] for x in case["added_distractors"]}]
            case["native_scene_obstacles"]=[obstacle]
            case["execution_gate"]["blockers"]=[
                b for b in case["execution_gate"]["blockers"]
                if b!="native_unselected_dynamic_geometry_not_in_CPU_screen"]
            case["native_initial_clearance_screen"]={
                "passed_for_bound_baseline_mesh":True,"requested_axis_separation_m":.025,
                "strict_adapter_axis_separation_m":.02,
                "runtime_initial_collision_check_must_still_run":True,
            }
            case["repair"]={"parent_case_id":original["case_id"],"parent_proposals":bind(proposals),
                            "moves":relevant,"requires_new_attempt_for_changed_geometry":bool(relevant),
                            "unchanged_baseline_or_camera_can_reuse_original_evidence":not relevant,
                            "task_roles_seed_camera_yaw_assets_unchanged":True,
                            "scope":"CPU proposal repair; not simulator approval"}
            cases.append(case)
        reports.append({"group":group,"native_obstacle":obstacle,"moves":moves})
    output.mkdir(parents=True,exist_ok=False)
    result={**{k:v for k,v in source.items() if k!="cases"},
            "schema":"native_geometry_repaired_household_proposals_v3","parent":bind(proposals),
            "cases":cases,"repair_report":reports,
            "changed_geometry_case_ids":[c["case_id"] for c in cases if c["repair"]["requires_new_attempt_for_changed_geometry"]],
            "simulation_executed":False,"training_admitted":False}
    save_new(output/"proposals.json",result)
    for case in cases:save_new(output/"cases"/(case["case_id"]+".json"),case)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--proposals",type=Path,required=True)
    p.add_argument("--baseline",action="append",required=True,help="SEED=/absolute/completed/baseline/folder")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    result=repair(args.proposals,dict(x.split("=",1) for x in args.baseline),args.output)
    print(json.dumps({"changed_geometry":result["changed_geometry_case_ids"],
                      "reports":result["repair_report"],"new_simulator_runs":0},ensure_ascii=False))


if __name__=="__main__":main()
