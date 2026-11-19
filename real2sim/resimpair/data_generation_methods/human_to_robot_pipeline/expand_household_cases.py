"""Finite, factor-separated household scene candidates; never runs a simulator."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from .planning_contract import validate_contract, check_binding
from .structured_randomization import screen_layout
from .household_assets import bind
from ..r2h_pipeline.common import write_json, json_sha256

WORK=Path("/mnt/sdf/dataset/zhouxunzhe/zhiqi")
CONTRACT=WORK/"data_generation_methods/human_to_robot_pipeline/evidence_20261020d/apple_contract.json"
FRUIT=Path("/mnt/sdf/datasets/hypericl_data_factory_20261016_v1/fruit_adapters_v2/catalog.json")
ENVIRONMENT=Path("/mnt/sdg/datasets/open_assets/household_scaleup_20261020_v1/environments/polyhaven/asset_catalog.json")
VARIANTS=[
    ("baseline",0,"base","native"),
    ("clutter2",2,"base","native"),
    ("clutter4",4,"base","native"),
    ("camera_only",0,"perturbed","native"),
    ("appearance_only",0,"base","explicit_candidate"),
    ("combined",4,"perturbed","explicit_candidate"),
]
PRIORITY_CATEGORIES=[
    "soda_can","water_bottle","tissue_box","soap_dispenser",
    "spatula","remote_control","closed_jam_jar","milk_carton",
    "kettle","mug","screwdriver","stapler",
]


def rng_for(seed,factor):
    return np.random.default_rng(np.random.SeedSequence([int(seed),int(factor)]))


def base_case(seed,active_arm):
    rng=rng_for(seed,10)
    midpoint=np.array([rng.uniform(-.08,.045),rng.uniform(-.065,.065),0.])
    theta=float(rng.uniform(-.40,.40))
    separation=float(rng.uniform(.25,.35))
    direction=np.array([np.sin(theta),np.cos(theta),0.])*(1 if active_arm else -1)
    src,dst=midpoint+direction*separation/2,midpoint-direction*separation/2
    return {
        "seed":seed,"split":"train","active_arm":active_arm,
        "layout_mode":"explicit_proposed_geometry",
        "roles":{"selected_object":{"category":"apple","initial_relation":"inside_source_container"},
                 "source_container":{"center":src.tolist(),"initially_occupied":True},
                 "destination_container":{"center":dst.tolist(),"initially_empty":True}},
        "added_distractors":[],
        "camera":{"eye":[-.50,0.,.73],"target":[0.,0.,.10],"fov_radians":1.30,
                  "fixed_during_episode":True},
        "camera_mode":"explicit_proposed_camera",
        "critical_event_probe_points":{"initial_grasp":(src+[0,0,.08]).tolist(),
                                      "lift_clear":(src+[0,0,.22]).tolist(),
                                      "arrival":(dst+[0,0,.22]).tolist(),
                                      "released_support":(dst+[0,0,.08]).tolist()},
        "simulation_executed":False,"training_admitted":False,
    }


def clutter_set(base,assets,group_index,max_attempts=3000):
    """Four distinct families with bounded rejection, never shrinking after failure."""
    rng=rng_for(base["seed"],20)
    rejected=Counter()
    # Cover each reviewed family before repeating. Readable household objects
    # enter the first qualification groups; the ambiguous stapler enters last.
    ordered=sorted(assets,key=lambda a:PRIORITY_CATEGORIES.index(a["category"]))
    preferred=np.roll(np.arange(len(ordered)),-group_index*4)
    selected=[ordered[int(i)] for i in preferred[:4]]
    rng.shuffle(selected)
    for attempt in range(max_attempts):
        obstacles=[]
        # Four corner sectors retain a clear central pick-transfer region.
        sectors=[(-1,-1),(1,-1),(-1,1),(1,1)]
        rng.shuffle(sectors)
        for i,(asset,(sx,sy)) in enumerate(zip(selected,sectors)):
            extent=np.asarray(asset["extents_m"])
            radius=float(np.linalg.norm(extent)/2)
            x=float(rng.uniform(-.27+radius,-.10) if sx<0 else rng.uniform(.09,.24-radius))
            y=float(sy*rng.uniform(.30,.45-radius))
            obstacles.append({
                "instance_id":f"household_{i}","asset_uid":asset["uid"],
                "category":asset["category"],"center":[x,y,float(extent[2]/2)],
                "radius_m":radius,"yaw_rad":float(rng.uniform(-np.pi,np.pi)),
                "visual":asset["visual"],"collision":asset["collision"],
                "asset_files":asset["files"],"must_remain_untouched":True,
                "asset_spec":bind(Path(asset["visual"]).parent/"asset.json"),
                "body_type":"static","collision_policy":asset["collision_policy"],
                "source_license":asset["license"],
                "no_runtime_scaling":True,
            })
        case=deepcopy(base);case["added_distractors"]=obstacles
        screen=screen_layout(case)
        if screen["coarse_screen_passed"]:
            return obstacles,{"attempts":attempt+1,"rejection_reasons":dict(rejected)}
        rejected.update(screen["reasons"])
    return None,{"attempts":max_attempts,"rejection_reasons":dict(rejected)}


def appearance_candidates(catalog=ENVIRONMENT):
    rows=json.loads(Path(catalog).read_text())["records"]
    wanted=["polyhaven:angli_veneer","polyhaven:bamboo_wall_03"]
    selected=[]
    for ident in wanted:
        row=next(r for r in rows if r["asset_id"]==ident)
        file_hashes={f["path"]:f["sha256"] for f in row["files"]}
        channels=[]
        for channel in row["channels"]:
            path=channel["path"]
            check_binding({"path":path,"sha256":file_hashes[path]})
            channels.append({**channel,"sha256":file_hashes[path]})
        selected.append({"asset_id":ident,"license":row["license"],"source":row["source"],
                         "channels":channels,"intended_use":"table_or_background_surface_candidate",
                         "runtime_override_implemented":False,"training_admitted":False})
    return selected


def variants_for(base,clutter,group_index,human,appearance):
    result=[]
    camera_rng=rng_for(base["seed"],30)
    camera=deepcopy(base["camera"])
    camera["eye"]=(np.array(camera["eye"])+camera_rng.uniform([-.025,-.035,-.015],[.025,.035,.035])).tolist()
    camera["fov_radians"]=float(camera_rng.uniform(1.27,1.34))
    for name,n,camera_level,appearance_level in VARIANTS:
        c=deepcopy(base)
        c["case_id"]=f"household_e_g{group_index:02d}_{name}_{base['seed']}"
        c["controlled_group_id"]=f"household_e_g{group_index:02d}_{base['seed']}"
        c["level"]="household_factor_qualification"
        c["added_distractors"]=deepcopy(clutter[:n])
        if camera_level=="perturbed":c["camera"]=deepcopy(camera)
        c["appearance"]={
            "mode":"native_layout_seed_default" if appearance_level=="native" else "explicit_asset_candidate",
            "wall_and_hdri_split":"train","sample_independently_of_goal":True,
            "material_seed":int(rng_for(base["seed"],40).integers(1,2**31)),
            "material_seed_consumed_by_current_adapter":False,
            "explicit_surface_asset":appearance[group_index%len(appearance)] if appearance_level!="native" else None,
        }
        c["factor_design"]={
            "layout_seed":base["seed"],"clutter_count":n,
            "camera_condition":camera_level,"appearance_condition":appearance_level,
            "independent_RNG_streams":{"layout":10,"clutter":20,"camera":30,"appearance":40},
            "within_group_same_native_seed_same_role_positions_same_human_reference":True,
            "not_independent_demonstrations_until_simulated":True,
        }
        c["human_window_id"]=human["id"]
        c["human_video"]=human["video"]
        c["human_source_window"]=deepcopy(human)
        c["planner_requirements"]={
            "obstacle_instance_ids":["source_container","destination_container"]+[x["instance_id"] for x in c["added_distractors"]],
            "include_attached_object_collision":True,
            "strict_full_q_path_validation_before_execution":True,
            "collision_mask_changes_after_failure_forbidden":True,
            "phase_labels_do_not_verify_contact":True,
        }
        blockers=["strict_adapter_small_qualification_must_pass",
                  "new_asset_scene_physics_and_visual_qualification_pending",
                  "native_unselected_dynamic_geometry_not_in_CPU_screen"]
        if appearance_level!="native":
            blockers.append("explicit_appearance_override_not_implemented")
        c["execution_gate"]={
            "status":"candidate_not_executed","automatic_launch":False,"blockers":blockers,
            "camera_override_interface_present_but_range_not_qualified":True,
        }
        c["screen"]=screen_layout(c)
        result.append(c)
    return result


def generate(asset_catalog,output,*,groups=4,seed_base=2026302000,contract_path=CONTRACT):
    if not 2<=groups<=4:raise ValueError("This finite qualification prepares 12..24 candidates")
    contract=json.loads(Path(contract_path).read_text());validate_contract(contract)
    windows=[x for x in contract["human_windows"] if x["split"]=="train"]
    if not windows:raise ValueError("No reviewed same-split human window")
    for row in windows:check_binding(row["video"])
    assets=json.loads(Path(asset_catalog).read_text())["assets"]
    for a in assets:
        if a["category"] in {"apple","bowl","container"}:raise ValueError("Ambiguous distractor")
        for path,sha in a["files"].items():check_binding({"path":path,"sha256":sha})
    fruit=json.loads(FRUIT.read_text())["assets"]
    backgrounds=appearance_candidates()
    selected=[];rejects=[];seed=seed_base;group=0
    while group<groups and seed<seed_base+500:
        trial_seed=seed;seed+=1
        picked=np.random.default_rng(trial_seed+730001).choice(len(fruit),2,replace=False)
        arms=[i for i,index in enumerate(picked) if fruit[int(index)]["name"]=="apple"]
        if len(arms)!=1 or arms[0]!=group%2:continue
        base=base_case(trial_seed,arms[0])
        if not screen_layout(base)["coarse_screen_passed"]:
            rejects.append({"seed":trial_seed,"stage":"layout","screen":screen_layout(base)})
            continue
        clutter,search=clutter_set(base,assets,group)
        if clutter is None:
            rejects.append({"seed":trial_seed,"stage":"four_distractor_geometry","search":search});continue
        cases=variants_for(base,clutter,group,windows[group%len(windows)],backgrounds)
        if not all(x["screen"]["coarse_screen_passed"] for x in cases):
            rejects.append({"seed":trial_seed,"stage":"factor_variant_camera","search":search,
                            "reasons":[x["screen"]["reasons"] for x in cases]});continue
        for c in cases:c["layout_search"]=search
        selected.extend(cases);group+=1
    if group!=groups:raise RuntimeError("Bounded qualification sampler exhausted; do not relax gates")
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    for c in selected:write_json(output/"cases"/(c["case_id"]+".json"),c)
    report={
        "schema":"household_factor_qualification_candidates_v1",
        "contract":bind(contract_path),"asset_catalog":bind(asset_catalog),
        "native_fruit_catalog":bind(FRUIT),"appearance_candidates":backgrounds,
        "controlled_layout_groups":groups,"candidate_configurations":len(selected),
        "cases":selected,"rejected_coarse_layouts":rejects,
        "counts_by_clutter":dict(Counter(str(len(c["added_distractors"])) for c in selected)),
        "used_household_categories":sorted({d["category"] for c in selected for d in c["added_distractors"]}),
        "asset_family_priority":PRIORITY_CATEGORIES,
        "cpu_preview_review_notes":{
            "source":"agent review of v3 CPU asset gallery; not in-simulator qualification",
            "stapler":"low semantic readability, resembles translucent flat box; excluded from first two groups",
            "mug":"stylized wooden mug; not photorealistic",
        },
        "original_four_pilot_seeds_modified":False,
        "simulation_executed":False,"new_robot_trajectories":0,"new_API_calls":0,"new_downloads":0,
        "training_admitted":False,
        "qualification_order":["First compare unchanged original four seeds with the new strict planner.",
                               "Then one new baseline plus its 2/4-clutter variants.",
                               "Only after native checks inspect camera perturbations.",
                               "Appearance-only/combined stay blocked until actual overrides are implemented."],
        "limitations":["Bounding volumes and event rays are coarse candidates, not collision/visibility proof.",
                       "Native unselected object, both robots and carried apple still require exact collision checks.",
                       "Same-group factor variants are correlated and must never cross dataset splits.",
                       "Explicit background assets are prepared but not consumed by current adapter."],
    }
    write_json(output/"candidate_manifest.json",report)
    first8=[c for c in selected if c["factor_design"]["appearance_condition"]=="native"][:8]
    write_json(output/"first8_candidate_configs.json",{
        "candidate_manifest":str(output/"candidate_manifest.json"),
        "selected_case_ids":[c["case_id"] for c in first8],
        "cases":first8,
        "status":"only_candidate_order; not launched or physically qualified",
        "first_four_original_pilot_seeds_unchanged":True,
    })
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--assets",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--groups",type=int,default=4)
    p.add_argument("--seed-base",type=int,default=2026302000)
    a=p.parse_args()
    r=generate(a.assets,a.output,groups=a.groups,seed_base=a.seed_base)
    print(json.dumps({"configs":r["candidate_configurations"],"categories":r["used_household_categories"],
                      "coarse_rejected_groups":len(r["rejected_coarse_layouts"]),
                      "new_robot_trajectories":0,"output":str(a.output)}))


if __name__=="__main__":main()
