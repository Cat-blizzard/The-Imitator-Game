"""Independent read-only aggregation of immutable strict-v3 scene evidence."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np

from .planning_contract import bind,check_binding,save_new
from .audit_household_coverage import camera_measurement,runtime_materials
from ..r2h_pipeline.common import json_sha256

ROOT=Path("/mnt/sdg/datasets/real_sim_human_pipeline")
STRICT=ROOT/"strict_20261020e"
EXPANSION=ROOT/"expansion_20261020e"
DYNAMIC_KEYS=("native_commands","states","object_states","timestamps_seconds")


def load_attempt(name):
    folder=STRICT/name
    result=json.loads((folder/"result.json").read_text())
    check_binding(result["receipt"])
    receipt_path=Path(result["receipt"]["path"])
    if receipt_path.resolve()!=(folder/"episode/receipt.json").resolve():
        raise ValueError("Mixed receipt")
    receipt=json.loads(receipt_path.read_text())
    for key,sha in receipt["sha256"].items():
        check_binding({"path":str(receipt_path.parent/key),"sha256":sha})
    check_binding({"path":receipt["qa_visibility"],"sha256":receipt["qa_visibility_sha256"]})
    arrays_path=Path(receipt["arrays"])
    if arrays_path.resolve()!=receipt_path.parent/"arrays.npz":raise ValueError("Mixed arrays")
    with np.load(arrays_path,allow_pickle=False) as data:
        arrays={k:data[k] for k in data.files}
    with np.load(receipt["qa_visibility"],allow_pickle=False) as data:
        qa={k:data[k] for k in data.files}
    proposal=json.loads((folder/"input.json").read_text())["proposal"]
    if result["passed"] is not True or result["native_verifier"]["passed"] is not True:
        raise ValueError("This completed strict-v3 audit found a failed native result")
    camera=camera_measurement(receipt["qa_visibility"],proposal["camera"])
    if not camera["matches_proposal"]:raise ValueError("Actual camera differs from requested config")
    materials=runtime_materials(receipt["randomization"])
    if any(x["status"]!="runtime_record_and_file_hash_verified" for x in materials.values()):
        raise ValueError("Unknown runtime material binding")
    return {"folder":folder,"result":result,"receipt":receipt,"arrays":arrays,
            "qa":qa,"proposal":proposal,"camera":camera,"materials":materials}


def dynamic_fingerprint(arrays):
    h=hashlib.sha256()
    for key in DYNAMIC_KEYS:
        array=np.ascontiguousarray(arrays[key])
        if not np.isfinite(array).all():raise ValueError("Nonfinite dynamic arrays")
        h.update((key+"|"+str(array.dtype)+"|"+repr(array.shape)+"\n").encode())
        h.update(array.tobytes())
    return h.hexdigest()


def visual_fingerprint(data):
    """Ignore attempt IDs and audit metadata; bind real scene/mat/camera content."""
    p=data["proposal"]
    clutter=[]
    for d in p["added_distractors"]:
        for path,sha in d["asset_files"].items():check_binding({"path":path,"sha256":sha})
        clutter.append({
            "category":d["category"],"center":d["center"],"yaw_rad":d["yaw_rad"],
            "visual_sha256":d["asset_files"][d["visual"]],
            "collision_sha256":d["asset_files"][d["collision"]],
        })
    signature={
        "seed":p["seed"],"split":p["split"],"active_arm":p["active_arm"],"roles":p["roles"],
        "initial_native_dynamic_states":data["arrays"]["object_states"][0].tolist(),
        "native_asset_uids":[a["uid"] for a in data["receipt"]["asset_specs"]],
        "clutter":clutter,
        "actual_camera_from_world":data["qa"]["camera_from_world"].tolist(),
        "actual_intrinsic_224":data["qa"]["intrinsic_224"].tolist(),
        "actual_material_sha256":{k:v["binding"]["sha256"] for k,v in data["materials"].items()},
    }
    return json_sha256(signature)


def camera_event_recheck(reference_report,attempts):
    recorded=json.loads(reference_report.read_text())
    base_name=recorded["cases"][0]["attempt"];base=attempts[base_name]
    rows=[]
    for previous in recorded["cases"]:
        name=previous["attempt"];data=attempts[name]
        arrays=data["arrays"]
        if set(arrays)!=set(base["arrays"]):raise ValueError("Camera variant array keys differ")
        equality={key:bool(np.array_equal(value,base["arrays"][key])) for key,value in arrays.items()}
        if not all(equality.values()):raise ValueError("Camera-only repair altered native arrays")
        numeric={
            key:float(np.max(np.abs(value.astype(float)-base["arrays"][key].astype(float)),initial=0))
            for key,value in arrays.items() if np.issubdtype(value.dtype,np.number)
        }
        if numeric!=previous["array_max_abs_diff_from_same_seed_baseline"]:
            raise ValueError("Root numeric difference audit cannot be reproduced")
        selected=data["proposal"]["active_arm"]
        counts=data["qa"]["object_pixels_224"][:,selected]
        stages=arrays["stage"]
        visibility={}
        for stage,old in previous["stage_visibility"].items():
            values=counts[stages==stage]
            if len(values)==0:raise ValueError("Event stage missing")
            actual={"frames":len(values),"minimum_pixels":int(values.min()),
                    "visible_ge16_count":int((values>=16).sum()),
                    "visible_ge16_fraction":float(np.mean(values>=16))}
            for key in ("frames","minimum_pixels","visible_ge16_fraction"):
                if actual[key]!=old[key]:raise ValueError("Root visibility audit cannot be reproduced")
            visibility[stage]=actual
        fraction=float(np.mean(counts>=16))
        if fraction!=previous["all_frames_ge16_fraction"]:raise ValueError("Global visibility mismatch")
        rows.append({
            "attempt":name,"receipt":bind(data["folder"]/"episode/receipt.json"),
            "arrays":bind(data["receipt"]["arrays"]),"qa_visibility":bind(data["receipt"]["qa_visibility"]),
            "native_pass":True,"frames":data["result"]["frames"],
            "all_array_fields_exactly_equal":equality,"numeric_max_abs_differences":numeric,
            "numeric_field_count":len(numeric),"all_array_field_count":len(equality),
            "stage_visibility_224":visibility,"all_frames_visible_ge16_fraction":fraction,
            "actual_camera":data["camera"],
            "selected_visible_pixel_count_is_not_contact_ground_truth":True,
        })
    return {"root_audit":bind(reference_report),"recomputed_from_raw_arrays":True,"cases":rows,
            "threshold_scope":"16 visible segmentation pixels at224 is a local diagnostic, not universal quality GT",
            "camera_candidates_were_posthoc":True}


def summarize():
    source=STRICT/"strict_v3_motion_duplicate_audit.json"
    root_dedup=json.loads(source.read_text())
    names=sorted({name for group in root_dedup["groups"].values() for name in group})
    if len(names)!=root_dedup["attempts"]:raise ValueError("Duplicate or missing attempt IDs")
    attempts={name:load_attempt(name) for name in names}
    motion_groups=defaultdict(list);visual_groups=defaultdict(list)
    for name,data in attempts.items():
        motion_groups[dynamic_fingerprint(data["arrays"])].append(name)
        visual_groups[visual_fingerprint(data)].append(name)
    ours={frozenset(v) for v in motion_groups.values()}
    theirs={frozenset(v) for v in root_dedup["groups"].values()}
    if ours!=theirs:raise ValueError("Root exact-motion partition cannot be reproduced")
    native_proposals=EXPANSION/"household_layouts_v3/proposals.json"
    case_ids={p["case_id"] for p in json.loads(native_proposals.read_text())["cases"]}
    paired={name:d for name,d in attempts.items() if d["proposal"]["case_id"] in case_ids}
    if len(paired)!=8:raise ValueError("Expected the exact predeclared 8 repaired configurations")
    categories=sorted({x["category"] for d in paired.values()
                        for x in d["proposal"]["added_distractors"]})
    rows=[]
    for name,data in paired.items():
        row={"attempt":name,"case_id":data["proposal"]["case_id"],
             "seed":data["proposal"]["seed"],"human_window_id":data["proposal"]["human_window_id"],
             "native_pass":True,"frames":data["result"]["frames"],
             "static_distractors":data["result"]["actual_extra_static_distractors"],
             "categories":[x["category"] for x in data["proposal"]["added_distractors"]],
             "max_extra_contact_force_N":data["result"]["max_extra_contact_force_N"],
             "camera_verified":data["camera"]["matches_proposal"],
             "receipt":bind(data["folder"]/"episode/receipt.json")}
        rows.append(row)
    original=EXPANSION/"household_layouts_v2/actual_coverage_snapshot_v1.json"
    old=json.loads(original.read_text())
    event=camera_event_recheck(STRICT/"event_camera_repair_audit.json",attempts)
    result={
        "schema":"independent_household_strictv3_qualification_summary_v1",
        "historical_first8":{
            "raw_snapshot":bind(original),"attempts":old["actual_attempts_found"],
            "native_pass":old["recorded_pass_attempts"],
            "failed_initial_checks":[a["case_id"] for a in old["attempts"] if a["recorded_physical_pass"] is False],
            "not_pooled_with_repaired_run":True,
        },
        "strictv3_paired_household":{
            "proposals":bind(native_proposals),"attempts":len(paired),"native_pass":len(paired),
            "distinct_rendered_configurations":len({visual_fingerprint(d) for d in paired.values()}),
            "exact_dynamic_rollouts":len({dynamic_fingerprint(d["arrays"]) for d in paired.values()}),
            "layout_seeds":sorted({d["proposal"]["seed"] for d in paired.values()}),
            "distinct_household_asset_categories":categories,
            "human_windows":sorted({d["proposal"]["human_window_id"] for d in paired.values()}),
            "rows":rows,
        },
        "event_camera_repair":event,
        "whole_strictv3_inventory":{
            "root_audit":bind(source),"independent_partition_matches_root":True,
            "successful_execution_attempts":len(attempts),
            "distinct_rendered_configurations":len(visual_groups),
            "exact_dynamic_rollouts":len(motion_groups),
            "dynamic_fingerprint_fields":list(DYNAMIC_KEYS),
            "motion_groups":dict(motion_groups),"visual_config_groups":dict(visual_groups),
            "duplicates_are_not_new_motion_diversity":True,
        },
        "cpu_only_household_asset_categories":12,
        "actual_new_household_asset_category_count":len(categories),
        "training_admitted":False,
        "interpretation":[
            "12 categories passed CPU checks, only 8 appeared in the current qualified household scene runs.",
            "Native outcome checks and exact dynamic consistency do not replace RGB task/contact/asset-readability review.",
            "Repeated g01/clutter2 is a regression rerun, not an extra scene or an independent motion.",
            "Clutter and view augmentations are valid distinct observations but can share exactly the same physical motion.",
            "All variants of a motion and human session must stay in one dataset split; no frame-wise random split.",
            "These adaptive engineering qualification counts are not an unbiased rollout success-rate estimate.",
        ],
    }
    target=EXPANSION/"household_actual_qualification_strictv3_v1.json"
    save_new(target,result)
    print(json.dumps({
        "paired":result["strictv3_paired_household"]["native_pass"],
        "household_categories":len(categories),
        "all_attempts":len(attempts),"visual_configs":len(visual_groups),
        "exact_dynamic_rollouts":len(motion_groups),
        "output":str(target)},ensure_ascii=False))
    return result


if __name__=="__main__":summarize()
