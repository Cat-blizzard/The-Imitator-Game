"""Read-only coverage audit from actual simulator evidence, never config counts."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .planning_contract import bind, check_binding, save_new
from ..r2h_pipeline.common import json_sha256


def expected_camera(camera,resolution):
    eye=np.asarray(camera["eye"],float)
    forward=np.asarray(camera["target"],float)-eye
    forward/=np.linalg.norm(forward)
    right=np.cross(forward,[0.,0.,1.])
    if np.linalg.norm(right)<1e-8:
        raise ValueError("Vertical look-at is undefined for the current camera convention")
    right/=np.linalg.norm(right)
    down=np.cross(forward,right)
    rotation=np.stack([right,down,forward])
    extrinsic=np.column_stack([rotation,-rotation@eye])
    focal=resolution/(2*np.tan(float(camera["fov_radians"])/2))
    intrinsic=np.array([[focal,0,resolution/2],[0,focal,resolution/2],[0,0,1]])
    return extrinsic,intrinsic


def camera_measurement(path,camera):
    with np.load(path,allow_pickle=False) as array:
        actual_e=np.asarray(array["camera_from_world"],float)
        actual_k=np.asarray(array["intrinsic_224"],float)
        native_k=np.asarray(array["intrinsic_512"],float) if "intrinsic_512" in array else None
    if actual_e.shape!=(3,4) or actual_k.shape!=(3,3):
        raise ValueError("Missing or malformed raw calibration")
    if not np.isfinite(actual_e).all() or not np.isfinite(actual_k).all():
        raise ValueError("Nonfinite camera calibration")
    expected_e,expected_k=expected_camera(camera,224)
    e_error=float(np.max(np.abs(actual_e-expected_e)))
    k_error=float(np.max(np.abs(actual_k-expected_k)))
    native_error=None
    if native_k is not None:
        _,k512=expected_camera(camera,512)
        if native_k.shape!=(3,3) or not np.isfinite(native_k).all():
            raise ValueError("Malformed native calibration")
        native_error=float(np.max(np.abs(native_k-k512)))
    r,t=actual_e[:,:3],actual_e[:,3]
    camera_eye=np.linalg.solve(r,-t)
    return {
        "source":bind(path),
        "measured_eye_world_m":camera_eye.tolist(),
        "measured_forward_world":r[2].tolist(),
        "measured_vertical_fov_radians":float(2*np.arctan(224/(2*actual_k[1,1]))),
        "camera_from_world":actual_e.tolist(),"intrinsic_224":actual_k.tolist(),
        "intrinsic_512":native_k.tolist() if native_k is not None else None,
        "max_absolute_extrinsic_error":e_error,
        "max_absolute_intrinsic224_error":k_error,
        "max_absolute_intrinsic512_error":native_error,
        "matches_proposal":bool(e_error<2e-5 and k_error<2e-4
                                and (native_error is None or native_error<5e-4)),
        "scope":"actual raw camera calibration; does not prove task event visibility",
    }


def runtime_materials(randomization):
    materials={}
    for key in ("wall_material","diffuse_surface","hdri"):
        row=randomization.get(key)
        if row is None:
            materials[key]={"status":"unknown_missing_runtime_record"}
            continue
        if "path" not in row or "sha256" not in row:
            materials[key]={"status":"unknown_missing_asset_binding","record":row}
            continue
        check_binding(row)
        materials[key]={"status":"runtime_record_and_file_hash_verified",
                        "binding":{"path":row["path"],"sha256":row["sha256"]},
                        "asset":row.get("asset"),"source":row.get("source",row.get("upstream")),
                        "license":row.get("license",row.get("license_status"))}
    return materials


def audit_attempt(folder,expected):
    """Missing evidence stays unknown. A failed episode may still calibrate a camera."""
    folder=Path(folder)
    original=json.loads((folder/"input.json").read_text())
    proposal=original["proposal"]
    result_path=folder/"result.json"
    result=json.loads(result_path.read_text()) if result_path.exists() else {}
    row={
        "attempt_directory":str(folder),"input":bind(folder/"input.json"),
        "result":bind(result_path) if result_path.exists() else None,
        "case_id":proposal["case_id"],
        "proposal_exact_match":json_sha256(proposal)==json_sha256(expected),
        "recorded_physical_pass":result.get("passed"),
        "production_approved":False,"training_admitted":False,"errors":[],
        "actual_distractor_count":result.get("actual_extra_static_distractors"),
        "requested_distractor_count":len(expected.get("added_distractors",[])),
        "camera":None,"materials":None,
    }
    receipt_path=folder/"episode/receipt.json"
    failure_camera=folder/"failure/complete_contact_trace.npz"
    randomization=None
    try:
        if receipt_path.exists():
            if result.get("receipt"):
                check_binding(result["receipt"])
                if Path(result["receipt"]["path"]).resolve()!=receipt_path.resolve():
                    raise ValueError("Result receipt points to another episode")
            receipt=json.loads(receipt_path.read_text())
            row["receipt"]=bind(receipt_path)
            randomization=receipt.get("randomization",{})
            camera_path=Path(receipt["qa_visibility"])
            check_binding({"path":str(camera_path),"sha256":receipt["qa_visibility_sha256"]})
            if camera_path.resolve().parent!=receipt_path.parent.resolve():
                raise ValueError("Calibration belongs to another episode")
            row["camera"]=camera_measurement(camera_path,expected["camera"])
            row["camera"]["episode_completed"]=bool(result.get("passed"))
        elif failure_camera.exists():
            row["camera"]=camera_measurement(failure_camera,expected["camera"])
            row["camera"]["episode_completed"]=False
            scene=folder/"failure/scene_evidence.json"
            if scene.exists():
                randomization=json.loads(scene.read_text()).get("randomization",{})
        if randomization is not None:
            row["materials"]=runtime_materials(randomization)
            row["runtime_native_layout_seed"]=randomization.get("layout_seed")
            row["runtime_proposal_exact_match"]=(
                json_sha256(randomization.get("structured_proposal"))==json_sha256(expected))
        row["distractor_count_matches_runtime"]=(
            row["actual_distractor_count"]==row["requested_distractor_count"]
            if row["actual_distractor_count"] is not None else None)
    except Exception as exc:
        row["errors"].append(str(exc))
    row["coverage_evidence_complete"]=bool(
        row["proposal_exact_match"] and row["recorded_physical_pass"] is True
        and row["camera"] is not None and row["camera"]["matches_proposal"]
        and row["distractor_count_matches_runtime"] is True
        and row.get("runtime_proposal_exact_match") is True
        and row["materials"] is not None
        and all(m["status"]=="runtime_record_and_file_hash_verified"
                for m in row["materials"].values())
        and not row["errors"])
    return row


def audit(proposals_path,result_roots):
    proposals=json.loads(Path(proposals_path).read_text())
    cases=proposals["cases"]
    by_id={c["case_id"]:c for c in cases}
    if len(by_id)!=len(cases):
        raise ValueError("Duplicate candidate case IDs")
    attempts=[]
    ignored=[]
    for root in map(Path,result_roots):
        if not root.exists():
            ignored.append({"root":str(root),"reason":"not_created"})
            continue
        for input_path in sorted(root.rglob("input.json")):
            try:
                case_id=json.loads(input_path.read_text())["proposal"]["case_id"]
            except (ValueError,KeyError):
                ignored.append({"input":str(input_path),"reason":"not_a_simulator_attempt"})
                continue
            if case_id not in by_id:
                ignored.append({"input":str(input_path),"reason":"case_not_in_requested_set"})
                continue
            attempts.append(audit_attempt(input_path.parent,by_id[case_id]))
    complete=[a for a in attempts if a["coverage_evidence_complete"]]
    seen={a["case_id"] for a in attempts}
    complete_ids={a["case_id"] for a in complete}
    signatures={tuple(m["binding"]["sha256"] for m in a["materials"].values()) for a in complete}
    return {
        "schema":"actual_household_coverage_audit_v1",
        "proposals":bind(proposals_path),"requested_cases":len(cases),
        "actual_attempts_found":len(attempts),
        "recorded_pass_attempts":sum(a["recorded_physical_pass"] is True for a in attempts),
        "evidence_complete_cases":sorted(complete_ids),
        "not_attempted_case_ids":sorted(set(by_id)-seen),
        "cases_still_without_complete_coverage":sorted(set(by_id)-complete_ids),
        "verified_clutter_counts":dict(Counter(
            str(by_id[i]["factor_design"]["clutter_count"]) for i in complete_ids)),
        "verified_household_categories":sorted({
            d["category"] for i in complete_ids for d in by_id[i]["added_distractors"]}),
        "distinct_runtime_material_combinations":len(signatures),
        "attempts":attempts,"ignored_inputs":ignored,
        "limitations":[
            "Counts include adaptive attempts; they are not an unbiased success rate.",
            "Raw K/E confirms camera implementation, not visibility of grasp/contact.",
            "Material evidence is runtime-bound file metadata, not semantic pixel segmentation.",
            "Material_seed is not treated as implemented appearance randomization.",
            "Coverage completion is not production or training approval.",
        ],
        "new_simulator_executions":0,"training_admitted":False,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals",type=Path,required=True)
    parser.add_argument("--results",type=Path,action="append",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.proposals,args.results)
    save_new(args.output,result)
    print(json.dumps({"attempts":result["actual_attempts_found"],
                      "complete_cases":len(result["evidence_complete_cases"]),
                      "not_attempted":len(result["not_attempted_case_ids"]),
                      "output":str(args.output)}))


if __name__=="__main__":
    main()
