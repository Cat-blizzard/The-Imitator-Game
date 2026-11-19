"""Ten attempts first, then a recorded cross-task review, then bounded revision.

No case is repaired inside the discovery loop. API budget stays zero; new
mechanism names do not silently inherit unrelated real-video source coverage.
"""
import fcntl
import os
from pathlib import Path
import shutil
import time

import numpy as np

from .core import activate, bind, checked, digest, read, require, replace_state, write


def register_phase(root,phase):
    from .novel_catalog import KINDS,scene,task
    from .novel_protocols import protocol
    from .library import task_semantic_key
    root=Path(root)
    require(phase in {"discovery","validation"},"unknown_novel_phase")
    if phase=="validation":
        review=read(root/"discovery_review.json")
        require(review["all_ten_attempted_before_any_revision"] is True,"complete_discovery_review_required")
        for item in review["discovery_results"]:checked(item)
        require(len(review["discovery_results"])==10,"ten_discovery_results_required")
    require(not (root/"cohorts"/(phase+".json")).exists(),"phase_already_frozen")
    groups=[]
    for kind in KINDS:
        spec=scene(kind);scene_path=root/"scenes"/phase/(kind+".json");write(scene_path,spec)
        members=[]
        # All three task branches are declared in discovery, only the first is run.
        for branch in range(3):
            intent=task(spec,branch,phase);protocol(spec,intent)
            path=root/"tasks"/(intent["id"]+".json");write(path,intent)
            row={"id":intent["id"],"scene":bind(scene_path),"task":bind(path),"semantic_key":task_semantic_key(intent),
                 "partition":"new_mechanism_"+phase,"request_human_reference":False,
                 "reference_policy":"exact_scene_execution_only; source contract missing",
                 "planner_plugin":"novel_mechanisms_v1"}
            write(root/"registered_cases"/(row["id"]+".json"),row)
            members.append({"case":row,"execute_in_phase":phase=="validation" or branch==0,"max_attempts":1,"attempt":"attempt_01"})
        groups.append({"scene_id":kind,"scene":bind(scene_path),"mechanism":spec["mechanism"],
                       "novelty":spec["novelty_vs_previous"],"tasks":members,"declared_task_count":3})
    cohort={"phase":phase,"groups":groups,"replace_failed_cases":False,
            "attempt_count":10 if phase=="discovery" else 30,"source_correspondence":"unknown"}
    write(root/"cohorts"/(phase+".json"),cohort)
    return {"phase":phase,"cohort":bind(root/"cohorts"/(phase+".json")),"planned_attempts":cohort["attempt_count"]}


def prepare(root,ancestor):
    from .budget import counts,Budget
    from .intake import sources
    activate()
    root,ancestor=Path(root).resolve(),Path(ancestor).resolve()
    require(not root.exists(),"new_novel_campaign_root_required")
    require(shutil.disk_usage(root.parent).free>100*2**30,"insufficient_novel_trial_space")
    old=read(ancestor/"budget.json");used=counts(old["ledger_path"])
    lineage={k:old["ancestor_started_calls"][k]+used[k] for k in used}
    source_rows=sources(ancestor);root.mkdir()
    for source in source_rows:
        src=(ancestor/"sources"/source["id"]/"source_contract.json").resolve()
        folder=root/"sources"/source["id"];folder.mkdir(parents=True)
        (folder/"source_contract.json").symlink_to(src)
    write(root/"ancestor_usage.json",{"ancestor_budget":bind(ancestor/"budget.json"),"lineage_started":lineage,"existing_experiments_not_reopened":True})
    write(root/"budget.json",{"schema":"resimpair_cumulative_budget_v4","campaign_id":"novel-mechanisms-local-20261024-v1",
        "ledger_path":str(root/"campaign_ledger"),"new_call_limits":{k:0 for k in lineage},"ancestor_started_calls":lineage,
        "ancestor_evidence":[bind(root/"ancestor_usage.json")],"lineage_cumulative_limits":lineage,
        "max_concurrent":1,"timeout_seconds":600,"currency_budget":0,"automatic_retry":False,
        "authorization":"User requested ten substantially different scenes, three task branches each; attempt all ten before review and shared repairs. No new paid allocation."})
    write(root/"config.json",read(ancestor/"config.json"))
    write(root/"experiment.json",{"schema":"resimpair_ten_novel_mechanisms_v1","sources":source_rows,"cases":[],
        "simulation_limit":50,"simulation_timeout_seconds":180,"gpu":2,"gpu_uuid":"GPU-807bd44d-36ea-31d0-427c-50e264428c67",
        "protected_gpus":[0,1,3],"no_training":True,"automatic_retry":False,"source_frame_budget":24,"review_frame_budget":49,
        "phase_plan":{"discovery":10,"validation_after_review":30,"explicit_new_repair_cases_after_review_max":10},
        "phase_gates":"No solver/asset revision until all ten discovery cases terminate. No validation before a bound discovery review.",
        "scope":"native rigid and passive joint mechanism probes; unverified real-source alignment; no soft-body/fluid claim",
        "ancestor":bind(ancestor/"experiment.json"),"new_api_limits":{k:0 for k in lineage}})
    Budget(root/"budget.json")
    return register_phase(root,"discovery")


def qualify(root,case,attempt,release):
    from .interface import validate_bundle
    from .event_protocol import compile_events
    from .compiler import verify_native_kinematics
    from .generic_compiler import compile_bundle
    from .media import packet
    root=Path(root);directory=root/"executions"/case["id"]/attempt
    output=root/"qualification"/release/case["id"]
    if (output/"result.json").exists():return read(output/"result.json")
    terminal=read(directory/"terminal.json")
    result={"case":case,"attempt":attempt,"terminal":bind(directory/"terminal.json"),"native_qualified":False,
            "source_contract_ready":False,"human_generated":False,"formal_pair":False,"training_admitted":False,
            "worker_status":terminal["status"],"worker_seconds":terminal["elapsed_seconds"],"errors":[],"videos":{}}
    bundle_path=directory/"execution_bundle.json"
    if bundle_path.exists():
        result["execution"]=bind(bundle_path);bundle=read(bundle_path)
        require(bundle["case"]==case and bundle["scene"]==read(checked(case["scene"])) and bundle["task_intent"]==read(checked(case["task"])),"frozen_input_binding_changed")
        with np.load(checked(bundle["measurements"]),allow_pickle=False) as archive:arrays={k:archive[k] for k in archive.files}
        result["frames"]=len(arrays["timestamps_seconds"])
        result["initial_physics_sha256"]=bundle["initial_physics_sha256"]
        result["initial_object_state_sha256"]=bundle["initial_object_state_sha256"]
        if bundle["planning_failure"]:result["errors"].append({"stage":"controller","detail":bundle["planning_failure"]})
        try:
            validate_bundle(bundle,arrays)
            measured=compile_events(read(checked(bundle["event_protocol"])),arrays)
            result["native_predicates"]=measured
            result["kinematics"]=verify_native_kinematics(arrays)
            if not measured["qualification"]["passed"]:
                result["errors"].append({"stage":"native_predicates","failed_checks":[x for x in measured["qualification"]["checks"] if not x["passed"]]})
        except (ValueError,FileNotFoundError) as exc:result["errors"].append({"stage":"native_predicates","detail":str(exc)})
        for name,camera in bundle["cameras"].items():
            result["videos"][name]=camera
            try:
                info=packet(checked(camera["video"]),output/(name+"_frames"),frame_budget=9)
                require(info["decoded_frames"]==len(arrays["timestamps_seconds"]) and
                        np.allclose(info["all_native_pts_seconds"],arrays["timestamps_seconds"],atol=1e-5,rtol=0),"native_media_timeline_mismatch")
            except (ValueError,FileNotFoundError) as exc:result["errors"].append({"stage":"media:"+name,"detail":str(exc)})
        if not result["errors"]:
            mapping={"mode":"unpaired_synthetic_mechanism_probe","sources":[],"source_coverage":"missing",
                     "exact_source_episode_equivalence":False,"manual_mapping_supplied":False,
                     "allowed_for_human_generation":False}
            try:
                contract=compile_bundle(bundle_path,mapping)
                write(output/"physics_contract.json",contract)
                result["native_qualified"]=True;result["physics_contract"]=bind(output/"physics_contract.json")
            except (ValueError,FileNotFoundError) as exc:result["errors"].append({"stage":"compiler","detail":str(exc)})
    else:result["errors"].append({"stage":"setup","detail":"worker produced no execution bundle; inspect original worker log"})
    result["required_new_information"]=["verified real human window for this mechanism", "source skill and event evidence before human pairing"]
    write(output/"result.json",result)
    return result


def run_phase(root,phase):
    from .pipeline import release_id
    from .simulation import launch
    from .budget import Budget
    root=Path(root);release=release_id();cohort=read(root/"cohorts"/(phase+".json"))
    require(all(v==0 for v in Budget(root/"budget.json").spec["new_call_limits"].values()),"novel_local_phase_requires_zero_api")
    if phase!="discovery":
        review=read(root/"discovery_review.json");require(review["all_ten_attempted_before_any_revision"],"ten_case_review_missing")
    members=[m for g in cohort["groups"] for m in g["tasks"] if m["execute_in_phase"]]
    with (root/".novel_phase.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        completed=sorted((root/"phase_results").glob(phase+"_*.json"))
        if completed:
            require(len(completed)==1,"ambiguous_completed_phase_results")
            previous=read(completed[0]);checked(previous["cohort"])
            require(previous["cohort"]==bind(root/"cohorts"/(phase+".json")),"completed_phase_inputs_changed")
            for result in previous["results"]:checked(result)
            return {"status":"existing_completed_phase_preserved","phase":phase,"requested_release":release,
                    "executed_release":previous["release"],"new_physical_attempts":0,
                    "qualified":previous["qualified"],"results":bind(completed[0])}
        from .core import implementation
        current=implementation()
        for m in members:
            reservation=root/"executions"/m["case"]["id"]/m["attempt"]/"reservation.json"
            if reservation.exists():
                require(read(reservation)["implementation"]==current,"incomplete_phase_must_resume_original_frozen_release")
        # Complete the entire physical cohort before inspecting detailed outcomes.
        for index,m in enumerate(members):
            cid=m["case"]["id"]
            replace_state(root/"phase_progress.json",{"phase":phase,"release":release,"pid":os.getpid(),"status":"executing",
                          "current_case":cid,"finished":index,"total":len(members),"updated_unix":time.time()})
            value=launch(root,cid,m["attempt"])
            print({"phase":phase,"attempted":index+1,"total":len(members),"case":cid,"worker":value["status"]},flush=True)
        results=[]
        for m in members:
            r=qualify(root,m["case"],m["attempt"],release)
            results.append(bind(root/"qualification"/release/m["case"]["id"]/"result.json"))
        final={"phase":phase,"release":release,"cohort":bind(root/"cohorts"/(phase+".json")),"attempts":len(members),
               "qualified":sum(read(x["path"])["native_qualified"] for x in results),"results":results,
               "new_human_videos":0,"api_usage":Budget(root/"budget.json").usage()}
        write(root/"phase_results"/(phase+"_"+release+".json"),final)
        replace_state(root/"phase_progress.json",{"phase":phase,"release":release,"status":"complete","finished":len(members),
                      "total":len(members),"active_worker":None,"updated_unix":time.time()})
    return {"phase":phase,"attempts":len(members),"qualified":final["qualified"],"results":bind(root/"phase_results"/(phase+"_"+release+".json"))}


def register_repairs(root,request_path):
    """Explicit, evidence-bound new cases inside the original cumulative ceiling."""
    from .novel_catalog import scene,task
    from .novel_protocols import protocol
    from .library import task_semantic_key
    root=Path(root);request=read(request_path)
    summary=read(checked(request["completed_validation"]))
    require(summary["phase"]=="validation" and summary["attempts"]==30,"complete_validation_required")
    require(not (root/"cohorts/repair.json").exists(),"repair_cohort_already_frozen")
    rows=request["cases"]
    require(0<len(rows)<=10,"at_most_ten_registered_repairs")
    require(len({r["previous_result"]["sha256"] for r in rows})==len(rows),"duplicate_repair_case")
    remaining=read(root/"experiment.json")["simulation_limit"]-len(list((root/"executions").glob("*/*/reservation.json")))
    require(len(rows)<=remaining,"cumulative_physical_ceiling_exceeded")
    allowed={r["sha256"] for r in summary["results"]};groups={}
    for row in rows:
        require(row["previous_result"]["sha256"] in allowed,"repair_not_bound_to_completed_validation")
        previous=read(checked(row["previous_result"]))
        require(not previous["native_qualified"],"repair_registration_requires_a_retained_failure")
        require(bool(row["diagnosis"]) and bool(row["expected_change"]),"repair_needs_diagnosis_and_expected_change")
        old_task=read(checked(previous["case"]["task"]))
        kind=old_task["scene_kind"];spec=scene(kind);intent=task(spec,old_task["branch"],"repair")
        protocol(spec,intent)
        sp=root/"scenes/repair"/(kind+".json");tp=root/"tasks"/(intent["id"]+".json")
        write(sp,spec);write(tp,intent)
        case={"id":intent["id"],"scene":bind(sp),"task":bind(tp),"semantic_key":task_semantic_key(intent),
              "partition":"repair_after_validation_not_independent_holdout","request_human_reference":False,
              "reference_policy":"exact_scene_execution_only; source contract missing","planner_plugin":"novel_mechanisms_v1",
              "repair_of":row["previous_result"],"repair_request":bind(request_path)}
        write(root/"registered_cases"/(case["id"]+".json"),case)
        group=groups.setdefault(kind,{"scene_id":kind,"scene":bind(sp),"mechanism":spec["mechanism"],
            "novelty":spec["novelty_vs_previous"],"tasks":[],"declared_task_count":3})
        group["tasks"].append({"case":case,"execute_in_phase":True,"max_attempts":1,"attempt":"attempt_01"})
    cohort={"phase":"repair","groups":list(groups.values()),"replace_failed_cases":False,
            "attempt_count":len(rows),"source_correspondence":"unknown","repair_request":bind(request_path)}
    write(root/"cohorts/repair.json",cohort)
    return {"phase":"repair","cohort":bind(root/"cohorts/repair.json"),"planned_attempts":len(rows)}
