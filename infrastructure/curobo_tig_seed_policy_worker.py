#!/usr/bin/env python3
"""Official conditional fitted-seed policy on frozen TIG task payload sessions.

All original planner construction and full warmup use always_run_trajopt=True.
Only afterwards the public policy flag becomes False. Fitted straight seeds
still pass native/dense/TimeOpt validation; invalid seeds continue through
TrajOpt and graph search. All limits, collision geometry, permissions, and
physical acceptance rules are inherited unchanged. No simulator is imported.
"""
from __future__ import annotations

import argparse
import contextlib
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace
from typing import Any

if __package__:
    from .curobo_tig_task_payload_worker import (
        TaskPayloadBackend, TaskSessionManager, canonical_task, dispatch_task, serve_task,
        self_test as task_self_test,
    )
    from .curobo_tig_payload_worker import _rotation, canonical_hash
    from .curobo_tig_worker import _finite_json, _matrix, _vector
else:
    from curobo_tig_task_payload_worker import (
        TaskPayloadBackend, TaskSessionManager, canonical_task, dispatch_task, serve_task,
        self_test as task_self_test,
    )
    from curobo_tig_payload_worker import _rotation, canonical_hash
    from curobo_tig_worker import _finite_json, _matrix, _vector


def native_policy(planner: Any) -> dict[str, Any]:
    """Read actual public planner/solver config, never echoed request values."""
    return {"always_run_trajopt": planner.config.always_run_trajopt,
            "enable_trajopt": planner.config.enable_trajopt,
            "run_trajopt_on_graph_seed": planner.config.run_trajopt_on_graph_seed,
            "fit_graph_seed_to_bspline": planner.config.fit_graph_seed_to_bspline,
            "enable_timeopt": planner.trajopt_solver.config.enable_timeopt,
            "native_fitted_seed_evaluator_available": planner.graph_seed_fitter is not None}


def apply_seed_policy(planner: Any, always_run: bool) -> dict[str, Any]:
    """Change exactly one public boolean and assert all mandatory gates remain."""
    if type(always_run) is not bool:
        raise ValueError("seed policy must be an explicit boolean")
    before = native_policy(planner)
    if any(before[key] is not True for key in before):
        raise RuntimeError("full original TrajOpt/graph/TimeOpt warmup configuration is required")
    planner.config.always_run_trajopt = always_run
    result = native_policy(planner)
    if result != {**before, "always_run_trajopt": always_run}:
        raise RuntimeError("conditional policy changed another native planner requirement")
    return result


class SeedPolicyBackend(TaskPayloadBackend):
    """Same native model/session contract, with official conditional seed return."""
    def __init__(self, task_id: str, *, always_run_trajopt: bool = False) -> None:
        super().__init__(task_id)
        if type(always_run_trajopt) is not bool:
            raise ValueError("explicit boolean planning policy required")
        self.always_run_trajopt = always_run_trajopt

    def configure(self, request: dict[str, Any]) -> dict[str, Any]:
        if request.get("warmup") is not True:
            raise ValueError("the complete original warmup must finish before policy activation")
        result = super().configure(request)
        result["native_seed_policy"] = apply_seed_policy(self.planner, self.always_run_trajopt)
        result["warmup_always_run_trajopt"] = True
        result["seed_policy_api"] = "MotionPlannerCfg.always_run_trajopt (public runtime configuration)"
        result["seed_policy_scope"] = "retime and validate fitted straight seeds first; retain TrajOpt/graph fallback"
        return result

    def plan_pose(self, request: dict[str, Any]) -> dict[str, Any]:
        policy = native_policy(self.planner)
        if (policy["always_run_trajopt"] is not self.always_run_trajopt
                or any(policy[key] is not True for key in policy if key != "always_run_trajopt")):
            raise RuntimeError("a required native planner or validation gate changed")
        result = super().plan_pose(request)
        result["native_seed_policy"] = native_policy(self.planner)
        return result


def base_goal(world_goal: Any, base_pose: Any) -> list[float]:
    """Transform finite XYZ/WXYZ[7] world goal into the frozen robot base."""
    goal, base = _vector(world_goal, 7, "world_goal"), _vector(base_pose, 7, "base_pose")
    for value in (goal, base):
        norm = math.sqrt(sum(x*x for x in value[3:]))
        if abs(norm-1.) > 1e-5:
            raise ValueError("base and goal require unit WXYZ quaternions")
        value[3:] = [x/norm for x in value[3:]]
    rotation = _rotation(base)
    delta = [goal[i]-base[i] for i in range(3)]
    xyz = [sum(rotation[j][i]*delta[j] for j in range(3)) for i in range(3)]
    w,x,y,z = [base[3],-base[4],-base[5],-base[6]]
    a,b,c,d = goal[3:]
    quat = [w*a-x*b-y*c-z*d, w*b+x*a+y*d-z*c,
            w*c-x*d+y*a+z*b, w*d+x*c-y*b+z*a]
    result = [*xyz,*quat]
    _finite_json(result)
    return result


def trajectory_statistics(result: dict[str, Any]) -> dict[str, Any]:
    """Read actual returned seven-joint trajectory and native validation flags."""
    if result.get("success") is not True:
        return {"success": False, "status": result.get("status"), "trajectory_source": result.get("trajectory_source")}
    q = _matrix(result["position"], 7, "planned positions")
    times = _vector(result["times_s"], len(q), "native timestamps")
    if len(q) < 2 or times[0] != 0. or any(b <= a for a,b in zip(times,times[1:])):
        raise ValueError("statistics require a complete increasing native trajectory")
    delta = [q[-1][i]-q[0][i] for i in range(7)]
    steps = [[right[i]-left[i] for i in range(7)] for left,right in zip(q,q[1:])]
    chord = math.sqrt(sum(x*x for x in delta))
    arc = sum(math.sqrt(sum(x*x for x in row)) for row in steps)
    validation = result.get("validation",{})

    def all_true(value: Any) -> bool:
        return all(all_true(x) for x in value) if isinstance(value,list) else value is True

    required = ("path_valid","collision_valid","limits_and_dynamics_valid")
    return {"success":True,"duration_s":times[-1],"scheduled_steps_20Hz":math.ceil(times[-1]/.05),
            "joint_chord_l2_rad":chord,"joint_arc_l2_rad":arc,
            "joint_arc_chord_ratio":arc/chord if chord else 1.,
            "maximum_joint_delta_rad":max(abs(x)for x in delta),"delta_q_rad":delta,
            "per_joint_total_variation_rad":[sum(abs(row[i])for row in steps)for i in range(7)],
            "trajectory_source":result.get("trajectory_source"),"planning_wall_s":result["planning_wall_s"],
            "native_validation":validation,
            "all_native_validation_flags_passed":all(key in validation and all_true(validation[key])for key in required),
            "native_column_selection":result["native_column_selection"],
            "native_seed_policy":result["native_seed_policy"]}


def benchmark(snapshot_dir: Path, output: Path, gpu_uuid: str, arm: int) -> None:
    """Two independent same-seed full-warmup planners, one frozen descend input."""
    if not gpu_uuid.startswith("GPU-") or os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_uuid:
        raise ValueError("benchmark requires one matching explicit CUDA_VISIBLE_DEVICES UUID")
    if arm not in (0,1):
        raise ValueError("one canonical physical arm is required")
    output.mkdir(parents=True,exist_ok=False)
    source = output/"source";source.mkdir()
    paths = [Path(__file__),*[Path(__file__).with_name(name)for name in (
        "curobo_tig_task_payload_worker.py","curobo_tig_fast_payload_worker.py",
        "curobo_tig_payload_worker.py","curobo_tig_session_worker.py","curobo_tig_worker.py")]]
    source_hashes = {}
    for path in paths:
        shutil.copyfile(path,source/path.name)
        source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    initial_path = snapshot_dir/f"initial-arm-{arm}-snapshot.json"
    actual_path = snapshot_dir/f"descend-arm-{arm}-snapshot.json"
    original_report = snapshot_dir/"report.json"
    initial, actual, prior = [json.loads(path.read_text())for path in (initial_path,actual_path,original_report)]
    task = canonical_task(actual["payload_model"]["task_id"])
    if initial["payload_model"] != actual["payload_model"]:
        raise ValueError("initial/descend target and fixed source model changed")
    goal = base_goal(prior["grasp_targets"][arm][1],actual["base_pose_wxyz"])
    side = "left" if arm == 0 else "right"
    session_id = side+":approach"
    frozen_input = {"task_id":task,"initial":initial,"actual":actual,"goal_base_xyz_wxyz":goal,
                    "seed":401,"interpolation_dt":.02,"max_attempts":4}
    report: dict[str,Any] = {"status":"running","gpu_uuid":gpu_uuid,"arm_index":arm,"task_id":task,
        "environment_steps":0,"source_sha256":source_hashes,"input_sha256":canonical_hash(frozen_input),
        "input_file_sha256":{str(path):hashlib.sha256(path.read_bytes()).hexdigest()for path in (initial_path,actual_path,original_report)},
        "cases":[],"policy_benefit_assumed":False,"native_validation_gates_changed":False}
    (output/"frozen-input.json").write_text(json.dumps(frozen_input,allow_nan=False)+"\n")

    def save() -> None:
        (output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")

    start=time.perf_counter()
    try:
        for name,always_run in (("always",True),("conditional",False)):
            manager=TaskSessionManager(task,backend_factory=lambda:SeedPolicyBackend(task,always_run_trajopt=always_run))
            row:dict[str,Any]={"policy":name,"input_sha256":report["input_sha256"],"always_run_trajopt":always_run}
            report["cases"].append(row)
            try:
                initialization=dispatch_task({"id":0,"op":"init_session","session_id":session_id,
                    "robot_cfg":initial["robot_cfg"],"world":initial["world"],"payload_model":initial["payload_model"],
                    "seed":401,"interpolation_dt":.02,"warmup":True},manager)["result"]
                backend=manager.sessions[session_id].backend
                updated=dispatch_task({"id":1,"op":"update_session","session_id":session_id,
                    "robot_cfg":actual["robot_cfg"],"world":actual["world"],"payload_state":actual["payload_state"],
                    "snapshot_id":"frozen-descend"},manager)["result"]
                # This resets the native IK (including LM), TrajOpt and graph
                # generators. Both independent instances used the same seed
                # and the same complete original always-True warmup.
                backend.planner.reset_seed()
                request={"id":2,"op":"plan_pose","session_id":session_id,"snapshot_id":"frozen-descend",
                         "start":actual["q_start"],"goal_pose":goal,"max_attempts":4}
                result=dispatch_task(request,manager)["result"]
                row.update(trajectory_statistics(result))
                row["native_geometry_identity"]={"robot_models":updated["native_update"]["native_readback"]["robot_models"],
                    "world":updated["native_update"]["native_readback"]["world"],
                    "payload_world":updated["native_update"]["payload"]["world"],
                    "source_policy":updated["native_update"]["payload"]["source_policy"]}
                row["storage_identity_unchanged"]=backend._storage_identity()==backend.initial_storage_identity
                for suffix,value in (("initialization",initialization),("update",updated),("native-result",result)):
                    (output/f"{name}-{suffix}.json").write_text(json.dumps(value,allow_nan=False)+"\n")
                if result.get("success") is True:
                    fk=dispatch_task({"id":3,"op":"fk","session_id":session_id,"snapshot_id":"frozen-descend",
                                      "positions":result["position"]},manager)["result"]
                    (output/f"{name}-trajectory-fk.json").write_text(json.dumps(fk,allow_nan=False)+"\n")
                    xyz=_matrix(fk["positions"],3,"TCP trajectory")
                    chord=[xyz[-1][i]-xyz[0][i]for i in range(3)]
                    length=math.sqrt(sum(x*x for x in chord))
                    arc=sum(math.sqrt(sum((b[i]-a[i])**2 for i in range(3)))for a,b in zip(xyz,xyz[1:]))
                    lateral=[]
                    for point in xyz:
                        delta=[point[i]-xyz[0][i]for i in range(3)]
                        fraction=sum(delta[i]*chord[i]for i in range(3))/(length*length)if length else 0.
                        lateral.append(math.sqrt(sum((delta[i]-fraction*chord[i])**2 for i in range(3))))
                    row.update(tcp_chord_m=length,tcp_arc_m=arc,tcp_maximum_lateral_from_line_m=max(lateral),
                        tcp_upward_reversal_m=sum(max(0.,b[2]-a[2])for a,b in zip(xyz,xyz[1:])))
                save()
            finally:
                manager.shutdown({})
        first,second=report["cases"]
        report["native_model_world_permissions_identical"]=(first["native_geometry_identity"]==second["native_geometry_identity"])
        if not report["native_model_world_permissions_identical"]:
            raise AssertionError("A/B geometry, limits, world or collision permissions differed")
        report["both_native_plans_passed"]=all(row.get("success")and row.get("all_native_validation_flags_passed")for row in report["cases"])
        report["conditional_duration_gain_s"]=(first["duration_s"]-second["duration_s"])if report["both_native_plans_passed"]else None
        report["status"]="completed"
    except Exception as error:
        report.update(status="failed",error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        report["wall_s"]=time.perf_counter()-start
        report["simulator_imported"]=any(name=="sapien"or name.startswith("mani_skill")for name in sys.modules)
        save()


def self_test() -> None:
    task_self_test()
    assert "torch"not in sys.modules and "curobo"not in sys.modules
    config=SimpleNamespace(always_run_trajopt=True,enable_trajopt=True,run_trajopt_on_graph_seed=True,fit_graph_seed_to_bspline=True)
    planner=SimpleNamespace(config=config,trajopt_solver=SimpleNamespace(config=SimpleNamespace(enable_timeopt=True)),graph_seed_fitter=object())
    readback=apply_seed_policy(planner,False)
    assert readback["always_run_trajopt"]is False and all(readback[key]is True for key in readback if key!="always_run_trajopt")
    config.always_run_trajopt=True;config.enable_trajopt=False
    try:apply_seed_policy(planner,False)
    except RuntimeError:pass
    else:raise AssertionError("disabled TrajOpt was accepted")
    identity=[0.,0.,0.,1.,0.,0.,0.]
    assert base_goal([1.,2.,3.,1.,0.,0.,0.],[.5,1.,2.,1.,0.,0.,0.])==[.5,1.,1.,1.,0.,0.,0.]
    q=[[0.]*7,[.1]+[0.]*6]
    sample={"success":True,"position":q,"times_s":[0.,1.],"planning_wall_s":.1,"validation":{
        name:[[True]]for name in ("path_valid","collision_valid","limits_and_dynamics_valid")},
        "trajectory_source":"linear_fit","native_column_selection":{"selection_by_name":True},"native_seed_policy":readback}
    stats=trajectory_statistics(sample)
    assert abs(stats["joint_arc_chord_ratio"]-1.)<1e-12 and stats["scheduled_steps_20Hz"]==20 and stats["all_native_validation_flags_passed"]
    assert trajectory_statistics({"success":False,"status":"NO_SOLUTION"})["success"]is False
    sample["validation"]["collision_valid"]=[[False]]
    assert trajectory_statistics(sample)["all_native_validation_flags_passed"]is False
    assert "torch"not in sys.modules and "curobo"not in sys.modules
    print(json.dumps({"seed_policy_self_test":"PASS","trajectory_shape":[2,7],"checks":["one_public_policy_change",
        "mandatory_TrajOpt_TimeOpt_graph_guards","native_validation_failure_preserved","native_trajectory_statistics",
        "base_frame_transform","inherited_task_source_and_payload_protocol","no_GPU_or_simulator_import"]}))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--serve",action="store_true")
    mode.add_argument("--self-test",action="store_true")
    mode.add_argument("--benchmark-snapshots",type=Path)
    parser.add_argument("--task-id")
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--gpu-uuid",default="")
    parser.add_argument("--arm",type=int,choices=(0,1),default=1)
    args=parser.parse_args()
    if args.self_test:self_test()
    elif args.benchmark_snapshots is not None:
        if args.output_dir is None:parser.error("--output-dir required for independent snapshot benchmark")
        benchmark(args.benchmark_snapshots,args.output_dir,args.gpu_uuid,args.arm)
    else:
        if args.task_id is None:parser.error("--task-id must fix one process task")
        protocol=os.fdopen(os.dup(sys.stdout.fileno()),"w",buffering=1)
        sys.stdout.flush();os.dup2(sys.stderr.fileno(),sys.stdout.fileno())
        with protocol,contextlib.redirect_stdout(sys.stderr):
            serve_task(sys.stdin,protocol,TaskSessionManager(args.task_id,backend_factory=lambda:SeedPolicyBackend(args.task_id)))


if __name__=="__main__":main()
