"""CPU audit of saved strict-v3 plans; never generates or relabels an episode."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
from .planning_contract import WORKSPACE,bind,save_new
from . import strict_motion
from .strict_motion import CheckedMotion,matrix_to_pose7,pose7_to_matrix,world_fk


def robot_dependencies():
    import mplib.pymp
    urdf=WORKSPACE/"PolicyGen/mani_skill/assets/robots/panda/panda_v3.urdf"
    paths={urdf,urdf.with_suffix(".srdf"),Path(mplib.pymp.__file__)}
    for mesh in ET.parse(urdf).getroot().iter("mesh"):
        path=(urdf.parent/mesh.attrib["filename"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.add(path)
    return {
        "captured_at_utc":datetime.now(timezone.utc).isoformat(),
        "capture_timing":"retrospective sidecar; not captured at original execution time",
        "actual_agent_type":"panda_wristcam",
        "actual_robot_model":"panda_v3.urdf",
        "identity_basis":"strict solver uses actual env_agent.urdf_path; PandaWristCam class points to v3; native receipts record dual_panda_wristcam and physical camera_link mesh",
        "legacy_pin_note":"Old source_pins lists panda_v2.urdf as an additional dependency; that entry is not evidence of the robot model actually used at runtime.",
        "historical_files_unchanged":True,
        "dependencies":[bind(p) for p in sorted(paths)],
    }


def audit(episode_root):
    from mplib.planner import Planner
    from mplib.pymp.collision_detection import fcl
    plans=episode_root/"plans"
    result=json.loads((episode_root/"result.json").read_text())
    if not result["passed"]:
        raise ValueError("This audit covers saved successful episodes; failure logs remain separate")
    sidecar=episode_root/"robot_model_dependencies_retrospective.json"
    if not sidecar.exists():
        save_new(sidecar,robot_dependencies())
    metadata=json.loads((episode_root/"input.json").read_text())
    source=bind(strict_motion.__file__)
    if metadata["source_sha256"].get(source["path"]) != source["sha256"]:
        raise ValueError("Guard source differs from this episode; audit its archived implementation explicitly")
    receipt=json.loads((episode_root/"episode/receipt.json").read_text())
    raw=np.load(episode_root/"episode/arrays.npz")
    world=json.loads((plans/"collision_world.json").read_text())
    arm=int(metadata["proposal"]["active_arm"])
    urdf=WORKSPACE/"PolicyGen/mani_skill/assets/robots/panda/panda_v3.urdf"
    names=[node.attrib["name"] for node in ET.parse(urdf).getroot().findall("link")]
    planner=Planner(urdf=str(urdf),srdf=str(urdf.with_suffix(".srdf")),
                    user_link_names=names,move_group="panda_hand_tcp")
    guard=object.__new__(CheckedMotion)
    guard.root=raw["robot_roots_world"][arm]
    guard.planner=planner
    planner.set_base_pose(matrix_to_pose7(guard.root))
    guard.environment_collision_objects=[]
    guard.physical_robot_objects=[]
    guard.objects=[]
    def collision(path):
        geometry=fcl.load_mesh_as_BVH(str(path),np.ones(3))
        item=fcl.CollisionObject(geometry)
        guard.objects.append((geometry,item))
        return item
    for row in world["objects"]:
        obj=collision(row["mesh"]["path"])
        planner.set_normal_object(row["instance"],obj)
        guard.environment_collision_objects.append((row["instance"],obj))
    for name in ["panda_link0"]+world["table_checked_physical_links"]:
        obj=collision(plans/(name+"_physical_local.obj"))
        guard.physical_robot_objects.append((name,planner.link_name_2_idx[name],obj))
    guard.table_collision=collision(world["table_mesh"]["path"])
    guard.selected_collision=collision(world["selected_object_physical_mesh"]["path"])
    guard.payload_collision=collision(world["selected_object_physical_mesh"]["path"])
    guard.support_floor_z=receipt["receptacle"]["floor_profile"]["center_inner_floor_z_m"]
    planner.update_point_cloud(np.load(world["point_cloud"]["path"]),radius=world["point_radius_m"])
    rows=[]
    maximum_fk=0.
    maximum_attachment_error=0.
    for decision_path in sorted(plans.glob("*/decision.json")):
        decision=json.loads(decision_path.read_text())
        if not decision["accepted"]:
            continue
        request=json.loads((decision_path.parent/"request.json").read_text())
        qpos=np.asarray(request["start_qpos"])
        stage=request["stage"]
        frame=request["source_frame"]
        guard.selected_world_pose=pose7_to_matrix(raw["object_states"][frame,arm,:7])
        guard.use_attach=bool(request["use_attach"])
        if guard.use_attach:
            attachment=json.loads((plans/("attached_object_grasp.json" if stage=="lift" else "attached_object.json")).read_text())
            guard.payload_to_tcp=np.asarray(attachment["object_to_tcp"])
            planner.update_attached_mesh(attachment["collision_mesh"]["path"],matrix_to_pose7(guard.payload_to_tcp))
            # Attachment may subsequently slip slightly; report, do not replace it.
            actual=np.linalg.inv(raw["tcp_world"][frame,arm])@guard.selected_world_pose
            maximum_attachment_error=max(maximum_attachment_error,float(np.max(np.abs(actual-guard.payload_to_tcp))))
        else:
            guard.payload_to_tcp=None
            planner.planning_world.remove_attach()
        path=dict(np.load(decision_path.parent/"planner_path.npz"))
        report,dense,poses=guard.check(path,qpos,np.asarray(request["world_goal"]),stage)
        for q,tcp in zip(dense,poses):
            full=qpos.copy();full[:7]=q
            planner.pinocchio_model.compute_forward_kinematics(full)
            base_tcp=pose7_to_matrix(planner.pinocchio_model.get_link_pose(planner.move_group_link_id))
            maximum_fk=max(maximum_fk,float(np.max(np.abs(guard.root@base_tcp-tcp))))
        rows.append({"plan":decision_path.parent.name,"passed":report["passed"],
                     "dense_samples":len(dense),"report":report})
    native_fk=world_fk(raw["states"][:,arm*9:arm*9+7],guard.root)
    summary={
        "scope":"CPU reconstruction of saved accepted plans and world meshes; not a new physics rollout",
        "auditor":bind(__file__),"guard":source,"robot_dependencies":bind(sidecar),
        "strict_source_matches_execution_hash":True,
        "passed":all(row["passed"] for row in rows) and maximum_fk<1e-5,
        "accepted_paths_replayed":len(rows),"dense_samples_replayed":sum(x["dense_samples"] for x in rows),
        "pinocchio_world_TCP_vs_independent_FK_max_abs":maximum_fk,
        "native_TCP_vs_independent_FK_max_abs":float(np.max(np.abs(native_fk-raw["tcp_world"][:,arm]))),
        "max_attachment_transform_change_at_later_plan_start":maximum_attachment_error,
        "attachment_metric_note":"includes actual contact slip between initial attachment and later stages; not a registration error",
        "plans":rows,
        "remaining_limits":["dense joint sampling not continuous-time proof",
                            "snapshot world does not model future passive falling motion",
                            "gripper opening/closing verified by native contact, not separate finger trajectory planner"],
        "receipt_labels_modified":False,
    }
    output=episode_root/"strict_v3_cpu_collision_replay.json"
    save_new(output,summary)
    return {"episode":str(episode_root),"passed":summary["passed"],
            "plans":len(rows),"samples":summary["dense_samples_replayed"],
            "FK_max_abs":maximum_fk,"output":str(output)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes",type=Path,nargs="+")
    args=parser.parse_args()
    sys.path.insert(0,str(WORKSPACE/"PolicyGen"))
    results=[audit(path) for path in args.episodes]
    print(json.dumps(results,indent=2))
    raise SystemExit(0 if all(x["passed"] for x in results) else 1)


if __name__=="__main__":
    main()
