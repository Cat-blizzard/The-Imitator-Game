"""Bounded planners for the shared workbench; never supplies generation prose.

Insertion starts from the retained contact-based expert and records planned
paths. All new physical results are checked through the same v5 interpreter.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation
from .simulation import array
from .core import require

PORTS = np.array([[.055, -.014, .022], [.055, .014, .022]])
BODY_HALF = np.array([.018, .014, .018])


def tracking_residual(actual, desired):
    actual, desired = np.asarray(actual), np.asarray(desired)
    require(actual.shape == desired.shape == (4, 4), "tcp_tracking_requires_transforms")
    return {"position_error_m": float(np.linalg.norm(actual[:3, 3] - desired[:3, 3])),
            "rotation_error_rad": float(Rotation.from_matrix(actual[:3, :3].T @ desired[:3, :3]).magnitude())}


def tracking_solver(native, recorder):
    class Tracked(native):
        def follow_path(self, result, *args, **kwargs):
            if isinstance(result, dict) and "position" in result:
                recorder.paths.append({"arm": self.multi_robot_id, "stage": recorder.stage,
                    "start_frame": len(recorder.rows), "positions": np.asarray(result["position"]).tolist(),
                    "planner_status": result.get("status")})
            return super().follow_path(result, *args, **kwargs)
    return Tracked


# Retained insertion expert source SHA256: 8b1b34f60f307276bce047f6c73bda5f352fcb639b09c5f74c742a916df94af7
def solve_insertion(rec,program):
    import sapien,trimesh
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver as NativeSolver
    PandaArmMotionPlanningSolver = tracking_solver(NativeSolver, rec)
    from mani_skill.examples.motionplanning.panda.utils import compute_grasp_info_by_obb
    b=rec.unwrapped;agents=b.agent.agents
    planners=[PandaArmMotionPlanningSolver(rec,base_pose=a.robot.pose,multi_robot_id=i,vis=False,debug=False,
        visualize_target_grasp_pose=False,print_env_info=False,joint_vel_limits=.5,joint_acc_limits=.45)for i,a in enumerate(agents)]
    from examples.baselines.hypericl_task_expansion.table_obstacle import install
    for planner in planners:install(planner,b)
    events=[]
    def move(arm,pose,stage,refine=8):
        rec.stage=stage;p=planners[arm]
        other_mesh=agents[1-arm].robot.get_first_collision_mesh(True)
        points=trimesh.sample.sample_surface(other_mesh,10000,seed=arm)[0]
        p.planner.update_point_cloud(points,radius=.004);p.use_point_cloud=True
        current=array(agents[arm].tcp.pose.to_transformation_matrix()).reshape(4,4);desired=pose.to_transformation_matrix()
        if np.linalg.norm(current[:3,3]-desired[:3,3])<.0003 and Rotation.from_matrix(current[:3,:3].T@desired[:3,:3]).magnitude()<.002:
            rec.hold(refine,tuple(p.gripper_state for p in planners));return
        path=p.move_to_pose_with_screw(pose,dry_run=True)
        if isinstance(path,int)and path==-1:
            from examples.baselines.hypericlzq.eef import panda_ik
            root=array(agents[arm].robot.pose.to_transformation_matrix()).reshape(4,4);joints=array(agents[arm].robot.qpos).reshape(9)
            q,ok,_=panda_ik(np.linalg.inv(root)@desired,joints[:7])
            if bool(ok):
                candidate=p.planner.plan_qpos_to_qpos([np.r_[q,joints[7:]]],joints,time_step=b.control_timestep)
                if candidate['status']=='Success':path=candidate
        if isinstance(path,int)and path==-1:path=p.move_to_pose_with_RRTConnect(pose,dry_run=True)
        require(not isinstance(path,int)and len(path['position'])<=400,'Connector path failed '+stage)
        p.follow_path(path,other_gripper_state=planners[1-arm].gripper_state,refine_steps=refine)
    def translated(pose,offset):return sapien.Pose(pose.p+offset,pose.q)
    def object_target(arm,index,desired):
        T=desired@np.linalg.inv(array(b.objects[index].pose.to_transformation_matrix()).reshape(4,4))@array(agents[arm].tcp.pose.to_transformation_matrix()).reshape(4,4)
        return sapien.Pose(T)
    receiver_T=array(b.objects[0].pose.to_transformation_matrix()).reshape(4,4)
    center=(receiver_T@np.array([-.065,0,.012,1.]))[:3]
    closing=receiver_T[:3,:3]@np.array([0,1.,0]);pose=agents[0].build_grasp_pose([0,0,-1],closing,center)
    move(0,translated(pose,[0,0,.17]),'receiver:approach');move(0,pose,'receiver:reach')
    rec.stage='receiver:grasp';planners[0].close_gripper(other_gripper_state=planners[1].gripper_state,t=15)
    move(0,translated(pose,[0,0,.18]),'receiver:lift');rec.hold(12,tuple(p.gripper_state for p in planners))
    require(bool(array(agents[0].is_grasping(b.objects[0])).item()),'Receiver not physically held')
    desired=array(b.objects[0].pose.to_transformation_matrix()).reshape(4,4);desired[:3,3]=[-.015,0.,.18]
    move(0,object_target(0,0,desired),'receiver:hold_in_workspace');rec.hold(12,tuple(p.gripper_state for p in planners))
    rec.fixed_arm_commands[0]=rec.commands[-1][:8].copy()
    events.append(dict(stage='receiver_held',frame=len(rec.rows),arm=0))
    plug_T=array(b.objects[1].pose.to_transformation_matrix()).reshape(4,4)
    obb=trimesh.primitives.Box(extents=BODY_HALF*2,transform=plug_T)
    hint=array(agents[1].tcp.pose.to_transformation_matrix()).reshape(4,4)[:3,1]
    grasp=compute_grasp_info_by_obb(obb,approaching=[0,0,-1],target_closing=hint,depth=.010)
    pose=agents[1].build_grasp_pose([0,0,-1],grasp['closing'],grasp['center'])
    move(1,translated(pose,[0,0,.16]),'plug:approach');move(1,pose,'plug:reach')
    rec.stage='plug:grasp';planners[1].close_gripper(other_gripper_state=planners[0].gripper_state,t=15)
    require(bool(array(agents[1].is_grasping(b.objects[1])).item()),'Plug not physically grasped')
    events.append(dict(stage='both_objects_held',frame=len(rec.rows),arms=[0,1]))
    if program.startswith('insert'):
        move(1,translated(pose,[0,0,.18]),'plug:lift')
        port=int(program[-1]);receiver_T=array(b.objects[0].pose.to_transformation_matrix()).reshape(4,4)
        desired=receiver_T.copy();desired[:3,3]=(receiver_T@np.r_[PORTS[port,:2],.040,1.])[:3]
        goal=object_target(1,1,desired);move(1,translated(goal,[0,0,.065]),'plug:align_above_socket')
        # Follow the measured receiver, held by the other arm. No actor state changes.
        for depth in np.linspace(.055,0.,12):
            receiver_T=array(b.objects[0].pose.to_transformation_matrix()).reshape(4,4)
            desired=receiver_T.copy();desired[:3,3]=(receiver_T@np.r_[PORTS[port,:2],.040+depth,1.])[:3]
            move(1,object_target(1,1,desired),'plug:insert_with_receiver_held',refine=5)
            require(bool(array(agents[0].is_grasping(b.objects[0])).item()),'Receiver lost during insertion')
        rec.stage='plug:release_inserted';planners[1].open_gripper(other_gripper_state=planners[0].gripper_state)
        tcp=array(agents[1].tcp.pose.raw_pose).reshape(7);move(1,sapien.Pose(tcp[:3]+[0,0,.16],tcp[3:]),'plug:retreat')
    else:
        move(1,translated(pose,[0,0,.12]),'plug:extract_with_receiver_held')
        target=0 if program.endswith('brown')else 1
        desired=np.eye(4);desired[:3,:3]=Rotation.from_euler('y',np.pi/4).as_matrix();desired[:3,3]=b.parking_centers[target]+[0,0,.055]
        # The inclined tongue lands first. Account for its geometric roll to a
        # flat resting pose; final placement is still independently measured.
        desired[0,3]-=.042+(.0065-.042)/np.sqrt(2.)
        goal=object_target(1,1,desired);move(1,translated(goal,[0,0,.15]),'plug:above_parking');move(1,goal,'plug:park')
        rec.stage='plug:release';planners[1].open_gripper(other_gripper_state=planners[0].gripper_state);move(1,translated(goal,[0,0,.17]),'plug:retreat')
    rec.stage='final_receiver_still_held';rec.hold(40,tuple(p.gripper_state for p in planners))
    return events


def planners(rec, velocity=.6):
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
    cls = tracking_solver(PandaArmMotionPlanningSolver, rec)
    return [cls(rec, base_pose=a.robot.pose, multi_robot_id=i, vis=False, debug=False,
                visualize_target_grasp_pose=False, print_env_info=False,
                joint_vel_limits=velocity, joint_acc_limits=velocity) for i,a in enumerate(rec.agent.agents)]


def move(rec, solvers, arm, pose, stage, refine=5):
    rec.stage=stage
    path=solvers[arm].move_to_pose_with_screw(pose,dry_run=True)
    if isinstance(path,int) or len(path["position"])>350:
        path=solvers[arm].move_to_pose_with_RRTConnect(pose,dry_run=True)
    require(not isinstance(path,int) and path.get("status")=="Success" and len(path["position"])<=400,
            "workbench_motion_planning_failed:"+stage)
    solvers[arm].follow_path(path,other_gripper_state=solvers[1-arm].gripper_state,refine_steps=refine)


def solve_chain(rec,task):
    import sapien
    b=rec.unwrapped
    ps=planners(rec,.65)
    direction=task["planning_parameters"]["direction"]
    index=0 if direction==1 else 4
    axis=np.array([float(direction),0.,0.])
    point=rec.initial["object_states"][index,:3].copy();point[2]=.045
    before=point-axis*.085
    q=b.agent.agents[0].build_grasp_pose([0,0,-1],np.cross([0,0,1.],axis),before).q
    initial=array(b.agent.agents[0].robot.qpos).reshape(9).copy()
    rec.stage="chain:close_pusher";ps[0].close_gripper(other_gripper_state=1,t=10)
    move(rec,ps,0,sapien.Pose(before+[0,0,.17],q),"chain:approach")
    move(rec,ps,0,sapien.Pose(before,q),"chain:reach")
    move(rec,ps,0,sapien.Pose(point+axis*.035,q),"chain:trigger")
    rec.stage="chain:passive_settle";rec.hold(24,(-1,1))
    tcp=array(b.agent.agents[0].tcp.pose.p).reshape(3)
    move(rec,ps,0,sapien.Pose(tcp+[0,0,.19],q),"chain:retreat")
    rec.stage="chain:return"
    path=ps[0].planner.plan_qpos_to_qpos([initial],array(b.agent.agents[0].robot.qpos).reshape(9).copy(),time_step=b.control_timestep)
    require(path["status"]=="Success","chain_return_failed")
    ps[0].follow_path(path,other_gripper_state=1,refine_steps=3)
    ps[0].open_gripper(other_gripper_state=1)
    rec.stage="final_settle";rec.hold(40)


def solve_lid(rec,task):
    import sapien,trimesh
    from examples.baselines.hypericl_task_expansion.door_kinematics import rotate_about_hinge,grasp_arc_candidates
    from examples.baselines.hypericlzq.eef import panda_ik
    b=rec.unwrapped;spec=b.scene_spec["lid"];arm=1
    agent=b.agent.agents[arm];ps=planners(rec,.5)
    link=b.door_links[0]
    transform=array(link.pose.to_transformation_matrix()).reshape(4,4)
    center=(transform@np.r_[spec["handle_center_local"],1.])[:3]
    robot_root=array(agent.robot.pose.to_transformation_matrix()).reshape(4,4)
    pivot=np.array(spec["pivot_world"]);axis=np.array(spec["axis_world"])
    # A valid initial grasp may have no continuous hinge path. Search a finite
    # set of handle approaches BEFORE any contact command; this is planning,
    # not a physical retry or a changed opening goal.
    candidates=[(np.array([0.,0.,-1.]),np.array([0.,1.,0.]))]
    for pitch in (0.,-.35,-.70):
        for yaw in (-1.05,-.70,-.35,.35,.70,1.05):
            approach=np.array([np.cos(yaw)*np.cos(pitch),np.sin(yaw)*np.cos(pitch),np.sin(pitch)])
            closing=np.array([-np.sin(yaw),np.cos(yaw),0.])
            candidates.extend([(approach,closing),(approach,np.cross(approach,closing))])
    evaluations=[];selected=None
    for approach,closing in candidates:
        grasp=agent.build_grasp_pose(approach,closing,center)
        choices=grasp_arc_candidates(grasp.to_transformation_matrix(),robot_root,pivot,axis,1.26,
                                      array(agent.robot.qpos).reshape(9)[:7])
        evaluations.append({"approach":approach.tolist(),"closing":closing.tolist(),"complete_arcs":len(choices)})
        if choices:
            selected=(grasp,approach,choices[0]);break
    from .core import write
    write(rec.directory/"grasp_preflight.json",{"candidates":evaluations,"before_contact_commands":True,
          "fixed_finite_search":True,"goal_changed":False})
    require(selected is not None,"lid_has_no_continuous_ik_arc")
    grasp,approach,choice=selected
    from examples.baselines.hypericl_task_expansion.table_obstacle import install
    install(ps[arm],b)
    obstacles=[o.get_first_collision_mesh(True) for o in b.objects+b.support_actors]
    obstacles.append(b.agent.agents[0].robot.get_first_collision_mesh(True))
    cloud=np.concatenate([trimesh.sample.sample_surface(mesh,1500,seed=i)[0] for i,mesh in enumerate(obstacles)])
    ps[arm].planner.update_point_cloud(cloud,radius=.002);ps[arm].use_point_cloud=True
    above=sapien.Pose(grasp.p-approach*.14+[0,0,.10],grasp.q)
    preferred,ok,_=panda_ik(np.linalg.inv(robot_root)@above.to_transformation_matrix(),choice["initial_q"])
    require(bool(ok),"lid_approach_ik_failed")
    state=array(agent.robot.qpos).reshape(9)
    path=ps[arm].planner.plan_qpos_to_qpos([np.r_[preferred,state[7:]]],state.copy(),time_step=b.control_timestep,
                                         use_point_cloud=True,planning_time=5.)
    require(path.get("status")=="Success" and len(path["position"])<=400,"lid_approach_path_failed")
    rec.stage="lid:approach";ps[arm].follow_path(path,other_gripper_state=1,refine_steps=5)
    move(rec,ps,arm,sapien.Pose(grasp.p-approach*.045,grasp.q),"lid:precontact")
    move(rec,ps,arm,grasp,"lid:grasp_reach")
    actual=array(agent.tcp.pose.to_transformation_matrix()).reshape(4,4)
    residual=tracking_residual(actual,grasp.to_transformation_matrix())
    write(rec.directory/"contact_approach.json",{"frame":len(rec.rows)-1,"arm":arm,
          "desired_tcp_world":grasp.to_transformation_matrix().tolist(),"actual_tcp_world":actual.tolist(),
          **residual,"scope":"controller tracking before closure; does not establish contact or grasp"})
    require(residual["position_error_m"]<=.005 and residual["rotation_error_rad"]<=.05,
            "functional_grasp_pose_not_reached")
    rec.stage="lid:close";ps[arm].close_gripper(other_gripper_state=1,t=18)
    require(bool(array(agent.is_grasping(link)).item()),"lid_handle_not_physically_grasped")
    local=np.linalg.inv(array(link.pose.to_transformation_matrix()).reshape(4,4))@array(agent.tcp.pose.to_transformation_matrix()).reshape(4,4)
    for _ in range(70):
        angle=float(array(b.cabinet.qpos).reshape(1)[0])
        if angle>=1.24:break
        transform=array(link.pose.to_transformation_matrix()).reshape(4,4)
        target=rotate_about_hinge(transform,pivot,axis,min(.04,1.27-angle))@local
        state=array(agent.robot.qpos).reshape(9)
        joints,ok,_=panda_ik(np.linalg.inv(robot_root)@target,state[:7])
        require(bool(ok) and np.max(np.abs(joints-state[:7]))<.2,"lid_continuous_ik_failed")
        rec.stage="lid:follow_hinge"
        rec.paths.append({"arm":arm,"stage":rec.stage,"start_frame":len(rec.rows),"positions":[joints.tolist()],"planner_status":"Success"})
        command={f"panda_wristcam-{arm}":np.r_[joints,-1.],
                 "panda_wristcam-0":np.r_[array(b.agent.agents[0].robot.qpos).reshape(9)[:7],1.]}
        for _ in range(5):rec.step(command)
        require(bool(array(agent.is_grasping(link)).item()),"lid_grasp_lost_during_opening")
    require(float(array(b.cabinet.qpos).reshape(1)[0])>=1.2,"lid_target_not_reached")
    rec.stage="lid:release";ps[arm].open_gripper(other_gripper_state=1)
    tcp=array(agent.tcp.pose.raw_pose).reshape(7)
    move(rec,ps,arm,sapien.Pose(tcp[:3]+[0,0,.13],tcp[3:]),"lid:withdraw")
    rec.stage="final_settle";rec.hold(40)


def solve(rec,task):
    if task["family"]=="passive_chain":return solve_chain(rec,task)
    if task["family"]=="articulation":return solve_lid(rec,task)
    if task["family"]=="insertion":
        from .workbench import ObjectView
        return solve_insertion(ObjectView(rec,["receiver","plug"]),"insert_port"+str(task["planning_parameters"]["port"]))
    raise ValueError("unsupported_planning_mechanism")
