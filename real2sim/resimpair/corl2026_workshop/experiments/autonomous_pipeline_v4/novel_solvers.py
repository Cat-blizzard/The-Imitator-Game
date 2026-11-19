"""Bounded robot motion for ten mechanisms; object state is never set here."""
import math
import numpy as np

from .core import require,write
from .simulation import array


def rolling_pusher_pose(agent,point):
    # Native Panda checks vector dot products; plain list @ list is invalid.
    return agent.build_grasp_pose(np.array([0.,0.,-1.]),np.array([0.,-1.,0.]),np.asarray(point,dtype=float))


def solve(rec,task):
    import sapien
    from .workbench_solvers import planners, move
    b=rec.unwrapped;kind=task["scene_kind"];params=task["planning_parameters"]
    variation=task.get("trajectory_variation",{})
    speed=float(variation.get("speed_scale",1.))
    clearance=float(variation.get("clearance_delta_m",0.))
    require(.65<=speed<=1.15 and 0.<=clearance<=.04,"trajectory_variation_out_of_qualified_bounds")
    ps=planners(rec,.40*speed)
    initial=[array(a.robot.qpos).reshape(9).copy() for a in b.agent.agents]

    def goto(arm,xyz,stage,q=None):
        xyz=np.asarray(xyz,dtype=float).copy()
        if "above" in stage:xyz[2]+=clearance
        if q is None:q=b.agent.agents[arm].build_grasp_pose(np.array([0.,0.,-1.]),np.array([0.,1.,0.]),np.asarray(xyz)).q
        move(rec,ps,arm,sapien.Pose(xyz,q),stage,refine=8)

    def hold(steps=12):rec.hold(steps,tuple(p.gripper_state for p in ps))

    def retreat(arm):
        xyz=array(b.agent.agents[arm].tcp.pose.p).reshape(3)
        q=array(b.agent.agents[arm].tcp.pose.q).reshape(4)
        goto(arm,xyz+[0,0,.15+clearance],"contact:retreat",q)

    def home(arm):
        path=ps[arm].planner.plan_qpos_to_qpos([initial[arm]],array(b.agent.agents[arm].robot.qpos).reshape(9),time_step=b.control_timestep)
        require(not isinstance(path,int) and path.get("status")=="Success" and len(path["position"])<=400,"novel_return_path_failed")
        rec.stage="robot:return";ps[arm].follow_path(path,other_gripper_state=ps[1-arm].gripper_state,refine_steps=5)

    if b.scene_spec["joints"]:
        chosen=params.get("selected",[0])
        for index in chosen:
            arm=index if len(b.articulations)>1 else 0
            spec=b.scene_spec["joints"][index];c=np.array(spec["pivot_world"])
            rec.stage="contact:close_pusher";ps[arm].close_gripper(other_gripper_state=ps[1-arm].gripper_state,t=8)
            if kind=="spring_console":
                goto(arm,c+[0,0,.15],"plunger:above")
                goto(arm,c+[0,0,.022],"plunger:contact")
                goto(arm,c+[0,0,-.008],"plunger:depress");hold(12)
                retreat(arm);hold(20)
            elif kind=="linear_faders":
                goto(arm,c+[-.07,0,.15],"fader:above")
                goto(arm,c+[-.065,0,0],"fader:behind")
                goto(arm,c+[.045,0,0],"fader:slide");hold(10);retreat(arm)
            elif kind in {"rocker_console","balance_beam"}:
                direction=params.get("direction",1 if index==0 else -1)
                radius=.052 if kind=="rocker_console" else .095
                target=direction*(.48 if kind=="rocker_console" else .32)
                point=c+[direction*radius,0,0]
                goto(arm,point+[0,0,.16],"lever:above")
                goto(arm,point+[0,0,.035],"lever:contact")
                end=c+[direction*radius*math.cos(target),0,-direction*radius*math.sin(target)+.012]
                goto(arm,end,"lever:tilt");hold(12);retreat(arm)
                if params.get("return_to_origin"):
                    other=c+[-direction*radius*math.cos(target),0,direction*radius*math.sin(target)]
                    goto(arm,other+[0,0,.14],"lever:return_above")
                    goto(arm,c+[-direction*radius,0,.012],"lever:return_center");hold(12);retreat(arm)
            else:
                direction=params["direction"];radius=.065
                def arc(start,end):
                    angles=np.linspace(start,end,9)
                    def point(a):
                        tangent=direction*np.array([-math.sin(a),math.cos(a),0.])
                        return c+[radius*math.cos(a),radius*math.sin(a),.022]-tangent*.038
                    first=point(start);goto(arm,first+[0,0,.15],"dial:above");goto(arm,first,"dial:engage")
                    for a in angles[1:]:goto(arm,point(a),"dial:turn")
                    hold(12);retreat(arm)
                arc(0.,direction*.95)
                if params.get("return_to_origin"):
                    start=float(array(b.articulations[0].qpos).reshape(-1)[0]);direction=-direction
                    arc(start,0.)
            home(arm);ps[arm].open_gripper(other_gripper_state=ps[1-arm].gripper_state)
        hold(40);return

    if kind=="rolling_lanes":
        from .tool_motion import linear
        from examples.baselines.hypericl_task_expansion.table_obstacle import install
        for index in params["selected"]:
            arm=index;ball=b.objects[b.object_ids.index("ball_"+str(index))]
            point=array(ball.pose.p).reshape(3);point[2]=.045
            rec.stage="ball:close_pusher";ps[arm].close_gripper(other_gripper_state=ps[1-arm].gripper_state,t=8)
            install(ps[arm],b)
            q=rolling_pusher_pose(b.agent.agents[arm],point).q
            goto(arm,point+[-.085,0,.16],"ball:above",q)
            linear(rec,ps,arm,sapien.Pose(point+[-.085,0,0],q),"ball:behind")
            linear(rec,ps,arm,sapien.Pose([.145,point[1],point[2]],q),"ball:roll_forward")
            retreat(arm);home(arm)
            ps[arm].open_gripper(other_gripper_state=ps[1-arm].gripper_state)
        hold(55);return

    tool_id={"hook_retrieval":"hook","granule_sweeping":"paddle","contact_drafting":"stylus","bead_stirring":"stirrer"}[kind]
    obj=b.objects[b.object_ids.index(tool_id)];item=b.scene_spec["objects"][b.object_ids.index(tool_id)]
    from scipy.spatial.transform import Rotation
    from .tool_motion import linear
    from .workbench_solvers import tracking_residual
    from examples.baselines.hypericl_task_expansion.table_obstacle import install
    write(rec.directory/"tool_table_obstacle.json",install(ps[0],b))
    transform=array(obj.pose.to_transformation_matrix()).reshape(4,4)
    require(Rotation.from_matrix(transform[:3,:3]).magnitude()<.12,"tool_not_in_declared_resting_orientation")
    feature=item["grasp_feature"]
    center=(transform@np.r_[feature["center_local"],1.])[:3]
    closing=transform[:3,:3]@np.array(feature["closing_local"])
    from examples.baselines.hypericlzq.eef import panda_ik,panda_chain
    limits=np.array([bounds for _,active,bounds in panda_chain() if active])
    robot_root=array(b.agent.agents[0].robot.pose.to_transformation_matrix()).reshape(4,4)
    state=array(b.agent.agents[0].robot.qpos).reshape(9)
    candidates=[]
    for sign in [1.,-1.]:
        pose=b.agent.agents[0].build_grasp_pose([0,0,-1],closing*sign,center)
        seed=state[:7].copy();margin=float("inf");valid=True;approach_q=None
        for dz in np.linspace(.15+clearance,0,25):
            target=sapien.Pose(center+[0,0,dz],pose.q).to_transformation_matrix()
            seed,ok,_=panda_ik(np.linalg.inv(robot_root)@target,seed)
            if approach_q is None:approach_q=seed.copy()
            valid &= bool(ok);margin=min(margin,float(np.minimum(seed-limits[:,0],limits[:,1]-seed).min()))
        candidates.append({"sign":sign,"valid":valid,"joint_margin_rad":margin,"pose":pose,"approach_q":approach_q})
    valid=[x for x in candidates if x["valid"] and x["joint_margin_rad"]>.02]
    write(rec.directory/"tool_grasp_candidates.json",{"finite_symmetry_search":True,
        "candidates":[{k:v for k,v in x.items() if k not in {"pose","approach_q"}} for x in candidates]})
    require(bool(valid),"no_tool_grasp_with_continuous_joint_margin")
    chosen=max(valid,key=lambda x:x["joint_margin_rad"]);grasp=chosen["pose"]
    path=ps[0].planner.plan_qpos_to_qpos([np.r_[chosen["approach_q"],state[7:]]],state,time_step=b.control_timestep)
    require(path.get("status")=="Success" and len(path["position"])<=400,"tool_approach_path_failed")
    require(np.all(path["position"]>=limits[:,0]) and np.all(path["position"]<=limits[:,1]),"tool_approach_command_limits")
    rec.stage="tool:approach";ps[0].follow_path(path,other_gripper_state=ps[1].gripper_state,refine_steps=8)
    linear(rec,ps,0,grasp,"tool:reach")
    residual=tracking_residual(array(b.agent.agents[0].tcp.pose.to_transformation_matrix()).reshape(4,4),grasp.to_transformation_matrix())
    now=array(obj.pose.to_transformation_matrix()).reshape(4,4)
    feature_drift=float(np.linalg.norm((now@np.r_[feature["center_local"],1.])[:3]-center))
    write(rec.directory/"tool_grasp_approach.json",{"frame":len(rec.rows)-1,**residual,"feature_drift_m":feature_drift})
    require(residual["position_error_m"]<.005 and residual["rotation_error_rad"]<.05 and feature_drift<.008,
            "functional_grasp_feature_not_reached:"+tool_id)
    rec.stage="tool:grasp";ps[0].close_gripper(other_gripper_state=ps[1].gripper_state,t=18)
    require(bool(array(b.agent.agents[0].is_grasping(obj)).item()),"tool_not_physically_grasped:"+tool_id)
    linear(rec,ps,0,sapien.Pose(center+[0,0,.16+clearance],grasp.q),"tool:lift");hold(12)
    require(bool(array(b.agent.agents[0].is_grasping(obj)).item()),"tool_not_physically_held:"+tool_id)
    rotation=transform[:3,:3].copy()
    feedback=[]
    def tool_to(point,stage,contact_goal=None):
        desired=np.eye(4);desired[:3,:3]=rotation;desired[:3,3]=point
        for correction in range(4):
            require(bool(array(b.agent.agents[0].is_grasping(obj)).item()),"tool_grasp_lost:"+stage)
            current=array(obj.pose.to_transformation_matrix()).reshape(4,4)
            offset=np.linalg.inv(current)@array(b.agent.agents[0].tcp.pose.to_transformation_matrix()).reshape(4,4)
            linear(rec,ps,0,sapien.Pose(desired@offset),stage)
            actual=array(obj.pose.to_transformation_matrix()).reshape(4,4)
            residual=tracking_residual(actual,desired)
            task_stop=bool(contact_goal()) if contact_goal is not None else False
            feedback.append({"stage":stage,"correction":correction,"frame":len(rec.rows)-1,
                             "desired_tool_world":desired.tolist(),"actual_tool_world":actual.tolist(),
                             "measured_cargo_stop_condition":task_stop,**residual})
            # Contact can compliantly rotate a held hook. Stop pulling when
            # the measured cargo is inside its declared region. Do not force
            # unnecessary six-DOF tool alignment against a completed contact.
            # This is a controller stop, never an acceptance flag: the frozen
            # independent contact/order/no-direct-hand predicates still run.
            if task_stop or (residual["position_error_m"]<.007 and residual["rotation_error_rad"]<.07):break
        write(rec.directory/("tool_tracking_%03d.json"%len(feedback)),{"measurements":feedback[-4:]})
        require(task_stop or (residual["position_error_m"]<.012 and residual["rotation_error_rad"]<.10),"tool_target_tracking_failed:"+stage)
    if kind=="hook_retrieval":
        for index in params["selected"]:
            cargo=b.objects[b.object_ids.index("cargo_"+str(index))]
            pos=array(cargo.pose.p).reshape(3)
            start=np.array([pos[0]+.040-.215,pos[1]-.045,.017])
            destination=next(o for o in b.scene_spec["objects"] if o["id"]=="cargo_destination_"+str(index))
            cx=destination["position"][0]
            shift=cx-pos[0]
            tool_to(start+[0,0,.17+clearance],"hook:above_far_face");tool_to(start,"hook:engage_far_face")
            def cargo_arrived():
                measured=array(cargo.pose.p).reshape(3)
                return cx-.07<measured[0]<cx+.065 and abs(measured[1]-pos[1])<.09 and measured[2]<.08
            distance=shift if b.scene_spec.get("batch_generation") else -.17
            if b.scene_spec.get("batch_generation"):
                # The goal is a cargo region, not exact alignment of the held
                # hook at its farthest nominal endpoint. Plan short contact
                # increments, inspect measured cargo after each, and stop on
                # the original declared goal. This avoids planning an unused
                # unreachable endpoint before any useful contact can happen.
                for delta in np.linspace(0., distance, max(2,int(np.ceil(abs(distance)/.015))+1))[1:]:
                    tool_to(start+[delta,0,0],"hook:pull",contact_goal=cargo_arrived)
                    if cargo_arrived(): break
                require(cargo_arrived(),"hook_cargo_not_in_declared_region_after_bounded_pull")
                hold(12)
                actual_tool=array(obj.pose.p).reshape(3)
                tool_to(actual_tool+[0,0,.16+clearance],"hook:disengage")
            else:
                tool_to(start+[distance,0,0],"hook:pull",contact_goal=cargo_arrived);hold(12)
                tool_to(start+[distance,0,.16+clearance],"hook:disengage")
    elif kind=="granule_sweeping":
        for index in params["selected"]:
            if b.scene_spec.get("batch_generation"):
                collector=next(o for o in b.scene_spec["objects"] if o["id"]==f"collector_{index}")
                y=collector["position"][1];dx=collector["position"][0]-.20
            else:y=[-.23,.12][index];dx=0.
            start=np.array([-.23+dx,y,.021]);end=np.array([.04+dx,y,.021])
            tool_to(start+[0,0,.16+clearance],"paddle:above");tool_to(start,"paddle:lower")
            tool_to(end,"paddle:sweep");hold(18);tool_to(end+[0,0,.16+clearance],"paddle:lift")
    else:
        tip_offset=rotation@np.array(item["tip_local"])
        if kind=="contact_drafting":points=[np.array(x+[params["height"]-.001]) for x in params["points"]]
        else:
            points=[np.array([-.10+.052*math.cos(a),-.18+.052*math.sin(a),.026]) for a in
                    np.linspace(0,params["direction"]*2*math.pi*params["turns"],12*params["turns"]+1)]
        tool_to(points[0]-tip_offset+[0,0,.16],"contact_tool:above")
        if kind=="bead_stirring":
            entry=np.array(b.scene_spec["tool_free_entry_world"])
            tool_to(entry-tip_offset+[0,0,.16],"stirrer:above_free_entry")
            tool_to(entry-tip_offset,"stirrer:enter_free_space")
            angle=-params["direction"]*math.pi/6
            gap=np.array([-.10+.052*math.cos(angle),-.18+.052*math.sin(angle),.026])
            tool_to(gap-tip_offset,"stirrer:approach_between_beads")
        for point in points:tool_to(point-tip_offset,"contact_tool:trace")
        hold(8);tool_to(points[-1]-tip_offset+[0,0,.16],"contact_tool:withdraw")
    hold(40)
