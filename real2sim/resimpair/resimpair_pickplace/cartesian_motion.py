"""Opt-in Cartesian lift candidate; all other retained grasp/place steps copied.

Not the default controller. Same independent guard and acceptance. Physical
validation is pending; the CPU regression uses recorded failures only.
"""
from pathlib import Path
import numpy as np
from resimpair_grasp.common import read,resolve,write
from resimpair_grasp.motion import MeshMotion
from .stability_motion import StabilityMotion,settle_window
from .stability import window_stability,opening_width,aperture_command
from .cartesian_lift import cartesian_path
from corl2026_workshop.experiments.autonomous_pipeline_v4.simulation import array,targets
from data_generation_methods.human_to_robot_pipeline.strict_motion import make_solver,world_fk,matrix_to_pose7


class CartesianLiftMotion(StabilityMotion):
    def plan(self,pose,stage,current=None):
        if stage!='lift':return super().plan(pose,stage,current)
        current=array(self.agent.robot.qpos).reshape(9).copy() if current is None else current.copy()
        if self.use_attach:self.refresh_attachment(stage)
        goal=np.asarray(pose.to_transformation_matrix())
        directory=self.directory/f'{self.counter:03d}_{stage}_cartesian'
        self.counter+=1;directory.mkdir()
        write(directory/'request.json',{'stage':stage,'current':current.tolist(),'goal':goal.tolist(),
            'frame':len(self.rec.rows),'attached':self.use_attach})
        try:
            with self.rec.performance.section('motion_plan_cartesian'):
                path=cartesian_path(self.planner,self.root,current,goal,self.rec.control_timestep)
            with self.rec.performance.section('dense_physical_collision_guard'):
                report,dense,poses=self.check(path,current,goal,stage)
            np.savez_compressed(directory/'path.npz',positions=path['position'],current=current,goal=goal)
            if dense is not None:np.savez_compressed(directory/'dense.npz',joints=dense,tcp=poses)
            write(directory/'decision.json',{'accepted':report['passed'],'report':report})
            if report['passed']:
                path['guard_evidence']=str(directory/'decision.json');return path
        except Exception as error:
            if not (directory/'decision.json').exists():write(directory/'decision.json',{'accepted':False,'error':repr(error)})
        return MeshMotion.plan(self,pose,stage,current)


def solve(rec, task, method):
    import sapien
    from scipy.spatial.transform import Rotation
    from mani_skill.examples.motionplanning.panda.utils import get_actor_obb, compute_grasp_info_by_obb
    from resimpair_grasp.grasps import load
    destinations = targets(rec.scene_spec, task)
    for op in task["operations"]:
        index = rec.object_ids.index(op["object"]); arm = op["arm"]
        obj = rec.objects[index]; agent = rec.agent.agents[arm]
        base = make_solver(rec, arm, rec.directory / (op["id"] + "_motion"))
        base.__class__ = CartesianLiftMotion
        guard = base; guard.install(index); motion = guard.motion
        initial_tcp = array(agent.tcp.pose.to_transformation_matrix()).reshape(4, 4).copy()
        selected = rec.scene_spec["objects"][index]
        def follow(path, stage):
            rec.stage = op["id"] + ":" + stage
            rec.paths.append({"arm": arm, "stage": rec.stage, "start_frame": len(rec.rows),
                "positions": np.asarray(path["position"]).tolist(), "planner_status": path["status"],
                "guard_evidence": path["guard_evidence"]})
            motion.follow_path(path, other_gripper_state=1, refine_steps=5)
        def move(pose, stage): follow(guard.plan(pose, stage), stage)
        object_T = array(obj.pose.to_transformation_matrix()).reshape(4, 4)
        if method == "baseline_obb":
            # Retained mesh baseline: a single OBB grasp, depth=min(.04,h/2).
            info = compute_grasp_info_by_obb(get_actor_obb(obj), approaching=[0, 0, -1],
                target_closing=initial_tcp[:3, 1], depth=min(.04, selected["half_size"][2]))
            if info["extents"][1] > .077: raise ValueError("baseline_opening_exceeded")
            poses = [("obb", agent.build_grasp_pose([0, 0, -1], info["closing"], info["center"]))]
        elif method == "cached_antipodal":
            cache = load(rec.scene_spec["grasp_caches"][op["object"]], selected["asset_descriptor"])
            poses = [(c["id"], sapien.Pose(object_T @ np.asarray(c["tcp_in_asset"]))) for c in cache["candidates"][:12]]
        else:
            raise ValueError("Unknown controller")
        decisions = []; chosen = None
        current = array(agent.robot.qpos).reshape(9).copy()
        for candidate_id, pose in poses:
            try:
                approach = sapien.Pose(pose.p + [0, 0, .17], pose.q)
                path_a = guard.plan(approach, "approach", current)
                at_a = current.copy(); at_a[:7] = path_a["position"][-1]
                path_b = guard.plan(pose, "grasp_reach", at_a)
                decisions.append({"candidate": candidate_id, "accepted_before_motion": True})
                chosen = candidate_id; break
            except RuntimeError as error:
                decisions.append({"candidate": candidate_id, "accepted_before_motion": False, "reason": str(error)})
        write(rec.directory / (op["id"] + "_grasp_selection.json"), {"method": method, "decisions": decisions,
            "selected": chosen, "maximum_candidates": 12, "physical_retries": 0})
        if chosen is None: raise RuntimeError("no_reachable_collision_checked_grasp")
        follow(path_a, "approach"); follow(path_b, "grasp_reach")
        after_reach = array(obj.pose.to_transformation_matrix()).reshape(4, 4)
        if np.linalg.norm(after_reach[:3, 3] - object_T[:3, 3]) > .008:
            raise RuntimeError("target_moved_during_open_finger_approach")
        rec.stage = op["id"] + ":close"; motion.close_gripper(other_gripper_state=1, t=12)
        if not bool(array(agent.is_grasping(obj)).item()): raise RuntimeError("no_measured_two_finger_grasp_after_close")
        guard.attach_lifted_object(obj, phase="grasp")
        # Close/contact changes the realized TCP slightly. Start a vertical lift
        # from that measured pose, not the earlier requested open-hand pose.
        # This avoids combining initial lift with an unnecessary reorientation.
        lift_T = array(agent.tcp.pose.to_transformation_matrix()).reshape(4, 4).copy()
        lift_T[2, 3] += .20
        move(sapien.Pose(lift_T), "lift")
        settle_window(rec,arm,index,op["id"]+":lift_settle")
        if not bool(array(agent.is_grasping(obj)).item()): raise RuntimeError("lost_grasp_after_lift")
        desired = np.eye(4); desired[:3, :3] = Rotation.from_euler("z", op["target"]["yaw"]).as_matrix()
        desired[:3, 3] = np.asarray(destinations[op["object"]]) + [0, 0, .002]
        actual = array(obj.pose.to_transformation_matrix()).reshape(4, 4)
        tcp = array(agent.tcp.pose.to_transformation_matrix()).reshape(4, 4)
        place = sapien.Pose(desired @ np.linalg.inv(actual) @ tcp)
        move(sapien.Pose(place.p + [0, 0, .18], place.q), "above_bowl")
        settle_window(rec,arm,index,op["id"]+":transfer_settle")
        actual=array(obj.pose.to_transformation_matrix()).reshape(4,4)
        tcp=array(agent.tcp.pose.to_transformation_matrix()).reshape(4,4)
        place=sapien.Pose(desired @ np.linalg.inv(actual) @ tcp)
        move(place, "release_above_rim")
        from corl2026_workshop.experiments.autonomous_pipeline_v4.release_control import readiness
        checks = []
        for correction in range(3):
            rec.hold(5, (-1, 1) if arm == 0 else (1, -1))
            actual = array(obj.pose.to_transformation_matrix()).reshape(4, 4)
            check = readiness(actual, destinations[op["object"]], bool(array(agent.is_grasping(obj)).item()))
            checks.append(check)
            if check["ready_to_release"] or correction == 2: break
            tcp = array(agent.tcp.pose.to_transformation_matrix()).reshape(4, 4)
            place = sapien.Pose(desired @ np.linalg.inv(actual) @ tcp)
            move(place, "release_above_rim")
        write(rec.directory / (op["id"] + "_release_control.json"), {"checks": checks})
        if not check["ready_to_release"]: raise RuntimeError("measured_release_precondition_failed")
        settle_window(rec,arm,index,op["id"]+":release_settle")
        current=array(agent.robot.qpos).reshape(9).copy()
        width=opening_width(float(current[7:].sum()))
        opening=guard.opening_sweep(width)
        write(rec.directory/(op["id"]+"_opening_sweep.json"),opening)
        if not opening['passed']:raise RuntimeError('native_opening_sweep_collision')
        # Check a retreat at the intended aperture BEFORE irreversible release.
        future=current.copy();future[7:]=width/2
        tcp=array(agent.tcp.pose.to_transformation_matrix()).reshape(4,4).copy()
        retreat=tcp.copy();retreat[2,3]+=.18
        guard.use_attach=False
        try:guard.plan(sapien.Pose(retreat),"retreat",future)
        finally:guard.use_attach=True
        rec.stage=op["id"]+":open"
        cfg=agent.controller.controllers["gripper"].config
        if not cfg.normalize_action or cfg.use_delta:raise ValueError("absolute_normalized_gripper_required")
        lower=float(cfg.lower);upper=float(cfg.upper)
        command=aperture_command(width,lower,upper)
        state=rec.measure()['states']
        for step in range(1,9):
            # Gradual opening to measured width + clearance, not full aperture
            # while the hand is still inside the container.
            aperture=float(current[7:].sum())+(width-float(current[7:].sum()))*step/8
            actions={f'panda_wristcam-{i}':np.r_[state[i*9:i*9+7],aperture_command(aperture,lower,upper) if i==arm else 1.] for i in range(2)}
            rec.step(actions)
        motion.gripper_state=command
        rec.hold(4,(command,1) if arm==0 else (1,command))
        if bool(array(agent.is_grasping(obj)).item()):raise RuntimeError('object_still_grasped_after_bounded_open')
        guard.detach()
        move(sapien.Pose(retreat), "retreat")
        # Only open fully after clearing the bowl; full aperture is checked too.
        opening=guard.opening_sweep(.08)
        write(rec.directory/(op["id"]+"_clear_opening_sweep.json"),opening)
        if not opening['passed']:raise RuntimeError('native_clear_opening_sweep_collision')
        motion.open_gripper(other_gripper_state=1)
        move(sapien.Pose(initial_tcp), "return")
        rec.hold(20)
    rec.stage = "final_settle"; rec.hold(40)
