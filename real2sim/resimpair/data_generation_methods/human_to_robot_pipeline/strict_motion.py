"""Independent, logged motion planning using the installed upstream mplib.

Every returned path is rechecked before execution. "Full path" means all
commands plus dense joint interpolation, not a formal continuous-time proof.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .planning_contract import bind, save_new
from .plan_geometry_validator import validate_geometry


def densify_joints(path, maximum_step=.015):
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 7 or len(path) < 1 or not np.isfinite(path).all():
        raise ValueError("Expected finite [N,7] arm joint path")
    if maximum_step <= 0:
        raise ValueError("Interpolation step must be positive")
    dense = [path[0]]
    for start, end in zip(path[:-1], path[1:]):
        steps = max(1, int(np.ceil(np.max(np.abs(end-start))/maximum_step)))
        dense.extend(start+(end-start)*(i/steps) for i in range(1,steps+1))
    return np.asarray(dense)


def validate_joint_path(path, limits, start, *, max_command_step=.08, maximum_start_error=.015):
    path = np.asarray(path, dtype=float)
    limits = np.asarray(limits)
    failures = []
    if path.ndim != 2 or path.shape[1] != 7 or not len(path) or not np.isfinite(path).all():
        return {"passed": False, "failures": ["invalid_or_nonfinite_joint_path"]}
    if len(path) > 300:
        failures.append("path_exceeds300_commands")
    if np.max(np.abs(path[0]-np.asarray(start)[:7])) > maximum_start_error:
        failures.append("path_start_discontinuous")
    if ((path < limits[:7,0]-1e-6) | (path > limits[:7,1]+1e-6)).any():
        failures.append("joint_limit")
    maximum = float(np.abs(np.diff(path,axis=0)).max(initial=0))
    if maximum > max_command_step:
        failures.append("joint_command_step")
    return {"passed": not failures, "failures": failures, "max_command_delta_rad": maximum,
            "max_command_step_rad": max_command_step, "commands": len(path)}


def world_fk(q, root):
    from examples.baselines.hypericlzq.eef import panda_fk
    q = np.asarray(q)
    return np.asarray(root)[None]@panda_fk(q[:,None,:])[:,0]


def sampled_mesh_points(mesh, count, seed):
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(mesh.faces), count, p=np.asarray(mesh.area_faces)/mesh.area)
    triangles = np.asarray(mesh.triangles)[ids]
    u, v = np.sqrt(rng.random(count)), rng.random(count)
    return ((1-u)[:,None]*triangles[:,0] +
            (u*(1-v))[:,None]*triangles[:,1] + (u*v)[:,None]*triangles[:,2])


def matrix_to_pose7(matrix):
    from scipy.spatial.transform import Rotation
    quat = Rotation.from_matrix(np.asarray(matrix)[:3,:3]).as_quat()
    return np.r_[np.asarray(matrix)[:3,3],quat[3],quat[:3]]


def pose7_to_matrix(pose):
    from scipy.spatial.transform import Rotation
    pose = np.asarray(pose)
    matrix = np.eye(4)
    matrix[:3,:3] = Rotation.from_quat(np.r_[pose[4:7],pose[3]]).as_matrix()
    matrix[:3,3] = pose[:3]
    return matrix


def payload_goal(vertices_in_tcp, rotation, target_xy, desired_bottom_z):
    """Place the held mesh's bounds, not an arbitrary TCP offset, at the goal."""
    rotated = np.asarray(vertices_in_tcp)@np.asarray(rotation).T
    bounds = np.array([rotated.min(0),rotated.max(0)])
    point = np.r_[np.asarray(target_xy)-bounds.mean(0)[:2],
                  desired_bottom_z-bounds[0,2]]
    return point


def corridor_height(bounds, start_xy, end_xy, payload_radius, minimum):
    """Conservative XY segment/AABB test; exact guard still checks actual path."""
    start, end = np.asarray(start_xy), np.asarray(end_xy)
    low, high = np.minimum(start,end)-payload_radius, np.maximum(start,end)+payload_radius
    selected = [box for box in bounds if
                np.all(np.minimum(box[1,:2],high) >= np.maximum(box[0,:2],low))]
    return max([minimum]+[float(box[1,2])+.04 for box in selected])


def make_solver(rec, arm, directory):
    """A fresh official Planner per robot; no monkeypatch or old Planner use."""
    import sapien
    import mplib.planner
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
    from examples.baselines.hypericl_task_expansion.collect import array

    class UpstreamSolver(PandaArmMotionPlanningSolver):
        def setup_planner(self):
            planner = mplib.planner.Planner(
                urdf=self.env_agent.urdf_path,
                srdf=self.env_agent.urdf_path.replace(".urdf",".srdf"),
                user_link_names=[link.get_name() for link in self.robot.get_links()],
                user_joint_names=[joint.get_name() for joint in self.robot.get_active_joints()],
                move_group="panda_hand_tcp", joint_vel_limits=np.ones(7)*.8,
                joint_acc_limits=np.ones(7)*.8)
            planner.set_base_pose(np.r_[self.base_pose.p,self.base_pose.q])
            return planner

    agent = rec.unwrapped.agent.agents[arm]
    motion = UpstreamSolver(
        rec, base_pose=agent.robot.pose, multi_robot_id=arm, vis=False, debug=False,
        visualize_target_grasp_pose=False, print_env_info=False,
        joint_vel_limits=.8, joint_acc_limits=.8)
    return CheckedMotion(rec, arm, motion, directory, bind(mplib.planner.__file__))


class CheckedMotion:
    def __init__(self, rec, arm, motion, directory, planner_binding):
        from examples.baselines.hypericl_task_expansion.collect import array
        self.rec, self.arm, self.motion = rec, arm, motion
        self.planner = motion.planner
        self.agent = rec.unwrapped.agent.agents[arm]
        self.root = array(self.agent.robot.pose.to_transformation_matrix()).reshape(4,4)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True,exist_ok=False)
        self.counter, self.use_attach = 0, False
        self.objects = []
        self.normal_objects = []
        self.table_link_objects = []
        self.environment_collision_objects = []
        self.physical_robot_objects = []
        self.payload_collision = None
        self.payload_to_tcp = None
        self.route_obstacle_bounds = []
        save_new(self.directory/"implementation.json", {
            "planner": planner_binding, "guard": bind(__file__),
            "old_custom_planner_used": False,
            "dense_joint_interval_rad": .015,
            "per_configuration_collision_recomputed": True,
            "formal_continuous_collision_claimed": False,
            "contact_permissions": {
                "selected_apple": "physical mesh checked against robot links; only hand and two fingers allowed in manipulation phases",
                "apple_after_actual_grasp": "independent physical mesh versus all world objects, starting before initial lift",
                "initial_source_support": "lift first2mm only; XY<=3mm; contacts below floor+5mm and depth<=1mm; no other source-bowl exemption",
                "other_objects": "no planning collision exemptions",
            }})

    def install_world(self):
        """Use actual actor meshes; source/target bowls stay genuinely hollow."""
        import trimesh
        from mplib.pymp.collision_detection import fcl
        b = self.rec.unwrapped
        actors = [("source_bowl",b.bins[0]), ("target_bowl",b.bins[1])]
        actors += [(f"clutter_{i}", actor) for i,actor in enumerate(b.added_static_distractors)]
        actors += [("unselected_object",b.objects[1-self.arm]),
                   ("other_robot",b.agent.agents[1-self.arm].robot)]
        meshes, manifest = [], []
        for name, actor in actors:
            mesh = actor.get_first_collision_mesh(to_world_frame=True)
            if mesh is None or not len(mesh.faces):
                raise ValueError("Missing actual collision geometry: "+name)
            path = self.directory/(name+"_world.obj")
            mesh.export(path)
            geometry = fcl.load_mesh_as_BVH(str(path), np.ones(3))
            collision = fcl.CollisionObject(geometry)
            self.planner.set_normal_object(name, collision)
            self.objects.append((geometry,collision))
            self.normal_objects.append(name)
            self.environment_collision_objects.append((name,collision))
            # Point samples are in world coordinates, same as normal meshes.
            meshes.append(mesh)
            if name != "other_robot":
                self.route_obstacle_bounds.append(mesh.bounds.copy())
            manifest.append({"instance":name,"mesh":bind(path),"world_bounds":mesh.bounds.tolist()})
        cloud = sampled_mesh_points(trimesh.util.concatenate(meshes),
                                    12000, int(b.randomization["layout_seed"])+1030003)
        table = b.table_scene.table.get_first_collision_mesh(to_world_frame=True)
        table_path = self.directory/"table_complete_world.obj"
        table.export(table_path)
        table_geometry = fcl.load_mesh_as_BVH(str(table_path),np.ones(3))
        self.table_collision = fcl.CollisionObject(table_geometry)
        self.objects.append((table_geometry,self.table_collision))
        # mplib in this environment has no ACM. The official planner receives a
        # table cloud without the fixed mounting footprint. The guard checks the
        # COMPLETE actual table mesh against every physical link, exempting only
        # the exact panda_link0/table mounting pair.
        from mani_skill.utils.geometry.trimesh_utils import get_component_meshes,merge_meshes
        base_mesh = None
        for link in self.agent.robot.get_links():
            name = link.get_name()
            mesh = merge_meshes(get_component_meshes(link._objs[0]))
            if mesh is None:
                continue
            link_path = self.directory/(name+"_physical_local.obj")
            mesh.export(link_path)
            geometry = fcl.load_mesh_as_BVH(str(link_path),np.ones(3))
            collision = fcl.CollisionObject(geometry)
            self.objects.append((geometry,collision))
            self.physical_robot_objects.append((name,self.planner.link_name_2_idx[name],collision))
            if name == "panda_link0":
                base_mesh = mesh.copy()
                base_mesh.apply_transform(self.root)
            else:
                self.table_link_objects.append((name,self.planner.link_name_2_idx[name],collision))
        if base_mesh is None or not self.table_link_objects:
            raise RuntimeError("Missing physical robot meshes for exact table guard")
        table_cloud = sampled_mesh_points(table,24000,int(b.randomization["layout_seed"])+1030033)
        base_bounds = base_mesh.bounds
        in_mount = np.all((table_cloud[:,:2] >= base_bounds[0,:2]-.005) &
                          (table_cloud[:,:2] <= base_bounds[1,:2]+.005),axis=1)
        cloud = np.concatenate([cloud,table_cloud[~in_mount]])
        cloud_path = self.directory/"collision_points_world.npy"
        np.save(cloud_path,cloud)
        self.planner.update_point_cloud(cloud,radius=.001)
        self.motion.use_point_cloud = True
        selected = b.objects[self.arm].get_first_collision_mesh(to_world_frame=False)
        self.selected_mesh_path = self.directory/"selected_object_physical_local.obj"
        selected.export(self.selected_mesh_path)
        self.selected_vertices = selected.vertices.copy()
        geometry = fcl.load_mesh_as_BVH(str(self.selected_mesh_path),np.ones(3))
        self.selected_collision = fcl.CollisionObject(geometry)
        self.objects.append((geometry,self.selected_collision))
        self.support_floor_z=float(b.floor_top)
        save_new(self.directory/"collision_world.json",{
            "objects":manifest,"point_cloud":bind(cloud_path),"point_radius_m":.001,
            "table_mesh":bind(table_path),
            "table_scope":"complete physical table mesh checked independently against each physical robot link and attached object",
            "table_only_exempt_pair":["panda_link0","table"],
            "planner_table_cloud_mount_crop_xy_m":[(base_bounds[0,:2]-.005).tolist(),(base_bounds[1,:2]+.005).tolist()],
            "table_guard_has_no_geometric_hole":True,
            "table_checked_physical_links":[x[0] for x in self.table_link_objects],
            "selected_object_physical_mesh":bind(self.selected_mesh_path),
            "selected_object_allowed_robot_links":["panda_hand","panda_leftfinger","panda_rightfinger"],
            "held_object_world_check":"independent FCL BVH against every listed object plus complete table; does not rely on mplib attachment collision",
            "inactive_arm_scope":"exact mesh snapshot; arm must stay fixed during active-arm plan"})

    def attach_lifted_object(self, obj, *, phase="lift_clear"):
        from scipy.spatial.transform import Rotation
        from examples.baselines.hypericl_task_expansion.collect import array
        world_object = array(obj.pose.to_transformation_matrix()).reshape(4,4)
        world_tcp = array(self.agent.tcp.pose.to_transformation_matrix()).reshape(4,4)
        relative = np.linalg.inv(world_tcp)@world_object
        quaternion = Rotation.from_matrix(relative[:3,:3]).as_quat()
        pose7 = np.r_[relative[:3,3],quaternion[3],quaternion[:3]]
        collision = str(self.selected_mesh_path)
        self.planner.update_attached_mesh(collision,pose7)
        from mplib.pymp.collision_detection import fcl
        geometry=fcl.load_mesh_as_BVH(collision,np.ones(3))
        self.payload_collision=fcl.CollisionObject(geometry)
        self.payload_to_tcp=relative
        self.objects.append((geometry,self.payload_collision))
        self.use_attach = True
        save_new(self.directory/("attached_object_grasp.json" if phase=="grasp" else "attached_object.json"),{
            "collision_mesh":bind(collision),"object_to_tcp":relative.tolist(),
            "enabled_after_actual_grasp":True,
            "source_support_cleared":phase=="lift_clear",
            "native_grasp_still_checked":True})

    def detach(self):
        self.planner.planning_world.remove_attach()
        self.use_attach = False

    def independent_table_collisions(self, full, tcp):
        """Physical mesh guard independent of mplib attachment collision support."""
        from mplib.pymp.collision_detection import fcl
        model = self.planner.pinocchio_model
        model.compute_forward_kinematics(full)
        pairs = []
        selected_transform = tcp@self.payload_to_tcp if self.use_attach else self.selected_world_pose
        self.selected_collision.set_transformation(matrix_to_pose7(selected_transform))
        allowed_hand = self.check_stage in (
            "grasp_reach","lift","above_bowl","release_above_rim","retreat")
        for name,index,collision in self.physical_robot_objects:
            transform = self.root@pose7_to_matrix(model.get_link_pose(index))
            collision.set_transformation(matrix_to_pose7(transform))
            if name != "panda_link0" and fcl.collide(collision,self.table_collision,fcl.CollisionRequest()).is_collision():
                pairs.append({"link1":name,"link2":"complete_table","source":"physical_mesh_guard"})
            if not (allowed_hand and name in ("panda_hand","panda_leftfinger","panda_rightfinger")):
                if fcl.collide(collision,self.selected_collision,fcl.CollisionRequest()).is_collision():
                    pairs.append({"link1":name,"link2":"selected_object","source":"physical_mesh_guard"})
        if self.use_attach:
            self.payload_collision.set_transformation(matrix_to_pose7(tcp@self.payload_to_tcp))
            for name,obstacle in self.environment_collision_objects+[("complete_table",self.table_collision)]:
                request=fcl.CollisionRequest(num_max_contacts=10000,enable_contact=True)
                contact=fcl.collide(self.payload_collision,obstacle,request)
                if not contact.is_collision():
                    continue
                contacts=contact.get_contacts()
                delta=tcp[:3,3]-self.check_initial_tcp[:3,3]
                support_allowed=(
                    name=="source_bowl" and self.check_stage=="lift" and
                    -.0005<=delta[2]<=.002 and np.linalg.norm(delta[:2])<=.003 and
                    0<len(contacts)<10000 and
                    all(c.pos[2]<=self.support_floor_z+.005 and c.penetration_depth<=.001 for c in contacts))
                if support_allowed:
                    self.allowed_support_contacts.append({
                        "dense_index":self.check_dense_index,"pairs":["held_object","source_bowl"],
                        "contact_count":len(contacts),"max_penetration_m":max(float(c.penetration_depth) for c in contacts),
                        "max_contact_z_m":max(float(c.pos[2]) for c in contacts)})
                else:
                    pairs.append({"link1":"held_object","link2":name,"source":"physical_mesh_guard"})
        return pairs

    def check(self, path, current, goal, stage):
        q = np.asarray(path.get("position",[]))
        joint = validate_joint_path(q,self.planner.joint_limits,current)
        if not joint["passed"]:
            return {"passed":False,"joint":joint},None,None
        dense = densify_joints(q)
        poses = world_fk(dense,self.root)
        self.check_stage=stage
        self.check_initial_tcp=poses[0].copy()
        self.allowed_support_contacts=[]
        settings = {"workspace_min":[-.45,-.65,-.015],"workspace_max":[.30,.65,.75],
                    "desired_end_pose":goal, "endpoint_tolerance_m":.002,
                    "endpoint_tolerance_rad":.03}
        if stage in ("grasp_reach","release_above_rim"):
            settings.update(xy_center=goal[:2,3],xy_radius_m=.10,
                            maximum_z=max(poses[0,2,3],goal[2,3])+.025)
        if stage=="lift":
            settings.update(xy_center=poses[0,:2,3],xy_radius_m=.015,
                            maximum_z=goal[2,3]+.01)
        geometry = validate_geometry(poses,**settings)
        # validate_geometry also serves a retrospective-only CLI. Its old scope
        # flags do not describe this live wrapper.
        geometry.pop("collision_verified",None)
        geometry.pop("live_execution_guard_installed",None)
        geometry["scope"]="geometry_only"
        if stage=="lift" and np.min(np.diff(poses[:,2,3]),initial=0)<-.001:
            geometry["geometry_proxy_passed"]=False
            geometry["failures"]["lift_monotonic_vertical"]=True
        collisions = []
        world = self.planner.planning_world
        world.set_use_point_cloud(True)
        world.set_use_attach(self.use_attach)
        try:
            for i,arm_q in enumerate(dense):
                self.check_dense_index=i
                full = np.asarray(current).copy()
                full[:7] = arm_q
                self.planner.robot.set_qpos(full,True)
                hits = world.collide_full()
                pairs = [
                        {"link1":str(hit.link_name1),"link2":str(hit.link_name2)}
                        for hit in hits]
                pairs += self.independent_table_collisions(full,poses[i])
                if pairs:
                    collisions.append({"dense_index":i,"pairs":pairs})
                    # Keep first evidence. No need to spend time on a rejected path.
                    break
        finally:
            self.planner.robot.set_qpos(np.asarray(current),True)
        result = {
            "passed":joint["passed"] and geometry["geometry_proxy_passed"] and not collisions,
            "joint":joint,"geometry":geometry,"collision_failures":collisions,
            "dense_samples":len(dense),"collision_samples_checked":i+1,
            "all_dense_collision_samples_checked":not collisions,
            "live_execution_guard_installed":True,
            "collision_scope":"all commands plus dense joint interpolation; official world/self/payload plus complete physical table",
            "independent_held_world_mesh_check":True,
            "independent_selected_nonhand_link_check":True,
            "allowed_initial_support_contacts":self.allowed_support_contacts,
            "use_point_cloud":True,"attached_object_enabled":self.use_attach,
            "world_settings":{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in settings.items()},
        }
        return result,dense,poses

    def plan_pose(self, goal_pose, stage, *, joint_goal=None):
        from examples.baselines.hypericl_task_expansion.collect import array
        goal = np.asarray(goal_pose.to_transformation_matrix())
        current = array(self.agent.robot.qpos).reshape(9).copy()
        self.selected_world_pose=array(self.rec.unwrapped.objects[self.arm].pose.to_transformation_matrix()).reshape(4,4)
        modes = ["RRTConnect_joint"] if joint_goal is not None else ["Screw","RRTConnect"]
        for mode in modes:
            folder = self.directory/f"{self.counter:03d}_{stage}_{mode}"
            self.counter += 1
            folder.mkdir()
            save_new(folder/"request.json",{
                "stage":stage,"mode":mode,"world_goal":goal.tolist(),"start_qpos":current.tolist(),
                "world_root":self.root.tolist(),"use_point_cloud":True,"use_attach":self.use_attach,
                "source_frame":len(self.rec.rows)})
            try:
                self.planner.robot.set_qpos(current.copy(),True)
                if mode == "Screw":
                    result = self.planner.plan_screw(np.r_[goal_pose.p,goal_pose.q],
                        current.copy(),time_step=self.rec.control_timestep,
                        use_point_cloud=True,use_attach=self.use_attach,wrt_world=True)
                elif mode == "RRTConnect":
                    result = self.planner.plan_qpos_to_pose(np.r_[goal_pose.p,goal_pose.q],
                        current.copy(),time_step=self.rec.control_timestep,
                        use_point_cloud=True,use_attach=self.use_attach,wrt_world=True)
                else:
                    result = self.planner.plan_qpos_to_qpos([np.asarray(joint_goal).copy()],
                        current.copy(),time_step=self.rec.control_timestep,
                        use_point_cloud=True,use_attach=False)
                if result.get("status") != "Success":
                    save_new(folder/"decision.json",{"accepted":False,"planner_status":result.get("status")})
                    continue
                pathfile = folder/"planner_path.npz"
                np.savez_compressed(pathfile,**{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
                report,dense,poses = self.check(result,current,goal,stage)
                if dense is not None:
                    np.savez_compressed(folder/"guard_path.npz",joint_positions=dense,world_tcp=poses)
                save_new(folder/"decision.json",{
                    "accepted":report["passed"],"planner_status":result["status"],
                    "guard":report,"planner_path":bind(pathfile)})
                if report["passed"]:
                    return result
            except Exception as error:
                if not (folder/"decision.json").exists():
                    save_new(folder/"decision.json",{"accepted":False,"error":repr(error)})
        raise RuntimeError("All independently checked plans rejected: "+stage)


def solve(rec, program, directory):
    """Task-equivalent bowl transfer; only checked paths can reach follow_path."""
    import sapien
    from mani_skill.examples.motionplanning.panda.utils import get_actor_obb,compute_grasp_info_by_obb
    from examples.baselines.hypericl_task_expansion.collect import array,require
    b=rec.unwrapped
    index=0 if program.startswith("pick_first") else 1
    checked=make_solver(rec,index,directory)
    checked.install_world()
    planner=checked.motion
    agent=b.agent.agents[index]
    obj=b.objects[index]
    initial=array(agent.robot.qpos).reshape(9).copy()
    initial_tcp=array(agent.tcp.pose.to_transformation_matrix()).reshape(4,4).copy()
    other_gripper=1

    def move(point,q,stage):
        rec.stage=stage
        goal=sapien.Pose(point,q)
        path=checked.plan_pose(goal,stage)
        planner.follow_path(path,other_gripper_state=other_gripper,refine_steps=4)
        if checked.use_attach:
            require(bool(array(agent.is_grasping(obj)).item()),"Object grasp lost during "+stage)

    hint=array(agent.tcp.pose.to_transformation_matrix()).reshape(4,4)[:3,1]
    grasp=compute_grasp_info_by_obb(get_actor_obb(obj),approaching=[0,0,-1],
                                  target_closing=hint,depth=.008)
    require(grasp["extents"][1]<=.077,"Fruit exceeds grasp aperture")
    pose=agent.build_grasp_pose([0,0,-1],grasp["closing"],grasp["center"])
    move(pose.p+[0,0,.16],pose.q,"approach")
    move(pose.p,pose.q,"grasp_reach")
    rec.stage="close"
    planner.close_gripper(other_gripper_state=other_gripper,t=16)
    require(bool(array(agent.is_grasping(obj)).item()),"Fruit was not grasped after closing")
    checked.attach_lifted_object(obj,phase="grasp")
    move(pose.p+[0,0,.19],pose.q,"lift")
    rec.hold(6,(-1,1) if index==0 else (1,-1))
    require(bool(array(agent.is_grasping(obj)).item()),"Fruit was not actually grasped")
    require(obj.get_first_collision_mesh(to_world_frame=True).bounds[0,2] > b.wall_height+.01,
            "Held object has not cleared source rim before transport")
    checked.attach_lifted_object(obj)
    from scipy.spatial.transform import Rotation
    world_mesh=obj.get_first_collision_mesh(to_world_frame=True)
    actual_tcp=array(agent.tcp.pose.to_transformation_matrix()).reshape(4,4)
    inverse_tcp=np.linalg.inv(actual_tcp)
    vertices_tcp=world_mesh.vertices@inverse_tcp[:3,:3].T+inverse_tcp[:3,3]
    destination=np.asarray(b.goal_centers[b.destination_index,:2])
    held_radius=float(np.linalg.norm(np.ptp(world_mesh.vertices[:,:2],axis=0))/2)
    bottom=corridor_height(checked.route_obstacle_bounds,world_mesh.bounds.mean(0)[:2],
                           destination,held_radius+.015,float(b.wall_height)+.08)
    candidate_logs=[]
    accepted=None
    rec.stage="above_bowl"
    for yaw in (0.,np.pi/2,-np.pi/2,np.pi):
        rotation=Rotation.from_rotvec([0,0,yaw]).as_matrix()@actual_tcp[:3,:3]
        above=payload_goal(vertices_tcp,rotation,destination,bottom)
        quat_xyzw=Rotation.from_matrix(rotation).as_quat()
        quaternion=np.r_[quat_xyzw[3],quat_xyzw[:3]]
        target=sapien.Pose(above,quaternion)
        try:
            path=checked.plan_pose(target,"above_bowl")
            candidate_logs.append({"yaw_radians":yaw,"world_goal":target.to_transformation_matrix().tolist(),
                                   "accepted":True})
            accepted=(path,above,quaternion)
            break
        except RuntimeError as error:
            candidate_logs.append({"yaw_radians":yaw,"world_goal":target.to_transformation_matrix().tolist(),
                                   "accepted":False,"reason":str(error)})
    save_new(checked.directory/"transfer_goal_candidates.json",{
        "held_bottom_transport_m":bottom,"release_bottom_m":float(b.wall_height)+.04,
        "corridor_payload_radius_m":held_radius+.015,"candidates":candidate_logs,
        "thresholds_relaxed":False,"fixed_tcp_height_removed":True})
    require(accepted is not None,"No independently checked task-equivalent transfer pose")
    path,above,quaternion=accepted
    planner.follow_path(path,other_gripper_state=other_gripper,refine_steps=4)
    require(bool(array(agent.is_grasping(obj)).item()),"Object grasp lost during above_bowl")
    # Reobserve payload bounds after real execution rather than assume a rigid
    # command was tracked exactly. The actual contact outcome remains decisive.
    bounds=obj.get_first_collision_mesh(to_world_frame=True).bounds
    release=array(agent.tcp.pose.p).reshape(3).copy()
    release[:2]+=destination-bounds.mean(0)[:2]
    release[2]+=float(b.wall_height)+.04-bounds[0,2]
    move(release,quaternion,"release_above_rim")
    rec.stage="open"
    planner.open_gripper(other_gripper_state=other_gripper)
    checked.detach()
    move(above,quaternion,"retreat")
    rec.stage="return"
    from scipy.spatial.transform import Rotation
    quat=Rotation.from_matrix(initial_tcp[:3,:3]).as_quat()
    goal=sapien.Pose(initial_tcp[:3,3],np.r_[quat[3],quat[:3]])
    path=checked.plan_pose(goal,"return",joint_goal=initial)
    planner.follow_path(path,other_gripper_state=other_gripper,refine_steps=4)
    rec.stage="final_settle"
    rec.hold(40)
    for _ in range(20):
        if b.stable_goal_steps[index]>=10:
            break
        rec.hold(5)
    return [{"object_index":index,"container_index":b.destination_index,
             "actual_grasp_after_lift":True,"strict_full_sampled_paths":True,
             "attached_mesh_checked_after_lift":True,"planner_logs":str(directory),
             "formal_continuous_collision_claimed":False,"frame":len(rec.rows)}]
