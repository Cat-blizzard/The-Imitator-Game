"""Mesh-cache controller using the existing strict planner's path mathematics.

No policy training. One active arm at a time, two native cameras, actual contacts.
Candidate search is planning-only; a failed physical grasp is a recorded attempt.
"""
from pathlib import Path
import numpy as np

from .common import read, resolve, write
from corl2026_workshop.experiments.autonomous_pipeline_v4.simulation import array, targets
from data_generation_methods.human_to_robot_pipeline.strict_motion import (
    CheckedMotion, make_solver, matrix_to_pose7, pose7_to_matrix, sampled_mesh_points,
    validate_joint_path, densify_joints, world_fk)
from data_generation_methods.human_to_robot_pipeline.plan_geometry_validator import validate_geometry


class MeshMotion(CheckedMotion):
    def install(self, object_index):
        import trimesh
        from mplib.pymp.collision_detection import fcl
        from mani_skill.utils.geometry.trimesh_utils import get_component_meshes, merge_meshes
        b = self.rec.unwrapped
        self.object_index = object_index
        self.selected_actor = b.objects[object_index]
        self.selected_world_pose = array(self.selected_actor.pose.to_transformation_matrix()).reshape(4, 4)
        actors = [(oid, o) for i, (oid, o) in enumerate(zip(b.object_ids, b.objects)) if i != object_index]
        actors += [(item["id"], actor) for item, actor in zip(b.scene_spec["supports"], b.support_actors)]
        actors += [("inactive_robot", b.agent.agents[1 - self.arm].robot)]
        manifest = []
        meshes = []
        def collider(mesh, name):
            path = self.directory / (name + ".obj")
            mesh.export(path)
            geometry = fcl.load_mesh_as_BVH(str(path), np.ones(3))
            obj = fcl.CollisionObject(geometry)
            self.objects.append((geometry, obj))
            return obj
        for name, actor in actors:
            mesh = actor.get_first_collision_mesh(to_world_frame=True)
            if mesh is None: raise ValueError("Missing physical collision mesh: " + name)
            obj = collider(mesh, name)
            self.planner.set_normal_object(name, obj)
            self.normal_objects.append(name)
            self.environment_collision_objects.append((name, obj))
            meshes.append(mesh)
            manifest.append({"id": name, "bounds": mesh.bounds.tolist()})
        table = b.table_scene.table.get_first_collision_mesh(to_world_frame=True)
        self.table_collision = collider(table, "complete_table")
        base = None
        for link in self.agent.robot.get_links():
            mesh = merge_meshes(get_component_meshes(link._objs[0]))
            if mesh is None: continue
            name = link.get_name()
            obj = collider(mesh, "robot_" + name)
            self.physical_robot_objects.append((name, self.planner.link_name_2_idx[name], obj))
            if name == "panda_link0":
                base = mesh.copy(); base.apply_transform(self.root)
        if base is None: raise ValueError("Missing exact robot mount mesh")
        cloud = sampled_mesh_points(table, 18000, 401)
        inside_mount = np.all((cloud[:, :2] >= base.bounds[0, :2] - .005) &
                              (cloud[:, :2] <= base.bounds[1, :2] + .005), axis=1)
        self.planner.update_point_cloud(cloud[~inside_mount], radius=.001)
        self.motion.use_point_cloud = True
        selected = self.selected_actor.get_first_collision_mesh(to_world_frame=False)
        self.selected_mesh_path = self.directory / "selected_physical.obj"
        selected.export(self.selected_mesh_path)
        self.selected_vertices = selected.vertices.copy()
        geometry = fcl.load_mesh_as_BVH(str(self.selected_mesh_path), np.ones(3))
        self.selected_collision = fcl.CollisionObject(geometry)
        self.objects.append((geometry, self.selected_collision))
        self.support_floor_z = b.scene_spec["table_top_z"]
        write(self.directory / "world.json", {"world": manifest, "table": "complete physical mesh",
            "payload_checker": "independent world-frame physical BVH; installed mplib 0.1.1 attached mesh ignores nonidentity base in collide_full",
            "table_robot_exemption": "panda_link0 mount only", "target_contact_exemption": "two fingertips during manipulation only",
            "inactive_arm": "fixed exact mesh; actual cross-arm contacts also independently checked",
            "source_support_exemption": "lift initial 2mm, penetration <=1mm, contact near tabletop; none elsewhere"})

    def independent_table_collisions(self, full, tcp):
        from mplib.pymp.collision_detection import fcl
        model = self.planner.pinocchio_model
        model.compute_forward_kinematics(full)
        selected = tcp @ self.payload_to_tcp if self.use_attach else self.selected_world_pose
        self.selected_collision.set_transformation(matrix_to_pose7(selected))
        # Open fingers must clear the target while reaching it. Contact is
        # permitted only after an actual close, or while withdrawing after release.
        allowed = self.check_stage in {"lift", "above_bowl", "release_above_rim", "retreat"}
        pairs = []
        for name, index, collision in self.physical_robot_objects:
            collision.set_transformation(matrix_to_pose7(self.root @ pose7_to_matrix(model.get_link_pose(index))))
            if name != "panda_link0" and fcl.collide(collision, self.table_collision, fcl.CollisionRequest()).is_collision():
                pairs.append({"link1": name, "link2": "complete_table"})
            if not (allowed and name in {"panda_leftfinger", "panda_rightfinger"}):
                if fcl.collide(collision, self.selected_collision, fcl.CollisionRequest()).is_collision():
                    pairs.append({"link1": name, "link2": "selected_object"})
            # Native physical robot meshes are also checked against all exact
            # obstacles; the planner's URDF collision approximations are not the
            # only authority for clutter or the inactive arm.
            for oid, obstacle in self.environment_collision_objects:
                if fcl.collide(collision, obstacle, fcl.CollisionRequest()).is_collision():
                    pairs.append({"link1": name, "link2": oid})
        if self.use_attach:
            self.payload_collision.set_transformation(matrix_to_pose7(selected))
            for name, obstacle in self.environment_collision_objects + [("complete_table", self.table_collision)]:
                result = fcl.collide(self.payload_collision, obstacle, fcl.CollisionRequest(num_max_contacts=10000, enable_contact=True))
                if not result.is_collision(): continue
                contacts = result.get_contacts()
                delta = tcp[:3, 3] - self.check_initial_tcp[:3, 3]
                permission = (name == "complete_table" and self.check_stage == "lift" and
                    -.0005 <= delta[2] <= .002 and np.linalg.norm(delta[:2]) <= .003 and 0 < len(contacts) < 10000 and
                    all(c.pos[2] <= self.support_floor_z + .005 and c.penetration_depth <= .001 for c in contacts))
                if permission:
                    self.allowed_support_contacts.append({"dense_index": self.check_dense_index,
                        "contact_count": len(contacts), "max_penetration_m": max(float(c.penetration_depth) for c in contacts)})
                else:
                    pairs.append({"link1": "held_object", "link2": name})
        return pairs

    def check(self, path, current, goal, stage):
        """Reuse strict path limits; avoid a reproduced mplib attachment bug.

        Native mplib attached geometry is evaluated in base coordinates against
        world obstacles for translated robot bases (0.6m error reproduced in
        smoke_r2). The independent exact payload BVH is transformed by world TCP
        at EVERY dense sample and is authoritative for carried-object collision.
        Robot/self checks remain enabled; no physical acceptance threshold changes.
        """
        q = np.asarray(path.get("position", []))
        joint = validate_joint_path(q, self.planner.joint_limits, current)
        if not joint["passed"]: return {"passed": False, "joint": joint}, None, None
        dense = densify_joints(q); poses = world_fk(dense, self.root)
        self.check_stage = stage; self.check_initial_tcp = poses[0].copy(); self.allowed_support_contacts = []
        settings = {"workspace_min": [-.45, -.65, -.015], "workspace_max": [.30, .65, .75],
                    "desired_end_pose": goal, "endpoint_tolerance_m": .002, "endpoint_tolerance_rad": .03}
        if stage in {"grasp_reach", "release_above_rim"}:
            settings.update(xy_center=goal[:2, 3], xy_radius_m=.10, maximum_z=max(poses[0, 2, 3], goal[2, 3]) + .025)
        if stage == "lift": settings.update(xy_center=poses[0, :2, 3], xy_radius_m=.015, maximum_z=goal[2, 3] + .01)
        geometry = validate_geometry(poses, **settings)
        if stage == "lift" and np.min(np.diff(poses[:, 2, 3]), initial=0) < -.001:
            geometry["geometry_proxy_passed"] = False; geometry["failures"]["lift_monotonic_vertical"] = True
        collisions = []; world = self.planner.planning_world
        world.set_use_point_cloud(True); world.set_use_attach(False)
        try:
            for i, arm_q in enumerate(dense):
                self.check_dense_index = i
                full = current.copy(); full[:7] = arm_q
                self.planner.robot.set_qpos(full, True)
                pairs = [{"link1": str(h.link_name1), "link2": str(h.link_name2)} for h in world.collide_full()]
                pairs += self.independent_table_collisions(full, poses[i])
                if pairs: collisions.append({"dense_index": i, "pairs": pairs}); break
        finally:
            self.planner.robot.set_qpos(current.copy(), True)
        report = {"passed": joint["passed"] and geometry["geometry_proxy_passed"] and not collisions,
            "joint": joint, "geometry": geometry, "collision_failures": collisions,
            "dense_samples": len(dense), "collision_samples_checked": i + 1, "all_dense_collision_samples_checked": not collisions,
            "native_mplib_attach_enabled": False, "independent_physical_payload_enabled": self.use_attach,
            "independent_full_table_and_obstacles_enabled": True, "allowed_support_contacts": self.allowed_support_contacts}
        return report, dense, poses

    def plan(self, pose, stage, current=None):
        current = array(self.agent.robot.qpos).reshape(9).copy() if current is None else current.copy()
        goal = np.asarray(pose.to_transformation_matrix())
        self.selected_world_pose = array(self.selected_actor.pose.to_transformation_matrix()).reshape(4, 4)
        for mode in ["screw", "rrt"]:
            directory = self.directory / f"{self.counter:03d}_{stage}_{mode}"
            self.counter += 1; directory.mkdir()
            write(directory / "request.json", {"stage": stage, "current": current.tolist(), "goal": goal.tolist(),
                  "frame": len(self.rec.rows), "attached": self.use_attach})
            try:
                options = dict(time_step=self.rec.control_timestep, use_point_cloud=True, use_attach=False, wrt_world=True)
                with self.rec.performance.section("motion_plan_" + mode):
                    if mode == "screw":
                        path = self.planner.plan_screw(np.r_[pose.p, pose.q], current.copy(),
                            qpos_step=.01 if stage == "lift" else .025, **options)
                    else:
                        path = self.planner.plan_qpos_to_pose(np.r_[pose.p, pose.q], current.copy(), **options)
                if path.get("status") != "Success":
                    write(directory / "decision.json", {"accepted": False, "planner_status": path.get("status")}); continue
                with self.rec.performance.section("dense_physical_collision_guard"):
                    report, dense, poses = self.check(path, current, goal, stage)
                np.savez_compressed(directory / "path.npz", positions=path["position"], current=current, goal=goal)
                if dense is not None:
                    np.savez_compressed(directory / "dense.npz", joints=dense, tcp=poses)
                write(directory / "decision.json", {"accepted": report["passed"], "report": report})
                if report["passed"]:
                    path["guard_evidence"] = str(directory / "decision.json")
                    return path
            except Exception as e:
                if not (directory / "decision.json").exists():
                    write(directory / "decision.json", {"accepted": False, "error": repr(e)})
        raise RuntimeError("independent_guard_rejected:" + stage)


def solve(rec, task, method):
    import sapien
    from scipy.spatial.transform import Rotation
    from mani_skill.examples.motionplanning.panda.utils import get_actor_obb, compute_grasp_info_by_obb
    from .grasps import load
    destinations = targets(rec.scene_spec, task)
    for op in task["operations"]:
        index = rec.object_ids.index(op["object"]); arm = op["arm"]
        obj = rec.objects[index]; agent = rec.agent.agents[arm]
        base = make_solver(rec, arm, rec.directory / (op["id"] + "_motion"))
        base.__class__ = MeshMotion
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
        rec.hold(8, (-1, 1) if arm == 0 else (1, -1))
        if not bool(array(agent.is_grasping(obj)).item()): raise RuntimeError("lost_grasp_after_lift")
        desired = np.eye(4); desired[:3, :3] = Rotation.from_euler("z", op["target"]["yaw"]).as_matrix()
        desired[:3, 3] = np.asarray(destinations[op["object"]]) + [0, 0, .002]
        actual = array(obj.pose.to_transformation_matrix()).reshape(4, 4)
        tcp = array(agent.tcp.pose.to_transformation_matrix()).reshape(4, 4)
        place = sapien.Pose(desired @ np.linalg.inv(actual) @ tcp)
        move(sapien.Pose(place.p + [0, 0, .18], place.q), "above_bowl")
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
        rec.stage = op["id"] + ":open"; motion.open_gripper(other_gripper_state=1)
        guard.detach()
        move(sapien.Pose(place.p + [0, 0, .18], place.q), "retreat")
        move(sapien.Pose(initial_tcp), "return")
        rec.hold(20)
    rec.stage = "final_settle"; rec.hold(40)
