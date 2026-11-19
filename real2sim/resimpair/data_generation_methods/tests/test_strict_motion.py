"""Failure-oriented tests for the execution guard; no GPU or native rollout."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import tempfile
from pathlib import Path

from data_generation_methods.human_to_robot_pipeline.strict_motion import (
    CheckedMotion,densify_joints,validate_joint_path,payload_goal,corridor_height)


class FakeWorld:
    def __init__(self, collision_at=None):
        self.collision_at=collision_at
        self.q=None
    def set_use_point_cloud(self,value): self.cloud=value
    def set_use_attach(self,value): self.attached=value
    def collide_full(self):
        if self.collision_at and self.collision_at(self.q):
            return [SimpleNamespace(link_name1="panda_link6",link_name2="obstacle")]
        return []


def checker(collision_at=None,attached=False,table_check=None):
    result=object.__new__(CheckedMotion)
    world=FakeWorld(collision_at)
    def set_qpos(q,*args): world.q=np.asarray(q)
    result.planner=SimpleNamespace(
        planning_world=world,robot=SimpleNamespace(set_qpos=set_qpos),
        joint_limits=np.tile([-3.,3.],(9,1)))
    result.root=np.eye(4)
    result.use_attach=attached
    result.independent_table_collisions=table_check or (lambda q,tcp: [])
    return result


def poses(q,root):
    result=np.repeat(np.eye(4)[None],len(q),axis=0)
    result[:,2,3]=.2
    return result


class StrictGuardTest(unittest.TestCase):
    def test_intermediate_collision_safe_endpoints(self):
        guard=checker(lambda q:.01 < q[0] < .03)
        path=np.zeros((2,7));path[1,0]=.04
        with patch("data_generation_methods.human_to_robot_pipeline.strict_motion.world_fk",poses):
            result,_,_=guard.check({"position":path},np.zeros(9),poses(path,None)[-1],"approach")
        self.assertFalse(result["passed"])
        self.assertEqual(result["collision_failures"][0]["dense_index"],1)
        self.assertTrue(result["live_execution_guard_installed"])

    def test_fake_success_wrong_endpoint(self):
        guard=checker()
        goal=np.eye(4);goal[2,3]=.2;goal[0,3]=.03
        with patch("data_generation_methods.human_to_robot_pipeline.strict_motion.world_fk",poses):
            result,_,_=guard.check({"position":np.zeros((2,7))},np.zeros(9),goal,"approach")
        self.assertFalse(result["passed"])
        self.assertFalse(result["geometry"]["geometry_proxy_passed"])
        self.assertEqual(result["geometry"]["scope"],"geometry_only")
        self.assertNotIn("collision_verified",result["geometry"])

    def test_local_descent_detour(self):
        guard=checker()
        def bad(q,root):
            value=poses(q,root);value[1,0,3]=.2;value[1,2,3]=.4
            return value
        path=np.zeros((3,7));path[:,0]=[0,.01,.02]
        with patch("data_generation_methods.human_to_robot_pipeline.strict_motion.world_fk",bad):
            result,_,_=guard.check({"position":path},np.zeros(9),poses(path,None)[-1],"release_above_rim")
        self.assertFalse(result["passed"])
        self.assertFalse(result["geometry"]["geometry_proxy_passed"])

    def test_payload_hits_table_robot_clear(self):
        guard=checker(attached=True,table_check=lambda q,tcp:[
            {"link1":"held_object","link2":"complete_table","source":"physical_mesh_guard"}])
        path=np.zeros((2,7))
        with patch("data_generation_methods.human_to_robot_pipeline.strict_motion.world_fk",poses):
            result,_,_=guard.check({"position":path},np.zeros(9),poses(path,None)[-1],"above_bowl")
        self.assertFalse(result["passed"])
        self.assertTrue(guard.planner.planning_world.attached)
        self.assertEqual(result["collision_failures"][0]["pairs"][0]["link1"],"held_object")

    def test_valid_path_checked_at_all_dense_samples(self):
        guard=checker()
        path=np.zeros((2,7));path[1,0]=.04
        with patch("data_generation_methods.human_to_robot_pipeline.strict_motion.world_fk",poses):
            result,dense,_=guard.check({"position":path},np.zeros(9),poses(path,None)[-1],"approach")
        self.assertTrue(result["passed"])
        self.assertGreater(len(dense),len(path))
        self.assertEqual(result["collision_samples_checked"],len(dense))

    def test_command_and_limit_rejection(self):
        path=np.zeros((2,7));path[1,0]=.2
        result=validate_joint_path(path,np.tile([-.1,.1],(7,1)),np.zeros(9))
        self.assertIn("joint_limit",result["failures"])
        self.assertIn("joint_command_step",result["failures"])

    def test_payload_goal_respects_rotated_geometry(self):
        vertices=np.array([[-.02,-.03,-.07],[.02,.03,-.01]])
        point=payload_goal(vertices,np.eye(3),[.1,.2],.12)
        world=vertices+point
        np.testing.assert_allclose(world.min(0)[2],.12)
        np.testing.assert_allclose((world.min(0)+world.max(0))[:2]/2,[.1,.2])
        # A tall obstacle off the corridor cannot force an unreachable height.
        height=corridor_height([np.array([[.5,.5,0],[.6,.6,.9]])],
                               [0,0],[.1,.1],.05,.16)
        self.assertAlmostEqual(height,.16)


class ActualFCLGuardTest(unittest.TestCase):
    """Exercise the actual installed BVH API, not a collision mock."""
    def fixture(self,directory):
        import trimesh
        from mplib.pymp.collision_detection import fcl
        path=Path(directory)/"selected.obj"
        trimesh.creation.box([.1,.1,.1]).export(path)
        geom=fcl.load_mesh_as_BVH(str(path),np.ones(3))
        guard=object.__new__(CheckedMotion)
        guard.root=np.eye(4)
        guard.use_attach=True
        guard.payload_to_tcp=np.eye(4)
        guard.selected_world_pose=np.eye(4);guard.selected_world_pose[2,3]=.3
        guard.selected_collision=fcl.CollisionObject(geom)
        guard.payload_collision=fcl.CollisionObject(geom)
        table=fcl.Box(.5,.5,.1);guard.table_collision=fcl.CollisionObject(table)
        guard.table_collision.set_transformation(np.array([0.,0.,-.5,1.,0.,0.,0.]))
        guard.physical_robot_objects=[]
        guard.environment_collision_objects=[]
        guard.check_stage="above_bowl"
        guard.check_initial_tcp=guard.selected_world_pose.copy()
        guard.check_dense_index=0
        guard.allowed_support_contacts=[]
        guard.support_floor_z=.01
        guard.planner=SimpleNamespace(pinocchio_model=SimpleNamespace(
            compute_forward_kinematics=lambda q:None,
            get_link_pose=lambda index:np.array([0.,0.,.3,1.,0.,0.,0.])))
        return guard,[geom,table]

    def test_actual_6mm_obstacle_intersects_payload_robot_clear(self):
        from mplib.pymp.collision_detection import fcl
        with tempfile.TemporaryDirectory() as directory:
            guard,keep=self.fixture(directory)
            obstacle_geom=fcl.Box(.006,.006,.006)
            obstacle=fcl.CollisionObject(obstacle_geom)
            obstacle.set_transformation(np.array([.05,0,.3,1.,0,0,0]))
            guard.environment_collision_objects=[("six_mm_obstacle",obstacle)]
            guard.use_attach=False
            without=guard.independent_table_collisions(np.zeros(9),guard.selected_world_pose)
            guard.use_attach=True
            with_payload=guard.independent_table_collisions(np.zeros(9),guard.selected_world_pose)
            self.assertEqual(without,[])
            self.assertIn("six_mm_obstacle",[x["link2"] for x in with_payload])

    def test_selected_object_disallows_nonhand_contact(self):
        from mplib.pymp.collision_detection import fcl
        with tempfile.TemporaryDirectory() as directory:
            guard,keep=self.fixture(directory)
            geometry=fcl.Box(.12,.06,.06);link=fcl.CollisionObject(geometry)
            guard.use_attach=False
            guard.check_stage="grasp_reach"
            guard.physical_robot_objects=[("panda_link6",0,link)]
            rejected=guard.independent_table_collisions(np.zeros(9),guard.selected_world_pose)
            self.assertIn("selected_object",[x["link2"] for x in rejected])
            guard.physical_robot_objects=[("panda_hand",0,link)]
            self.assertEqual(guard.independent_table_collisions(np.zeros(9),guard.selected_world_pose),[])
            guard.check_stage="return"
            self.assertTrue(guard.independent_table_collisions(np.zeros(9),guard.selected_world_pose))

    def test_only_named_base_table_pair_is_exempt(self):
        from mplib.pymp.collision_detection import fcl
        with tempfile.TemporaryDirectory() as directory:
            guard,keep=self.fixture(directory)
            geometry=fcl.Box(.06,.06,.12);link=fcl.CollisionObject(geometry)
            guard.use_attach=False
            guard.planner.pinocchio_model.get_link_pose=lambda index:np.array([0.,0.,-.5,1.,0.,0.,0.])
            guard.physical_robot_objects=[("panda_link0",0,link)]
            self.assertEqual(guard.independent_table_collisions(np.zeros(9),guard.selected_world_pose),[])
            guard.physical_robot_objects=[("panda_link1",0,link)]
            pairs=guard.independent_table_collisions(np.zeros(9),guard.selected_world_pose)
            self.assertIn("complete_table",[x["link2"] for x in pairs])


if __name__=="__main__":
    unittest.main()
