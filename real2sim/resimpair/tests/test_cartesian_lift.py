import unittest
import numpy as np
from resimpair_pickplace.cartesian_lift import cartesian_path
from resimpair_pickplace.reachability import ArmSelector,INITIAL
from data_generation_methods.human_to_robot_pipeline.strict_motion import world_fk,validate_joint_path


class CartesianLiftTests(unittest.TestCase):
    def test_native_ik_chain_tracks_vertical_pose_before_collision_qualification(self):
        selector=ArmSelector([[-.6,.35,0.]])
        planner=selector.planners[0];root=selector.roots[0]
        goal=world_fk(INITIAL[None,:7],root)[0];goal[2,3]+=.06
        result=cartesian_path(planner,root,INITIAL,goal)
        joints=np.asarray(result['position']);poses=world_fk(joints,root)
        self.assertTrue(validate_joint_path(joints,planner.joint_limits,INITIAL)['passed'])
        self.assertLess(np.linalg.norm(poses[-1,:3,3]-goal[:3,3]),.002)
        self.assertLess(np.linalg.norm(poses[:,:2,3]-poses[0,:2,3],axis=1).max(),.002)
        self.assertTrue(result['requires_retained_full_path_guard'])

    def test_sideways_request_cannot_masquerade_as_vertical_lift(self):
        root=np.eye(4);goal=world_fk(INITIAL[None,:7],root)[0]
        goal[2,3]+=.10;goal[0,3]+=.05
        with self.assertRaisesRegex(ValueError,'vertical_lift'):
            cartesian_path(None,root,INITIAL,goal)


if __name__=='__main__':unittest.main()
