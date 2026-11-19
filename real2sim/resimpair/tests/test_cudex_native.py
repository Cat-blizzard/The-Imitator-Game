import unittest
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from resimpair_grasp.common import ROOT
from resimpair_grasp.cudex_native import hand_to_tcp
from resimpair_grasp.cartesian_clearance import swept_vertices, VerticalClearance


class CuDexNativeTests(unittest.TestCase):
    def test_root_to_tcp_matches_native_fixed_joint_without_snapping(self):
        robot=ET.parse(ROOT/'PolicyGen/mani_skill/assets/robots/panda/panda_v3.urdf').getroot()
        joint=next(j for j in robot.findall('joint') if j.find('child').get('link')=='panda_hand_tcp')
        xyz=np.fromstring(joint.find('origin').get('xyz'),sep=' ')
        r=Rotation.from_euler('xyz',[.2,2.7,-.4])
        q=np.r_[[.1,-.3,.2],r.as_quat(scalar_first=True),.03,.03]
        T=hand_to_tcp(q)
        np.testing.assert_allclose(T[:3,3],q[:3]+r.apply(xyz),atol=1e-12)
        np.testing.assert_allclose(T[:3,:3],r.as_matrix(),atol=1e-12)
        self.assertNotAlmostEqual(T[2,2],-1.)

    def test_invalid_optimizer_pose_is_not_repaired_silently(self):
        for q in [np.zeros(9),np.full(9,np.nan),np.ones(8)]:
            with self.assertRaises(ValueError):hand_to_tcp(q)

    def test_tilted_grasp_sweep_matches_controller_world_z(self):
        T=np.eye(4);T[:3,:3]=Rotation.from_euler('y',1.2).as_matrix();T[:3,3]=[.1,.2,.3]
        vertices=trimesh.creation.box(extents=[.01,.02,.03]).vertices
        sweep=swept_vertices(vertices,T,.17)
        np.testing.assert_allclose(sweep[len(vertices):]-sweep[:len(vertices)],np.tile([0,0,.17],(len(vertices),1)),atol=1e-12)

    def test_world_z_guard_detects_collision_in_vertical_approach(self):
        # A box high above the final wrist is hit during descent, even though
        # the final hand is clear. This catches reverting to endpoint-only QA.
        T=np.eye(4);T[:3,:3]=Rotation.from_euler('x',np.pi).as_matrix();T[:3,3]=[0,0,.10]
        obstacle=trimesh.creation.box(extents=[.12,.25,.025]);obstacle.apply_translation([0,0,.29])
        short=VerticalClearance(obstacle,approach=0)
        full=VerticalClearance(obstacle,approach=.17)
        self.assertTrue(short.check(T,-1,.002)[0])
        self.assertFalse(full.check(T,-1,.002)[0])


if __name__=='__main__':unittest.main()
