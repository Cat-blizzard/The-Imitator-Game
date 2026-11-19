import copy
import unittest
import numpy as np
import trimesh
from resimpair_pickplace.stability import aperture_command, opening_width, window_stability
from resimpair_pickplace.closure_quality import ClosingSweep
from resimpair_pickplace.support_surface import check_support


class GraspReleaseTests(unittest.TestCase):
    def test_backfacing_floor_ray_hit_does_not_qualify_support(self):
        floor=trimesh.creation.box([.2,.2,.01])
        cavity={'center_xy_m':[0,0],'flat_floor_radius_m':.08,'floor_z_m':.005}
        self.assertTrue(check_support(floor,cavity)['passed'])
        inverted=trimesh.Trimesh(vertices=floor.vertices,faces=floor.faces[:,::-1],process=False)
        bad=check_support(inverted,cavity)
        self.assertEqual(bad['hit_count'],bad['sample_count'])
        self.assertFalse(bad['passed'])

    def test_native_mimic_negative_lower_bound_is_respected(self):
        # ManiSkill Panda has [-.01,.04] per finger, not [0,.04].
        cmd = aperture_command(.052, -.01, .04)
        physical_joint_target = -.01 + (cmd + 1) / 2 * .05
        self.assertAlmostEqual(2 * physical_joint_target, .052)
        self.assertNotAlmostEqual(cmd, .052/.04-1)

    def test_no_opening_reserve_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'reserve'): opening_width(.076)
        with self.assertRaises(ValueError): aperture_command(.09, -.01, .04)

    def test_native_finger_union_detects_blocked_inner_contact_pair(self):
        mesh = trimesh.creation.box([.035, .06, .04])
        sweep = ClosingSweep(mesh)
        T = np.eye(4)
        self.assertEqual(len(sweep.fingers), 2)
        self.assertFalse(sweep.measure(T, .019)['contact_pair_reachable'])
        self.assertTrue(sweep.measure(T, .06)['contact_pair_reachable'])

    def test_window_rejects_slip_despite_two_finger_contact(self):
        sample = {'object_states': np.array([[0,0,.1,1,0,0,0,0,0,0,0,0,0]]),
                  'tcp_world': np.array([np.eye(4),np.eye(4)]),
                  'grasp': np.ones((2,1),dtype=bool)}
        stable = [copy.deepcopy(sample) for _ in range(6)]
        self.assertTrue(window_stability(stable,0,0)['stable'])
        sliding = copy.deepcopy(stable)
        for i,s in enumerate(sliding): s['object_states'][0,0] = i*.001
        self.assertFalse(window_stability(sliding,0,0)['stable'])

    def test_window_rejects_common_motion_even_without_attachment_slip(self):
        rows=[]
        for i in range(6):
            tcp=np.eye(4);tcp[0,3]=.002*i
            rows.append({'object_states':np.array([[.002*i,0,.1,1,0,0,0,0,0,0,0,0,0]]),
                         'tcp_world':np.array([tcp,np.eye(4)]),'grasp':np.ones((2,1),bool)})
        r=window_stability(rows,0,0)
        self.assertLess(r['relative_translation_range_m'],1e-10)
        self.assertFalse(r['stable'])


if __name__ == '__main__': unittest.main()
