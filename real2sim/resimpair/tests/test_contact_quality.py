import unittest
import tempfile
from pathlib import Path
import numpy as np
from resimpair_pickplace.contact_quality import load_capacity, dfc_residual, assess, VERSION
from resimpair_pickplace.quality_reachability import selection_key


class ContactQualityTests(unittest.TestCase):
    def setUp(self):
        self.points = np.array([[0., -.03, 0.], [0., .03, 0.]])
        self.normals = np.array([[0., -1., 0.], [0., 1., 0.]])

    def test_symmetric_point_contacts_obey_friction_and_force_cap(self):
        q = load_capacity(self.points, self.normals, [0,0,0], [0,0,1], .25)
        self.assertTrue(q['valid'])
        self.assertAlmostEqual(q['capacity'], .5, places=6)
        lower = load_capacity(self.points, self.normals, [0,0,0], [0,0,1], .15)
        self.assertLess(lower['capacity'], q['capacity'])

    def test_dfc_cannot_distinguish_load_about_offset_center(self):
        centered = dfc_residual(self.points, self.normals, np.zeros(3), .1)
        offset = dfc_residual(self.points, self.normals, np.array([.01,0,0]), .1)
        self.assertAlmostEqual(centered, offset)
        q = load_capacity(self.points, self.normals, [.01,0,0], [0,0,1], .25)
        self.assertLess(q['capacity'], 1e-7)

    def test_contact_patch_is_explicit_assumption_not_free_torque(self):
        q = load_capacity(self.points, self.normals, [.01,0,0], [0,0,1], .25, .004)
        self.assertGreater(q['capacity'], 0.)
        self.assertLess(q['capacity'], .5)
        zero = load_capacity(self.points, self.normals, [.01,0,0], [0,0,1], .25, 0.)
        self.assertLess(zero['capacity'], q['capacity'])

    def test_invariant_to_contact_order_and_world_translation(self):
        a = load_capacity(self.points, self.normals, [0,0,0], [.2,0,1], .25, .004)
        delta = np.array([.3,-.6,.2])
        b = load_capacity(self.points[::-1]+delta, self.normals[::-1], delta, [.2,0,1], .25, .004)
        self.assertAlmostEqual(a['capacity'], b['capacity'], places=6)

    def test_degenerate_or_nonfinite_inputs_rejected(self):
        with self.assertRaises(ValueError):
            load_capacity(self.points, self.normals*0, [0,0,0], [0,0,1], .25)
        with self.assertRaises(ValueError):
            load_capacity(self.points, self.normals, [0,0,np.nan], [0,0,1], .25)
        with self.assertRaises(ValueError):
            load_capacity(self.points, self.normals, [0,0,0], [0,0,1], -.1)

    def test_quality_is_opt_in_and_cannot_be_silently_ignored(self):
        strong = {'id':'strong', 'task_load_quality': {'version':VERSION,
                  'model_valid':True, 'worst_load_capacity':.3, 'dfc_residual':.01}}
        weak = {'id':'weak', 'task_load_quality': {'version':VERSION,
                  'model_valid':True, 'worst_load_capacity':.1, 'dfc_residual':.01}}
        self.assertLess(selection_key(weak, 1., None), selection_key(strong, 3., None))
        policy='task_load_proxy_then_joint_travel_v1'
        self.assertLess(selection_key(strong, 3., policy), selection_key(weak, 1., policy))
        with self.assertRaises(ValueError): selection_key({'id':'unknown'}, 1., policy)
        with self.assertRaises(ValueError): selection_key(strong, 1., 'typo')

    def test_recovery_records_zero_pose_and_unknown_without_submission(self):
        from resimpair_grasp.common import binding,write
        from resimpair_pickplace.contact_refinement import recover_index
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);write(root/'descriptor.json',{'source_id':'source_a'})
            entry={'asset':binding(root/'descriptor.json'),'split':'heldout'}
            write(root/'request_00.json',{'entry':entry})
            report=recover_index(root)
            self.assertEqual(len(report['pending']),1)
            self.assertEqual(len(report['failures']),0)
            write(root/'terminal_00.json',{'returncode':0,'report':None})
            report=recover_index(root)
            self.assertEqual(len(report['pending']),0)
            self.assertEqual(report['failures'][0]['reason'],'no_qualified_pose_under_declared_model')
            self.assertEqual(report['entries'],[])


if __name__ == '__main__': unittest.main()
