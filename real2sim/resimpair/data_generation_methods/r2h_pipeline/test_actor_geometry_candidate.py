"""CPU tests of mathematical scope and negative preflight behavior."""
from copy import deepcopy
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

from data_generation_methods.r2h_pipeline import actor_geometry_candidate as m
from data_generation_methods.r2h_pipeline.common import read_json, write_json

BASE = Path('/mnt/sdg/datasets/real_sim_human_pipeline')
PACKAGE = BASE / 'harness_20261021_v2/unified_anchor/development/package.json'
WORLD = BASE / 'harness_20261021_v1/multiview/development/plans/collision_world.json'
PROTOCOL = BASE / 'harness_20261021_v3/independent_review/actor_geometry_protocol.json'


class GeometryMathTests(unittest.TestCase):
    def rig(self, lean=0):
        return {'schema': m.RIG_SCHEMA, 'frame': 'world_m_z_up',
                'torso': {'waist_world_m': [-.87, 0, 0], 'facing_world_xy': [1, 0],
                          'lean_degrees': lean, 'dimensions_depth_width_height_m': [.24, .36, .6],
                          'shoulder_height_from_waist_m': .54},
                'arms': {'upper_arm_m': .30, 'forearm_m': .26, 'shoulder_width_m': .36}}

    def test_both_shoulders_same_body_and_anatomical_left(self):
        g = m.rig_geometry(self.rig())
        np.testing.assert_allclose(g['shoulders']['left'], [-.87, .18, .54])
        np.testing.assert_allclose(g['shoulders']['right'], [-.87, -.18, .54])

    def test_lean_rotates_one_body_and_both_shoulders(self):
        g = m.rig_geometry(self.rig(30))
        np.testing.assert_allclose((g['shoulders']['left']+g['shoulders']['right'])/2,
                                  [-.87+.27, 0, .54*np.cos(np.pi/6)])
        self.assertAlmostEqual(np.linalg.norm(g['shoulders']['left']-g['shoulders']['right']), .36)

    def test_refuses_length_extension(self):
        rig = self.rig(); rig['arms']['forearm_m'] = .80
        with self.assertRaises(ValueError): m.rig_geometry(rig)

    def test_refuses_unfrozen_angle(self):
        with self.assertRaises(ValueError): m.rig_geometry(self.rig(35))

    def test_too_near_and_far_are_unreachable(self):
        for point in ([0, 0, .02], [0, 0, .57]):
            self.assertEqual(m.reach_chain([0, 0, 0], point)['status'], 'point_unreachable')

    def test_chain_lengths_and_scope(self):
        r = m.reach_chain([0, 0, 0], [.40, .1, .1])
        chain = np.asarray(r['chain_world_m'])
        np.testing.assert_allclose(np.linalg.norm(np.diff(chain, axis=0), axis=1), [.30, .26])
        self.assertFalse(r['joint_or_collision_feasibility_verified'])

    def test_solid_angle_containment(self):
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        self.assertEqual(m._contains_points(mesh, [[0, 0, 0], [2, 0, 0]]).tolist(), [True, False])

    def test_actual_surface_collision_is_not_aabb_only(self):
        table = trimesh.creation.box(extents=[2, 2, .1])
        torso = trimesh.creation.box(extents=[.2, .2, 1])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'table.obj'; table.export(path)
            r = m.torso_table_check(torso, table, path)
        self.assertTrue(r['surface_intersection']); self.assertTrue(r['mesh_collision'])
        self.assertEqual(r['arm_hand_collision_status'], 'unknown')

    def test_projection_clips_behind_camera(self):
        camera = {'camera_from_world': np.c_[np.eye(3), np.zeros(3)].tolist(),
                  'intrinsic_matrix': np.eye(3).tolist(), 'near': .01, 'far': 10,
                  'width': 1, 'height': 1}
        self.assertFalse(m.project([0, 0, -1], camera)['in_frame'])


class RealDevelopmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = m.unified.validate_package(PACKAGE)
        cls.world, cls.table, _ = m._load_world(cls.data, WORLD)
        cls.rig = m.nominal_rig(cls.data, cls.table)

    def compute(self, rig=None, data=None, protocol=PROTOCOL):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'rig.json'; write_json(path, rig or self.rig)
            return m._compute(data or self.data, PACKAGE, path, WORLD, protocol)

    def test_real_far_upright_reject_and_full_held_path(self):
        r = self.compute()
        self.assertEqual(r['status'], 'reject')
        self.assertGreater(r['held_frame_count'], 100)
        self.assertEqual(len(r['held_trajectory_targets']), r['held_frame_count'])
        self.assertEqual(r['unknown_events'][0]['event_id'], 'withdraw_hand')
        self.assertTrue(any(x['all_surface_points_beyond_maximum_reach'] for x in r['event_targets']))
        for key in ('clears_geometry_preflight', 'can_generate_video', 'human_anatomy_verified',
                    'generated_image_verified', 'physical_execution_verified'):
            self.assertFalse(r[key])

    def test_wrong_camera_rejected(self):
        rig = deepcopy(self.rig); rig['camera_binding']['camera_id'] = 'native'
        with self.assertRaisesRegex(ValueError, 'camera'): self.compute(rig)

    def test_far_side_is_actual_world_binding(self):
        rig = deepcopy(self.rig); rig['torso']['waist_world_m'][0] *= -1
        with self.assertRaisesRegex(ValueError, 'far edge'): self.compute(rig)

    def test_wrong_actor_and_tcp_mapping_rejected(self):
        for field, value in [('actor_id', 'operator_2'), ('target_assumption', {'kind': 'robot_tcp'})]:
            rig = deepcopy(self.rig); rig[field] = value
            with self.assertRaises(ValueError): self.compute(rig)

    def test_foreign_collision_world_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'collision_world.json'; write_json(path, self.world)
            with self.assertRaisesRegex(ValueError, 'receipt execution'): m._load_world(self.data, path)

    def test_larger_protocol_tolerance_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = read_json(PROTOCOL); p['fixed_engineering_parameters']['numerical_distance_tolerance_m'] = .5
            path = Path(tmp)/'protocol.json'; write_json(path, p)
            with self.assertRaisesRegex(ValueError, 'protocol/parameters'): self.compute(protocol=path)

    def test_real_wrapper_binds_static_rig_and_blocks_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rig_path, report_path = tmp/'rig.json', tmp/'report.json'
            write_json(rig_path, self.rig)
            m.compile_geometry(PACKAGE, rig_path, WORLD, PROTOCOL, report_path)
            config = {'anchor_model': 'test_model_no_network'}
            result = m.prepare_with_geometry(report_path, tmp/'prepared', config)
            self.assertEqual(result['status'], 'blocked_geometry_preflight')
            request = read_json(result['request']['path'])
            self.assertEqual(request['initial_conditions']['operator']['synthetic_geometry']['rig'], self.rig)
            self.assertNotIn('event_targets', request['initial_conditions']['operator']['synthetic_geometry'])
            with patch.object(m.unified, 'execute_initial') as provider:
                with self.assertRaisesRegex(ValueError, 'Geometry preflight blocked'):
                    m.execute_with_geometry(tmp/'prepared/wrapper.json', config, tmp/'nonexistent_budget.json')
                provider.assert_not_called()

    def test_tampered_report_cannot_upgrade_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); rig_path = tmp/'rig.json'; report_path = tmp/'report.json'
            write_json(rig_path, self.rig)
            m.compile_geometry(PACKAGE, rig_path, WORLD, PROTOCOL, report_path)
            report = read_json(report_path); report['clears_geometry_preflight'] = True
            report_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'recomputed'): m.validate_geometry(report_path)


if __name__ == '__main__':
    unittest.main()
