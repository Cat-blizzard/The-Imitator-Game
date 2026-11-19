from copy import deepcopy
import unittest

from resimpair_tasks.container_affordance_consumer_v1 import measured_affordance, describe_goal, place_relation


def fixture(label='deeper_open_cavity_geometry_candidate', passed=True):
    row = {'descriptor': {'path': 'unit_fixture', 'sha256': 'a' * 64}, 'source_id': 'source',
        'variant_id': 'variant', 'partition': 'development', 'native_body_height_m': .10,
        'native_min_z_m': 0., 'native_max_z_m': .10, 'floor_z_m': .01, 'rim_z_m': .10,
        'cavity_depth_m': .09, 'floor_relative_body_height': .1, 'rim_relative_body_height': 1.,
        'depth_relative_body_height': .9, 'flat_floor_radius_m': .07,
        'depth_to_floor_diameter_ratio': .09 / .14, 'declared_and_actual_collision_bounds_max_difference_m': 0.,
        'descriptive_geometry_bin': label,
        'retained_oriented_floor_support': {'passed': passed, 'reason': None if passed else 'bad_support'}}
    descriptor = {'name': 'Source title', 'dimensions_m': [.2, .2, .1], 'visual': {}, 'collision': {},
        'cavity': {'floor_z_m': .01, 'rim_z_m': .10, 'flat_floor_radius_m': .07, 'minimum_wall_radius_m': .076}}
    return row, descriptor


class AffordanceTests(unittest.TestCase):
    def test_deep_geometry_is_not_semantic_containment(self):
        row, d = fixture(); value = measured_affordance(row, d)
        self.assertEqual('into_open_container', value['place_relation']['geometry_candidate'])
        self.assertEqual('unknown', value['place_relation']['value'])
        self.assertFalse(value['place_relation']['into_open_container_allowed'])
        self.assertFalse(value['place_relation']['assertable_by_R2H'])

    def test_all_bins_separate_geometry_candidates(self):
        for label, expected in [('shallow_tray_like_cavity', 'on_tray'),
                                ('elevated_shallow_recess_or_support_surface', 'on_support_surface'),
                                ('intermediate_cavity_geometry_unknown_semantics', 'unknown')]:
            row, _ = fixture(label)
            self.assertEqual(expected, place_relation(row)['geometry_candidate'])

    def test_failed_native_support_cannot_be_used_as_geometry_fallback(self):
        row, d = fixture(passed=False); value = measured_affordance(row, d)
        self.assertFalse(value['consumer_admitted'])
        self.assertIsNone(value['place_relation']['geometry_fallback'])
        self.assertEqual('unknown', value['place_relation']['geometry_candidate'])

    def test_payload_rim_margin_preserves_protrusion_without_declaring_failure(self):
        row, d = fixture(); value = measured_affordance(row, d)
        small = describe_goal(value, payload_height_m=.05)
        tall = describe_goal(value, payload_height_m=.15)
        self.assertAlmostEqual(.04, small['payload_top_below_rim_margin_m'])
        self.assertAlmostEqual(-.06, tall['payload_top_below_rim_margin_m'])
        self.assertTrue(small['whole_nominal_payload_below_rim'])
        self.assertFalse(tall['whole_nominal_payload_below_rim'])
        self.assertFalse(small['place_relation']['into_open_container_allowed'])
        self.assertEqual('unknown', tall['place_relation']['value'])

    def test_measurements_do_not_mutate_original_evidence(self):
        row, d = fixture(); original = deepcopy((row, d))
        value = measured_affordance(row, d)
        self.assertEqual(original, (row, d))
        self.assertEqual(row['descriptor'], value['descriptor'])
        self.assertAlmostEqual(.006, value['measurements']['flat_floor_to_minimum_wall_radius_margin_m'])

    def test_inconsistent_or_invalid_measurements_rejected(self):
        for key, value in [('cavity_depth_m', .2), ('native_body_height_m', float('nan')), ('rim_z_m', .2)]:
            row, d = fixture(); row[key] = value
            with self.assertRaises(ValueError):
                measured_affordance(row, d)
        row, d = fixture()
        for height in [0, -1, float('nan')]:
            with self.assertRaises(ValueError):
                describe_goal(measured_affordance(row, d), payload_height_m=height)


if __name__ == '__main__':
    unittest.main()
