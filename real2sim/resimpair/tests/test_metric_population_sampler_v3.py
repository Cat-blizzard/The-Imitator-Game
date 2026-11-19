import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from resimpair_grasp.common import binding, write, read
from resimpair_pickplace.requested_tasks import Rejected
from resimpair_tasks import metric_population_sampler_v2 as retained
from resimpair_tasks import metric_population_sampler_v3 as sampler


def item(source, radius, variant='original'):
    descriptor = {'source_id': source, 'partition': 'development',
                  'half_size': [radius / math.sqrt(2), radius / math.sqrt(2), .02],
                  'dimensions_m': [radius * math.sqrt(2), radius * math.sqrt(2), .04]}
    return ({'source_id': source, 'variant_id': source + '_' + variant}, descriptor)


def bins(capacity=.082):
    return [({'path': 'fixture_' + str(i), 'sha256': str(i) * 64},
             {'source_id': 'bin_' + str(i), 'partition': 'development', 'dimensions_m': [.20, .20, .06],
              'cavity': {'flat_floor_radius_m': capacity}}) for i in range(2)]


def outside_table(**kwargs):
    return {'xy': [10., 0.], 'yaw': 0.}


def never_qualify(scene, tasks):
    return {'passed': False, 'reason': 'unit_fixture_unreachable', 'original_scene_id': scene['scene_instance_id']}


class CapacityTests(unittest.TestCase):
    def test_original_slots_and_exact_boundary(self):
        self.assertIs(sampler.slots, retained.slots)
        self.assertTrue(sampler.slots([.025, .025], .0745)['passed'])
        self.assertFalse(sampler.slots([.025, .025], .0745 - 1e-7)['passed'])

    def test_stable_global_identity(self):
        base = sampler.scene_identity('a' * 64, 42, 0)
        self.assertEqual(base, sampler.scene_identity('a' * 64, 42, 0))
        self.assertEqual(4, len({base, sampler.scene_identity('b' * 64, 42, 0),
                                sampler.scene_identity('a' * 64, 43, 0),
                                sampler.scene_identity('a' * 64, 42, 1)}))
        self.assertLess(len(base + '_distribute_groups'), 81)

    def test_eight_distinct_sources_can_split_without_all_in_one(self):
        pool = {str(i): [item(str(i), .02)] for i in range(20)}
        original = copy.deepcopy(pool)
        group, selected, receipt = sampler.capacity_select(pool, bins(), 8, 'development', 17)
        self.assertEqual(8, len({d['source_id'] for _, d in group}))
        self.assertEqual(original, pool)
        self.assertTrue(all(sampler.slots([sampler.footprint(x) for x in group[c::2]], .082)['passed'] for c in range(2)))
        self.assertFalse(sampler.slots([sampler.footprint(x) for x in group], .082)['passed'])
        spec, report = sampler.propose('scene_eight', group, selected, 18, ({}, {}))
        self.assertEqual(8, len(spec['objects']))
        self.assertNotIn('collect_all', report['compiled_goal_types'])
        self.assertEqual(8, len(next(t for t in spec['tasks'] if t['id'] == 'distribute_groups')['placements']))
        self.assertEqual(8, sum(receipt['group_sizes']))

    def test_insufficient_capacity_is_failure_not_shrink_or_reduce_n(self):
        pool = {str(i): [item(str(i), .09)] for i in range(10)}
        with self.assertRaises(Rejected) as caught:
            sampler.capacity_select(pool, bins(.05), 8, 'development', 3)
        self.assertEqual('no_same_partition_container_for_requested_subgroup_size', caught.exception.code)

    def test_completion_preserves_source_uniqueness_and_fixed_variants(self):
        pool = {'large': [item('large', .06)], 'small': [item('small', .01), item('small', .02, 'large')],
                'other': [item('other', .02)], 'third': [item('third', .03)]}
        selected = [[pool['large'][0]], []]
        completed = sampler.completion(selected, pool, [2, 2], bins(.1))
        self.assertIsNotNone(completed)
        sources = [d['source_id'] for group in completed for _, d in group]
        self.assertEqual(4, len(set(sources)))
        self.assertEqual('large', completed[0][0][1]['source_id'])

    def test_partition_conflict_is_rejected(self):
        pool = {str(i): [item(str(i), .02)] for i in range(8)}
        pool['0'][0][1]['partition'] = 'heldout_asset'
        with self.assertRaisesRegex(Rejected, 'source_partition_or_identity_conflict'):
            sampler.capacity_select(pool, bins(), 3, 'development', 4)

    def test_pose_hook_cannot_bypass_table_bounds(self):
        pool = {str(i): [item(str(i), .02)] for i in range(8)}
        group, selected, _ = sampler.capacity_select(pool, bins(), 3, 'development', 4)
        with self.assertRaises(Rejected) as caught:
            sampler.propose('scene_hook', group, selected, 4, ({}, {}), object_pose_proposal=outside_table)
        self.assertEqual('bounded_initial_object_layout_exhausted', caught.exception.code)
        self.assertEqual(128, len(caught.exception.details['pose_attempts']))

    def test_count_schedule_validation_precedes_io(self):
        for schedule in ([3], [3, 9], [3, True]):
            with self.assertRaisesRegex(ValueError, 'count_schedule'):
                sampler.build('/missing', '/unused', scenes=2, seed=1, count_schedule=schedule)

    def test_layout_rejection_saved_for_each_proposal_no_count_change(self):
        pool = {str(i): [item(str(i), .02)] for i in range(8)}
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp); bundle = temp / 'bundle.json'; static = temp / 'static.json'
            write(static, {'assets': []}); write(bundle, {'static_catalog': binding(static)})
            b = read(bundle)
            with patch.object(sampler, 'admitted_pool', return_value=(b, [], pool, bins())), \
                 patch.object(sampler.factory, 'static_source_catalog', return_value=({}, {})), \
                 patch.object(sampler.factory, 'compile_exact_scene', return_value=({'scene_instance_id': 'fixture'}, [], {})):
                result = sampler.build(bundle, temp / 'out', scenes=1, seed=8, count_schedule=[3],
                                       proposals=2, layout_qualifier=never_qualify)
            self.assertEqual(0, result['accepted_geometry_scenes'])
            attempts = read(temp / 'out/sampling/0000/attempts.json')['attempts']
            self.assertEqual(2, len(attempts))
            self.assertTrue(all(a['reason'] == 'layout_qualifier_rejected' for a in attempts))
            self.assertTrue(all(a['requested_object_count'] == 3 for a in attempts))
            self.assertTrue(all(a['layout_qualification']['sha256'] for a in attempts))


if __name__ == '__main__':
    unittest.main()
