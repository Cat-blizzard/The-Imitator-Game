from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from resimpair_grasp.common import binding, write
from resimpair_tasks import metric_population_sampler_v3 as old
from resimpair_tasks import metric_population_sampler_v4 as candidate


class GenerationIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.bundle = self.directory / 'bundle.json'
        self.catalog = self.directory / 'catalog.json'
        write(self.bundle, {'fixture': 'bundle'})
        write(self.catalog, {'fixture': 'catalog'})
        self.kwargs = dict(scenes=2, seed=42, minimum=3, maximum=8, proposals=16,
            container_catalog=self.catalog, count_schedule=[3, 4],
            hooks={'object_pose_proposal': None, 'layout_qualifier': None})
        self.contract = candidate.generation_contract(self.bundle, **self.kwargs)

    def tearDown(self):
        self.temporary.cleanup()

    def test_same_contract_same_identity(self):
        again = candidate.generation_contract(self.bundle, **self.kwargs)
        self.assertEqual(candidate.scene_identity(self.contract, 0, 3), candidate.scene_identity(again, 0, 3))

    def test_changed_N_even_at_same_seed_changes_identity(self):
        other = candidate.generation_contract(self.bundle, **{**self.kwargs, 'count_schedule': [4, 4]})
        self.assertNotEqual(candidate.scene_identity(self.contract, 0, 3), candidate.scene_identity(other, 0, 4))

    def test_prior_and_future_counts_bound_for_same_scene_index(self):
        for schedule in ([8, 4], [3, 8]):
            other = candidate.generation_contract(self.bundle, **{**self.kwargs, 'count_schedule': schedule})
            self.assertNotEqual(candidate.scene_identity(self.contract, 1, 4),
                                candidate.scene_identity(other, 1, schedule[1]))

    def test_all_generation_inputs_bound(self):
        cases = [({'seed': 43}, 3), ({'minimum': 2}, 3), ({'maximum': 7}, 3),
                 ({'proposals': 15}, 3), ({'count_schedule': None}, None),
                 ({'scenes': 1, 'count_schedule': [3]}, 3), ({'container_catalog': None}, 3),
                 ({'hooks': {'object_pose_proposal': {'implementation': {'sha256': 'a'}, 'qualified_name': 'one'},
                             'layout_qualifier': None}}, 3),
                 ({'hooks': {'object_pose_proposal': None, 'layout_qualifier': {'implementation': {'sha256': 'b'},
                                                                           'qualified_name': 'two'}}}, 3)]
        baseline = candidate.scene_identity(self.contract, 0, 3)
        for delta, n in cases:
            other = candidate.generation_contract(self.bundle, **{**self.kwargs, **delta})
            self.assertNotEqual(baseline, candidate.scene_identity(other, 0, n or other['effective_count_schedule'][0]))

    def test_exact_catalog_content_binding_changes_identity(self):
        alternate = self.directory / 'other.json'; write(alternate, {'fixture': 'changed'})
        other = candidate.generation_contract(self.bundle, **{**self.kwargs, 'container_catalog': alternate})
        self.assertNotEqual(candidate.scene_identity(self.contract, 0, 3), candidate.scene_identity(other, 0, 3))

    def test_N_and_index_must_agree_with_contract(self):
        with self.assertRaisesRegex(ValueError, 'requested_N'):
            candidate.scene_identity(self.contract, 0, 8)
        with self.assertRaisesRegex(ValueError, 'scene_index'):
            candidate.scene_identity(self.contract, 2, 3)

    def test_contract_copies_input_schedule_and_hooks(self):
        expected = deepcopy(self.contract)
        self.kwargs['count_schedule'][0] = 8
        self.kwargs['hooks']['layout_qualifier'] = 'changed'
        self.assertEqual(expected, self.contract)

    def test_frozen_sampling_implementation_is_reused(self):
        self.assertIs(candidate.capacity_select, old.capacity_select)
        self.assertIs(candidate.propose, old.propose)
        self.assertIs(candidate.factory, old.factory)
        self.assertIs(candidate._consumer_bins, old._consumer_bins)


if __name__ == '__main__':
    unittest.main()
