from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from resimpair_grasp.common import binding, write
from resimpair_tasks import metric_task_factory_v3 as factory


class ContainerBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.refs = []
        for i, partition in enumerate(['development', 'development', 'heldout_asset']):
            path = self.root / f'{i}.json'
            write(path, {'source_id': 'source', 'variant_id': str(i), 'partition': partition,
                         'roles': {'stationary_container': True}})
            self.refs.append(binding(path))
        self.requests = [{'id': 'bin', 'source_id': 'source', 'xy': [0, 0]}]

    def tearDown(self):
        self.temp.cleanup()

    def test_selected_descriptor_not_smaller_same_source(self):
        selected = factory.select_container_variants(self.refs, self.requests,
                                                      {'source': self.refs[1]}, 'development')
        self.assertEqual([self.refs[1]], selected)

    def test_partition_or_missing_or_extra_source_rejected(self):
        for explicit in ({}, {'wrong': self.refs[0]}, {'source': self.refs[2]},
                         {'source': self.refs[0], 'extra': self.refs[1]}):
            with self.assertRaises(ValueError):
                factory.select_container_variants(self.refs, self.requests, explicit, 'development')

    def test_nonmember_rejected_even_if_source_matches(self):
        with self.assertRaisesRegex(ValueError, 'not_in_immutable_bundle'):
            factory.select_container_variants(self.refs[:1], self.requests,
                                              {'source': self.refs[1]}, 'development')

    def test_changed_descriptor_hash_rejected(self):
        bad = {**self.refs[0], 'sha256': '0' * 64}
        with self.assertRaisesRegex(ValueError, 'hash changed'):
            factory.select_container_variants(self.refs, self.requests, {'source': bad}, 'development')

    def test_compiler_only_receives_bound_variant_and_drift_rejected(self):
        asset = self.root / 'object.json'; write(asset, {'partition': 'development'})
        spec = {'id': 'scene', 'containers': self.requests, 'container_variant_choices': {'source': self.refs[1]},
                'objects': [], 'tasks': []}
        expected_scene = {'supports': [{'id': 'bin', 'container_descriptor': self.refs[1]}]}
        with patch.object(factory.retained, 'compile_scene', return_value=(expected_scene, [], {})) as compiler:
            scene, _, checks = factory.compile_exact_scene(spec, [{'asset': binding(asset)}], self.refs, 1)
            self.assertEqual([self.refs[1]], compiler.call_args.args[2])
            self.assertTrue(checks['exact_container_bindings'][0]['exact_match'])
        wrong = {'supports': [{'id': 'bin', 'container_descriptor': self.refs[0]}]}
        with patch.object(factory.retained, 'compile_scene', return_value=(wrong, [], {})):
            with self.assertRaisesRegex(ValueError, 'compiled_container_variant_changed'):
                factory.compile_exact_scene(spec, [{'asset': binding(asset)}], self.refs, 1)


if __name__ == '__main__':
    unittest.main()
