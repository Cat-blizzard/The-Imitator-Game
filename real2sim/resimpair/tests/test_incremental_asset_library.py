import tempfile
from pathlib import Path
import unittest

from resimpair_grasp.common import binding, read, write
from resimpair_tasks.incremental_asset_library_v1 import ordered, content_key, one, inspect_output, status


class IncrementalAssetLibraryTests(unittest.TestCase):
    def test_ordering_is_repeatable_and_not_input_order_dependent(self):
        rows = [{'source_id': f's{i}', 'source': f'library{i%3}',
                 'partition': 'heldout', 'category': 'object'} for i in range(18)]
        self.assertEqual(ordered(rows, 7), ordered(list(reversed(rows)), 7))
        self.assertEqual(len({r['source'] for r in ordered(rows, 7)[:3]}), 3)

    def test_native_content_key_separates_scale_and_geometry(self):
        d = {'visual': {'sha256': 'mesh_a'}}
        v = {'source_to_asset': [1, 0, 0], 'dimensions_m': [.06, .04, .04]}
        a = content_key(d, v, [{'sha256': 'converter'}])
        self.assertNotEqual(a, content_key(d, {**v, 'dimensions_m': [.07, .04, .04]}, [{'sha256': 'converter'}]))
        self.assertNotEqual(a, content_key({'visual': {'sha256': 'mesh_b'}}, v, [{'sha256': 'converter'}]))

    def test_unknown_local_reservation_is_never_resubmitted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); write(root/'jobs/key/reservation.json', {'started': True})
            with self.assertRaisesRegex(ValueError, 'Unknown reserved'):
                one(root, {'content_key': 'key'}, {})

    def test_unknown_global_reservation_is_never_resubmitted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); asset = root/'asset.json'; write(asset, {'test': True})
            registry = root/'registry'; write(registry/'jobs/key.json', {'output': str(root/'other')})
            with self.assertRaisesRegex(ValueError, 'unresolved global'):
                one(root, {'content_key': 'key', 'asset': binding(asset)},
                    {'code': [], 'min_free_gib': 0, 'registry': str(registry)})

    def test_geometric_readiness_does_not_depend_on_legacy_pose_height_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); p = root/'asset.json'
            write(p, {'qualification': {'geometry': 'pass'}, 'roles': {'rigid_manipulation_candidate': True}})
            write(root/'refined/catalog.json', {'assets': [binding(p)]})
            result = inspect_output(root, {'source_id': 'asset'}, 1)
            self.assertEqual(result['state'], 'native_geometry_ready')
            self.assertEqual(result['offline_antipodal_status'], 'unavailable_legacy_height_gate')
            self.assertEqual(result['dynamic_status'], 'unmeasured')

    def test_geometry_failure_is_never_admitted_as_unknown_pose(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); p = root/'asset.json'
            write(p, {'qualification': {'geometry': 'fail'}, 'roles': {'rigid_manipulation_candidate': False}})
            write(root/'refined/catalog.json', {'assets': [binding(p)]})
            (root/'worker.log').write_text('ValueError: independent_collision_geometry_rejected\n')
            result = inspect_output(root, {'source_id': 'asset'}, 1)
            self.assertEqual(result['state'], 'native_rejected')
            self.assertEqual(result['failure_class'], 'independent_native_geometry_rejected')

    def test_unknown_status_never_enters_ready_index(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write(root/'protocol.json', {'prior_cpu_conversion_jobs': 288})
            write(root/'selection.json', {'rows': [{'source_id': 'one', 'content_key': 'key'}]})
            write(root/'jobs/key/reservation.json', {'started': True})
            report = status(root)
            self.assertEqual(report['unknown_keys'], ['key'])
            self.assertEqual(report['native_ready_count'], 0)
            self.assertEqual(read(root/'native_ready_index.json')['entries'], [])


if __name__ == '__main__': unittest.main()
