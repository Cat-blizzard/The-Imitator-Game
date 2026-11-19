"""Actual saved-scene composition boundaries; no GPU/API/physics execution."""
from copy import deepcopy
import subprocess
import unittest
from resimpair_grasp.common import ROOT, digest, read, resolve
from resimpair_tasks import closure_background_replay_v1 as c


class SavedSourceComposition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        specs = read(ROOT / 'reports/closure_background_20261030_v1/source_specs.json')['sources']
        cls.actual = []
        for spec in specs:
            source = c.batch.appearance.replay.validate_source(resolve(spec['execution']).parent)
            base, _ = c.render.baseline_scene(source)
            envelope = c.furniture.domain.build_motion_envelope(base, source['source'] / 'measurements.npz')
            cls.actual.append((spec, source, base, envelope))

    def candidate(self, item, **changes):
        spec, source, base, envelope = item
        args = dict(seed=spec['seed'], room=spec['room'], partition=spec['partition'],
                    motion_envelope=envelope, catalog=c.furniture.CATALOG)
        args.update(changes)
        return c.candidate_scene(base, **args)

    def test_both_actual_native_sources_keep_physics_and_original_camera_rays(self):
        for item in self.actual:
            with self.subTest(source=item[0]['id']):
                before = deepcopy(item[2]); value, receipt, diagnosis = self.candidate(item)
                self.assertEqual(item[2], before)
                self.assertEqual(c.furniture.domain.physical_scene(before), c.furniture.domain.physical_scene(value))
                expected = deepcopy(before['cameras']); expected['opposite_high'].update(width=1280, height=720)
                self.assertEqual(value['cameras'], expected)
                self.assertEqual(len(value['scene_domain']['geometry']['furniture']), 4)
                self.assertTrue(all(x['scale'] == [1., 1., 1.] for x in value['scene_domain']['geometry']['furniture']))
                self.assertEqual(diagnosis['finite_structure_blocked_count'], 0)
                self.assertFalse(receipt['quality_improvement_established'])

    def test_wrong_partition_is_rejected_without_mutating_native_scene(self):
        item = self.actual[0]; before = digest(item[2])
        with self.assertRaisesRegex(ValueError, 'source_partition_must_match'):
            self.candidate(item, partition='development')
        self.assertEqual(digest(item[2]), before)

    def test_missing_visual_catalog_fails_without_changing_physical_scene(self):
        item = self.actual[0]; before = digest(item[2])
        with self.assertRaises(FileNotFoundError):
            self.candidate(item, catalog=ROOT / 'assets/nonexistent_review_fixture/catalog.json')
        self.assertEqual(digest(item[2]), before)

    def test_dispatch_does_not_monkeypatch_global_subprocess(self):
        original = subprocess.Popen
        with c.dispatch():
            self.assertIs(subprocess.Popen, original)
            self.assertIsNot(c.render.subprocess, subprocess)
            with self.assertRaisesRegex(ValueError, 'unexpected_child_worker_command'):
                c.render.subprocess.Popen(['nvidia-smi'])
        self.assertIs(c.render.subprocess, subprocess)


if __name__ == '__main__': unittest.main()
