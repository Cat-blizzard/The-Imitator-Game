import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from resimpair_grasp.common import ROOT, read, write, resolve, sha
from resimpair_pickplace.pose_bundle import export_bundle, inspect, materialize, checked_file
from resimpair_grasp.grasps import load


class PoseBundleTests(unittest.TestCase):
    def sample(self, root):
        demo = ROOT / 'assets/general_pickplace_demo_v1'
        entry = read(demo / 'cache_index.json')['entries'][0]
        index = root / 'source_index.json'; write(index, {'entries': [entry]})
        container = read(demo / 'container_catalog.json')['assets'][0]
        cat = root / 'source_containers.json'; write(cat, {'assets': [container]})
        export_bundle({'cudex': index}, cat, root / 'bundle')
        return entry

    def test_relocation_preserves_poses_and_needs_no_optimizer(self):
        import builtins
        real_import = builtins.__import__
        def no_optimizer(name, *args, **kwargs):
            if name.split('.')[0] in {'curobo', 'graspgenx'}:
                raise AssertionError('Cached runtime attempted to load an optimizer')
            return real_import(name, *args, **kwargs)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); original = self.sample(root)
            shutil.move(root / 'bundle', root / 'relocated')
            with patch('builtins.__import__', side_effect=no_optimizer):
                paths = materialize(root / 'relocated', root / 'imported')
                entry = read(paths['cudex'])['entries'][0]
                new = load(entry['cache'], entry['asset'])
            old = load(original['cache'], original['asset'])
            self.assertEqual(new['candidates'], old['candidates'])
            self.assertEqual(read(resolve(entry['asset']))['visual']['sha256'],
                             read(resolve(original['asset']))['visual']['sha256'])
            self.assertEqual(materialize(root / 'relocated', root / 'imported'), paths)

    def test_missing_or_changed_geometry_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); self.sample(root)
            manifest = inspect(root / 'bundle')
            visual = next(v for v in manifest['files'] if v['path'].endswith('visual.glb'))
            p = root / 'bundle' / visual['path']; p.write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'changed'): inspect(root / 'bundle')

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, 'escapes'):
                checked_file(temporary, {'path': '../outside', 'sha256': 'invalid'})

    def test_physical_lineage_survives_clone_without_double_counting(self):
        from resimpair_pickplace.cache_runtime import physical_lineage
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); budget = root / 'budget'
            write(budget / 'limits.json', {'physical_attempt_limit': 3})
            write(budget / 'reservations/0.json', {'attempt': 'already_published'})
            write(root / 'provenance/grasp_compute_lineage.json', {'physical_reserved': 50,
                'carried_reservation_hashes': [sha(budget / 'reservations/0.json')]})
            self.assertEqual(physical_lineage(root), 50)
            for name in ['cached_pose_campaign_registry', 'grasp_benchmark_registry']:
                write(root / 'runtime' / name / 'campaign.json', {'budget': str(budget)})
            self.assertEqual(physical_lineage(root), 50)
            write(budget / 'reservations/1.json', {'attempt': 'new'})
            self.assertEqual(physical_lineage(root), 51)


if __name__ == '__main__': unittest.main()
