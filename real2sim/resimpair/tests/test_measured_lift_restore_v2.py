"""Real-state-derived counter selection and exact native replay regression."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from resimpair_grasp.common import ROOT, binding, read, resolve
from resimpair_pickplace.measured_lift_restore_v2 import select_attachment, restore_guard


FIXTURE = ROOT/'tests/fixtures/measured_lift_restore_v2'


class MeasuredLiftRestoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='resimpair_measured_restore_')
        self.addCleanup(self.temp.cleanup)
        self.fixture = Path(self.temp.name)/'fixture'
        shutil.copytree(FIXTURE, self.fixture)
        self.execution = self.fixture/'execution'
        self.motion = self.execution/'transfer_0_motion'
        bundle = read(self.execution/'execution_bundle.json')
        bundle['measurements'] = binding(self.fixture/'measurements.npz')
        self.put(self.execution/'execution_bundle.json', bundle)

    @staticmethod
    def put(path, value):
        Path(path).write_text(json.dumps(value, indent=2)+'\n')

    def request(self, counter):
        return next(self.motion.glob(f'{counter:03d}_lift_*/request.json'))

    def test_old_request_cannot_select_future_same_frame_refresh(self):
        for counter in (10, 12):
            result = select_attachment(self.request(counter), expected_request=binding(self.request(counter)))
            self.assertEqual(result['selected_attachment']['counter'], 10)
            self.assertEqual(len(result['other_same_frame_attachments']), 1)
            future = result['other_same_frame_attachments'][0]
            self.assertEqual(future['counter'], 13)
            self.assertFalse(future['eligible_before_or_at_request'])
            self.assertEqual(result['measurement_alignment']['qpos_max_error_rad'], 0)

    def test_later_request_selects_latest_available_refresh(self):
        result = select_attachment(self.request(13))
        self.assertEqual(result['selected_attachment']['counter'], 13)
        self.assertEqual(result['other_same_frame_attachments'][0]['counter'], 10)
        self.assertTrue(result['other_same_frame_attachments'][0]['eligible_before_or_at_request'])

    def test_request_before_first_refresh_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'no_attachment_at_or_before'):
            select_attachment(self.request(9))

    def test_same_numeric_latest_counter_is_ambiguous(self):
        shutil.copyfile(self.motion/'attachment_013_lift.json', self.motion/'attachment_0013_lift.json')
        with self.assertRaisesRegex(ValueError, 'ambiguous_attachment_counter'):
            select_attachment(self.request(13))

    def test_inconsistent_future_refresh_is_not_silently_ignored(self):
        path = self.motion/'attachment_013_lift.json'
        data = read(path); data['measured_object_in_tcp'][0][3] += .005
        self.put(path, data)
        with self.assertRaisesRegex(ValueError, 'attachment_disagrees_with_measured_frame'):
            select_attachment(self.request(10))

    def test_request_joint_state_must_match_same_recorded_frame(self):
        path = self.request(12)
        request = read(path); request['current'][0] += .01
        self.put(path, request)
        with self.assertRaisesRegex(ValueError, 'request_state_does_not_match'):
            select_attachment(path)

    def test_measurement_hash_is_checked(self):
        path = self.fixture/'measurements.npz'
        path.write_bytes(path.read_bytes()+b'changed-fixture-copy')
        with self.assertRaisesRegex(ValueError, 'hash changed'):
            select_attachment(self.request(12))

    def test_other_frame_is_not_used_for_this_request(self):
        data = read(self.motion/'attachment_013_lift.json')
        data['frame'] = 1
        self.put(self.motion/'attachment_014_lift.json', data)
        result = select_attachment(self.request(13))
        self.assertEqual(result['selected_attachment']['counter'], 13)
        self.assertEqual(len(result['other_same_frame_attachments']), 1)

    def test_real_native_world_reproduces_original_rejection_with_two_refreshes(self):
        provenance = read(FIXTURE/'provenance.json')
        try:
            source_request = resolve(provenance['source_request'])
        except (FileNotFoundError, ValueError):
            self.skipTest('Historical native mesh source is not installed; portable counter tests still run')
        source_motion = source_request.parent.parent
        for source in source_motion.glob('*.obj'):
            (self.motion/source.name).symlink_to(source)
        shutil.copyfile(source_motion/'world.json', self.motion/'world.json')
        request_path = self.request(12)
        request = read(request_path)
        guard, audit = restore_guard(request_path)
        self.assertEqual(audit['selected_attachment']['counter'], 10)
        self.assertEqual(audit['other_same_frame_attachments'][0]['counter'], 13)
        with np.load(self.fixture/'measurements.npz', allow_pickle=False) as values:
            np.testing.assert_array_equal(guard.root, values['robot_roots_world'][0])
        with np.load(source_motion/'012_lift_rrt/path.npz', allow_pickle=False) as values:
            report, dense, _ = guard.check({'position': values['positions']},
                                           np.asarray(request['current']), np.asarray(request['goal']), 'lift')
        old = read(source_motion/'012_lift_rrt/decision.json')['report']
        self.assertEqual(report, old)
        self.assertFalse(report['passed'])
        self.assertEqual(len(dense), 76)
        self.assertTrue(report['all_dense_collision_samples_checked'])
        self.assertFalse(audit['physical_execution'])


if __name__ == '__main__':
    unittest.main()
