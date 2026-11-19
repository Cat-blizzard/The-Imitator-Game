import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from resimpair_pickplace.payload_descent_v1 import payload_corridor, install
from resimpair_pickplace import whole_path_motion_v1


class PayloadDescentTests(unittest.TestCase):
    def test_payload_offset_rotation_is_not_tcp_centre(self):
        poses = np.repeat(np.eye(4)[None], 3, axis=0)
        poses[1, :3, :3] = Rotation.from_euler('z', .3).as_matrix()
        local = np.eye(4); local[0, 3] = .08
        self.assertFalse(payload_corridor(poses, local, np.eye(4))['passed'])

    def test_vertical_translation_is_allowed_and_xy_bow_is_rejected(self):
        poses = np.repeat(np.eye(4)[None], 3, axis=0)
        poses[:, 2, 3] = [.2, .1, 0.]
        self.assertTrue(payload_corridor(poses, np.eye(4), np.eye(4))['passed'])
        poses[1, 0, 3] = .016
        self.assertFalse(payload_corridor(poses, np.eye(4), np.eye(4))['passed'])

    def test_unknown_or_nonfinite_geometry_is_rejected(self):
        for value in (np.empty((0, 4, 4)), np.full((1, 4, 4), np.nan)):
            with self.assertRaises(ValueError):
                payload_corridor(value, np.eye(4), np.eye(4))

    def test_original_dispatch_restored_on_failure(self):
        original = whole_path_motion_v1.WholePathMotion.plan
        with self.assertRaisesRegex(RuntimeError, 'scope_exit'):
            with install():
                self.assertIsNot(whole_path_motion_v1.WholePathMotion.plan, original)
                raise RuntimeError('scope_exit')
        self.assertIs(whole_path_motion_v1.WholePathMotion.plan, original)


if __name__ == '__main__':
    unittest.main()
