import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from resimpair_pickplace.contact_pose_refinement_v1 import centered_pose


class ContactPoseRefinementTest(unittest.TestCase):
    def test_asymmetric_contact_recenters_in_rotated_closing_frame(self):
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler('zyx', [.7, .3, -.2]).as_matrix()
        pose[:3, 3] = [.2, -.4, .1]
        left, right = .020, .024
        corrected, shift = centered_pose(pose, left, right)
        left_surface = pose[:3, 3] + pose[:3, 1]*(left/2)
        right_surface = pose[:3, 3] - pose[:3, 1]*(right/2)
        np.testing.assert_allclose(corrected[:3, 3], (left_surface+right_surface)/2)
        np.testing.assert_array_equal(corrected[:3, :3], pose[:3, :3])
        np.testing.assert_array_equal(pose[:3, 3], [.2, -.4, .1])
        self.assertAlmostEqual(shift, -.001)

    def test_unbounded_or_nonfinite_contact_does_not_generate_pose(self):
        for left, right in [(float('nan'), .02), (-.01, .02), (.075, .01)]:
            with self.assertRaises(ValueError): centered_pose(np.eye(4), left, right)


if __name__ == '__main__': unittest.main()
