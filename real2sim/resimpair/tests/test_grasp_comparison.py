import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from resimpair_benchmark.ledger import reserve, check_request
from resimpair_benchmark.proposals import tcp_to_hand_seed
from resimpair_grasp.common import write
from resimpair_grasp.cudex_native import hand_to_tcp


class ComparisonTest(unittest.TestCase):
    def test_hybrid_seeds_preserve_native_tcp_on_arbitrary_rotations(self):
        rng = np.random.default_rng(412)
        poses = np.repeat(np.eye(4)[None], 32, axis=0)
        poses[:, :3, :3] = Rotation.random(32, random_state=rng).as_matrix()
        poses[:, :3, 3] = rng.normal(size=(32, 3)) * .1
        seeds = tcp_to_hand_seed(poses)
        np.testing.assert_allclose(np.array([hand_to_tcp(q) for q in seeds]), poses, atol=1e-12)
        np.testing.assert_allclose(seeds[:, 7:], .04)

    def test_output_changes_do_not_reset_budget_or_retry_unknown(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root / 'limits.json', {'gpu_job_limit': 1})
            identity, request = {'asset': 'a', 'method': 'hybrid'}, {'input': 7}
            record, fresh = reserve(root, identity, request, root / 'first')
            self.assertTrue(fresh)
            self.assertFalse(reserve(root, identity, request, root / 'first')[1])
            self.assertEqual(check_request(root, identity, request, root / 'first'), record)
            with self.assertRaises(ValueError): reserve(root, identity, request, root / 'second')
            with self.assertRaises(RuntimeError): reserve(root, {'asset': 'b'}, request, root / 'second')
            with self.assertRaises(ValueError): check_request(root, identity, {'input': 8}, root / 'first')


if __name__ == '__main__': unittest.main()
