"""CPU regression of cuDex's ESDF-to-cost primitive sparse-buffer hazard."""
from types import SimpleNamespace
import unittest

import torch

from resimpair_grasp.cudex_constraint_v3 import query_both_modes


def buffer():
    return SimpleNamespace(primitive_collision_buffer=SimpleNamespace(
        distance_buffer=torch.zeros((2, 1, 1)),
        sparsity_index_buffer=torch.zeros((2, 1, 1), dtype=torch.uint8)),
        mesh_collision_buffer=None)


class KernelBranch:
    """CPU reproduction of the actual upstream sparse zero early-return.

    Signed query writes both values without marking sparsity. Cost query
    writes penetration, but skips the safe zero whose sparse flag is zero.
    """
    def __init__(self, fail_on_cost=False):
        self.compute_esdf = False
        self._collision_query_buffer = buffer()
        self.fail_on_cost = fail_on_cost
        self.used = []

    def forward(self, spheres, env_query_idx, opt_progress):
        b = self._collision_query_buffer.primitive_collision_buffer
        self.used.append(self._collision_query_buffer)
        if self.compute_esdf:
            b.distance_buffer[:] = torch.tensor([-0.2, 0.03]).reshape(2, 1, 1)
        else:
            if self.fail_on_cost:
                raise RuntimeError('diagnostic_failure')
            b.distance_buffer[1] = .03
            b.sparsity_index_buffer[1] = 1
            if b.sparsity_index_buffer[0].item() != 0:
                b.distance_buffer[0] = 0
                b.sparsity_index_buffer[0] = 0
        return b.distance_buffer.max(dim=-1)[0]


class IsolatedQueryRegression(unittest.TestCase):
    def test_reproduces_old_negative_on_clear_geometry(self):
        primitive = KernelBranch()
        primitive.compute_esdf = True
        primitive.forward(None, None, 1)
        primitive.compute_esdf = False
        result = primitive.forward(None, None, 1)
        self.assertLess(result[0].item(), 0)
        self.assertGreater(result[1].item(), 0)

    def test_fresh_mode_buffers_remove_stale_value_and_keep_penetration(self):
        primitive = KernelBranch()
        original = primitive._collision_query_buffer
        # A previously contaminated optimizer buffer must also be ignored.
        original.primitive_collision_buffer.distance_buffer.fill_(-9)
        values, data = query_both_modes(primitive, None, None, buffer)
        self.assertEqual(values['nonnegative_violation'][0].item(), 0)
        self.assertAlmostEqual(values['nonnegative_violation'][1].item(), .03)
        self.assertLess(values['signed_esdf'][0].item(), 0)
        self.assertIs(primitive._collision_query_buffer, original)
        self.assertFalse(primitive.compute_esdf)
        self.assertIsNot(primitive.used[0], primitive.used[1])
        self.assertIsNot(primitive.used[0], original)
        self.assertIsNot(primitive.used[1], original)
        self.assertTrue(torch.all(original.primitive_collision_buffer.distance_buffer == -9))
        self.assertEqual(data['primitive_nonnegative_violation_per_sphere'][0].item(), 0)

    def test_exception_restores_flag_and_original_optimizer_buffer(self):
        primitive = KernelBranch(fail_on_cost=True)
        primitive.compute_esdf = True
        original = primitive._collision_query_buffer
        with self.assertRaisesRegex(RuntimeError, 'diagnostic_failure'):
            query_both_modes(primitive, None, None, buffer)
        self.assertTrue(primitive.compute_esdf)
        self.assertIs(primitive._collision_query_buffer, original)


if __name__ == '__main__':
    unittest.main()
