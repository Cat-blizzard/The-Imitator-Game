import unittest
import numpy as np

from resimpair_pickplace.seed_compatibility import simulator_seed, seed_receipt


class SimulatorSeedCompatibilityTests(unittest.TestCase):
    def test_frozen_large_seed_is_accepted_by_actual_legacy_rng(self):
        source = 20261027500
        with self.assertRaises(ValueError): np.random.RandomState(source)
        receipt = seed_receipt(source)
        np.random.RandomState(receipt['simulator_reset_seed'])
        self.assertEqual(receipt['source_scene_seed'], source)
        self.assertTrue(receipt['layout_objects_targets_and_distractors_unchanged'])

    def test_all_previous_valid_seeds_preserve_rng_sequences(self):
        for source in [0, 20261027, 2026102708, 2026102780, 2**32 - 1]:
            np.testing.assert_array_equal(np.random.RandomState(source).normal(size=20),
                np.random.RandomState(simulator_seed(source)).normal(size=20))

    def test_invalid_seeds_fail_explicitly(self):
        for source in [-1, True, 1.2, '123', None]:
            with self.subTest(source=source), self.assertRaises(ValueError): simulator_seed(source)


if __name__ == '__main__': unittest.main()
