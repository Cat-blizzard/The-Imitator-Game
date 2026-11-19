import unittest
from resimpair_tasks.cudex_seed_repair_campaign_v1 import allocation


class RepairBudget(unittest.TestCase):
    def test_replays_consume_original_slots(self):
        pending = list(range(2657))
        admitted, deferred = allocation(2871, 5528, pending)
        self.assertEqual(len(admitted), 2653)
        self.assertEqual(deferred, pending[-4:])
        self.assertEqual(2871 + 4 + len(admitted), 5528)

    def test_no_hidden_allowance_or_manifest_loss(self):
        for args in [(5527, 5528, [1]), (2871, 5528, [1]), (5530, 5528, [])]:
            with self.assertRaises(ValueError): allocation(*args)


if __name__ == '__main__': unittest.main()
