import unittest
from resimpair_tasks import container_affordance_consumer_v2 as candidate
from resimpair_tasks.container_affordance_consumer_v1 import measured_affordance
from test_container_affordance_consumer_v1 import fixture


class MetadataTests(unittest.TestCase):
    def test_unbound_caller_height_is_not_evidence_of_actual_task_goal(self):
        row, descriptor = fixture()
        result = candidate.describe_goal(measured_affordance(row, descriptor),
            payload_height_m=.04, nominal_payload_bottom_z_m=.01)
        self.assertEqual('caller_supplied_nominal_goal_unverified', result['bottom_evidence'])
        self.assertFalse(result['place_relation']['into_open_container_allowed'])

    def test_missing_bottom_retains_explicit_floor_assumption(self):
        row, descriptor = fixture()
        result = candidate.describe_goal(measured_affordance(row, descriptor), payload_height_m=.04)
        self.assertEqual('descriptor_floor_assumption', result['bottom_evidence'])


if __name__ == '__main__':
    unittest.main()
