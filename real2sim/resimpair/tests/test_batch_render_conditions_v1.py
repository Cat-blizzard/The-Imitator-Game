import unittest
import numpy as np
from resimpair_tasks.batch_render_conditions_v1 import keyframes


class EventFrames(unittest.TestCase):
    def setUp(self):
        self.bundle = {'object_ids': ['object_1'], 'task_intent': {'operations': [{'arm': 1, 'object': 'object_1'}]}}
        self.arrays = {'grasp': np.zeros((6, 2, 1), dtype=bool), 'object_states': np.zeros((6, 1, 13))}
        self.arrays['grasp'][2:4, 1, 0] = True
        self.arrays['object_states'][3:5, 0, 2] = .04

    def test_real_contact_and_lift_joint_evidence(self):
        indices, evidence = keyframes(self.bundle, self.arrays)
        self.assertEqual(indices, [0, 3, 5])
        self.assertEqual(evidence['arm'], 1)

    def test_no_arbitrary_midpoint_or_multievent_claim(self):
        self.arrays['grasp'][:] = False
        with self.assertRaises(ValueError): keyframes(self.bundle, self.arrays)
        self.bundle['task_intent']['operations'] *= 2
        with self.assertRaisesRegex(ValueError, 'multi_event'): keyframes(self.bundle, self.arrays)


if __name__ == '__main__': unittest.main()
