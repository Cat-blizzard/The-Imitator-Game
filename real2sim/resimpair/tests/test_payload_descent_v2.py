import unittest
from types import SimpleNamespace
from unittest.mock import patch
from resimpair_pickplace import whole_path_motion_v1 as original
from resimpair_pickplace.payload_descent_v2 import install


class ConditionalDescentTests(unittest.TestCase):
    def test_existing_collision_rejection_is_never_replaced(self):
        def rejection(*a, **k):
            raise RuntimeError('independent_collision')
        with patch.object(original.WholePathMotion, 'plan', rejection):
            with patch.object(original, 'geometric_transport') as alternative:
                with install(), self.assertRaisesRegex(RuntimeError, 'independent_collision'):
                    original.WholePathMotion.plan(SimpleNamespace(), None, 'release_above_rim')
                alternative.assert_not_called()
            self.assertIs(original.WholePathMotion.plan, rejection)

    def test_preview_and_unrelated_stages_return_exact_original_path(self):
        path = {'position': object(), 'guard_evidence': 'retained'}
        def retained(*a, **k):
            return path
        with patch.object(original.WholePathMotion, 'plan', retained):
            with install():
                for stage, preview in [('release_above_rim', True), ('lift', False)]:
                    result = original.WholePathMotion.plan(
                        SimpleNamespace(use_attach=True, preview=preview), None, stage)
                    self.assertIs(result, path)
            self.assertIs(original.WholePathMotion.plan, retained)


if __name__ == '__main__':
    unittest.main()
