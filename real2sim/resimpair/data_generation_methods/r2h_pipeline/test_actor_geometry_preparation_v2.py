from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from data_generation_methods.r2h_pipeline import actor_geometry_preparation_v2 as m
from data_generation_methods.r2h_pipeline.common import read_json

REPORT = Path('/mnt/sdg/datasets/real_sim_human_pipeline/harness_20261021_v3/actor_geometry/development_01/lean_00_waist_+00cm/report.json')


class RejectedPreparationTests(unittest.TestCase):
    def test_rejected_rig_has_diagnostic_spec_and_no_executable_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'prepared'
            config = {'anchor_model': 'no_network_test'}
            state = m.prepare(REPORT, output, config)
            self.assertIsNone(state['executable_unified_package'])
            self.assertEqual(list(output.rglob('package.json')), [])
            self.assertEqual(list(output.rglob('initial_request.json')), [])
            spec = read_json(state['diagnostic_request_spec']['path'])
            self.assertFalse(spec['dispatch_allowed'])
            self.assertEqual(spec['initial_conditions']['stage'], 'initial')
            with patch.object(m.geometry.unified, '_campaign_budgeted_call') as paid:
                with self.assertRaisesRegex(ValueError, 'no executable unified package'):
                    m.execute(output/'preparation.json', config, output/'absent_budget.json')
                # Directly invoking the original executor cannot use a package
                # from this rejected preparation because no such artifact exists.
                with self.assertRaises(FileNotFoundError):
                    m.geometry.unified.execute_initial(output/'unified/package.json', config,
                                                         output/'absent_budget.json')
                paid.assert_not_called()


if __name__ == '__main__':
    unittest.main()
