"""Real g00 preparation plus the actual independently observed g00/g01 splice."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from data_generation_methods.r2h_pipeline import actor_geometry_preparation_v3 as m
from data_generation_methods.r2h_pipeline.batch_pipeline import binding
from data_generation_methods.r2h_pipeline.common import read_json, write_json

ROOT = Path('/mnt/sdg/datasets/real_sim_human_pipeline/harness_20261021_v3/actor_geometry')
LEFT = ROOT/'left_hand_candidate'
FOREIGN = ROOT/'left_hand_candidate_validation/preparation/preparation.json'


class PreparationBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.tmp.name)
        cls.config = read_json(LEFT/'public_config.json')
        cls.state = m.prepare(LEFT/'geometry_report.json', cls.directory/'prepared', cls.config)
        cls.foreign = read_json(FOREIGN)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def assert_before_budget_rejected(self, state, regex):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'preparation.json'; write_json(path, state)
            with patch.object(m.geometry.unified, '_campaign_budgeted_call') as paid:
                with self.assertRaisesRegex(ValueError, regex):
                    m.execute(path, self.config, Path(tmp)/'absent_budget.json')
                paid.assert_not_called()

    def test_positive_actual_zero_budget_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            budget = {'schema': 'r2h_unified_budget_v1', 'campaign_id': 'v3-preparation-positive-test',
                      'ledger_path': str(tmp/'ledger'),
                      'cumulative_call_limits': {'image': 0, 'video': 0, 'review': 0},
                      'max_concurrent': 1, 'timeout_seconds': 600}
            write_json(tmp/'budget.json', budget)
            with patch.object(m.geometry.unified, 'CAMPAIGN_REGISTRY', tmp/'registry'):
                with self.assertRaisesRegex(RuntimeError, 'image cumulative campaign budget exhausted'):
                    m.execute(self.directory/'prepared/preparation.json', self.config, tmp/'budget.json')
            self.assertTrue((tmp/'ledger/budget.json').exists())
            self.assertEqual(list((tmp/'ledger').rglob('call_started.json')), [])

    def test_actual_cross_source_inner_wrapper_splice(self):
        state = deepcopy(self.state)
        state['qualified_wrapper'] = self.foreign['qualified_wrapper']
        self.assert_before_budget_rejected(state, 'Outer and inner geometry differ')

    def test_outer_executable_package_splice(self):
        state = deepcopy(self.state)
        state['executable_unified_package'] = self.foreign['executable_unified_package']
        self.assert_before_budget_rejected(state, 'Outer executable package differs')

    def test_outer_actual_request_splice(self):
        state = deepcopy(self.state)
        inner = read_json(self.foreign['qualified_wrapper']['path'])
        state['actual_initial_request'] = inner['request']
        self.assert_before_budget_rejected(state, 'Outer actual request differs')

    def test_same_source_other_rig_inner_geometry(self):
        state = deepcopy(self.state)
        inner = read_json(state['qualified_wrapper']['path'])
        inner['geometry'] = binding(ROOT/'native_camera_probe/grid/lean_45_waist_+05cm/report.json')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'inner.json'; write_json(path, inner)
            state['qualified_wrapper'] = binding(path)
            self.assert_before_budget_rejected(state, 'Outer and inner geometry differ')

    def test_diagnostic_reference_and_display_mutations(self):
        mutations = {
            'geometry_report': binding(ROOT/'native_camera_probe/grid/lean_45_waist_+05cm/report.json'),
            'source_image': {'path': '/unrelated/image.png', 'sha256': 'a'*64},
            'model': 'another-model', 'prompt': 'A different same-source instruction',
            'initial_conditions': {'camera_id': 'opposite'},
        }
        for field, value in mutations.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                state = deepcopy(self.state)
                diagnostic = read_json(state['diagnostic_request_spec']['path'])
                diagnostic[field] = value
                path = Path(tmp)/'diagnostic.json'; write_json(path, diagnostic)
                state['diagnostic_request_spec'] = binding(path)
                self.assert_before_budget_rejected(state, 'Diagnostic')

    def test_changed_bound_implementation(self):
        state = deepcopy(self.state)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'implementation.py'; path.write_text('different code\n')
            state['implementation'] = binding(path)
            self.assert_before_budget_rejected(state, 'implementation changed')

    def test_stale_diagnostic_sha(self):
        state = deepcopy(self.state)
        state['diagnostic_request_spec']['sha256'] = '0'*64
        self.assert_before_budget_rejected(state, 'SHA256|sha256|hash')

    def test_coordinated_diagnostic_and_actual_prompt_mutation(self):
        # Isolated mutable test preparation; no archived input is edited.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            state = m.prepare(LEFT/'geometry_report.json', tmp/'prepared', self.config)
            diagnostic_path = Path(state['diagnostic_request_spec']['path'])
            diagnostic = read_json(diagnostic_path); diagnostic['prompt'] += '\nChanged operator instruction.'
            diagnostic_path.write_text(json.dumps(diagnostic))
            state['diagnostic_request_spec'] = binding(diagnostic_path)
            request_path = Path(state['actual_initial_request']['path'])
            request = read_json(request_path); request['prompt'] = diagnostic['prompt']
            request_path.write_text(json.dumps(request))
            state['actual_initial_request'] = binding(request_path)
            inner_path = Path(state['qualified_wrapper']['path'])
            inner = read_json(inner_path); inner['request'] = binding(request_path)
            inner_path.write_text(json.dumps(inner))
            state['qualified_wrapper'] = binding(inner_path)
            self.assert_before_budget_rejected(state, 'Diagnostic differs from recomputed')

    def test_rejected_geometry_still_no_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'prepared'
            state = m.prepare(ROOT/'native_camera_probe/grid/lean_45_waist_+05cm/report.json',
                              output, self.config)
            self.assertIsNone(state['executable_unified_package'])
            self.assertIsNone(state['actual_initial_request'])
            self.assertIsNone(state['qualified_wrapper'])
            self.assertEqual(list(output.rglob('package.json')), [])
            self.assert_before_budget_rejected(state, 'Geometry preflight blocked')


if __name__ == '__main__':
    unittest.main()
