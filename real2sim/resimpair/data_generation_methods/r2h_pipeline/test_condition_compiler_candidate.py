"""Real saved-source probes plus isolated provider serialization tests; no paid API."""
from copy import deepcopy
from pathlib import Path
import base64
import io
import tempfile
import unittest
from unittest.mock import patch

from data_generation_methods.r2h_pipeline import condition_compiler_candidate as c
from data_generation_methods.r2h_pipeline import unified_anchor_candidate as u
from data_generation_methods.r2h_pipeline.common import read_json, write_json
from data_generation_methods.r2h_pipeline.providers import GeminiClient, SeedanceClient
from data_generation_methods.r2h_pipeline.sim_state_conditions import compile_conditions

OLD = Path('/mnt/sdg/datasets/real_sim_human_pipeline/harness_20261021_v2/unified_anchor/development/package.json')
MANIFEST = '/mnt/sdg/datasets/real_sim_human_pipeline/controlled_20261020e/experiment.json'
HIDDEN = '/mnt/sdg/datasets/real_sim_human_pipeline/harness_20261021_v3/hidden_conditions/development_05/hidden_conditions.json'
GEOMETRY_PROBE = Path('/mnt/sdg/datasets/real_sim_human_pipeline/optimization_20261021_180115/conditions/rig_consumption_v1')


@unittest.skipUnless(OLD.exists(), 'Archived qualified source is not installed')
class ConditionCompilerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        old = read_json(OLD)
        packet = read_json(old['inputs']['conditions']['path'])
        compile_conditions(packet['bindings']['receipt']['path'], packet['bindings']['verification']['path'],
                           packet['selected_camera'], cls.root/'source', partition=packet['experiment_partition'])
        cls.config = {'anchor_model': 'gemini-nano-banana-2.1', 'condition_compiler': c.VERSION,
                      'inactive_hand_visibility': 'may_remain_out_of_frame',
                      'gemini': {'model': 'gemini-3.1-pro-preview', 'api_key_env': 'CONDITION_OFFLINE_TEST_KEY'},
                      'seedance': {'model': 'doubao-seedance-2-5-260628'}}
        u.prepare_unified(cls.root/'source/conditions.json', old['inputs']['execution_input']['path'],
                          old['inputs']['placement']['path'], cls.root/'package', cls.config)
        cls.package = cls.root/'package/package.json'
        cls.data = u.validate_package(cls.package)
        cls.compiled = c.compile_unified(cls.data, inactive_policy='may_remain_out_of_frame')

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_initial_does_not_consume_later_state_or_event_text(self):
        data = deepcopy(self.data)
        data['packet']['state_delta']['edit_instruction'] = 'FUTURE_ONLY_SENTINEL'
        data['plan']['event_assignments'][0]['role'] = 'FUTURE_EVENT_SENTINEL'
        compiled = c.compile_unified(data, inactive_policy='may_remain_out_of_frame')
        self.assertEqual(compiled['stages']['initial'], self.compiled['stages']['initial'])
        self.assertIn('FUTURE_ONLY_SENTINEL', compiled['stages']['terminal']['prompt'])
        self.assertNotIn('FUTURE_EVENT_SENTINEL', compiled['stages']['initial']['prompt'])

    def test_same_operator_and_event_id_order_are_required(self):
        for mutation in ('owner', 'event_order', 'event_id'):
            data = deepcopy(self.data)
            if mutation == 'owner':
                data['plan']['actor']['hands']['left']['owner_id'] = 'operator_2'
            elif mutation == 'event_order':
                data['plan']['event_assignments'].reverse()
            else:
                data['plan']['event_assignments'][0]['event_id'] = 'another_task'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                c.compile_unified(data)

    def test_each_event_hand_is_really_in_model_text(self):
        data = deepcopy(self.data)
        data['plan']['event_assignments'][1]['hands'] = ['left']
        compiled = c.compile_unified(data)
        prompt = compiled['stages']['video']['prompt']
        self.assertIn('with the left hand, lift', prompt)
        self.assertIn('with the right hand, grasp', prompt)

    def test_supplied_geometry_cannot_be_silently_dropped(self):
        data = deepcopy(self.data)
        data['plan']['actor']['synthetic_geometry'] = {
            'rig': {}, 'geometry_preflight_report': {'path': str(self.root/'missing-geometry.json'), 'sha256': '0'*64}}
        with self.assertRaises((ValueError, FileNotFoundError)):
            c.compile_unified(data)

    @unittest.skipUnless((GEOMETRY_PROBE/'geometry_report.json').exists(), 'Real geometry probe not installed')
    def test_real_rig_summary_and_cross_hand_splice(self):
        from data_generation_methods.r2h_pipeline.batch_pipeline import binding
        package = GEOMETRY_PROBE/'source_package/package.json'
        data = u.validate_package(package)
        data['plan']['actor']['synthetic_geometry'] = {
            'rig': read_json(GEOMETRY_PROBE/'rig.json'),
            'geometry_preflight_report': binding(GEOMETRY_PROBE/'geometry_report.json')}
        request = u._initial_request(package, data, self.config)
        self.assertIn('Lean torso 45 degrees', request['prompt'])
        self.assertIn('left (0.669,0.123), right (0.331,0.123)', request['prompt'])
        self.assertIn('left hand is the working hand', request['prompt'])
        self.assertTrue(request['compiled_stage']['static_actor_geometry']['clears_geometry_preflight'])
        data['plan']['event_assignments'][0]['hands'] = ['right']
        with self.assertRaisesRegex(ValueError, 'event hand assignment differs'):
            c.compile_unified(data)

    def test_real_measured_events_and_state_delta_retained(self):
        compiled = self.compiled
        self.assertEqual(compiled['measured_event_sidecar']['measured_events'], self.data['packet']['measured_events'])
        self.assertIn('source bowl must no longer contain', compiled['stages']['terminal']['prompt'])
        self.assertTrue(compiled['stages']['video']['source_simulator_event_timing_is_not_video_frame_timing'])
        self.assertFalse(compiled['model_receives_sim_keyframes_as_pose_control'])
        self.assertTrue(150 <= compiled['stages']['video']['prompt_stats']['english_whitespace_words'] <= 250)

    def test_actual_initial_request_consumes_candidate_prompt(self):
        request = read_json(self.root/'package/initial_request.json')
        self.assertEqual(request['prompt'], self.compiled['stages']['initial']['prompt'])
        legacy = u._initial_request(self.package, self.data, {'anchor_model': self.config['anchor_model']})
        self.assertNotIn('condition_compiler', legacy)
        self.assertNotEqual(request['prompt'], legacy['prompt'])

    def test_real_zero_budget_before_credentials_or_transport(self):
        budget = {'schema': 'r2h_unified_budget_v1', 'campaign_id': 'condition-compiler-offline-test',
                  'ledger_path': str(self.root/'ledger'), 'cumulative_call_limits': {'image':0,'video':0,'review':0},
                  'max_concurrent': 1, 'timeout_seconds': 600}
        write_json(self.root/'budget.json', budget)
        with patch.object(u, 'CAMPAIGN_REGISTRY', self.root/'registry'), patch.object(u, 'load_env_file') as credentials:
            with self.assertRaisesRegex(RuntimeError, 'image cumulative campaign budget exhausted'):
                u.execute_initial(self.package, self.config, self.root/'budget.json')
            credentials.assert_not_called()
        self.assertEqual(list((self.root/'ledger').rglob('call_started.json')), [])

    def test_terminal_and_video_still_block_without_independent_reviews(self):
        image = self.data['initial']['frame']['path']
        result = c.prepare_terminal(self.package, image, None, None, self.config, self.root/'terminal.json')
        self.assertEqual(result['status'], 'blocked_initial_reviews')
        with self.assertRaisesRegex(ValueError, 'independent actor/state approval'):
            c.execute_terminal(self.root/'terminal.json', self.config, self.root/'missing-budget.json')
        result = u.prepare_video(self.package, image, None, None, self.config)
        self.assertEqual(result['status'], 'blocked')

    def test_real_three_source_probes_and_foreign_hidden_packet(self):
        for case in ['apple', 'book', 'cabinet']:
            result = c.compile_legacy_probe(MANIFEST, case, hidden_path=HIDDEN if case == 'cabinet' else None)
            self.assertFalse(result['dispatch_allowed'])
            self.assertFalse(result['quality_release_approved'])
            self.assertTrue(150 <= result['stages']['video']['prompt_stats']['english_whitespace_words'] <= 250)
            if case == 'cabinet':
                self.assertEqual(result['hidden_identity_sidecar']['asset_uid'], 'robotwin_073_rubikscube_0')
                self.assertEqual(len(result['stages']['terminal']['reference_images_after_approved_initial']), 1)
                self.assertEqual(result['stages']['initial']['reference_images'], [])
        with self.assertRaisesRegex(ValueError, 'another execution'):
            c.compile_legacy_probe(MANIFEST, 'book', hidden_path=HIDDEN)

    def test_provider_adapter_serializes_exact_text_and_separate_images(self):
        # Transport stub validates existing adapter serialization only; not visual evidence.
        from PIL import Image
        output = io.BytesIO(); Image.new('RGB', (4, 4)).save(output, format='PNG')
        calls = []
        def transport(method, url, **kwargs):
            calls.append(kwargs['json'])
            return {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'inlineData': {
                'mimeType': 'image/png', 'data': base64.b64encode(output.getvalue()).decode()}}]}}]}
        stage = self.compiled['stages']['terminal']
        with patch.dict('os.environ', {'CONDITION_OFFLINE_TEST_KEY': 'offline-test-only'}):
            GeminiClient(self.config['gemini'], transport=transport).generate_image_with_references(
                Path(stage['target']['path']), [Path(self.data['initial']['frame']['path'])],
                stage['prompt'], model=self.config['anchor_model'])
        parts = calls[0]['contents'][0]['parts']
        self.assertEqual(parts[-1]['text'], stage['prompt'])
        self.assertEqual(sum('inlineData' in x for x in parts), 2)
        self.assertEqual(parts[1]['text'], 'SUPPORTING REFERENCE IMAGE 1:')
        payload = {'model': self.config['seedance']['model'], 'content': [
            {'type':'text', 'text':self.compiled['stages']['video']['prompt']},
            {'type':'image_url','role':'first_frame','image_url':{'url':'https://example.invalid/source.png'}}],
            'duration':8, 'ratio':'adaptive'}
        request = SeedanceClient(self.config['seedance']).prepare_submit(payload)
        self.assertEqual(request['json']['content'][0]['text'], self.compiled['stages']['video']['prompt'])


if __name__ == '__main__':
    unittest.main()
