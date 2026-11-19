"""Actual CLI transport and allowlist boundaries without physical execution."""
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from resimpair_scene.api import MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES, call

ROOT = Path(__file__).resolve().parents[1]


class SceneTransportTests(unittest.TestCase):
    def serve(self, payload):
        result = subprocess.run([sys.executable, '-m', 'resimpair_scene', 'serve'],
                                input=payload, text=True, capture_output=True, cwd=ROOT, check=True)
        return [json.loads(x) for x in result.stdout.splitlines()]

    def test_capabilities_and_errors_preserve_request_identity(self):
        rows = self.serve('\n'.join([
            json.dumps({'id': 1, 'method': 'execution_profiles'}),
            json.dumps({'id': 2, 'method': '__import__', 'params': {'name': 'os'}}),
            json.dumps({'id': 3, 'method': 'scene_measure', 'params': []}),
            json.dumps({'id': 4, 'method': 'execution_profiles', 'params': {'run': True}}),
            json.dumps({'id': 5, 'method': 'execution_profiles'})]) + '\n')
        self.assertEqual([x['id'] for x in rows], [1, 2, 3, 4, 5])
        self.assertEqual(rows[0]['result']['default_profile'], 'checked_withdrawal')
        for row in rows[1:4]:
            self.assertIn('error', row)
        self.assertIn('result', rows[-1])

    def test_oversized_request_is_drained_before_next_request(self):
        rows = self.serve('x' * (2**20 + 3000) + '\n' + json.dumps(
            {'id': 'next', 'method': 'execution_profiles'}) + '\n')
        self.assertEqual(len(rows), 2)
        self.assertIn('request_exceeds', rows[0]['error']['message'])
        self.assertEqual(rows[1]['id'], 'next')

    def test_response_limit(self):
        with patch('resimpair_scene.api._tools', return_value={'large': lambda: 'x' * MAX_RESPONSE_BYTES}):
            with self.assertRaisesRegex(ValueError, 'response_exceeds'):
                call('large')

    def test_python_request_limit_precedes_tool_side_effects(self):
        invoked = []
        def tool(value):
            invoked.append(value)
        with patch('resimpair_scene.api._tools', return_value={'tool': tool}):
            with self.assertRaisesRegex(ValueError, 'request_exceeds'):
                call('tool', {'value': 'x' * MAX_REQUEST_BYTES})
        self.assertFalse(invoked)

    def test_large_mutation_response_preserves_completion_and_recovery_path(self):
        invoked = []
        def tool(cohort, limit):
            invoked.append((cohort, limit))
            return {'large': 'x' * MAX_RESPONSE_BYTES}
        with patch('resimpair_scene.api._tools', return_value={'execution_run': tool}):
            result = call('execution_run', {'cohort': 'saved_cohort', 'limit': 1})
        self.assertEqual(len(invoked), 1)
        self.assertTrue(result['do_not_resubmit'])
        self.assertEqual(result['recover_from']['cohort'], 'saved_cohort')
        self.assertEqual(result['status'], 'operation_returned_result_omitted')

    def test_native_and_python_progress_stays_out_of_protocol_stdout(self):
        program = '''
import os
from unittest.mock import patch
from resimpair_scene.__main__ import invoke,emit
def noisy():
    print('python_progress')
    os.write(1,b'native_progress\\n')
    return {'ok':True}
with patch('resimpair_scene.api._tools',return_value={'noisy':noisy}):
    emit(invoke('noisy',{}))
'''
        result = subprocess.run([sys.executable, '-c', program], capture_output=True,
                                text=True, cwd=ROOT, check=True)
        self.assertEqual(json.loads(result.stdout), {'ok': True})
        self.assertIn('native_progress', result.stderr)
        self.assertIn('python_progress', result.stderr)


if __name__ == '__main__':
    unittest.main()
