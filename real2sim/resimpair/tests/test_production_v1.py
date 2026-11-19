"""Executable requests must match the audited manifest before GPU submission."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from resimpair_grasp.common import binding, write
from resimpair_tasks.production_v1 import validate_chunks, validate_cohort, qualification_scene_ids, run, require_recorded_execution


class ExecutableCoverageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.rows = []; self.scenes = []; self.chunks = []
        self.requests = []
        for i in range(2):
            request = {'id': f'scene{i}', 'objects': [{'id': 'item', 'source_id': f'source{i}', 'xy': [0, 0]}],
                'tasks': [{'id': 'move', 'placements': [{'object': 'item', 'target': {'kind': 'table', 'xy': [.2, .1]}}]}]}
            self.requests.append(request)
            path = self.root / f'scene{i}.json'; write(path, request)
            self.rows.append({'id': f'scene{i}_move', 'scene_id': f'scene{i}',
                'requested_task_id': 'move', 'source_request': binding(path), 'partition': 'development'})
            self.scenes.append({'scene_id': f'scene{i}', 'seed': 100 + i, 'request': binding(path), 'partition': 'development'})
            c = self.root / f'chunk{i}.json'; write(c, {'schema': 'cached_pickplace_requests_v1', 'scenes': [request]})
            self.chunks.append({'chunk': i, 'requests': binding(c), 'scene_ids': [f'scene{i}'], 'task_count': 1, 'seed_start': 100 + i})

    def test_complete_set_passes(self):
        result = validate_chunks(self.rows, self.chunks, self.scenes)
        self.assertEqual(set(result), {'scene0_move', 'scene1_move'})

    def test_omitted_chunk_rejected(self):
        with self.assertRaisesRegex(ValueError, 'full audited manifest'):
            validate_chunks(self.rows, self.chunks[:1], self.scenes)

    def test_same_id_changed_goal_rejected(self):
        altered = deepcopy(self.requests[0]); altered['tasks'][0]['placements'][0]['target']['xy'][0] = .4
        path = self.root / 'changed_goal.json'; write(path, {'schema': 'cached_pickplace_requests_v1', 'scenes': [altered]})
        chunks = deepcopy(self.chunks); chunks[0]['requests'] = binding(path)
        with self.assertRaisesRegex(ValueError, 'changed executable scene'):
            validate_chunks(self.rows, chunks, self.scenes)

    def test_repeated_chunk_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate executable chunk'):
            validate_chunks(self.rows, self.chunks + self.chunks[:1], self.scenes)

    def test_clutter_seed_change_rejected(self):
        chunks = deepcopy(self.chunks); chunks[0]['seed_start'] += 1
        with self.assertRaisesRegex(ValueError, 'clutter seed'):
            validate_chunks(self.rows, chunks, self.scenes)

    def test_false_heldout_metadata_rejected(self):
        scenes = deepcopy(self.scenes); scenes[0]['partition'] = 'heldout'
        with self.assertRaisesRegex(ValueError, 'metadata partition'):
            validate_chunks(self.rows, self.chunks, scenes)

    def test_missing_cohort_case_rejected(self):
        expected = validate_chunks(self.rows, self.chunks, self.scenes)
        campaign = self.root / 'campaign'; write(campaign / 'cohort.json', {'cases': []})
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            validate_cohort(campaign, self.chunks[0], expected)

    def test_cohort_request_substitution_rejected(self):
        expected = validate_chunks(self.rows, self.chunks, self.scenes)
        campaign = self.root / 'campaign'
        write(campaign / 'cohort.json', {'cases': [{'id': 'scene0_move', 'execution_id': 'scene0_move',
            'requested_task_id': 'move', 'source_request': self.rows[1]['source_request']}]})
        with self.assertRaisesRegex(ValueError, 'source request changed'):
            validate_cohort(campaign, self.chunks[0], expected)

    def test_qualification_includes_both_partitions(self):
        scenes = [{'scene_id': f'scene{i}', 'partition': 'heldout' if i > 6 else 'development'} for i in range(10)]
        self.assertEqual(qualification_scene_ids(scenes), ['scene0', 'scene1', 'scene7', 'scene8'])

    def test_infrastructure_stop_cannot_silently_resume(self):
        root = self.root / 'stopped'; write(root / 'infrastructure_stop.json', {'reason': 'no_bundle'})
        with self.assertRaisesRegex(RuntimeError, 'infrastructure failure'): run(root)

    def test_existing_terminal_without_stop_receipt_is_also_blocked(self):
        root = self.root / 'interrupted'; execution = root / 'executions' / 'a'
        write(execution / 'terminal.json', {'physical_attempts': 1})
        with self.assertRaisesRegex(RuntimeError, 'Infrastructure failure'):
            require_recorded_execution(root, {'id': 'a', 'eligible': True}, execution)
        self.assertTrue((root / 'infrastructure_stop.json').exists())

    def test_geometry_rejection_does_not_need_a_bundle(self):
        root = self.root / 'preflight'; execution = root / 'executions' / 'a'
        write(execution / 'terminal.json', {'physical_attempts': 0})
        require_recorded_execution(root, {'id': 'a', 'eligible': False}, execution)
        self.assertFalse((root / 'infrastructure_stop.json').exists())


if __name__ == '__main__': unittest.main()
