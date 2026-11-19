"""Multi-operation replay selection and clock tests; no GPU render or physics."""
from fractions import Fraction
import hashlib

import numpy as np
import pytest

from resimpair_grasp.common import read, write
from resimpair_tasks import multitask_render_v1 as replay
from resimpair_tasks.deferred_render_v1 import NativeClockVideo


def report_rows():
    structures = ['collect_group', 'split_group', 'clear_then_place', 'arrange_group']
    return [{'arm': 'candidate', 'case_id': f'case_{i}', 'terminal': True,
             'task_structure': structures[i % 4], 'dataset_physical_passed': i >= 4} for i in range(80)]


def test_selection_uses_first_pass_per_structure_in_original_order():
    rows = report_rows(); selected, missing = replay.select_sources({'rows': rows})
    assert [r['audit_row']['case_id'] for r in selected] == ['case_4', 'case_5', 'case_6', 'case_7']
    assert missing == []
    rows[0]['terminal'] = False
    with pytest.raises(ValueError, match='80_candidate_requests'):
        replay.select_sources({'rows': rows})


def test_missing_structure_has_no_fabricated_example():
    rows = report_rows()
    for row in rows:
        if row['task_structure'] == 'clear_then_place': row['dataset_physical_passed'] = False
    selected, missing = replay.select_sources({'rows': rows})
    assert len(selected) == 3 and missing[0]['group'] == 'clear'


def test_each_operation_requires_its_own_six_native_events():
    task = {'operations': [{'id': f'op_{i}', 'object': f'object_{i}', 'arm': i, 'target': {'support': 'table'}} for i in range(2)]}
    events = [{'id': f'op_{i}.{name}', 'frame': i * 10 + j + 1} for i in range(2) for j, name in enumerate(replay.EVENTS)]
    result = replay.event_keyframes(task, events, 20)
    assert len(result['operations_in_measured_time_order']) == 2
    assert result['frame_indices'][0] == 0 and result['frame_indices'][-1] == 19
    assert not result['visibility_or_human_generation_qualification']
    with pytest.raises(ValueError, match='all_six'):
        replay.event_keyframes(task, events[:-1], 20)


def test_render_allocation_retains_60_ceiling_and_counts_actual58(tmp_path):
    prior = tmp_path / 'prior/budget/limits.json'
    write(prior, {'historical_actual_render_jobs': 54, 'maximum_lineage_render_jobs_including_history': 60})
    for i in range(4): write(prior.parent / 'reservations' / f'{i}.json', {'attempt': i})
    receipt, budget = replay.register_budget(tmp_path / 'output', 4, prior, tmp_path / 'registry')
    assert budget['historical_actual_render_jobs'] == 58
    assert budget['maximum_attempts'] == 4 and budget['maximum_lineage_render_jobs_including_history'] == 64
    assert budget['allocation']['screen'] == 0
    with pytest.raises(ValueError, match='cannot_reset'):
        replay.register_budget(tmp_path / 'other', 1, prior, tmp_path / 'registry')


def test_state_mapping_cannot_reorder_or_skip_frames():
    states = np.arange(6, dtype=np.float32).reshape(2, 1, 3)
    source = {'states': states, 'timestamps': np.array([0., .05])}
    errors = dict(state_max_abs=0., qpos_max_abs=0., object_state_max_abs=0., TCP_max_abs=0.)
    mapping = {'env_step_calls': 0, 'all_states_rendered_once': True, 'original_actions_and_timeline_preserved': True,
        'frames': [{'frame': i, 'source_state_index': i, 'cameras': ['robot_side', 'opposite_high'],
                    'source_timestamp_seconds': i * .05,
                    'state_sha256': hashlib.sha256(np.ascontiguousarray(states[i]).tobytes()).hexdigest(),
                    'errors': errors} for i in range(2)]}
    replay.validate_frame_mapping(mapping, source)
    mapping['frames'][1]['source_state_index'] = 0
    with pytest.raises(ValueError, match='mapping_changed'):
        replay.validate_frame_mapping(mapping, source)


def test_keyframes_are_decoded_at_native_video_clock(tmp_path):
    video = tmp_path / 'native.mp4'; writer = NativeClockVideo(video, 16, 16, Fraction(20, 1))
    for i in range(5): writer.append(np.full((16, 16, 3), i * 40, dtype=np.uint8), i)
    writer.close()
    proof = replay.decode_keyframes(video, tmp_path / 'images', [0, 2, 4], np.arange(5) / 20., 'opposite_high')
    assert proof['decoded_frame_count'] == 5
    assert [r['source_state_index'] for r in proof['images']] == [0, 2, 4]
    assert proof['clock_matches_all_source_timestamps']
    with pytest.raises(ValueError, match='native_clock'):
        replay.decode_keyframes(video, tmp_path / 'wrong_clock', [0, 2, 4], np.arange(5) / 10., 'opposite_high')
