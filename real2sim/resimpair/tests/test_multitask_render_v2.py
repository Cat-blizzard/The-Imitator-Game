"""One audited example per scene and finite ten-job budget; no GPU or physics."""
from fractions import Fraction
import hashlib

import numpy as np
import pytest

from resimpair_grasp.common import write
from resimpair_tasks import multitask_render_v2 as replay
from resimpair_tasks.deferred_render_v1 import NativeClockVideo


def report_rows():
    structures = ['relocate_selected', 'contain_selected', 'orient_selected', 'arrange_group',
                  'collect_group', 'selected_subset', 'split_group', 'clear_then_place']
    return [{'arm': 'candidate', 'case_id': f'scene_{scene:03d}_{name}', 'scene_id': f'multitask_scene_{scene:03d}_fixed',
             'terminal': True, 'task_structure': name, 'operation_count': 1 if i < 3 else 2,
             'dataset_physical_passed': True}
            for scene in range(10) for i, name in enumerate(structures)]


def test_fixed_scene_priorities_cover_ten_distinct_scenes():
    rows = report_rows(); selected, missing = replay.select_sources({'rows': rows})
    assert [r['scene_index'] for r in selected] == list(range(10))
    assert [r['group'] for r in selected] == list(replay.SCENE_PRIORITY_GROUPS)
    assert len({r['audit_row']['scene_id'] for r in selected}) == 10
    assert not any(r['fallback'] for r in selected)
    assert missing == []
    # Selection is keyed by the original scene index, not report ordering.
    shuffled = list(reversed(rows))
    again, _ = replay.select_sources({'rows': shuffled})
    assert [r['audit_row']['case_id'] for r in selected] == [r['audit_row']['case_id'] for r in again]


def test_preference_rejection_uses_fixed_multistep_then_single_step_fallback():
    rows = report_rows()
    rows[7]['dataset_physical_passed'] = False  # scene 0 preferred clear.
    for row in rows[8:16]:
        if row['operation_count'] > 1: row['dataset_physical_passed'] = False
    selected, missing = replay.select_sources({'rows': rows})
    assert selected[0]['group'] == 'collect' and selected[0]['fallback']
    assert selected[0]['selection_reason'] == 'preferred_structure_unavailable_fixed_multi_operation_fallback'
    assert selected[1]['audit_row']['task_structure'] == 'relocate_selected'
    assert selected[1]['group'] == 'single' and selected[1]['fallback']
    assert selected[1]['selection_reason'] == 'no_multi_operation_pass_fixed_single_operation_fallback'
    assert not missing


def test_missing_scene_is_retained_not_replaced_by_another_scene_or_baseline():
    rows = report_rows()
    for row in rows[:8]: row['dataset_physical_passed'] = False
    rows.append({**rows[7], 'arm': 'baseline', 'dataset_physical_passed': True})
    selected, missing = replay.select_sources({'rows': rows})
    assert len(selected) == 9
    assert [r['scene_index'] for r in missing] == [0]
    assert missing[0]['original_requests'] == 8
    assert all(r['scene_index'] != 0 for r in selected)


def test_partial_or_duplicate_request_cohort_cannot_select_examples():
    rows = report_rows(); rows[-1]['terminal'] = False
    with pytest.raises(ValueError, match='80_candidate_requests'):
        replay.select_sources({'rows': rows})
    rows = report_rows(); rows[-1]['case_id'] = rows[-2]['case_id']
    with pytest.raises(ValueError, match='unique_frozen'):
        replay.select_sources({'rows': rows})


def test_single_operation_fallback_still_requires_all_six_native_events():
    task = {'operations': [{'id': 'op_0', 'object': 'object_0', 'arm': 0, 'target': {'support': 'table'}}]}
    events = [{'id': f'op_0.{name}', 'frame': j + 1} for j, name in enumerate(replay.EVENTS)]
    result = replay.event_keyframes(task, events, 8)
    assert len(result['operations_in_measured_time_order']) == 1
    assert result['frame_indices'] == list(range(8))
    with pytest.raises(ValueError, match='all_six'):
        replay.event_keyframes(task, events[:-1], 8)


def prior_budget(tmp_path):
    prior = tmp_path / 'prior/budget/limits.json'
    write(prior, {'historical_actual_render_jobs': 54, 'maximum_lineage_render_jobs_including_history': 60})
    for i in range(4): write(prior.parent / 'reservations' / f'{i}.json', {'attempt': i})
    return prior


def test_ten_job_allowance_preserves_actual58_and_ceiling60_then70(tmp_path):
    prior = prior_budget(tmp_path)
    _, budget = replay.register_budget(tmp_path / 'output', 10, prior, tmp_path / 'registry', tmp_path / 'unused_v1')
    assert budget['historical_actual_render_jobs'] == 58
    assert budget['maximum_attempts'] == 10 and budget['allocation_limit'] == 10
    assert budget['maximum_lineage_render_jobs_including_history'] == 70
    assert budget['maximum_actual_jobs_including_history'] == 68
    assert budget['v1_example_allowance_unallocated_not_carried_forward']
    assert budget['allocation'] == {'screen': 0, 'video': 10}
    assert budget['gpu'] == 2 and budget['minimum_free_gpu_mib'] == 12288
    assert budget['parallel_workers'] == 1 and budget['timeout_seconds']['video'] == 240
    with pytest.raises(ValueError, match='cannot_reset'):
        replay.register_budget(tmp_path / 'other', 1, prior, tmp_path / 'registry', tmp_path / 'unused_v1')


def test_registered_v1_allowance_cannot_be_silently_dropped_from_new_lineage(tmp_path):
    prior = prior_budget(tmp_path)
    write(tmp_path / 'used_v1/registrations/allocated.json', {'allocation_limit': 4})
    with pytest.raises(ValueError, match='v1_example_allowance_was_allocated'):
        replay.register_budget(tmp_path / 'output', 10, prior, tmp_path / 'registry', tmp_path / 'used_v1')
    assert not (tmp_path / 'output').exists()


def test_more_than_ten_sources_cannot_register(tmp_path):
    prior = prior_budget(tmp_path)
    with pytest.raises(ValueError, match='one_to_ten'):
        replay.register_budget(tmp_path / 'output', 11, prior, tmp_path / 'registry', tmp_path / 'unused_v1')


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
    with pytest.raises(ValueError, match='native_clock'):
        replay.decode_keyframes(video, tmp_path / 'wrong_clock', [0, 2, 4], np.arange(5) / 10., 'opposite_high')
