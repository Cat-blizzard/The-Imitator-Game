"""Counterexamples to inflated diversity and source-split claims."""
from copy import deepcopy
import hashlib
import json
import math

import pytest

from resimpair_tasks.coverage_audit_v1 import audit


def scene(name='scene_a'):
    return {'id': name, 'objects': [
        {'id': 'bottle', 'source_id': 'gso:bottle', 'xy': [-.25, -.3], 'yaw': .2},
        {'id': 'book', 'source_id': 'abo:book', 'xy': [-.25, .3], 'yaw': 0.}],
        'containers': [
            {'id': 'left_bin', 'source_id': 'gso:bin', 'xy': [.10, -.3]},
            {'id': 'right_bin', 'source_id': 'gso:bin', 'xy': [.10, .3]}],
        'distractors': {'count': 4, 'max_dimension_m': .16},
        'tasks': [{'id': 'collect', 'placements': [
            {'object': 'bottle', 'target': {'kind': 'container', 'container': 'left_bin'}}]}]}


def bound_row(tmp_path, spec, rid, **extra):
    path = tmp_path / (rid + '.json')
    path.write_text(json.dumps(spec))
    return {'id': rid, 'scene_id': spec['id'], 'partition': 'development',
            'mechanism': 'pick_place', 'goal_family': 'collect',
            'source_ids': sorted(o['source_id'] for o in spec['objects']),
            'requested_task_id': spec['tasks'][0]['id'],
            'source_request': {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()},
            **extra}


def codes(report):
    return {e['code'] for e in report['errors']}


def test_one_bound_request_reports_scope_without_claiming_physical_or_language_success(tmp_path):
    r = audit([bound_row(tmp_path, scene(), 'one')], 1)
    assert r['passed']
    assert r['counts']['source_assets'] == 2
    assert r['counts']['manipulated_source_assets'] == 1
    assert r['counts']['unique_relation_structure'] == 1
    assert r['rows'][0]['physical_status'] == r['rows'][0]['human_video_status'] == 'not_evaluated'
    assert r['rows'][0]['semantic_status'] == 'unverified'
    assert r['rows'][0]['source_novelty'] == 'unknown_history'


def test_text_seed_camera_and_family_names_cannot_create_different_tasks(tmp_path):
    a = scene(); b = scene('scene_b')
    b.update(seed=984, camera={'position': [0, 1, 2]}, description='An entirely different name!')
    b['tasks'][0]['instruction'] = 'Another phrasing of the same thing'
    rows = [bound_row(tmp_path, a, 'one'), bound_row(tmp_path, b, 'two', goal_family='packing_new')]
    r = audit(rows, 2)
    assert 'duplicate_quantized_task_request' in codes(r)
    assert r['counts']['declared_goal_families'] == 2
    assert r['counts']['unique_exact_metric'] == r['counts']['unique_asset_goal'] == 1


def test_object_container_and_task_aliases_do_not_inflate_coverage(tmp_path):
    a = scene(); b = scene('scene_b')
    b['objects'][0]['id'] = 'renamed_object'; b['containers'][0]['id'] = 'renamed_bin'
    b['tasks'][0]['id'] = 'renamed_task'
    b['tasks'][0]['placements'][0] = {'object': 'renamed_object', 'target': {'kind': 'container', 'container': 'renamed_bin'}}
    r = audit([bound_row(tmp_path, a, 'one'), bound_row(tmp_path, b, 'two')], 2)
    assert 'duplicate_quantized_task_request' in codes(r)
    assert r['counts']['unique_asset_goal'] == 1


def test_submillimeter_changes_do_not_qualify_as_distinct_instances(tmp_path):
    a = scene(); b = scene('scene_b'); b['objects'][0]['xy'][0] += .0001
    r = audit([bound_row(tmp_path, a, 'one'), bound_row(tmp_path, b, 'two')], 2)
    assert r['counts']['unique_exact_metric'] == 2
    assert r['counts']['unique_quantized_metric'] == 1
    assert 'duplicate_quantized_task_request' in codes(r)


def test_metric_layouts_do_not_become_new_goal_structures(tmp_path):
    a = scene(); b = scene('scene_b')
    for obj in b['objects'] + b['containers']:
        obj['xy'][0] += .05
    r = audit([bound_row(tmp_path, a, 'one'), bound_row(tmp_path, b, 'two')], 2)
    assert r['passed']
    assert r['counts']['unique_quantized_metric'] == 2
    assert r['counts']['unique_asset_goal'] == r['counts']['unique_relation_structure'] == 1
    assert r['counts']['layout_or_metric_variants_of_same_asset_goal'] == 1


def test_different_selected_roles_are_detected_with_same_relation_type(tmp_path):
    a = scene(); b = scene('scene_b'); b['tasks'][0]['placements'][0]['object'] = 'book'
    r = audit([bound_row(tmp_path, a, 'one'), bound_row(tmp_path, b, 'two')], 2)
    assert r['passed']
    assert r['counts']['unique_asset_goal'] == 2
    assert r['counts']['unique_relation_structure'] == 1
    assert r['counts']['manipulated_source_assets'] == 2


def test_collecting_together_and_distributing_have_different_goal_structures(tmp_path):
    a = scene(); a['tasks'][0]['placements'].append({'object': 'book', 'target': {'kind': 'container', 'container': 'left_bin'}})
    b = deepcopy(a); b['id'] = 'scene_b'; b['tasks'][0]['placements'][1]['target']['container'] = 'right_bin'
    r = audit([bound_row(tmp_path, a, 'one'), bound_row(tmp_path, b, 'two')], 2)
    assert r['passed']
    assert r['counts']['unique_relation_structure'] == 2


def test_declared_order_is_bound_to_object_identity(tmp_path):
    a = scene(); a['tasks'][0]['placements'].append({'object': 'book', 'target': {'kind': 'container', 'container': 'right_bin'}})
    b = deepcopy(a); b['id'] = 'scene_b'; b['tasks'][0]['placements'].reverse()
    r = audit([bound_row(tmp_path, a, 'one'), bound_row(tmp_path, b, 'two')], 2)
    assert r['passed']
    assert r['counts']['unique_asset_goal'] == 2
    assert r['counts']['contact_mechanisms'] == 1


def test_unselected_source_cannot_cross_data_partitions(tmp_path):
    a = scene(); b = scene('scene_b'); b['objects'][0]['source_id'] = 'gso:another_bottle'
    r = audit([bound_row(tmp_path, a, 'one'), bound_row(tmp_path, b, 'two', partition='validation')], 2)
    assert 'source_partition_leakage' in codes(r)
    assert r['source_partition_leaks'] == {'abo:book': ['development', 'validation']}


def test_declared_sources_cannot_hide_actual_request_sources(tmp_path):
    row = bound_row(tmp_path, scene(), 'one', source_ids=['fake:asset'])
    assert 'source_ids_differ_from_bound_request' in codes(audit([row], 1))


def test_binding_changes_are_detected(tmp_path):
    row = bound_row(tmp_path, scene(), 'one')
    (tmp_path / 'one.json').write_text('{}')
    assert 'source_request_binding_changed' in codes(audit([row], 1))


def test_known_sources_cannot_be_declared_globally_unseen(tmp_path):
    row = bound_row(tmp_path, scene(), 'one', partition='heldout_asset', source_novelty='global_unseen_asset')
    assert 'unsupported_global_unseen_source_claim' in codes(audit([row], 1, ['gso:bottle']))
    row.pop('source_novelty')
    r = audit([row], 1, ['gso:bottle'])
    assert r['passed']
    assert r['rows'][0]['source_novelty'] == 'contains_previously_seen'


def test_unseen_claim_requires_history_even_when_no_known_match(tmp_path):
    row = bound_row(tmp_path, scene(), 'one', source_novelty='global_unseen_asset')
    assert 'unsupported_global_unseen_source_claim' in codes(audit([row], 1))


def test_scene_id_cannot_hide_different_initial_states(tmp_path):
    a = scene(); b = scene(); b['objects'][0]['xy'][0] += .02
    r = audit([bound_row(tmp_path, a, 'one'), bound_row(tmp_path, b, 'two')], 2)
    assert 'same_scene_id_has_different_initial_request' in codes(r)


def test_expected_count_and_unique_ids_are_hard_requirements(tmp_path):
    row = bound_row(tmp_path, scene(), 'one')
    assert 'row_count_mismatch' in codes(audit([row], 500))
    assert 'duplicate_row_id' in codes(audit([row, deepcopy(row)], 2))


def test_whole_turn_yaw_is_not_a_new_task(tmp_path):
    a = scene(); b = scene('scene_b'); a['objects'][0]['yaw'] = 0.; b['objects'][0]['yaw'] = 2 * math.pi
    r = audit([bound_row(tmp_path, a, 'one'), bound_row(tmp_path, b, 'two')], 2)
    assert r['counts']['unique_quantized_metric'] == 1


@pytest.mark.parametrize('problem,code', [('object', 'unknown_placement_object'), ('container', 'unknown_target_container')])
def test_unknown_roles_are_rejected(tmp_path, problem, code):
    spec = scene()
    if problem == 'object': spec['tasks'][0]['placements'][0]['object'] = 'missing'
    else: spec['tasks'][0]['placements'][0]['target']['container'] = 'missing'
    r = audit([bound_row(tmp_path, spec, 'one')], 1)
    assert code in codes(r)


def test_missing_binding_is_unknown_only_when_explicitly_allowed():
    s = scene()
    row = {'id': 'one', 'scene_id': s['id'], 'partition': 'development', 'mechanism': 'pick_place',
           'goal_family': 'collect', 'source_ids': ['gso:bottle', 'abo:book'],
           'requested_task_id': 'collect', 'request': s}
    assert 'source_request_binding_required' in codes(audit([row], 1))
    report = audit([row], 1, require_bindings=False)
    assert report['passed'] and not report['rows'][0]['source_request_bound']


def test_new_asset_does_not_automatically_create_new_pose_layout(tmp_path):
    a = scene(); b = scene('scene_b'); b['objects'][0]['source_id'] = 'gso:another_bottle'
    r = audit([bound_row(tmp_path, a, 'one'), bound_row(tmp_path, b, 'two')], 2)
    assert r['passed']
    assert r['counts']['unique_quantized_layout'] == 2
    assert r['counts']['unique_quantized_pose_layout'] == 1


def test_nonfinite_request_is_a_saved_rejection_not_a_fingerprint(tmp_path):
    s = scene(); s['objects'][0]['xy'][0] = float('nan')
    r = audit([bound_row(tmp_path, s, 'one')], 1)
    assert not r['passed']
    assert 'nonfinite_or_invalid_metric_value' in codes(r)
    json.dumps(r, allow_nan=False)


def test_malformed_placement_is_rejected_without_aborting_other_rows(tmp_path):
    s = scene(); s['tasks'][0]['placements'] = ['not_an_operation']
    r = audit([bound_row(tmp_path, s, 'bad'), bound_row(tmp_path, scene('scene_b'), 'good')], 2)
    assert not r['passed'] and r['counts']['valid_rows'] == 1
    assert 'invalid_placement_or_target' in codes(r)
