"""Read-only contract regressions, including four retained native failures.

No simulator or paid service is called. The four artifact paths are explicit test
inputs, not a discovery convention of the observation API.
"""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from resimpair_scene import observe


ROOT = Path(__file__).resolve().parents[1]
EXECUTIONS = ROOT / 'runs/production_integration_20261030_v1/final_metric_groups_closure_r1/executions'
AUDITS = ROOT / 'validation/paired_assets_20261030/independent/final_metric_groups_closure_r1'
CASES = {
    'container': ('metric_group_63301d78de427f1d55784def45a53738_distribute_groups',
                  'hand_vs_container', 'transfer_6', 'placement_hand_clearance'),
    'placed_object': ('metric_group_b61e71434d87d8089768f9d33ed8a237_distribute_groups',
                      'hand_vs_placed_object', 'transfer_2', 'placement_slot_or_order'),
    'thin_object': ('metric_group_e5bd1e5109221146d7e20c6829fe1165_distribute_groups',
                    'grasp_lost_during_lift', 'transfer_2', 'grasp_contact_and_attachment_control'),
    'transient': ('metric_group_35f5f2b23d9520e3f311c57adee97856_distribute_groups',
                  'release_region_violation', 'transfer_4', 'payload_pose_during_descent'),
}


def actual(name):
    case, *_ = CASES[name]
    execution, audit = EXECUTIONS / case, AUDITS / case / 'independent_result.json'
    if not execution.is_dir() or not audit.is_file():
        pytest.skip('retained local execution/audit not installed; never synthesize a real regression')
    return execution, audit


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(paths):
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns, digest(p)) for p in paths if p.is_file()}


@pytest.mark.parametrize('name', CASES)
def test_real_failure_has_specific_operation_route_and_preserves_evidence(name):
    execution, audit = actual(name)
    paths = [execution / x for x in ('execution_bundle.json', 'effective_scene.json', 'effective_task.json',
                                     'measurements.npz', 'terminal.json', 'worker_result.json', 'planning_failure.json')]
    paths += [audit, audit.parent / 'new_event_audit.json']
    before = snapshot(paths)
    result = observe.run_diagnose(execution, independent_audit=audit)
    _, expected_type, expected_operation, expected_route = CASES[name]
    finding = result['findings'][0]
    assert (finding['type'], finding['operation'], finding['route']) == (expected_type, expected_operation, expected_route)
    assert result['failed_operation'] == expected_operation
    assert result['status']['worker_status'] == 'completed'
    assert result['status']['physical']['status'] == 'failed'
    assert result['status']['human_pair']['status'] == 'unknown'
    assert result['new_physics'] == result['GPU_calls'] == result['API_calls'] == 0
    assert result['causal_proof_claimed'] is False
    for ref in finding['evidence']:
        assert Path(ref['path']).is_file() and digest(Path(ref['path'])) == ref['sha256']
    json.dumps(result, allow_nan=False)
    assert snapshot(paths) == before


def test_grasp_was_present_at_close_then_lost_at_first_lift():
    execution, audit = actual('thin_object')
    result = observe.run_diagnose(execution, independent_audit=audit)['findings'][0]
    values = result['observations']
    assert values['close_end_native_grasp'] is True
    assert values['grasp_loss_frames'] == [966]
    assert values['object_XY_change_at_first_loss_m'] == pytest.approx(0.01506329, abs=1e-7)
    assert result['root_cause'].startswith('unknown')
    measured = []
    for frame in (964, 965, 966):
        output = observe.run_observe(execution, frame=frame, independent_audit=audit)
        obj = next(x for x in output['frame_samples'][0]['objects'] if x['id'] == 'object_4')
        measured.append(obj['native_grasp_by_arm'][0])
    assert measured == [True, True, False]
    other = observe.run_diagnose(execution, operation='transfer_0', independent_audit=audit)
    assert all(x['type'] != 'grasp_lost_during_lift' for x in other['findings'])


def test_final_goal_pass_does_not_erase_original_transient_failure():
    execution, _ = actual('transient')
    result = observe.run_diagnose(execution, independent_audit=AUDITS / 'FINAL_REVIEW.json')
    assert result['status']['physical']['subchecks']['original_requested_goal'] is True
    assert result['status']['physical']['subchecks']['task_events'] is False
    values = result['findings'][0]['observations']
    assert values['original_XY_tolerance_m'] == 0.015
    assert values['maximum_XY_error_m'] == pytest.approx(0.016036692, abs=1e-8)
    assert values['maximum_error_frame'] == 2031
    assert values['first_failure_frame'] == 2029


def test_no_independent_audit_never_means_physical_pass():
    execution, _ = actual('transient')
    result = observe.run_status(execution)
    assert result['worker_completed'] is True
    assert result['physical']['status'] == 'unknown'
    assert result['physical']['reason'] == 'no_bound_independent_audit'
    assert result['RGB']['status'] == result['human_pair']['status'] == 'unknown'


def test_unrelated_audit_cannot_qualify_execution():
    execution, _ = actual('container')
    _, wrong_audit = actual('transient')
    result = observe.run_status(execution, independent_audit=wrong_audit)
    assert result['physical']['status'] == 'unknown'
    assert 'invalid_or_unavailable_audit' in result['physical']['reason']


def test_reservation_without_terminal_remains_unknown_even_with_a_pass_record(tmp_path):
    execution = tmp_path / 'reserved_case'; execution.mkdir()
    bundle = write_json(execution / 'execution_bundle.json', {'scene': {}, 'task_intent': {}})
    write_json(execution / 'reservation.json', {'pid': 0})
    audit = write_json(tmp_path / 'independent_result.json', {
        'case_id': execution.name, 'physical_qualified': True,
        'source_files_unchanged_during_review': True, 'audit_error': None,
        'source_execution': {'path': str(bundle), 'sha256': digest(bundle)},
    })
    result = observe.run_status(execution, independent_audit=audit)
    assert result['worker_status'] == 'unknown_reserved_without_terminal'
    assert result['physical']['status'] == 'unknown'
    assert result['automatic_resubmission_allowed'] is False
    assert observe.run_observe(execution)['frame_samples'] == []


def test_self_asserted_audit_never_qualifies_even_with_valid_source_hash(tmp_path):
    execution = tmp_path / 'case'; execution.mkdir()
    bundle = write_json(execution / 'execution_bundle.json', {})
    write_json(execution / 'terminal.json', {'returncode': 0})
    source = {'path': str(bundle)}
    record = {'case_id': execution.name, 'physical_qualified': True,
              'source_files_unchanged_during_review': True, 'audit_error': None, 'source_execution': source}
    audit = write_json(tmp_path / 'independent_result.json', record)
    assert observe.run_status(execution, independent_audit=audit)['physical']['status'] == 'unknown'
    source['sha256'] = '0' * 64
    write_json(audit, record)
    assert observe.run_status(execution, independent_audit=audit)['physical']['status'] == 'unknown'
    source['sha256'] = digest(bundle)
    write_json(audit, record)
    result = observe.run_status(execution, independent_audit=audit)
    assert result['physical']['status'] == 'unknown'
    assert result['RGB']['status'] == result['human_pair']['status'] == 'unknown'


def test_modified_true_flag_is_rejected_despite_valid_protocol_and_source_chain(tmp_path):
    execution, audit = actual('container')
    protocol = json.loads((AUDITS / 'PROTOCOL.json').read_text())
    final = json.loads((AUDITS / 'FINAL_REVIEW.json').read_text())
    fake_protocol = write_json(tmp_path / 'PROTOCOL.json', protocol)
    final['protocol'] = {'path': str(fake_protocol), 'sha256': digest(fake_protocol)}
    write_json(tmp_path / 'FINAL_REVIEW.json', final)
    changed = json.loads(audit.read_text())
    assert changed['physical_qualified'] is False
    changed['physical_qualified'] = True
    fake_leaf = write_json(tmp_path / execution.name / 'independent_result.json', changed)
    result = observe.run_status(execution, independent_audit=fake_leaf)
    assert result['physical']['status'] == 'unknown'
    assert 'independent_leaf_and_summary_disagree' in result['physical']['reason']


def test_retained_success_requires_full_chain_and_does_not_imply_RGB_or_human_pass():
    if not (AUDITS / 'FINAL_REVIEW.json').is_file():
        pytest.skip('retained local independent cohort not installed')
    final = json.loads((AUDITS / 'FINAL_REVIEW.json').read_text())
    row = next(x for x in final['rows'] if x['physical_qualified'] is True)
    execution = EXECUTIONS / row['case_id']
    result = observe.run_status(execution, independent_audit=AUDITS / 'FINAL_REVIEW.json')
    assert result['physical']['status'] == 'passed'
    assert result['physical']['provenance_chain'] == 'verified_metric_cohort_v1'
    assert result['RGB']['status'] == result['human_pair']['status'] == 'unknown'


def test_real_dimensions_mass_and_support_are_qualified_by_evidence():
    execution, _ = actual('transient')
    result = observe.scene_measure(execution, object_ids=['object_0', 'container_0'])
    obj, container = result['objects']
    assert result['frame'] == 2218
    assert obj['geometry_basis'] == 'saved_native_collision_vertices'
    assert obj['pose_basis'] == 'measured_native_actor_pose'
    assert obj['actor_local_collision_dimensions_m'] == pytest.approx([0.0355705097, 0.0352077801, 0.0336927790], abs=1e-9)
    assert obj['mass']['native_kg'] == pytest.approx(0.00872527528554)
    assert obj['mass']['actual_real_object_mass'] == 'unknown'
    support = container['support_region']
    assert support['kind'] == 'qualified_floor_disk'
    assert support['full_inner_cavity_bounds'] == 'unknown'
    assert support['outer_bounds_are_not_inner_space'] is True
    assert support['radius_m'] == pytest.approx(0.09169645775)
    assert Path(obj['geometry_evidence']['path']).name == 'native_geometry.json'
    assert Path(container['geometry_evidence']['path']).name == 'native_containers.json'
    assert container['geometry_is_visual_or_real_world_truth'] is False
    json.dumps(result, allow_nan=False)


def test_declared_scene_box_is_explicitly_approximate_and_not_a_cavity(tmp_path):
    scene = write_json(tmp_path / 'scene.json', {
        'id': 'declared', 'objects': [{'id': 'box', 'position': [1, 2, 3], 'yaw': np.pi/2,
                                     'half_size': [1, 2, 0.5]}], 'supports': [],
    })
    result = observe.scene_measure(scene)
    obj = result['objects'][0]
    assert result['frame'] is None
    assert obj['geometry_basis'] == 'descriptor_or_declared_box_approximation'
    assert np.allclose(obj['world_bounds_m'], [[-1, 1, 2.5], [3, 3, 3.5]])
    assert obj['support_region']['status'] == 'unknown'
    assert obj['mass']['native_kg'] == 'unknown'


def test_operation_samples_and_actions_keep_original_timeline():
    execution, audit = actual('transient')
    result = observe.run_observe(execution, operation='transfer_4', max_frames=3, independent_audit=audit)
    record, frames = result['execution_record'], result['frame_samples']
    assert len(frames) == 3
    assert frames[0]['frame'] == record['first_frame']
    assert frames[-1]['frame'] == record['last_frame']
    for frame in frames:
        assert frame['action_maps_state_interval'] == [frame['frame']-1, frame['frame']]
        assert frame['timestamp_seconds'] == pytest.approx(frame['frame'] / 20)
        assert len(frame['incoming_native_action']) == 16


def test_measurements_cache_avoids_repeated_decompression_and_is_readonly():
    execution, _ = actual('thin_object')
    observe._npz_cached.cache_clear()
    with patch.object(observe.np, 'load', wraps=np.load) as loader:
        first = observe.run_observe(execution, frame=964)
        second = observe.run_observe(execution, frame=965)
        assert loader.call_count == 1
    assert first['frame_samples'][0]['frame'] != second['frame_samples'][0]['frame']
    bundle = json.loads((execution / 'execution_bundle.json').read_text())
    arrays, _ = observe._measurements(execution, bundle)
    assert not any(x.flags.writeable for x in arrays.values())


def test_queries_have_bounds_and_do_not_accept_private_configuration(tmp_path):
    private = write_json(tmp_path / 'private' / 'config.json', {'secret': 'do-not-read'})
    with pytest.raises(ValueError, match='private_configuration'):
        observe.scene_measure(private)
    execution, _ = actual('transient')
    with pytest.raises(ValueError, match='bounded_max_frames'):
        observe.run_observe(execution, max_frames=100000)
    with pytest.raises(ValueError, match='bounded_max_objects'):
        observe.scene_measure(execution, max_objects=100000)
    with pytest.raises(ValueError, match='bounded_max_findings'):
        observe.run_diagnose(execution, max_findings=100000)
    with pytest.raises(ValueError, match='choose_frame_or_operation'):
        observe.run_observe(execution, frame=1, operation='transfer_0')
    with pytest.raises(ValueError, match='plain_operation_id'):
        observe.run_diagnose(execution, operation='../outside')
    with pytest.raises(ValueError, match='frame_out_of_range'):
        observe.run_observe(execution, frame=1000000)


def test_real_descent_rejection_starts_outside_planning_margin_without_executing():
    execution = ROOT / 'runs/production_integration_20261030_v1/scene_tools_revision_r1/executions/revision_a8e6542ec9_distribute_groups'
    if not execution.is_dir():
        pytest.skip('retained revised execution not installed')
    folder = execution / 'transfer_2_motion/006_release_above_rim_conditional_payload'
    paths = [folder / x for x in ('request.json', 'decision.json', 'dense.npz', 'trigger.json')]
    paths += [execution / x for x in ('measurements.npz', 'event_protocol.json', 'planned_paths.json')]
    before = snapshot(paths)
    result = observe.run_diagnose(execution)
    finding = next(x for x in result['findings'] if x['type'] == 'payload_corridor_start_outside')
    assert finding['operation'] == 'transfer_2'
    assert finding['route'] == 'payload_alignment_before_descent'
    assert finding['descent_execution'] == 'not_executed'
    assert finding['physical_acceptance_inferred'] is False
    assert result['status']['physical']['status'] != 'passed'
    values = finding['observations']
    assert values['native_path_guard_passed'] is True
    assert values['native_dense_checks_complete'] is True
    assert values['dense_samples'] == 63
    assert values['maximum_error_dense_index'] == values['first_violating_dense_index'] == 0
    assert values['planning_limit_m'] == 0.012
    assert values['original_physical_XY_tolerance_m'] == 0.015
    assert values['predicted_start_XY_error_m'] == pytest.approx(0.012195477772, abs=1e-12)
    assert values['predicted_end_XY_error_m'] < 1e-8
    assert values['measured_current_frame'] == 697
    assert values['measured_native_grasp'] is True
    assert values['current_payload_XY_error_to_exact_task_target_m'] == pytest.approx(0.012195303512, abs=1e-12)
    for ref in finding['evidence']:
        assert digest(Path(ref['path'])) == ref['sha256']
    assert snapshot(paths) == before


def test_corridor_diagnosis_distinguishes_interior_excursion_from_start_bias(tmp_path):
    # An analytic fixture tests the alternate route; it is not a native trial.
    folder = tmp_path / 'motion'; folder.mkdir()
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[:, 0, 3] = [0, 0.013, 0]
    np.savez_compressed(folder / 'dense.npz', tcp=poses)
    write_json(folder / 'request.json', {'goal': np.eye(4).tolist(), 'payload_to_tcp': np.eye(4).tolist(),
                                       'stage': 'release_above_rim', 'preview': False})
    decision = {'accepted': False, 'report': {'passed': True}, 'payload_corridor': {
        'passed': False, 'planning_limit_m': 0.012, 'maximum_predicted_payload_XY_error_m': 0.013}}
    path = write_json(folder / 'decision.json', decision)
    result = observe._corridor_finding(tmp_path, 'transfer_0', {'object': 'payload', 'arm': 0}, path, decision, {})
    assert result['type'] == 'payload_corridor_interior_excursion'
    assert result['route'] == 'payload_descent_path_shape'
    assert result['observations']['maximum_error_dense_index'] == 1
    assert result['observations']['predicted_start_XY_error_m'] == 0
    assert result['descent_execution'] == 'unknown'
    assert result['physical_acceptance_inferred'] is False
