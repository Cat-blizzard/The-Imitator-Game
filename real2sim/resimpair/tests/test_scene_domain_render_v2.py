"""CPU-only contracts for saved-state visual comparisons; no GPU jobs."""
from copy import deepcopy

import numpy as np
import pytest

from resimpair_grasp.common import binding, digest, write
from resimpair_tasks import scene_domain_render_v2 as render


def scene():
    camera = {'eye': [1, 0, 1], 'target': [0, 0, 0], 'width': 64, 'height': 64, 'fov': 1.}
    return {'objects': [{'id': 'o', 'position': [.1, .2, .3]}], 'supports': [],
        'cameras': {'robot_side': deepcopy(camera), 'opposite_high': deepcopy(camera)},
        'appearance': {'table_rgb': [1., 1., 1.]}, 'grasp_caches': {'o': 'original'}}


def test_visual_derivative_cannot_change_task_geometry_or_original_camera():
    source = scene(); candidate = deepcopy(source); candidate['scene_domain'] = {'visual_only': True}
    render.validate_visual_derivative(source, candidate, 'environment_3d')
    candidate['objects'][0]['position'][0] += .1
    with pytest.raises(ValueError, match='physical_scene'):
        render.validate_visual_derivative(source, candidate, 'environment_3d')
    candidate = deepcopy(source); candidate['cameras']['robot_side']['fov'] = 1.1
    with pytest.raises(ValueError, match='original_robot_side'):
        render.validate_visual_derivative(source, candidate, 'environment_3d_camera')


def test_only_declared_camera_variant_may_change_opposite_camera():
    source = scene(); candidate = deepcopy(source); candidate['cameras']['opposite_high']['fov'] = 1.1
    with pytest.raises(ValueError, match='camera_change_only_allowed'):
        render.validate_visual_derivative(source, candidate, 'environment_3d')
    render.validate_visual_derivative(source, candidate, 'environment_3d_camera')


def test_baseline_cannot_be_silently_restyled():
    source = scene(); candidate = deepcopy(source)
    candidate['appearance']['table_rgb'] = [.1, .2, .3]
    with pytest.raises(ValueError, match='baseline_render_values'):
        render.validate_visual_derivative(source, candidate, 'baseline')


def test_stale_binding_repair_requires_proven_cache_only_change(tmp_path):
    original = scene()
    original['appearance'].update(schema='resimpair_visual_appearance_v1',
        physical_scene_digest=digest({k: v for k, v in original.items() if k != 'appearance'}))
    write(tmp_path / 'original.json', original)
    effective = deepcopy(original); effective['grasp_caches']['o'] = 'selected_subset'
    write(tmp_path / 'effective.json', effective)
    source = {'scene': effective, 'bundle': {'case': {'scene': binding(tmp_path / 'original.json')}},
        'bindings': {'effective_scene.json': binding(tmp_path / 'effective.json')}}
    result, receipt = render.baseline_scene(source)
    assert result['appearance']['table_rgb'] == original['appearance']['table_rgb']
    assert receipt['render_values_changed'] is False and receipt['source_modified'] is False
    assert result['appearance']['physical_scene_digest'] != effective['appearance']['physical_scene_digest']
    render.validate_visual_derivative(effective, result, 'baseline')
    source['scene']['objects'][0]['position'][0] += .1
    with pytest.raises(ValueError, match='not_explained'):
        render.baseline_scene(source)


def test_screen_uses_fixed_grasp_frame_and_video_uses_every_native_frame():
    assert render.frame_indices({'stage': 'screen', 'state_indices': [0, 3, 9]}, 10) == [0, 3, 9]
    assert render.frame_indices({'stage': 'video'}, 10) == list(range(10))
    for indices in ([1, 3, 9], [0, 10, 9], [0, 3, 8], [0, 0, 9], [0, 3.5, 9]):
        with pytest.raises(ValueError, match='frozen_initial_grasp_final'):
            render.frame_indices({'stage': 'screen', 'state_indices': indices}, 10)


def test_state_readback_must_match_full_original_measurements():
    state, q, objects, tcp = np.zeros((1, 10)), np.zeros(18), np.zeros((2, 13)), np.zeros((2, 4, 4))
    arrays = {'states': q[None], 'object_states': objects[None], 'tcp_world': tcp[None]}
    errors = render.state_errors(state, state, q, objects, tcp, arrays, 0)
    render.validate_restored(errors)
    changed = state.copy(); changed[0, 0] = 1e-4
    with pytest.raises(ValueError, match='differs_from_original'):
        render.validate_restored(render.state_errors(changed, state, q, objects, tcp, arrays, 0))
    with pytest.raises(ValueError, match='shape_changed'):
        render.state_errors(state[:, :-1], state, q, objects, tcp, arrays, 0)
    errors['TCP_max_abs'] = float('nan')
    with pytest.raises(ValueError): render.validate_restored(errors)


def gate_fixture(tmp_path):
    independent = {'predeclared_control_video_case_ids': ['control'], 'cases': [
        {'id': 'control', 'predeclared_control': True, 'screen_case_id': None, 'variant': 'baseline'}]}
    write(tmp_path / 'independent.json', independent)
    write(tmp_path / 'protocol.json', {'independent_protocol': binding(tmp_path / 'independent.json')})
    return independent['cases'][0], {'id': 'candidate', 'screen_case_id': 'screen_00', 'variant': 'environment_3d_camera'}


def test_only_predeclared_controls_can_run_without_screen_gate(tmp_path):
    control, candidate = gate_fixture(tmp_path)
    assert render.video_gate(tmp_path, control, None)['execution_basis'].startswith('predeclared_control')
    with pytest.raises(ValueError, match='screen_gate_required'):
        render.video_gate(tmp_path, candidate, None)
    damaged = deepcopy(control); damaged['variant'] = 'environment_3d_camera'
    with pytest.raises(ValueError, match='control_exception'):
        render.video_gate(tmp_path, damaged, None)


def test_screen_gate_binds_corresponding_successful_render_and_protocol(tmp_path):
    _, candidate = gate_fixture(tmp_path)
    folder = tmp_path / 'cases/screen_00'
    write(folder / 'terminal.json', {'returncode': 0})
    write(folder / 'worker_result.json', {'physical_execution_count': 0})
    gate = {'schema': render.GATE_VERSION, 'protocol': binding(tmp_path / 'protocol.json'),
        'allowed_video_case_ids': ['candidate'], 'screen_receipts': [{'case_id': 'screen_00', 'passed': True,
            'terminal': binding(folder / 'terminal.json'), 'worker_result': binding(folder / 'worker_result.json')}]}
    write(tmp_path / 'gate.json', gate)
    assert render.video_gate(tmp_path, candidate, tmp_path / 'gate.json') == binding(tmp_path / 'gate.json')
    gate['screen_receipts'][0]['passed'] = False; write(tmp_path / 'failed_gate.json', gate)
    with pytest.raises(ValueError, match='screen_did_not_qualify'):
        render.video_gate(tmp_path, candidate, tmp_path / 'failed_gate.json')


def test_missing_gate_refuses_before_any_gpu_acquisition(tmp_path, monkeypatch):
    _, candidate = gate_fixture(tmp_path); candidate.update(stage='video', predeclared_control=False)
    monkeypatch.setattr(render, 'verify', lambda _: ({'cases': [candidate]},
        {'allocation': {'screen': 20, 'video': 8}, 'maximum_attempts': 32}))
    from resimpair_tasks import cudex_shared_gpu_v1
    monkeypatch.setattr(cudex_shared_gpu_v1, 'shared_snapshot', lambda *_: pytest.fail('must not query GPU'))
    with pytest.raises(ValueError, match='screen_gate_required'):
        render.run(tmp_path, 'video', 1)
    assert not (tmp_path / 'budget/reservations').exists()


def test_same_scope_cannot_gain_budget_by_changing_directory(tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'; registry = tmp_path / 'registry'
    for root in (a, b): write(root / 'protocol.json', {'independent': 'same'})
    protocol = {'independent_protocol': {'path': 'fixed', 'sha256': 'fixed'}, 'budget': {'path': 'budget', 'sha256': 'fixed'}}
    first = render.register_scope(a, protocol, registry=registry)
    assert render.register_scope(a, protocol, registry=registry) == first
    with pytest.raises(ValueError, match='already_bound'):
        render.register_scope(b, protocol, registry=registry)


def test_gallery_keeps_failed_and_unattempted_cases(tmp_path):
    cases = [{'id': 'failed-screen', 'stage': 'screen', 'source_id': 'a', 'variant': 'environment_3d_camera'},
        {'id': 'unattempted-video', 'stage': 'video', 'source_id': 'b', 'variant': 'baseline'}]
    write(tmp_path / 'protocol.json', {'cases': cases})
    write(tmp_path / 'cases/failed-screen/terminal.json', {'returncode': 1, 'elapsed_seconds': 2.5})
    output = tmp_path / 'gallery/index.html'
    report = render.gallery(tmp_path, output)
    assert report['counts'] == {'rendered': 0, 'failed': 1, 'unknown': 0, 'not_run': 1}
    assert all(case['id'] in output.read_text() for case in cases)
    assert report['failures_and_unattempted_cases_retained'] is True
