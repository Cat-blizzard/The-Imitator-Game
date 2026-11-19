"""Compile R2H conditions from a verified native execution and derived RGB.

This is a separate lineage-aware input adapter. It does not relabel a render
as a new physical execution, modify the old v5 interface or bypass physical
checks. The output condition structure reuses the existing compiler functions.
"""
import argparse
import copy
import hashlib
from pathlib import Path

import numpy as np

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write

VERSION = 'native_state_render_condition_adapter_v1'
PHYSICAL_CHECKS = ('task_events', 'kinematics', 'geometry', 'executed_paths_guarded')


def same_binding(left, right):
    """Legacy exporters use absolute paths, newer binders use repo-relative."""
    return resolve(left) == resolve(right)


def verify_source(parent_path):
    from .requested_goal_audit import evaluate
    parent_path = Path(parent_path); parent = read(parent_path)
    if (parent['native_execution'] is not True or parent['physical_execution_count'] != 1
            or parent['planning_failure'] is not None):
        raise ValueError('successful_native_physical_source_required')
    evaluate(parent_path.parent, parent['case'])
    native = read(parent_path.parent / 'audit.json')
    containers = read(parent_path.parent / 'container_audit.json')
    goals = read(parent_path.parent / 'requested_goal_audit.json')
    if not all(native['checks'][k]['passed'] is True for k in PHYSICAL_CHECKS):
        raise ValueError('original_native_physical_checks_failed')
    for key in ('native_container_geometry', 'payload_inside_qualified_floor'):
        if containers['checks'][key]['passed'] is not True:
            raise ValueError('original_container_check_failed')
    if not goals['goals'] or not all(g['passed'] is True for g in goals['goals']):
        raise ValueError('original_requested_goal_failed')
    declaration = read(resolve(parent['planning_declaration']))
    if (declaration['scope'] != 'before_physical_execution' or declaration['case'] != parent['case']
            or not same_binding(declaration['event_protocol'], parent['event_protocol'])):
        raise ValueError('native_preexecution_declaration_mismatch')
    task_ref = declaration.get('effective_task', parent['case']['task'])
    scene_ref = declaration.get('effective_scene', parent['case']['scene'])
    if read(resolve(task_ref)) != parent['task_intent']:
        raise ValueError('native_execution_changed_its_predeclared_effective_task')
    return parent, read(resolve(scene_ref)), declaration


def verify_mapping(render, arrays, directory):
    mapping = read(directory / 'frame_mapping.json')
    if (not same_binding(mapping['source_execution'], render['source_execution'])
            or not same_binding(mapping['source_measurements'], render['measurements'])):
        raise ValueError('render_mapping_source_mismatch')
    with np.load(resolve(mapping['source_states']), allow_pickle=False) as saved:
        states = saved['physics_states']
    rows = mapping['frames']; expected_times = arrays['timestamps_seconds']
    if mapping.get('env_step_calls') != 0 or mapping.get('all_states_rendered_once') is not True:
        raise ValueError('render_must_not_advance_or_subsample_physics')
    if len(rows) != len(states) or len(rows) != len(expected_times):
        raise ValueError('missing_or_extra_state_mapping_rows')
    for i, row in enumerate(rows):
        expected_hash = hashlib.sha256(np.ascontiguousarray(states[i]).tobytes()).hexdigest()
        if (row['frame'] != i or row['source_state_index'] != i or row['state_sha256'] != expected_hash
                or abs(row['source_timestamp_seconds'] - float(expected_times[i])) > 1e-9
                or set(row['cameras']) != {'robot_side', 'opposite_high'}):
            raise ValueError('render_state_or_timeline_identity_changed')
        for key, tolerance in [('state_max_abs', 1e-6), ('qpos_max_abs', 1e-6),
                               ('object_state_max_abs', 1e-6), ('TCP_max_abs', 1e-5)]:
            value = row['errors'][key]
            if not np.isfinite(value) or not 0 <= value <= tolerance:
                raise ValueError('rendered_state_readback_mismatch:' + key)
    return binding(directory / 'frame_mapping.json')


def export(directory, output):
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import media, event_protocol
    from corl2026_workshop.experiments.autonomous_pipeline_v4.generic_compiler import generation_information
    from corl2026_workshop.experiments.autonomous_pipeline_v4.compiler import verify_native_kinematics
    from corl2026_workshop.experiments.autonomous_pipeline_v4.hand_correspondence import compile_correspondence
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if output.exists(): raise ValueError('preserve_existing_derived_contract')
    path = directory / 'execution_bundle.json'; render = read(path)
    if (render['schema'] != 'resimpair_saved_state_render_bundle_v1' or render['native_execution'] is not False
            or render['physical_execution_count'] != 0 or render['new_trajectory'] is not False
            or render['saved_state_rerender'] is not True):
        raise ValueError('explicit_derived_render_bundle_required')
    parent_path = resolve(render['source_execution']); parent, declared_scene, declaration = verify_source(parent_path)
    if render['task_intent'] != parent['task_intent'] or render['case'] != parent['case']:
        raise ValueError('render_changed_task_identity')
    for key in ('measurements', 'event_protocol', 'planning_declaration', 'planned_paths'):
        if not same_binding(render[key], parent[key]): raise ValueError('render_changed_native_evidence:' + key)
    expected = copy.deepcopy(declared_scene); actual = copy.deepcopy(render['scene'])
    expected.pop('appearance', None); actual.pop('appearance', None)
    if actual != expected: raise ValueError('render_changed_nonappearance_scene_fields')
    with np.load(resolve(parent['measurements']), allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    mapping = verify_mapping(render, arrays, directory)
    if set(render['cameras']) != {'robot_side', 'opposite_high'}:
        raise ValueError('both_synchronous_views_required')
    media_checks = []
    for name, camera in render['cameras'].items():
        packet = media.packet(resolve(camera['video']), output / 'media_checks' / name, frame_budget=9)
        if (packet['decoded_frames'] != len(arrays['timestamps_seconds'])
                or not np.allclose(packet['all_native_pts_seconds'], arrays['timestamps_seconds'], atol=1e-5)
                or camera['frame_count'] != packet['decoded_frames']
                or camera['synchronous_with_measurements'] is not True
                or camera['config'] != declared_scene['cameras'][name]):
            raise ValueError('derived_video_not_aligned_with_original_physics')
        for key in ('initial_image', 'final_image'): resolve(camera[key])
        media_checks.append({'camera': name, 'decoded_frames': packet['decoded_frames'], 'passed': True})
    protocol = read(resolve(parent['event_protocol'])); event_protocol.validate(protocol)
    measured = event_protocol.compile_events(protocol, arrays)
    kinematics = verify_native_kinematics(arrays)
    if not measured['qualification']['passed'] or not kinematics['passed']:
        raise ValueError('native_event_or_kinematic_check_failed')
    measured['qualification']['native_kinematics'] = kinematics
    camera = render['cameras']['opposite_high']
    generation, initial = generation_information(protocol, measured, arrays, camera)
    semantic_key = digest(parent['task_intent'])
    contract = {'schema': 'resimpair_compiled_task_v5', 'input_adapter': VERSION,
        'execution': binding(path), 'source_native_execution': render['source_execution'],
        'event_protocol': parent['event_protocol'], 'planning_declaration': parent['planning_declaration'],
        'source_mapping': {'correspondence': 'synthetic_only', 'source_status': 'missing', 'formal_pair_qualified': False},
        'task_id': protocol['task_id'], 'semantic_key': semantic_key, 'measured': measured,
        'generation_information': generation, 'initial_scene_information': initial,
        'hand_correspondence_evidence': compile_correspondence(protocol, arrays, camera),
        'selected_camera': 'opposite_high', 'initial_image': camera['initial_image'], 'final_image': camera['final_image'],
        'simulation_video': camera['video'], 'other_simulation_video': render['cameras']['robot_side']['video'],
        'lineage': {'source_native_execution': render['source_execution'], 'render_derivative': binding(path),
            'new_physical_executions': 0, 'new_physical_trajectories': 0, 'frame_mapping': mapping,
            'dataset_partition': parent['case']['partition'], 'split_group': parent['case']['split_group'],
            'appearance_derivative_must_stay_in_source_split': True},
        'manual_downstream_annotation_count': 0, 'training_admitted': False,
        'human_generation_started': False, 'api_calls': 0}
    from resimpair_tabletop.sample_report import native_scalars
    write(output / 'contract.json', native_scalars(contract))
    write(output / 'qualification.json', {'schema': VERSION, 'implementation': binding(__file__),
          'parent_physical_checks_unchanged': True, 'media_checks': media_checks, 'frame_mapping': mapping,
          'source_declaration': parent['planning_declaration'], 'render_is_not_new_native_execution': True,
          'original_v5_interface_unchanged': True, 'contract': binding(output / 'contract.json'),
          'api_calls': 0, 'human_generation_started': False})
    return {'contract': str(output / 'contract.json'), 'api_calls': 0, 'new_physical_executions': 0}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execution', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); print(export(args.execution, args.output))
