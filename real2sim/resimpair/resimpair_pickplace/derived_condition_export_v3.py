"""Compile source-bound R2H conditions for scene/camera render derivatives.

Uses the independent v4 reference-view validator and the existing event, projection and
anatomical-hand compiler. Native execution, original audits and old adapters
are read-only. Conditions with unknown readability are marked not ready for
paid human generation; compiling a contract is not a generated human pair.
"""
import argparse
from pathlib import Path

import numpy as np

from resimpair_grasp.common import binding, digest, read, resolve, write

VERSION = 'scene_domain_render_condition_adapter_v3_rules_v4'
PHYSICAL_CHECKS = ('task_events', 'kinematics', 'geometry', 'executed_paths_guarded')


def compile_information(parent, render, arrays):
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import event_protocol
    from corl2026_workshop.experiments.autonomous_pipeline_v4.generic_compiler import generation_information
    from corl2026_workshop.experiments.autonomous_pipeline_v4.compiler import verify_native_kinematics
    from corl2026_workshop.experiments.autonomous_pipeline_v4.hand_correspondence import compile_correspondence
    protocol = read(resolve(parent['event_protocol'])); event_protocol.validate(protocol)
    measured = event_protocol.compile_events(protocol, arrays)
    kinematics = verify_native_kinematics(arrays)
    if not measured['qualification']['passed'] or not kinematics['passed']:
        raise ValueError('original_event_or_native_kinematic_check_failed')
    measured['qualification']['native_kinematics'] = kinematics
    camera = render['cameras']['opposite_high']
    generation, initial = generation_information(protocol, measured, arrays, camera)
    domain = render['scene'].get('scene_domain')
    visual = {'preserve_environment_from_reference_images': True,
              'environment_static_throughout_video': True,
              'camera_static_throughout_video': True,
              'background_furniture_is_not_a_task_participant': True}
    if domain: visual['scene_family'] = domain['category']
    # The existing image/video prompt builders already consume these two dicts.
    generation['visual_environment'] = dict(visual)
    initial['visual_environment'] = dict(visual)
    return protocol, measured, generation, initial, compile_correspondence(protocol, arrays, camera)


def export(directory, output):
    from . import render_lineage_validator_v4 as validator
    from resimpair_tabletop.sample_report import native_scalars
    from corl2026_workshop.experiments.autonomous_pipeline_v4.core import bind as canonical, implementation, checked
    from corl2026_workshop.experiments.autonomous_pipeline_v4.prompts import anchor_prompt, video_prompt
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if output.exists(): raise ValueError('condition_output_must_be_new')
    path = directory / 'execution_bundle.json'; before = binding(path); render = read(path)
    if (render['schema'] != 'resimpair_saved_state_scene_domain_bundle_v2'
            or render['native_execution'] is not False or render['physical_execution_count'] != 0
            or render['new_trajectory'] is not False):
        raise ValueError('explicit_scene_domain_render_derivative_required')
    proof = validator.validate(directory)
    if proof.get('passed') is not True: raise ValueError('independent_scene_domain_lineage_validation_failed')
    parent_path = resolve(render['source_execution']); parent = read(parent_path)
    # Read original independent results; do not rerun an evaluator that writes
    # over historical audit files merely to export a new visual derivative.
    native = read(parent_path.parent / 'audit.json')
    container = read(parent_path.parent / 'container_audit.json')
    goal = read(parent_path.parent / 'requested_goal_audit.json')
    if (native['errors'] or not all(native['checks'][k]['passed'] is True for k in PHYSICAL_CHECKS)
            or container['errors'] or not all(c['passed'] is True for c in container['checks'].values())
            or not goal['goals'] or not all(g['passed'] is True for g in goal['goals'])):
        raise ValueError('original_physical_acceptance_required')
    with np.load(resolve(parent['measurements']), allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}
    protocol, measured, generation, initial, hands = compile_information(parent, render, arrays)
    if before != binding(path): raise ValueError('render_changed_during_compilation')
    write(output / 'render_validation.json', native_scalars(proof))
    camera = render['cameras']['opposite_high']; domain = render['scene'].get('scene_domain')
    contract = {'schema': 'resimpair_compiled_task_v5', 'input_adapter': VERSION,
        'execution': canonical(path), 'source_native_execution': render['source_execution'],
        'event_protocol': parent['event_protocol'], 'planning_declaration': parent['planning_declaration'],
        'source_mapping': {'correspondence': 'synthetic_only', 'source_status': 'missing', 'sources': [], 'formal_pair_qualified': False},
        'task_id': protocol['task_id'], 'semantic_key': digest(parent['task_intent']),
        'measured': measured, 'generation_information': generation, 'initial_scene_information': initial,
        'hand_correspondence_evidence': hands, 'selected_camera': 'opposite_high',
        'initial_image': canonical(resolve(camera['initial_image'])), 'final_image': canonical(resolve(camera['final_image'])),
        'simulation_video': canonical(resolve(camera['video'])), 'other_simulation_video': canonical(resolve(render['cameras']['robot_side']['video'])),
        'adapter_validation': binding(output / 'render_validation.json'),
        'render_readability': proof.get('pixel_readability', {'status': 'unknown'}),
        'generation_ready': proof.get('generation_ready') is True,
        'lineage': {'source_native_execution': render['source_execution'], 'render_derivative': before,
            'new_physical_executions': 0, 'new_physical_trajectories': 0,
            'frame_mapping': render['frame_mapping'], 'dataset_partition': parent['case']['partition'],
            'split_group': parent['case']['split_group'], 'appearance_derivative_must_stay_in_source_split': True,
            'scene_domain': domain, 'actual_camera_configuration': camera['config'],
            'actual_camera_parameters': camera['parameters'],
            'camera_dependent_projections_recomputed': True},
        'condition_consumption': {'initial_scene_information': 'existing_initial_image_prompt',
            'generation_information': 'existing_video_and_review_prompt',
            'render_readability': 'separate_generation_gate_not_inferred_task_success'},
        'adapter_code': {'adapter': binding(__file__), 'validator': binding(validator.__file__)},
        'code': implementation(),
        'manual_downstream_annotation_count': 0, 'training_admitted': False,
        'human_generation_started': False, 'new_paid_model_calls': 0}
    write(output / 'contract.json', native_scalars(contract))
    for key in ('execution', 'initial_image', 'final_image', 'simulation_video', 'other_simulation_video'):
        checked(contract[key])
    write(output / 'prompt_preview.json', {'schema': VERSION, 'contract': binding(output / 'contract.json'),
        'initial_image_prompt': anchor_prompt(initial), 'video_prompt': video_prompt(contract),
        'initial_simulation_image': contract['initial_image'], 'terminal_simulation_image': contract['final_image'],
        'generation_ready': contract['generation_ready'], 'submitted': False,
        'model_configuration_and_finite_budget_required_before_paid_submission': True,
        'prompt_functions_are_existing_pipeline_functions': True})
    write(output / 'qualification.json', {'schema': VERSION, 'contract': binding(output / 'contract.json'),
        'original_physical_rules_and_inputs_unchanged': True, 'new_physical_executions': 0,
        'new_paid_model_calls': 0, 'generation_ready': contract['generation_ready'],
        'not_a_formal_Real_Sim_Human_pair': True})
    return {'contract': str(output / 'contract.json'), 'generation_ready': contract['generation_ready'],
            'new_paid_model_calls': 0, 'new_physical_executions': 0}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--execution', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args(); print(export(a.execution, a.output))
