"""Finite 720p camera candidate over an already accepted visual derivative.

The baseline is the exact retained 512px video. Only opposite_high pixel extent
changes; physics, camera poses/FOV, robot_side, materials, lighting and furniture
are held fixed. Rendering and independent qualification use separate modules.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import shutil

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write
from . import visual_furniture_bulk_render_v4 as batch
from . import scene_domain_v2 as domain

VERSION = 'registered_native_camera_resolution_replay_v1'


def candidate_scene(baseline):
    scene = deepcopy(baseline)
    camera = scene['cameras']['opposite_high']
    if (camera['width'], camera['height']) != (512, 512):
        raise ValueError('this_finite_candidate_requires_original_512_square_source')
    camera.update(width=1280, height=720)
    sampling = scene['scene_domain']['camera_sampling']
    sampling.update(actual_configs=deepcopy(scene['cameras']), modified_camera='opposite_high',
                    status='explicit_resolution_only_candidate',
                    resolution_policy=VERSION)
    restored = deepcopy(scene)
    restored['cameras'] = deepcopy(baseline['cameras'])
    restored['scene_domain']['camera_sampling'] = deepcopy(baseline['scene_domain']['camera_sampling'])
    if restored != baseline:
        raise ValueError('resolution_candidate_changed_other_scene_values')
    domain.validate_scene(scene)
    return scene


def prepare(baseline_video, output, independent_review, *, prior_budget, allocation_id):
    baseline_video, output = Path(baseline_video).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('new_resolution_scope_required')
    if shutil.disk_usage(ROOT).free < 10 * 2**30:
        raise ValueError('insufficient_space_preserve_existing_data')
    old_worker = read(baseline_video / 'worker_result.json')
    old_reservation = read(baseline_video / 'reservation.json')
    old_protocol = read(resolve(old_reservation['protocol']))
    old_case = next(c for c in old_protocol['cases'] if c['id'] == old_worker['case_id'])
    if old_case['stage'] != 'video' or read(baseline_video / 'terminal.json')['returncode'] != 0:
        raise ValueError('completed_native_baseline_video_required')
    old_review = read(independent_review)
    reviewed = next((r for r in old_review['rows'] if r['case_id'] == old_case['id']), None)
    if not reviewed or not reviewed['render_candidate_qualified']:
        raise ValueError('independently_qualified_baseline_required')
    original = read(resolve(old_case['scene']))
    scene = candidate_scene(original)
    source_row = deepcopy(next(s for s in old_protocol['sources'] if s['id'] == old_case['source_id']))
    source = batch.appearance.replay.validate_source(resolve(source_row['source_execution']).parent)
    batch.appearance.render.validate_visual_derivative(source['scene'], scene, 'environment_3d_camera')
    if source['bindings'] != source_row['artifacts']:
        raise ValueError('native_source_bindings_changed')
    baseline_screen = next(c for c in old_protocol['cases'] if c['id'] == old_case['screen_case_id'])
    fixed = {
        'schema': VERSION, 'changed_camera': 'opposite_high', 'source_dimensions': [512, 512],
        'candidate_dimensions': [1280, 720], 'eye_target_vertical_FOV_unchanged': True,
        'other_camera_unchanged': True, 'geometry_materials_lighting_unchanged': True,
        'source_execution': source_row['source_execution'],
        'baseline_protocol': old_reservation['protocol'], 'baseline_video_case_id': old_case['id'],
        'baseline_screen_case_id': baseline_screen['id'], 'baseline_scene': old_case['scene'],
        'baseline_worker_result': binding(baseline_video / 'worker_result.json'),
        'baseline_bundle': binding(baseline_video / 'execution_bundle.json'),
        'baseline_independent_review': binding(independent_review),
        'baseline_screen_worker_result': binding(resolve(old_reservation['protocol']).parent / 'cases' / baseline_screen['id'] / 'worker_result.json'),
        'linear_pixel_gain_at_fixed_pose_vertical_FOV': 720 / 512,
        'approximate_area_pixel_gain': (720 / 512)**2,
        'dual_camera_total_pixel_multiplier': (1280 * 720 + 512 * 512) / (2 * 512 * 512),
        'actual_projection_and_calibration_required': True,
        'generation_or_visual_quality_improvement_claimed': False,
        'new_physics_executions': 0, 'API_calls': 0}
    write(output / 'resolution_ablation.json', fixed)
    cases = []
    for stage in ('screen', 'video'):
        case_id = stage + '_metric_group_distribute_dining_720p'
        folder = output / 'cases' / case_id
        write(folder / 'render_scene.json', scene)
        shutil.copy2(baseline_video / 'motion_envelope.json', folder / 'motion_envelope.json')
        write(folder / 'source_artifacts.json', source['bindings'])
        for name in batch.appearance.replay.COPY_EVIDENCE:
            shutil.copy2(source['source'] / name, folder / name)
        cases.append({'id': case_id, 'source_id': source_row['id'], 'stage': stage,
            'variant': 'environment_3d_camera', 'scene': binding(folder / 'render_scene.json'),
            'source_execution': source_row['source_execution'],
            'state_indices': source_row['screen_state_indices'] if stage == 'screen' else None,
            'visualization_only': True, 'R2H_ready': False, 'predeclared_control': False,
            'screen_case_id': 'screen_metric_group_distribute_dining_720p' if stage == 'video' else None,
            'metadata_repair': None, 'recipe_id': 'metric_group_distribute_dining_720p',
            'room': old_case['room'], 'seed': old_case['seed'],
            'resolution_ablation': binding(output / 'resolution_ablation.json')})
    registration, limits = batch.register_budget(output, prior_budget, allocation_id, screens=1, videos=1)
    old_selection = read(resolve(old_protocol['independent_protocol']))
    selection = {'schema': VERSION, 'author': 'generator_requires_separate_resolution_validator_and_gate',
        'sources': [source_row], 'cases': cases,
        'frozen_original_evaluation_dependencies': old_selection['frozen_original_evaluation_dependencies'],
        'camera_resolution_ablation': binding(output / 'resolution_ablation.json'),
        'predeclared_control_video_case_ids': [], 'old_512_baseline_reused_no_new_attempt': True,
        'independent_rules': 'versioned_resolution_branch_required_original_rules_retained'}
    write(output / 'selection_protocol.json', selection)
    protocol = deepcopy(old_protocol)
    protocol.update(sources=[source_row], cases=cases, budget=binding(output / 'budget/limits.json'),
        budget_registration=registration, independent_protocol=binding(output / 'selection_protocol.json'),
        selection_protocol_author=selection['author'],
        camera_resolution_adapter=binding(__file__),
        camera_resolution_ablation=binding(output / 'resolution_ablation.json'))
    protocol['code'][str(Path(__file__).resolve().relative_to(ROOT))] = binding(__file__)
    write(output / 'protocol.json', protocol)
    refs = {'protocol': binding(output / 'protocol.json')}
    for ref in batch.appearance.render.bindings_in(protocol):
        resolve(ref)
        refs['dependency_' + digest(ref)] = ref
    for ref in batch.appearance.render.bindings_in(scene):
        resolve(ref)
        refs['scene_resource_' + digest(ref)] = ref
    write(output / 'frozen_inputs.json', refs)
    batch.validate_prepared_interface(output)
    batch.appearance.render.verify(output)
    result = {'schema': VERSION, 'new_screen_cases': 1, 'new_video_cases': 1,
        'baseline_reused': fixed['baseline_bundle'], 'source_native_frames': source_row['frame_count'],
        'historical_actual_render_jobs': limits['historical_actual_render_jobs'],
        'maximum_lineage_render_jobs': limits['maximum_lineage_render_jobs_including_history'],
        'new_physical_executions': 0, 'API_calls': 0}
    write(output / 'prepared.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-video', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--independent-review', type=Path, required=True)
    p.add_argument('--prior-budget', type=Path, required=True)
    p.add_argument('--allocation-id', required=True)
    a = p.parse_args()
    print(prepare(a.baseline_video, a.output, a.independent_review,
                  prior_budget=a.prior_budget, allocation_id=a.allocation_id))
