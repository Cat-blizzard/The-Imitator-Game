"""Finite saved-state screen/video comparison for scene-domain variants.

The original renderer and native executions remain frozen. This candidate
reuses their source validation and native-clock encoder, changes visual fields
only, and never calls env.step. Full videos require a separate screen receipt.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import fcntl
import hashlib
import html
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write
from . import deferred_render_v1 as original

VERSION = 'saved_state_scene_domain_render_v2'
BUNDLE_VERSION = 'resimpair_saved_state_scene_domain_bundle_v2'
GATE_VERSION = 'scene_domain_screen_qualification_v2'
SCOPE_REGISTRY = ROOT / 'runs/scene_domain_render_registry_v2'
VISUAL_FIELDS = {'appearance', 'scene_domain', 'cameras'}
VARIANTS = {'baseline', 'catalog_only', 'environment_3d', 'environment_3d_camera'}


def bindings_in(value):
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
            yield value
        else:
            for child in value.values(): yield from bindings_in(child)
    elif isinstance(value, list):
        for child in value: yield from bindings_in(child)


def validate_visual_derivative(source_scene, scene, variant):
    if variant not in VARIANTS:
        raise ValueError('unknown_controlled_visual_variant')
    if {k: v for k, v in source_scene.items() if k not in VISUAL_FIELDS} != {
            k: v for k, v in scene.items() if k not in VISUAL_FIELDS}:
        raise ValueError('visual_derivative_changed_native_physical_scene')
    if variant == 'baseline' and scene != source_scene:
        # Only a separately recorded stale appearance-binding repair is allowed.
        comparable = deepcopy(scene)
        comparable['appearance']['physical_scene_digest'] = source_scene['appearance'].get('physical_scene_digest')
        if comparable != source_scene:
            raise ValueError('baseline_render_values_must_remain_identical')
    if (set(scene['cameras']) != {'robot_side', 'opposite_high'}
            or scene['cameras']['robot_side'] != source_scene['cameras']['robot_side']):
        raise ValueError('original_robot_side_and_exact_two_camera_roles_required')
    if variant != 'environment_3d_camera' and scene['cameras'] != source_scene['cameras']:
        raise ValueError('camera_change_only_allowed_in_camera_ablation')
    for ref in bindings_in(scene): resolve(ref)


def baseline_scene(source):
    """Repair only the proven selected-cache metadata change, never render values."""
    scene = deepcopy(source['scene']); appearance = scene.get('appearance', {})
    if appearance.get('schema') != 'resimpair_visual_appearance_v1': return scene, None
    expected = digest({k: v for k, v in scene.items() if k != 'appearance'})
    if appearance['physical_scene_digest'] == expected: return scene, None
    original_ref = source['bundle']['case']['scene']; original_scene = read(resolve(original_ref))
    if ({k: v for k, v in original_scene.items() if k not in ('appearance', 'grasp_caches')} !=
            {k: v for k, v in scene.items() if k not in ('appearance', 'grasp_caches')}
            or original_scene['appearance'] != appearance
            or appearance['physical_scene_digest'] != digest({k: v for k, v in original_scene.items() if k != 'appearance'})):
        raise ValueError('inherited_appearance_digest_mismatch_not_explained_by_selected_cache_metadata')
    before = appearance['physical_scene_digest']; appearance['physical_scene_digest'] = expected
    return scene, {'schema': 'appearance_selected_cache_binding_repair_v1',
        'original_prepared_scene': original_ref, 'effective_source_scene': source['bindings']['effective_scene.json'],
        'field': 'appearance.physical_scene_digest', 'before': before, 'after': expected,
        'reason': 'pre-action grasp-cache subset changed its reference after appearance binding',
        'physical_geometry_or_pose_changed': False, 'render_values_changed': False, 'source_modified': False}


def prepare(independent, output, catalog, validator, quality_rules):
    from . import scene_domain_v2 as domain
    independent, output, catalog, validator, quality_rules = (Path(p).resolve() for p in
        (independent, output, catalog, validator, quality_rules))
    if output.exists(): raise ValueError('scene_domain_output_must_be_new')
    fixed = read(independent)
    if fixed['schema'] != 'independent_scene_domain_render_protocol_v2':
        raise ValueError('independently_frozen_scene_domain_protocol_required')
    for ref in bindings_in(fixed): resolve(ref)
    if (fixed['stage_caps'] != {'screen': 20, 'video': 8, 'explicit_repair': 4}
            or fixed['maximum_new_render_jobs'] != 32 or fixed['historical_actual_render_jobs'] != 8
            or fixed['maximum_lineage_render_jobs_including_history'] != 40
            or fixed['physical_execution_count'] != 0 or fixed['new_trajectories'] != 0 or fixed['api_calls'] != 0):
        raise ValueError('fixed_32_render_zero_physics_scope_required')
    history = []
    for ref in fixed['historical_render_protocols']:
        parent = resolve(ref).parent
        reservations = [binding(p) for p in sorted(parent.glob('*/reservation.json'))]
        history.append({'protocol': ref, 'reservations': reservations, 'actual_render_jobs': len(reservations)})
    if sum(row['actual_render_jobs'] for row in history) != fixed['historical_actual_render_jobs']:
        raise ValueError('historical_render_reservation_count_mismatch')
    sources = []; values = {}; envelopes = {}; source_code = {}
    for row in fixed['sources']:
        value = original.validate_source(resolve(row['source_execution']).parent)
        if any(value['bindings'][key] != ref for key, ref in row['source_artifacts'].items()):
            raise ValueError('source_artifact_differs_from_frozen_independent_source')
        if len(value['states']) != row['frame_count']:
            raise ValueError('source_frame_count_changed')
        for ref in bindings_in(value['bundle']['code']):
            resolve(ref); source_code[digest(ref)] = ref
        sources.append({**deepcopy(row), 'artifacts': value['bindings'], 'source_execution_code': value['bundle']['code']})
        values[row['id']] = value
        envelopes[row['id']] = domain.build_motion_envelope(value['scene'], value['source'] / 'measurements.npz')
    cases = []; scenes = {}
    for raw in fixed['cases']:
        case = deepcopy(raw); source = values[case['source_id']]
        source_row = next(s for s in sources if s['id'] == case['source_id'])
        folder = case_folder(output, case)
        if case['variant'] == 'baseline':
            scene, repair = baseline_scene(source)
        else:
            partition = 'development' if case['source_id'] == 'source_0' else 'heldout_appearance'
            scene = domain.sample_domain(source['scene'], case['seed'], catalog_path=catalog,
                preset_id=case['config']['preset_id'],
                mode='catalog_only' if case['variant'] == 'catalog_only' else 'workspace',
                camera_mode='bounded' if case['variant'] == 'environment_3d_camera' else 'fixed',
                motion_envelope=envelopes[case['source_id']], asset_partition=partition)
            repair = None
        validate_visual_derivative(source['scene'], scene, case['variant'])
        if case['stage'] == 'screen':
            case['state_indices'] = case['screen_state_indices']
            if case['state_indices'] != source_row['screen_state_indices']:
                raise ValueError('independently_selected_grasp_frame_must_be_retained')
        frame_indices(case, len(source['states']))
        write(folder / 'render_scene.json', scene); write(folder / 'source_artifacts.json', source['bindings'])
        for name in original.COPY_EVIDENCE:
            shutil.copy2(source['source'] / name, folder / name)
        if repair:
            write(folder / 'metadata_repair.json', repair); case['metadata_repair'] = binding(folder / 'metadata_repair.json')
        case.update(scene=binding(folder / 'render_scene.json'), source_execution=source_row['source_execution'],
            source_partition=source_row['dataset_partition'], split_group=source_row['split_group'],
            appearance_partition='original' if case['variant'] == 'baseline' else partition)
        cases.append(case); scenes[case['id']] = scene
    for case in cases:
        if case.get('screen_case_id') and scenes[case['id']] != scenes[case['screen_case_id']]:
            raise ValueError('candidate_video_must_use_exact_same_frozen_screen_scene')
        if case.get('same_catalog_assets_and_lights_as_matched_full_case'):
            actual = scenes[case['id']]['scene_domain']; matched = scenes[case['matched_full_case_id']]['scene_domain']
            if any(actual[key] != matched[key] for key in ('resources', 'lighting')):
                raise ValueError('controlled_ablation_changed_catalog_or_lighting')
    allocation = fixed['stage_caps']
    if any(sum(c['stage'] == stage for c in cases) != allocation[stage] for stage in ('screen', 'video')):
        raise ValueError('screen_and_video_case_counts_must_match_frozen_limits')
    limits = {'schema': VERSION, 'maximum_attempts': fixed['maximum_new_render_jobs'], 'allocation': allocation,
        'historical_actual_render_jobs': fixed['historical_actual_render_jobs'], 'historical_evidence': history,
        'maximum_lineage_render_jobs_including_history': fixed['maximum_lineage_render_jobs_including_history'],
        'physical_execution_limit': 0, 'api_limit': 0, 'gpu': fixed['gpu'],
        'minimum_free_gpu_mib': fixed['minimum_free_gpu_mib'], 'parallel_workers': fixed['parallel_workers'],
        'timeout_seconds': fixed['timeout_seconds_by_stage'], 'resource_wait_seconds': fixed['resource_wait_seconds'],
        'repair_authority': 'reserved_only; a separately frozen explicit repair receipt and independent gate are required',
        'unknown_submission_must_be_recovered': True, 'automatic_retries': 0,
        'independent_protocol': binding(independent)}
    write(output / 'budget/limits.json', limits)
    code_paths = [Path(__file__), Path(original.__file__), Path(domain.__file__),
        ROOT / 'resimpair_tasks/appearance_v1.py', ROOT / 'resimpair_pickplace/environment.py',
        ROOT / 'corl2026_workshop/experiments/autonomous_pipeline_v4/simulation.py',
        ROOT / 'PolicyGen/mani_skill/render/shaders.py', ROOT / 'PolicyGen/mani_skill/sensors/camera.py',
        ROOT / 'resimpair_tasks/cudex_shared_gpu_v1.py',
        ROOT / 'corl2026_workshop/experiments/autonomous_pipeline_v4/resources.py']
    protocol = {'schema': VERSION, 'independent_protocol': binding(independent), 'catalog': binding(catalog),
        'independent_validator': binding(validator), 'independent_quality_rules': binding(quality_rules),
        'sources': sources, 'cases': cases, 'budget': binding(output / 'budget/limits.json'),
        'code': {str(p.relative_to(ROOT)): binding(p) for p in code_paths},
        'state_roundtrip_tolerance': 1e-6, 'TCP_tolerance_m': 1e-5,
        'physical_execution_count': 0, 'native_execution': False, 'new_trajectory': False, 'api_calls': 0,
        'appearance_partition_policy': {'source_0': 'development', 'source_1': 'heldout_appearance', 'source_2': 'heldout_appearance'},
        'full_state_videos_without_resampling': True, 'raw_entity_segmentation_all_rendered_frames': True,
        'reserved_repair_jobs_registered': 0, 'default_promoted': False}
    write(output / 'protocol.json', protocol)
    refs = {'protocol': binding(output / 'protocol.json'), 'budget': binding(output / 'budget/limits.json'),
        'independent_protocol': binding(independent), 'catalog': binding(catalog),
        'independent_validator': binding(validator), 'independent_quality_rules': binding(quality_rules),
        **{'source_code_' + key: ref for key, ref in source_code.items()}}
    for i, ref in enumerate(bindings_in(protocol)):
        resolve(ref); refs['protocol_dependency_' + str(i)] = ref
    for case in cases:
        for ref in bindings_in(scenes[case['id']]):
            resolve(ref); refs['scene_resource_' + digest(ref)] = ref
    write(output / 'frozen_inputs.json', refs)
    result = {'schema': VERSION, 'sources': len(sources), 'screen_cases': 20, 'video_cases': 8,
        'explicit_repair_slots_reserved': 4, 'registered_repair_cases': 0, 'maximum_new_render_jobs': 32,
        'historical_actual_render_jobs': 8, 'maximum_lineage_render_jobs': 40,
        'new_render_jobs': 0, 'physical_execution_count': 0, 'api_calls': 0,
        'baseline_metadata_repairs': [c['id'] for c in cases if c.get('metadata_repair')]}
    write(output / 'prepared.json', result); verify(output)
    return result


def frame_indices(case, count):
    if case['stage'] == 'video': return list(range(count))
    if case['stage'] != 'screen': raise ValueError('unsupported_render_stage')
    indices = case['state_indices']
    if (len(indices) != 3 or any(type(i) is not int for i in indices)
            or not 0 == indices[0] < indices[1] < indices[2] == count - 1):
        raise ValueError('screen_requires_frozen_initial_grasp_final_state_indices')
    return indices


def state_errors(state_readback, state, robot_q, objects, tcp, arrays, index):
    expected = [state, arrays['states'][index], arrays['object_states'][index], arrays['tcp_world'][index]]
    actual = [state_readback, robot_q, objects, tcp]
    if any(np.shape(a) != np.shape(b) for a, b in zip(actual, expected)):
        raise ValueError('native_saved_state_or_measurement_shape_changed')
    return {name: float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
            for name, a, b in zip(('state_max_abs', 'qpos_max_abs', 'object_state_max_abs', 'TCP_max_abs'), actual, expected)}


def validate_restored(errors, tolerance=1e-6, tcp_tolerance=1e-5):
    if (any(not np.isfinite(v) for v in errors.values())
            or max(errors[k] for k in ('state_max_abs', 'qpos_max_abs', 'object_state_max_abs')) > tolerance
            or errors['TCP_max_abs'] > tcp_tolerance):
        raise ValueError('restored_state_differs_from_original_measurements')


def case_folder(root, case):
    return Path(root) / 'cases' / case['id']


def register_scope(root, protocol, *, registry=SCOPE_REGISTRY):
    """A different output directory cannot reuse the same render allowance."""
    root, registry = Path(root).resolve(), Path(registry).resolve(); registry.mkdir(parents=True, exist_ok=True)
    receipt_path = registry / (digest(protocol['independent_protocol']) + '.json')
    receipt = {'schema': VERSION, 'independent_protocol': protocol['independent_protocol'],
        'campaign_root': str(root), 'runtime_protocol': binding(root / 'protocol.json'),
        'budget': protocol['budget']}
    with (registry / '.lock').open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if receipt_path.exists() and read(receipt_path) != receipt:
            raise ValueError('same_independent_render_scope_already_bound_recover_original_campaign')
        write(receipt_path, receipt)
    return binding(receipt_path)


def verify(root):
    root = Path(root).resolve(); protocol = read(root / 'protocol.json')
    if protocol['schema'] != VERSION: raise ValueError('scene_domain_protocol_required')
    for ref in read(root / 'frozen_inputs.json').values(): resolve(ref)
    for ref in protocol['code'].values(): resolve(ref)
    limits = read(resolve(protocol['budget']))
    if (limits['physical_execution_limit'] != 0 or limits['api_limit'] != 0
            or limits['gpu'] != 2 or limits['minimum_free_gpu_mib'] < 12288
            or limits['parallel_workers'] != 1 or limits['maximum_attempts'] != sum(limits['allocation'].values())):
        raise ValueError('finite_serial_render_only_budget_required')
    for case in protocol['cases']:
        if case['stage'] not in ('screen', 'video') or case['variant'] not in VARIANTS:
            raise ValueError('unsupported_predeclared_case')
        resolve(case['scene'])
    return protocol, limits


def video_gate(root, case, gate_path):
    protocol = read(Path(root) / 'protocol.json')
    independent = read(resolve(protocol['independent_protocol']))
    if case['id'] in independent['predeclared_control_video_case_ids']:
        declared = next(c for c in independent['cases'] if c['id'] == case['id'])
        if (case.get('predeclared_control') is not True or case.get('screen_case_id') is not None
                or any(case[k] != v for k, v in declared.items())):
            raise ValueError('control_exception_must_match_predeclared_case')
        return {'execution_basis': 'predeclared_control_still_requires_same_post_render_validation',
                'independent_protocol': protocol['independent_protocol']}
    if gate_path is None: raise ValueError('independent_screen_gate_required_before_video')
    gate_path = Path(gate_path).resolve(); gate = read(gate_path)
    if (gate.get('schema') != GATE_VERSION or gate['protocol'] != binding(Path(root) / 'protocol.json')
            or case['id'] not in gate['allowed_video_case_ids']):
        raise ValueError('video_not_authorized_by_frozen_screen_gate')
    rows = [r for r in gate['screen_receipts'] if r['case_id'] == case['screen_case_id']]
    if len(rows) != 1 or rows[0].get('passed') is not True:
        raise ValueError('corresponding_screen_did_not_qualify')
    row = rows[0]
    folder = Path(root) / 'cases' / case['screen_case_id']
    if (row['terminal'] != binding(folder / 'terminal.json')
            or row['worker_result'] != binding(folder / 'worker_result.json')):
        raise ValueError('screen_gate_receipt_does_not_bind_actual_render')
    result = read(resolve(row['worker_result'])); terminal = read(resolve(row['terminal']))
    if terminal['returncode'] != 0 or result['physical_execution_count'] != 0:
        raise ValueError('screen_not_successfully_rendered')
    return binding(gate_path)


def entity_mapping(env):
    """Use the actual scene entity IDs; never infer ownership from image color."""
    def ids(actor):
        return [int(getattr(obj, 'entity', obj).per_scene_id) for obj in actor._objs]
    result = {'objects': {name: ids(actor) for name, actor in zip(env.object_ids, env.objects)},
        'supports': {item['id']: ids(actor) for item, actor in zip(env.scene_spec['supports'], env.support_actors)},
        'robot_links': {}, 'scene_domain': {}, 'all_entities': []}
    for arm, agent in enumerate(env.agent.agents):
        result['robot_links']['arm' + str(arm)] = {link.name: ids(link) for link in agent.robot.links}
    for scene_index, scene in enumerate(env.scene.sub_scenes):
        for entity in scene.entities:
            row = {'entity_id': int(entity.per_scene_id), 'name': entity.name, 'scene_index': scene_index}
            result['all_entities'].append(row)
            if entity.name.startswith('scene_domain/'):
                result['scene_domain'].setdefault(entity.name[len('scene_domain/'):], []).append(row['entity_id'])
    return result


def worker(root, case_id):
    from resimpair_runtime import configure
    configure()
    import gymnasium as gym
    import torch
    from PIL import Image
    from resimpair_pickplace import environment
    from corl2026_workshop.experiments.autonomous_pipeline_v4.simulation import array
    from corl2026_workshop.experiments.autonomous_pipeline_v4.timing import Accumulator
    from corl2026_workshop.experiments.autonomous_pipeline_v4.core import bind as canonical
    from . import scene_domain_v2 as domain

    root = Path(root).resolve(); protocol, limits = verify(root)
    case = next(c for c in protocol['cases'] if c['id'] == case_id); folder = case_folder(root, case)
    if os.environ.get('RESIMPAIR_SCENE_DOMAIN_RESERVATION') != str(folder / 'reservation.json'):
        raise ValueError('bounded_render_supervisor_required')
    source_row = next(s for s in protocol['sources'] if s['id'] == case['source_id'])
    for ref in source_row['artifacts'].values(): resolve(ref)
    source = original.validate_source(resolve(source_row['source_execution']).parent)
    for ref in bindings_in(source['bundle']['code']): resolve(ref)
    scene = read(resolve(case['scene'])); validate_visual_derivative(source['scene'], scene, case['variant'])
    if case['variant'] == 'baseline':
        expected_scene, expected_repair = baseline_scene(source)
        if scene != expected_scene or (expected_repair is not None and read(resolve(case['metadata_repair'])) != expected_repair):
            raise ValueError('baseline_metadata_repair_must_match_proven_original_source')
    indices = frame_indices(case, len(source['states']))
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    contact = read(resolve(source_row['artifacts']['native_contact_policy.json']))['policy']
    if case['variant'] == 'baseline':
        environment.register(scene, Accumulator(), contact_policy=contact); env_id = environment.ENV_ID
    else:
        domain.register(scene, Accumulator(), contact_policy=contact); env_id = domain.ENV_ID
    env = None; writers = {}; rows = []; cameras = {}; last_frames = {}; images = {}; segmentation = {}
    mapping = None; segmentation_formats = {}
    start = time.monotonic(); runtime_evidence = None
    try:
        env = gym.make(env_id, sim_backend='physx_cpu', render_backend='cuda:0',
            control_mode='pd_joint_pos', obs_mode='rgb+segmentation', reward_mode='none', render_mode=None)
        env.reset(seed=int(scene['seed'] % (2**32)), options={'reconfigure': True})
        b = env.unwrapped
        if b.gpu_sim_enabled or list(b.object_ids) != source['bundle']['object_ids']:
            raise ValueError('saved_state_runtime_identity_mismatch')
        if tuple(array(b.get_state()).shape) != tuple(source['states'][0].shape):
            raise ValueError('visual_environment_changed_native_state_layout')
        runtime_evidence = getattr(b, 'scene_domain_runtime_evidence', None)
        mapping = entity_mapping(b)
        for name, config in scene['cameras'].items():
            params = b._sensors[name].get_params()
            shader = b._sensors[name].config.shader_pack
            if shader not in ('minimal', 'default', 'rt'): raise ValueError('unrecognized_segmentation_shader')
            segmentation_formats[name] = {'shader_pack': shader,
                'source_texture': 'PositionSegmentation' if shader == 'minimal' else 'Segmentation',
                'segmentation_channel_index': 3 if shader == 'minimal' else 1,
                'stored_channels': 1, 'semantic': 'actor_or_link_level_scene_entity_id'}
            segmentation[name] = []
            cameras[name] = {'config': config, 'source_config': source['scene']['cameras'][name],
                'config_changed': config != source['scene']['cameras'][name],
                'parameters': {k: array(v).reshape((-1,) + tuple(array(v).shape[-2:]))[0].tolist() for k, v in params.items()},
                'synchronous_with_measurements': True, 'frame_count': len(indices),
                'fps': float(source['fps']), 'capture_mode': 'saved_state_rerender',
                'source_state_index_equals_video_frame_index': case['stage'] == 'video'}
            if case['stage'] == 'video':
                writers[name] = original.NativeClockVideo(folder / (name + '.mp4'), config['width'], config['height'], source['fps'])
        for output_index, index in enumerate(indices):
            state = source['states'][index]
            b.set_state(torch.as_tensor(state, device='cpu'))
            readback = array(b.get_state()).copy()
            qpos = np.concatenate([array(a.robot.qpos).reshape(9) for a in b.agent.agents])
            objects = np.stack([array(o.get_state()).reshape(13) for o in b.objects])
            tcp = np.stack([array(a.tcp.pose.to_transformation_matrix()).reshape(4, 4) for a in b.agent.agents])
            errors = state_errors(readback, state, qpos, objects, tcp, source['arrays'], index)
            try: validate_restored(errors, protocol['state_roundtrip_tolerance'], protocol['TCP_tolerance_m'])
            except ValueError:
                write(folder / 'state_mismatch.json', {'source_state_index': index, 'errors': errors}); raise
            obs = b.get_obs()
            if not np.array_equal(array(b.get_state()), readback):
                raise ValueError('rendering_changed_the_physics_state')
            frame_images = {}
            for name, config in scene['cameras'].items():
                rgb = array(obs['sensor_data'][name]['rgb']).reshape(config['height'], config['width'], 3)
                ids = array(obs['sensor_data'][name]['segmentation']).reshape(config['height'], config['width'])
                if not np.issubdtype(ids.dtype, np.integer): raise ValueError('integer_entity_segmentation_required')
                segmentation[name].append(ids.astype(np.int32, copy=True))
                if name in writers: writers[name].append(rgb, output_index)
                last_frames[name] = rgb
                if case['stage'] == 'screen':
                    path = folder / f'{name}_state_{index:06d}.png'
                    Image.fromarray(rgb).save(path); frame_images[name] = binding(path)
                elif output_index == 0:
                    Image.fromarray(rgb).save(folder / (name + '_initial.png'))
            rows.append({'frame': output_index, 'source_state_index': index,
                'source_timestamp_seconds': float(source['timestamps'][index]),
                'state_sha256': hashlib.sha256(np.ascontiguousarray(state).tobytes()).hexdigest(),
                'cameras': list(scene['cameras']), 'images': frame_images, 'errors': errors})
            if frame_images: images[str(index)] = frame_images
        if case['stage'] == 'video':
            for name, rgb in last_frames.items(): Image.fromarray(rgb).save(folder / (name + '_final.png'))
    finally:
        for writer in writers.values(): writer.close()
        if env is not None: env.close()
    if len(rows) != len(indices): raise ValueError('incomplete_saved_state_render')
    write(folder / 'frame_mapping.json', {'schema': VERSION, 'source_execution': source_row['source_execution'],
        'source_states': source_row['artifacts']['deferred_render_states.npz'],
        'source_measurements': source_row['artifacts']['measurements.npz'], 'frames': rows,
        'all_states_rendered_once': case['stage'] == 'video', 'env_step_calls': 0,
        'contact_signals_recomputed': False, 'original_actions_and_timeline_preserved': True})
    np.savez_compressed(folder / 'segmentation.npz', **{name: np.stack(rows) for name, rows in segmentation.items()},
        source_state_indices=np.asarray(indices, dtype=np.int64))
    write(folder / 'visibility_manifest.json', {'schema': 'native_entity_segmentation_evidence_v2',
        'segmentation': binding(folder / 'segmentation.npz'), 'arrays': {name: name for name in segmentation},
        'state_indices': indices, 'entity_mapping': mapping, 'segmentation_formats': segmentation_formats,
        'segmentation_channel_index': (next(iter(segmentation_formats.values()))['segmentation_channel_index']
            if len({v['segmentation_channel_index'] for v in segmentation_formats.values()}) == 1 else None),
        'RGB_and_segmentation_same_get_obs': True,
        'source_execution': source_row['source_execution'], 'render_scene': case['scene'],
        'render_protocol': binding(root / 'protocol.json'), 'render_case_id': case['id'],
        'all_frames_included': case['stage'] == 'video', 'physical_execution_count': 0})
    bundle_ref = None
    if case['stage'] == 'video':
        for name in cameras:
            cameras[name].update(video=canonical(folder / (name + '.mp4')),
                initial_image=canonical(folder / (name + '_initial.png')), final_image=canonical(folder / (name + '_final.png')))
        bundle = deepcopy(source['bundle'])
        bundle.update(schema=BUNDLE_VERSION, scene=scene, cameras=cameras,
            native_execution=False, physical_execution_count=0, saved_state_rerender=True, new_trajectory=False,
            source_execution=source_row['source_execution'], source_native_execution=True,
            source_execution_id=source['bundle']['case'].get('execution_id', source['bundle']['case']['id']),
            source_physical_execution_count=source['bundle']['physical_execution_count'],
            measurements=canonical(source['source'] / 'measurements.npz'),
            planned_paths=canonical(source['source'] / 'planned_paths.json'),
            event_protocol=canonical(source['source'] / 'event_protocol.json'),
            planning_declaration=canonical(source['source'] / 'planning_declaration.json'),
            code=protocol['code'], source_execution_code=source['bundle']['code'],
            frame_mapping=binding(folder / 'frame_mapping.json'),
            visibility_manifest=binding(folder / 'visibility_manifest.json'),
            render_protocol=binding(root / 'protocol.json'), render_case_id=case['id'],
            qualification_scope='saved-state visual derivative; original actions, time and contact evidence inherited',
            not_eligible_for_current_R2H_schema=True, training_admitted=False)
        write(folder / 'execution_bundle.json', bundle); bundle_ref = binding(folder / 'execution_bundle.json')
    write(folder / 'worker_result.json', {'schema': VERSION, 'case_id': case['id'], 'stage': case['stage'],
        'bundle': bundle_ref, 'source_execution': source_row['source_execution'], 'cameras': cameras,
        'visibility_manifest': binding(folder / 'visibility_manifest.json'),
        'render_protocol': binding(root / 'protocol.json'), 'render_case_id': case['id'],
        'screen_images': images, 'physical_execution_count': 0, 'native_execution': False,
        'new_trajectory': False, 'env_step_calls': 0, 'api_calls': 0, 'frames_per_camera': len(rows),
        'state_indices': indices, 'scene_domain_runtime_evidence': runtime_evidence,
        'render_and_encoding_seconds': time.monotonic() - start,
        'maximum_errors': {k: max(r['errors'][k] for r in rows) for k in rows[0]['errors']}})


def run(root, stage, limit, *, gate=None, video_group='all'):
    from corl2026_workshop.experiments.autonomous_pipeline_v4.resources import lock_paths
    from .cudex_shared_gpu_v1 import shared_snapshot
    root = Path(root).resolve(); protocol, limits = verify(root)
    if (stage not in ('screen', 'video') or type(limit) is not int
            or not 1 <= limit <= limits['allocation'][stage]):
        raise ValueError('finite_stage_launch_limit_required')
    if video_group not in ('all', 'controls', 'candidates') or (stage == 'screen' and video_group != 'all'):
        raise ValueError('valid_video_group_required')
    cases = [c for c in protocol['cases'] if c['stage'] == stage]
    if video_group != 'all': cases = [c for c in cases if c.get('predeclared_control', False) == (video_group == 'controls')]
    launched = []
    with (root / '.supervisor.lock').open('a+') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for case in cases:
            if len(launched) >= limit: break
            folder = case_folder(root, case)
            if (folder / 'terminal.json').exists(): continue
            if (folder / 'reservation.json').exists():
                raise RuntimeError('existing_render_submission_requires_recovery_no_resubmit')
            receipt = video_gate(root, case, gate) if stage == 'video' else None
            scope_receipt = register_scope(root, protocol)
            reservations = [read(p) for p in (root / 'budget/reservations').glob('*.json')]
            if (len(reservations) >= limits['maximum_attempts']
                    or sum(r['stage'] == stage for r in reservations) >= limits['allocation'][stage]):
                raise ValueError('render_budget_exhausted')
            deadline = time.monotonic() + limits['resource_wait_seconds']; acquired = None; snap = None; fds = []
            while time.monotonic() < deadline:
                candidate = ExitStack()
                try:
                    fds = []
                    for path in lock_paths(limits['gpu']):
                        handle = candidate.enter_context(path.open('r+'))
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB); fds.append(handle.fileno())
                    snap = shared_snapshot(limits['gpu'])
                    if snap['free_memory_MiB'] < limits['minimum_free_gpu_mib']:
                        raise RuntimeError('shared_gpu_headroom_unavailable')
                    acquired = candidate; break
                except (BlockingIOError, RuntimeError):
                    candidate.close(); time.sleep(min(3, max(0, deadline - time.monotonic())))
            if acquired is None: raise TimeoutError('bounded_GPU_resource_wait_expired_no_attempt_reserved')
            with acquired:
                reservation = {'case': case, 'stage': stage, 'protocol': binding(root / 'protocol.json'),
                    'gpu': snap, 'started_unix': time.time(), 'physical_execution_count': 0,
                    'api_calls': 0, 'external_processes_preserved': True, 'screen_gate': receipt,
                    'scope_registration': scope_receipt}
                write(root / 'budget/reservations' / f'{len(reservations):04d}.json', reservation)
                write(folder / 'reservation.json', reservation)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=snap['uuid'], OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                    RESIMPAIR_SCENE_DOMAIN_RESERVATION=str(folder / 'reservation.json'))
                started = time.monotonic()
                with (folder / 'worker.log').open('x') as log:
                    proc = subprocess.Popen([sys.executable, '-m', 'resimpair_tasks.scene_domain_render_v2', 'worker',
                        '--root', str(root), '--case', case['id']], cwd=ROOT, env=env,
                        stdout=log, stderr=subprocess.STDOUT, pass_fds=tuple(fds), start_new_session=True)
                    write(folder / 'process.json', {'pid': proc.pid, 'started_unix': time.time()})
                    try: code = proc.wait(timeout=limits['timeout_seconds'][stage]); timeout = False
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGTERM)
                        try: code = proc.wait(timeout=10)
                        except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL); code = proc.wait(timeout=10)
                        timeout = True
                result = {'returncode': code, 'timeout': timeout, 'stage': stage,
                    'elapsed_seconds': time.monotonic() - started, 'physical_execution_count': 0,
                    'new_trajectory': False, 'api_calls': 0}
                write(folder / 'terminal.json', result); launched.append(result); print({'case_id': case['id'], **result}, flush=True)
                if code: raise RuntimeError('render_failure_preserved_no_automatic_retry:' + case['id'])
    return {'new_render_attempts': len(launched), 'stage': stage, 'physical_execution_count': 0, 'api_calls': 0}


def gallery(root, output):
    """Local complete-cohort gallery: failures and unattempted cases stay visible."""
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists(): raise ValueError('gallery_output_must_be_new_preserve_previous_review')
    protocol = read(root / 'protocol.json'); blocks = []; counts = {'rendered': 0, 'failed': 0, 'unknown': 0, 'not_run': 0}
    def href(path): return html.escape(os.path.relpath(Path(path), output.parent), quote=True)
    def esc(value): return html.escape(str(value))
    for case in protocol['cases']:
        folder = case_folder(root, case); terminal_path = folder / 'terminal.json'
        terminal = read(terminal_path) if terminal_path.exists() else None
        status = ('rendered' if terminal['returncode'] == 0 else 'failed') if terminal else (
            'unknown' if (folder / 'reservation.json').exists() else 'not_run')
        counts[status] += 1; media = []
        mapping = read(folder / 'frame_mapping.json') if (folder / 'frame_mapping.json').exists() else {'frames': []}
        if case['stage'] == 'screen':
            for row in mapping['frames']:
                for camera, ref in row.get('images', {}).items():
                    path = resolve(ref)
                    media.append(f'<figure><a href="{href(path)}"><img loading="lazy" src="{href(path)}"></a>'
                        f'<figcaption>{esc(camera)} · native frame {row["source_state_index"]} · {row["source_timestamp_seconds"]:.2f}s</figcaption></figure>')
        else:
            for camera in ('robot_side', 'opposite_high'):
                path = folder / (camera + '.mp4'); poster = folder / (camera + '_initial.png')
                if path.exists():
                    media.append(f'<figure><video controls preload="none" poster="{href(poster)}" src="{href(path)}"></video>'
                        f'<figcaption>{esc(camera)} · full native timeline</figcaption></figure>')
        links = [f'<a href="{href(folder / name)}">{esc(name)}</a>' for name in
            ('render_scene.json', 'terminal.json', 'worker.log', 'worker_result.json', 'visibility_manifest.json') if (folder / name).exists()]
        elapsed = f'{terminal["elapsed_seconds"]:.2f}s' if terminal and 'elapsed_seconds' in terminal else '—'
        blocks.append(f'<article class="{status}"><h2>{esc(case["id"])}</h2><p><strong>{status}</strong> · '
            f'{esc(case["variant"])} · {esc(case["source_id"])} · {elapsed}</p>'
            f'<div class="media">{"".join(media) or "<p>没有已完成媒体；保留该例及原失败，不用其它场景替代。</p>"}</div>'
            f'<p class="links">{" · ".join(links)}</p></article>')
    document = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    document += '<title>Scene domain v2 · 完整对照集</title><style>body{margin:0;background:#edf1f5;color:#172435;font:16px/1.55 "Noto Sans CJK SC","Microsoft YaHei",sans-serif}main{max-width:1400px;margin:auto;padding:28px}h1{font-size:28px}h2{font-size:19px;overflow-wrap:anywhere}article{padding:20px;margin:20px 0;background:white;border-radius:10px;border-left:6px solid #8393a8}.rendered{border-color:#34836f}.failed{border-color:#bd4747}.unknown{border-color:#b57c21}.media{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}figure{margin:0}img,video{width:100%;background:#e3e8ec}figcaption,.links{font-size:13px;color:#4b5f73}a{color:#195a99}header{padding:22px;background:white;border-radius:10px}@media(max-width:700px){.media{grid-template-columns:1fr}}</style><main><header>'
    document += '<h1>Scene domain v2 · 全部对照结果</h1><p>来源、状态、动作和时间轴均继承原物理执行。本页包括未运行、失败和成功渲染的所有预登记例；rendered 只表示渲染进程完成，人物参考质量及像素可读性需独立审查。</p>'
    document += f'<p>{esc(counts)} · 新物理执行 0 · API 调用 0</p></header>{"".join(blocks)}</main></html>'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream: stream.write(document)
    receipt = {'schema': VERSION, 'protocol': binding(root / 'protocol.json'), 'gallery': binding(output),
        'counts': counts, 'registered_cases': len(protocol['cases']), 'failures_and_unattempted_cases_retained': True,
        'new_render_jobs': 0, 'physical_execution_count': 0, 'api_calls': 0}
    write(output.with_suffix('.json'), receipt)
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    for name in ('independent', 'output', 'catalog', 'validator', 'quality-rules'):
        prep.add_argument('--' + name, type=Path, required=True)
    launch = sub.add_parser('run'); launch.add_argument('--root', type=Path, required=True)
    launch.add_argument('--stage', choices=['screen', 'video'], required=True)
    launch.add_argument('--limit', type=int, required=True); launch.add_argument('--gate', type=Path)
    launch.add_argument('--video-group', choices=['all', 'controls', 'candidates'], default='all')
    child = sub.add_parser('worker'); child.add_argument('--root', type=Path, required=True)
    child.add_argument('--case', required=True)
    view = sub.add_parser('gallery'); view.add_argument('--root', type=Path, required=True)
    view.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.command == 'prepare': print(prepare(a.independent, a.output, a.catalog, a.validator, a.quality_rules))
    elif a.command == 'run': print(run(a.root, a.stage, a.limit, gate=a.gate, video_group=a.video_group))
    elif a.command == 'worker': worker(a.root, a.case)
    else: print(gallery(a.root, a.output))
