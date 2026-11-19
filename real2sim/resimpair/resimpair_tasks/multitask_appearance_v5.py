"""Optional saved-state appearance for any completed multi-task execution.

Appearance is composed after physics and never gates physical collection. The
source execution, tasks, actions, state slots and both cameras remain immutable.
Missing visual resources produce an explicit degraded appearance receipt; they
do not relabel a valid trajectory or claim a human-generation-ready result.

`apply_overlay` is a CPU-only integration point. `prepare` and `run` connect it
to the existing state-checked renderer and shared-GPU scheduler. No new physical
execution, model API request or independent trajectory is created here.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
from pathlib import Path
import shutil
import time

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write
from . import deferred_render_v1 as replay
from . import scene_domain_v2 as domain
from . import scene_quality_v3 as quality
from . import scene_quality_render_v3 as render

VERSION = 'multitask_optional_appearance_v5'
CATALOG = ROOT / 'assets/appearance_v2/curated/catalog.json'
REGISTRY = ROOT / 'runtime/multitask_appearance_v5_registry'
MODES = ('disabled', 'catalog_only', 'workspace', 'metric_room')


def assert_invariants(source, result):
    if domain.physical_scene(source) != domain.physical_scene(result):
        raise RuntimeError('optional_appearance_changed_physical_scene')
    if source['cameras'] != result['cameras']:
        raise RuntimeError('optional_appearance_changed_source_cameras')


def apply_overlay(scene, seed, *, mode='metric_room', preset='office_bookcase',
                  motion_envelope=None, catalog=CATALOG,
                  material_index=quality.MATERIALS, furniture_catalog=quality.FURNITURE,
                  material_partition='development', allow_visual_fallback=True):
    """Return (scene copy, receipt); never mutate the input or any asset.

    Size, collision, robot, control and task fields are immutable. A missing or
    corrupt appearance resource falls back to the exact source scene only when
    requested; this is recorded as degraded, never as a successful augmentation.
    Geometry/camera invariant failures always raise, even in fallback mode.
    """
    if mode not in MODES or type(seed) is not int or seed < 0:
        raise ValueError('invalid_optional_appearance_configuration')
    if preset not in domain.PRESETS or material_partition not in ('development', 'heldout_appearance'):
        raise ValueError('explicit_known_preset_and_material_partition_required')
    if type(allow_visual_fallback) is not bool:
        raise ValueError('boolean_visual_fallback_required')
    before = deepcopy(scene)
    receipt = {'schema': VERSION, 'seed': seed, 'mode': mode, 'preset': preset,
        'source_scene_digest': digest(scene), 'physical_scene_digest': digest(domain.physical_scene(scene)),
        'camera_digest': digest(scene['cameras']), 'material_partition': material_partition,
        'implementation': binding(__file__), 'physical_success_label_modified': False,
        'new_physical_executions': 0, 'API_calls': 0,
        'R2H_ready': False, 'render_quality_status': 'not_measured',
        'failure_does_not_discard_source_execution': True}
    if mode == 'disabled':
        return before, {**receipt, 'status': 'disabled', 'appearance_applied': False,
                        'result_scene_digest': digest(before), 'error': None}
    if mode in ('workspace', 'metric_room') and motion_envelope is None:
        raise ValueError('workspace_requires_actual_motion_envelope')
    try:
        result = domain.sample_domain(scene, seed, catalog_path=catalog, preset_id=preset,
            mode='catalog_only' if mode == 'catalog_only' else 'workspace', camera_mode='fixed',
            motion_envelope=motion_envelope, asset_partition=material_partition)
        from .scene_material_policy_v3 import apply_policy
        result = apply_policy(result, seed, index_path=material_index,
                              mode='semantic_rgb', asset_partition=material_partition)
        if mode == 'metric_room':
            result = quality.apply_furniture(result, catalog_path=furniture_catalog)
    except (FileNotFoundError, ValueError, OSError) as exc:
        # This boundary is after collection. The original successful physical
        # execution remains available even when a visual derivative is absent.
        if not allow_visual_fallback:
            raise
        assert_invariants(scene, before)
        return before, {**receipt, 'status': 'visual_degraded_original_retained',
            'appearance_applied': False, 'result_scene_digest': digest(before),
            'error': {'type': type(exc).__name__, 'message': str(exc)}}
    assert_invariants(before, result)
    if scene != before:
        raise RuntimeError('optional_appearance_mutated_input_scene')
    domain.validate_scene(result)
    return result, {**receipt, 'status': 'prepared_requires_render_validation',
        'appearance_applied': True, 'result_scene_digest': digest(result), 'error': None,
        'resources': {role: row['id'] if row else None
                      for role, row in result['scene_domain']['resources'].items()},
        'furniture_ids': [r['source_id'] for r in (result['scene_domain'].get('geometry') or {}).get('furniture', [])]}


def _register_budget(output, prior_budget, allocation_id, cases, *, registry=REGISTRY):
    """Append a new finite allowance; never spend an old unallocated allowance."""
    if not allocation_id or len(cases) not in (1, 2):
        raise ValueError('named_one_or_two_case_visual_allocation_required')
    prior_path = Path(prior_budget).resolve(); prior = read(prior_path)
    previous_reservations = list((prior_path.parent / 'reservations').glob('*.json'))
    for path in previous_reservations:
        reservation = read(path)
        source_protocol = read(resolve(reservation['protocol']))
        previous_case = next(c for c in source_protocol['cases'] if c['id'] == reservation['case']['id'])
        previous_root = resolve(reservation['protocol']).parent
        if not (render.case_folder(previous_root, previous_case) / 'terminal.json').exists():
            raise ValueError('prior_render_attempt_unresolved_recover_before_extending_lineage')
    historical = prior['historical_actual_render_jobs'] + len(previous_reservations)
    prior_limit = prior['maximum_lineage_render_jobs_including_history']
    registry = Path(registry); registry.mkdir(parents=True, exist_ok=True)
    receipt_path = registry / 'registrations' / (digest(allocation_id) + '.json')
    with (registry / '.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt_path.exists():
            raise ValueError('appearance_allocation_already_registered_recover_original_output')
        count = len(cases)
        limits = {'schema': VERSION, 'maximum_attempts': count,
            'allocation_id': allocation_id, 'allocation_limit': count,
            'allocation': {stage: sum(c['stage'] == stage for c in cases) for stage in ('screen', 'video')},
            'prior_budget': binding(prior_path), 'historical_actual_render_jobs': historical,
            'prior_lineage_limit': prior_limit,
            'maximum_lineage_render_jobs_including_history': prior_limit + count,
            'maximum_actual_jobs_including_history': historical + count,
            'unspent_prior_allowance_not_reassigned': True,
            'physical_execution_limit': 0, 'api_limit': 0, 'gpu': 2,
            'minimum_free_gpu_mib': 12288, 'parallel_workers': 1,
            'timeout_seconds': {'screen': 240, 'video': 240}, 'resource_wait_seconds': 180,
            'automatic_retries': 0, 'unknown_submission_must_be_recovered': True}
        write(output / 'budget/limits.json', limits)
        write(receipt_path, {'schema': VERSION, 'allocation_id': allocation_id,
            'root': str(output), 'budget': binding(output / 'budget/limits.json'),
            'maximum_attempts': count, 'created_unix': time.time()})
    return binding(receipt_path), limits


def prepare(source, output, *, seed, prior_budget, allocation_id,
            mode='metric_room', preset='office_bookcase', stage='screen',
            include_baseline=True, catalog=CATALOG,
            material_index=quality.MATERIALS, furniture_catalog=quality.FURNITURE,
            material_partition='development', registry=REGISTRY):
    """Bind actual native evidence and prepare a separate screen/video derivative."""
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('preserve_existing_optional_appearance_output')
    if stage not in ('screen', 'video') or type(include_baseline) is not bool:
        raise ValueError('invalid_render_stage_or_baseline_option')
    if shutil.disk_usage(ROOT).free < 2 * 2**30:
        raise ValueError('insufficient_disk_preserve_history')
    value = replay.validate_source(source)
    baseline, repair = render.baseline_scene(value)
    envelope = domain.build_motion_envelope(baseline, value['source'] / 'measurements.npz')
    scene, receipt = apply_overlay(baseline, seed, mode=mode, preset=preset, motion_envelope=envelope,
        catalog=catalog, material_index=material_index, furniture_catalog=furniture_catalog,
        material_partition=material_partition)
    write(output / 'appearance_receipt.json', receipt)
    write(output / 'motion_envelope.json', envelope)
    count = len(value['states'])
    grasp = next(e['frame'] for e in read(value['source'] / 'audit.json')['event_frames']
                 if e['id'].endswith('.grasp'))
    if not 0 < grasp < count - 1:
        raise ValueError('native_grasp_event_required_for_screen')
    indices = [0, grasp, count - 1] if stage == 'screen' else list(range(count))
    variants = [('baseline', baseline)] if include_baseline else []
    if receipt['appearance_applied']:
        variants.append(('appearance', scene))
    elif not variants:
        # An explicit original-scene fallback is independently labeled. It must
        # never be counted as a successful background or human-video example.
        variants.append(('baseline', baseline))
    cases = []
    source_row = {'id': 'source_0', 'source_execution': value['bindings']['execution_bundle.json'],
        'artifacts': value['bindings'], 'source_execution_code': value['bundle']['code'],
        'frame_count': count, 'fps': float(value['fps']),
        'source_execution_id': value['bundle']['case'].get('execution_id', value['bundle']['case']['id']),
        'task_operations': len(value['task']['operations'])}
    for label, candidate in variants:
        case_id = stage + '_' + label; folder = output / 'cases' / case_id
        variant = ('baseline' if label == 'baseline' else
                   'catalog_only' if mode == 'catalog_only' else 'environment_3d')
        render.validate_visual_derivative(value['scene'], candidate, variant)
        write(folder / 'render_scene.json', candidate)
        write(folder / 'source_artifacts.json', value['bindings'])
        for name in replay.COPY_EVIDENCE:
            shutil.copy2(value['source'] / name, folder / name)
        case = {'id': case_id, 'source_id': 'source_0', 'stage': stage, 'variant': variant,
            'scene': binding(folder / 'render_scene.json'), 'state_indices': indices,
            'source_execution': source_row['source_execution'],
            'predeclared_control': True, 'screen_case_id': None,
            'visualization_only': True, 'R2H_ready': False,
            'appearance_status': receipt['status'] if label == 'appearance' else 'retained_source_appearance'}
        if label == 'baseline' and repair:
            write(folder / 'metadata_repair.json', repair)
            case['metadata_repair'] = binding(folder / 'metadata_repair.json')
        cases.append(case)
    registration, limits = _register_budget(output, prior_budget, allocation_id, cases, registry=registry)
    selection = {'schema': VERSION, 'author': 'implementation_agent_using_retained_physical_acceptance',
        'cases': cases, 'source_execution': source_row['source_execution'],
        'predeclared_control_video_case_ids': [c['id'] for c in cases if c['stage'] == 'video'],
        'appearance_receipt': binding(output / 'appearance_receipt.json'),
        'qualification_scope': 'integration and state preservation, not an independent visual-quality verdict',
        'physical_execution_count': 0, 'api_calls': 0}
    write(output / 'selection_protocol.json', selection)
    modules = [Path(__file__), Path(render.__file__), Path(replay.__file__), Path(domain.__file__),
        Path(quality.__file__), ROOT / 'resimpair_tasks/scene_material_policy_v3.py',
        ROOT / 'resimpair_tasks/appearance_v1.py', ROOT / 'resimpair_pickplace/environment.py',
        ROOT / 'resimpair_tasks/cudex_shared_gpu_v1.py',
        ROOT / 'corl2026_workshop/experiments/autonomous_pipeline_v4/simulation.py',
        ROOT / 'corl2026_workshop/experiments/autonomous_pipeline_v4/resources.py',
        ROOT / 'PolicyGen/mani_skill/render/shaders.py', ROOT / 'PolicyGen/mani_skill/sensors/camera.py']
    protocol = {'schema': render.VERSION, 'appearance_adapter': binding(__file__),
        'independent_protocol': binding(output / 'selection_protocol.json'),
        'selection_protocol_author': selection['author'], 'sources': [source_row], 'cases': cases,
        'budget': binding(output / 'budget/limits.json'), 'budget_registration': registration,
        'code': {str(p.relative_to(ROOT)): binding(p) for p in modules},
        'state_roundtrip_tolerance': 1e-6, 'TCP_tolerance_m': 1e-5,
        'physical_execution_count': 0, 'native_execution': False, 'new_trajectory': False,
        'api_calls': 0, 'R2H_ready': False, 'default_promoted': False}
    write(output / 'protocol.json', protocol)
    refs = {'protocol': binding(output / 'protocol.json'),
            'appearance_receipt': binding(output / 'appearance_receipt.json'),
            'motion_envelope': binding(output / 'motion_envelope.json')}
    for ref in render.bindings_in(protocol):
        resolve(ref); refs['dependency_' + digest(ref)] = ref
    for candidate in (read(resolve(c['scene'])) for c in cases):
        for ref in render.bindings_in(candidate):
            resolve(ref); refs['scene_resource_' + digest(ref)] = ref
    write(output / 'frozen_inputs.json', refs)
    render.verify(output)
    result = {'schema': VERSION, 'source_execution': source_row['source_execution'],
        'appearance_status': receipt['status'], 'prepared_cases': len(cases), 'stage': stage,
        'frames_per_camera_per_case': len(indices), 'cameras_per_case': len(scene['cameras']),
        'maximum_render_jobs': limits['maximum_attempts'],
        'historical_actual_render_jobs': limits['historical_actual_render_jobs'],
        'lineage_render_ceiling': limits['maximum_lineage_render_jobs_including_history'],
        'physical_scene_and_cameras_unchanged': True,
        'new_physical_executions': 0, 'API_calls': 0, 'render_jobs_started': 0}
    write(output / 'prepared.json', result)
    return result


def run(output):
    protocol, limits = render.verify(output)
    stages = {c['stage'] for c in protocol['cases']}
    if len(stages) != 1:
        raise ValueError('one_stage_per_optional_appearance_campaign')
    stage = next(iter(stages))
    return render.run(output, stage, limits['allocation'][stage])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--source', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True); p.add_argument('--prior-budget', type=Path, required=True)
    p.add_argument('--allocation-id', required=True); p.add_argument('--mode', choices=MODES, default='metric_room')
    p.add_argument('--preset', choices=domain.PRESET_IDS, default='office_bookcase')
    p.add_argument('--stage', choices=('screen', 'video'), default='screen')
    p.add_argument('--without-baseline', action='store_true')
    p = commands.add_parser('run'); p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(prepare(args.source, args.output, seed=args.seed, prior_budget=args.prior_budget,
                      allocation_id=args.allocation_id, mode=args.mode, preset=args.preset,
                      stage=args.stage, include_baseline=not args.without_baseline))
    else:
        print(run(args.output))
