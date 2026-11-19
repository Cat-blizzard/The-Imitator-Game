"""Prepare/run up to four fixed-source screens using the bulk furniture library."""
from __future__ import annotations

import argparse
import fcntl
import html
import os
from pathlib import Path
import shutil
import time

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write
from . import visual_furniture_bulk_scene_v1 as furniture
from . import multitask_appearance_v5 as appearance

VERSION = 'bulk_furniture_fixed_source_screen_v1'
REGISTRY = ROOT / 'runtime/visual_furniture_bulk_render_v1'
PRIOR_BUDGET = ROOT / 'runs/multitask_appearance_20261030_v5/budget/limits.json'


def register_budget(output, prior_budget, allocation_id, count):
    if type(count) is not int or not 1 <= count <= 4 or not allocation_id:
        raise ValueError('named_one_to_four_screen_allocation_required')
    prior_path = Path(prior_budget); prior = read(prior_path)
    reservations = list((prior_path.parent / 'reservations').glob('*.json'))
    for path in reservations:
        reservation = read(path); old_root = resolve(reservation['protocol']).parent
        if not (old_root / 'cases' / reservation['case']['id'] / 'terminal.json').exists():
            raise ValueError('prior_render_submission_unresolved')
    historical = prior['historical_actual_render_jobs'] + len(reservations)
    REGISTRY.mkdir(parents=True, exist_ok=True)
    target = REGISTRY / 'registrations' / (digest(allocation_id) + '.json')
    with (REGISTRY / '.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists(): raise ValueError('allocation_already_registered_recover_original_directory')
        limits = {'schema': VERSION, 'maximum_attempts': count, 'allocation_id': allocation_id,
            'allocation_limit': count, 'allocation': {'screen': count, 'video': 0},
            'prior_budget': binding(prior_path), 'historical_actual_render_jobs': historical,
            'prior_lineage_limit': prior['maximum_lineage_render_jobs_including_history'],
            'maximum_lineage_render_jobs_including_history': prior['maximum_lineage_render_jobs_including_history']+count,
            'maximum_actual_jobs_including_history': historical+count,
            'unspent_prior_allowance_not_reassigned': True, 'physical_execution_limit': 0, 'api_limit': 0,
            'gpu': 2, 'minimum_free_gpu_mib': 12288, 'parallel_workers': 1,
            'timeout_seconds': {'screen': 240, 'video': 240}, 'resource_wait_seconds': 180,
            'automatic_retries': 0, 'unknown_submission_must_be_recovered': True}
        write(output / 'budget/limits.json', limits)
        write(target, {'allocation_id': allocation_id, 'maximum_attempts': count,
            'root': str(output.resolve()), 'budget': binding(output / 'budget/limits.json'),
            'created_unix': time.time()})
    return binding(target), limits


def prepare(source_specs, output, *, catalog=furniture.CATALOG, prior_budget=PRIOR_BUDGET,
            allocation_id='bulk_furniture_20261030_four_screens'):
    output = Path(output).resolve(); source_specs = Path(source_specs)
    if output.exists(): raise ValueError('preserve_previous_bulk_furniture_render')
    if shutil.disk_usage(ROOT).free < 2*2**30: raise ValueError('insufficient_disk_preserve_history')
    specs = read(source_specs)['sources']
    if not 1 <= len(specs) <= 4 or len({digest(s['execution']) for s in specs}) != len(specs):
        raise ValueError('one_to_four_unique_actual_source_executions_required')
    cases = []; sources = []; receipts = []
    for i, spec in enumerate(specs):
        source = appearance.replay.validate_source(resolve(spec['execution']).parent)
        source_partition = source['bundle']['case'].get('partition')
        partition = spec.get('partition', source_partition)
        if partition != source_partition:
            raise ValueError('background_partition_must_preserve_original_task_partition')
        baseline, repair = appearance.render.baseline_scene(source)
        envelope = furniture.domain.build_motion_envelope(baseline, source['source'] / 'measurements.npz')
        scene, receipt = furniture.apply_overlay(baseline, spec['seed'], motion_envelope=envelope,
            catalog_path=catalog, group=spec['group'], preset=spec['preset'], partition=partition)
        if not receipt['appearance_applied']: raise ValueError('bulk_background_visual_resources_unavailable')
        appearance.assert_invariants(baseline, scene)
        appearance.render.validate_visual_derivative(source['scene'], scene, 'environment_3d')
        case_id = f'screen_{i:02d}_{spec["group"]}'; folder = output / 'cases' / case_id
        write(folder / 'render_scene.json', scene); write(folder / 'appearance_receipt.json', receipt)
        write(folder / 'motion_envelope.json', envelope)
        write(folder / 'source_artifacts.json', source['bindings'])
        for name in appearance.replay.COPY_EVIDENCE: shutil.copy2(source['source'] / name, folder / name)
        count = len(source['states'])
        grasp = next(e['frame'] for e in read(source['source'] / 'audit.json')['event_frames'] if e['id'].endswith('.grasp'))
        source_id = f'source_{i:02d}'
        sources.append({'id': source_id, 'source_execution': source['bindings']['execution_bundle.json'],
            'artifacts': source['bindings'], 'source_execution_code': source['bundle']['code'],
            'frame_count': count, 'fps': float(source['fps']), 'source_spec': spec})
        cases.append({'id': case_id, 'source_id': source_id, 'stage': 'screen', 'variant': 'environment_3d',
            'scene': binding(folder / 'render_scene.json'), 'source_execution': source['bindings']['execution_bundle.json'],
            'state_indices': [0, grasp, count-1], 'visualization_only': True, 'R2H_ready': False,
            'predeclared_control': False, 'screen_case_id': None})
        receipts.append(binding(folder / 'appearance_receipt.json'))
    registration, limits = register_budget(output, prior_budget, allocation_id, len(cases))
    selection = {'schema': VERSION, 'author': 'implementation_agent_using_existing_physical_audits',
        'input_sources': binding(source_specs), 'catalog': binding(catalog), 'sources': sources, 'cases': cases,
        'predeclared_control_video_case_ids': [], 'appearance_receipts': receipts,
        'qualification_scope': 'fixed-source background sampling and state preservation, not all-library visual qualification'}
    write(output / 'selection_protocol.json', selection)
    modules = [Path(__file__), Path(furniture.__file__), Path(appearance.__file__),
        Path(appearance.render.__file__), Path(appearance.replay.__file__), Path(furniture.domain.__file__),
        ROOT/'resimpair_tasks/scene_quality_v3.py', ROOT/'resimpair_tasks/scene_material_policy_v3.py',
        ROOT/'resimpair_tasks/appearance_v1.py', ROOT/'resimpair_pickplace/environment.py',
        ROOT/'resimpair_tasks/cudex_shared_gpu_v1.py',
        ROOT/'corl2026_workshop/experiments/autonomous_pipeline_v4/simulation.py',
        ROOT/'corl2026_workshop/experiments/autonomous_pipeline_v4/resources.py',
        ROOT/'PolicyGen/mani_skill/render/shaders.py', ROOT/'PolicyGen/mani_skill/sensors/camera.py']
    protocol = {'schema': appearance.render.VERSION, 'bulk_background_adapter': binding(__file__),
        'independent_protocol': binding(output/'selection_protocol.json'),
        'selection_protocol_author': selection['author'], 'sources': sources, 'cases': cases,
        'budget': binding(output/'budget/limits.json'), 'budget_registration': registration,
        'code': {str(p.relative_to(ROOT)): binding(p) for p in modules},
        'state_roundtrip_tolerance': 1e-6, 'TCP_tolerance_m': 1e-5,
        'physical_execution_count': 0, 'native_execution': False, 'new_trajectory': False,
        'api_calls': 0, 'R2H_ready': False, 'default_promoted': False}
    write(output/'protocol.json', protocol)
    refs = {'protocol': binding(output/'protocol.json')}
    for reference in appearance.render.bindings_in(protocol):
        resolve(reference); refs['dependency_'+digest(reference)] = reference
    for scene in (read(resolve(c['scene'])) for c in cases):
        for reference in appearance.render.bindings_in(scene):
            resolve(reference); refs['scene_resource_'+digest(reference)] = reference
    write(output/'frozen_inputs.json', refs)
    appearance.render.verify(output)
    result = {'schema': VERSION, 'sources': len(sources), 'prepared_cases': len(cases),
        'actual_render_jobs': 0, 'frames_per_camera_per_case': 3, 'cameras_per_case': 2,
        'distinct_selected_furniture': len({x for p in receipts for x in read(resolve(p))['bulk_policy']['selected']}),
        'historical_actual_render_jobs': limits['historical_actual_render_jobs'],
        'maximum_lineage_render_jobs': limits['maximum_lineage_render_jobs_including_history'],
        'new_physical_executions': 0, 'API_calls': 0}
    write(output/'prepared.json', result)
    return result


def run(output, limit=None):
    protocol, limits = appearance.render.verify(output)
    return appearance.render.run(output, 'screen', limits['allocation']['screen'] if limit is None else limit)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--sources', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--catalog', type=Path, default=furniture.CATALOG)
    p.add_argument('--prior-budget', type=Path, default=PRIOR_BUDGET)
    p.add_argument('--allocation-id', default='bulk_furniture_20261030_four_screens')
    p = sub.add_parser('run'); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(prepare(args.sources, args.output, catalog=args.catalog, prior_budget=args.prior_budget,
                      allocation_id=args.allocation_id))
    else: print(run(args.output, args.limit))
