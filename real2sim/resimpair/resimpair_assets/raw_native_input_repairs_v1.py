"""One source attempt: exact visual transforms, then bounded native proposals.

The single convex hull is actually qualified first. Only a failed first proposal
allows the original VHACD32 recipe. Both use the unchanged v5/v4/v3 native tests
with v2's exact-return-file cooker. No old queue, cache or geometry is edited.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
import trimesh

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write
from . import exact_scene_transforms_v1 as exact
from . import metric_native_v4, metric_native_v5, metric_native_v6
from . import metric_variants_v2, raw_mesh_io_v1, raw_manipulation_v2
from .metric_admission_v1 import inspect
from .raw_size_policy_v1 import decide_raw, metric_policy

VERSION = 'raw_native_input_repairs_v1'
WORKER_ENV = 'RESIMPAIR_NATIVE_INPUT_REPAIRS_JOB_SHA256'
MAX_SETTLE_STEPS = 1440
PROPOSALS = ('single_convex_hull', 'original_VHACD32_fallback')
IMPLEMENTATIONS = tuple(dict.fromkeys((*raw_manipulation_v2.IMPLEMENTATIONS,
    'resimpair_assets/raw_native_input_repairs_v1.py',
    'resimpair_assets/exact_scene_transforms_v1.py',
    'resimpair_assets/native_reload_preflight_v1.py',
    'resimpair_assets/raw_metric_containers_v1.py',
    'resimpair_assets/metric_visual_precision_v3.py',
    'resimpair_assets/metric_inventory_v7.py',
    'resimpair_assets/metric_semantics_v1.py',
    'runtime/collision_cpu_deps/vhacdx.cpython-312-x86_64-linux-gnu.so',
    '.venv/lib/python3.12/site-packages/trimesh/scene/transforms.py',
    '.venv/lib/python3.12/site-packages/trimesh/scene/scene.py',
    '.venv/lib/python3.12/site-packages/trimesh/base.py',
    'configs/metric_numeric_bounds_v2.json',
    'configs/metric_population_v7.json')))


def _env():
    return {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}


@contextmanager
def settle_budget(output):
    previous = metric_native_v4.settle
    started = 0

    def bounded(reference, destination):
        nonlocal started
        if started + 360 > MAX_SETTLE_STEPS:
            raise ValueError('source_native_settle_step_budget_exhausted')
        started += 360
        write(Path(output) / 'settle_reservations' / f'{started // 360:02d}.json', {
            'descriptor': reference, 'result_path': str(destination),
            'maximum_steps_this_call': 360, 'source_reserved_step_upper_bound': started,
            'original_native_settle_implementation': binding(metric_native_v4.__file__)})
        return previous(reference, destination)

    metric_native_v4.settle = bounded
    try:
        yield
    finally:
        metric_native_v4.settle = previous


def prepare_raw(job, folder):
    """Retained source policy and support-frame algorithm; no initial hull gate."""
    row = job['raw_row']
    scene, mesh, raw_receipt = raw_mesh_io_v1.load_raw_scene(row)
    raw_receipt['exact_scene_transform_adapter'] = exact.receipt()
    write(folder / 'raw_input.json', raw_receipt)
    hull = mesh.convex_hull
    original_obb = np.asarray(hull.bounding_box_oriented.primitive.extents)
    decision = decide_raw(row, original_obb, metadata_records=job.get('metadata_records', ()), seed=job['seed'])
    write(folder / 'size_decision.json', decision)
    if decision['role'] != 'manipulation':
        return None, decision, None
    scale = decision['source_unit_to_m']
    scaled_hull = hull.copy(); scaled_hull.apply_scale(scale)
    stable = scaled_hull.simplify_quadric_decimation(face_count=400).convex_hull if len(scaled_hull.faces) > 400 else scaled_hull
    poses, probabilities = trimesh.poses.compute_stable_poses(stable, n_samples=1, sigma=0., threshold=.005)
    if not len(poses):
        raise ValueError('raw_metric_no_geometric_support_pose')
    orientation = poses[int(np.argmax(probabilities))]
    transform = orientation @ np.diag([scale, scale, scale, 1.])
    visual = scene.copy(); visual.apply_transform(transform)
    translation = -visual.bounds.mean(axis=0)
    transform[:3, 3] += translation; visual.apply_translation(translation)
    write(folder / 'raw_frame.json', {'source_to_initial_asset': transform.tolist(),
        'source_basis_directions_in_initial_asset': orientation[:3, :3].tolist(),
        'initial_support_pose_selection': 'maximum_geometric_stable_pose_probability',
        'semantic_upright_axis': 'unknown_not_inferred_from_geometric_support',
        'native_settling_may_apply_a_further_canonical_transform': True})
    actual = visual.to_geometry()
    visual_path = folder / 'raw_metric/visual.glb'
    material = raw_mesh_io_v1.export_visual(visual, visual_path, actual)
    write(folder / 'raw_metric/material_export.json', material)
    canonical_hull = actual.convex_hull
    collision_path = folder / 'raw_metric/initial_hull.obj'
    canonical_hull.export(collision_path, digits=12)
    parent = {'schema': 'resimpair_mesh_asset_v1', 'id': digest([row['id'], raw_receipt, decision])[:24],
        'source_id': row['id'], 'name': row.get('name', row['id']), 'category': decision['category'],
        'partition': row['partition'], 'provenance': {'source': row['source'], 'original_sha256': row['source_sha'],
            'raw_input': binding(folder / 'raw_input.json'), 'raw_size_decision': binding(folder / 'size_decision.json'),
            'license': job.get('license'), 'prior_native': job.get('prior_native'),
            'source_aliases': job.get('source_aliases', []), 'ingestion_group': job['ingestion_group']},
        'visual': binding(visual_path), 'collisions': [binding(collision_path)],
        'source_to_asset': transform.tolist(), 'dimensions_m': actual.extents.tolist(),
        'bounds_m': actual.bounds.tolist(), 'half_size': (actual.extents / 2).tolist(),
        'collision_bounds_m': canonical_hull.bounds.tolist(),
        'semantic_role': 'manipulation', 'semantic_category': decision['category'],
        'qualification': {'geometry': 'pending', 'physical_execution': 'unmeasured'},
        'roles': {'rigid_manipulation_candidate': False, 'static_distractor': False},
        'mass_kg': float(canonical_hull.volume * decision['effective_density_kg_m3']),
        'density_kg_m3': decision['effective_density_kg_m3'], 'raw_source_uniform_scale': scale,
        'new_grasp_required': True}
    write(folder / 'raw_metric/parent.json', parent)
    policy = metric_policy(decision, binding(folder / 'size_decision.json'))
    write(folder / 'metric_policy.json', policy)
    state = {'source_id': row['id'], 'role': 'manipulation', 'profile': decision['category'],
        'partition': row['partition'], 'semantic_admission': 'engineered_role_and_scale_candidate',
        'base_longest_m': decision['target_longest_m'], 'size_confidence': decision['size_confidence'],
        'policy': binding(folder / 'metric_policy.json'), 'raw_size_decision': binding(folder / 'size_decision.json')}
    write(folder / 'source_state.json', state)
    return binding(folder / 'raw_metric/parent.json'), decision, state


def final_replay(entry_path, output):
    """A fresh process, one exact-file actor build, zero simulation steps."""
    from . import native_reload_preflight_v1
    from .stable_cook_confirmed_v2 import pointset_error
    entry = read(entry_path); descriptor = read(resolve(entry['asset']))
    output = Path(output)
    result = native_reload_preflight_v1.check([entry], output / 'retained_reload')
    row = result['rows'][0]
    checks = {'original_geometry_mass_evaluator_passed': row['passed'],
              'native_geometry_and_settle_qualified': entry['geometry_qualified'],
              'original_admission_passed': entry['semantic_scale_admission']['passed']}
    part_rows = []
    if row.get('actual_native_geometry'):
        actual = read(resolve(row['actual_native_geometry']))['objects'][0]
        checks['all_final_parts_loaded'] = len(actual['shapes']) == len(descriptor['collisions']) == len(descriptor['cooking'])
        for number, (ref, shape, passes) in enumerate(zip(descriptor['collisions'], actual['shapes'], descriptor['cooking'])):
            mesh = trimesh.load(resolve(ref), force='mesh', process=False)
            error = pointset_error(mesh.vertices, shape['vertices'])
            confirmation = read(resolve(passes[-1]['confirmation']))
            part_rows.append({'part': number, 'collision': ref, 'input_vertices': len(mesh.vertices),
                'actual_native_vertices': len(shape['vertices']), 'vertex_error_m': error,
                'strict_1um_passed': error < 1e-6, 'original_10um_passed': error < 1e-5,
                'confirmed_exact_return_file': ref == passes[-1]['input'] == passes[-1]['output'] == confirmation['input']
                    and passes[-1]['returned_file_exported_after_check'] is False,
                'helper_confirmation': passes[-1]['confirmation'], 'helper_pid': confirmation['worker_pid']})
        checks['all_parts_pass_exact_file_confirmation'] = bool(part_rows) and all(
            p['strict_1um_passed'] and p['original_10um_passed'] and p['confirmed_exact_return_file'] for p in part_rows)
    else:
        checks['all_final_parts_loaded'] = False
    value = {'schema': VERSION, 'entry': binding(entry_path), 'asset': entry['asset'],
        'passed': all(checks.values()), 'checks': checks, 'parts': part_rows,
        'retained_original_evaluator': binding(output / 'retained_reload/RESULTS.json'),
        'worker_pid': os.getpid(), 'parent_pid': os.getppid(), 'native_actor_builds': 1,
        'simulation_steps': 0, 'robot_executions': 0, 'GPU_jobs': 0, 'API_calls': 0}
    write(output / 'RESULT.json', value)
    return value


def _fresh_replay(entry_path, output):
    output.mkdir(parents=True)
    with (output / 'stdout.log').open('x') as stdout, (output / 'stderr.log').open('x') as stderr:
        proc = subprocess.run([sys.executable, '-m', 'resimpair_assets.raw_native_input_repairs_v1',
            '_review', '--input', str(entry_path), '--output', str(output)], cwd=ROOT, env=_env(),
            stdout=stdout, stderr=stderr, timeout=120, check=False)
    if proc.returncode or not (output / 'RESULT.json').exists():
        raise RuntimeError('fresh_final_native_actor_review_did_not_complete')
    result = read(output / 'RESULT.json')
    if result['parent_pid'] != os.getpid() or result['worker_pid'] == os.getpid():
        raise ValueError('fresh_final_native_replay_process_required')
    return result


def compile_source(original_job_path, folder):
    original = read(original_job_path); row = original['raw_row']; folder = Path(folder)
    parent, decision, state = prepare_raw(original, folder)
    if parent is None:
        return {'source_id': row['id'], 'status': 'source_size_or_role_requires_other_use',
                'decision': binding(folder / 'size_decision.json'), 'candidates': []}
    candidates = []
    selected = None
    for number, representation in enumerate(PROPOSALS):
        candidate = folder / 'candidates' / f'{number:02d}_{representation}'
        write(candidate / 'PROPOSAL.json', {'source_id': row['id'], 'source_attempt': 1,
            'proposal_index': number, 'representation': representation, 'raw_parent': parent,
            'trigger': 'predeclared_first_candidate' if number == 0 else 'first_candidate_failed_complete_native_qualification',
            'maximum_native_settle_steps': 720, 'no_nested_native_v6_refinement': True})
        began = time.monotonic()
        native_started = False
        try:
            proposed = parent if number == 0 else metric_native_v6.refine(parent, candidate / 'decomposition')
            variant = metric_variants_v2.materialize(resolve(proposed), folder / 'metric_policy.json', candidate / 'metric_variant')
            descriptor_path = resolve(variant['variants'][0]['descriptor'])
            write(candidate / 'NATIVE_START.json', {'descriptor': binding(descriptor_path), 'unix': time.time(),
                'native_compiler': binding(metric_native_v5.__file__), 'source_attempt': 1, 'proposal_index': number})
            native_started = True
            native = metric_native_v5.compile_variant(descriptor_path, candidate / 'native')
            entry = {**native, 'source_aliases': original.get('source_aliases', []),
                'ingestion_group': original['ingestion_group'], 'role': 'manipulation',
                'source_state': binding(folder / 'source_state.json'), 'new_grasp_required': True,
                'previous_grasp_cache_reuse_authorized': False, 'native_cook_method': 'stable_cook_confirmed_v2',
                'native_input_repair_implementation': binding(__file__), 'proposal': binding(candidate / 'PROPOSAL.json')}
            gate = inspect(entry, state) if native['geometry_qualified'] else {
                'passed': False, 'reasons': ['native_geometry_or_support_not_qualified'],
                'current_policy': binding(folder / 'metric_policy.json')}
            entry['semantic_scale_admission'] = gate
            write(candidate / 'native_entry.json', entry)
            passed = native['geometry_qualified'] and gate['passed']
            replay = None
            if passed:
                replay = _fresh_replay(candidate / 'native_entry.json', candidate / 'final_reload')
                passed = replay['passed']
            value = {'representation': representation, 'status': 'native_ready' if passed else 'native_qualification_failed',
                'native_started': True, 'native_result': binding(candidate / 'native/result.json'),
                'entry': binding(candidate / 'native_entry.json'), 'asset': native['asset'],
                'surface': native['surface'], 'semantic_scale_admission': gate,
                'final_fresh_reload': binding(candidate / 'final_reload/RESULT.json') if replay is not None else None,
                'elapsed_seconds': time.monotonic() - began}
            if passed:
                selected = {**entry, 'fresh_final_native_reload': value['final_fresh_reload']}
        except Exception as error:
            from scipy.spatial import QhullError
            known = isinstance(error, (ValueError, QhullError))
            value = {'representation': representation,
                'status': 'geometry_or_policy_failed' if known else 'unknown_system_failure',
                'native_started': native_started, 'reason': type(error).__name__ + ':' + str(error),
                'traceback': traceback.format_exc(), 'elapsed_seconds': time.monotonic() - began}
        write(candidate / 'RESULT.json', value)
        candidates.append({'proposal': binding(candidate / 'PROPOSAL.json'), 'result': binding(candidate / 'RESULT.json'), **value})
        if selected is not None or value['status'] == 'unknown_system_failure':
            break
    if selected is not None:
        write(folder / 'native_entry.json', selected)
    return {'source_id': row['id'], 'status': 'native_ready' if selected is not None else 'native_qualification_failed',
        'entry': binding(folder / 'native_entry.json') if selected is not None else None,
        'state': binding(folder / 'source_state.json'), 'size_decision': binding(folder / 'size_decision.json'),
        'candidates': candidates, 'selected_representation': next((c['representation'] for c in candidates if c['status'] == 'native_ready'), None)}


def _validate_job(path):
    path = Path(path).resolve(); job = read(path)
    if job.get('schema') != VERSION or job['maximum_source_attempts'] != 1 or job['timeout_seconds'] != 900:
        raise ValueError('registered_single_900s_source_attempt_required')
    protocol = read(resolve(job['protocol']))
    if protocol['maximum_sources'] > 6 or protocol['maximum_CPU_workers'] != 2:
        raise ValueError('six_source_two_worker_protocol_required')
    selected = {item['source']['source_id']: item['source'] for item in read(resolve(protocol['preregistration']))['selected']}
    if selected[job['source_id']]['job'] != job['original_job']:
        raise ValueError('source_not_in_frozen_six_representatives')
    if Path(job['output']).resolve() != path.parent / 'compiled':
        raise ValueError('fresh_bound_output_required')
    for ref in [job['original_job'], job['old_cpu_result'], *protocol['implementation_bindings']]:
        resolve(ref)
    return path, job, protocol


def worker(job_path):
    job_path, job, protocol = _validate_job(job_path)
    if os.environ.get(WORKER_ENV) != binding(job_path)['sha256']:
        raise ValueError('separate_registered_worker_required')
    folder = job_path.parent
    if (folder / 'worker_started.json').exists() or Path(job['output']).exists():
        raise ValueError('one_source_attempt_only_no_retry')
    write(folder / 'worker_started.json', {'job': binding(job_path), 'pid': os.getpid(), 'unix': time.time()})
    began = time.monotonic()
    try:
        with exact.retained_io_scope(), raw_manipulation_v2._confirmed_cook_hook(), settle_budget(job['output']):
            value = compile_source(resolve(job['original_job']), job['output'])
    except Exception as error:
        value = {'source_id': job['source_id'], 'status': 'geometry_or_policy_failed',
            'reason': type(error).__name__ + ':' + str(error), 'traceback': traceback.format_exc()}
    for ref in [job['original_job'], job['old_cpu_result'], *protocol['implementation_bindings']]:
        resolve(ref)
    accounting = raw_manipulation_v2.accounting(job['output'])
    reservations = list(Path(job['output']).glob('settle_reservations/*.json'))
    accounting['reserved_settle_steps_upper_bound'] = len(reservations) * 360
    if accounting['reserved_settle_steps_upper_bound'] > MAX_SETTLE_STEPS:
        raise ValueError('source_settle_budget_violation')
    value = {**value, 'schema': VERSION, 'source_attempts': 1,
        'representation_proposals': len(list(Path(job['output']).glob('candidates/*/PROPOSAL.json'))),
        'native_candidates_started': len(list(Path(job['output']).glob('candidates/*/NATIVE_START.json'))),
        'original_job': job['original_job'], 'original_result': job['old_cpu_result'], 'protocol': job['protocol'],
        'worker_pid': os.getpid(), 'parent_pid': os.getppid(), 'wall_seconds': time.monotonic() - began,
        'cpu_accounting': accounting, 'new_pose_candidates': 0, 'physical_success': 'unmeasured',
        'robot_executions': 0, 'GPU_jobs': 0, 'API_calls': 0, 'old_queue_modified': False,
        'all_original_bindings_unchanged': True}
    write(folder / 'cpu_result.json', value)
    return value


def run_source(job_path):
    job_path, job, _ = _validate_job(job_path)
    folder = job_path.parent
    if (folder / 'LAUNCH.json').exists() or (folder / 'SUPERVISOR.json').exists():
        raise ValueError('one_source_launch_only_no_retry')
    write(folder / 'LAUNCH.json', {'job': binding(job_path), 'source_attempt': 1, 'unix': time.time(),
        'supervisor_pid': os.getpid(), 'timeout_seconds': 900})
    env = _env(); env[WORKER_ENV] = binding(job_path)['sha256']
    began = time.monotonic(); status = 'worker_completed'
    with (folder / 'stdout.log').open('x') as stdout, (folder / 'stderr.log').open('x') as stderr:
        process = subprocess.Popen([sys.executable, '-m', 'resimpair_assets.raw_native_input_repairs_v1',
            '_worker', '--job', str(job_path)], cwd=ROOT, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        write(folder / 'WORKER_PROCESS.json', {'pid': process.pid, 'supervisor_pid': os.getpid(), 'process_group_id': process.pid})
        try:
            process.wait(timeout=900)
        except subprocess.TimeoutExpired:
            status = 'worker_timeout'
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
    result_path = folder / 'cpu_result.json'
    if status == 'worker_completed' and (process.returncode != 0 or not result_path.exists()):
        status = 'worker_failed_without_complete_result'
    report = {'schema': VERSION, 'source_id': job['source_id'], 'status': status,
        'job': binding(job_path), 'worker_result': binding(result_path) if result_path.exists() else None,
        'returncode': process.returncode, 'worker_pid': process.pid, 'wall_seconds': time.monotonic() - began,
        'cpu_accounting': raw_manipulation_v2.accounting(job['output']), 'maximum_source_attempts': 1,
        'GPU_jobs': 0, 'API_calls': 0, 'robot_executions': 0}
    write(folder / 'SUPERVISOR.json', report)
    return report


def prepare(preregistration, report_root, run_root):
    report_root, run_root = Path(report_root).resolve(), Path(run_root).resolve()
    selected = read(preregistration)['selected']
    if len(selected) > 6 or run_root.exists() or (report_root / 'PROTOCOL.json').exists():
        raise ValueError('fresh_at_most_six_source_campaign_required')
    implementations = [binding(ROOT / p) for p in IMPLEMENTATIONS]
    protocol = {'schema': VERSION, 'preregistration': binding(preregistration),
        'implementation_bindings': implementations, 'maximum_sources': len(selected),
        'maximum_CPU_workers': 2, 'source_timeout_seconds': 900, 'campaign_timeout_seconds': 3600,
        'maximum_source_attempts': 1, 'automatic_retries': 0, 'maximum_representation_proposals_per_source': 2,
        'proposal_order': list(PROPOSALS), 'fallback': 'only when complete first representation qualification fails',
        'decomposition_recipe': metric_native_v6.RECIPE, 'native_qualifier': 'unchanged metric_native_v5/v4/v3',
        'native_cooker': 'stable_cook_confirmed_v2 via raw_manipulation_v2._confirmed_cook_hook',
        'exact_visual_transform_transport': exact.receipt(), 'maximum_dynamic_steps_per_source': MAX_SETTLE_STEPS,
        'surface_p95_threshold_m': .003, 'native_convergence_threshold_m': 1e-6,
        'native_runtime_replay_threshold_m': 1e-5, 'numeric_and_admission_bounds_changed': False,
        'maximum_fresh_full_asset_replays_per_proposal': 1, 'fresh_final_replay_simulation_steps': 0,
        'robot_executions': 0, 'GPU_jobs': 0, 'API_calls': 0, 'old_queue_modified': False,
        'new_pose_candidates': 0, 'physical_success': 'unmeasured', 'run_root': str(run_root), 'created_unix': time.time()}
    write(report_root / 'PROTOCOL.json', protocol)
    jobs = []
    for index, item in enumerate(selected):
        row = item['source']; folder = run_root / f'{index:02d}_{digest(row["source_id"])[:16]}'
        job = {'schema': VERSION, 'source_id': row['source_id'], 'original_job': row['job'],
            'old_cpu_result': row['cpu_result'], 'protocol': binding(report_root / 'PROTOCOL.json'),
            'output': str(folder / 'compiled'), 'maximum_source_attempts': 1, 'timeout_seconds': 900}
        write(folder / 'job.json', job); jobs.append(binding(folder / 'job.json'))
    write(report_root / 'JOBS.json', {'protocol': binding(report_root / 'PROTOCOL.json'), 'jobs': jobs})
    return jobs


def campaign(report_root):
    root = Path(report_root); planned = read(root / 'JOBS.json')
    if (root / 'CAMPAIGN_START.json').exists():
        raise ValueError('single_campaign_launch_only_no_retry')
    write(root / 'CAMPAIGN_START.json', {'jobs': binding(root / 'JOBS.json'), 'unix': time.time(), 'pid': os.getpid()})
    began = time.monotonic(); reports = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(run_source, resolve(ref)): ref for ref in planned['jobs']}
        for future in as_completed(futures, timeout=3600):
            report = future.result(); reports.append(report)
            print(report['source_id'], report['status'], round(report['wall_seconds'], 2), flush=True)
    for ref in read(resolve(planned['protocol']))['implementation_bindings']:
        resolve(ref)
    result = {'schema': VERSION, 'source_attempts': len(reports), 'reports': reports,
        'wall_seconds': time.monotonic() - began, 'maximum_CPU_workers': 2,
        'all_implementation_bindings_unchanged': True, 'robot_executions': 0, 'GPU_jobs': 0, 'API_calls': 0}
    write(root / 'CAMPAIGN_RESULT.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'campaign', '_worker', '_review'))
    parser.add_argument('--job', type=Path); parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path); parser.add_argument('--report-root', type=Path)
    parser.add_argument('--run-root', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        value = prepare(args.input, args.report_root, args.run_root)
    elif args.command == 'campaign':
        value = campaign(args.report_root)
    elif args.command == '_worker':
        value = worker(args.job)
    else:
        value = final_replay(args.input, args.output)
    print(value)


if __name__ == '__main__':
    main()
