"""Qualify every metric container-role source with one shared cavity procedure.

Preserve current source-aware scale, source identity and split. A stationary
triangle collider preserves the measured opening; this is geometric container
qualification, not a claim that every payload can be placed successfully.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from collections import Counter
import fcntl
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
from resimpair_assets.metric_variants_v2 import audit_variant
from resimpair_assets.metric_visual_precision_v3 import bake_visual_world
from resimpair_pickplace.containers import cavity
from resimpair_tasks.cudex_bulk_queue_v2 import atomic, process_alive, process_identity

VERSION = 'all_metric_container_source_population_v5'


def classify_error(error):
    from scipy.spatial import QhullError
    geometry_reasons = {
        'no_triangle_geometry', 'container_face_budget', 'nonfinite_container',
        'no_upright_container_support_pose_at_current_metric_scale',
        'container_visual_export_changed_scale_or_origin', 'native_triangle_geometry_changed',
        'visual_graph_contains_non_triangle_geometry', 'visual_graph_transform_must_be_finite',
        'visual_graph_has_no_geometry', 'baked_visual_topology_count_changed', 'baked_visual_world_bounds_changed',
    }
    if isinstance(error, QhullError) or isinstance(error, ValueError) and str(error) in geometry_reasons:
        return 'cavity_or_numeric_failed'
    return 'unknown_system_failure'


def compile_one(job_path):
    import sapien
    from scipy.spatial import cKDTree

    job_path = Path(job_path)
    job = read(job_path)
    output = Path(job['output'])
    if (output / 'worker_started.json').exists():
        raise ValueError('existing_container_attempt_needs_recovery')
    write(output / 'worker_started.json', {'job': binding(job_path),
        'identity': process_identity(os.getpid()), 'unix': time.time()})
    start = time.monotonic()
    if job.get('prerequisite_status') != 'ready':
        result = {'source_id': job['source_id'], 'status': job['prerequisite_status'],
            'reason': job.get('prerequisite_reason'), 'cavity_attempted': False,
            'job': binding(job_path), 'GPU_jobs': 0, 'API_calls': 0}
        write(output / 'result.json', result)
        return result
    reference = job['metric_descriptor']
    descriptor_path = resolve(reference)
    descriptor = read(descriptor_path)
    if descriptor['source_id'] != job['source_id'] or descriptor.get('semantic_role') != 'container':
        raise ValueError('bound_source_and_container_role_required')
    # Reconstruct metric/mass provenance from the original bound policy.
    numeric = audit_variant(descriptor_path)
    write(output / 'numeric_reload.json', numeric)
    scene = trimesh.load(resolve(descriptor['visual']), force='scene', process=False)
    scene = bake_visual_world(scene)
    mesh = scene.to_geometry()
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError('no_triangle_geometry')
    if len(mesh.faces) > 300000:
        raise ValueError('container_face_budget')
    if not np.isfinite(mesh.vertices).all():
        raise ValueError('nonfinite_container')
    hull = mesh.convex_hull
    if len(hull.faces) > 400:
        hull = hull.simplify_quadric_decimation(face_count=400).convex_hull
    poses, probabilities = trimesh.poses.compute_stable_poses(hull, n_samples=1, sigma=0, threshold=.005)
    accepted = []
    orientation_rows = []
    for index, (pose, probability) in enumerate(zip(poses[:10], probabilities[:10])):
        placed = mesh.copy()
        placed.apply_transform(pose)
        center = placed.bounds.mean(axis=0)
        center[2] = placed.bounds[0, 2]
        placed.apply_translation(-center)
        translation = np.eye(4)
        translation[:3, 3] = -center
        transform = translation @ pose
        try:
            opening = cavity(placed)
        except ValueError as error:
            orientation_rows.append({'index': index, 'status': 'no_open_supported_cavity', 'reason': str(error)})
            continue
        accepted.append((opening['flat_floor_radius_m'], placed, transform, opening, float(probability), index))
        orientation_rows.append({'index': index, 'status': 'geometry_candidate', 'cavity': opening})
    write(output / 'orientation_search.json', {'maximum_stable_candidates': 10,
        'rows': orientation_rows, 'size_rescaled': False, 'manual_per_asset_parameters': False})
    if not accepted:
        raise ValueError('no_upright_container_support_pose_at_current_metric_scale')
    _, mesh, transform, opening, probability, pose_index = max(accepted, key=lambda row: row[0])
    visual = scene.copy()
    visual.apply_transform(transform)
    visual = bake_visual_world(visual)
    visual_path = output / 'visual.glb'
    visual.export(visual_path)
    reloaded = trimesh.load(visual_path, force='scene', process=False).to_geometry()
    if not np.allclose(reloaded.bounds, mesh.bounds, rtol=1e-6, atol=1e-7):
        raise ValueError('container_visual_export_changed_scale_or_origin')
    collision_path = output / 'collision.obj'
    mesh.export(collision_path, digits=12)
    native = sapien.physx.PhysxCollisionShapeTriangleMesh(filename=str(collision_path), scale=[1, 1, 1],
        material=sapien.physx.PhysxMaterial(.6, .6, 0.))
    error = max(cKDTree(mesh.vertices).query(native.vertices)[0].max(),
        cKDTree(native.vertices).query(mesh.vertices)[0].max())
    if not np.isfinite(error) or error > 1e-5:
        raise ValueError('native_triangle_geometry_changed')
    variant_id = digest({'parent_metric_descriptor': reference, 'implementation': binding(__file__),
        'transform': transform.tolist(), 'cavity_implementation': binding(ROOT / 'resimpair_pickplace/containers.py')})
    value = {'schema': 'stationary_mesh_container_v1', 'id': variant_id[:24], 'variant_id': variant_id,
        'source_id': descriptor['source_id'], 'name': descriptor['name'], 'partition': descriptor['partition'],
        'provenance': descriptor.get('provenance'), 'source_aliases': job.get('source_aliases', []),
        'visual': binding(visual_path), 'collision': binding(collision_path),
        'source_to_asset': (transform @ np.asarray(descriptor['source_to_asset'])).tolist(),
        'bounds_m': mesh.bounds.tolist(), 'dimensions_m': mesh.extents.tolist(), 'cavity': opening,
        'stable_pose_index': pose_index, 'geometric_support_probability': probability,
        'native_triangle_vertex_error_m': float(error),
        'native_policy': 'static_triangle_mesh_only; shared_metric_cavity_measurement; no_dynamic_grasp_claim',
        'roles': {'stationary_container': True, 'graspable_container': False},
        'metric_size': {'mode': descriptor['size_provenance']['confidence'],
            'parent_metric_descriptor': reference, 'size_policy': descriptor['size_provenance']['policy'],
            'metric_stage_uniform_factor': descriptor['size_provenance']['scale_from_parent'],
            'additional_scale_in_container_stage': 1.0},
        'semantic_status': 'current_source_role_policy_plus_measured_cavity_not_identity_ground_truth',
        'source_alias_split_consistent': job['source_alias_split_consistent'],
        'source_group_partitions': job['source_group_partitions'],
        'physical_status': 'unmeasured', 'relative_scale': descriptor['size_provenance']['selected_relative_scale'],
        'numeric_provenance_reload': binding(output / 'numeric_reload.json'),
        'job': binding(job_path), 'generator': binding(__file__)}
    write(output / 'descriptor.json', value)
    result = {'source_id': descriptor['source_id'], 'variant_id': variant_id,
        'status': 'ready_geometry_only' if job['source_alias_split_consistent'] else 'geometry_qualified_split_conflict',
        'asset': binding(output / 'descriptor.json'),
        'dimensions_m': mesh.extents.tolist(), 'cavity': opening,
        'inner_flat_floor_diameter_m': 2 * opening['flat_floor_radius_m'],
        'source_aliases': job.get('source_aliases', []), 'cavity_attempted': True,
        'elapsed_seconds': time.monotonic() - start, 'physical_success_inferred': False,
        'GPU_jobs': 0, 'API_calls': 0}
    write(output / 'result.json', result)
    return result


def prepare(cpu_root, root):
    cpu_root, root = Path(cpu_root), Path(root).resolve()
    if root.exists():
        raise FileExistsError(root)
    status = read(cpu_root / 'status.json')
    if status['counts'].get('pending', 0) or status['counts'].get('running', 0):
        raise ValueError('finish_current_5524_CPU_scope_before_container_batch')
    states = read(cpu_root / 'source_state_index.json')
    cpu_protocol = read(cpu_root / 'protocol.json')
    manifest = read(resolve(cpu_protocol['manifest']))
    by_source = {row['source_id']: row for row in states['entries']}
    selected = [job for job in manifest['jobs'] if job['decision']['role'] == 'container']
    if len(selected) != states['summary']['roles']['container']:
        raise ValueError('container_source_scope_disagrees_with_frozen_plan')
    root.mkdir(parents=True)
    write(root / 'source_state_snapshot.json', states)
    jobs = []
    for parent in selected:
        source = by_source[parent['source_id']]
        partitions = sorted({by_source[sid]['partition'] for sid in
            [parent['source_id']] + parent.get('source_aliases', []) if sid in by_source})
        variants = [e for e in source.get('variants', []) if e.get('role') == 'container']
        ready = source.get('semantic_admission') != 'pending' and len(variants) == 1
        row = {'source_id': parent['source_id'], 'key': digest(parent['source_id'])[:24],
            'source_aliases': parent.get('source_aliases', []), 'source_decision': parent['decision'],
            'source_group_partitions': partitions, 'source_alias_split_consistent': len(partitions) == 1,
            'metric_descriptor': variants[0]['asset'] if ready else None,
            'CPU_terminal': source.get('terminal'), 'CPU_status': source.get('batch_status'),
            'prerequisite_status': 'ready' if ready else 'prerequisite_pending_or_failed',
            'prerequisite_reason': None if ready else 'no_single_numerically_qualified_container_descriptor_under_current_policy'}
        jobs.append(row)
    protocol = {'schema': VERSION, 'source_plan': manifest['source_state_index'],
        'CPU_protocol': binding(cpu_root / 'protocol.json'), 'source_state_snapshot': binding(root / 'source_state_snapshot.json'),
        'jobs': jobs, 'maximum_sources': len(jobs), 'maximum_CPU_workers': 16,
        'per_source_timeout_seconds': 120, 'per_source_attempts': 1, 'automatic_retries': False,
        'deadline_unix': cpu_protocol['deadline_unix'], 'created_unix': time.time(),
        'code': [binding(__file__), binding(ROOT / 'resimpair_assets/metric_variants_v2.py'),
            binding(ROOT / 'resimpair_assets/metric_visual_precision_v3.py'),
            binding(ROOT / 'resimpair_pickplace/containers.py')],
        'previous_container_catalogs_preserved': True, 'size_source_policy_unchanged': True,
        'GPU_jobs': 0, 'API_calls': 0, 'dynamic_steps': 0}
    write(root / 'protocol.json', protocol)
    for row in jobs:
        folder = root / 'jobs' / row['key']
        write(folder / 'job.json', {**row, 'output': str(folder), 'protocol': binding(root / 'protocol.json')})
    return {'planned_sources': len(jobs), 'ready_prerequisites': sum(x['prerequisite_status'] == 'ready' for x in jobs)}


def refresh(root):
    root = Path(root)
    protocol = read(root / 'protocol.json')
    rows = []
    assets = []
    for job in protocol['jobs']:
        folder = root / 'jobs' / job['key']
        terminal = read(folder / 'terminal.json') if (folder / 'terminal.json').exists() else {'status': 'running' if (folder / 'process.json').exists() else 'pending'}
        result = read(folder / 'result.json') if (folder / 'result.json').exists() else {}
        if terminal['status'] == 'ready_geometry_only':
            assets.append(result['asset'])
        rows.append({'source_id': job['source_id'], **result, **terminal})
    summary = {'schema': VERSION, 'source_count': len(rows), 'counts': dict(Counter(x['status'] for x in rows)),
        'ready_container_sources': len(assets), 'updated_unix': time.time(), 'API_calls': 0, 'GPU_jobs': 0}
    atomic(root / 'catalog.json', {'schema': 'stationary_container_catalog_v1', 'assets': assets,
        'rows': rows, 'parents': [binding(root / 'protocol.json')], 'physics_executions': 0, 'api_calls': 0})
    atomic(root / 'status.json', summary)
    return summary


def run(root, workers=16, maximum=None):
    root = Path(root).resolve()
    protocol = read(root / 'protocol.json')
    for ref in protocol['code']:
        resolve(ref)
    if not 1 <= workers <= protocol['maximum_CPU_workers']:
        raise ValueError('CPU_worker_budget_exceeded')
    if maximum is not None and maximum <= 0:
        raise ValueError('positive_source_limit_required')
    with (root / '.run.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pending = []
        for job in protocol['jobs']:
            path = root / 'jobs' / job['key'] / 'job.json'
            folder = path.parent
            if (folder / 'terminal.json').exists():
                continue
            if (folder / 'process.json').exists() or (folder / 'worker_started.json').exists():
                if (folder / 'process.json').exists() and process_alive(read(folder / 'process.json')['identity']):
                    raise RuntimeError('existing_live_container_process_do_not_relaunch')
                if (folder / 'result.json').exists():
                    result = read(folder / 'result.json')
                    write(folder / 'terminal.json', {'status': result['status'], 'recovered_saved_result': True,
                        'result': binding(folder / 'result.json')})
                    continue
                raise RuntimeError('existing_container_process_or_unknown_result_recover_without_resubmission')
            pending.append(path)
        if maximum is not None:
            pending = pending[:maximum]

        def execute(path):
            folder = path.parent
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            start = time.monotonic()
            with (folder / 'worker.log').open('x') as log:
                child = subprocess.Popen([sys.executable, '-u', '-m', 'resimpair_assets.metric_container_population_v5',
                    'worker', '--job', str(path)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                write(folder / 'process.json', {'identity': process_identity(child.pid), 'unix': time.time()})
                try:
                    code = child.wait(timeout=protocol['per_source_timeout_seconds'])
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    code = 124
            if not code and (folder / 'result.json').exists():
                result = read(folder / 'result.json')
                terminal = {'status': result['status'], 'result': binding(folder / 'result.json')}
            elif code == 124:
                terminal = {'status': 'source_timeout', 'returncode': code, 'retry': False}
            else:
                failure = read(folder / 'failure.json') if (folder / 'failure.json').exists() else {}
                terminal = {'status': failure.get('classification', 'unknown_system_failure'),
                    'reason': failure.get('error'), 'returncode': code, 'retry': False,
                    'failure': binding(folder / 'failure.json') if failure else None}
            terminal['elapsed_seconds'] = time.monotonic() - start
            write(folder / 'terminal.json', terminal)
            return terminal

        iterator = iter(pending)
        active = {}
        exhausted = False
        unknown = False
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while active or not exhausted:
                while len(active) < workers and not exhausted and not unknown:
                    if time.time() >= protocol['deadline_unix'] or (root / 'STOP').exists():
                        exhausted = True
                        break
                    path = next(iterator, None)
                    if path is None:
                        exhausted = True
                        break
                    active[pool.submit(execute, path)] = path
                if not active:
                    break
                done, _ = wait(active, timeout=5, return_when=FIRST_COMPLETED)
                for future in done:
                    active.pop(future)
                    value = future.result()
                    completed += 1
                    if completed % 25 == 0:
                        print(refresh(root), flush=True)
                    if value['status'] == 'unknown_system_failure':
                        unknown = True
                        exhausted = True
        result = refresh(root)
        if unknown:
            raise RuntimeError('unknown_container_system_failure_paused_new_admission')
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'worker', 'status'])
    parser.add_argument('--root', type=Path)
    parser.add_argument('--cpu-root', type=Path)
    parser.add_argument('--job', type=Path)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--maximum', type=int)
    args = parser.parse_args()
    if args.command == 'worker':
        try:
            print(compile_one(args.job))
        except Exception as error:
            # Bad geometry is isolated; missing bindings and other unexpected
            # infrastructure errors remain separate and stop new admission.
            write(args.job.parent / 'failure.json', {'error_type': type(error).__name__, 'error': str(error),
                'traceback': traceback.format_exc(),
                'classification': classify_error(error)})
            raise
    elif args.command == 'prepare':
        print(prepare(args.cpu_root, args.root))
    elif args.command == 'run':
        print(run(args.root, args.workers, args.maximum))
    else:
        print(refresh(args.root))
