"""Independent, finite CPU native-asset queue; no trajectory or model API.

The full source population is frozen once. Runs consume new rows incrementally
and recover completed outputs; an unresolved reservation is never resubmitted.
Native geometry, geometric pose proposals, source semantics and dynamic success
remain different fields. The retained converter and independent rules are read
without modification. PhysX is used only for CPU collision cooking/mass queries;
the worker never advances a physics scene or opens a renderer.
"""
import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write

VERSION = 'incremental_native_asset_library_v1'
CONVERTER = 'resimpair_pickplace.vhacd_conversion_v2'
CODE_PATHS = [__file__, ROOT/'resimpair_pickplace/vhacd_conversion_v2.py',
              ROOT/'resimpair_grasp/assets.py', ROOT/'resimpair_tabletop/materialize.py',
              ROOT/'resimpair_tabletop/native_mass.py']


def replace_derived(path, value):
    """Only mutable index pointers use replacement; raw attempts stay immutable."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.{time.time_ns()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    os.replace(temporary, path)


@contextmanager
def lock(path, blocking=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


def ordered(rows, seed):
    """Deterministic source/partition/category interleaving, never outcome based."""
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row['source'], row['partition'], row.get('category') or '')].append(row)
    for key, values in buckets.items():
        values.sort(key=lambda x: hashlib.sha256(f"{seed}:{x['source_id']}".encode()).hexdigest())
        buckets[key] = deque(values)
    sources = defaultdict(dict)
    for key, bucket in buckets.items(): sources[key[0]][key] = bucket
    per_source = {}
    for source, groups in sources.items():
        values = deque()
        while any(groups.values()):
            for key in sorted(groups):
                if groups[key]: values.append(groups[key].popleft())
        per_source[source] = values
    result = []
    while any(per_source.values()):
        for source in sorted(per_source):
            if per_source[source]: result.append(per_source[source].popleft())
    return result


def content_key(descriptor, variant, converter_bindings):
    return digest({'visual': descriptor['visual']['sha256'],
                   'source_to_asset': variant['source_to_asset'],
                   'dimensions_m': variant['dimensions_m'],
                   'converter': converter_bindings})[:32]


def semantic_notes(descriptor):
    text = str(descriptor.get('name', ''))+' '+str(descriptor.get('category', ''))
    words = ['mattress', 'sofa', 'bed ', 'bedframe', 'dining table', 'treadmill',
             'rowing machine', 'wardrobe', 'chandelier', 'micro scene', 'room scene']
    flags = [word for word in words if word in text.lower()]
    return {'semantic_identity': descriptor.get('semantic_identity', 'unknown'),
            'real_world_scale': 'unknown', 'size': descriptor.get('size'),
            'furniture_or_multi_object_name_flags': flags,
            'flags_are_retrieval_hints_not_semantic_truth': True,
            'task_category_verified': False}


def initialize(root, population, containers, workers=8, timeout=120,
               maximum_wall_seconds=43200, min_free_gib=50, seed=20261028):
    root = Path(root).resolve()
    if root.exists(): raise ValueError('Existing library queue; use status/run, never reset it')
    if not 1 <= workers <= 16 or not 30 <= timeout <= 300 or not 60 <= maximum_wall_seconds <= 86400:
        raise ValueError('Finite worker/time limits required')
    data = read(population); rows = data['unprocessed_pool']
    source_ids = [r['source_id'] for r in rows]
    if len(set(source_ids)) != len(source_ids): raise ValueError('Duplicate source IDs in input population')
    code = [binding(p) for p in CODE_PATHS]
    canonical = {}; selected = []; excluded = []
    for row in ordered(rows, seed):
        descriptor = read(resolve(row['asset']))
        variant = next(v for v in descriptor['variants'] if v['id'] == descriptor['selected_variant'])
        size = variant['dimensions_m']
        if max(size) > .18 or size[2] < .018:
            excluded.append({'source_id': row['source_id'], 'reason': 'outside_retained_size_gate'}); continue
        key = content_key(descriptor, variant, code[1:])
        item = {**row, 'id': 'asset_'+key, 'content_key': key,
                'name': descriptor.get('name'), 'dimensions_m': size,
                'offline_geometric_pose_count': len(variant['candidates']),
                'semantic_notes': semantic_notes(descriptor)}
        if key in canonical:
            item['content_alias_of'] = canonical[key]
        else: canonical[key] = row['source_id']
        selected.append(item)
    root.mkdir(parents=True)
    registry = ROOT/'runtime/incremental_native_asset_registry_v1'
    protocol = {'schema': VERSION, 'population': binding(population), 'code': code,
        'converter_module': CONVERTER, 'source_count': len(selected), 'unique_geometry_jobs': len(canonical),
        'size_gate': {'maximum_dimension_m': .18, 'minimum_selected_height_m': .018},
        'max_cpu_workers': workers, 'worker_timeout_seconds': timeout,
        'max_wall_seconds': maximum_wall_seconds, 'min_free_gib': min_free_gib,
        'maximum_new_conversion_jobs': len(canonical), 'seed': seed,
        'registry': str(registry), 'prior_cpu_conversion_jobs': 288, 'prior_new_source_conversions': 256,
        'prior_evidence': binding(ROOT/'runs/batch500_20261027/asset_native_summary.json'),
        'source_selection_does_not_depend_on_future_outcomes': True,
        'independent_surface_and_native_rules_unchanged': True,
        'qualification_gate_first_jobs': 8, 'gpu_jobs': 0, 'api_calls': 0,
        'dynamic_simulation_steps': 0, 'physical_executions': 0,
        'index_scope': 'native geometry ready for offline grasp solver; no dynamic or semantic certificate',
        'mutability': 'Only latest status/index pointers may be replaced; attempts and snapshots immutable'}
    write(root/'protocol.json', protocol)
    write(root/'selection.json', {'schema': VERSION, 'protocol': binding(root/'protocol.json'),
                                 'rows': selected, 'size_excluded': excluded})
    descriptors = []
    for asset in read(containers)['assets']:
        d = read(resolve(asset))
        descriptors.append({'source_id': d['source_id'], 'asset': asset, 'cavity': d['cavity'],
            'dimensions_m': d['dimensions_m'], 'partition': d['partition'],
            'qualification': 'retained stationary container geometry; payload/path validation required'})
    write(root/'place_roles.json', {'schema': 'place_role_inventory_v1',
        'container_source': binding(containers), 'containers': descriptors,
        'non_container_place_targets': [
            {'id': 'table_xy', 'request_kind': 'table', 'asset_required': False,
             'constraints': ['metric_xy', 'payload_footprint_inside_table', 'collision_clearance']},
            {'id': 'relative_position', 'request_kind': 'table', 'asset_required': False,
             'relations': ['left_of', 'right_of', 'in_front_of', 'behind', 'near'],
             'constraints': ['anchor_pose', 'offset_m', 'nonoverlap', 'metric_target_audit']},
            {'id': 'arrangement', 'request_kind': 'table', 'asset_required': False,
             'relations': ['row', 'group', 'separate_groups', 'distribute_destinations'],
             'constraints': ['per_object_targets', 'nonoverlap', 'final_pose_and_stability']},
            {'id': 'orientation', 'request_kind': 'table', 'asset_required': False,
             'constraints': ['metric_xy', 'target_yaw', 'orientation_tolerance']}],
        'mesh_support_surface_status': 'not_inferred_from_category_or_pickability; separate upward-surface qualification required',
        'articulated_or_insertion_mechanisms': 'not_covered_by_rigid_pick_place',
        'physical_certification': False})
    write(root/'lineage.json', {'prior_cpu_conversion_jobs': 288, 'new_job_limit': len(canonical),
        'cumulative_cpu_conversion_ceiling': 288+len(canonical),
        'old_physical_budget_is_not_consumed_or_reset': True, 'gpu_pose_budget_managed_by_root': True})
    return status(root)


def inspect_output(folder, row, returncode):
    """Geometry admission is independent of legacy height-gated pose export."""
    result = {'source_id': row['source_id'], 'state': 'native_rejected', 'returncode': returncode}
    path = folder/'refined/catalog.json'
    if path.exists():
        assets = read(path)['assets']
        if len(assets) != 1: raise ValueError('Unexpected refined catalog count')
        d = read(resolve(assets[0]))
        result['geometry_details'] = {'surface_comparison': d.get('surface_comparison'),
            'qualification': d.get('qualification'), 'roles': d.get('roles')}
        if d['qualification']['geometry'] == 'pass' and d['roles']['rigid_manipulation_candidate']:
            result.update(state='native_geometry_ready', asset=assets[0],
                          geometry_evidence=binding(path), dynamic_status='unmeasured')
            entry = folder/'entry.json'
            if entry.exists(): result['offline_antipodal_entry'] = read(entry)
            result['offline_antipodal_status'] = 'geometric_proposals_only' if entry.exists() else 'unavailable_legacy_height_gate'
    if result['state'] != 'native_geometry_ready':
        log = folder/'worker.log'; lines = log.read_text(errors='replace').splitlines() if log.exists() else []
        result['reason'] = lines[-1] if lines else str(returncode)
        if returncode == 'timeout': result['failure_class'] = 'native_conversion_timeout'
        elif 'independent_collision_geometry_rejected' in result['reason']: result['failure_class'] = 'independent_native_geometry_rejected'
        elif 'cooking failed' in result['reason']: result['failure_class'] = 'native_cook_nonconvergent'
        else: result['failure_class'] = 'worker_error'
    return result


def stop_worker(process):
    os.killpg(process.pid, signal.SIGTERM)
    try: process.wait(timeout=5)
    except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL); process.wait()


def one(root, row, protocol):
    folder = root/'jobs'/row['content_key']; terminal = folder/'terminal.json'
    if terminal.exists(): return read(terminal)
    if (folder/'reservation.json').exists(): raise ValueError('Unknown reserved CPU job; never resubmit')
    for value in protocol['code']+[row['asset']]: resolve(value)
    if shutil.disk_usage(root).free < protocol['min_free_gib']*1024**3:
        raise RuntimeError('Free disk reserve reached before submission')
    registry = Path(protocol['registry'])
    record = registry/'jobs'/(row['content_key']+'.json')
    with lock(registry/'.registry.lock', blocking=True):
        if record.exists():
            owner = read(record); existing = Path(owner['output'])/'terminal.json'
            if not existing.exists(): raise ValueError('Content has unresolved global reservation; no duplicate')
            value = read(existing)
            write(terminal, {**value, 'reused_global_terminal': binding(existing), 'new_cpu_job': False})
            return read(terminal)
        write(record, {'source_id': row['source_id'], 'asset': row['asset'],
                       'output': str(folder), 'protocol': binding(root/'protocol.json')})
        write(folder/'reservation.json', {'row': row, 'started_unix': time.time(),
              'protocol': binding(root/'protocol.json'), 'global_reservation': binding(record)})
    start = time.monotonic()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OMP_THREAD_LIMIT='2', MALLOC_ARENA_MAX='2')
    with (folder/'worker.log').open('x') as log:
        process = subprocess.Popen([sys.executable, '-u', '-m', CONVERTER, 'worker',
            '--asset', str(resolve(row['asset'])), '--output', str(folder)], cwd=ROOT,
            env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(folder/'process.json', {'pid': process.pid, 'started_unix': time.time(),
                                     'gpu_visible': False, 'cpu_physics_queries_only': True})
        try: code = process.wait(timeout=protocol['worker_timeout_seconds'])
        except subprocess.TimeoutExpired: stop_worker(process); code = 'timeout'
    result = inspect_output(folder, row, code)
    result.update(elapsed_seconds=time.monotonic()-start, new_cpu_job=True,
                  api_calls=0, gpu_jobs=0, physical_executions=0, dynamic_simulation_steps=0)
    for value in protocol['code']+[row['asset']]: resolve(value)
    write(terminal, result)
    return result


def status(root):
    root = Path(root).resolve(); p = read(root/'protocol.json'); selected = read(root/'selection.json')['rows']
    results = {}; entries = []; unknown = []; aliases = defaultdict(list)
    for row in selected: aliases[row['content_key']].append(row['source_id'])
    for row in selected:
        key = row['content_key']; folder = root/'jobs'/key
        if key in results: continue
        terminal = folder/'terminal.json'
        if terminal.exists(): results[key] = read(terminal)
        elif (folder/'reservation.json').exists():
            results[key] = {'state': 'unknown_reserved', 'source_id': row['source_id']}; unknown.append(key)
        else: results[key] = {'state': 'pending', 'source_id': row['source_id']}
        result = results[key]
        if result['state'] == 'native_geometry_ready':
            entries.append({'source_id': result['source_id'], 'asset': result['asset'],
                'geometry_evidence': result['geometry_evidence'], 'content_key': key,
                'terminal': binding(terminal), 'semantic_notes': row['semantic_notes'],
                'source_aliases': [s for s in aliases[key] if s != result['source_id']],
                'optimizer_status': 'not_run', 'physical_status': 'unmeasured'})
    counts = dict(Counter(x['state'] for x in results.values()))
    attempted = sum(bool(r.get('new_cpu_job')) for r in results.values())+len(unknown)
    summary = {'schema': VERSION, 'source_count': len(selected), 'unique_geometry_jobs': len(results),
        'states': counts, 'new_cpu_attempts': attempted, 'prior_cpu_attempts': p['prior_cpu_conversion_jobs'],
        'cumulative_cpu_attempts': attempted+p['prior_cpu_conversion_jobs'],
        'native_ready_count': len(entries), 'unknown_keys': unknown,
        'worker_seconds_completed': sum(r.get('elapsed_seconds', 0.) for r in results.values()),
        'api_calls': 0, 'gpu_jobs': 0, 'physical_executions': 0, 'dynamic_simulation_steps': 0,
        'updated_unix': time.time()}
    stamp = str(time.time_ns())
    snapshot = root/'snapshots'/('native_ready_'+stamp+'.json')
    value = {'schema': 'incremental_native_ready_index_v1', 'protocol': binding(root/'protocol.json'),
             'entries': entries, 'scope': 'geometry-ready inputs to offline solver; not grasp-certified or physical-success labels'}
    write(snapshot, value)
    replace_derived(root/'native_ready_index.json', {**value, 'snapshot': binding(snapshot)})
    replace_derived(root/'status.json', summary)
    return summary


def run(root, limit, workers=None):
    root = Path(root).resolve(); p = read(root/'protocol.json')
    workers = workers or p['max_cpu_workers']
    if not 1 <= workers <= p['max_cpu_workers'] or not 1 <= limit <= p['maximum_new_conversion_jobs']:
        raise ValueError('Requested execution exceeds frozen queue bounds')
    with lock(root/'.runner.lock'):
        for code in p['code']: resolve(code)
        before = status(root)
        if before['unknown_keys']: raise ValueError('Unknown prior CPU reservations; inspect before resume')
        started = root/'started.json'
        if not started.exists(): write(started, {'unix': time.time(), 'protocol': binding(root/'protocol.json')})
        deadline = read(started)['unix']+p['max_wall_seconds']
        if time.time() >= deadline: raise TimeoutError('Frozen queue wall limit exhausted')
        if before['new_cpu_attempts']+limit > p['maximum_new_conversion_jobs']:
            limit = p['maximum_new_conversion_jobs']-before['new_cpu_attempts']
        rows = []; seen = set()
        for row in read(root/'selection.json')['rows']:
            key = row['content_key']
            if key in seen: continue
            seen.add(key)
            if not (root/'jobs'/key/'terminal.json').exists(): rows.append(row)
        rows = rows[:limit]
        if len(rows)>p['qualification_gate_first_jobs'] and not (root/'qualification_gate.json').exists():
            raise ValueError('Complete and review first finite qualification batch before expansion')
        invocation = str(time.time_ns())
        write(root/'invocations'/(invocation+'.json'), {'pid': os.getpid(), 'requested_limit': limit,
            'workers': workers, 'started_unix': time.time(), 'deadline_unix': deadline})
        todo = iter(rows); completed = 0; submitted = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            active = {}
            while True:
                while len(active)<workers and time.time()<deadline:
                    row = next(todo, None)
                    if row is None: break
                    active[pool.submit(one, root, row, p)] = row; submitted += 1
                if not active: break
                done, _ = wait(active, timeout=10, return_when=FIRST_COMPLETED)
                for future in done:
                    row = active.pop(future); result = future.result(); completed += 1
                    print(json.dumps({'source_id': row['source_id'], 'state': result['state'],
                        'seconds': result.get('elapsed_seconds'), 'new_cpu_job': result.get('new_cpu_job')}, ensure_ascii=False), flush=True)
                if done and (completed % 16 == 0 or not active): status(root)
        result = status(root)
        write(root/'invocations'/(invocation+'_terminal.json'), {'completed': completed,
            'submitted': submitted, 'deadline_reached': time.time()>=deadline, 'summary': result})
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init'); init.add_argument('--root', type=Path, required=True)
    init.add_argument('--population', type=Path, required=True); init.add_argument('--containers', type=Path, required=True)
    init.add_argument('--workers', type=int, default=8); init.add_argument('--timeout', type=int, default=120)
    init.add_argument('--wall-seconds', type=int, default=43200); init.add_argument('--min-free-gib', type=float, default=50)
    execute = sub.add_parser('run'); execute.add_argument('--root', type=Path, required=True)
    execute.add_argument('--limit', type=int, required=True); execute.add_argument('--workers', type=int)
    query = sub.add_parser('status'); query.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'init':
        value = initialize(args.root, args.population, args.containers, args.workers, args.timeout, args.wall_seconds, args.min_free_gib)
    elif args.command == 'run': value = run(args.root, args.limit, args.workers)
    else: value = status(args.root)
    print(json.dumps(value, ensure_ascii=False, indent=2))
