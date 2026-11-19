"""Finite, resumable native asset preparation and cuDex cache expansion.

Uses retained conversion/qualification, never per-asset dynamics or controller
special cases. Cached sources are reused; new optimizations are offline jobs.
"""
import argparse
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from resimpair_grasp.common import ROOT, read, write, binding, resolve

BASE = ROOT / 'runs/batch500_20261027'


def initialize(root=BASE):
    root = Path(root)
    if (root / 'asset_selection.json').exists():
        raise ValueError('Asset selection already frozen')
    from resimpair_pickplace.pose_bundle import materialize
    indices = materialize(ROOT / 'assets/precomputed_grasps_v1', root / 'base_library')
    native = {read(resolve(e['asset']))['source_id']: e for e in read(indices['cudex'])['entries']}
    previous = ROOT / 'runs/grasp_comparison_20261027'
    from resimpair_benchmark.campaign import proposal_folder
    known = read(previous / 'selection.json')
    seen = set(known['excluded_previous_sources']) | {r['source_id'] for r in known['entries']}
    rows = []
    for row in known['entries']:
        result = proposal_folder('cudex', row['id']) / 'result.json'
        visual = previous / 'qualified_visual/cudex' / row['id'] / 'decisions.json'
        rows.append({'id': 'cached_' + row['id'], 'source_id': row['source_id'],
            'source': row['source'], 'asset': row['asset'], 'new_source': False,
            'native_entry': native.get(row['source_id']),
            'raw': read(result)['raw'] if result.exists() else None,
            'cached_visual_count': read(visual)['retained'] if visual.exists() else 0,
            'previous_result': binding(result) if result.exists() else None})
    catalogs = [ROOT / 'runs/tabletop_pose_library_20261026_v2/reports/8dd98501d2598ec7.json',
                ROOT / 'runs/tabletop_pose_library_20261026_v3/reports/e2fa53e6c671dcef.json']
    population = {r['source_id']: r for p in catalogs for r in read(p)['rows']
                  if r['status'] == 'offline_tabletop_feasible' and r['source_id'] not in seen}
    groups = defaultdict(list)
    for row in population.values():
        descriptor = read(resolve(row['asset']))
        variant = next(v for v in descriptor['variants'] if v['id'] == descriptor['selected_variant'])
        size = variant['dimensions_m']
        if max(size) > .18 or size[2] < .018:
            continue
        key = (row['partition'], row['source'], row['category'])
        groups[key].append(row)
    for values in groups.values():
        values.sort(key=lambda r: hashlib.sha256(('batch500-v1:' + r['source_id']).encode()).hexdigest())
    selected = []
    while len(selected) < 256 and any(groups.values()):
        for key in sorted(groups):
            if groups[key]: selected.append(groups[key].pop(0))
            if len(selected) == 256: break
    for i, row in enumerate(selected):
        rows.append({'id': f'new_{i:04d}', 'source_id': row['source_id'], 'source': row['source'],
                     'category': row['category'], 'partition': row['partition'], 'asset': row['asset'],
                     'new_source': True, 'native_entry': None, 'raw': None})
    write(root / 'known_source_ids.json', {'source_ids': sorted(seen), 'scope': 'retained comparison and its earlier-source inventory'})
    write(root / 'asset_selection.json', {'schema': 'batch500_asset_selection_v1', 'rows': rows,
        'known_source_ids': binding(root / 'known_source_ids.json'),
        'catalogs': [binding(p) for p in catalogs], 'protocol': binding(root / 'protocol.json'),
        'new_count': len(selected), 'reused_source_count': len(known['entries']),
        'sampling': 'deterministic round robin across partition/source/category; fixed before new outcomes',
        'native_geometry_status': 'unmeasured_for_new_sources', 'category_semantics': 'metadata_unverified',
        'no_post_physics_replacement': True})
    return {'registered': len(rows), 'new': len(selected), 'cached': len(known['entries']),
            'category_counts': dict(Counter(r['category'] for r in selected))}


def terminate(process):
    os.killpg(process.pid, signal.SIGTERM)
    try: process.wait(timeout=8)
    except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL); process.wait()


def native_worker(root, row):
    out = root / 'asset_work' / row['id']
    terminal = out / 'native_terminal.json'
    if terminal.exists(): return read(terminal)
    if (out / 'native_reservation.json').exists():
        raise ValueError('Unknown prior native conversion; recover before resubmitting')
    if row['native_entry']:
        result = {'source_id': row['source_id'], 'state': 'reused_native', 'entry': row['native_entry'],
                  'elapsed_seconds': 0., 'native_conversion_started': False}
        write(terminal, result); return result
    if not row['new_source'] and (not row['raw'] or not row['cached_visual_count']):
        result = {'source_id': row['source_id'], 'state': 'known_cached_pose_rejection',
                  'elapsed_seconds': 0., 'native_conversion_started': False}
        write(terminal, result); return result
    out.mkdir(parents=True, exist_ok=True)
    write(out / 'native_reservation.json', {'source_id': row['source_id'], 'asset': row['asset'],
        'selection': binding(root / 'asset_selection.json'), 'started_unix': time.time()})
    start = time.monotonic()
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_THREAD_LIMIT='2')
    with (out / 'native_worker.log').open('x') as log:
        process = subprocess.Popen([sys.executable, '-u', '-m', 'resimpair_pickplace.vhacd_conversion_v2',
            'worker', '--asset', str(resolve(row['asset'])), '--output', str(out)],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(out / 'native_process.json', {'pid': process.pid, 'started_unix': time.time()})
        try: code = process.wait(timeout=120)
        except subprocess.TimeoutExpired: terminate(process); code = 'timeout'
    result = {'source_id': row['source_id'], 'state': 'native_rejected', 'returncode': code,
              'elapsed_seconds': time.monotonic() - start, 'native_conversion_started': True}
    catalog = out / 'refined/catalog.json'
    if catalog.exists():
        assets = read(catalog)['assets']
        if len(assets) == 1:
            descriptor = read(resolve(assets[0]))
            if descriptor['qualification']['geometry'] == 'pass' and descriptor['roles']['rigid_manipulation_candidate']:
                result.update(state='native_geometry_ready', asset=assets[0],
                    legacy_cache_required=False, geometry_evidence=binding(catalog))
    if result['state'] == 'native_rejected':
        result['reason'] = (out / 'native_worker.log').read_text(errors='replace').splitlines()[-1]
    write(terminal, result); return result


def native(root=BASE):
    root = Path(root)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(native_worker, root, row) for row in read(root / 'asset_selection.json')['rows']]
        for future in as_completed(futures):
            row = future.result()
            print(json.dumps({k: row.get(k) for k in ['source_id', 'state', 'reason', 'elapsed_seconds']}, ensure_ascii=False), flush=True)


def optimize_one(root, row):
    """Reserved single-GPU offline job; no model generation inside rollouts."""
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import resources
    from resimpair_benchmark.ledger import reserve
    from resimpair_grasp.cudex_native import UPSTREAM
    out = root / 'asset_work' / row['id'] / 'cudex'
    if (out / 'terminal.json').exists():
        return read(out / 'result.json')['raw'] if (out / 'result.json').exists() else None
    if (out / 'reservation.json').exists(): raise ValueError('Unresolved cuDex submission; recover existing job')
    with contextlib.ExitStack() as stack:
        for path in resources.lock_paths(2):
            handle = stack.enter_context(path.open('r+')); fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for attempt in range(16):
            snapshot = resources.gpu_snapshot(2)
            try:
                resources.check_idle(snapshot)
                break
            except RuntimeError:
                if snapshot['compute_owners'] or attempt == 15:
                    raise
                time.sleep(1)
        request = {'method': 'cudex', 'asset': row['asset'],
                   'seed': 2026102700 + int(row['id'].split('_')[-1]),
                   'protocol': binding(root / 'protocol.json'),
                   'manip': read(ROOT / 'runs/grasp_comparison_20261027/protocol.json')['manip'],
                   'code': {name: binding(ROOT / name) for name in ['resimpair_tasks/asset_pool_v1.py',
                       'resimpair_benchmark/workers.py', 'resimpair_benchmark/proposals.py']}}
        identity = {'campaign': 'batch500_20261027', 'source_id': row['source_id'], 'method': 'cudex'}
        reservation, fresh = reserve(root / 'compute_budget', identity, request, out)
        if not fresh: raise ValueError('Existing unresolved GPU reservation')
        write(out / 'reservation.json', reservation)
        jobfile = out / 'job.json'
        write(jobfile, {'method': 'cudex', 'jobs': [{'identity': identity, 'request': request, 'output': str(out)}],
                        'ledger': str(root / 'compute_budget'), 'batch_output': str(out)})
        deps = ROOT / 'runtime/cudex_deps'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=snapshot['uuid'], OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                   RESIMPAIR_BENCHMARK_WORKER=str(jobfile.resolve()),
                   PYTHONPATH=os.pathsep.join(map(str, [ROOT, deps, deps / 'cmeel.prefix/lib/python3.12/site-packages',
                       UPSTREAM / 'src', UPSTREAM / 'src/curobo/geom/cpp'])),
                   LD_LIBRARY_PATH=str(deps / 'cmeel.prefix/lib') + os.pathsep + os.environ.get('LD_LIBRARY_PATH', ''))
        start = time.monotonic()
        with (out / 'worker.log').open('x') as log:
            process = subprocess.Popen([sys.executable, '-u', '-m', 'resimpair_benchmark.workers', str(jobfile)],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write(out / 'process.json', {'pid': process.pid, 'started_unix': time.time(), 'gpu': snapshot})
            try: code = process.wait(timeout=120)
            except subprocess.TimeoutExpired: terminate(process); code = 'timeout'
        write(out / 'terminal.json', {'returncode': code, 'seconds': time.monotonic() - start,
                                     'result_exists': (out / 'result.json').exists(), 'api_calls': 0})
        return read(out / 'result.json')['raw'] if (out / 'result.json').exists() else None


def qualify_bounded(asset, raw, folder):
    """CPU-only closure check with a retained input, timeout and terminal state."""
    output = folder / 'native_cudex_qualification'
    expected = {'asset': asset, 'raw': raw, 'geometry': 'native'}
    decisions = output / 'decisions.json'
    if decisions.exists():
        result = read(decisions)
        if any(result[k] != v for k, v in expected.items()):
            raise ValueError('Qualification input changed')
        resolve(result['cache'])
        return result
    terminal = folder / 'qualification_terminal.json'
    if terminal.exists(): return None
    request = folder / 'qualification_request.json'
    if request.exists(): raise ValueError('Unresolved CPU qualification; recover before resubmitting')
    write(request, {**expected, 'output': str(output), 'rule': binding(ROOT / 'resimpair_benchmark/proposals.py')})
    start = time.monotonic()
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with (folder / 'qualification_worker.log').open('x') as log:
        process = subprocess.Popen([sys.executable, '-u', '-m', 'resimpair_tasks.asset_pool_v1',
            'qualify-worker', '--request', str(request)], cwd=ROOT, env=env,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(folder / 'qualification_process.json', {'pid': process.pid, 'started_unix': time.time()})
        try: code = process.wait(timeout=120)
        except subprocess.TimeoutExpired: terminate(process); code = 'timeout'
    write(terminal, {'returncode': code, 'elapsed_seconds': time.monotonic() - start,
                     'decisions_exist': decisions.exists(), 'api_calls': 0, 'gpu_jobs': 0})
    return read(decisions) if code == 0 and decisions.exists() else None


def caches(root=BASE, only_reused=False):
    root = Path(root)
    with (root / '.cache_preparation.lock').open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _caches(root, only_reused)


def _caches(root=BASE, only_reused=False):
    root = Path(root)
    for row in read(root / 'asset_selection.json')['rows']:
        folder = root / 'asset_work' / row['id']; terminal = folder / 'cache_terminal.json'
        if terminal.exists() or not (folder / 'native_terminal.json').exists(): continue
        native_result = read(folder / 'native_terminal.json')
        if row['native_entry']:
            n = len(read(resolve(row['native_entry']['cache']))['candidates'])
            result = {'source_id': row['source_id'], 'state': 'ready' if n else 'native_cache_empty',
                      'entry': row['native_entry'], 'retained': n, 'gpu_job_started': False}
        elif native_result['state'] != 'native_geometry_ready':
            result = {'source_id': row['source_id'], 'state': native_result['state'], 'gpu_job_started': False}
        else:
            if only_reused and not row['raw']: continue
            raw = row['raw'] or optimize_one(root, row)
            if raw:
                qualified = qualify_bounded(native_result['asset'], raw, folder)
                if qualified:
                    result = {'source_id': row['source_id'], 'state': 'ready' if qualified['retained'] else 'native_cache_empty',
                              'entry': {'asset': native_result['asset'], 'cache': qualified['cache']},
                              'retained': qualified['retained'], 'gpu_job_started': row['raw'] is None}
                else:
                    result = {'source_id': row['source_id'], 'state': 'cpu_qualification_failed',
                              'gpu_job_started': row['raw'] is None}
            else: result = {'source_id': row['source_id'], 'state': 'optimizer_failed', 'gpu_job_started': True}
        write(terminal, result)
        print(json.dumps({k: result.get(k) for k in ['source_id', 'state', 'retained', 'gpu_job_started']}), flush=True)


def report(root=BASE):
    root = Path(root); rows = []; entries = []
    for row in read(root / 'asset_selection.json')['rows']:
        terminal = root / 'asset_work' / row['id'] / 'cache_terminal.json'
        result = read(terminal) if terminal.exists() else {'state': 'pending'}
        rows.append({'id': row['id'], 'source_id': row['source_id'], **result})
        if result['state'] == 'ready': entries.append(result['entry'])
    value = {'entries': entries, 'rows': rows, 'ready': len(entries),
             'statuses': dict(Counter(r['state'] for r in rows)), 'physical_executions': 0,
             'qualification': 'native geometry and cached closure; dynamics remain unmeasured'}
    target = root / 'asset_pool_reports' / (str(time.time_ns()) + '.json'); write(target, value)
    print(json.dumps({'path': str(target), 'ready': len(entries), 'statuses': value['statuses']}))
    return value


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['initialize', 'native', 'caches', 'report', 'qualify-worker'])
    p.add_argument('--root', type=Path, default=BASE)
    p.add_argument('--only-reused', action='store_true')
    p.add_argument('--request', type=Path)
    a = p.parse_args()
    if a.command == 'qualify-worker':
        from resimpair_benchmark.proposals import qualify
        value = read(a.request); resolve(value['rule'])
        qualify(value['asset'], value['raw'], Path(value['output']), geometry=value['geometry'])
    elif a.command == 'caches': caches(a.root, a.only_reused)
    else:
        result = globals()[a.command](a.root)
        if result is not None and a.command != 'report': print(json.dumps(result, ensure_ascii=False))
