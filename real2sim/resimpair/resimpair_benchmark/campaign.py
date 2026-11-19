"""Frozen 64-source, three-method comparison with shared finite resources."""
import argparse
from collections import Counter
import contextlib
import fcntl
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write
from .ledger import reserve

BASE = ROOT / 'runs/grasp_comparison_20261027'
LEDGER = ROOT / 'runs/grasp_comparison_budget_20261027'
METHODS = ('graspgenx', 'cudex', 'hybrid')


def proposal_folder(method, asset_id):
    repair = BASE / 'integration_repairs.json'
    suffix = read(repair).get('outputs', {}).get(method + '/' + asset_id, asset_id) if repair.exists() else asset_id
    return BASE / 'proposals' / method / suffix


def initialize():
    if (BASE / 'protocol.json').exists():
        raise ValueError('Comparison already registered; resume it')
    old = read(ROOT / 'validation/cudex_contact_quality_20261026/selection.json')['entries'][:2]
    entries = []
    for i, entry in enumerate(old):
        d = read(resolve(entry['asset']))
        entries.append({'id': f'a{i:03d}', 'source_id': d['source_id'],
            'source': d['provenance']['source'], 'split': 'development',
            'asset': d['offline_asset'], 'native_entry': entry, 'physical_subset': True})
    catalogs = [ROOT / 'runs/tabletop_pose_library_20261026_v2/reports/8dd98501d2598ec7.json',
                ROOT / 'runs/tabletop_pose_library_20261026_v3/reports/e2fa53e6c671dcef.json']
    used = set()
    for path in (ROOT / 'runs').glob('**/cohort.json'):
        for row in read(path).get('cases', []):
            value = row.get('source_id')
            if isinstance(value, str): used.add(value)
            elif isinstance(value, list): used.update(value)
            # Historical generic campaigns kept source IDs only in the asset
            # descriptor. Missing top-level metadata is not evidence of novelty.
            asset = row.get('asset')
            if isinstance(asset, dict) and 'path' in asset:
                try:
                    source = read(resolve(asset)).get('source_id')
                    if source: used.add(source)
                except (FileNotFoundError, ValueError):
                    raise ValueError('Historical asset provenance unavailable; cannot assert untouched sampling')
    used.update(e['source_id'] for e in read(ROOT / 'validation/cudex_contact_quality_20261026/selection.json')['entries']
                if 'source_id' in e)
    for e in read(ROOT / 'validation/cudex_contact_quality_20261026/selection.json')['entries']:
        used.add(read(resolve(e['asset']))['source_id'])
    population = {r['source_id']: r for catalog in catalogs for r in read(catalog)['rows']
        if r['status'] == 'offline_tabletop_feasible' and r['partition'] == 'heldout_asset'
        and r['source_id'] not in used}
    quotas = {'objaverse': 32, 'abo': 14, 'google_scanned_objects_original_fuel': 8,
              'polyhaven': 4, 'ycb_official': 4}
    groups = {}
    for source, count in quotas.items():
        rows = [r for r in population.values() if r['source'] == source]
        rows.sort(key=lambda r: hashlib.sha256(('20261027:' + r['source_id']).encode()).hexdigest())
        if len(rows) < count:
            raise ValueError(f'Insufficient untouched {source} population: {len(rows)} < {count}')
        groups[source] = rows[:count]
    chosen = []
    while any(groups.values()):
        for source in sorted(groups):
            if groups[source]: chosen.append(groups[source].pop(0))
    for i, row in enumerate(chosen, 2):
        entries.append({'id': f'a{i:03d}', 'source_id': row['source_id'], 'source': row['source'],
            'split': 'heldout_asset', 'asset': row['asset'], 'sample_row': row,
            'physical_subset': i < 26})
    write(BASE / 'selection.json', {'schema': 'three_way_grasp_source_selection_v1',
        'entries': entries, 'catalogs': [binding(p) for p in catalogs],
        'untouched_heldout_population': len(population), 'excluded_previous_sources': sorted(used),
        'sampling': 'source-stratified deterministic SHA256; not population-weighted',
        'condition': 'already offline-tabletop-feasible selected support/scale; not arbitrary raw assets',
        'no_post_outcome_replacement': True})
    from resimpair_grasp.cudex_native import prepare
    manip = prepare(BASE / 'native_config')
    from resimpair_grasp import evaluator as grasp_eval
    from resimpair_pickplace import evaluator as pick_eval
    protocol = {'schema': 'three_way_grasp_comparison_v1',
        'selection': binding(BASE / 'selection.json'), 'manip': binding(manip),
        'methods': list(METHODS), 'assets': 64, 'development': 2, 'untouched_heldout': 62,
        'physical_subset_sources': 26, 'same_physical_cases_per_method': True,
        'learned_proposals': 256, 'optimization_seeds': 256, 'optimization_iterations': 500,
        'common_geometry_checks_first_ranked': 128,
        'common_rule': 'resimpair_benchmark.proposals:common_native_hand_6d_v1',
        'no_vertical_snapping': True, 'no_method_fallback': True,
        'hybrid': 'all 256 learned native TCPs converted to hand roots; no random seed substitution',
        'geometry_scope': 'all sources visual; frozen physical subset rechecked against native colliders',
        'physical_controller': 'retained measured stability; no controller changes in comparison',
        'unchanged_evaluators': [binding(grasp_eval.__file__), binding(pick_eval.__file__)],
        'physics_limit': 84, 'api_limit': 0, 'gpu': 2, 'max_gpu_concurrency': 1,
        'gpu_job_limit': 198, 'per_gpu_job_timeout_seconds': 120,
        'prior_optimizer_jobs': 10, 'prior_graspgenx_inference_batches_including_failures': 3,
        'prior_shared_physical_attempts': 44,
        'statistics': 'all 64 asset-method attempts retained; stages and physical conditional rate separate; development and heldout separate',
        'baseline_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    write(BASE / 'protocol.json', protocol)
    write(LEDGER / 'limits.json', {'gpu_job_limit': 198, 'physical_attempt_limit': 84,
        'max_concurrent': 1, 'api_limit': 0, 'gpu': 2, 'per_attempt_timeout_seconds': 360,
        'prior_spent': 44, 'cumulative_ceiling': 128,
        'prior_budget': binding(ROOT / 'runs/cudex_physical_budget_20261026/limits.json'),
        'prior_reservations': [binding(p) for p in sorted((ROOT / 'runs/cudex_physical_budget_20261026/reservations').glob('*.json'))],
        'prior_gpu_jobs': 10, 'protocol': binding(BASE / 'protocol.json'),
        'authorization': 'User requested larger three-method asset comparison and cached-pose main publication',
        'created_unix': time.time()})
    write(ROOT / 'runtime/grasp_benchmark_registry/grasp_comparison_20261027.json', {
        'root': str(BASE), 'ledger': str(LEDGER), 'protocol': binding(BASE / 'protocol.json')})
    write(BASE / 'conversion_plan.json', {'sample': [e['sample_row'] for e in entries
        if e['physical_subset'] and e['split'] != 'development'], 'frozen_before_pose_results': True})
    print({'registered_assets': len(entries), 'physical_subset': 26, 'sources': dict(Counter(e['source'] for e in entries))})


def launch(method, entries, label):
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import resources
    entries = [e for e in entries if not (proposal_folder(method, e['id']) / 'terminal.json').exists()]
    if not entries: return
    protocol = read(BASE / 'protocol.json')
    registry = read(ROOT / 'runtime/grasp_benchmark_registry/grasp_comparison_20261027.json')
    if registry['ledger'] != str(LEDGER) or resolve(registry['protocol']) != BASE / 'protocol.json':
        raise ValueError('Shared campaign registry changed')
    code = {str(p.relative_to(ROOT)): binding(p) for p in Path(__file__).parent.glob('*.py')}
    repaired = any(proposal_folder(method, e['id']).name != e['id'] for e in entries)
    batch_output = BASE / 'batches' / (label + ('_repair2' if repaired else ''))
    jobs = []
    with contextlib.ExitStack() as stack:
        for path in resources.lock_paths(2):
            handle = stack.enter_context(path.open('r+'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # A completed CUDA process can leave the driver's utilization sample
        # high briefly. Wait boundedly under the same lock; never weaken idle
        # checks or evict an owner to make the next job start.
        lease_start = time.monotonic()
        for attempt in range(16):
            snapshot = resources.gpu_snapshot(2)
            try:
                resources.check_idle(snapshot)
                break
            except RuntimeError:
                if attempt == 15:
                    write(BASE / 'resource_waits' / (label + '_' + str(time.time_ns()) + '.json'),
                          {'snapshot': snapshot, 'wait_seconds': time.monotonic() - lease_start,
                           'submitted': False, 'action': 'stop before reservation; resume existing campaign when idle'})
                    raise
                time.sleep(1)
        lease_wait_seconds = time.monotonic() - lease_start
        for entry in entries:
            out = proposal_folder(method, entry['id'])
            if (out / 'terminal.json').exists(): continue
            request = {'method': method, 'asset': entry['asset'], 'seed': 2026102700 + int(entry['id'][1:]),
                       'protocol': binding(BASE / 'protocol.json'), 'code': code}
            if method != 'graspgenx': request['manip'] = protocol['manip']
            if method == 'hybrid':
                learned = proposal_folder('graspgenx', entry['id']) / 'raw.npz'
                if not learned.exists():
                    write(out / 'terminal.json', {'returncode': 'upstream_missing', 'physical_attempts': 0,
                        'reason': 'learned_proposals_unavailable; no random fallback', 'gpu_job_started': False})
                    continue
                request['learned_raw'] = binding(learned)
            identity = {'campaign': 'grasp_comparison_20261027', 'asset_id': entry['id'], 'method': method}
            if out.name != entry['id']: identity['integration_repair'] = out.name
            reservation, fresh = reserve(LEDGER, identity, request, out)
            if not fresh:
                raise RuntimeError('Unresolved previous submission; inspect process/result before recovery')
            write(out / 'reservation.json', reservation)
            jobs.append({'identity': identity, 'request': request, 'output': str(out)})
        if not jobs: return
        batch_output.mkdir(parents=True, exist_ok=False)
        spec = {'method': method, 'jobs': jobs, 'ledger': str(LEDGER), 'batch_output': str(batch_output)}
        jobfile = batch_output / 'job.json'; write(jobfile, spec)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=snapshot['uuid'], OMP_NUM_THREADS='2',
                   OPENBLAS_NUM_THREADS='2', RESIMPAIR_BENCHMARK_WORKER=str(jobfile.resolve()))
        if method == 'graspgenx':
            python = ROOT / 'runtime/envs/graspgenx/bin/python'
        else:
            from resimpair_grasp.cudex_native import UPSTREAM
            deps = ROOT / 'runtime/cudex_deps'; python = Path(sys.executable)
            env.update(PYTHONPATH=os.pathsep.join(map(str, [ROOT, deps,
                deps / 'cmeel.prefix/lib/python3.12/site-packages', UPSTREAM / 'src', UPSTREAM / 'src/curobo/geom/cpp'])),
                LD_LIBRARY_PATH=str(deps / 'cmeel.prefix/lib') + os.pathsep + os.environ.get('LD_LIBRARY_PATH', ''))
        start = time.monotonic()
        with (batch_output / 'worker.log').open('x') as log:
            process = subprocess.Popen([str(python), '-u', '-m', 'resimpair_benchmark.workers', str(jobfile)],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write(batch_output / 'process.json', {'pid': process.pid, 'start_unix': time.time(), 'gpu': snapshot})
            try: returncode = process.wait(timeout=120 * len(jobs))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait(); returncode = 'timeout'
        wall = time.monotonic() - start
        write(batch_output / 'terminal.json', {'returncode': returncode, 'elapsed_seconds': wall,
            'lease_wait_seconds': lease_wait_seconds, 'reserved': len(jobs)})
        for item in jobs:
            out = Path(item['output'])
            value = {'returncode': 0 if (out / 'result.json').exists() else returncode if returncode else 'worker_error',
                     'batch': binding(batch_output / 'terminal.json'), 'gpu_job_started': (out / 'worker_started.json').exists(),
                     'batch_wall_seconds': wall, 'batch_jobs': len(jobs), 'api_calls': 0}
            if (out / 'error.json').exists(): value['error'] = read(out / 'error.json')
            write(out / 'terminal.json', value)
            print(method, item['identity']['asset_id'], value['returncode'], flush=True)


def gpu(split, methods):
    entries = [e for e in read(BASE / 'selection.json')['entries'] if e['split'] == split]
    for method in methods:
        if method == 'graspgenx':
            for i in range(0, len(entries), 8): launch(method, entries[i:i + 8], f'{split}_{method}_{i:03d}')
        else:
            for entry in entries: launch(method, [entry], f'{split}_{method}_{entry["id"]}')


def qualify(split):
    from .proposals import qualify as one
    for entry in read(BASE / 'selection.json')['entries']:
        if split and entry['split'] != split: continue
        for method in METHODS:
            path = proposal_folder(method, entry['id']) / 'result.json'
            if not path.exists(): continue
            result = one(entry['asset'], read(path)['raw'], BASE / 'qualified_visual' / method / entry['id'])
            print(entry['id'], method, result['retained'], round(result['elapsed_seconds'], 3), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['init', 'gpu', 'qualify'])
    parser.add_argument('--split', choices=['development', 'heldout_asset'])
    parser.add_argument('--method', action='append', choices=METHODS)
    args = parser.parse_args()
    if args.mode == 'init': initialize()
    elif args.mode == 'gpu':
        if not args.split: parser.error('--split required for GPU execution')
        gpu(args.split, args.method or METHODS)
    else: qualify(args.split)
