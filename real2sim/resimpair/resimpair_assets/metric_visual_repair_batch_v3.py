"""One bounded precision repair for the frozen batch's exact GLB export failures.

Reuse the real five-case numeric qualification where available. All sources use
the original native compiler/pose qualifier; no optimizer or API is submitted.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

from resimpair_grasp.common import ROOT, binding, read, resolve, write
from resimpair_tasks.cudex_bulk_queue_v2 import process_identity, process_alive
from resimpair_assets import metric_batch_scaled_v7 as retained
from resimpair_assets import metric_visual_precision_v3 as precision
from resimpair_assets.metric_variants_v2 import audit_variant

VERSION = 'bounded_GLB_precision_repair_batch_v3'


def worker(job_path):
    job = read(job_path)

    def materialize(descriptor_path, policy_path, output):
        prior = job.get('numeric_pilot_catalog')
        if not prior:
            return precision.materialize(descriptor_path, policy_path, output)
        catalog_path = resolve(prior)
        catalog = read(catalog_path)
        recipe = read(catalog_path.parent / 'recipe.json')
        if recipe['implementation'] != binding(precision.__file__):
            raise ValueError('precision_pilot_code_binding_mismatch')
        for reference in catalog['assets']:
            d = read(resolve(reference))
            if d['lineage']['parent_descriptor'] != binding(descriptor_path) or d['original_metric_policy'] != binding(policy_path):
                raise ValueError('precision_pilot_input_binding_mismatch')
            audit_variant(resolve(reference))
        write(Path(output) / 'reuse_receipt.json', {'prior_catalog': prior,
            'reason': 'reuse_identical_actual_numeric_pilot_without_repeating_geometry_generation',
            'new_geometry_attempts': 0, 'original_failure': job['original_failure']})
        write(Path(output) / 'catalog.json', catalog)
        return {'reused_numeric_catalog': prior}

    original_module = retained.metric_variants_v2
    retained.metric_variants_v2 = SimpleNamespace(materialize=materialize)
    try:
        return retained.worker(job_path)
    finally:
        retained.metric_variants_v2 = original_module


def prepare(root):
    root = Path(root).resolve()
    protocol = read(root / 'protocol.json')
    receipt = root / 'driver_receipt.json'
    if receipt.exists():
        resolve(read(receipt)['driver'])
        return
    for ref in [protocol['implementation'], protocol['native_qualification'], protocol['source_plan']]:
        resolve(ref)
    write(receipt, {'schema': VERSION, 'protocol': binding(root / 'protocol.json'),
        'driver': binding(__file__), 'retained_worker': binding(retained.__file__),
        'maximum_CPU_workers': protocol['maximum_CPU_workers'], 'new_GPU_jobs': 0, 'API_calls': 0})
    for row in protocol['rows']:
        old = read(resolve(row['job']))
        if read(resolve(row['failure']))['error'] != 'visual_export_changed_scale_or_origin':
            raise ValueError('typed_visual_export_failure_required')
        folder = root / 'jobs' / row['key']
        prior = ROOT / 'runs/metric_assets_20261030_v7/visual_precision_diagnostic_v3' / row['key'] / 'catalog.json'
        job = {**old, 'output': str(folder), 'protocol': binding(root / 'protocol.json'),
            'original_job': row['job'], 'original_failure': row['failure'],
            'numeric_pilot_catalog': binding(prior) if prior.exists() else None,
            'visual_precision_driver': binding(receipt), 'previous_compilation_result': None}
        job.pop('previous_failed_terminal', None)
        write(folder / 'job.json', job)


def run(root):
    root = Path(root).resolve()
    prepare(root)
    protocol = read(root / 'protocol.json')
    receipt = read(root / 'driver_receipt.json')
    for ref in [receipt['protocol'], receipt['driver'], receipt['retained_worker'],
            protocol['implementation'], protocol['native_qualification'], protocol['source_plan']]:
        resolve(ref)
    if protocol['per_source_attempts'] != 1 or len(protocol['rows']) != protocol['maximum_sources']:
        raise ValueError('frozen_precision_source_budget_changed')
    original = read(resolve(protocol['original_CPU_protocol']))
    # Keep the shared CPU cap: the container campaign must have drained first.
    container_status = read(root.parent / 'container_population_v5/status.json')
    if container_status['counts'].get('pending', 0) or container_status['counts'].get('running', 0):
        raise RuntimeError('container_CPU_stage_must_drain_before_precision_repair')
    with (root / '.run.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pending = []
        for row in protocol['rows']:
            path = root / 'jobs' / row['key'] / 'job.json'
            if (path.parent / 'terminal.json').exists():
                continue
            if (path.parent / 'process.json').exists() or (path.parent / 'worker_started.json').exists():
                if (path.parent / 'process.json').exists() and process_alive(read(path.parent / 'process.json')['identity']):
                    raise RuntimeError('live_precision_repair_process_do_not_duplicate')
                if (path.parent / 'result.json').exists():
                    write(path.parent / 'terminal.json', {'status': 'completed', 'recovered_saved_result': True,
                        'result': binding(path.parent / 'result.json')})
                    continue
                raise RuntimeError('unknown_precision_repair_state_do_not_resubmit')
            pending.append(path)

        def execute(path):
            if time.time() >= original['deadline_unix']:
                return {'status': 'deadline_no_submission'}
            folder = path.parent
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            start = time.monotonic()
            with (folder / 'worker.log').open('x') as log:
                process = subprocess.Popen([sys.executable, '-u', '-m', 'resimpair_assets.metric_visual_repair_batch_v3',
                    'worker', '--job', str(path)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                write(folder / 'process.json', {'identity': process_identity(process.pid), 'unix': time.time()})
                try:
                    code = process.wait(timeout=protocol['per_source_timeout_seconds'])
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    code = 124
            if not code and (folder / 'result.json').exists():
                terminal = {'status': 'completed', 'result': binding(folder / 'result.json')}
            elif code == 124:
                terminal = {'status': 'source_timeout', 'retry': False}
            else:
                error = read(folder / 'failure.json') if (folder / 'failure.json').exists() else {}
                terminal = {'status': error.get('classification', 'unknown_system_failure'),
                    'failure': binding(folder / 'failure.json') if error else None, 'reason': error.get('error'), 'retry': False}
            terminal['elapsed_seconds'] = time.monotonic() - start
            write(folder / 'terminal.json', terminal)
            return terminal

        # Only ten typed records exist. Process in bounded waves so an unknown
        # infrastructure fault prevents admission of the next wave.
        for start in range(0, len(pending), protocol['maximum_CPU_workers']):
            with ThreadPoolExecutor(max_workers=protocol['maximum_CPU_workers']) as pool:
                values = list(pool.map(execute, pending[start:start + protocol['maximum_CPU_workers']]))
            if any(value['status'] == 'unknown_system_failure' for value in values):
                raise RuntimeError('unknown_precision_repair_infrastructure_failure')
    rows = []
    entries = []
    for row in protocol['rows']:
        folder = root / 'jobs' / row['key']
        terminal = read(folder / 'terminal.json') if (folder / 'terminal.json').exists() else {'status': 'pending'}
        result = read(folder / 'result.json') if (folder / 'result.json').exists() else {}
        rows.append({'source_id': row['source_id'], 'original_failure': row['failure'],
            **terminal, 'entries': result.get('entries', [])})
        entries.extend(e for e in result.get('entries', []) if e.get('geometry_qualified'))
    write(root / 'native_ready_index.json', {'schema': 'resimpair_grasp_cache_index_v1', 'entries': entries})
    write(root / 'native_cache_index.json', {'schema': 'resimpair_grasp_cache_index_v1',
        'entries': [e for e in entries if e.get('retained', 0) > 0]})
    write(root / 'pending_grasp_index.json', {'schema': 'resimpair_grasp_cache_index_v1',
        'entries': [e for e in entries if not e.get('retained', 0)]})
    result = {'schema': VERSION, 'rows': rows, 'counts': dict(Counter(row['status'] for row in rows)),
        'native_ready': len(entries), 'new_GPU_jobs': 0, 'API_calls': 0,
        'prior_failures_preserved': True, 'native_rules_changed': False}
    write(root / 'RESULTS.json', result)
    return {key: value for key, value in result.items() if key != 'rows'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'worker'])
    parser.add_argument('--root', type=Path)
    parser.add_argument('--job', type=Path)
    args = parser.parse_args()
    if args.command == 'worker':
        try:
            print(worker(args.job))
        except Exception as error:
            text = str(error)
            known = isinstance(error, ValueError) and not any(token in text for token in
                ['hash changed', 'hash mismatch', 'binding_mismatch', 'previous_worker'])
            write(args.job.parent / 'failure.json', {'error_type': type(error).__name__, 'error': text,
                'traceback': traceback.format_exc(),
                'classification': 'asset_numeric_or_geometry_failed' if known else 'unknown_system_failure'})
            raise
    else:
        print(run(args.root))
