"""Run pp2 scene task sets on the retained default controller with parallel CPU workers.

The worker is resimpair_tasks.production_integration_v4 (profile
"checked_withdrawal"), unchanged. This supervisor only differs from the
retained ones in mode handling: production cohorts run in parallel and record
every attempt; evaluation cohorts are frozen before execution and never
replace a failed case. Neither mode reruns a case that has a reservation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from resimpair_grasp.common import ROOT, binding, read, resolve, write

VERSION = 'pp2_execution_v1'
PARENT_COHORT = ROOT / 'runs/production_integration_20261030_v1/scene_tools_revision_r1'
WORKER_MODULE = 'resimpair_tasks.production_integration_v4'
TIMEOUT_SECONDS = 480
MODES = ('production', 'evaluation')


def prepare(factory, root, mode, attempt_limit, worker_module=WORKER_MODULE):
    from resimpair_pickplace.runner import verify
    factory, root = Path(factory).resolve(), Path(root).resolve()
    if root.exists():
        raise ValueError('preserve_existing_cohort')
    if mode not in MODES:
        raise ValueError('unknown_mode')
    spec = deepcopy(verify(PARENT_COHORT))
    spec.update(schema=VERSION, budget=str(root / 'budget'), timeout_seconds=TIMEOUT_SECONDS,
                integration_prefix_reuse=False, integration_full_open=True,
                parent_experiment=binding(PARENT_COHORT / 'experiment.json'), pp2_mode=mode)
    # Freeze only the modules that define the executed inputs; tooling (render, grid, labels...) may change.
    spec['code'].update({f'resimpair_pp2/{n}': binding(ROOT / 'resimpair_pp2' / n)
                         for n in ('__init__.py', 'grammar.py', 'scenes.py', 'execute.py', 'worker.py')})
    cases = []
    for row in read(factory / 'factory.json')['scenes']:
        if not row['tasks']:
            continue
        directory = Path(row['directory'])
        directory = directory if directory.is_absolute() else ROOT / directory
        scene_binding = binding(directory / 'compiled_scene.json')
        scene = read(directory / 'compiled_scene.json')
        sources = sorted(o['pp2']['source_id'] for o in scene['objects'] + scene['supports'])
        for task in read(directory / 'compiled_tasks.json')['tasks']:
            case_id = row['scene_id'] + '__' + task['requested_task_id']
            path = root / 'inputs' / case_id / 'task.json'; write(path, task)
            cases.append({'id': case_id, 'execution_id': case_id, 'scene_id': row['scene_id'],
                'partition': row['partition'], 'source_ids': sources, 'split_group': sources,
                'source_coverage': 'synthetic_only', 'requested_task_id': task['requested_task_id'],
                'scene': scene_binding, 'task': binding(path), 'eligible': True,
                'method': 'pp2_catalog_native_grasps', 'pp2_family': task['pp2_program']['family'],
                'instruction': task['instruction'], 'physical_status': 'not_executed'})
    if len(cases) > attempt_limit:
        raise ValueError('cohort_exceeds_registered_attempt_limit')
    write(root / 'budget/limits.json', {'schema': VERSION, 'mode': mode, 'physical_attempt_limit': attempt_limit,
        'timeout_seconds': TIMEOUT_SECONDS, 'retries_per_case': 0, 'API_calls': 0, 'GPU_jobs': 0,
        'registered_unix': time.time()})
    write(root / 'experiment.json', spec)
    write(root / 'rules.json', read(PARENT_COHORT / 'rules.json'))
    write(root / 'cohort.json', {'cases': cases, 'selected_before_execution': True, 'failed_cases_replaced': False,
                                 'factory': binding(factory / 'factory.json'), 'mode': mode})
    write(root / 'protocol.json', {'schema': VERSION, 'driver': binding(Path(__file__)), 'worker_module': worker_module,
        'worker': binding(ROOT / 'resimpair_tasks/production_integration_v4.py'), 'mode': mode,
        'budget': binding(root / 'budget/limits.json'), 'API_calls': 0, 'GPU_jobs': 0})
    return {'root': str(root), 'cases': len(cases), 'mode': mode}


def _launch(root, case, limits):
    from resimpair_tasks.multitask_campaign_v3 import admission_check
    directory = root / 'executions' / case['id']
    reservation = {'case': case, 'campaign': str(root), 'protocol': binding(root / 'protocol.json'),
                   'started_unix': time.time()}
    write(directory / 'reservation.json', reservation)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none',
               VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/lvp_icd.x86_64.json', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONPATH=str(ROOT),
               RESIMPAIR_HEADLESS_RESERVATION=str(directory / 'reservation.json'))
    start = time.monotonic()
    with (directory / 'worker.log').open('x') as stream:
        child = subprocess.Popen([sys.executable, '-u', '-m', read(root / 'protocol.json')['worker_module'], 'worker', '--root', str(root),
                                  '--case', case['id']], cwd=ROOT, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        write(directory / 'process.json', {'pid': child.pid, 'started_unix': time.time()})
        try:
            rc = child.wait(timeout=limits['timeout_seconds'])
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
            rc = 'timeout'
    check = admission_check(directory, rc)
    terminal = {'case': case['id'], 'physical_attempts': 1, 'returncode': rc,
                'elapsed_seconds': time.monotonic() - start, 'admission': check, 'API_calls': 0, 'GPU_jobs': 0}
    write(directory / 'terminal.json', terminal)
    return terminal


def run(root, workers, limit=None):
    from resimpair_pickplace.runner import verify
    root = Path(root).resolve(); verify(root)
    limits = read(root / 'budget/limits.json')
    cases = read(root / 'cohort.json')['cases']
    with (root / '.supervisor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pending = []
        for case in cases:
            directory = root / 'executions' / case['id']
            if (directory / 'terminal.json').exists():
                continue
            if (directory / 'reservation.json').exists():
                raise RuntimeError('unknown_attempt_recover_without_rerun:' + case['id'])
            pending.append(case)
        used = sum((root / 'executions' / c['id'] / 'reservation.json').exists() for c in cases)
        room = limits['physical_attempt_limit'] - used
        pending = pending[:min(room, limit if limit is not None else room)]
        started = time.monotonic(); rows = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for terminal in pool.map(lambda c: _launch(root, c, limits), pending):
                rows.append(terminal)
                print(json.dumps({'case': terminal['case'], 'rc': terminal['returncode'],
                                  'seconds': round(terminal['elapsed_seconds'], 1),
                                  'failure': terminal['admission'].get('reported_failure')})[:300], flush=True)
    result = {'new_attempts': len(rows), 'wall_seconds': time.monotonic() - started,
              'worker_seconds': sum(r['elapsed_seconds'] for r in rows), 'workers': workers}
    write(root / 'run_receipts' / f'{time.time_ns()}.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare'); a.add_argument('--factory', type=Path, required=True)
    a.add_argument('--root', type=Path, required=True); a.add_argument('--mode', choices=MODES, required=True)
    a.add_argument('--attempt-limit', type=int, required=True); a.add_argument('--worker', default=WORKER_MODULE)
    b = sub.add_parser('run'); b.add_argument('--root', type=Path, required=True)
    b.add_argument('--workers', type=int, default=8); b.add_argument('--limit', type=int)
    args = p.parse_args()
    print(json.dumps(prepare(args.factory, args.root, args.mode, args.attempt_limit, args.worker) if args.command == 'prepare'
                     else run(args.root, args.workers, args.limit), indent=1))
