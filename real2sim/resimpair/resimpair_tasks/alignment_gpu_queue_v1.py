"""Bounded alignment diagnostics that wait for an unused GPU, never preempt one.

The CPU asset library has an independent lifetime. This queue first compares
four native assets with/without the cuDex constraint fix (eight jobs). It does
not promote the solver or expand to the whole library without evidence.
"""
import argparse
from contextlib import ExitStack
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from resimpair_grasp.common import ROOT, binding, read, resolve, write
from resimpair_grasp.cudex_native import UPSTREAM
from corl2026_workshop.experiments.autonomous_pipeline_v4 import resources

VERSION = 'idle_only_cudex_alignment_queue_v1'


def prepare(root, index, original, candidate):
    root = Path(root).resolve()
    if root.exists():
        raise ValueError('existing_alignment_queue_resume_only')
    entries = read(index)['entries']
    chosen = []
    # Freeze one source from each available collection before GPU outcomes.
    for prefix in ('abo:', 'google_scanned_objects_original_fuel:', 'objaverse:', 'ycb_official:'):
        matches = sorted((e for e in entries if e['source_id'].startswith(prefix)), key=lambda e: e['source_id'])
        if matches:
            chosen.append(matches[0])
    if len(chosen) != 4:
        raise ValueError('four_predeclared_collections_required')
    write(root / 'frozen_inputs.json', {'entries': chosen, 'selection': 'first_source_per_four_collections',
        'original_index_is_incremental': True, 'native_geometry_only': True})
    jobs = []
    for i, entry in enumerate(chosen):
        for variant, manip in [('retained_signed_constraint', original), ('nonnegative_constraint', candidate)]:
            folder = root / f'{i:02d}_{variant}'
            path = folder / 'job.json'
            write(path, {'asset': entry['asset'], 'manip': binding(manip), 'seed': 20261028,
                'output': str(folder), 'variant': variant, 'source_id': entry['source_id']})
            jobs.append(binding(path))
    write(root / 'protocol.json', {'schema': VERSION, 'jobs': jobs, 'gpu_job_limit': 8,
        'per_job_timeout_seconds': 180, 'maximum_parallel_gpu_jobs': 1, 'api_calls': 0,
        'gpu_ownership': 'no_compute_owner_memory_le_512MiB_utilization_le_10_for_120s_then_exclusive_project_lock',
        'maximum_wait_seconds': 7200, 'unknown_never_resubmitted': True,
        'scope': 'four_new_sources_controlled_comparison_not_historical_139_recount',
        'code': [binding(__file__), binding(ROOT / 'resimpair_grasp/cudex_constraint_v2.py')],
        'prior_expansion_optimization_evidence': binding(ROOT / 'runs/batch500_20261027/asset_capacity_audit_20261028/FINAL_SUMMARY.json'),
        'prior_expansion_jobs': 141, 'prior_count_scope': 'retained_expansion_campaign_not_all_historical_projects',
        'default_promoted': False})
    return {'prepared_jobs': len(jobs), 'gpu_jobs_started': 0}


def _status(root, value):
    import json
    temporary = root / ('status.' + str(os.getpid()) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(root / 'status.json')


def run(root):
    root = Path(root).resolve()
    protocol = read(root / 'protocol.json')
    for item in protocol['code']:
        resolve(item)
    began = time.time()
    quiet_since = {}
    with (root / '.supervisor.lock').open('a+') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(root / 'supervisor.json', {'pid': os.getpid(), 'started_unix': began,
            'protocol': binding(root / 'protocol.json')})
        for job_ref in protocol['jobs']:
            path = resolve(job_ref); folder = path.parent
            if (folder / 'terminal.json').exists():
                continue
            if (folder / 'reservation.json').exists():
                raise RuntimeError('unknown_gpu_submission_recover_existing_process')
            selected = None
            while time.time()-began < protocol['maximum_wait_seconds']:
                for i in range(4):
                    snap = resources.gpu_snapshot(i)
                    try:
                        resources.check_idle(snap)
                    except RuntimeError:
                        quiet_since.pop(i, None)
                        continue
                    quiet_since.setdefault(i, time.time())
                    if time.time()-quiet_since[i] >= 120:
                        selected = i
                        break
                if selected is not None:
                    break
                _status(root, {'status': 'waiting_unused_gpu', 'pid': os.getpid(),
                    'updated_unix': time.time(), 'started_jobs': len(list(root.glob('*/reservation.json'))),
                    'terminal_jobs': len(list(root.glob('*/terminal.json'))), 'api_calls': 0})
                time.sleep(30)
            if selected is None:
                _status(root, {'status': 'wait_deadline_no_resource', 'updated_unix': time.time(),
                    'started_jobs': len(list(root.glob('*/reservation.json'))), 'api_calls': 0})
                return
            with ExitStack() as locks:
                for lock_path in resources.lock_paths(selected):
                    lock = locks.enter_context(lock_path.open('r+'))
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                snapshot = resources.gpu_snapshot(selected); resources.check_idle(snapshot)
                if len(list(root.glob('*/reservation.json'))) >= protocol['gpu_job_limit']:
                    raise ValueError('finite_gpu_budget_exhausted')
                write(folder / 'reservation.json', {'job': job_ref, 'gpu': snapshot, 'started_unix': time.time()})
                deps = ROOT / 'runtime/cudex_deps'
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=snapshot['uuid'],
                    RESIMPAIR_CUDEX_ALIGNMENT_JOB=str(path), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                    PYTHONPATH=os.pathsep.join(map(str, [ROOT, deps, deps/'cmeel.prefix/lib/python3.12/site-packages',
                        UPSTREAM/'src', UPSTREAM/'src/curobo/geom/cpp'])),
                    LD_LIBRARY_PATH=str(deps/'cmeel.prefix/lib')+os.pathsep+os.environ.get('LD_LIBRARY_PATH',''))
                start = time.monotonic()
                with (folder / 'worker.log').open('x') as stream:
                    child = subprocess.Popen([sys.executable, '-u', '-m', 'resimpair_grasp.cudex_constraint_v2',
                        'worker', '--job', str(path)], cwd=ROOT, env=env, stdout=stream,
                        stderr=subprocess.STDOUT, start_new_session=True)
                    write(folder / 'process.json', {'pid': child.pid, 'started_unix': time.time()})
                    try:
                        code = child.wait(timeout=protocol['per_job_timeout_seconds'])
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=8)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL); child.wait()
                        code = 'timeout'
                write(folder / 'terminal.json', {'returncode': code, 'elapsed_seconds': time.monotonic()-start,
                    'result_exists': (folder / 'result.json').exists(), 'api_calls': 0})
                # Stop after the first infrastructure/interface failure. Preserve
                # the denominator; never burn the remaining assets on a known bug.
                if code != 0 or not (folder / 'result.json').exists():
                    _status(root, {'status': 'stopped_for_diagnosis', 'job': job_ref,
                        'updated_unix': time.time(), 'api_calls': 0})
                    return
        _status(root, {'status': 'terminal_qualification_needs_independent_review',
            'updated_unix': time.time(), 'started_jobs': len(list(root.glob('*/reservation.json'))),
            'default_promoted': False, 'api_calls': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--index', type=Path, required=True); p.add_argument('--original', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p = sub.add_parser('run'); p.add_argument('--root', type=Path, required=True)
    a = parser.parse_args()
    print(prepare(a.root, a.index, a.original, a.candidate) if a.command == 'prepare' else run(a.root))
