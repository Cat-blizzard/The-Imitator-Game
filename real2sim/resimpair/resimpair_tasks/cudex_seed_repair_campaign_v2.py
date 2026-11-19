"""Independent held-out qualification and bounded seed-fix continuation.

All eleven diagnostic replays consume previously unused original campaign slots.
Old failed attempts and the v4 stop receipt stay immutable. Continuation needs
a successful preregistered qualification; any new failure stops admission.
"""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from resimpair_grasp.common import ROOT, binding, read, resolve, write
from resimpair_grasp import cudex_seed_initialization_v1 as candidate
from . import cudex_bulk_queue_v2 as queue
from . import cudex_multigpu_v2 as multi

VERSION = 'bounded_cudex_seed_repair_campaign_v2'
MODULE = 'resimpair_tasks.cudex_seed_repair_campaign_v2'
FAILURES = ('00df3e5dd06234a2414023456ad6b950',
            '0abb2b9bbee50e90ddf5fc41049e67b6',
            '0ce912ff408783ac51ce2257448dbca5')


def allocation(original_reserved, original_limit, pending, replays=4):
    if not 0 <= original_reserved <= original_limit or replays <= 0:
        raise ValueError('invalid_finite_budget')
    if original_reserved + len(pending) != original_limit:
        raise ValueError('original_manifest_budget_not_reconciled')
    capacity = original_limit - original_reserved - replays
    if capacity < 0:
        raise ValueError('no_remaining_diagnostic_budget')
    return pending[:capacity], pending[capacity:]


def prepare(root):
    root = Path(root).resolve(); out = root / 'seed_repair_holdout_v2'
    if out.exists(): raise ValueError('preserve_existing_holdout_experiment')
    independent_path = ROOT / 'validation/cudex_seed_repair_20261029/independent/HOLDOUT_PROTOCOL_V2.json'
    evaluator = independent_path.with_name('evaluate_holdout_v2.py')
    independent = read(independent_path); protocol = read(root / 'protocol.json')
    counts = queue.status(root)
    if counts['active_processes'] or counts['unresolved_stages']:
        raise ValueError('recover_original_work_before_preparing')
    for ref in independent['frozen_implementation']: resolve(ref)
    resolve(independent['development_failed_gate']); resolve(independent['development_audit'])
    if counts['gpu_reserved'] != independent['budget']['original_reserved']:
        raise ValueError('frozen_initial_count_changed')
    jobs = []
    for i, item in enumerate(independent['jobs']):
        source = resolve(item['parent_job']); job = read(source)
        folder = out / 'qualification' / item['id']
        write(folder / 'job.json', {**job, 'output': str(folder), 'phase': 'seed_repair_holdout',
              'parent_job': item['parent_job'], 'seed_candidate': binding(candidate.__file__)})
        jobs.append({'job': binding(folder / 'job.json'), 'role': item['role'],
                     'parent_job': item['parent_job'], 'parent_terminal': item['parent_terminal'],
                     'parent_retained': item['prior_retained']})
    write(out / 'original_counts.json', counts)
    prior = read(resolve(independent['development_policy']))
    prior_jobs = [item['job'] for item in prior['qualification']]
    write(out / 'policy.json', {'schema': VERSION, 'driver': binding(__file__),
        'candidate': binding(candidate.__file__), 'campaign_root': str(root),
        'original_protocol': binding(root / 'protocol.json'),
        'original_stop': independent['original_stop'],
        'original_counts': binding(out / 'original_counts.json'),
        'resource_policy': independent['budget']['resource_policy'],
        'independent_protocol': binding(independent_path), 'gate_validator': binding(evaluator),
        'prior_diagnostic_jobs': prior_jobs, 'prior_diagnostic_GPU_jobs': len(prior_jobs),
        'deadline_unix': independent['budget']['original_deadline_unix'],
        'gpu_job_limit': protocol['gpu_job_limit'], 'lineage_gpu_job_limit': protocol['lineage_gpu_job_limit'],
        'new_diagnostic_limit': len(jobs), 'qualification': jobs,
        'continuation': independent['continuation_after_pass_only'],
        'deferred_to_preserve_original_budget': independent['deferred_to_preserve_original_budget'],
        'qualification_rule': independent['rules'],
        'new_failure_behavior': 'stop_admission_and_drain_no_quarantine_or_retry',
        'api_calls': 0, 'native_acceptance_changed': False,
        'physical_success': 'unmeasured', 'default_promoted': False})
    return {'policy': binding(out / 'policy.json'), 'diagnostic_jobs': len(jobs),
            'continuation_jobs': len(independent['continuation_after_pass_only']),
            'deferred_for_budget': len(independent['deferred_to_preserve_original_budget'])}


def validate(policy_path):
    p = read(policy_path); root = Path(p['campaign_root'])
    if (p['schema'] != VERSION or p['driver'] != binding(__file__)
            or p['candidate'] != binding(candidate.__file__)
            or p['original_protocol'] != binding(root / 'protocol.json')):
        raise ValueError('frozen_seed_repair_policy_changed')
    independent = read(resolve(p['independent_protocol'])); resolve(p['gate_validator'])
    for ref in independent['frozen_implementation']: resolve(ref)
    resolve(p['original_stop']); prior = read(resolve(p['original_counts']))
    original = read(resolve(p['original_protocol']))
    resource = multi.validate_policy(root, resolve(p['resource_policy']))
    if p['deadline_unix'] != read(root / 'campaign_clock.json')['deadline_unix']:
        raise ValueError('original_deadline_must_not_reset')
    if p['gpu_job_limit'] != original['gpu_job_limit'] or p['lineage_gpu_job_limit'] != original['lineage_gpu_job_limit']:
        raise ValueError('original_budget_must_not_reset_or_grow')
    if (len(p['qualification']) != independent['budget']['new_diagnostic_GPU_limit']
            or p['qualification_rule'] != independent['rules']
            or p['continuation'] != independent['continuation_after_pass_only']
            or p['deferred_to_preserve_original_budget'] != independent['deferred_to_preserve_original_budget']):
        raise ValueError('independent_holdout_protocol_changed')
    prior_policy = read(resolve(independent['development_policy']))
    if p['prior_diagnostic_jobs'] != [r['job'] for r in prior_policy['qualification']]:
        raise ValueError('prior_diagnostics_must_not_disappear')
    for ref in p['prior_diagnostic_jobs']:
        queue.validate_result(resolve(ref))
    refs = [r['job'] for r in p['qualification']] + p['continuation']
    if (len({r['path'] for r in refs}) != len(refs)
            or prior['gpu_reserved'] + len(p['prior_diagnostic_jobs']) + len(refs) != original['gpu_job_limit']):
        raise ValueError('cumulative_allowance_not_reconciled')
    for row, expected in zip(p['qualification'], independent['jobs']):
        if row['parent_job'] != expected['parent_job'] or row['role'] != expected['role']:
            raise ValueError('independent_fixed_holdout_selection_changed')
    return p, original, resource


def worker(policy_path, job_path):
    p, _, _ = validate(policy_path); job_path = Path(job_path).resolve(); ref = binding(job_path)
    refs = [r['job'] for r in p['qualification']] + p['continuation']
    if ref not in refs or os.environ.get('RESIMPAIR_CUDEX_ALIGNMENT_JOB') != str(job_path):
        raise ValueError('bounded_frozen_job_required')
    job = read(job_path)
    if (job_path.parent / 'worker_started.json').exists():
        raise ValueError('no_duplicate_optimizer')
    write(job_path.parent / 'seed_repair_policy.json', {'policy': binding(policy_path), 'job': ref})
    from resimpair_grasp.cudex_constraint_v3 import worker as retained
    with candidate.install(job['output'], job['seed']): return retained(job_path)


def execute(policy_path, job_path, stage, timeout, lock_fds=()):
    """Durable owned process group, same timeouts and inherited GPU locks."""
    validate(policy_path); job_path = Path(job_path).resolve(); out = job_path.parent
    process_path = out / (stage + '_process.json')
    if process_path.exists(): raise ValueError('stage_already_launched_no_implicit_retry')
    started = time.time()
    write(process_path, {'identity': queue.process_identity(os.getpid()), 'started_unix': started,
                        'job': binding(job_path), 'stage': stage, 'policy': binding(policy_path)})
    command = ([sys.executable, '-u', '-m', MODULE, 'worker', '--policy', str(policy_path), '--job', str(job_path)]
               if stage == 'gpu' else [sys.executable, '-u', '-m', queue.WORKER_MODULE, 'export', '--job', str(job_path)])
    returncode, error = None, None
    try:
        with (out / ('worker.log' if stage == 'gpu' else 'export.log')).open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True, pass_fds=tuple(lock_fds))
            write(out / (stage + '_worker_process.json'), {'identity': queue.process_identity(child.pid), 'started_unix': time.time()})
            try: returncode = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try: child.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
                returncode = 'timeout'
    except Exception as exc: error = f'{type(exc).__name__}: {exc}'
    finally:
        for fd in lock_fds: os.close(fd)
    terminal = {'returncode': returncode, 'error': error, 'stage': stage, 'started_unix': started,
                'finished_unix': time.time(), 'elapsed_seconds': time.time() - started, 'api_calls': 0}
    write(out / ('optimizer_terminal.json' if stage == 'gpu' else 'export_terminal.json'), terminal)
    return terminal


def qualify(policy_path):
    p, _, _ = validate(policy_path); output = Path(policy_path).parent / 'qualification_gate.json'
    evaluator = resolve(p['gate_validator'])
    if not output.exists():
        subprocess.run([sys.executable, str(evaluator), '--root', str(Path(policy_path).parent),
                        '--output', str(output)], cwd=ROOT, check=True,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'),
                       timeout=180)
    gate = read(output)
    if gate['execution_policy'] != binding(policy_path) or gate['auditor'] != p['gate_validator']:
        raise ValueError('gate_does_not_bind_actual_policy_and_independent_evaluator')
    for row in gate['rows']:
        for key in ('result', 'terminal', 'native_replay', 'seed_pool', 'seed_shape'):
            if key in row: resolve(row[key])
    for row in p['qualification']:
        queue.validate_result(resolve(row['job']))
    return gate


def run(policy_path, phase):
    policy_path = Path(policy_path).resolve(); p, original, resource = validate(policy_path)
    root = Path(p['campaign_root']); out = policy_path.parent
    stop = threading.Event(); mutex = threading.RLock(); states = {}; last = [0.]
    def emit(_root, state, **extra):
        with mutex:
            if extra.get('job'): states[extra['job']['path']] = {'state': state, **extra}
            queue.atomic(out / 'status.json', {'schema': VERSION, 'phase': phase, 'state': state,
                'identity': queue.process_identity(os.getpid()), 'updated_unix': time.time(),
                'active': list(states.values()), 'policy': binding(policy_path), **({} if extra.get('job') else extra)})
    adapter = multi.PoolResources(root, resolve(p['resource_policy']), emit, stop)
    def launch(job, stage, timeout, env, lock_fds=()):
        cmd = [sys.executable, '-u', '-m', MODULE, 'execute', '--policy', str(policy_path), '--job', str(job),
               '--stage', stage, '--timeout', str(timeout)]
        for fd in lock_fds: cmd += ['--lock-fd', str(fd)]
        with (job.parent / (stage + '_runner.log')).open('x') as stream:
            return subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                    start_new_session=True, pass_fds=tuple(lock_fds))
    def refresh(force=False):
        if force or time.monotonic() - last[0] > 60:
            counts = queue.status(root, write_status=True); queue.update_candidate_index(root, original)
            n = sum((resolve(x['job']).parent / 'reservation.json').exists() for x in p['qualification'])
            counts.update(diagnostic_jobs_reserved=n + len(p['prior_diagnostic_jobs']), cumulative_gpu_jobs=counts['gpu_reserved'] + n + len(p['prior_diagnostic_jobs']),
                          deferred_assets=len(p['deferred_to_preserve_original_budget']))
            queue.atomic(out / 'counts.json', counts); last[0] = time.monotonic()
    def finished(ref, result):
        with mutex: states.pop(ref['path'], None)
        emit(root, phase + '_running', last_finished={'job': ref, 'result': result})
    originals = queue._acquire_gpu, queue._launch_stage, queue._state
    with (root / '.supervisor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inherited = queue.status(root)
        if inherited['active_processes'] or inherited['unresolved_stages']:
            raise ValueError('active_original_worker_requires_recovery_before_new_admission')
        if (out / (phase + '_stop.json')).exists(): raise ValueError('recorded_stop_requires_diagnosis')
        refs = [r['job'] for r in p['qualification']]
        if phase == 'continuation':
            gate = read(out / 'qualification_gate.json')
            if gate != qualify(policy_path) or gate['decision'] != 'proceed_continuation':
                raise ValueError('qualification_did_not_pass')
            refs = p['continuation']
        # Any inherited reservation is resolved before admitting new work.
        refs = sorted(refs, key=lambda r: not (resolve(r).parent / 'reservation.json').exists())
        queue._acquire_gpu, queue._launch_stage, queue._state = adapter.acquire, launch, emit
        try:
            emit(root, phase + '_running'); refresh(True)
            result = multi.dispatch(refs, lambda ref: queue._run_job(root, original, ref, p['deadline_unix'], {}),
                                    resource['maximum_parallel_gpu_jobs'], p['deadline_unix'], stop, finished, refresh)
            refresh(True)
            write(out / (phase + ('_stop.json' if result['error'] else '_complete.json')), result)
            emit(root, 'stopped_for_diagnosis' if result['error'] else phase + '_complete', result=result)
            if phase == 'qualification' and not result['error']: qualify(policy_path)
            return result
        finally: queue._acquire_gpu, queue._launch_stage, queue._state = originals


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'run', 'execute', 'worker', 'qualify'])
    p.add_argument('--root', type=Path); p.add_argument('--policy', type=Path); p.add_argument('--job', type=Path)
    p.add_argument('--phase', choices=['qualification', 'continuation'], default='qualification')
    p.add_argument('--stage', choices=['gpu', 'cpu']); p.add_argument('--timeout', type=int)
    p.add_argument('--lock-fd', type=int, action='append', default=[])
    a = p.parse_args()
    if a.command == 'prepare': print(prepare(a.root))
    elif a.command == 'run': print(run(a.policy, a.phase))
    elif a.command == 'worker': print(worker(a.policy, a.job))
    elif a.command == 'execute': print(execute(a.policy, a.job, a.stage, a.timeout, a.lock_fd))
    else: print(qualify(a.policy))
