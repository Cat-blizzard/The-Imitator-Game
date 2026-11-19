"""Finite seed-initialization experiment within the original cuDex allowance.

The four diagnostic replays consume four previously unused campaign slots.
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

VERSION = 'bounded_cudex_seed_repair_campaign_v1'
MODULE = 'resimpair_tasks.cudex_seed_repair_campaign_v1'
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
    root = Path(root).resolve(); out = root / 'seed_repair_v1'
    if out.exists(): raise ValueError('preserve_existing_repair_experiment')
    protocol = read(root / 'protocol.json'); counts = queue.status(root)
    if counts['active_processes'] or counts['unresolved_stages']:
        raise ValueError('recover_original_work_before_preparing')
    queue._verify_frozen(protocol); queue.validate_gate(root, protocol)
    stop = read(root / 'quarantine_stop_v4.json')
    if stop['reason'] != 'recognized_failure_total_limit_reached':
        raise ValueError('expected_diagnosed_stop_required')
    pending = [r for r in protocol['bulk_jobs'] if not (resolve(r).parent / 'reservation.json').exists()]
    admitted, deferred = allocation(counts['gpu_reserved'], protocol['gpu_job_limit'], pending)
    originals = [root / 'bulk' / key / 'job.json' for key in FAILURES]
    originals += [root / 'qualification/03_nonnegative_constraint/job.json']
    jobs = []
    for i, source in enumerate(originals):
        job = read(source); old = read(source.parent / 'terminal.json')
        role = 'known_initialization_failure' if i < 3 else 'successful_control'
        if old['status'] != ('failed' if i < 3 else 'completed'):
            raise ValueError('fixed_input_history_mismatch')
        folder = out / 'qualification' / f'{i:02d}_{job["content_key"]}'
        write(folder / 'job.json', {**job, 'output': str(folder), 'phase': 'seed_repair_qualification',
              'parent_job': binding(source), 'seed_candidate': binding(candidate.__file__)})
        jobs.append({'job': binding(folder / 'job.json'), 'role': role,
                     'parent_job': binding(source), 'parent_terminal': binding(source.parent / 'terminal.json'),
                     'parent_retained': old.get('retained', 0)})
    write(out / 'original_counts.json', counts)
    write(out / 'policy.json', {'schema': VERSION, 'driver': binding(__file__),
        'candidate': binding(candidate.__file__), 'campaign_root': str(root),
        'original_protocol': binding(root / 'protocol.json'),
        'original_stop': binding(root / 'quarantine_stop_v4.json'),
        'original_counts': binding(out / 'original_counts.json'),
        'resource_policy': binding(root / 'resource_policy_multigpu_v2.json'),
        'deadline_unix': read(root / 'campaign_clock.json')['deadline_unix'],
        'gpu_job_limit': protocol['gpu_job_limit'], 'lineage_gpu_job_limit': protocol['lineage_gpu_job_limit'],
        'new_diagnostic_limit': 4, 'qualification': jobs, 'continuation': admitted,
        'deferred_to_preserve_original_budget': deferred,
        'qualification_rule': {'all_jobs_complete': True, 'nonempty_filtered_pool': True,
            'finite_256_by_1_by_9_seeds': True, 'each_failure_retains_at_least': 1,
            'successful_control_minimum_retention_ratio': .8},
        'new_failure_behavior': 'stop_admission_and_drain_no_quarantine_or_retry',
        'api_calls': 0, 'native_acceptance_changed': False,
        'physical_success': 'unmeasured', 'default_promoted': False})
    return {'policy': binding(out / 'policy.json'), 'diagnostic_jobs': 4,
            'continuation_jobs': len(admitted), 'deferred_for_budget': len(deferred)}


def validate(policy_path):
    p = read(policy_path); root = Path(p['campaign_root'])
    if (p['schema'] != VERSION or p['driver'] != binding(__file__)
            or p['candidate'] != binding(candidate.__file__)
            or p['original_protocol'] != binding(root / 'protocol.json')):
        raise ValueError('frozen_seed_repair_policy_changed')
    resolve(p['original_stop']); prior = read(resolve(p['original_counts']))
    original = read(resolve(p['original_protocol']))
    resource = multi.validate_policy(root, resolve(p['resource_policy']))
    if p['deadline_unix'] != read(root / 'campaign_clock.json')['deadline_unix']:
        raise ValueError('original_deadline_must_not_reset')
    if p['gpu_job_limit'] != original['gpu_job_limit'] or p['lineage_gpu_job_limit'] != original['lineage_gpu_job_limit']:
        raise ValueError('original_budget_must_not_reset_or_grow')
    if len(p['qualification']) != 4 or p['new_diagnostic_limit'] != 4:
        raise ValueError('four_fixed_diagnostic_jobs_required')
    refs = [r['job'] for r in p['qualification']] + p['continuation']
    if len({r['path'] for r in refs}) != len(refs) or prior['gpu_reserved'] + len(refs) != original['gpu_job_limit']:
        raise ValueError('cumulative_allowance_not_reconciled')
    original_refs = {r['path']: r for r in original['bulk_jobs']}
    for ref in p['continuation'] + p['deferred_to_preserve_original_budget']:
        if original_refs.get(ref['path']) != ref: raise ValueError('new_asset_outside_original_manifest')
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
    p, _, _ = validate(policy_path); rows = []; passed = True
    for item in p['qualification']:
        job = resolve(item['job']); terminal = read(job.parent / 'terminal.json')
        checks = {'completed': terminal['status'] == 'completed'}
        row = {**item, 'terminal': binding(job.parent / 'terminal.json'), 'checks': checks}
        if checks['completed']:
            r = queue.validate_result(job); pool = read(job.parent / 'seed_pool.json'); shape = read(job.parent / 'seed_shape.json')
            checks.update(nonempty_pool=bool(pool['counts_per_environment']) and min(pool['counts_per_environment']) > 0,
                finite_seeds=shape['shape'] == [256, 1, 9] and shape['finite'] is True,
                nonempty_native_cache=r['retained'] >= 1)
            if item['role'] == 'successful_control':
                checks['control_retention'] = r['retained'] >= .8 * item['parent_retained']
            row.update(retained=r['retained'], result=binding(job.parent / 'result.json'),
                seed_pool=binding(job.parent / 'seed_pool.json'), seed_shape=binding(job.parent / 'seed_shape.json'))
        passed = passed and all(checks.values()); rows.append(row)
    report = {'schema': VERSION, 'policy': binding(policy_path), 'rows': rows,
              'decision': 'proceed_continuation' if passed else 'stop_for_diagnosis',
              'native_acceptance_changed': False, 'physical_success': 'unmeasured',
              'review_scope': 'frozen_preregistered_gate_executed_by_root_not_new_independent_agent_review'}
    write(Path(policy_path).parent / 'qualification_gate.json', report)
    return report


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
            counts.update(diagnostic_jobs_reserved=n, cumulative_gpu_jobs=counts['gpu_reserved'] + n,
                          deferred_assets=len(p['deferred_to_preserve_original_budget']))
            queue.atomic(out / 'counts.json', counts); last[0] = time.monotonic()
    def finished(ref, result):
        with mutex: states.pop(ref['path'], None)
        emit(root, phase + '_running', last_finished={'job': ref, 'result': result})
    originals = queue._acquire_gpu, queue._launch_stage, queue._state
    with (root / '.supervisor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inherited = queue.status(root)
        if inherited['active_processes']:
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
