"""Continue the original finite campaign past explicitly recorded job failures.

Quarantine is an operational disposition, never a successful grasp result.
Only hash-bound, terminal bulk failures may be excluded from dispatch. New
failures still stop admission; unknown submissions must first be recovered.
The original solver, native validator, limits and multi-GPU locks are reused.
"""
import argparse
import fcntl
import os
from pathlib import Path
import threading
import time

from resimpair_grasp.common import binding, read, resolve, write
from . import cudex_bulk_queue_v2 as queue
from . import cudex_multigpu_v2 as multi

VERSION = 'cudex_recorded_failure_quarantine_v3'
EMPTY_INPUT_ERROR = "RuntimeError: shape '[256, 9]' is invalid for input of size 0"


def validate_quarantine(protocol, entries):
    """An exact terminal/log binding cannot hide a new failure or retry."""
    bulk = {r['path']: r for r in protocol['bulk_jobs']}
    accepted = set()
    for entry in entries:
        ref = entry['job']
        if bulk.get(ref['path']) != ref or ref['path'] in accepted:
            raise ValueError('quarantine_requires_unique_original_bulk_job')
        folder = resolve(ref).parent
        for key, filename in [('terminal', 'terminal.json'),
                              ('optimizer_terminal', 'optimizer_terminal.json'),
                              ('worker_log', 'worker.log'), ('reservation', 'reservation.json')]:
            if resolve(entry[key]) != (folder / filename).resolve():
                raise ValueError('quarantine_evidence_from_wrong_job')
        terminal = read(folder / 'terminal.json')
        optimizer = read(folder / 'optimizer_terminal.json')
        if (terminal.get('status') != 'failed' or terminal.get('failure_stage') != 'gpu'
                or terminal.get('returncode') != 1 or optimizer.get('returncode') != 1
                or optimizer.get('complete_output_present') is not False
                or read(folder / 'reservation.json').get('job') != ref):
            raise ValueError('quarantine_requires_known_terminal_GPU_failure')
        if (entry.get('reason_code') != 'empty_newton_input_runtime_error'
                or EMPTY_INPUT_ERROR not in (folder / 'worker.log').read_text()
                or entry.get('physical_success') != 'unmeasured'
                or entry.get('retry') is not False):
            raise ValueError('quarantine_reason_or_no_retry_rule_invalid')
        if any((folder / p).exists() for p in (
                'optimizer_result.json', 'result.json', 'export_reservation.json')):
            raise ValueError('completed_or_ambiguous_output_requires_recovery')
        for filename in ('gpu_process.json', 'gpu_worker_process.json', 'cpu_process.json'):
            path = folder / filename
            if path.exists() and queue.process_alive(read(path).get('identity')):
                raise RuntimeError('quarantined_job_process_still_alive')
        accepted.add(ref['path'])
    return accepted


def partition_jobs(protocol, quarantined):
    pending, recover = [], []
    for ref in protocol['bulk_jobs']:
        folder = resolve(ref).parent
        if (folder / 'terminal.json').exists():
            if (read(folder / 'terminal.json')['status'] != 'completed'
                    and ref['path'] not in quarantined):
                raise RuntimeError('unreviewed_failure_requires_diagnosis:' + ref['path'])
        elif ref['path'] in quarantined:
            raise RuntimeError('quarantined_terminal_missing')
        elif (folder / 'reservation.json').exists():
            recover.append(ref)
        else:
            pending.append(ref)
    return pending, recover


def validate_policy(root, path):
    root = Path(root).resolve(); policy = read(path)
    if (policy.get('schema') != VERSION or Path(policy['campaign_root']).resolve() != root
            or policy['driver'] != binding(__file__)
            or policy['protocol'] != binding(root / 'protocol.json')
            or policy.get('new_failures_stop_admission') is not True):
        raise ValueError('invalid_quarantine_policy')
    resource_path = resolve(policy['resource_policy'])
    resource = multi.validate_policy(root, resource_path)
    protocol = read(root / 'protocol.json')
    quarantined = validate_quarantine(protocol, policy['quarantined_jobs'])
    if not quarantined:
        raise ValueError('use_original_driver_without_recorded_failures')
    return policy, resource_path, resource, quarantined


def run(root, policy_path):
    root = Path(root).resolve(); policy_path = Path(policy_path).resolve()
    policy, resource_path, resource, quarantined = validate_policy(root, policy_path)
    reference = binding(policy_path); protocol = read(root / 'protocol.json')
    registry = read(resolve(protocol['registry']))
    if Path(registry['root']).resolve() != root or registry['recipe'] != protocol['recipe']:
        raise ValueError('same_registered_campaign_required')
    refs = protocol['qualification_jobs'] + protocol['bulk_jobs']
    if len(refs) != protocol['gpu_job_limit'] or len({r['path'] for r in refs}) != len(refs):
        raise ValueError('original_finite_manifest_invalid')
    resolve(protocol['frozen_inputs']); resolve(protocol['candidate_manifest'])
    queue.validate_gate(root, protocol)
    for ref in protocol['qualification_jobs']:
        queue.validate_result(resolve(ref))
    stop = threading.Event(); mutex = threading.RLock(); states = {}; last_refresh = [0.]
    old_acquire, old_state, old_version = queue._acquire_gpu, queue._state, queue.VERSION

    def state(directory, label, **extra):
        with mutex:
            job = extra.get('job')
            if job:
                states[job['path']] = {'stage': label, **extra, 'observed_unix': time.time()}
            overall = (label if label in {'stopped_for_diagnosis', 'completed_with_failures'}
                       else 'parallel_bulk_running_with_quarantine')
            old_state(directory, overall, recovery_policy=reference,
                      resource_policy=policy['resource_policy'],
                      maximum_parallel_gpu_jobs=resource['maximum_parallel_gpu_jobs'],
                      quarantined_jobs=len(quarantined), active_job_states=list(states.values()),
                      **({} if job else extra))

    adapter = multi.PoolResources(root, resource_path, state, stop)

    def refresh(force=False):
        if force or time.monotonic() - last_refresh[0] >= 60:
            queue.status(root, write_status=True); queue.update_candidate_index(root, protocol)
            last_refresh[0] = time.monotonic()

    def finished(ref, result):
        with mutex:
            states.pop(ref['path'], None)
        write(root / 'quarantine_results_v3' / Path(ref['path']).parent.name / f'{time.time_ns()}.json', {
            'job': ref, 'recovery_policy': reference, 'result': result, 'finished_unix': time.time()})
        state(root, 'parallel_bulk_running_with_quarantine')

    with (root / '.supervisor.lock').open('a+') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        session = root / 'quarantine_sessions_v3' / f'{time.time_ns()}_{os.getpid()}'
        write(session.with_suffix('.json'), {'identity': queue.process_identity(os.getpid()),
            'started_unix': time.time(), 'recovery_policy': reference,
            'original_protocol': binding(root / 'protocol.json'), 'qualification_repeated': False})
        queue._acquire_gpu, queue._state, queue.VERSION = adapter.acquire, state, VERSION
        try:
            refresh(True)
            if (root / 'completion.json').exists():
                state(root, 'completed_with_failures', counts=queue.status(root)); return
            deadline = read(root / 'campaign_clock.json')['deadline_unix']
            pending, recover = partition_jobs(protocol, quarantined)
            write(session.with_name(session.name + '_admission.json'), {
                'pending': len(pending), 'recover_first': len(recover),
                'quarantined': sorted(quarantined), 'deadline_unix': deadline,
                'gpu_job_limit': protocol['gpu_job_limit'], 'no_new_allowance': True})
            for ref in recover:
                result = queue._run_job(root, protocol, ref, deadline, {})
                if result['status'] != 'completed':
                    raise RuntimeError('inherited_job_failed_no_retry:' + ref['path'])
                finished(ref, result)
            result = multi.dispatch(pending, lambda ref: queue._run_job(root, protocol, ref, deadline, {}),
                                    resource['maximum_parallel_gpu_jobs'], deadline, stop, finished, refresh)
            refresh(True)
            write(session.with_name(session.name + '_terminal.json'), result)
            if result['error']:
                state(root, 'stopped_for_diagnosis', error=result['error'], counts=queue.status(root))
                return result
            counts = queue.status(root)
            if (counts['terminal_jobs'] != protocol['gpu_job_limit'] or counts['unresolved_stages']
                    or counts['failed_jobs'] != len(quarantined)):
                raise RuntimeError('manifest_not_fully_accounted_for')
            write(root / 'completion.json', {'status': 'completed_with_failures',
                'finished_unix': time.time(), 'protocol': binding(root / 'protocol.json'),
                'recovery_policy': reference, 'counts': counts, 'api_calls': 0,
                'physical_success': 'unmeasured', 'default_promoted': False})
            state(root, 'completed_with_failures', counts=counts)
            return result
        except Exception as exc:
            stop.set(); refresh(True)
            state(root, 'stopped_for_diagnosis', error=f'{type(exc).__name__}: {exc}', counts=queue.status(root))
            raise
        finally:
            queue._acquire_gpu, queue._state, queue.VERSION = old_acquire, old_state, old_version


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--recovery-policy', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.recovery_policy)
