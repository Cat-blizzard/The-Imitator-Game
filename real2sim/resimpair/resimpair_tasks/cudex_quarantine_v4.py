"""Bounded isolation of one evidenced cuDex runtime failure signature.

Failed assets remain failed, consume their original attempt, and are never
resubmitted. Only terminal empty-Newton-input failures with no complete output
may continue; all unknown or different errors stop admission. The same frozen
job manifest, native acceptance, resource policy and deadline remain in force.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
import os
from pathlib import Path
import threading
import time

from resimpair_grasp.common import binding, read, resolve, write
from . import cudex_bulk_queue_v2 as queue
from . import cudex_multigpu_v2 as multi
from . import cudex_quarantine_v3 as exact

VERSION = 'cudex_bounded_signature_quarantine_v4'


def failure_entry(protocol, ref):
    """Classify an already terminal failure; do not infer grasp infeasibility."""
    folder = resolve(ref).parent
    entry = {'job': ref, 'reason_code': 'empty_newton_input_runtime_error',
             'physical_success': 'unmeasured', 'retry': False}
    for key, name in [('terminal', 'terminal.json'), ('optimizer_terminal', 'optimizer_terminal.json'),
                      ('worker_log', 'worker.log'), ('reservation', 'reservation.json')]:
        entry[key] = binding(folder / name)
    exact.validate_quarantine(protocol, [entry])
    log = (folder / 'worker.log').read_text()
    if (log.rstrip().splitlines()[-1] != exact.EMPTY_INPUT_ERROR
            or '/curobo/opt/newton/newton_base.py' not in log
            or 'q = q.view(self.n_problems, self.action_horizon * self.d_action)' not in log):
        raise ValueError('unrecognized_error_not_quarantinable')
    return entry


class FailureCounter:
    def __init__(self, maximum_total=32, maximum_consecutive=3):
        self.maximum_total = maximum_total; self.maximum_consecutive = maximum_consecutive
        self.total = 0; self.consecutive = 0; self.peak_consecutive = 0

    def observe(self, failed):
        if failed:
            self.total += 1; self.consecutive += 1
            self.peak_consecutive = max(self.peak_consecutive, self.consecutive)
        else:
            self.consecutive = 0
        return self.stop_reason()

    def stop_reason(self):
        if self.total >= self.maximum_total:
            return 'recognized_failure_total_limit_reached'
        if self.peak_consecutive >= self.maximum_consecutive:
            return 'recognized_failure_consecutive_limit_reached'
        return None

    def snapshot(self):
        return {'recognized_failure_total': self.total, 'consecutive': self.consecutive,
                'peak_consecutive': self.peak_consecutive,
                'maximum_total': self.maximum_total, 'maximum_consecutive': self.maximum_consecutive}


def validate_policy(root, path):
    root = Path(root).resolve(); policy = read(path)
    if (policy.get('schema') != VERSION or Path(policy['campaign_root']).resolve() != root
            or policy['driver'] != binding(__file__)
            or policy['protocol'] != binding(root / 'protocol.json')
            or policy.get('maximum_signature_failures') != 32
            or policy.get('maximum_consecutive_signature_failures') != 3
            or policy.get('unknown_or_other_errors_stop_admission') is not True
            or policy.get('no_retries') is not True
            or policy['exact_failure_validator'] != binding(exact.__file__)):
        raise ValueError('invalid_bounded_failure_policy')
    resource_path = resolve(policy['resource_policy'])
    resource = multi.validate_policy(root, resource_path)
    original = read(root / 'protocol.json')
    reviewed = policy['observed_signature_examples']
    if len(reviewed) < 2:
        raise ValueError('two_bound_matching_observations_required')
    exact.validate_quarantine(original, reviewed)
    for entry in reviewed:
        if failure_entry(original, entry['job']) != entry:
            raise ValueError('reviewed_signature_evidence_changed')
    return policy, resource_path, resource


def history(root, protocol, policy_ref, counter):
    """Replay terminal chronology, preserving threshold stops across restarts."""
    events = []; quarantined = set()
    for ref in protocol['bulk_jobs']:
        folder = resolve(ref).parent; terminal_path = folder / 'terminal.json'
        if not terminal_path.exists():
            continue
        terminal = read(terminal_path)
        if terminal['status'] == 'completed':
            finished = read(folder / 'export_terminal.json')
            events.append((finished.get('finished_unix', finished.get('recovered_unix')), ref['path'], False))
        else:
            entry = failure_entry(protocol, ref)
            write(Path(root) / 'quarantined_failures_v4' / (folder.name + '.json'),
                  {'entry': entry, 'policy': policy_ref, 'disposition': 'failed_not_retried_not_admitted_to_cache'})
            quarantined.add(ref['path'])
            finished = read(folder / 'optimizer_terminal.json')
            events.append((finished['finished_unix'], ref['path'], True))
    for _, _, failed in sorted(events):
        counter.observe(failed)
    return quarantined


def dispatch(refs, execute, maximum, deadline, stop, finished, poll, counter, classify):
    """Keep original failed results; use classification only for admission."""
    pending = iter(refs); futures = {}; exhausted = False; error = counter.stop_reason()
    submitted = completed = 0
    if error:
        stop.set()
    def timed(ref):
        try:
            return execute(ref), time.monotonic()
        except Exception as exc:
            return {'status': 'needs_recovery', 'error': f'{type(exc).__name__}: {exc}'}, time.monotonic()
    with ThreadPoolExecutor(max_workers=maximum, thread_name_prefix='cudex-v4-lane') as pool:
        while futures or not exhausted:
            # Every in-flight job could fail: don't overbook the finite error allowance.
            capacity = min(maximum, counter.maximum_total - counter.total)
            while not exhausted and not stop.is_set() and len(futures) < capacity:
                if time.time() >= deadline:
                    error = error or 'original_campaign_deadline_no_new_submission'
                    stop.set(); exhausted = True; break
                ref = next(pending, None)
                if ref is None:
                    exhausted = True; break
                futures[pool.submit(timed, ref)] = ref; submitted += 1
            if not futures:
                break
            done, _ = wait(futures, timeout=5, return_when=FIRST_COMPLETED)
            for future in sorted(done, key=lambda f: f.result()[1]):
                ref = futures.pop(future); result, _ = future.result()
                try:
                    if result['status'] == 'completed':
                        reason = counter.observe(False)
                    elif result['status'] == 'failed':
                        classify(ref, result)
                        reason = counter.observe(True)
                    else:
                        raise RuntimeError(result.get('error', 'unknown_worker_result_requires_recovery'))
                    if reason:
                        error = error or reason; stop.set(); exhausted = True
                except Exception as exc:
                    error = error or f'{type(exc).__name__}: {exc}'
                    stop.set(); exhausted = True
                completed += 1; finished(ref, result)
            poll()
    return {'submitted': submitted, 'completed_futures': completed, 'error': error,
            'counts': counter.snapshot(), 'failed_results_remain_failed': True,
            'no_new_admission_after_stop': True}


def run(root, policy_path):
    root = Path(root).resolve(); policy_path = Path(policy_path).resolve()
    policy, resource_path, resource = validate_policy(root, policy_path)
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
    counter = FailureCounter(); quarantined = set()
    old_acquire, old_state, old_version = queue._acquire_gpu, queue._state, queue.VERSION

    def state(directory, label, **extra):
        with mutex:
            job = extra.get('job')
            if job:
                states[job['path']] = {'stage': label, **extra, 'observed_unix': time.time()}
            overall = (label if label in {'stopped_for_diagnosis', 'completed_with_failures'}
                       else 'parallel_bulk_running_bounded_quarantine')
            old_state(directory, overall, recovery_policy=reference,
                      resource_policy=policy['resource_policy'],
                      maximum_parallel_gpu_jobs=resource['maximum_parallel_gpu_jobs'],
                      quarantined_jobs=len(quarantined), failure_counter=counter.snapshot(),
                      active_job_states=list(states.values()), **({} if job else extra))
    adapter = multi.PoolResources(root, resource_path, state, stop)

    def refresh(force=False):
        if force or time.monotonic() - last_refresh[0] >= 60:
            queue.status(root, write_status=True); queue.update_candidate_index(root, protocol)
            last_refresh[0] = time.monotonic()

    def classify(ref, _result):
        entry = failure_entry(protocol, ref); folder = resolve(ref).parent
        write(root / 'quarantined_failures_v4' / (folder.name + '.json'),
              {'entry': entry, 'policy': reference, 'disposition': 'failed_not_retried_not_admitted_to_cache'})
        quarantined.add(ref['path'])

    def finished(ref, result):
        with mutex:
            states.pop(ref['path'], None)
        write(root / 'quarantine_results_v4' / Path(ref['path']).parent.name / f'{time.time_ns()}.json', {
            'job': ref, 'recovery_policy': reference, 'result': result,
            'failure_counter': counter.snapshot(), 'finished_unix': time.time()})
        state(root, 'parallel_bulk_running_bounded_quarantine')

    def retain_threshold_stop():
        path = root / 'quarantine_stop_v4.json'
        if counter.stop_reason() and not path.exists():
            write(path, {'reason': counter.stop_reason(), 'failure_counter': counter.snapshot(),
                         'recovery_policy': reference, 'stopped_unix': time.time(),
                         'requires_diagnosis_before_any_new_submission': True})

    with (root / '.supervisor.lock').open('a+') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        session = root / 'quarantine_sessions_v4' / f'{time.time_ns()}_{os.getpid()}'
        write(session.with_suffix('.json'), {'identity': queue.process_identity(os.getpid()),
            'started_unix': time.time(), 'recovery_policy': reference,
            'original_protocol': binding(root / 'protocol.json'), 'qualification_repeated': False})
        queue._acquire_gpu, queue._state, queue.VERSION = adapter.acquire, state, VERSION
        try:
            refresh(True)
            if (root / 'quarantine_stop_v4.json').exists():
                raise RuntimeError('recorded_failure_threshold_stop_requires_diagnosis')
            quarantined.update(history(root, protocol, reference, counter))
            if (root / 'completion.json').exists():
                state(root, 'completed_with_failures', counts=queue.status(root)); return
            if counter.stop_reason():
                raise RuntimeError(counter.stop_reason())
            deadline = read(root / 'campaign_clock.json')['deadline_unix']
            pending, recover = exact.partition_jobs(protocol, quarantined)
            write(session.with_name(session.name + '_admission.json'), {
                'pending': len(pending), 'recover_first': len(recover), 'quarantined': sorted(quarantined),
                'deadline_unix': deadline, 'gpu_job_limit': protocol['gpu_job_limit'], 'no_new_allowance': True})
            for ref in recover:
                result = queue._run_job(root, protocol, ref, deadline, {})
                if result['status'] == 'completed':
                    counter.observe(False)
                elif result['status'] == 'failed':
                    classify(ref, result); counter.observe(True)
                else:
                    raise RuntimeError('unknown_inherited_job_no_retry:' + ref['path'])
                finished(ref, result)
                if counter.stop_reason():
                    raise RuntimeError(counter.stop_reason())
            result = dispatch(pending, lambda ref: queue._run_job(root, protocol, ref, deadline, {}),
                              resource['maximum_parallel_gpu_jobs'], deadline, stop,
                              finished, refresh, counter, classify)
            refresh(True); write(session.with_name(session.name + '_terminal.json'), result)
            if result['error']:
                retain_threshold_stop()
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
            retain_threshold_stop()
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
