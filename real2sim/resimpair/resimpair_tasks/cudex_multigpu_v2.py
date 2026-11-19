"""Bounded multi-GPU scheduling over the existing, qualified cuDex job ledger.

An explicit resource-policy revision may increase concurrent jobs, never the
frozen job list, total allowance, solve settings, deadline or acceptance rules.
Each thread reuses the original durable GPU/export/recovery implementation.
Independent processes isolate CUDA state; inherited per-GPU locks allow at
most one project optimizer per card. Historical qualification is not rerun.
"""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import threading
import time

from resimpair_grasp.common import binding, read, resolve, write
from . import cudex_bulk_queue_v2 as queue
from .cudex_shared_gpu_v1 import shared_snapshot

VERSION = 'authorized_multi_gpu_cudex_v2'


def validate_policy(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    p, original = read(path), read(root / 'protocol.json')
    if (p.get('schema') != VERSION or p.get('authorization_status') != 'authorized'
            or not p.get('user_authorization', {}).get('message')):
        raise ValueError('explicit_parallel_resource_authorization_required')
    if Path(p['campaign_root']).resolve() != root or p['protocol'] != binding(root / 'protocol.json'):
        raise ValueError('parallel_policy_campaign_mismatch')
    if p['driver'] != binding(__file__):
        raise ValueError('parallel_driver_changed')
    resolve(p['parent_resource_policy'])
    resolve(p['shared_snapshot_implementation'])
    for key in ('gpu_job_limit', 'lineage_gpu_job_limit', 'per_gpu_job_timeout_seconds',
                'per_cpu_export_timeout_seconds'):
        if p.get(key) != original[key]:
            raise ValueError('parallel_resource_policy_cannot_change_' + key)
    if p.get('deadline_unix') != read(root / 'campaign_clock.json')['deadline_unix']:
        raise ValueError('parallel_policy_cannot_reset_deadline')
    cards = p['gpus']; maximum = p['maximum_parallel_gpu_jobs']
    if (type(maximum) is not int or not 1 <= maximum <= len(cards) <= 4
            or len({g['index'] for g in cards}) != len(cards)
            or len({g['uuid'] for g in cards}) != len(cards)
            or any(g['index'] not in original['gpu_indices'] or not g['uuid'].startswith('GPU-') for g in cards)
            or p['maximum_parallel_cpu_exports'] != maximum
            or type(p['minimum_free_gpu_mib']) is not int or p['minimum_free_gpu_mib'] < 4096):
        raise ValueError('bounded_distinct_GPU_lanes_and_memory_headroom_required')
    queue._verify_frozen(original)
    return p


class PoolResources:
    def __init__(self, root, policy_path, emit, stop):
        self.root = Path(root).resolve(); self.path = Path(policy_path).resolve()
        self.policy = validate_policy(root, policy_path)
        self.reference = binding(policy_path); self.emit = emit; self.stop = stop

    def check(self, snapshot, card):
        if snapshot['uuid'] != card['uuid'] or snapshot['gpu'] != card['index']:
            raise RuntimeError('pinned_GPU_identity_changed')
        if snapshot['free_memory_MiB'] < self.policy['minimum_free_gpu_mib']:
            raise RuntimeError('shared_GPU_memory_headroom_unavailable')

    def acquire(self, root, protocol, deadline, quiet_since, job_ref):
        if Path(root).resolve() != self.root or deadline != self.policy['deadline_unix']:
            raise ValueError('parallel_resource_campaign_or_deadline_mismatch')
        resolve(self.reference)
        if binding(self.root / 'protocol.json') != self.policy['protocol']:
            raise ValueError('original_protocol_changed')
        while time.time() < deadline and not self.stop.is_set():
            observations, candidates = [], []
            for card in self.policy['gpus']:
                try:
                    snap = shared_snapshot(card['index']); observations.append(snap)
                    self.check(snap, card)
                    candidates.append((snap['utilization'], -snap['free_memory_MiB'], card))
                except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
                    observations.append({'gpu': card['index'], 'resource_error': str(exc)})
            for _, _, card in sorted(candidates, key=lambda x: x[:2]):
                locks = ExitStack()
                try:
                    fds = []
                    for path in queue.resources.lock_paths(card['index']):
                        lock = locks.enter_context(path.open('r+'))
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        fds.append(lock.fileno())
                    snap = shared_snapshot(card['index']); self.check(snap, card)
                    if self.stop.is_set():
                        locks.close(); break
                    return {**snap, 'resource_mode': VERSION, 'resource_policy': self.reference,
                            'external_compute_sharing_authorized': True,
                            'project_lock_exclusive': True}, locks, fds
                except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
                    locks.close()
            self.emit(root, 'waiting_shared_GPU_slot_or_memory', job=job_ref, observations=observations)
            self.stop.wait(min(5, max(0, deadline - time.time())))
        if self.stop.is_set():
            raise RuntimeError('dispatch_stopped_before_new_GPU_reservation')
        raise TimeoutError('original_deadline_reached_before_new_GPU_reservation')


def dispatch(refs, execute, maximum, deadline, stop, finished, poll):
    """Submit each frozen reference at most once; bound both stages by lanes.

    Any worker failure stops new admission. Already running durable attempts
    finish and are accounted for; no failure is retried under another name.
    """
    pending = iter(refs); futures = {}; exhausted = False; error = None
    submitted, completed = 0, 0
    with ThreadPoolExecutor(max_workers=maximum, thread_name_prefix='cudex-lane') as executor:
        try:
            while futures or not exhausted:
                while not exhausted and not stop.is_set() and len(futures) < maximum:
                    if time.time() >= deadline:
                        error = error or 'original_campaign_deadline_no_new_submission'
                        stop.set(); exhausted = True; break
                    ref = next(pending, None)
                    if ref is None:
                        exhausted = True; break
                    futures[executor.submit(execute, ref)] = ref; submitted += 1
                if not futures:
                    break
                done, _ = wait(futures, timeout=5, return_when=FIRST_COMPLETED)
                for future in done:
                    ref = futures.pop(future)
                    try:
                        result = future.result()
                        if result['status'] != 'completed':
                            error = error or 'job_failure_needs_diagnosis:' + ref['path']
                            stop.set(); exhausted = True
                    except Exception as exc:
                        result = {'status': 'needs_recovery', 'error': f'{type(exc).__name__}: {exc}'}
                        error = error or result['error']; stop.set(); exhausted = True
                    completed += 1; finished(ref, result)
                poll()
        except BaseException:
            stop.set()
            raise
    return {'submitted': submitted, 'completed_futures': completed, 'error': error,
            'no_new_admission_after_failure': True}


def run(root, policy_path):
    root = Path(root).resolve(); policy_path = Path(policy_path).resolve()
    policy = validate_policy(root, policy_path); protocol = read(root / 'protocol.json')
    registry = read(resolve(protocol['registry']))
    if Path(registry['root']).resolve() != root or registry['recipe'] != protocol['recipe']:
        raise ValueError('same_registered_campaign_required')
    all_refs = protocol['qualification_jobs'] + protocol['bulk_jobs']
    if len(all_refs) != protocol['gpu_job_limit'] or len({r['path'] for r in all_refs}) != len(all_refs):
        raise ValueError('original_finite_manifest_invalid')
    for ref in (protocol['frozen_inputs'], protocol['candidate_manifest']):
        resolve(ref)
    queue.validate_gate(root, protocol)
    for ref in protocol['qualification_jobs']:
        queue.validate_result(resolve(ref))
    reference = binding(policy_path); stop = threading.Event(); mutex = threading.RLock()
    states, last_refresh = {}, [0.]
    old_acquire, old_state, old_version = queue._acquire_gpu, queue._state, queue.VERSION

    def state(directory, label, **extra):
        with mutex:
            job = extra.get('job')
            if job:
                states[job['path']] = {'stage': label, **extra, 'observed_unix': time.time()}
            overall = (label if label in {'stopped_for_diagnosis', 'completed', 'deadline_reached'}
                       else 'parallel_bulk_running' if states else label)
            old_state(directory, overall, resource_mode=VERSION, resource_policy=reference,
                maximum_parallel_gpu_jobs=policy['maximum_parallel_gpu_jobs'],
                maximum_parallel_cpu_exports=policy['maximum_parallel_cpu_exports'],
                active_job_states=list(states.values()), **({} if job else extra))

    adapter = PoolResources(root, policy_path, state, stop)

    def refresh(force=False):
        if force or time.monotonic() - last_refresh[0] >= 60:
            queue.status(root, write_status=True)
            queue.update_candidate_index(root, protocol)
            last_refresh[0] = time.monotonic()

    def finished(ref, result):
        with mutex:
            states.pop(ref['path'], None)
        write(root / 'parallel_results' / Path(ref['path']).parent.name / f'{time.time_ns()}.json', {
            'job': ref, 'resource_policy': reference, 'result': result, 'finished_unix': time.time()})
        state(root, 'parallel_bulk_running')

    with (root / '.supervisor.lock').open('a+') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        session = root / 'parallel_sessions' / f'{time.time_ns()}_{os.getpid()}'
        write(session.with_suffix('.json'), {'identity': queue.process_identity(os.getpid()),
            'started_unix': time.time(), 'resource_policy': reference,
            'original_protocol': binding(root / 'protocol.json'), 'qualification_repeated': False})
        queue._acquire_gpu, queue._state, queue.VERSION = adapter.acquire, state, VERSION
        try:
            deadline = read(root / 'campaign_clock.json')['deadline_unix']
            refresh(True)
            if (root / 'completion.json').exists():
                state(root, 'completed', counts=queue.status(root)); return
            pending, recover = [], []
            for ref in protocol['bulk_jobs']:
                folder = resolve(ref).parent
                if (folder / 'terminal.json').exists():
                    if read(folder / 'terminal.json')['status'] != 'completed':
                        raise RuntimeError('existing_failed_job_requires_diagnosis:' + str(folder))
                elif (folder / 'reservation.json').exists():
                    recover.append(ref)
                else:
                    pending.append(ref)
            # Drain inherited reservations before new parallel admission. This
            # includes a serial predecessor's CPU stage without new CPU locks.
            for ref in recover:
                state(root, 'recovering_inherited_reservation', job=ref)
                result = queue._run_job(root, protocol, ref, deadline, {})
                if result['status'] != 'completed':
                    raise RuntimeError('inherited_job_failed_no_retry:' + ref['path'])
                finished(ref, result)
            result = dispatch(pending, lambda ref: queue._run_job(root, protocol, ref, deadline, {}),
                policy['maximum_parallel_gpu_jobs'], deadline, stop, finished, refresh)
            refresh(True)
            write(session.with_name(session.name + '_terminal.json'), result)
            if result['error']:
                state(root, 'stopped_for_diagnosis', error=result['error'], counts=queue.status(root))
                return result
            counts = queue.status(root)
            if counts['terminal_jobs'] != protocol['gpu_job_limit'] or counts['unresolved_stages']:
                raise RuntimeError('manifest_not_fully_terminal')
            write(root / 'completion.json', {'finished_unix': time.time(),
                'protocol': binding(root / 'protocol.json'), 'resource_policy': reference,
                'counts': counts, 'api_calls': 0, 'physical_success': 'unmeasured', 'default_promoted': False})
            state(root, 'completed', counts=counts)
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
    parser.add_argument('--resource-policy', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.resource_policy)
