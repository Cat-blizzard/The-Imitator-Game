"""Explicit user-authorized GPU sharing overlay for an existing cuDex ledger.

The frozen queue, manifest, solver, exporter, gate and deadlines are unchanged.
Only GPU acquisition and resource metadata are replaced in this supervisor
process. The original durable stage runners retain inherited project locks.
Never calls prepare(), stops another GPU owner, or resubmits an unknown stage.
"""
from contextlib import ExitStack
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import time

from resimpair_grasp.common import binding, read, resolve, write
from . import cudex_bulk_queue_v2 as queue

VERSION = 'user_authorized_shared_cudex_queue_v1'


def validate_policy(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    policy = read(path)
    protocol = read(root / 'protocol.json')
    if (policy.get('schema') != VERSION or policy.get('authorization_status') != 'authorized'
            or not policy.get('user_authorization', {}).get('message')):
        raise ValueError('explicit_user_shared_resource_authorization_required')
    if Path(policy['campaign_root']).resolve() != root:
        raise ValueError('resource_policy_bound_to_different_campaign')
    if policy.get('protocol') != binding(root / 'protocol.json'):
        raise ValueError('frozen_campaign_protocol_changed')
    if policy.get('driver') != binding(__file__):
        raise ValueError('shared_resource_driver_changed')
    resolve(policy['prior_resource_proposal'])
    if (policy.get('deadline_unix') != read(root / 'campaign_clock.json')['deadline_unix']
            or policy.get('gpu_job_limit') != protocol['gpu_job_limit']
            or policy.get('lineage_gpu_job_limit') != protocol['lineage_gpu_job_limit']
            or policy.get('maximum_parallel_gpu_jobs') != 1
            or protocol['maximum_parallel_gpu_jobs'] != 1
            or policy.get('per_gpu_job_timeout_seconds') != protocol['per_gpu_job_timeout_seconds']
            or policy.get('per_cpu_export_timeout_seconds') != protocol['per_cpu_export_timeout_seconds']):
        raise ValueError('resource_overlay_must_preserve_budget_timeouts_and_deadline')
    if (type(policy.get('gpu_index')) is not int or policy['gpu_index'] not in protocol['gpu_indices']
            or not str(policy.get('gpu_uuid', '')).startswith('GPU-')
            or type(policy.get('minimum_free_gpu_mib')) is not int
            or policy['minimum_free_gpu_mib'] < 4096):
        raise ValueError('pinned_GPU_and_finite_memory_headroom_required')
    queue._verify_frozen(protocol)
    return policy


def shared_snapshot(index):
    snapshot = queue.resources.gpu_snapshot(index)
    value = subprocess.check_output(['nvidia-smi', '-i', str(index),
        '--query-gpu=uuid,memory.total,memory.free', '--format=csv,noheader,nounits'],
        text=True, timeout=20)
    uuid, total, free = [v.strip() for v in value.strip().split(',')]
    if uuid != snapshot['uuid']:
        raise RuntimeError('GPU_identity_changed_during_memory_query')
    return {**snapshot, 'total_memory_MiB': int(total), 'free_memory_MiB': int(free)}


def check_shared(snapshot, policy):
    if snapshot['gpu'] != policy['gpu_index'] or snapshot['uuid'] != policy['gpu_uuid']:
        raise RuntimeError('shared_GPU_identity_does_not_match_authorization')
    if snapshot['free_memory_MiB'] < policy['minimum_free_gpu_mib']:
        raise RuntimeError('insufficient_free_GPU_memory_for_shared_submission')
    # External compute owners and utilization are recorded, not relabelled idle.


class SharedResources:
    def __init__(self, root, policy_path, emit_state):
        self.root = Path(root).resolve()
        self.policy_path = Path(policy_path).resolve()
        self.policy = validate_policy(self.root, self.policy_path)
        self.reference = binding(self.policy_path)
        self.emit_state = emit_state

    def acquire(self, root, protocol, deadline, quiet_since, job_ref):
        if Path(root).resolve() != self.root or deadline != self.policy['deadline_unix']:
            raise ValueError('shared_acquisition_campaign_or_deadline_mismatch')
        resolve(self.reference)
        if binding(self.root / 'protocol.json') != self.policy['protocol']:
            raise ValueError('shared_acquisition_protocol_changed')
        index = self.policy['gpu_index']
        while time.time() < deadline:
            observation = None; reason = None; locks = ExitStack()
            try:
                observation = shared_snapshot(index)
                check_shared(observation, self.policy)
                fds = []
                for path in queue.resources.lock_paths(index):
                    lock = locks.enter_context(path.open('r+'))
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fds.append(lock.fileno())
                latest = shared_snapshot(index)
                check_shared(latest, self.policy)
                # The original queue embeds this exact snapshot in reservation.json.
                return {**latest, 'resource_mode': VERSION, 'resource_policy': self.reference,
                        'external_compute_sharing_authorized': True,
                        'project_lock_exclusive': True}, locks, fds
            except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as error:
                reason = f'{type(error).__name__}: {error}'
                locks.close()
            self.emit_state(root, 'waiting_shared_gpu_memory_or_project_lock', job=job_ref,
                observation=observation, reason=reason, deadline_unix=deadline)
            time.sleep(min(protocol['poll_seconds'], max(0, deadline - time.time())))
        raise TimeoutError('original_campaign_deadline_reached_waiting_for_shared_gpu')


def run(root, policy_path):
    root = Path(root).resolve()
    policy_path = Path(policy_path).resolve()
    policy = validate_policy(root, policy_path)
    reference = binding(policy_path)
    old_acquire, old_state, old_version = queue._acquire_gpu, queue._state, queue.VERSION

    def state(directory, label, **extra):
        old_state(directory, label, resource_policy=reference, resource_mode=VERSION, **extra)

    adapter = SharedResources(root, policy_path, state)
    write(root / 'resource_sessions' / f'{time.time_ns()}_{os.getpid()}.json', {
        'identity': queue.process_identity(os.getpid()), 'started_unix': time.time(),
        'resource_policy': reference, 'original_protocol': policy['protocol'],
        'actual_resource_mode': VERSION, 'same_manifest_ledger_and_deadline': True})
    queue._acquire_gpu, queue._state, queue.VERSION = adapter.acquire, state, VERSION
    try:
        return queue.run(root)
    finally:
        queue._acquire_gpu, queue._state, queue.VERSION = old_acquire, old_state, old_version


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--resource-policy', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.resource_policy)
