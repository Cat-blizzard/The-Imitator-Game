"""Authorization, retained budgets and real project locks; no GPU submissions."""
import fcntl

import pytest

from resimpair_grasp.common import binding, read, write
from resimpair_tasks import cudex_shared_gpu_v1 as shared


@pytest.fixture
def campaign(tmp_path):
    root = tmp_path / 'campaign'; root.mkdir()
    gate = tmp_path / 'gate.py'; gate.write_text('# not executed\n')
    write(root / 'protocol.json', {'gpu_job_limit': 5528, 'lineage_gpu_job_limit': 5670,
        'maximum_parallel_gpu_jobs': 1, 'per_gpu_job_timeout_seconds': 180,
        'per_cpu_export_timeout_seconds': 180, 'gpu_indices': [0, 1, 2, 3],
        'code': [], 'configuration': [], 'gate_validator': binding(gate), 'poll_seconds': 0})
    write(root / 'campaign_clock.json', {'deadline_unix': 10})
    write(root / 'proposal.json', {'status': 'not_authorization'})
    policy = root / 'policy.json'
    write(policy, {'schema': shared.VERSION, 'authorization_status': 'authorized',
        'user_authorization': {'message': 'test fixture explicit resource approval'},
        'campaign_root': str(root), 'protocol': binding(root / 'protocol.json'),
        'driver': binding(shared.__file__), 'prior_resource_proposal': binding(root / 'proposal.json'),
        'deadline_unix': 10, 'gpu_job_limit': 5528, 'lineage_gpu_job_limit': 5670,
        'maximum_parallel_gpu_jobs': 1, 'per_gpu_job_timeout_seconds': 180,
        'per_cpu_export_timeout_seconds': 180, 'gpu_index': 2, 'gpu_uuid': 'GPU-fixture',
        'minimum_free_gpu_mib': 12288})
    return root, policy


def snapshot():
    return {'gpu': 2, 'uuid': 'GPU-fixture', 'free_memory_MiB': 22000,
            'utilization': 85, 'compute_owners': ['external_training_fixture']}


def test_shared_owner_is_allowed_only_for_pinned_gpu_and_sufficient_memory(campaign):
    root, path = campaign; policy = shared.validate_policy(root, path)
    shared.check_shared(snapshot(), policy)
    with pytest.raises(RuntimeError, match='identity'):
        shared.check_shared({**snapshot(), 'uuid': 'GPU-different'}, policy)
    with pytest.raises(RuntimeError, match='insufficient_free'):
        shared.check_shared({**snapshot(), 'free_memory_MiB': 1024}, policy)


@pytest.mark.parametrize('key,value', [
    ('authorization_status', 'pending'), ('deadline_unix', 20),
    ('gpu_job_limit', 5536), ('maximum_parallel_gpu_jobs', 2),
    ('per_gpu_job_timeout_seconds', 360), ('lineage_gpu_job_limit', 5680),
])
def test_overlay_cannot_implicitly_authorize_or_expand_old_budget(campaign, key, value):
    root, path = campaign; policy = read(path); policy[key] = value
    bad = root / ('invalid_' + key + '.json'); write(bad, policy)
    with pytest.raises(ValueError):
        shared.validate_policy(root, bad)


def test_returned_project_lock_is_exclusive_and_snapshot_binds_policy(campaign, monkeypatch):
    root, path = campaign
    lock_path = root / 'gpu.lock'; lock_path.touch()
    monkeypatch.setattr(shared.time, 'time', lambda: 0)
    monkeypatch.setattr(shared, 'shared_snapshot', lambda _: snapshot())
    monkeypatch.setattr(shared.queue.resources, 'lock_paths', lambda _: [lock_path])
    adapter = shared.SharedResources(root, path, lambda *_a, **_k: None)
    snap, locks, fds = adapter.acquire(root, read(root / 'protocol.json'), 10, {}, {})
    assert snap['resource_policy'] == binding(path)
    assert snap['compute_owners'] == ['external_training_fixture']
    assert len(fds) == 1
    with lock_path.open('r+') as competitor:
        with pytest.raises(BlockingIOError):
            fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.close()
        fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_memory_drop_after_lock_prevents_submission_and_releases_lock(campaign, monkeypatch):
    root, path = campaign
    lock_path = root / 'gpu.lock'; lock_path.touch()
    clocks = iter([0, 0, 11])
    snaps = iter([snapshot(), {**snapshot(), 'free_memory_MiB': 1024}])
    monkeypatch.setattr(shared.time, 'time', lambda: next(clocks))
    monkeypatch.setattr(shared.time, 'sleep', lambda _: None)
    monkeypatch.setattr(shared, 'shared_snapshot', lambda _: next(snaps))
    monkeypatch.setattr(shared.queue.resources, 'lock_paths', lambda _: [lock_path])
    adapter = shared.SharedResources(root, path, lambda *_a, **_k: None)
    with pytest.raises(TimeoutError, match='original_campaign_deadline'):
        adapter.acquire(root, read(root / 'protocol.json'), 10, {}, {})
    with lock_path.open('r+') as competitor:
        fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not list(root.glob('**/reservation.json'))
