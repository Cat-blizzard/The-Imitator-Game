"""CPU-only scheduling fixtures; none of these outputs is grasp evidence."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time

import pytest

from resimpair_grasp.common import binding, read, write
from resimpair_tasks import cudex_multigpu_v2 as multi
from resimpair_tasks import cudex_shared_gpu_v1 as prior


@pytest.fixture
def campaign(tmp_path):
    root = tmp_path / 'campaign'; root.mkdir()
    gate = root / 'fixture_gate.py'; gate.write_text('# not executed\n')
    write(root / 'protocol.json', {'gpu_job_limit': 5528, 'lineage_gpu_job_limit': 5670,
        'maximum_parallel_gpu_jobs': 1, 'maximum_parallel_cpu_exports': 1,
        'per_gpu_job_timeout_seconds': 180, 'per_cpu_export_timeout_seconds': 180,
        'gpu_indices': [0, 1, 2, 3], 'code': [], 'configuration': [], 'gate_validator': binding(gate)})
    deadline = time.time() + 10
    write(root / 'campaign_clock.json', {'deadline_unix': deadline})
    write(root / 'parent_policy.json', {'fixture': True})
    path = root / 'parallel_policy.json'
    write(path, {'schema': multi.VERSION, 'authorization_status': 'authorized',
        'user_authorization': {'message': 'explicit parallel test fixture'},
        'campaign_root': str(root), 'protocol': binding(root / 'protocol.json'),
        'driver': binding(multi.__file__), 'parent_resource_policy': binding(root / 'parent_policy.json'),
        'shared_snapshot_implementation': binding(prior.__file__), 'gpu_job_limit': 5528,
        'lineage_gpu_job_limit': 5670, 'per_gpu_job_timeout_seconds': 180,
        'per_cpu_export_timeout_seconds': 180, 'deadline_unix': deadline,
        'gpus': [{'index': i, 'uuid': f'GPU-fixture-{i}'} for i in (0, 1)],
        'maximum_parallel_gpu_jobs': 2, 'maximum_parallel_cpu_exports': 2,
        'minimum_free_gpu_mib': 12288})
    return root, path


@pytest.mark.parametrize('field,value', [('gpu_job_limit', 5530), ('lineage_gpu_job_limit', 5672),
    ('deadline_unix', 99999999999), ('maximum_parallel_gpu_jobs', 3), ('authorization_status', 'pending')])
def test_parallelism_never_creates_a_new_allowance(campaign, field, value):
    root, path = campaign; p = read(path)
    assert multi.validate_policy(root, path)['maximum_parallel_gpu_jobs'] == 2
    p[field] = value; altered = root / (field + '.json'); write(altered, p)
    with pytest.raises(ValueError):
        multi.validate_policy(root, altered)


def test_simultaneous_lanes_claim_distinct_GPU_locks(campaign, monkeypatch):
    root, path = campaign
    for i in (0, 1): (root / f'gpu_{i}.lock').touch()
    monkeypatch.setattr(multi, 'shared_snapshot', lambda i: {
        'gpu': i, 'uuid': f'GPU-fixture-{i}', 'free_memory_MiB': 24000,
        'utilization': 40, 'compute_owners': ['external_owner_fixture']})
    monkeypatch.setattr(multi.queue.resources, 'lock_paths', lambda i: [root / f'gpu_{i}.lock'])
    stop = threading.Event(); adapter = multi.PoolResources(root, path, lambda *_a, **_k: None, stop)
    def get(i):
        return adapter.acquire(root, read(root / 'protocol.json'), read(path)['deadline_unix'], {}, {'path': str(i)})
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = [f.result(timeout=3) for f in [pool.submit(get, 0), pool.submit(get, 1)]]
    try:
        assert {first[0]['gpu'], second[0]['gpu']} == {0, 1}
        assert first[0]['resource_policy'] == second[0]['resource_policy'] == binding(path)
        assert first[0]['compute_owners'] == ['external_owner_fixture']
    finally:
        first[1].close(); second[1].close()


def test_memory_wait_creates_no_reservation(campaign, monkeypatch):
    root, path = campaign; stop = threading.Event()
    monkeypatch.setattr(multi, 'shared_snapshot', lambda i: {
        'gpu': i, 'uuid': f'GPU-fixture-{i}', 'free_memory_MiB': 1000, 'utilization': 99})
    adapter = multi.PoolResources(root, path, lambda *_a, **_k: stop.set(), stop)
    with pytest.raises(RuntimeError, match='before_new_GPU_reservation'):
        adapter.acquire(root, read(root / 'protocol.json'), read(path)['deadline_unix'], {}, {'path': 'fixture'})
    assert not list(root.glob('**/reservation.json'))


def test_unique_jobs_and_at_most_four_simultaneous_assets():
    stop = threading.Event(); lock = threading.Lock(); called = []; done = []; live = [0, 0]
    barrier = threading.Barrier(4)
    def execute(ref):
        with lock:
            called.append(ref['path']); live[0] += 1; live[1] = max(live)
        barrier.wait(timeout=2); time.sleep(.01)
        with lock: live[0] -= 1
        return {'status': 'completed'}
    result = multi.dispatch([{'path': str(i)} for i in range(8)], execute, 4, time.time() + 5, stop,
        lambda ref, _r: done.append(ref['path']), lambda: None)
    assert result['error'] is None and result['submitted'] == 8
    assert sorted(called) == sorted(done) == [str(i) for i in range(8)]
    assert live == [0, 4]


def test_failure_drains_admitted_jobs_and_does_not_admit_replacements():
    stop = threading.Event(); barrier = threading.Barrier(2); called = []; done = []
    def execute(ref):
        called.append(ref['path']); barrier.wait(timeout=2)
        if ref['path'] == '0': return {'status': 'failed'}
        time.sleep(.05); return {'status': 'completed'}
    result = multi.dispatch([{'path': str(i)} for i in range(10)], execute, 2, time.time() + 5, stop,
        lambda ref, _r: done.append(ref['path']), lambda: None)
    assert result['error'] and result['submitted'] == 2
    assert sorted(called) == sorted(done) == ['0', '1']


def test_original_deadline_blocks_all_new_submissions():
    result = multi.dispatch([{'path': 'fixture'}], lambda _r: pytest.fail('must not execute'),
        4, time.time() - 1, threading.Event(), lambda *_a: None, lambda: None)
    assert result['submitted'] == 0 and 'deadline' in result['error']
