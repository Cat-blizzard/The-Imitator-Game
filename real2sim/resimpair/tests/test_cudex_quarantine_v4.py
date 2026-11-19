"""Finite-failure admission and immutable ledger tests, never grasp evidence."""
from copy import deepcopy
import threading
import time

import pytest

from resimpair_grasp.common import binding, read, write
from resimpair_tasks import cudex_quarantine_v4 as recovery


@pytest.fixture
def failed_job(tmp_path):
    folder = tmp_path / 'asset'
    write(folder / 'job.json', {'fixture': 'empty Newton input'})
    ref = binding(folder / 'job.json')
    write(folder / 'terminal.json', {'status': 'failed', 'failure_stage': 'gpu', 'returncode': 1})
    write(folder / 'optimizer_terminal.json', {'returncode': 1, 'complete_output_present': False,
                                             'finished_unix': 100.})
    write(folder / 'reservation.json', {'job': ref})
    (folder / 'worker.log').write_text(
        '/curobo/opt/newton/newton_base.py\n'
        'q = q.view(self.n_problems, self.action_horizon * self.d_action)\n'
        + recovery.exact.EMPTY_INPUT_ERROR + '\n')
    return folder, {'bulk_jobs': [ref]}, ref


def test_only_exact_terminal_error_classified_and_original_preserved(failed_job):
    folder, protocol, ref = failed_job
    before = binding(folder / 'terminal.json')
    entry = recovery.failure_entry(protocol, ref)
    assert entry['retry'] is False and entry['physical_success'] == 'unmeasured'
    assert before == binding(folder / 'terminal.json')
    (folder / 'worker.log').write_text((folder / 'worker.log').read_text() + 'RuntimeError: different error\n')
    with pytest.raises(ValueError, match='unrecognized_error'):
        recovery.failure_entry(protocol, ref)


@pytest.mark.parametrize('name', ['optimizer_result.json', 'result.json', 'export_reservation.json'])
def test_never_skips_ambiguous_or_complete_output(failed_job, name):
    folder, protocol, ref = failed_job
    write(folder / name, {'pending_recovery': True})
    with pytest.raises(ValueError, match='requires_recovery'):
        recovery.failure_entry(protocol, ref)


def test_two_isolated_failures_continue_and_remain_failed():
    called, finished, classified = [], [], []
    def execute(ref):
        called.append(ref['path'])
        return {'status': 'failed' if ref['path'] in ('1', '3') else 'completed'}
    counter = recovery.FailureCounter()
    result = recovery.dispatch([{'path': str(i)} for i in range(5)], execute, 1,
        time.time() + 5, threading.Event(), lambda ref, res: finished.append((ref['path'], res['status'])),
        lambda: None, counter, lambda ref, _r: classified.append(ref['path']))
    assert result['error'] is None and result['submitted'] == 5
    assert classified == ['1', '3'] and counter.total == 2
    assert finished == [('0', 'completed'), ('1', 'failed'), ('2', 'completed'), ('3', 'failed'), ('4', 'completed')]


def test_three_consecutive_failures_stop_and_cannot_be_reset_by_success():
    called = []; counter = recovery.FailureCounter()
    def execute(ref):
        called.append(ref['path']); return {'status': 'failed'}
    result = recovery.dispatch([{'path': str(i)} for i in range(10)], execute, 1,
        time.time() + 5, threading.Event(), lambda *_: None, lambda: None, counter, lambda *_: None)
    assert called == ['0', '1', '2'] and 'consecutive' in result['error']
    counter.observe(False)
    assert 'consecutive' in counter.stop_reason()


def test_total_allowance_counts_historical_failures_and_inflight_capacity():
    called = []; counter = recovery.FailureCounter()
    for _ in range(31): counter.observe(True); counter.observe(False)
    def execute(ref):
        called.append(ref['path']); time.sleep(.01); return {'status': 'failed'}
    result = recovery.dispatch([{'path': str(i)} for i in range(10)], execute, 4,
        time.time() + 5, threading.Event(), lambda *_: None, lambda: None, counter, lambda *_: None)
    assert called == ['0'] and counter.total == 32 and 'total_limit' in result['error']


def test_unrecognized_failure_drains_workers_without_new_admission():
    called = []; finished = []; barrier = threading.Barrier(2)
    def execute(ref):
        called.append(ref['path']); barrier.wait(timeout=2)
        if ref['path'] == '1': time.sleep(.03)
        return {'status': 'failed' if ref['path'] == '0' else 'completed'}
    def reject(*_): raise ValueError('different_signature')
    result = recovery.dispatch([{'path': str(i)} for i in range(10)], execute, 2,
        time.time() + 5, threading.Event(), lambda ref, _r: finished.append(ref['path']),
        lambda: None, recovery.FailureCounter(), reject)
    assert 'different_signature' in result['error']
    assert sorted(called) == sorted(finished) == ['0', '1']


def test_deadline_and_unknown_submission_never_retry():
    call = lambda _: pytest.fail('must not submit')
    result = recovery.dispatch([{'path': '0'}], call, 4, time.time() - 1,
        threading.Event(), lambda *_: None, lambda: None, recovery.FailureCounter(), lambda *_: None)
    assert result['submitted'] == 0
    result = recovery.dispatch([{'path': str(i)} for i in range(3)],
        lambda _: {'status': 'needs_recovery', 'error': 'unknown_submission'}, 1,
        time.time() + 5, threading.Event(), lambda *_: None, lambda: None,
        recovery.FailureCounter(), lambda *_: pytest.fail('not terminal'))
    assert result['submitted'] == 1 and 'unknown_submission' in result['error']


def test_history_replays_failures_without_overwriting_evidence(failed_job, tmp_path):
    folder, protocol, ref = failed_job
    policy = tmp_path / 'policy.json'; write(policy, {'fixture': True})
    before = binding(folder / 'terminal.json')
    counter = recovery.FailureCounter()
    assert recovery.history(tmp_path, protocol, binding(policy), counter) == {ref['path']}
    assert counter.total == 1 and before == binding(folder / 'terminal.json')
    again = recovery.FailureCounter()
    recovery.history(tmp_path, protocol, binding(policy), again)
    assert again.total == 1
