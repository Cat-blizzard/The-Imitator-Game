"""Recovery invariants on temporary ledgers; these are not grasp-success tests."""
from copy import deepcopy

import pytest

from resimpair_grasp.common import binding, write
from resimpair_tasks import cudex_quarantine_v3 as recovery


@pytest.fixture
def ledger(tmp_path):
    refs = []
    for name in ('failed', 'complete', 'unknown', 'pending'):
        folder = tmp_path / name
        write(folder / 'job.json', {'fixture': name}); refs.append(binding(folder / 'job.json'))
    bad = tmp_path / 'failed'
    write(bad / 'terminal.json', {'status': 'failed', 'failure_stage': 'gpu', 'returncode': 1})
    write(bad / 'optimizer_terminal.json', {'returncode': 1, 'complete_output_present': False})
    write(bad / 'reservation.json', {'job': refs[0]})
    (bad / 'worker.log').write_text(recovery.EMPTY_INPUT_ERROR)
    write(tmp_path / 'complete/terminal.json', {'status': 'completed'})
    write(tmp_path / 'unknown/reservation.json', {'job': refs[2]})
    entry = {'job': refs[0], 'terminal': binding(bad / 'terminal.json'),
             'optimizer_terminal': binding(bad / 'optimizer_terminal.json'),
             'worker_log': binding(bad / 'worker.log'), 'reservation': binding(bad / 'reservation.json'),
             'reason_code': 'empty_newton_input_runtime_error', 'physical_success': 'unmeasured', 'retry': False}
    return tmp_path, {'bulk_jobs': refs}, entry


def test_preserve_failure_recover_unknown_and_only_dispatch_unattempted(ledger):
    root, protocol, entry = ledger
    before = binding(root / 'failed/terminal.json')
    quarantined = recovery.validate_quarantine(protocol, [entry])
    pending, recover = recovery.partition_jobs(protocol, quarantined)
    assert pending == protocol['bulk_jobs'][3:]
    assert recover == protocol['bulk_jobs'][2:3]
    assert before == binding(root / 'failed/terminal.json')


def test_unlisted_failure_still_stops_campaign(ledger):
    _, protocol, _ = ledger
    with pytest.raises(RuntimeError, match='unreviewed_failure'):
        recovery.partition_jobs(protocol, set())


def test_changed_failure_cannot_reuse_old_review(ledger):
    root, protocol, entry = ledger
    (root / 'failed/worker.log').write_text('different failure')
    with pytest.raises(ValueError, match='hash changed'):
        recovery.validate_quarantine(protocol, [entry])


@pytest.mark.parametrize('filename', ['optimizer_result.json', 'result.json', 'export_reservation.json'])
def test_ambiguous_outputs_cannot_be_skipped(ledger, filename):
    root, protocol, entry = ledger
    write(root / 'failed' / filename, {'fixture': 'requires recovery'})
    with pytest.raises(ValueError, match='requires_recovery'):
        recovery.validate_quarantine(protocol, [entry])


def test_live_process_cannot_be_quarantined(ledger, monkeypatch):
    root, protocol, entry = ledger
    write(root / 'failed/gpu_process.json', {'identity': {'pid': 123}})
    monkeypatch.setattr(recovery.queue, 'process_alive', lambda _: True)
    with pytest.raises(RuntimeError, match='still_alive'):
        recovery.validate_quarantine(protocol, [entry])


def test_cannot_exclude_qualification_or_count_failure_as_success(ledger):
    _, protocol, entry = ledger
    with pytest.raises(ValueError, match='unique_original_bulk_job'):
        recovery.validate_quarantine({'bulk_jobs': protocol['bulk_jobs'][1:]}, [entry])
    changed = deepcopy(entry); changed['physical_success'] = 'passed'
    with pytest.raises(ValueError, match='no_retry_rule_invalid'):
        recovery.validate_quarantine(protocol, [changed])
    changed = deepcopy(entry); changed['retry'] = True
    with pytest.raises(ValueError, match='no_retry_rule_invalid'):
        recovery.validate_quarantine(protocol, [changed])
