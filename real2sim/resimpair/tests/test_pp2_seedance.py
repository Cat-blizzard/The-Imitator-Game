import time

from resimpair_pp2 import seedance as sd


def test_estimate_matches_the_published_720p_rate():
    # Ark price table: Seedance 2.0, 720p, 5 s without input video ~ 4.97 CNY
    assert abs(sd.estimate_cny('2.0', '720p', 5) - 4.97) < .01


def test_duration_grows_with_steps_within_model_range():
    one, three = {'steps': [1]}, {'steps': [1, 2, 3]}
    assert sd.duration_s(one, '2.0') == 6 and sd.duration_s(three, '2.0') == 12
    assert sd.duration_s({'steps': list(range(9))}, '2.0') == 15


def test_budget_counts_billed_and_unresolved_submissions(tmp_path):
    ledger = sd.Ledger(tmp_path / 'ledger.jsonl')
    meta = lambda rid: {'request_id': rid, 'model': 'm', 'resolution': '720p', 'duration_s': 6, 'estimate_cny': 6.}
    ledger.add(event='submitting', **meta('a')); ledger.add(event='succeeded', cost_cny=5.5, **meta('a'))
    ledger.add(event='submitting', **meta('b')); ledger.add(event='unknown', **meta('b'))
    ledger.add(event='submitting', **meta('c')); ledger.add(event='rejected', **meta('c'))
    latest, spent = ledger.state()
    assert latest['b']['event'] == 'unknown' and spent == 11.5


def test_reconcile_adopts_the_one_matching_task_and_never_guesses(tmp_path, monkeypatch):
    ledger = sd.Ledger(tmp_path / 'ledger.jsonl')
    meta = lambda rid: {'request_id': rid, 'model': 'm', 'resolution': '720p', 'duration_s': 6, 'estimate_cny': 6.}
    ledger.add(event='submitting', **meta('a')); ledger.add(event='unknown', **meta('a'))
    now = time.time()
    monkeypatch.setattr(sd, 'call', lambda *a, **k: {'items': [{'id': 't1', 'model': 'm', 'created_at': now}]})
    assert sd.reconcile(ledger) == ['adopted']
    assert ledger.state()[0]['a'] == {**ledger.rows()[-1]} and ledger.rows()[-1]['task_id'] == 't1'
    ledger.add(event='submitting', **meta('b')); ledger.add(event='unknown', **meta('b'))
    monkeypatch.setattr(sd, 'call', lambda *a, **k: {'items': []})
    assert sd.reconcile(ledger) == []  # too recent to conclude nothing was created
    assert sd.reconcile(ledger, settle_s=-1) == ['not_created']
