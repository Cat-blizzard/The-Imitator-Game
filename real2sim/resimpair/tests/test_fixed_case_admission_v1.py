from pathlib import Path
import pytest

from resimpair_grasp.common import write
from resimpair_tasks import fixed_case_admission_v1 as module


def example(tmp_path):
    parent = tmp_path / 'campaign'; root = parent / 'two_cases'
    write(root / 'protocol.json', {'campaign_root': str(parent)})
    return parent, root, {'id': 'first'}


def test_existing_reservation_cannot_submit_second_case(tmp_path, monkeypatch):
    parent, root, case = example(tmp_path)
    write(root / 'executions' / case['id'] / 'reservation.json', {'case': case})
    monkeypatch.setattr(module.retained, 'run', lambda *a, **kw: pytest.fail('driver must not run'))
    with pytest.raises(ValueError, match='never_retry'):
        module.guarded_driver(root, case)


def test_completed_first_case_does_not_call_limit_one_again(tmp_path, monkeypatch):
    parent, root, case = example(tmp_path)
    folder = root / 'executions' / case['id']
    write(folder / 'reservation.json', {'case': case}); write(folder / 'terminal.json', {'actual': True})
    monkeypatch.setattr(module.retained, 'run', lambda *a, **kw: pytest.fail('would select second case'))
    assert module.guarded_driver(root, case)['state'] == 'already_terminal_no_submission'


def test_unapproved_case_cannot_consume_budget_even_after_race(tmp_path, monkeypatch):
    parent, root, case = example(tmp_path)
    def raced_driver(*args, **kwargs):
        module.retained.write(parent / 'budget/reservations/0000.json',
            {'case': {'id': 'second'}, 'campaign': str(root), 'cohort': root.name})
    monkeypatch.setattr(module.retained, 'run', raced_driver)
    with pytest.raises(ValueError, match='before_budget_reservation'):
        module.guarded_driver(root, case)
    assert not (parent / 'budget/reservations/0000.json').exists()
