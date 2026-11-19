"""Required-goal admission tests; fixtures are never physical success evidence."""
import pytest

from resimpair_grasp.common import read, write
from resimpair_tasks import spatial_reachability_v5 as layout


def test_required_goal_set_is_explicit_and_cannot_drop_a_missing_goal():
    request = {'tasks': [{'id': 'pick'}, {'id': 'place'}]}
    assert layout.required_goals(request, ['all']) == ['pick', 'place']
    assert layout.required_goals(request, ['place']) == ['place']
    for goals in ([], ['pick', 'pick'], ['unknown']):
        with pytest.raises(ValueError): layout.required_goals(request, goals)


def test_one_endpoint_failure_prevents_layout_admission(tmp_path, monkeypatch):
    checked = []
    def select(scene, task, folder):
        checked.append(task['requested_task_id'])
        passed = task['requested_task_id'] != 'second'
        write(folder / 'arm_selection.json', {'fixture': True, 'passed': passed})
        return ({'fixture': True}, task) if passed else (None, None)
    monkeypatch.setattr(layout, 'bind_selection', select)
    tasks = [{'requested_task_id': name} for name in ('first', 'second', 'third')]
    report = layout.qualify({}, tasks, tmp_path)
    assert report['passed'] is False
    assert checked == ['first', 'second']
    assert report['not_checked'] == ['third']
    assert report['physical_success'] == 'unmeasured'


def test_all_required_goals_must_have_retained_evidence(tmp_path, monkeypatch):
    def select(scene, task, folder):
        write(folder / 'arm_selection.json', {'fixture': True, 'passed': True})
        return {'fixture': True}, task
    monkeypatch.setattr(layout, 'bind_selection', select)
    tasks = [{'requested_task_id': name} for name in ('first', 'second')]
    report = layout.qualify({}, tasks, tmp_path)
    assert report['passed'] is True
    assert len(report['checked']) == 2 and report['not_checked'] == []
    assert read(tmp_path / 'qualification.json') == report
