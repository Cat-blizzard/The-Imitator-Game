import copy

import numpy as np
import pytest

from resimpair_pickplace.requested_tasks import Rejected
from resimpair_tasks.multitask_scene_factory_v1 import (
    goal_signature, select_tasks, sequential_vacancy, simple_slots, task_count,
)


def test_simple_packing_uses_capacity_and_keeps_payloads_separate():
    for radii in [[.03, .04], [.025, .03, .033], [.025, .03, .033, .03]]:
        evidence = simple_slots(radii, .13, angle=.937)
        assert evidence['passed']
        xy = np.array(evidence['offsets_local_xy_m'])
        assert np.all(np.linalg.norm(xy, axis=1) + radii <= .13 - .012 + 1e-6)
        for i in range(len(radii)):
            for j in range(i):
                assert np.linalg.norm(xy[i] - xy[j]) >= radii[i] + radii[j] + .025 - 1e-6
        assert not evidence['container_specific_settings']
    assert not simple_slots([.06] * 4, .10)['passed']


def test_task_count_range_is_bounded_and_reproducible():
    draws = [task_count([3, 8], seed) for seed in range(100)]
    assert set(draws) == set(range(3, 9))
    assert draws == [task_count([3, 8], seed) for seed in range(100)]
    for invalid in [2, 9, [4, 3], [3, 9], True, [3]]:
        with pytest.raises(ValueError): task_count(invalid, 1)


def vacancy_fixture():
    spec = {'objects': [{'id': 'a', 'xy': [-.3, -.4]}, {'id': 'b', 'xy': [0., 0.]}], 'containers': []}
    task = {'id': 'clear_then_place', 'placements': [
        {'object': 'b', 'target': {'kind': 'table', 'xy': [0., .4]}},
        {'object': 'a', 'target': {'kind': 'table', 'xy': [0., 0.]}}]}
    return spec, task, {'a': .04, 'b': .03}


def test_vacancy_updates_occupancy_and_demands_order():
    spec, task, radii = vacancy_fixture()
    evidence = sequential_vacancy(spec, task, radii, {})
    assert evidence['required_object_precedence'] == [['b', 'a']]
    assert evidence['checks'][1]['occupancy_before']['b'] == [0., .4]
    reverse = copy.deepcopy(task)
    reverse['placements'].reverse()
    # This also rejects an undeclared arbitrary dependency relation.
    with pytest.raises(Rejected): sequential_vacancy(spec, reverse, radii, {})


def test_vacancy_does_not_ignore_remaining_objects_or_containers():
    spec, task, radii = vacancy_fixture()
    spec['objects'].append({'id': 'c', 'xy': [.12, 0.]})
    radii.update(a=.08, c=.025)
    # C fits next to old B but the larger A cannot use B's vacated footprint.
    with pytest.raises(Rejected, match='footprint_overlap'):
        sequential_vacancy(spec, task, radii, {})
    spec, task, radii = vacancy_fixture()
    spec['containers'] = [{'id': 'bin', 'xy': [0., .4]}]
    with pytest.raises(Rejected, match='footprint_overlap'):
        sequential_vacancy(spec, task, radii, {'bin': .12})


def test_vacancy_rejects_regrasp_and_out_of_table_destinations():
    spec, task, radii = vacancy_fixture()
    task['placements'][1]['object'] = 'b'
    with pytest.raises(Rejected, match='two_distinct'):
        sequential_vacancy(spec, task, radii, {})
    spec, task, radii = vacancy_fixture()
    task['placements'][0]['target']['xy'] = [2., 0.]
    with pytest.raises(Rejected, match='outside_table'):
        sequential_vacancy(spec, task, radii, {})


def test_order_only_does_not_inflate_goal_count():
    _, task, _ = vacancy_fixture()
    reverse = copy.deepcopy(task); reverse['placements'].reverse()
    assert goal_signature(task) == goal_signature(reverse)
    changed = copy.deepcopy(task); changed['placements'][0]['target']['xy'][0] += .1
    assert goal_signature(task) != goal_signature(changed)


def test_task_subset_keeps_structural_coverage_and_is_seed_stable():
    ids = ['relocate_selected', 'contain_selected', 'orient_selected', 'arrange_group',
           'collect_group', 'selected_subset', 'split_group', 'clear_then_place']
    tasks = [{'id': name} for name in ids]
    for count in range(3, 9):
        selected = select_tasks(tasks, count, 55)
        assert len(selected) == count
        assert {'relocate_selected', 'collect_group', 'clear_then_place'} <= {x['id'] for x in selected}
        assert selected == select_tasks(tasks, count, 55)
