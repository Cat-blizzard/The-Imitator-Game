from copy import deepcopy

import numpy as np
import pytest

from resimpair_tasks.metric_population_sampler_v2 import slots
from resimpair_tasks.source_first_slot_layout_v1 import disk_interval, group_proposal, propose


def group_example():
    model = slots([.035, .035], .13)
    vertices = np.array([[-.025, -.02, 0.], [-.025, .02, 0.], [.025, -.02, 0.], [.025, .02, 0.]])
    return {'container': 'receiver', 'support': {'yaw': 0.},
            'descriptor': {'cavity': {'flat_floor_radius_m': .13}},
            'objects': ['a', 'b'], 'vertices': [vertices, vertices],
            'yaws': [.3, -.5], 'packing': model, 'offsets': np.array(model['offsets']),
            'direction': np.array([-1., 0.])}


def test_ray_bound_rejects_disjoint_disks_without_widening_them():
    assert disk_interval([[0., 0.], [.04, 0.]], [.1, .08], [1., 0.]) == pytest.approx([0., .04])
    assert disk_interval([[0., .2]], .1, [1., 0.]) is None
    assert disk_interval([[.2, 0.]], .1, [1., 0.]) is None


def test_shift_preserves_original_spacing_and_floor_margins():
    group = group_example(); row = group_proposal(group, np.pi / 2, True)
    assert row['passed'] and 0 < row['shift_distance_m'] <= .04
    assert row['minimum_circle_boundary_margin_m'] >= .012
    assert all(x['passed'] and x['required_margin_m'] == .008 for x in row['final_floor_fit'])
    before, after = group['offsets'], np.array(row['new_offsets_xy_m'])
    assert np.linalg.norm(before[0] - before[1]) == pytest.approx(np.linalg.norm(after[0] - after[1]), abs=1e-12)


def test_actual_vertices_can_reject_despite_circular_summary():
    group = group_example(); group['vertices'][0] = np.array([[.2, 0., 0.]])
    row = group_proposal(group, 0., True)
    assert not row['passed']
    assert not all(x['passed'] for x in row['final_floor_fit'])
    assert row['shift_distance_m'] == 0.


def test_candidate_changes_only_target_offsets_and_request_identity():
    group = group_example()
    request = {'id': 'fixed', 'objects': [{'id': 'a', 'xy': [-.2, -.2], 'yaw': .3},
                                         {'id': 'b', 'xy': [-.2, .2], 'yaw': -.5}],
               'containers': [{'id': 'receiver', 'xy': [.1, 0.], 'yaw': 0.}],
               'variant_choices': {'a': 'fixed_a', 'b': 'fixed_b'},
               'tasks': [{'id': 'both', 'placements': [
                   {'object': oid, 'target': {'kind': 'container', 'container': 'receiver',
                                            'offset_xy_m': offset}}
                   for oid, offset in zip(group['objects'], group['packing']['offsets'])]}]}
    before = deepcopy(request)
    changed, receipt = propose(request, [group], np.pi / 2, True, 2)
    assert request == before
    for field in ('objects', 'containers', 'variant_choices'):
        assert changed[field] == before[field]
    assert changed['tasks'] != before['tasks']
    assert receipt['task_variant'] == 'new_target_positions'
    assert not receipt['same_goal_control_fix']
