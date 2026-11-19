from copy import deepcopy
import pytest

from resimpair_tasks.source_first_reachable_layout_v1 import propose
from resimpair_pickplace.requested_tasks import Rejected


def example():
    request = {'id': 'test_scene', 'objects': [
        {'id': f'object_{i}', 'source_id': f'item:{i}', 'xy': [-.2, .3 * (i - 1)], 'yaw': 0.}
        for i in range(3)],
        'containers': [{'id': 'receiving_0', 'source_id': 'bin:0', 'xy': [.26, -.25], 'yaw': 0.},
                       {'id': 'receiving_1', 'source_id': 'bin:1', 'xy': [.26, .25], 'yaw': 0.}],
        'variant_choices': {f'item:{i}': f'fixed_size_{i}' for i in range(3)},
        'container_variant_choices': {'bin:0': {'sha256': 'a'}, 'bin:1': {'sha256': 'b'}},
        'tasks': [{'id': 'distribute_all', 'placements': [
            {'object': f'object_{i}', 'target': {'kind': 'container', 'container': f'receiving_{i % 2}',
             'offset_xy_m': [0., -.02 if i == 0 else .02]}} for i in range(3)]}]}
    assets = {f'item:{i}': {'half_size': [.025, .03, .05]} for i in range(3)}
    bins = {f'bin:{i}': {'dimensions_m': [.18, .18, .04]} for i in range(2)}
    return request, assets, bins


def test_no_source_size_variant_goal_mutation():
    request, assets, bins = example(); before = deepcopy((request, assets, bins))
    changed, receipt = propose(request, assets, bins, 123)
    assert (request, assets, bins) == before
    for key in ('tasks', 'variant_choices', 'container_variant_choices'):
        assert changed[key] == request[key]
    assert [(o['id'], o['source_id']) for o in changed['objects']] == [(o['id'], o['source_id']) for o in request['objects']]
    assert len(changed['objects']) == 3
    assert receipt['runtime_rescaling'] is False
    assert changed == propose(request, assets, bins, 123)[0]
    assert changed != propose(request, assets, bins, 124)[0]


def test_oversize_rejects_instead_of_shrinking_or_dropping_source():
    request, assets, bins = example(); assets['item:0']['half_size'] = [.3, .3, .1]
    with pytest.raises(Rejected):
        propose(request, assets, bins, 123)
    assert len(request['objects']) == 3
    assert assets['item:0']['half_size'] == [.3, .3, .1]


def test_table_task_not_silently_converted():
    request, assets, bins = example()
    request['tasks'][0]['placements'][0]['target'] = {'kind': 'table', 'xy': [0., 0.]}
    with pytest.raises(Rejected, match='explicit_container_goals'):
        propose(request, assets, bins, 123)
