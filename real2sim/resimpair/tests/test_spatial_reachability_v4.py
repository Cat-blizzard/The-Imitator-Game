"""Necessary workspace checks, not IK/physics success assertions."""
import copy

import numpy as np
import pytest

from resimpair_pickplace.cartesian_reachability import pose
from resimpair_tasks import spatial_reachability_v4 as layout


def candidate(offset=(0., 0., .02)):
    tcp = np.eye(4); tcp[:3, 3] = offset
    return {'id': 'candidate', 'tcp_in_asset': tcp.tolist()}


def test_different_arms_reaching_individual_endpoints_is_insufficient():
    roots = [[0., -.6, 0.], [0., .6, 0.]]
    source, target = pose([.25, -.6, .04]), pose([.25, .6, .04])
    check = layout.common_arm(source, target, [candidate()], roots, .90)
    assert not check['passed']
    assert check['candidates'][0]['endpoint_distances_m']['grasp'] < .90
    assert check['candidates'][1]['endpoint_distances_m']['release'] < .90


def test_actual_cached_TCP_offset_and_height_are_used_not_object_center():
    roots = [[0., -.35, 0.], [0., .35, 0.]]
    start, end = pose([.65, -.2, .05]), pose([.65, .2, .05])
    assert layout.common_arm(start, end, [candidate()], roots)['passed']
    assert not layout.common_arm(start, end, [candidate((.3, 0., 0.))], roots)['passed']
    assert not layout.common_arm(start, end, [candidate((0., 0., .8))], roots)['passed']


def test_world_rotation_changes_cached_local_TCP_consistently():
    roots = [[0., -.35, 0.], [0., .35, 0.]]
    a, b = pose([.7, 0., .05], 0.), pose([.7, 0., .05], np.pi)
    assert not layout.common_arm(a, a, [candidate((.3, 0., 0.))], roots)['passed']
    assert layout.common_arm(b, b, [candidate((.3, 0., 0.))], roots)['passed']


def test_sampling_cannot_relax_original_runtime_reach_bound():
    with pytest.raises(ValueError, match='bounded_radius'):
        layout.common_arm(np.eye(4), np.eye(4), [candidate()], [[0., 0., 0.]]*2, 1.1)


def test_packed_collection_offsets_rotate_together_and_original_is_preserved():
    original = {'id': 'fixture_spatialv2', 'objects': [
        {'id': 'a', 'source_id': 'a', 'xy': [-.3, -.3], 'yaw': 0.},
        {'id': 'b', 'source_id': 'b', 'xy': [-.3, .3], 'yaw': 0.}],
        'containers': [{'id': 'c', 'source_id': 'c', 'xy': [.1, 0.], 'yaw': 0.}],
        'distractors': {'count': 3}, 'tasks': [{'id': 'collect_pair', 'placements': [
            {'object': 'a', 'target': {'kind': 'container', 'container': 'c', 'offset_xy_m': [-.04, 0.]}},
            {'object': 'b', 'target': {'kind': 'container', 'container': 'c', 'offset_xy_m': [.04, 0.]}}]}]}
    assets = {k: {'half_size': [.02, .02, .02]} for k in ('a', 'b')}
    containers = {'c': {'dimensions_m': [.25, .25, .08], 'cavity': {'flat_floor_radius_m': .10}}}
    before = copy.deepcopy(original)
    request, trace = layout.proposal(original, assets, containers, 12)
    offsets = [p['target']['offset_xy_m'] for p in request['tasks'][0]['placements']]
    assert np.linalg.norm(np.asarray(offsets[0]) - offsets[1]) == pytest.approx(.08, abs=2e-6)
    assert original == before
    assert request == layout.proposal(original, assets, containers, 12)[0]
    assert request['objects'][0]['source_id'] == 'a'
