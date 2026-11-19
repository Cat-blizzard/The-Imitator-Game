"""Layout contracts independent of simulator, GPU and API availability."""
import copy

import numpy as np
import pytest

from resimpair_pickplace.requested_tasks import Rejected
from resimpair_tasks.spatial_factory_v2 import sample_request


def inputs():
    spec = {'id': 'scene', 'objects': [
        {'id': 'object_0', 'source_id': 'a', 'xy': [-.33, -.4], 'yaw': .5},
        {'id': 'object_1', 'source_id': 'b', 'xy': [-.33, .4], 'yaw': 0.}],
        'containers': [{'id': 'container_0', 'source_id': 'c', 'xy': [.12, 0.], 'yaw': 0.}],
        'distractors': {'count': 4, 'exclude_source_ids': ['a', 'b', 'c']},
        'tasks': [
            {'id': 'relocate', 'placements': [{'object': 'object_0', 'target': {'kind': 'table', 'xy': [0., -.4]}}]},
            {'id': 'orient', 'placements': [{'object': 'object_0', 'target': {'kind': 'table', 'xy': [0., -.4], 'yaw': .5 + np.pi / 2}}]},
            {'id': 'contain', 'placements': [{'object': 'object_1', 'target': {'kind': 'container', 'container': 'container_0', 'offset_xy_m': [0., 0.]}}]},
            {'id': 'arrange', 'placements': [
                {'object': 'object_0', 'target': {'kind': 'table', 'xy': [0., -.4]}},
                {'object': 'object_1', 'target': {'kind': 'table', 'xy': [0., .4]}}]},
        ]}
    assets = {'a': {'half_size': [.03, .04, .04]}, 'b': {'half_size': [.02, .02, .05]}}
    containers = {'c': {'dimensions_m': [.18, .18, .06], 'cavity': {'flat_floor_radius_m': .07}}}
    return spec, assets, containers


def test_replay_keeps_identity_shared_goals_and_relative_orientation():
    args = inputs(); before = copy.deepcopy(args)
    first, trace = sample_request(*args, 30)
    assert (first, trace) == sample_request(*args, 30)
    assert args == before
    assert [o['source_id'] for o in first['objects']] == ['a', 'b']
    assert first['distractors'] == args[0]['distractors']
    goals = [t['placements'][0]['target'] for t in first['tasks']]
    assert goals[0]['xy'] == goals[1]['xy'] == goals[3]['xy']
    delta = (goals[1]['yaw'] - first['objects'][0]['yaw']) % (2 * np.pi)
    assert delta == pytest.approx(np.pi / 2, abs=2e-6)
    offset = goals[2]['offset_xy_m']
    assert np.linalg.norm(offset) + np.linalg.norm([.02, .02]) + .008 <= .07
    assert trace['outcome_based_resampling'] is False


def test_many_seeds_span_workspace_and_keep_initial_and_goals_separate():
    points, stations, goals = [], [], []
    for seed in range(80):
        spec, trace = sample_request(*inputs(), seed)
        points.extend(o['xy'] for o in spec['objects'])
        stations.append(spec['containers'][0]['xy'])
        placements = spec['tasks'][3]['placements']
        goals.extend(p['target']['xy'] for p in placements)
        radii = [.05, np.linalg.norm([.02, .02]), np.linalg.norm([.09, .09]),
                 .05, np.linalg.norm([.02, .02])]
        positions = [o['xy'] for o in spec['objects']] + [stations[-1]] + [p['target']['xy'] for p in placements]
        for i in range(len(positions)):
            for j in range(i):
                assert np.linalg.norm(np.array(positions[i]) - positions[j]) > radii[i] + radii[j] + .03499
    assert np.ptp(np.array(points), axis=0).tolist() == pytest.approx([.50, 1.08], abs=.06)
    assert min(np.ptp(np.array(stations), axis=0) / [.48, 1.08]) > .8
    assert min(np.ptp(np.array(goals), axis=0) / [.48, 1.08]) > .8


def test_impossible_footprint_does_not_get_rescaled_or_replaced():
    spec, assets, containers = inputs()
    assets['a']['half_size'] = [1., 1., 1.]
    before = copy.deepcopy(spec)
    with pytest.raises(Rejected, match='asset_does_not_fit_sampling_workspace'):
        sample_request(spec, assets, containers, 1)
    assert spec == before
