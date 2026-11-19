import copy
import math

import numpy as np
import pytest

from resimpair_grasp.common import ROOT, read
from resimpair_pickplace.pose_bundle import materialize
from resimpair_pickplace.requested_tasks import Rejected, compile_scene, floor_fit, request_index
from resimpair_pickplace.requested_goal_audit import measured_goal


@pytest.fixture(scope='module')
def native_library(tmp_path_factory):
    out = tmp_path_factory.mktemp('cached_requested')
    indices = materialize(ROOT / 'assets/precomputed_grasps_v1', out)
    return read(indices['cudex'])['entries'], read(out / 'container_catalog.json')['assets']


def scene():
    return copy.deepcopy(read(ROOT / 'configs/cached_pickplace_requests_example.json')['scenes'][0])


def test_one_compiler_preserves_metric_goals_and_scene(native_library):
    from corl2026_workshop.experiments.autonomous_pipeline_v4.simulation import targets
    spec = scene(); original = copy.deepcopy(spec)
    compiled, tasks, checks = compile_scene(spec, *native_library, seed=701)
    assert spec == original
    assert compiled['objects'][0]['position'][:2] == spec['objects'][0]['xy']
    assert len(tasks) == 2 and compiled['randomization']['distractor_count'] == 4
    assert all(t['operations'][0]['arm'] is None for t in tasks)
    assert targets(compiled, tasks[1])['object'][:2] == spec['tasks'][1]['placements'][0]['target']['xy']
    c = compiled['supports'][0]
    assert np.allclose(targets(compiled, tasks[0])['object'][:2],
                       np.array(c['position'][:2]) + [.01*math.cos(.25), .01*math.sin(.25)], atol=1e-8)
    assert checks['objects'][0]['rescaled'] is False


@pytest.mark.parametrize('source', ['abo:B00WRDS0AU', 'objaverse:27abbf7bfb5043aaa73a4f9030f06f43'])
def test_different_shapes_use_identical_logic(native_library, source):
    spec = scene(); spec['objects'][0]['source_id'] = source
    compiled, tasks, checks = compile_scene(spec, *native_library, seed=702)
    assert checks['objects'][0]['source_id'] == source
    assert len(tasks) == 2 and checks['objects'][0]['cached_poses'] > 0


@pytest.mark.parametrize('problem,reason', [
    ('size', 'asset_exceeds_requested_size'),
    ('offset', 'no_container_with_requested_capacity'),
    ('table', 'footprint_outside_table'),
    ('scale', 'unsupported_or_missing_request_fields'),
    ('cache', 'no_cached_grasp'),
])
def test_invalid_requests_fail_before_rollout(native_library, problem, reason):
    spec = scene()
    if problem == 'size': spec['objects'][0]['max_dimension_m'] = .02
    if problem == 'offset': spec['tasks'][0]['placements'][0]['target']['offset_xy_m'] = [.15, 0.]
    if problem == 'table': spec['tasks'][1]['placements'][0]['target']['xy'] = [.8, 0.]
    if problem == 'scale': spec['objects'][0]['scale'] = .5
    if problem == 'cache': spec['objects'][0]['source_id'] = 'ycb_official:073-d_lego_duplo'
    with pytest.raises(Rejected) as error: compile_scene(spec, *native_library, seed=703)
    assert error.value.code == reason


def test_container_inner_capacity_and_offset_are_geometric():
    vertices = np.array([[-.02, -.01, 0.], [.02, .01, .04]])
    cavity = {'flat_floor_radius_m': .04}
    assert floor_fit(vertices, cavity, 0., 0., [0., 0.])['passed']
    assert not floor_fit(vertices, cavity, 0., 0., [.03, 0.])['passed']


def test_independent_requested_goal_rejects_other_successful_destination():
    states = np.zeros((20, 13)); states[:, :3] = [.1, -.3, .06]; states[:, 3] = 1.
    assert measured_goal(states, [.1, -.3, .06], 0.)['passed']
    assert not measured_goal(states, [.15, -.3, .06], 0.)['passed']
    assert not measured_goal(states, [.1, -.3, .06], .3)['passed']


def test_request_paths_cannot_escape_campaign():
    spec = read(ROOT / 'configs/cached_pickplace_requests_example.json')
    spec['scenes'][0]['id'] = '../outside'
    with pytest.raises(Rejected): request_index(spec)
