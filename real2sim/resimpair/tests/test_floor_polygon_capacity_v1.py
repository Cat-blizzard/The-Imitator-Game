"""CPU boundary, identity and compiler-dispatch checks; no actors or dynamics."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from resimpair_grasp.common import binding, digest, read, write
from resimpair_pickplace import requested_tasks as legacy
from resimpair_tasks import floor_polygon_capacity_v1 as geometry
from resimpair_tasks import floor_polygon_factory_v1 as adapter


def cavity(center=(0., 0.)):
    return {'center_xy_m': list(center), 'floor_z_m': 0., 'floor_tolerance_m': .003,
            'flat_floor_radius_m': .10}


def disk(boundary=None, z=None):
    boundary = np.asarray(boundary if boundary is not None else
        [[-.3, -.2], [.3, -.2], [.3, .2], [-.3, .2]], float)
    vertices = np.c_[np.vstack((boundary.mean(0), boundary)), np.zeros(len(boundary) + 1)]
    if z is not None:
        vertices[:, 2] = z
    faces = [[0, i + 1, (i + 1) % len(boundary) + 1] for i in range(len(boundary))]
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def footprint(source, dimensions=(.06, .04), shift=(0., 0.)):
    size = np.asarray(dimensions) / 2
    points = np.c_[geometry.rectangle_corners(np.array([-size, size])) + shift, np.zeros(4)]
    return {'native': points.copy(), 'visual': points.copy(), 'envelope': points.copy(),
            'receipt': {'source_id': source}, 'evidence': {'path': source, 'sha256': 'synthetic'}}


def bare_branch(floor, payload, center=(0., 0.)):
    branch = object.__new__(adapter.GeometryBranch)
    branch.floors = {digest(cavity(center)): {**floor, 'evidence': {'path': 'floor', 'sha256': 'synthetic'}}}
    branch.by_vertices = {adapter.vertices_key(payload['visual']): [payload]}
    branch.floor_checks = []; branch.pair_checks = []
    branch.goals = {}; branch.goals_by_id = {}
    return branch


def test_floor_requires_actual_connected_convex_triangle_cover():
    floor = geometry.extract_floor(disk(), cavity())
    assert floor['triangle_count'] == 4 and floor['euler_number'] == 1
    assert floor['floor_polygon_area_m2'] == pytest.approx(.24)
    assert floor['minimum_normal_z'] == 1.
    assert geometry.floor_margin(floor, [[0., 0.]]) == pytest.approx(.2)
    assert not floor['topology_repaired']


def test_bbox_corners_outside_beveled_floor_are_rejected():
    mesh = disk([[-.3, -.15], [-.25, -.2], [.25, -.2], [.3, -.15],
                 [.3, .15], [.25, .2], [-.25, .2], [-.3, .15]])
    floor = geometry.extract_floor(mesh, cavity())
    assert geometry.floor_margin(floor, [[.299, .199]]) < 0
    rectangle = geometry.inner_rectangle(floor)
    assert geometry.floor_margin(floor, geometry.rectangle_corners(rectangle)) >= .012
    assert np.prod(rectangle[1] - rectangle[0]) < .24


def test_topological_hole_cannot_be_filled_by_convex_hull():
    outer = np.array([[-.3, -.3], [.3, -.3], [.3, .3], [-.3, .3]])
    inner = outer / 3; vertices = np.c_[np.vstack((outer, inner)), np.zeros(8)]
    faces = []
    for i in range(4):
        j = (i + 1) % 4
        faces += [[i, j, 4 + j], [i, 4 + j, 4 + i]]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    with pytest.raises(ValueError, match='manifold_disk_without_holes'):
        geometry.extract_floor(mesh, cavity())


@pytest.mark.parametrize('defect', ['disconnected', 'duplicate', 'overlap', 'downward', 'height_span', 'steep'])
def test_invalid_native_floor_faces_do_not_admit(defect):
    mesh = disk()
    if defect == 'disconnected':
        other = disk(); other.apply_translation([.8, 0., 0.]); mesh = trimesh.util.concatenate([mesh, other])
    elif defect == 'duplicate':
        mesh.faces = np.vstack((mesh.faces, mesh.faces[0]))
    elif defect == 'overlap':
        other = disk(); other.apply_translation([.05, .05, 0.]); mesh = trimesh.util.concatenate([mesh, other])
    elif defect == 'downward':
        mesh.faces = mesh.faces[:, ::-1]
    elif defect == 'height_span':
        mesh.vertices[:, 2] = [-.002, .002, .002, -.002, -.002]
    else:
        mesh.vertices[:, :2] *= .001; mesh.vertices[0, 2] = .002
    with pytest.raises(ValueError):
        geometry.extract_floor(mesh, cavity())


def test_concave_connected_floor_is_not_treated_as_its_bbox():
    vertices = [[0, 0, 0], [.4, 0, 0], [.4, .1, 0], [.1, .1, 0], [.1, .4, 0], [0, .4, 0]]
    faces = [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5]]
    with pytest.raises(ValueError, match='nonconvex_floor'):
        geometry.extract_floor(trimesh.Trimesh(vertices=vertices, faces=faces, process=False), cavity((.05, .05)))


def test_original_safe_circle_positive_control_and_polygon_extra_capacity_are_separate():
    floor = geometry.extract_floor(disk(), cavity()); item = footprint('original')
    branch = bare_branch(floor, item)
    safe = branch.floor_fit(item['visual'], cavity(), 0., 0., [0., 0.])
    assert safe['passed'] and safe['legacy_circle_diagnostic']['passed']
    extra = branch.floor_fit(item['visual'], cavity(), 0., 0., [.21, 0.])
    assert extra['passed'] and not extra['legacy_circle_diagnostic']['passed']
    assert extra['required_margin_m'] == .012
    assert extra['circle_diagnostic_is_not_the_polygon_acceptance_rule']


@pytest.mark.parametrize('offset', [[.27, 0.], [.28, 0.], [.259, 0.]])
def test_payload_boundary_touch_overrun_and_eleven_mm_fail(offset):
    floor = geometry.extract_floor(disk(), cavity()); item = footprint('original')
    result = bare_branch(floor, item).floor_fit(item['visual'], cavity(), 0., 0., offset)
    assert not result['passed']


def test_cavity_center_applied_once_and_relative_yaw_retained():
    center = np.array([.12, -.08]); mesh = disk(); mesh.apply_translation([*center, 0.])
    floor = geometry.extract_floor(mesh, cavity(center)); item = footprint('long', (.2, .04), (.03, -.01))
    branch = bare_branch(floor, item, center)
    result = branch.floor_fit(item['visual'], cavity(center), np.pi / 2, 0., [0., 0.])
    transformed = item['visual'][:, :2] @ legacy.rotation(np.pi / 2).T + center
    assert result['passed']
    assert result['floor_margin_m'] == pytest.approx(geometry.floor_margin(floor, transformed))


def test_native_vertex_outside_visual_footprint_cannot_be_ignored():
    floor = geometry.extract_floor(disk(), cavity()); item = footprint('native')
    item['native'] = np.vstack((item['native'], [.31, 0., 0.]))
    result = bare_branch(floor, item).floor_fit(item['visual'], cavity(), 0., 0., [0., 0.])
    assert result['visual_floor_margin_m'] > .012
    assert not result['passed'] and result['native_floor_margins_m'][0] < 0


def test_first_fit_shelves_preserve_original_payload_count_offsets_and_clearances():
    floor = geometry.extract_floor(disk(), cavity())
    payloads = [footprint(str(i), (.22, .14), (.03, -.01)) for i in range(8)]
    packed = geometry.shelf_pack(payloads, floor, maximum_bins=4)
    assert packed['receiver_instances'] == 2
    assert [p['original_index'] for p in packed['placements']] == list(range(8))
    assert min(p['floor_boundary_margin_m'] for p in packed['placements']) >= .012
    assert min(p['gap_m'] for p in packed['pair_clearances']) >= .025
    for p in packed['placements']:
        translated = payloads[p['original_index']]['envelope'][:, :2] + p['origin_xy_in_receiver_m']
        assert geometry.floor_margin(floor, translated) >= .012
    with pytest.raises(ValueError, match='capacity_exhausted'):
        geometry.shelf_pack(payloads, floor, maximum_bins=1)
    for kwargs in ({'margin': .0119}, {'gap': .0249}, {'maximum_bins': 5}):
        with pytest.raises(ValueError):
            geometry.shelf_pack(payloads, floor, **kwargs)


def goal_branch(gap):
    branch = object.__new__(adapter.GeometryBranch); branch.pair_checks = []
    first = {'object': 'a', 'bounds': np.array([[0., 0.], [.05, .05]]), 'footprint': {'id': 'a'}}
    second = {'object': 'b', 'bounds': np.array([[.05 + gap, 0.], [.10 + gap, .05]]), 'footprint': {'id': 'b'}}
    branch.goals = {adapter.point_key([.075 + gap, .025], .04): [second]}
    branch.goals_by_id = {('a', adapter.point_key([.025, .025], .04)): first}
    return branch, [.075 + gap, .025], [('a', [.025, .025], .04)]


@pytest.mark.parametrize('gap', [0., -.001, .024])
def test_task_goal_pair_touch_overlap_and_twenty_four_mm_fail(gap):
    branch, xy, occupied = goal_branch(gap)
    with pytest.raises(legacy.Rejected, match='native_goal_footprints_too_close'):
        branch.separation(xy, .04, occupied, margin=.015)
    assert not branch.pair_checks[-1]['passed']


def test_only_original_goal_pair_call_uses_new_predicate():
    branch, xy, occupied = goal_branch(.026)
    branch.separation(xy, .04, occupied, margin=.015)
    assert branch.pair_checks[-1]['passed']
    # Same geometry still fails the original exterior-circle rule. That guard
    # must remain in force for source/goal, initial object and receiver checks.
    with pytest.raises(legacy.Rejected, match='footprint_overlap'):
        branch.separation(xy, .04, occupied)
    branch.separation(xy, .04, occupied, exclude=['a'])
    with pytest.raises(ValueError, match='bind_original_native'):
        branch.separation([.2, .2], .04, [], margin=.015)


def test_private_function_binding_preserves_legacy_code_and_globals():
    prior = legacy.compile_scene.__globals__['floor_fit']
    bound = adapter.locally_bound(legacy.compile_scene, floor_fit=lambda *args: None)
    assert bound.__code__ is legacy.compile_scene.__code__
    assert bound.__globals__ is not legacy.compile_scene.__globals__
    assert legacy.compile_scene.__globals__['floor_fit'] is prior


def test_saved_native_vertices_are_bound_and_not_silently_substituted(tmp_path):
    mesh = trimesh.creation.box(extents=[.06, .04, .03])
    collision = tmp_path / 'original.obj'; visual = tmp_path / 'visual.obj'
    mesh.export(collision); mesh.export(visual)
    descriptor = {'source_id': 'fixed', 'partition': 'heldout_asset', 'roles': {'rigid_manipulation_candidate': True},
        'collisions': [binding(collision)], 'visual': binding(visual), 'density_kg_m3': 400., 'mass_kg': .1,
        'dimensions_m': [.06, .04, .03], 'half_size': [.03, .02, .015]}
    asset = tmp_path / 'descriptor.json'; write(asset, descriptor)
    cache = tmp_path / 'cache.json'; write(cache, {'synthetic_geometry_test_only': True})
    entry = {'asset': binding(asset), 'cache': binding(cache), 'variant_id': 'unchanged'}
    vertices = np.asarray(trimesh.load(collision, process=False).vertices)
    shape = {'vertices': vertices.tolist(), 'scale': [1., 1., 1.], 'pose': np.eye(4).tolist(), 'density': 400.}
    native = {'objects': [{'id': 'asset', 'descriptor': binding(asset), 'mass': .1, 'shapes': [shape]}]}
    proof = tmp_path / 'native.json'; write(proof, native)
    payload = geometry.load_footprint(entry, binding(proof))
    assert payload['receipt']['saved_native_geometry'] == binding(proof)
    assert payload['receipt']['maximum_native_vertex_error_m'] == 0.
    bad = deepcopy(native); bad['objects'][0]['shapes'][0]['vertices'][0][0] += .01
    bad_path = tmp_path / 'bad_native.json'; write(bad_path, bad)
    with pytest.raises(ValueError, match='original_vertex_pose_scale_or_density'):
        geometry.load_footprint(entry, binding(bad_path))


def test_semantic_hold_survives_real_factory_output_writer(tmp_path, monkeypatch):
    """Even an otherwise compiled case cannot enter the default execution set."""
    request = tmp_path / 'requests.json'
    write(request, {'schema': adapter.factory.VERSION, 'scenes': [{'id': 'fixed',
        'objects': [{'id': 'object_0', 'source_id': 'semantic_conflict', 'xy': [0., 0.]}],
        'tasks': [{'id': 'one', 'placements': [{'object': 'object_0', 'target': {'kind': 'table', 'xy': [.1, 0.]}}]}]}]})
    evidence = tmp_path / 'hold.json'; write(evidence, {'source_id': 'semantic_conflict'})
    hold = {'source_id': 'semantic_conflict', 'reason': 'semantic_size_conflict', 'evidence': binding(evidence)}
    # The real local writer is exercised with a synthetic completed factory
    # result. No fake native cache is admitted or passed to the real compiler.
    def fake_factory(bundle, requests, output, seed):
        write(output / 'protocol.json', {})
        scene_path = output / 'scenes' / 'fixed' / 'scene.json'; write(scene_path, {})
        write(output / 'cohort.json', {'cases': [{'id': 'case', 'scene': binding(scene_path), 'eligible': True}]})
        write(output / 'summary.json', {'compiled_scenes': 1, 'compiled_tasks': 1})
    monkeypatch.setattr(adapter.factory, 'compile_requests', fake_factory)
    output = tmp_path / 'compiled'
    adapter.compile_requests(tmp_path / 'unused_bundle', request, output, 17,
        native_geometry_refs={}, source_holds=[hold])
    case = read(output / 'cohort.json')['cases'][0]
    assert case['eligible'] is False and case['admission_status'] == 'review_hold'
    assert case['geometry_admission']['semantic_status'] == 'hold'
    assert hold in case['geometry_admission']['holds']
    assert read(output / 'summary.json')['usable_candidates'] == 0
