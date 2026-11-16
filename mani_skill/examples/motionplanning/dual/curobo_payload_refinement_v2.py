#!/usr/bin/env python3
"""Isolated fixed-task-v2 geometry: only canonical right fruit uses 6.25mm.

Input and output are complete actual Task snapshots. Array shapes are finite
physical shape-local vertices[M,3], integer faces[F,3], actor-local spheres[N,4]
in metres, and normalized XYZ/WXYZ poses[7]. The frozen v1 recipe remains the
sole implementation for apples, pills and the left fruit. Every convex solid
of the right fruit is covered by the original six-LP/cell circumsphere method,
with fixed 6.25mm pitch. No current pose, seed, fingers or obstacle selects or
removes cells. Only proven empty cells are omitted by that unchanged algorithm.

Reserved payload slots, proof, count, hash and attachment sphere rows change
together. All robot geometry, world geometry, actual state, source manifest,
limits and collision permissions are preserved. The geometry-only bounded
cache contains immutable sphere tuples and proof JSON; every return is copied.
The generation runner imports this fixed geometry recipe. --emit-patch prints a
reviewable fixed-task-v2 lazy-import proposal and never modifies runtime.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import difflib
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np
from scipy.spatial import ConvexHull

from infrastructure.tig_runtime import resolve_runtime_paths
RUNTIME_PATHS = resolve_runtime_paths()
ROOT = RUNTIME_PATHS.workspace_root
DEV = RUNTIME_PATHS.code_root
from . import curobo_payload_refinement as v1  # noqa: E402
from infrastructure.curobo_tig_payload_worker import validate_request as legacy_validate_request  # noqa: E402

E = ROOT / 'infrastructure/validation/dual-arm/curobo-complete-20261116'
CANDIDATES = E / 'candidates/fruit_payload_v2'
FRUIT = v1.TASK_IDS[2]
ARM = 1
PITCH_M = .00625
RECIPE = 'fixed-task-payload-refinement-v2-right-fruit-6_25mm'
POLICY = 'fixed-task-v2'
RUNNER = DEV / 'mani_skill/examples/motionplanning/dual/validate_curobo_complete.py'
BASE_RUNNER_SHA256 = 'ff8b9230c0510c4cb529eb0d41e197508d6b8b66cebe178c0f945bd502a6b18f'


def _recipe() -> dict[str, Any]:
    # Reuse the strict finite canonical cache schema, with one fixed new recipe.
    return {**v1._fixed_recipe(FRUIT), 'recipe': RECIPE, 'pitch_m': PITCH_M}


@lru_cache(maxsize=64)
def _cached_geometry(key: str) -> tuple[tuple[tuple[float, ...], ...], str]:
    """Only geometry+recipe enters this immutable bounded LRU; no state enters."""
    data = json.loads(key)
    if data.get('schema') != 'actor-local-geometry-cache-v1' or data.get('recipe') != _recipe():
        raise ValueError('Expected the exact fixed v2 right-fruit geometry recipe')
    rows, proofs = [], []
    for local in data['components']:
        vertices = v1._array(local['vertices'], 3, 'cached full solid vertices')
        transform = v1._pose_matrix(local['shape_local_pose_wxyz'])
        spheres, proof = v1.tight_convex_spheres(vertices, PITCH_M, max_depth=2)
        spheres[:, :3] = v1._transform(spheres[:, :3], transform)
        proof.update(world_shape_name='', geometry_policy=local['geometry_policy'],
            shape_local_pose_wxyz=deepcopy(local['shape_local_pose_wxyz']),
            adaptive_rule_task_independent=True, candidate_refinement={
                'recipe': RECIPE, 'task_id': FRUIT, 'physical_arm_index': ARM,
                'policy': 'full_convex_solid_6_25mm_grid',
                'source_frame': 'physical_shape_local', 'output_frame': 'physical_actor_local',
                'mesh_sha256': local['mesh_sha256'], 'geometry_only': True,
                'seed_or_hand_dependent': False})
        rows.extend(spheres.tolist())
        proofs.append(proof)
    result = v1._array(rows, 4, 'complete v2 actor-local sphere cover')
    if np.any(result[:, 3] <= 0):
        raise ValueError('Conservative complete coverage requires positive sphere radii')
    return (tuple(tuple(float(v) for v in row) for row in result),
            json.dumps(proofs, sort_keys=True, allow_nan=False, separators=(',', ':')))


def _geometry(components: list[dict[str, Any]]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Fresh [N,4] array and proof mappings on every lookup, including hits."""
    key = v1._geometry_cache_key(components, _recipe())
    rows, proof_json = _cached_geometry(key)
    spheres, proofs = np.array(rows, dtype=np.float64, copy=True), json.loads(proof_json)
    for item, proof in zip(components, proofs, strict=True):
        proof['world_shape_name'] = item['name']
    return spheres, proofs


def refine_task_snapshot_v2(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Validate the full input and return a new complete policy-specific snapshot.

    Accept exact baseline, exact v1, or exact idempotent v2 snapshots. Geometry,
    proof and recipe identities are checked before any cached result is reused.
    v1's full validation/caching remains responsible for every other task/arm.
    """
    task, arm = v1._validate_snapshot(snapshot)
    if task != FRUIT or arm != ARM:
        return v1.refine_task_snapshot(snapshot)
    components = v1._components(snapshot)
    model = snapshot['payload_model']
    tags = []
    for proof in model['shape_coverage']:
        refinement = proof.get('candidate_refinement', {})
        if not isinstance(refinement, dict):
            raise ValueError('Payload refinement provenance must be a mapping')
        tags.append(refinement.get('recipe'))
    is_v2 = all(tag == RECIPE for tag in tags)
    if not is_v2:
        # This checks raw 25mm geometry or the exact v1 geometry/proofs, including
        # actual local mesh identity. A foreign/mixed/stale recipe fails closed.
        v1.refine_task_snapshot(snapshot)
    result = deepcopy(snapshot)
    spheres, proofs = _geometry(components)
    target = result['payload_model']
    target.update(capacity=len(spheres), sphere_count=len(spheres),
        spheres_actor_local=spheres.tolist(), shape_coverage=proofs, sphere_pitch_m=PITCH_M,
        geometry_sha256=v1.canonical_hash({'spheres': spheres.tolist(), 'coverage': proofs}))
    kin = result['robot_cfg']['kinematics']
    kin['collision_spheres'][v1.PAYLOAD_LINK] = [
        {'center': [0., 0., 0.], 'radius': -100.} for _ in range(len(spheres))]
    kin['extra_collision_spheres'][v1.PAYLOAD_LINK] = len(spheres)
    if result['payload_state']['attachment'] is not None:
        result['payload_state']['attachment']['spheres'] = spheres.tolist()
    result['coverage']['payload'].update(reserved_sphere_capacity=len(spheres), sphere_count=len(spheres),
        payload_sphere_pitch_m=PITCH_M, payload_sphere_policy=v1.BASE_POLICY,
        candidate_refinement={'recipe': RECIPE, 'isolated_candidate': True,
            'geometry_only': True, 'seed_or_hand_dependent': False,
            'task_id': FRUIT, 'physical_arm_index': ARM})
    if is_v2 and result != snapshot:
        raise ValueError('Existing v2 geometry, proof or recipe identity is stale')
    assert v1._nonpayload_view(result) == v1._nonpayload_view(snapshot)
    v1._validate_snapshot(result)
    return result


def _base_runner_source() -> str:
    """Read immutable promoted proposal rather than depend on future runtime edits."""
    path = E / 'candidates/pill_direct/validate_curobo_pill_direct.py'
    spec = importlib.util.spec_from_file_location('saved_pill_direct_candidate_for_v2_patch', path)
    if spec is None or spec.loader is None:
        raise ValueError('Archived CPU recipe proposal is unavailable')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = module.proposed_runner_source(module.archived_runner_source())
    if hashlib.sha256(source.encode()).hexdigest() != BASE_RUNNER_SHA256:
        raise ValueError('The reviewed complete runner base SHA changed')
    return source


def proposed_runner_source(source: str) -> str:
    """Minimal explicit policy/late-import extension; fixed-task-v1 stays exact."""
    edits = (
        ("PAYLOAD_REFINEMENTS = ('baseline', 'fixed-task-v1')",
         "PAYLOAD_REFINEMENTS = ('baseline', 'fixed-task-v1', 'fixed-task-v2')"),
        ("    if policy != 'fixed-task-v1':\n        raise ValueError('Unknown payload refinement policy')\n    from .curobo_payload_refinement import refine_task_snapshot\n    return refine_task_snapshot(snapshot)\n",
         "    if policy == 'fixed-task-v1':\n        from .curobo_payload_refinement import refine_task_snapshot\n        return refine_task_snapshot(snapshot)\n    if policy == 'fixed-task-v2':\n        from .curobo_payload_refinement_v2 import refine_task_snapshot_v2\n        return refine_task_snapshot_v2(snapshot)\n    raise ValueError('Unknown payload refinement policy')\n"),
        ("    assert configured_payload_snapshot(apple,'baseline') is apple\n",
         "    assert configured_payload_snapshot(apple,'baseline') is apple\n    assert configured_payload_snapshot(apple,'fixed-task-v2') == apple\n"),
    )
    result = source
    for before, after in edits:
        if result.count(before) != 1:
            raise ValueError('Reviewed complete runner policy anchor changed')
        result = result.replace(before, after, 1)
    import ast
    ast.parse(result)
    return result


def interface_patch() -> str:
    source = _base_runner_source()
    name = RUNNER.relative_to(DEV).as_posix()
    return ''.join(difflib.unified_diff(source.splitlines(keepends=True),
        proposed_runner_source(source).splitlines(keepends=True), fromfile='a/' + name, tofile='b/' + name))


def _native_saved_geometry_check(snapshot: dict[str, Any], refined: dict[str, Any]) -> dict[str, Any]:
    """Independent saved native FK + unchanged URDF + FCL, without GPU/PhysX."""
    path = E / 'diagnostics/fruit-lift-v1/diagnose.py'
    spec = importlib.util.spec_from_file_location('saved_fruit_geometry_diagnostic_for_v2', path)
    if spec is None or spec.loader is None:
        raise ValueError('Saved read-only geometry diagnostic is unavailable')
    diagnostic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnostic)
    directory = E / 'extensions-v1/curobo-fruit-s401-dev01'
    saved_fk = json.loads((directory / 'lift_and_hold-arm-1-fk.json').read_text())
    report = json.loads((directory / 'report.json').read_text())
    update = report['planning'][-1]['updates'][ARM]['native_update']
    assert report['planning'][-1]['phase'] == 'lift_and_hold'
    native_tcp = v1._pose_matrix([*saved_fk['positions'][0], *saved_fk['quaternions'][0]])
    tcp_object = v1._pose_matrix(snapshot['payload_state']['attachment']['tcp_to_object_wxyz'])
    native_object = native_tcp @ tcp_object
    old = np.asarray(snapshot['payload_model']['spheres_actor_local'])
    new = np.asarray(refined['payload_model']['spheres_actor_local'])
    old_base = np.c_[v1._transform(old[:, :3], native_object), old[:, 3]]
    saved_spheres = np.asarray(saved_fk['link_spheres'][v1.PAYLOAD_LINK])[0]
    error = float(np.abs(old_base - saved_spheres).max())
    assert error < 1e-6
    new_base = np.c_[v1._transform(new[:, :3], native_object), new[:, 3]]
    robot_spheres, _, urdf_errors = diagnostic.robot_spheres(snapshot, saved_fk)
    world = diagnostic.world_objects(snapshot, update)
    mask = update['payload']['source_policy']['robot_link_collision_masks']['planner'][v1.PAYLOAD_LINK]
    assert mask == [3] and world['scene::table::shape_0']['enabled']
    assert world['scene::table::shape_0']['layers'] == 1
    new_robot = {**robot_spheres, v1.PAYLOAD_LINK: new_base}
    self_pairs = [row for row in diagnostic.self_scan(new_robot, snapshot['robot_cfg']['kinematics'])
                  if v1.PAYLOAD_LINK in (row['first'], row['second'])]
    assert self_pairs and self_pairs[0]['minimum_gap_m'] > 0
    world_scan = diagnostic.world_scan({v1.PAYLOAD_LINK: new_base}, world, {v1.PAYLOAD_LINK: mask})
    assert not world_scan['nonpositive_pairs']
    table = snapshot['world']['cuboid']['scene::table::shape_0']
    table_transform = v1._pose_matrix(table['pose'])
    assert np.allclose(table_transform[:3, 2], [0., 0., 1.], atol=1e-7)
    top = table_transform[2, 3] + table['dims'][2] / 2.
    old_gap = float(np.min(old_base[:, 2] - old_base[:, 3]) - top)
    new_gap = float(np.min(new_base[:, 2] - new_base[:, 3]) - top)
    assert old_gap < 0 < new_gap
    physical = []
    for name in snapshot['payload_model']['target_obstacle_names']:
        a, b = world[name]['object'], world['scene::table::shape_0']['object']
        gap = float(diagnostic.fcl.distance(a, b, diagnostic.fcl.DistanceRequest(enable_signed_distance=True)).min_distance)
        assert gap > 0
        physical.append({'shape': name, 'actual_convex_table_gap_m': gap})
    return {'saved_native_model_capacity': len(old), 'candidate_capacity': len(new),
        'old_payload_native_FK_reconstruction_maximum_error_m': error,
        'unchanged_URDF_robot_native_FK_maximum_errors_m': urdf_errors,
        'old_table_minimum_sphere_gap_m': old_gap, 'new_table_minimum_sphere_gap_m': new_gap,
        'minimum_enabled_payload_self_pair': self_pairs[0],
        'payload_enabled_world_FCL_check': world_scan,
        'physical_actual_convex_table_gaps': physical,
        'native_mask_unchanged': mask, 'table_enabled_layer_unchanged': 1,
        'scope': 'saved native FK, unchanged robot URDF and CPU FCL; no native planner rerun'}


def self_test() -> dict[str, Any]:
    started = time.perf_counter()
    records, rejected, cache_checks = [], [], []
    _cached_geometry.cache_clear()

    def reject(label: str, action: Callable[[], Any]) -> None:
        try:
            action()
        except (ValueError, RuntimeError):
            rejected.append(label)
        else:
            raise AssertionError('Unsafe candidate accepted: ' + label)

    failed_path = E / 'extensions-v1/curobo-fruit-s401-dev01/lift_and_hold-arm-1-snapshot.json'
    failed = json.loads(failed_path.read_text())
    original = deepcopy(failed)
    cold = time.perf_counter()
    refined = refine_task_snapshot_v2(failed)
    cold = time.perf_counter() - cold
    first_info = _cached_geometry.cache_info()
    warm = time.perf_counter()
    assert refine_task_snapshot_v2(failed) == refined
    warm = time.perf_counter() - warm
    assert _cached_geometry.cache_info().hits == first_info.hits + 1
    assert refine_task_snapshot_v2(refined) == refined and failed == original
    assert v1._nonpayload_view(refined) == v1._nonpayload_view(failed)
    components = v1._components(failed)
    cover = v1._body_coverage(components, np.asarray(refined['payload_model']['spheres_actor_local']))
    evidence = _native_saved_geometry_check(failed, refined)
    records.append({'input': str(failed_path), 'input_sha256': hashlib.sha256(failed_path.read_bytes()).hexdigest(),
        'task': FRUIT, 'arm': ARM, 'cold_CPU_wall_s': cold, 'second_CPU_wall_s': warm,
        'coverage': cover, 'geometry_sha256': refined['payload_model']['geometry_sha256'],
        'nonpayload_exact_identity': True, 'idempotent': True, 'input_unmodified': True})

    # All six baseline snapshots validate and the five unaffected arms exactly
    # equal v1, including complete proof/metadata and reserved topology.
    for task in v1.TASK_IDS:
        for arm in (0, 1):
            path = E / f'task-source-snapshots-dev02/{task}-arm-{arm}.json'
            raw = json.loads(path.read_text())
            before = deepcopy(raw)
            value = refine_task_snapshot_v2(raw)
            assert raw == before
            same_v1 = task != FRUIT or arm != ARM
            if same_v1:
                assert value == v1.refine_task_snapshot(raw)
                assert refine_task_snapshot_v2(value) == value
            else:
                assert value['payload_model'] == refined['payload_model']
            for profile in ('approach', 'transport'):
                for operation in ('init_session', 'update_session'):
                    request = v1._request(value, operation, profile)
                    v1.validate_task_request(request, task)
                    legacy_validate_request(v1.validation_view(request, task))
            records.append({'input': str(path), 'task': task, 'arm': arm,
                'sphere_count': value['payload_model']['sphere_count'],
                'exactly_equals_v1': same_v1, 'Task_and_legacy_CPU_validators': 'passed'})

    # Native held-state schemas retain the exact actual grasp, fingers, world
    # and source exception, with only model/attachment sphere rows updated.
    for profile in ('approach', 'transport'):
        for operation in ('init_session', 'update_session'):
            request = v1._request(refined, operation, profile)
            v1.validate_task_request(request, FRUIT)
            legacy_validate_request(v1.validation_view(request, FRUIT))

    pristine = deepcopy(refined)
    refined['payload_model']['spheres_actor_local'][0][0] += 1.
    refined['payload_model']['shape_coverage'][0]['candidate_refinement']['recipe'] = 'polluted'
    assert refine_task_snapshot_v2(failed) == pristine
    array, proof = _geometry(components)
    array[0, 0] += 1.
    proof[0]['candidate_refinement']['recipe'] = 'polluted'
    array2, proof2 = _geometry(components)
    assert np.array_equal(array2, pristine['payload_model']['spheres_actor_local'])
    assert proof2 == pristine['payload_model']['shape_coverage']
    cache_checks.append('returned_snapshot_array_and_proof_pollution_isolated')

    variants = []
    world = deepcopy(failed)
    world['world']['mesh']['peer::panda_hand::shape_0']['pose'][0] += .013
    variants.append(('world', world))
    fingers = deepcopy(failed)
    kin = fingers['robot_cfg']['kinematics']
    for name, value in (('panda_finger_joint1', .031), ('panda_finger_joint2', .027)):
        kin['lock_joints'][name] = value
        kin['cspace']['default_joint_position'][kin['cspace']['joint_names'].index(name)] = value
    variants.append(('fingers', fingers))
    attached = deepcopy(failed)
    attached['payload_state'].update(source_support_allowed=False, source_support_obstacle_names=[])
    attached['payload_state']['attachment']['evidence_id'] += '-CPU_STATE_IDENTITY_TEST'
    attached['payload_state']['attachment']['object_pose_base_wxyz'][0] += .01
    variants.append(('attachment_and_source_permission', attached))
    detached = deepcopy(failed)
    detached['payload_state'].update(attachment=None, source_support_allowed=False, source_support_obstacle_names=[])
    variants.append(('detached_state', detached))
    cache_before = _cached_geometry.cache_info()
    for label, variant in variants:
        value = refine_task_snapshot_v2(variant)
        assert value['payload_model'] == pristine['payload_model']
        assert v1._nonpayload_view(value) == v1._nonpayload_view(variant)
        cache_checks.append(label + '_fresh_state_preserved_on_geometry_hit')
    assert _cached_geometry.cache_info().misses == cache_before.misses
    assert _cached_geometry.cache_info().hits == cache_before.hits + len(variants)

    # Tiny actual solids exercise cell-boundary cases and cheap key separation.
    vertices = np.asarray([[x, y, z] for x in (0., .001) for y in (0., .001) for z in (0., .001)])
    component = {'name': 'synthetic', 'vertices': vertices, 'faces': ConvexHull(vertices).simplices,
        'actor_vertices': vertices, 'transform': np.eye(4), 'mesh_sha256': v1.canonical_hash(vertices.tolist()),
        'proof': {'geometry_policy': 'synthetic_full_solid', 'shape_local_pose_wxyz': [0., 0., 0., 1., 0., 0., 0.]}}
    synthetic, _ = _geometry([component])
    v1._body_coverage([component], synthetic)
    changed_component = deepcopy(component)
    changed_component['vertices'] *= 1.125
    changed_component['actor_vertices'] = changed_component['vertices'].copy()
    changed_component['mesh_sha256'] = v1.canonical_hash(changed_component['vertices'].tolist())
    prior = _cached_geometry.cache_info()
    changed_spheres, _ = _geometry([changed_component])
    assert _cached_geometry.cache_info().misses == prior.misses + 1
    assert not np.array_equal(synthetic, changed_spheres)
    v1._body_coverage([changed_component], changed_spheres)
    key = v1._geometry_cache_key([component], _recipe())
    other_recipe_key = v1._geometry_cache_key([component], {**_recipe(), 'pitch_m': .0125})
    assert key != other_recipe_key
    reject('different_recipe_key_not_accepted_as_v2', lambda: _cached_geometry(other_recipe_key))
    reject('nonfinite_cache_recipe', lambda: v1._geometry_cache_key([component], {**_recipe(), 'pitch_m': float('nan')}))
    reject('mutable_state_in_cache_recipe', lambda: v1._geometry_cache_key([component], {**_recipe(), 'world': {}}))
    cache_checks.extend(['changed_actual_local_mesh_misses_cache', 'changed_recipe_cannot_hit_or_impersonate_v2'])

    mutations: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
        ('foreign_task', lambda s: s['payload_model'].update(task_id='foreign')),
        ('wrong_arm', lambda s: s.update(arm_index=0)),
        ('foreign_target', lambda s: s['payload_model'].update(target_actor_name='foreign')),
        ('stale_geometry_hash', lambda s: s['payload_model'].update(geometry_sha256='bad')),
        ('bad_capacity', lambda s: s['payload_model'].update(capacity=1)),
        ('active_reserved_slot', lambda s: s['robot_cfg']['kinematics']['collision_spheres'][v1.PAYLOAD_LINK][0].update(radius=.001)),
        ('forbidden_hand_ignore', lambda s: s['robot_cfg']['kinematics']['self_collision_ignore'][v1.PAYLOAD_LINK].append('panda_hand')),
        ('foreign_source_permission', lambda s: s['payload_state'].update(source_support_obstacle_names=['scene::table::shape_0'])),
        ('nonfinite_q', lambda s: s['q_start'].__setitem__(0, float('nan'))),
        ('attachment_mismatch', lambda s: s['payload_state']['attachment']['spheres'][0].__setitem__(3, .1)),
    ]
    name = failed['payload_model']['target_obstacle_names'][0]
    mutations.extend([
        ('missing_target_mesh', lambda s: s['world']['mesh'].pop(name)),
        ('stale_actual_local_mesh', lambda s: s['world']['mesh'][name]['vertices'][0].__setitem__(0, .5)),
        ('nonfinite_mesh', lambda s: s['world']['mesh'][name]['vertices'][0].__setitem__(0, float('inf'))),
        ('moving_frozen_source', lambda s: s['world']['mesh'][s['payload_model']['source_obstacle_names'][0]]['pose'].__setitem__(0, .5)),
    ])
    for label, mutate in mutations:
        invalid = deepcopy(failed)
        mutate(invalid)
        reject(label, lambda invalid=invalid: refine_task_snapshot_v2(invalid))
    for label, mutate in (
        ('stale_v2_recipe', lambda s: s['payload_model']['shape_coverage'][0]['candidate_refinement'].update(recipe='foreign')),
        ('malformed_v2_recipe_proof', lambda s: s['payload_model']['shape_coverage'][0].update(candidate_refinement=None)),
        ('missing_complete_solid_proof', lambda s: s['payload_model']['shape_coverage'][0].update(complete_solid_coverage=False)),
        ('stale_v2_sphere', lambda s: s['payload_model']['spheres_actor_local'][0].__setitem__(0, .5)),
    ):
        invalid = deepcopy(pristine)
        mutate(invalid)
        invalid['payload_model']['geometry_sha256'] = v1.canonical_hash({
            'spheres': invalid['payload_model']['spheres_actor_local'], 'coverage': invalid['payload_model']['shape_coverage']})
        if label == 'stale_v2_sphere':
            invalid['payload_state']['attachment']['spheres'] = deepcopy(invalid['payload_model']['spheres_actor_local'])
        reject(label, lambda invalid=invalid: refine_task_snapshot_v2(invalid))
    reject('malformed_snapshot', lambda: refine_task_snapshot_v2({}))

    # Original immutable manager retains topology, stale-token checks and
    # invalidation. This CPU double emits no native planning/grasp evidence.
    manager = v1.TaskSessionManager(FRUIT, v1._CPUBackend)
    for profile in ('approach', 'transport'):
        v1.dispatch_task(v1._request(pristine, 'init_session', profile), manager)
        first = v1.dispatch_task(v1._request(pristine, 'update_session', profile, 'v2-a'), manager)['result']
        assert first['version'] == 1
        v1.dispatch_task(v1._request(pristine, 'update_session', profile, 'v2-b'), manager)
        stale = {'id': 2, 'op': 'fk', 'session_id': 'right:' + profile,
            'snapshot_id': 'v2-a', 'positions': [pristine['q_start']]}
        reject(profile + '_stale_token', lambda stale=stale: v1.dispatch_task(stale, manager))

    source = _base_runner_source()
    proposed = proposed_runner_source(source)
    # The proposed runtime policy extends only the constant, routing function
    # and a CPU apple identity assertion. All original v1 code remains frozen.
    import ast
    old, new = ast.parse(source), ast.parse(proposed)
    changed = {'PAYLOAD_REFINEMENTS', 'configured_payload_snapshot', 'self_test'}
    def node_id(node: ast.AST) -> str:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            return node.name
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            return node.targets[0].id
        return ast.dump(node)
    for before, after in zip(old.body, new.body, strict=True):
        assert node_id(before) == node_id(after)
        if node_id(before) not in changed:
            assert ast.dump(before) == ast.dump(after)
    reject('changed_runner_policy_anchor', lambda: proposed_runner_source(proposed))
    expected = json.loads((CANDIDATES / 'prechange-sha256.json').read_text())['source_sha256']
    differences = [name for name, sha in expected.items() if hashlib.sha256(Path(name).read_bytes()).hexdigest() != sha]
    torch = sys.modules.get('torch')
    assert torch is None or not torch.cuda.is_initialized()
    return {'status': 'passed', 'policy': POLICY, 'recipe': RECIPE,
        'environment_steps': 0, 'GPU_initialized': False, 'native_planner_rerun': 'NOT_RUN',
        'actual_saved_snapshot_records': records, 'failed_lift_geometry_check': evidence,
        'cache_checks': cache_checks, 'negative_tests_rejected': len(rejected), 'negative_tests': rejected,
        'geometry_cache': str(_cached_geometry.cache_info()),
        'candidate_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'proposed_runner_base_sha256': BASE_RUNNER_SHA256,
        'proposed_runner_sha256': hashlib.sha256(proposed.encode()).hexdigest(),
        'current_formal_source_count': len(expected), 'current_formal_source_differences': differences,
        'native_success_or_full_route_effect': 'NOT_MEASURED; requires a new explicit native protocol',
        'CPU_wall_s': time.perf_counter() - started}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-test', action='store_true')
    mode.add_argument('--emit-patch', action='store_true')
    arguments = parser.parse_args()
    if arguments.emit_patch:
        print(interface_patch(), end='')
    else:
        print(json.dumps(self_test(), indent=2, allow_nan=False))
