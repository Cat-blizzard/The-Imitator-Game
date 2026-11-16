"""Fixed fruit-only actual link7 full-solid coverage, isolated from runtime.

Both fruit arms use the same 12.5mm grid and the existing six-LP/cell tight
circumsphere algorithm. Every native convex shape is covered in shape axes,
then rigidly transformed once into owning link axes. No current q, obstacle,
gap, seed or grasp selects/removes a sphere. Existing fixed-task-v2 payload
geometry, physical scene, masks, limits and state are preserved exactly.

Geometry arrays: finite native vertices[V,3] metres, integer triangles[F,3],
positive mesh scale[3], normalized shape-local XYZ/WXYZ pose[7]. Sphere arrays
are link-local [N,4] XYZ/radius. Live export reads actual PhysX collision shapes
only; offline export reads the complete saved actual initial-scene manifest.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import difflib
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np
from scipy.spatial import ConvexHull

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p/'infrastructure/development/tig-dual-arm').is_dir())
DEV = ROOT/'infrastructure/development/tig-dual-arm'
for directory in (DEV, ROOT/'infrastructure'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from mani_skill.examples.motionplanning.dual import curobo_payload_refinement as v1
from mani_skill.examples.motionplanning.dual import curobo_payload_refinement_v2 as v2

E = ROOT/'infrastructure/validation/dual-arm/curobo-complete-20261116'
P = E/'extensions-v2/curobo-fruit-s401-refined-dev01'
LINK = 'panda_link7'
FRUIT = v1.TASK_IDS[2]
PITCH_M = .0125
POLICY = 'fixed-task-v3'
RECIPE = 'fixed-fruit-both-link7-actual-convex-solid-tight12_5mm-v3'
SCHEMA = 'tig-actual-panda-link7-local-geometry-v1'
BASE_RUNNER_SHA256 = 'cc21cd18715c332465a088cf6877d13f539b49a62f5865c08b67907e278c5a59'


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _hash(value: Any) -> str:
    """Same canonical JSON+newline SHA as actual initial-scene native geometry."""
    return hashlib.sha256((json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False)+'\n').encode()).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _shape_record(index: int, record: dict[str, Any]) -> dict[str, Any]:
    _require(record.get('type') == 'PhysxCollisionShapeConvexMesh',
             'Actual complete link7 must contain only supported PhysX convex mesh solids')
    native = record.get('native_local_geometry')
    _require(isinstance(native, dict) and record.get('native_local_geometry_sha256') == _hash(native),
             'Actual native collision geometry digest differs')
    vertices = v1._array(native.get('vertices_shape_local_unscaled_m'), 3, 'actual native vertices')
    scale = np.asarray(native.get('native_scale_xyz'), dtype=np.float64)
    _require(scale.shape == (3,) and bool(np.isfinite(scale).all()) and bool(np.all(scale > 0)),
             'Positive finite native mesh scale[3] required')
    faces = np.asarray(native.get('triangles'))
    _require(faces.ndim == 2 and faces.shape[1] == 3 and len(faces) > 0
             and faces.dtype.kind in 'iu' and bool(np.all(faces >= 0)) and bool(np.all(faces < len(vertices))),
             'Actual triangles must be integer [F,3] valid native indices')
    scaled = np.ascontiguousarray(vertices*scale, dtype=np.float64)
    _require(native.get('scaled_vertices_sha256') == hashlib.sha256(scaled.tobytes()).hexdigest(),
             'Actual scaled native vertex digest differs')
    _require(len(scaled) >= 4 and np.linalg.matrix_rank(scaled-scaled.mean(0)) == 3,
             'Actual full-dimensional convex solid required')
    ConvexHull(scaled)
    pose = record.get('shape_local_pose_raw_xyz_wxyz')
    v1._pose_matrix(pose)
    groups = record.get('collision_groups_uint32')
    _require(isinstance(groups, list) and len(groups) == 4 and all(
        type(x) is int and 0 <= x <= 0xffffffff for x in groups), 'Actual four uint32 collision groups required')
    return {'shape_index': index, 'type': record['type'], 'native_local_geometry': deepcopy(native),
        'native_local_geometry_sha256': record['native_local_geometry_sha256'],
        'shape_local_pose_raw_xyz_wxyz': deepcopy(pose), 'collision_groups_uint32': deepcopy(groups)}


def _manifest(arm: int, shapes: list[dict[str, Any]], source: dict[str, Any]) -> dict[str, Any]:
    _require(type(arm) is int and arm in (0, 1), 'Canonical physical arm0/1 required')
    _require(isinstance(shapes, list) and len(shapes) > 0, 'Complete nonempty actual link7 shape inventory required')
    normalized = [_shape_record(i, row) for i, row in enumerate(shapes)]
    return {'schema': SCHEMA, 'task_id': FRUIT, 'physical_arm_index': arm, 'link_name': LINK,
        'native_shape_count': len(normalized), 'shapes': normalized,
        'geometry_sha256': _hash(normalized), 'source': deepcopy(source)}


def geometry_from_initial_scene(initial: dict[str, Any], arm_index: int) -> dict[str, Any]:
    """Extract exact actual collision solids; no URDF visual mesh is consulted."""
    _require(initial.get('schema') == 'tig-complete-actual-initial-scene-v1', 'Complete actual initial-scene schema required')
    scene = initial.get('scene')
    _require(isinstance(scene, dict) and initial.get('scene_sha256') == _hash(scene), 'Canonical initial-scene hash differs')
    coverage, proof = initial.get('coverage', {}), initial.get('read_only_proof', {})
    _require(coverage.get('all_covered') is True
        and coverage.get('native_body_and_link_inventories_exactly_matched') is True
        and coverage.get('all_collision_shapes_included_without_mask_filtering') is True,
        'Initial scene must include the complete actual native inventory without collision-mask filtering')
    _require(proof.get('state_and_elapsed_unchanged') is True
        and proof.get('full_scene_repeated_read_consistent') is True
        and proof.get('environment_steps_during_export') == 0
        and proof.get('elapsed_steps_before') == proof.get('elapsed_steps_after') == [0]
        and proof.get('state_digest_before') == proof.get('state_digest_after'), 'Actual initial-scene read-only proof missing')
    _require(type(arm_index) is int and arm_index in (0, 1), 'Canonical physical arm0/1 required')
    robots = scene.get('robots')
    _require(isinstance(robots, list) and len(robots) == 2
        and [r.get('arm_index') for r in robots] == [0, 1], 'Actual canonical physical robot inventory differs')
    link = robots[arm_index].get('links', {}).get(LINK)
    _require(isinstance(link, dict), 'Actual native link7 missing')
    return _manifest(arm_index, link.get('shapes'), {'kind': 'complete_actual_initial_scene',
        'scene_sha256': initial['scene_sha256'], 'environment_steps_during_export': 0,
        'physical_geometry_source': 'raw PhysX collision mesh arrays, mesh scale and shape-local pose',
        'all_actual_link_shapes_exported': True, 'physical_state_unchanged': True,
        'urdf_or_visual_mesh_used': False})


def export_link7_geometry(raw: Any, arm_index: int) -> dict[str, Any]:
    """Read the selected robot's actual PhysX shapes without stepping or setters."""
    from mani_skill.examples.motionplanning.dual.initial_scene_manifest import _native_geometry, _raw_pose
    from mani_skill.examples.motionplanning.dual.validate_curobo_payload import state_digest
    _require(type(arm_index) is int and arm_index in (0, 1), 'Canonical physical arm0/1 required')
    _require(len(raw.agent.agents) == 2 and raw.num_envs == 1 and not raw.gpu_sim_enabled,
             'One actual CPU TIG physical Panda pair required')
    before = state_digest(raw)
    elapsed_before = np.asarray(raw.elapsed_steps.cpu()).copy()
    robot = raw.agent.agents[arm_index].robot
    links = [link for link in robot.get_links() if link.get_name() == LINK]
    _require(len(links) == 1 and len(links[0]._objs) == 1, 'One unambiguous actual link7 body required')
    shapes = list(links[0]._objs[0].get_collision_shapes())
    records = []
    for shape in shapes:
        native = _native_geometry(shape)
        records.append({'type': type(shape).__name__, 'native_local_geometry': native,
            'native_local_geometry_sha256': _hash(native),
            'shape_local_pose_raw_xyz_wxyz': _raw_pose(shape.get_local_pose()),
            'collision_groups_uint32': [int(x) for x in shape.get_collision_groups()]})
    result = _manifest(arm_index, records, {'kind': 'live_actual_PhysX',
        'physical_geometry_source': 'raw PhysX collision mesh arrays, mesh scale and shape-local pose',
        'all_actual_link_shapes_exported': True, 'environment_steps_during_export': 0,
        'physical_state_unchanged': True, 'urdf_or_visual_mesh_used': False})
    _require(state_digest(raw) == before and np.array_equal(np.asarray(raw.elapsed_steps.cpu()), elapsed_before),
             'Reading actual link7 changed physical state or elapsed steps')
    return result


def _validated_geometry(value: Any, arm: int) -> list[dict[str, Any]]:
    _require(isinstance(value, dict) and value.get('schema') == SCHEMA
        and value.get('task_id') == FRUIT and value.get('link_name') == LINK
        and type(value.get('physical_arm_index')) is int and value['physical_arm_index'] == arm,
        'Explicit actual native link7 geometry for the selected fruit arm required')
    source = value.get('source', {})
    _require(source.get('kind') in ('complete_actual_initial_scene', 'live_actual_PhysX')
        and source.get('all_actual_link_shapes_exported') is True and source.get('physical_state_unchanged') is True
        and source.get('urdf_or_visual_mesh_used') is False and source.get('environment_steps_during_export') == 0,
        'Actual full collision-shape source and no-step read-only receipt required')
    shapes = value.get('shapes')
    _require(isinstance(shapes, list) and len(shapes) > 0 and value.get('native_shape_count') == len(shapes),
             'Actual native link7 shape count differs')
    checked = [_shape_record(i, shape) for i, shape in enumerate(shapes)]
    _require(checked == shapes and value.get('geometry_sha256') == _hash(checked),
             'Actual link7 local geometry or order/hash differs')
    return checked


@lru_cache(maxsize=32)
def _cached_cover(key: str) -> tuple[tuple[tuple[float, ...], ...], str]:
    """Immutable finite geometry+fixed-policy cache; no arm/q/scene enters."""
    data = json.loads(key)
    _require(data.get('recipe') == RECIPE and data.get('pitch_m') == PITCH_M, 'Exact fixed link7 recipe required')
    rows, proofs = [], []
    for shape in data['shapes']:
        native = shape['native_local_geometry']
        vertices = np.asarray(native['vertices_shape_local_unscaled_m'], dtype=np.float64)*np.asarray(native['native_scale_xyz'])
        spheres, proof = v1.tight_convex_spheres(vertices, PITCH_M, max_depth=2)
        spheres[:, :3] = v1._transform(spheres[:, :3], v1._pose_matrix(shape['shape_local_pose_raw_xyz_wxyz']))
        proof.update(link=LINK, shape_index=shape['shape_index'], type=shape['type'],
            native_local_geometry_sha256=shape['native_local_geometry_sha256'],
            shape_local_pose_raw_xyz_wxyz=deepcopy(shape['shape_local_pose_raw_xyz_wxyz']),
            source_frame='physical_shape_local', output_frame='physical_link_local',
            policy=RECIPE, task_or_seed_or_pose_or_obstacle_dependent=False,
            physical_collision_geometry_unchanged=True, collision_masks_unchanged=True)
        rows.extend(spheres.tolist());proofs.append(proof)
    array = v1._array(rows, 4, 'complete link-local sphere cover')
    _require(bool(np.all(array[:, 3] > 0)), 'Positive complete-cover sphere radii required')
    return tuple(tuple(float(x) for x in row) for row in array), json.dumps(proofs, sort_keys=True, allow_nan=False)


def _cover(shapes: list[dict[str, Any]]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    key = json.dumps({'recipe': RECIPE, 'pitch_m': PITCH_M, 'shapes': shapes}, sort_keys=True, allow_nan=False)
    rows, proofs = _cached_cover(key)
    return np.asarray(rows, dtype=np.float64).copy(), json.loads(proofs)


def _nonlink7_view(snapshot: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(snapshot)
    result['robot_cfg']['kinematics']['collision_spheres'].pop(LINK)
    coverage = result['coverage']
    coverage['active_robot_shapes'] = [row for row in coverage['active_robot_shapes'] if row['link'] != LINK]
    coverage.pop('robot_collision_sphere_count')
    coverage.pop('link7_sphere_refinement', None)
    return result


def refine_task_snapshot_v3(snapshot: dict[str, Any], geometry_manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """Keep v2 payloads; replace only both fruit arms' actual link7 full cover."""
    task, arm = v1._validate_snapshot(snapshot)
    base = v2.refine_task_snapshot_v2(snapshot)
    if task != FRUIT:
        return base
    shapes = _validated_geometry(geometry_manifest, arm)
    old = [row for row in base['coverage']['active_robot_shapes'] if row['link'] == LINK]
    _require(len(old) == len(shapes) and [row['shape_index'] for row in old] == list(range(len(shapes))),
             'Actual shape inventory must equal the complete original link7 export')
    spheres, proofs = _cover(shapes)
    result = deepcopy(base)
    kin, coverage = result['robot_cfg']['kinematics'], result['coverage']
    old_count = len(kin['collision_spheres'][LINK])
    kin['collision_spheres'][LINK] = [{'center': row[:3].tolist(), 'radius': float(row[3])} for row in spheres]
    coverage['active_robot_shapes'] = [row for row in coverage['active_robot_shapes'] if row['link'] != LINK]+deepcopy(proofs)
    coverage['robot_collision_sphere_count'] = sum(len(rows) for rows in kin['collision_spheres'].values())
    previous = coverage.get('link7_sphere_refinement')
    original_count = previous['original_sphere_count'] if previous else old_count
    coverage['link7_sphere_refinement'] = {'recipe': RECIPE, 'sphere_pitch_m': PITCH_M,
        'physical_arm_index': arm, 'original_sphere_count': original_count,
        'refined_sphere_count': len(spheres), 'native_shape_count': len(shapes),
        'actual_geometry_manifest': deepcopy(geometry_manifest), 'shape_proofs': deepcopy(proofs),
        'all_actual_link7_solids_covered': True, 'physical_geometry_unchanged': True,
        'world_and_nonlink7_geometry_exactly_preserved': True, 'collision_permissions_and_limits_unchanged': True,
        'task_or_seed_or_pose_or_obstacle_dependent_sphere_selection': False}
    _require(_nonlink7_view(result) == _nonlink7_view(base), 'A non-link7 field changed')
    if previous is not None:
        _require(previous == coverage['link7_sphere_refinement'] and result == base,
                 'Stale or foreign pre-existing link7 recipe/geometry')
    v1._validate_snapshot(result)
    json.dumps(result, allow_nan=False)
    return result


def proposed_runner_source(source: str) -> str:
    """Explicit new policy and live actual-geometry export; no runtime writes."""
    _require(hashlib.sha256(source.encode()).hexdigest() == BASE_RUNNER_SHA256, 'Frozen fruit-v2 runner base SHA differs')
    edits = (
        ("PAYLOAD_REFINEMENTS = ('baseline', 'fixed-task-v1', 'fixed-task-v2')",
         "PAYLOAD_REFINEMENTS = ('baseline', 'fixed-task-v1', 'fixed-task-v2', 'fixed-task-v3')"),
        ("def configured_payload_snapshot(snapshot: dict[str,Any], policy: str) -> dict[str,Any]:",
         "def configured_payload_snapshot(snapshot: dict[str,Any], policy: str, geometry_manifest: dict[str,Any] | None = None) -> dict[str,Any]:"),
        ("        return refine_task_snapshot_v2(snapshot)\n    raise ValueError('Unknown payload refinement policy')\n",
         "        return refine_task_snapshot_v2(snapshot)\n    if policy == 'fixed-task-v3':\n        from .curobo_fruit_link7_v3 import refine_task_snapshot_v3\n        return refine_task_snapshot_v3(snapshot,geometry_manifest)\n    raise ValueError('Unknown payload refinement policy')\n"),
        ("            return configured_payload_snapshot(snapshot,payload_refinement)\n",
         "            geometry_manifest = None\n            if payload_refinement == 'fixed-task-v3' and args.task == TASKS[2]:\n                from .curobo_fruit_link7_v3 import export_link7_geometry\n                geometry_manifest = export_link7_geometry(raw,index)\n            return configured_payload_snapshot(snapshot,payload_refinement,geometry_manifest)\n"),
        ("    assert configured_payload_snapshot(apple,'fixed-task-v2') == apple\n",
         "    assert configured_payload_snapshot(apple,'fixed-task-v2') == apple\n    assert configured_payload_snapshot(apple,'fixed-task-v3') == apple\n"),
        ("        'payload_refinement_scope': 'cuRobo native payload representation only; physical model, control and acceptance thresholds unchanged',\n",
         "        'payload_refinement_scope': (\n            'cuRobo native payload and both fruit-arm panda_link7 conservative sphere geometry only; physical model, control and acceptance thresholds unchanged'\n            if payload_refinement == 'fixed-task-v3' else\n            'cuRobo native payload representation only; physical model, control and acceptance thresholds unchanged'),\n"),
        ("            'cuRobo native payload spheres and reserved capacity only; no physical state, collision permissions or thresholds changed')\n",
         "            'cuRobo native payload spheres/reserved capacity and both fruit-arm panda_link7 conservative spheres only; no physical state, collision permissions or thresholds changed'\n            if payload_refinement == 'fixed-task-v3' else\n            'cuRobo native payload spheres and reserved capacity only; no physical state, collision permissions or thresholds changed')\n"))
    result = source
    for before, after in edits:
        _require(result.count(before) == 1, 'Frozen minimal policy/export anchor differs')
        result = result.replace(before, after, 1)
    ast.parse(result)
    return result


def interface_patch() -> str:
    source = (P/'source/validate_curobo_complete.py').read_text()
    name = 'infrastructure/development/tig-dual-arm/mani_skill/examples/motionplanning/dual/validate_curobo_complete.py'
    return ''.join(difflib.unified_diff(source.splitlines(True), proposed_runner_source(source).splitlines(True),
                                      fromfile='a/'+name, tofile='b/'+name))


def self_test() -> dict[str, Any]:
    """Synthetic full-solid/rigid tests plus six actual snapshots and negatives."""
    started = time.perf_counter();rejected: list[str] = [];actual = []
    patch = interface_patch();assert POLICY in patch
    def reject(label: str, call: Callable[[], Any]) -> None:
        try:
            call()
        except (ValueError, TypeError, KeyError):
            rejected.append(label)
        else:
            raise AssertionError('Expected rejection: '+label)
    initial = json.loads((P/'initial-scene.json').read_text())
    geometries = [geometry_from_initial_scene(initial, i) for i in range(2)]
    _cached_cover.cache_clear()
    for arm in (0, 1):
        path = P/f'carry_lower-arm-{arm}-snapshot.json'
        snapshot = json.loads(path.read_text());unchanged = deepcopy(snapshot)
        result = refine_task_snapshot_v3(snapshot, geometries[arm])
        assert snapshot == unchanged and result == refine_task_snapshot_v3(result, geometries[arm])
        assert _nonlink7_view(result) == _nonlink7_view(v2.refine_task_snapshot_v2(snapshot))
        info = _cached_cover.cache_info();again = refine_task_snapshot_v3(snapshot, geometries[arm])
        assert again == result and _cached_cover.cache_info().hits == info.hits+1
        for profile in ('approach', 'transport'):
            for operation in ('init_session', 'update_session'):
                request = v1._request(result, operation, profile)
                v1.validate_task_request(request, FRUIT)
                v2.legacy_validate_request(v1.validation_view(request, FRUIT))
        actual.append({'arm': arm, 'input': str(path), 'input_sha256': _sha(path),
            'original_link7_spheres': len(snapshot['robot_cfg']['kinematics']['collision_spheres'][LINK]),
            'refined_link7_spheres': len(result['robot_cfg']['kinematics']['collision_spheres'][LINK]),
            'complete_actual_shapes': len(geometries[arm]['shapes']), 'nonlink7_v2_identity_exact': True,
            'Task_and_legacy_CPU_validators': 'passed'})
    for task in v1.TASK_IDS[:2]:
        for arm in (0, 1):
            snapshot = json.loads((E/f'task-source-snapshots-dev02/{task}-arm-{arm}.json').read_text())
            assert refine_task_snapshot_v3(snapshot) == v2.refine_task_snapshot_v2(snapshot)
    # Every cube point is covered before and after an arbitrary rigid transform;
    # this numerical test supplements the complete LP/partition proof.
    vertices = np.array([[x,y,z] for x in (-.013,.017) for y in (-.011,.019) for z in (-.007,.023)])
    spheres, proof = v1.tight_convex_spheres(vertices, PITCH_M, max_depth=2)
    grid = np.stack(np.meshgrid(*[np.linspace(vertices[:,i].min(),vertices[:,i].max(),13) for i in range(3)],indexing='ij'),axis=-1).reshape(-1,3)
    transform = v1._pose_matrix([.1,-.2,.3,np.sqrt(.5),0.,0.,np.sqrt(.5)])
    transformed = np.c_[v1._transform(spheres[:,:3],transform),spheres[:,3]]
    for points, cover in ((grid,spheres),(v1._transform(grid,transform),transformed)):
        gaps = np.linalg.norm(points[:,None]-cover[None,:,:3],axis=2)-cover[None,:,3]
        assert gaps.min(axis=1).max() <= 1e-8 and proof['complete_solid_coverage']
    base = json.loads((P/'carry_lower-arm-1-snapshot.json').read_text())
    geometry = geometries[1]
    reject('missing_geometry',lambda:refine_task_snapshot_v3(base))
    reject('wrong_arm_geometry',lambda:refine_task_snapshot_v3(base,geometries[0]))
    for label, mutate in (
        ('manifest_hash',lambda m:m.update(geometry_sha256='0'*64)),
        ('wrong_link',lambda m:m.update(link_name='panda_hand')),
        ('missing_shape',lambda m:m.update(shapes=[])),
        ('shape_count',lambda m:m.update(native_shape_count=2)),
        ('visual_mesh',lambda m:m['source'].update(urdf_or_visual_mesh_used=True)),
        ('export_stepped',lambda m:m['source'].update(environment_steps_during_export=1)),
        ('native_hash',lambda m:m['shapes'][0].update(native_local_geometry_sha256='0'*64)),
        ('local_pose',lambda m:m['shapes'][0].update(shape_local_pose_raw_xyz_wxyz=[0.,0.,0.,2.,0.,0.,0.])),
        ('unsupported_type',lambda m:m['shapes'][0].update(type='VisualMesh'))):
        bad = deepcopy(geometry);mutate(bad)
        reject(label,lambda bad=bad:refine_task_snapshot_v3(base,bad))
    for label, mutate in (
        ('nonfinite_vertices',lambda n:n['vertices_shape_local_unscaled_m'][0].__setitem__(0,float('nan'))),
        ('zero_scale',lambda n:n.update(native_scale_xyz=[0.,1.,1.])),
        ('negative_scale',lambda n:n.update(native_scale_xyz=[-1.,1.,1.])),
        ('bad_triangles',lambda n:n.update(triangles=[[0,1,999999]])),
        ('fractional_triangles',lambda n:n.update(triangles=[[0.,1.,2.]])),
        ('scaled_vertex_digest',lambda n:n.update(scaled_vertices_sha256='0'*64))):
        bad = deepcopy(geometry);native = bad['shapes'][0]['native_local_geometry'];mutate(native)
        if label != 'nonfinite_vertices':bad['shapes'][0]['native_local_geometry_sha256'] = _hash(native);bad['geometry_sha256'] = _hash(bad['shapes'])
        reject(label,lambda bad=bad:refine_task_snapshot_v3(base,bad))
    bad = deepcopy(initial);bad['read_only_proof']['environment_steps_during_export'] = 1
    reject('initial_scene_stepped',lambda:geometry_from_initial_scene(bad,0))
    bad = deepcopy(initial);bad['scene_sha256'] = '0'*64
    reject('initial_scene_hash',lambda:geometry_from_initial_scene(bad,0))
    refined = refine_task_snapshot_v3(base,geometry);refined['coverage']['link7_sphere_refinement']['recipe']='foreign'
    reject('stale_recipe',lambda:refine_task_snapshot_v3(refined,geometry))
    return {'status':'passed','fixed_pitch_m':PITCH_M,'recipe':RECIPE,'actual_snapshots':actual,
        'four_nonfruit_arms_exactly_v2':True,'synthetic_solid_grid_points':len(grid),'rigid_transform_checks':2,
        'rejected_inputs':rejected,'cache':_cached_cover.cache_info()._asdict(),
        'world_state_payload_limits_masks_unchanged':True,'environment_steps':0,'native_planning':'NOT_RUN',
        'helper_sha256':_sha(Path(__file__)),'wall_s':time.perf_counter()-started}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--emit-patch',action='store_true')
    parser.add_argument('--output-json',type=Path)
    args = parser.parse_args()
    if args.emit_patch:
        _require(not args.self_test and args.output_json is None,'Patch output is independent of self-test JSON')
        print(interface_patch(),end='');return 0
    if not args.self_test:parser.error('Provide --self-test or --emit-patch')
    result = self_test();text = json.dumps(result,indent=2,allow_nan=False)+'\n'
    if args.output_json is not None:
        with args.output_json.open('x') as handle:handle.write(text)
    print(text,end='');return 0


if __name__ == '__main__':
    raise SystemExit(main())
