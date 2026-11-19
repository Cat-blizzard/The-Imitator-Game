"""Role-specific stationary support meshes at a declared 24 cm scene scale.

Pick assets and grasp caches are unchanged. These are support_surface assets,
not containers; all source candidates are retained as successes or failures.
CPU-only native triangle cooking and sampled upward-surface checks precede any
runtime use. No dynamic trajectory, GPU operation or model API is involved.
"""
import argparse
from pathlib import Path
import time

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from resimpair_grasp.common import read, write, binding, resolve, digest, ROOT
from resimpair_tasks.place_surface_candidates_v1 import propose_patch
from resimpair_pickplace.support_surface import check_support


def prepare(row, folder, maximum_size_m):
    import sapien
    source = read(resolve(row['asset']))
    mesh = trimesh.load(resolve(source['visual']), force='scene', process=False).to_geometry()
    original_bounds = mesh.bounds.copy()
    scale = maximum_size_m / float(np.max(mesh.extents))
    transform = np.diag([scale, scale, scale, 1.])
    mesh.apply_scale(scale)
    translation = np.array([-mesh.bounds[:, 0].mean(), -mesh.bounds[:, 1].mean(), -mesh.bounds[0, 2]])
    mesh.apply_translation(translation); transform[:3, 3] = translation
    folder.mkdir(parents=True, exist_ok=True)
    mesh.export(folder/'visual.glb'); mesh.export(folder/'collision.obj')
    shape = sapien.physx.PhysxCollisionShapeTriangleMesh(
        filename=str(folder/'collision.obj'), scale=[1, 1, 1],
        material=sapien.physx.PhysxMaterial(.6, .6, 0.))
    native = trimesh.Trimesh(vertices=np.asarray(shape.vertices), faces=np.asarray(shape.triangles), process=False)
    np.savez_compressed(folder/'native_triangle_mesh.npz', vertices=native.vertices, faces=native.faces)
    distance = max(float(cKDTree(native.vertices).query(mesh.vertices)[0].max()),
                   float(cKDTree(mesh.vertices).query(native.vertices)[0].max()))
    proposed = propose_patch(native)
    checks = {'native_vertex_hausdorff_m': distance,
              'native_vertex_tolerance_m': .00001,
              'input_faces': len(mesh.faces), 'native_faces': len(native.faces),
              'triangle_topology_full_equivalence': 'not_certified_by_vertex_distance',
              'native_triangle_mesh': binding(folder/'native_triangle_mesh.npz'),
              'support_query': proposed}
    write(folder/'geometry_checks.json', checks)
    if distance > .00001:
        return {'source_id': source['source_id'], 'status': 'rejected_native_vertex_geometry',
                'geometry_checks': binding(folder/'geometry_checks.json')}
    if proposed['status'] != 'upward_flat_patch_candidate':
        return {'source_id': source['source_id'], 'status': 'rejected_no_upward_flat_patch',
                'geometry_checks': binding(folder/'geometry_checks.json')}
    patch = proposed['patch']
    # Independent visual support also prevents a collision-only flat patch
    # from being presented as a usable rendered surface.
    visual_check = check_support(mesh, patch)
    write(folder/'visual_support_check.json', visual_check)
    if not visual_check['passed']:
        return {'source_id': source['source_id'], 'status': 'rejected_visual_native_support_disagreement',
                'geometry_checks': binding(folder/'geometry_checks.json'),
                'visual_check': binding(folder/'visual_support_check.json')}
    usable = max(0., patch['flat_floor_radius_m']-.008)
    identity = digest({'source': row['asset'], 'role': 'support_surface', 'maximum_size_m': maximum_size_m})[:20]
    descriptor = {'schema': 'stationary_support_surface_v1', 'id': 'surface_'+identity,
        'source_id': source['source_id'], 'name': source.get('name'), 'partition': source['partition'],
        'parent_pick_asset': row['asset'], 'visual': binding(folder/'visual.glb'),
        'collision': binding(folder/'collision.obj'), 'native_policy': 'static_triangle_mesh_only',
        'bounds_m': mesh.bounds.tolist(), 'dimensions_m': mesh.extents.tolist(),
        'source_to_asset': (transform@np.asarray(source['source_to_asset'])).tolist(),
        'pick_asset_to_support_asset': transform.tolist(),
        'role': 'support_surface', 'task_relation': 'on',
        'container': False, 'enclosing_walls_claimed': False,
        'support_region': {'center_xy_m': patch['center_xy_m'], 'surface_z_m': patch['floor_z_m'],
            'flat_radius_m': patch['flat_floor_radius_m'], 'normal_world_local': [0., 0., 1.],
            'height_tolerance_m': .003, 'minimum_surface_normal_z': .85},
        'payload_fit': {'maximum_footprint_radius_m': usable,
            'maximum_inscribed_square_side_m': float(np.sqrt(2)*usable), 'edge_margin_m': .008,
            'criterion': 'all payload XY vertices relative to support center must lie within maximum radius'},
        'clearance': {'own_mesh_vertical_prism_above_patch': 'sampled_clear',
            'verified_height_m': .25, 'gripper_and_robot_clearance': 'unknown_runtime_required',
            'other_scene_obstacle_clearance': 'unknown_runtime_required',
            'basis': 'first downward native and visual hits at97 rays lie on declared upward patch'},
        'metric_size': {'mode': 'role_specific_synthetic_scale', 'maximum_extent_m': maximum_size_m,
            'scale_from_pick_asset': scale, 'pick_bounds_unchanged_m': original_bounds.tolist(),
            'real_world_scale_recovered': False},
        'semantic_status': 'source_metadata_unverified; geometry does not certify object identity',
        'physical_status': 'unmeasured', 'dynamic_stability': 'unknown',
        'native_geometry_checks': binding(folder/'geometry_checks.json'),
        'visual_support_check': binding(folder/'visual_support_check.json'),
        'support_rule': binding(ROOT/'resimpair_pickplace/support_surface.py'),
        'runtime_adapter_status': 'requires_stationary_support_surface_target_adapter'}
    path = folder/'descriptor.json'; write(path, descriptor)
    return {'source_id': source['source_id'], 'status': 'support_surface_candidate',
            'asset': binding(path), 'maximum_payload_footprint_radius_m': usable,
            'support_height_m': patch['floor_z_m']}


def run(candidates, output, maximum_size_m=.24):
    if not .20 <= maximum_size_m <= .30: raise ValueError('Explicit20–30cm support role scale required')
    output = Path(output).resolve()
    if output.exists(): raise ValueError('Keep previous support variant evidence')
    inputs = read(candidates)['entries']
    if not 1 <= len(inputs) <= 128: raise ValueError('Bounded static-role input required')
    output.mkdir(parents=True)
    write(output/'protocol.json', {'schema': 'stationary_surface_role_scale_protocol_v1',
        'input': binding(candidates), 'sources': len(inputs), 'maximum_size_m': maximum_size_m,
        'all_input_sources_attempted': True, 'original_pick_assets_unchanged': True,
        'code': [binding(__file__), binding(ROOT/'resimpair_tasks/place_surface_candidates_v1.py'),
                 binding(ROOT/'resimpair_pickplace/support_surface.py')],
        'api_calls': 0, 'gpu_jobs': 0, 'dynamic_simulation_steps': 0})
    start = time.monotonic(); rows = []
    for i, row in enumerate(inputs):
        folder = output/'assets'/f'{i:03d}'; tick = time.monotonic()
        write(folder/'reservation.json', {'input': row, 'started_unix': time.time()})
        try: result = prepare(row, folder, maximum_size_m)
        except Exception as error:
            result = {'source_id': row['source_id'], 'status': 'unknown_preparation_error', 'error': repr(error)}
        result['elapsed_seconds'] = time.monotonic()-tick
        write(folder/'terminal.json', result); rows.append(result)
    assets = [r['asset'] for r in rows if r['status']=='support_surface_candidate']
    write(output/'catalog.json', {'schema': 'stationary_support_surface_catalog_v1', 'assets': assets,
        'protocol': binding(output/'protocol.json'), 'relation': 'on', 'not_container_cavities': True,
        'physical_status': 'unmeasured', 'runtime_adapter_implemented': False})
    summary = {'attempted': len(rows), 'support_surface_candidates': len(assets),
        'unknown': sum(r['status'].startswith('unknown') for r in rows),
        'rejected': sum(r['status'].startswith('rejected') for r in rows),
        'rows': rows, 'wall_seconds': time.monotonic()-start,
        'api_calls': 0, 'gpu_jobs': 0, 'physical_executions': 0}
    write(output/'SUMMARY.json', summary)
    return {k:v for k,v in summary.items() if k!='rows'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--maximum-size-m', type=float, default=.24)
    a = p.parse_args(); print(run(a.candidates, a.output, a.maximum_size_m))
