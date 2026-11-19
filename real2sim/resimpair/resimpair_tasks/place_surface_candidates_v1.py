"""Find upward, near-flat support patches on native rigid meshes, CPU only.

This proposes stationary place targets independently of container semantics.
Each retained patch is checked by the existing oriented-support verifier. It
does not certify payload stability, stationarity or a manipulation trajectory.
"""
import argparse
from pathlib import Path
import time

import numpy as np
import trimesh

from resimpair_tabletop import geometry as _geometry
from resimpair_grasp.common import ROOT, read, resolve, write, binding
from resimpair_pickplace.support_surface import check_support


def propose_patch(mesh):
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    lower, upper = mesh.bounds; extent = upper-lower; center = (lower+upper)/2
    ray = RayMeshIntersector(mesh); angles = np.linspace(0, 2*np.pi, 32, endpoint=False)
    unit = np.column_stack([np.cos(angles), np.sin(angles)])
    attempts = []
    for dx, dy in [(0., 0.), (-.15, 0.), (.15, 0.), (0., -.15), (0., .15)]:
        xy = center[:2]+extent[:2]*[dx, dy]
        hit, _, _ = ray.intersects_location([[*xy, upper[2]+.01]], [[0., 0., -1.]], multiple_hits=False)
        if len(hit) != 1: continue
        z = float(hit[0, 2])
        for radius in [.06, .05, .04, .03, .02]:
            if np.any(xy-radius < lower[:2]) or np.any(xy+radius > upper[:2]): continue
            points = np.vstack([xy, *[xy+f*radius*unit for f in [.25, .6, .9]]])
            origins = np.column_stack([points, np.full(len(points), upper[2]+.01)])
            hits, ids, faces = ray.intersects_location(origins, np.tile([0., 0., -1.], (len(points), 1)), multiple_hits=False)
            fast_pass = bool(len(set(ids)) == len(points) and np.all(np.abs(hits[:, 2]-z)<=.003)
                             and np.all(mesh.face_normals[faces, 2]>=.85))
            attempts.append({'center_xy_m': xy.tolist(), 'radius_m': radius, 'fast_candidate': fast_pass})
            if not fast_pass: continue
            patch = {'center_xy_m': xy.tolist(), 'floor_z_m': z, 'flat_floor_radius_m': radius,
                     'floor_tolerance_m': .003}
            evidence = check_support(mesh, patch)
            if evidence['passed']:
                return {'status': 'upward_flat_patch_candidate', 'patch': patch,
                    'independent_support_check': evidence, 'attempts': attempts,
                    'payload_footprint_radius_limit_m': max(0., radius-.008),
                    'requires_stationary_support_role': True, 'physical_status': 'unmeasured',
                    'runtime_adapter_status': 'requires_explicit_stationary_mesh_support_adapter'}
    return {'status': 'no_patch_in_bounded_search', 'attempts': attempts,
            'physical_status': 'unmeasured', 'not_a_global_support_impossibility_proof': True}


def run(index, output):
    output = Path(output).resolve()
    if output.exists(): raise ValueError('Preserve earlier support-role evidence')
    output.mkdir(parents=True)
    entries = read(index)['entries']; rows = []; start = time.monotonic()
    rule = binding(ROOT/'resimpair_pickplace/support_surface.py')
    code = binding(__file__)
    for number, entry in enumerate(entries):
        asset = entry['asset']; d = read(resolve(asset)); tick = time.monotonic()
        if d['qualification']['geometry'] != 'pass':
            result = {'status': 'native_geometry_not_qualified'}
        else:
            try:
                mesh = trimesh.util.concatenate([trimesh.load(resolve(b), force='mesh', process=False) for b in d['collisions']])
                result = propose_patch(mesh)
            except Exception as error:
                result = {'status': 'unknown_support_query_error', 'error': repr(error)}
        row = {'source_id': d['source_id'], 'name': d.get('name'), 'asset': asset,
               'dimensions_m': d['dimensions_m'], 'semantic_status': d.get('semantic_status', 'unknown'),
               'support_rule': rule, 'seconds': time.monotonic()-tick, **result}
        write(output/'sources'/f'{number:05d}.json', row); rows.append(row)
    for b in [rule, code]: resolve(b)
    good = [r for r in rows if r['status']=='upward_flat_patch_candidate']
    write(output/'support_surface_candidates.json', {'schema': 'native_stationary_surface_candidates_v1',
        'input': binding(index), 'generator': code, 'independent_rule': rule, 'entries': good,
        'qualified_geometry_only': True, 'physical_success_claimed': False})
    summary = {'tested': len(rows), 'candidate_support_surfaces': len(good),
        'unknown': sum(r['status'].startswith('unknown') for r in rows),
        'wall_seconds': time.monotonic()-start, 'api_calls': 0, 'gpu_jobs': 0,
        'physical_executions': 0, 'new_task_mechanisms': 0,
        'rows': rows, 'missing_runtime_work': 'stationary mesh target adapter, payload footprint/path/stability check'}
    write(output/'SUMMARY.json', summary)
    return {k:v for k,v in summary.items() if k!='rows'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); print(run(a.index, a.output))
