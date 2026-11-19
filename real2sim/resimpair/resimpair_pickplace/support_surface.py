"""Native triangle orientation must support a load from ABOVE.

A two-sided ray hit alone is insufficient for PhysX single-sided triangle
mesh contacts. Reject inconsistent surfaces; never silently reverse a mesh.
"""
import argparse
from pathlib import Path
import numpy as np
import trimesh
from trimesh.ray.ray_triangle import RayMeshIntersector
from resimpair_grasp.common import read, resolve, binding, write


def check_support(mesh, cavity):
    center = np.asarray(cavity['center_xy_m'])
    radius = cavity['flat_floor_radius_m']
    angles = np.linspace(0, 2*np.pi, 32, endpoint=False)
    xy = np.vstack([center, *[center + f*radius*np.column_stack([np.cos(angles), np.sin(angles)])
                             for f in [.25, .6, .9]]])
    origins = np.column_stack([xy, np.full(len(xy), mesh.bounds[1, 2]+.01)])
    hits, rayids, triangles = RayMeshIntersector(mesh).intersects_location(
        origins, np.tile([0, 0, -1.], (len(xy), 1)), multiple_hits=False)
    normals = mesh.face_normals[triangles]
    near_floor = abs(hits[:, 2]-cavity['floor_z_m']) <= cavity.get('floor_tolerance_m', .003)
    up = normals[:, 2] >= .85
    return {'passed': bool(len(set(rayids)) == len(xy) and near_floor.all() and up.all()),
            'version': 'native_oriented_support_preflight_v1', 'sample_count': len(xy),
            'hit_count': len(hits), 'upward_hits': int(up.sum()),
            'near_declared_floor_hits': int(near_floor.sum()),
            'minimum_normal_z': float(normals[:, 2].min()) if len(hits) else None,
            'reason': None if len(hits) == len(xy) and near_floor.all() and up.all()
                      else 'declared_floor_not_a_consistent_upward_collision_surface',
            'scope': 'sampled native-facing support; not a dynamic stability certificate'}


def catalog_check(catalog, output):
    data = read(catalog)
    entries = data.get('containers', data.get('assets', []))
    rows = []
    for entry in entries:
        # Candidate catalog stores descriptor bindings as its assets.
        b = entry.get('descriptor', entry)
        d = read(resolve(b))
        mesh = trimesh.load(resolve(d['collision']), force='mesh', process=False)
        rows.append({'descriptor': b, 'source_id': d['source_id'], **check_support(mesh, d['cavity'])})
    write(output, {'input': binding(catalog), 'version': 'native_oriented_support_preflight_v1',
                   'rows': rows, 'tested': len(rows), 'passed': sum(r['passed'] for r in rows),
                   'api_calls': 0, 'physical_executions': 0})
    return {'tested': len(rows), 'passed': sum(r['passed'] for r in rows)}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(catalog_check(a.catalog,a.output))
