"""CPU preflight for the exact visual-mesh inputs consumed by the frozen solver.

This does not repair a mesh, substitute an assumed COM, rescale geometry, or
claim physical graspability. Invalid numerical inputs are attributable asset
failures; unreadable files, changed hashes and unexpected exceptions propagate.
"""
from pathlib import Path
import warnings

import numpy as np

from .common import binding, read, resolve

VERSION = 'cudex_finite_visual_input_v4'


def inspect_mesh(mesh, table_z):
    vertices = np.asarray(mesh.vertices, dtype=float)
    reasons = []
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices):
        reasons.append('empty_or_invalid_mesh_vertices')
    elif not np.isfinite(vertices).all():
        reasons.append('nonfinite_mesh_vertices')
    if not np.isfinite(table_z):
        reasons.append('nonfinite_native_table_support')
    # Do not ask trimesh for derived quantities on nonfinite/empty geometry.
    if reasons:
        return {'status': 'asset_invalid', 'reasons': reasons,
                'vertices': len(vertices), 'center_mass_m': None, 'signed_volume_m3': None}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        com = np.asarray(mesh.center_mass, dtype=float)
        volume = float(mesh.volume)
        bounds = np.asarray(mesh.bounds, dtype=float)
    if com.shape != (3,) or not np.isfinite(com).all():
        reasons.append('nonfinite_mesh_center_mass')
    if not np.isfinite(volume):
        reasons.append('nonfinite_mesh_volume')
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        reasons.append('nonfinite_mesh_bounds')
    return {'status': 'asset_invalid' if reasons else 'finite_input', 'reasons': reasons,
            'vertices': len(vertices), 'faces': len(mesh.faces),
            'is_watertight': bool(mesh.is_watertight),
            'signed_volume_m3': volume if np.isfinite(volume) else None,
            'center_mass_m': com.tolist() if com.shape == (3,) and np.isfinite(com).all() else None,
            'bounds_m': bounds.tolist() if bounds.shape == (2, 3) and np.isfinite(bounds).all() else None,
            'native_table_z_m': float(table_z),
            'finite_does_not_certify_physical_mass_or_grasp': True}


def inspect_job(job_path):
    import trimesh
    job_path = Path(job_path).resolve()
    job = read(job_path)
    descriptor = read(resolve(job['asset']))
    visual = resolve(descriptor['visual'])
    # Identical loader and frame to cudex_constraint_v3.worker.
    mesh = trimesh.load(visual, force='scene', process=False).to_geometry()
    measured = inspect_mesh(mesh, float(descriptor['collision_bounds_m'][0][2]))
    return {'schema': VERSION, 'implementation': binding(__file__), 'job': binding(job_path),
            'asset': job['asset'], 'visual': descriptor['visual'], **measured,
            'geometry_modified': False, 'COM_replaced': False, 'GPU_jobs': 0, 'API_calls': 0}


def inspect_unsorted(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        q = data['q_unsorted']
        if q.shape != (256, 9):
            raise ValueError('unexpected_solver_state_shape_requires_diagnosis')
        finite = np.isfinite(q).all(axis=1)
        return {'state': binding(path), 'shape': list(q.shape),
                'finite_rows': int(finite.sum()), 'nonfinite_rows': int((~finite).sum())}
