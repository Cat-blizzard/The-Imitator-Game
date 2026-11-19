"""Bind cuDex's gravity center to the actual native collision mass model.

The retained v3 optimizer and native qualifier are unchanged. A process-local
configuration hook supplies measured-from-native COM instead of the visual
surface's possibly undefined signed-volume COM. This is not measured real mass.
"""
import argparse
from pathlib import Path
from unittest.mock import patch
import numpy as np
import trimesh
from .common import binding, read, resolve, write
from . import cudex_constraint_v3 as retained

VERSION = 'cudex_native_COM_v4'


def condition(asset):
    d = read(resolve(asset)); native = read(resolve(d['native_properties']))
    support = read(resolve(d['native_support']))
    center = np.asarray(native['center_of_mass_m'], float)
    bounds = np.asarray(d['collision_bounds_m'], float)
    if (center.shape != (3,) or not np.isfinite(center).all()
            or np.any(center < bounds[0]) or np.any(center > bounds[1])
            or not np.isfinite(native['mass_kg']) or native['mass_kg'] <= 0
            or support['status'] != 'pass' or not d['roles']['rigid_manipulation_candidate']):
        raise ValueError('bound_native_COM_support_or_geometry_invalid')
    mesh = trimesh.load(resolve(d['visual']), force='scene', process=False).to_geometry()
    if not np.isfinite(mesh.vertices).all() or np.any(mesh.extents <= 0):
        raise ValueError('nonfinite_or_degenerate_visual_geometry')
    return {'schema': VERSION, 'asset': asset, 'native_properties': d['native_properties'],
        'native_support': d['native_support'], 'gravity_center_m': center.tolist(),
        'visual_COM_finite': bool(np.isfinite(mesh.center_mass).all()),
        'visual_COM_m': mesh.center_mass.tolist() if np.isfinite(mesh.center_mass).all() else None,
        'COM_source': 'PhysX_compound_uniform_density_engineering_mass_model',
        'physical_real_COM_measured': False, 'qualification_rules_changed': False}


def worker(job_path):
    from curobo.wrap.reacher.grasp_solver import GraspSolverConfig
    from .cudex_seed_initialization_v2 import install
    job = read(job_path); value = condition(job['asset'])
    write(Path(job['output']) / 'native_COM_condition.json',
          {**value, 'implementation': binding(__file__), 'job': binding(job_path)})
    original = GraspSolverConfig.load_from_robot_config
    def config(*args, **kwargs):
        if len(kwargs.get('obj_gravity_center', [])) != 1:
            raise ValueError('single_bound_asset_required')
        kwargs['obj_gravity_center'] = [value['gravity_center_m']]
        return original(*args, **kwargs)
    with patch.object(GraspSolverConfig, 'load_from_robot_config', side_effect=config):
        with install(Path(job['output']), job['seed']):
            return retained.worker(job_path)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['worker', 'export'])
    p.add_argument('--job', type=Path, required=True)
    a=p.parse_args(); print(worker(a.job) if a.stage == 'worker' else retained.export(a.job))
