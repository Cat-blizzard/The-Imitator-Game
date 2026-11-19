"""Candidate cuDex seed repair: offset sampled convex faces, not mesh vertices.

The native collision filter, Newton optimizer, 256-seed solve and pose qualifier
are retained. Empty/invalid seed batches fail explicitly before Newton; they
are never turned into successful empty pose caches. Frozen upstream files are
not edited. install() is scoped to one separately budgeted worker process.
"""
from contextlib import contextmanager
import argparse
import inspect
import math
from pathlib import Path

import numpy as np

from .common import binding, read, resolve, write

VERSION = 'cudex_convex_face_offset_seed_v1'


class SeedInitializationError(ValueError):
    """Invalid initialization is not evidence that the asset is ungraspable."""


def surface_samples(mesh, count, inflate=.1, with_radius=True, *, seed=None):
    import trimesh
    if type(count) is not int or count <= 0 or not np.isfinite(inflate) or inflate <= 0:
        raise ValueError('positive_finite_surface_sampling_parameters_required')
    hull = mesh.copy().convex_hull
    if not hull.is_watertight or not hull.is_winding_consistent or hull.volume <= 0:
        raise SeedInitializationError('convex_hull_requires_positive_closed_geometry')
    points, faces = trimesh.sample.sample_surface_even(hull, count, seed=seed)
    if len(points) == 0: raise SeedInitializationError('surface_sampler_returned_no_points')
    repeat = math.ceil(count / len(points))
    points = np.tile(points, (repeat, 1))[:count]
    faces = np.tile(faces, repeat)[:count]
    normals = np.asarray(hull.face_normals)[faces].copy()
    points = np.asarray(points) + inflate * normals
    if not np.isfinite(points).all() or not np.isfinite(normals).all():
        raise SeedInitializationError('nonfinite_surface_initialization')
    # Every point lies inflate metres outside a supporting plane of the hull;
    # therefore its distance to that hull (and enclosed asset) is >= inflate.
    if with_radius: points = np.column_stack([points, np.full(count, .9 * inflate)])
    return points, normals


def validate_surface_pool(positions, lower, upper, environments):
    """Do not let bincount without minlength silently erase an empty environment."""
    def array(value):
        if hasattr(value, 'detach'): value = value.detach().cpu().numpy()
        return np.asarray(value)
    positions, lower, upper = array(positions), array(lower), array(upper)
    if (type(environments) is not int or environments <= 0 or lower.shape != (environments,)
            or upper.shape != (environments,)):
        raise SeedInitializationError('environment_axis_lost_by_empty_surface_filter')
    counts = upper - lower
    if (np.any(counts <= 0) or upper[-1] != len(positions)
            or lower[0] != 0 or not np.array_equal(lower[1:], upper[:-1])):
        raise SeedInitializationError('no_collision_free_surface_seed_for_one_or_more_environments')
    if positions.shape != (int(upper[-1]), 3) or not np.isfinite(positions).all():
        raise SeedInitializationError('invalid_filtered_surface_geometry')
    return counts.astype(int).tolist()


def validate_seed_tensor(q, batch, num_seeds, dof):
    expected = (batch * num_seeds, 1, dof)
    if tuple(q.shape) != expected:
        raise SeedInitializationError(f'initial_seed_shape_mismatch_expected_{expected}_actual_{tuple(q.shape)}')
    if hasattr(q, 'isfinite'):
        finite = bool(q.isfinite().all())
    else: finite = bool(np.isfinite(q).all())
    if not finite: raise SeedInitializationError('nonfinite_initial_seed_tensor')
    return q


@contextmanager
def install(output, seed):
    """Patch only initialization methods; restore originals even on a failure."""
    from curobo.geom.types import Obstacle
    from curobo.util.sample_grasp import HeurGraspSeedGenerator
    from curobo.wrap.reacher.grasp_solver import GraspSolver
    import trimesh
    output = Path(output).resolve(); calls = [0]
    old_surface = Obstacle.get_samples_on_ch_surface
    old_base = HeurGraspSeedGenerator._set_base_trq
    old_seed = GraspSolver.get_seed
    write(output / 'seed_candidate_implementation.json', {'schema': VERSION, 'implementation': binding(__file__),
        'upstream_surface': binding(inspect.getfile(Obstacle)),
        'upstream_seeder': binding(inspect.getfile(HeurGraspSeedGenerator)),
        'upstream_solver': binding(inspect.getfile(GraspSolver)),
        'collision_filter_changed': False, 'Newton_changed': False, 'native_qualifier_changed': False,
        'initialization_only': True, 'new_physical_execution': False})

    def candidate_surface(self, sample_num, inflate=.1, with_radius=True):
        mesh = self.get_trimesh_mesh().copy()
        mesh.apply_transform(trimesh.scene.transforms.kwargs_to_matrix(
            translation=self.pose[:3], quaternion=self.pose[3:]))
        index = calls[0]; calls[0] += 1
        positions, normals = surface_samples(mesh, sample_num, inflate, with_radius, seed=seed + index)
        np.savez_compressed(output / f'seed_surface_{index:03d}.npz', positions=positions, normals=normals)
        write(output / f'seed_surface_{index:03d}.json', {'schema': VERSION, 'mesh_name': self.name,
            'sample_count': sample_num, 'inflate_m': inflate, 'sphere_radius_m': .9 * inflate,
            'seed': seed + index, 'samples': binding(output / f'seed_surface_{index:03d}.npz'),
            'sampled_convex_face_then_offset_outward': True, 'collision_filter_applied_upstream': True})
        return positions, normals

    def checked_base(self, t, r, q, extra_info=None):
        if t is None and r is None and self.tr_link_names is not None:
            try:
                counts = validate_surface_pool(extra_info.surface_sample_positions,
                    extra_info.surface_sample_ind_lower, extra_info.surface_sample_ind_upper,
                    len(extra_info._contact_mesh_idx))
                write(output / 'seed_pool.json', {'schema': VERSION, 'counts_per_environment': counts,
                    'status': 'nonempty_collision_filtered_initialization_only'})
            except SeedInitializationError as exc:
                write(output / 'seed_initialization_failure.json', {'schema': VERSION, 'reason': str(exc),
                    'stage': 'filtered_surface_pool', 'optimizer_called': False,
                    'asset_grasp_feasibility': 'unknown', 'successful_cache_written': False})
                raise
        return old_base(self, t, r, q, extra_info)

    def checked_seed(self, num_seeds, goal_pose, use_nn_seed, seed_config=None):
        value = old_seed(self, num_seeds, goal_pose, use_nn_seed, seed_config)
        try: validate_seed_tensor(value, goal_pose.batch, num_seeds, self.dof)
        except SeedInitializationError as exc:
            write(output / 'seed_initialization_failure.json', {'schema': VERSION, 'reason': str(exc),
                'stage': 'solver_seed_tensor', 'optimizer_called': False,
                'asset_grasp_feasibility': 'unknown', 'successful_cache_written': False})
            raise
        write(output / 'seed_shape.json', {'shape': list(value.shape), 'schema': VERSION,
            'batch': goal_pose.batch, 'num_seeds': num_seeds, 'dof': self.dof, 'finite': True})
        return value

    Obstacle.get_samples_on_ch_surface, HeurGraspSeedGenerator._set_base_trq, GraspSolver.get_seed = candidate_surface, checked_base, checked_seed
    try: yield
    finally:
        Obstacle.get_samples_on_ch_surface, HeurGraspSeedGenerator._set_base_trq, GraspSolver.get_seed = old_surface, old_base, old_seed


def worker(job_path):
    from .cudex_constraint_v3 import worker as retained_worker
    import os
    job_path = Path(job_path).resolve(); job = read(job_path)
    if os.environ.get('RESIMPAIR_CUDEX_ALIGNMENT_JOB') != str(job_path):
        raise ValueError('exclusive_bounded_supervisor_required')
    if (Path(job['output']) / 'worker_started.json').exists():
        raise ValueError('original_attempt_must_not_be_retried_or_overwritten')
    if job.get('seed_candidate') != binding(__file__):
        raise ValueError('separately_frozen_candidate_job_required')
    with install(job['output'], job['seed']): return retained_worker(job_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args(); print(worker(args.job))
