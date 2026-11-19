"""Preserve the original cuDex seeds; repair only a measured empty CUDA pool.

Exactly one initialization fallback is allowed before Newton. A nonempty pool
uses the original sampler, collision filter and random stream. This version
does not retry an optimizer, adjust geometry or change pose qualification.
"""
from contextlib import contextmanager
import inspect
from pathlib import Path

import numpy as np

from .common import binding, write
from .cudex_seed_initialization_v1 import (
    SeedInitializationError, surface_samples, validate_seed_tensor, validate_surface_pool)

VERSION = 'cudex_measured_empty_pool_fallback_v2'


def array(value):
    if hasattr(value, 'detach'): value = value.detach().cpu().numpy()
    return np.asarray(value)


def pool_action(positions, lower, upper, environments):
    positions, lower, upper = map(array, (positions, lower, upper))
    if environments != 1:
        raise SeedInitializationError('conditional_repair_scope_is_one_environment_per_job')
    if positions.shape == (0, 3):
        if ((lower.shape == upper.shape == (0,)) or
                (lower.shape == upper.shape == (1,) and lower[0] == upper[0] == 0)):
            return 'repair_measured_empty_pool'
        raise SeedInitializationError('empty_pool_has_inconsistent_indices')
    validate_surface_pool(positions, lower, upper, environments)
    return 'preserve_original_nonempty_pool'


@contextmanager
def install(output, seed):
    from curobo.geom.types import Obstacle
    from curobo.geom.sdf.world_mesh import WorldMeshCollision
    from curobo.util.sample_grasp import HeurGraspSeedGenerator
    from curobo.wrap.reacher.grasp_solver import GraspSolver
    import trimesh

    output = Path(output).resolve()
    old_surface = Obstacle.get_samples_on_ch_surface
    old_load = WorldMeshCollision.load_contact_obj
    old_base = HeurGraspSeedGenerator._set_base_trq
    old_seed = GraspSolver.get_seed
    mode = {'fallback': False, 'surface_calls': 0, 'load_calls': 0}
    write(output / 'seed_candidate_implementation.json', {'schema': VERSION, 'implementation': binding(__file__),
        'upstream_surface': binding(inspect.getfile(Obstacle)), 'upstream_world_mesh': binding(inspect.getfile(WorldMeshCollision)),
        'upstream_seeder': binding(inspect.getfile(HeurGraspSeedGenerator)), 'upstream_solver': binding(inspect.getfile(GraspSolver)),
        'collision_filter_changed': False, 'Newton_changed': False, 'native_qualifier_changed': False,
        'initialization_only': True, 'trigger': 'one_environment_actual_CUDA_filtered_positions_empty',
        'maximum_initialization_fallbacks': 1, 'optimizer_retries': 0,
        'nonempty_original_pool_and_random_stream_preserved': True})

    def measured_surface(self, sample_num, inflate=.1, with_radius=True):
        if not mode['fallback']:
            # No local RNG or alternative sampling is called on the healthy path.
            return old_surface(self, sample_num, inflate, with_radius)
        mesh = self.get_trimesh_mesh().copy()
        mesh.apply_transform(trimesh.scene.transforms.kwargs_to_matrix(translation=self.pose[:3], quaternion=self.pose[3:]))
        index = mode['surface_calls']; mode['surface_calls'] += 1
        positions, normals = surface_samples(mesh, sample_num, inflate, with_radius, seed=seed + index)
        np.savez_compressed(output / f'fallback_surface_{index:03d}.npz', positions=positions, normals=normals)
        return positions, normals

    def checked_load(self, world_model, contact_obj_names):
        if mode['load_calls']:
            raise SeedInitializationError('unexpected_second_contact_load_not_an_optimizer_retry')
        mode['load_calls'] += 1
        result = old_load(self, world_model, contact_obj_names)
        environments = len(self._contact_mesh_idx)
        before = {name: array(getattr(self, 'surface_sample_' + name)).copy()
                  for name in ('positions', 'normals', 'ind_lower', 'ind_upper')}
        np.savez_compressed(output / 'original_filtered_seed_pool.npz', **before)
        action = pool_action(before['positions'], before['ind_lower'], before['ind_upper'], environments)
        if action == 'repair_measured_empty_pool':
            mode['fallback'] = True
            try: result = old_load(self, world_model, contact_obj_names)
            finally: mode['fallback'] = False
        counts = validate_surface_pool(self.surface_sample_positions, self.surface_sample_ind_lower,
                                       self.surface_sample_ind_upper, environments)
        unchanged = all(np.array_equal(before[name], array(getattr(self, 'surface_sample_' + name))) for name in before)
        if action == 'preserve_original_nonempty_pool' and not unchanged:
            raise SeedInitializationError('healthy_original_pool_was_modified')
        write(output / 'seed_path.json', {'schema': VERSION, 'action': action,
            'original_filtered_count': len(before['positions']), 'final_filtered_counts': counts,
            'original_pool': binding(output / 'original_filtered_seed_pool.npz'),
            'original_nonempty_pool_bitwise_unchanged': unchanged if len(before['positions']) else None,
            'initialization_fallbacks': int(action == 'repair_measured_empty_pool'), 'optimizer_retries': 0})
        return result

    def checked_base(self, t, r, q, extra_info=None):
        if t is None and r is None and self.tr_link_names is not None:
            counts = validate_surface_pool(extra_info.surface_sample_positions, extra_info.surface_sample_ind_lower,
                extra_info.surface_sample_ind_upper, len(extra_info._contact_mesh_idx))
            write(output / 'seed_pool.json', {'schema': VERSION, 'counts_per_environment': counts,
                'status': 'nonempty_collision_filtered_initialization_only'})
        return old_base(self, t, r, q, extra_info)

    def checked_seed(self, num_seeds, goal_pose, use_nn_seed, seed_config=None):
        value = old_seed(self, num_seeds, goal_pose, use_nn_seed, seed_config)
        validate_seed_tensor(value, goal_pose.batch, num_seeds, self.dof)
        write(output / 'seed_shape.json', {'schema': VERSION, 'shape': list(value.shape), 'batch': goal_pose.batch,
            'num_seeds': num_seeds, 'dof': self.dof, 'finite': True})
        return value

    Obstacle.get_samples_on_ch_surface = measured_surface
    WorldMeshCollision.load_contact_obj = checked_load
    HeurGraspSeedGenerator._set_base_trq = checked_base
    GraspSolver.get_seed = checked_seed
    try: yield
    finally:
        Obstacle.get_samples_on_ch_surface = old_surface
        WorldMeshCollision.load_contact_obj = old_load
        HeurGraspSeedGenerator._set_base_trq = old_base
        GraspSolver.get_seed = old_seed
