"""Observe actual solver inputs in a separately budgeted repeatability probe.

No inputs are substituted, rounded, sorted or reseeded. Each worker runs the
retained optimizer once, optionally with the frozen measured-empty fallback.
"""
from contextlib import nullcontext
import inspect
from pathlib import Path

import numpy as np

from .common import binding, read, write

VERSION = 'cudex_actual_seed_input_probe_v1'


def worker(job_path):
    import torch
    from curobo.wrap.reacher.grasp_solver import GraspSolver
    from . import cudex_constraint_v3 as retained
    from . import cudex_seed_initialization_v2 as conditional
    job_path = Path(job_path).resolve(); job = read(job_path); output = Path(job['output'])
    if job.get('seed_probe') != binding(__file__) or job['probe_variant'] not in ('original', 'conditional'):
        raise ValueError('frozen_explicit_probe_job_required')
    original = GraspSolver.get_seed; calls = [0]

    def observe(self, *args, **kwargs):
        if calls[0]: raise ValueError('probe_permits_one_optimizer_seed_call_only')
        calls[0] += 1
        seed_tensor = original(self, *args, **kwargs)
        world = self.world_coll_checker
        def array(value): return value.detach().cpu().numpy().copy()
        values = {'q_start': array(seed_tensor), 'filtered_positions': array(world.surface_sample_positions),
            'filtered_normals': array(world.surface_sample_normals),
            'filtered_lower': array(world.surface_sample_ind_lower), 'filtered_upper': array(world.surface_sample_ind_upper),
            'torch_cpu_rng_state': array(torch.get_rng_state()),
            'torch_cuda_rng_state': array(torch.cuda.get_rng_state())}
        state = np.random.get_state()
        values.update(numpy_mt_state=state[1], numpy_mt_position=np.asarray(state[2]),
                      numpy_has_gauss=np.asarray(state[3]), numpy_cached_gaussian=np.asarray(state[4]))
        np.savez_compressed(output / 'actual_optimizer_input.npz', **values)
        write(output / 'actual_optimizer_input.json', {'schema': VERSION,
            'implementation': binding(__file__), 'original_get_seed_source': binding(inspect.getfile(GraspSolver)),
            'arrays': binding(output / 'actual_optimizer_input.npz'), 'variant': job['probe_variant'],
            'optimizer_seed_calls': calls[0], 'inputs_substituted': False, 'RNG_reseeded_by_probe': False,
            'numpy_rng_algorithm': state[0], 'synchronization': 'CPU snapshots before first Newton call'})
        return seed_tensor

    GraspSolver.get_seed = observe
    try:
        context = conditional.install(output, job['seed']) if job['probe_variant'] == 'conditional' else nullcontext()
        with context: return retained.worker(job_path)
    finally: GraspSolver.get_seed = original
