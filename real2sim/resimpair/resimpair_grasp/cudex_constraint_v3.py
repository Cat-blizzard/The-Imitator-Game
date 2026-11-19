"""cuDex diagnostics with isolated query buffers and durable solver output.

The v2 diagnostic reused a primitive CUDA buffer across signed ESDF and
collision-cost queries. Safe primitive queries can skip writing a zero, so
the previous signed distance leaked into the next query. Each mode here owns
a newly initialized buffer. The optimizer and native pose qualification
remain separate; no pose translation or collision-score clamping is used.
"""
import argparse
import inspect
import os
from pathlib import Path
import time
import traceback

import numpy as np
import yaml

from .common import ROOT, binding, read, resolve, write
from .cudex_native import UPSTREAM, hand_to_tcp

VERSION = 'native_cudex_constraint_v3_isolated_queries'


def query_both_modes(primitive, spheres, env_query_idx, buffer_factory):
    """Never expose a query to another mode's distance/sparsity buffers.

    This helper is injectable for a CPU regression of the actual stale-buffer
    branch. A fresh allocation is mandatory, not a clone of the old buffer.
    Restore the optimizer's original buffer even if either diagnostic fails.
    """
    previous_flag = primitive.compute_esdf
    previous_buffer = primitive._collision_query_buffer
    values, per_sphere = {}, {}
    try:
        for name, flag in [('signed_esdf', True), ('nonnegative_violation', False)]:
            primitive.compute_esdf = flag
            primitive._collision_query_buffer = buffer_factory()
            values[name] = primitive.forward(
                spheres, env_query_idx=env_query_idx, opt_progress=1.0).clone()
            for kind in ('mesh', 'primitive'):
                buf = getattr(primitive._collision_query_buffer, kind + '_collision_buffer', None)
                if buf is not None:
                    per_sphere[kind + '_' + name + '_per_sphere'] = buf.distance_buffer.clone()
    finally:
        primitive.compute_esdf = previous_flag
        primitive._collision_query_buffer = previous_buffer
    return values, per_sphere


def component_snapshot(rollout, joint_state):
    import torch
    from curobo.geom.sdf.world import CollisionQueryBuffer
    with torch.no_grad():
        state = rollout._get_augmented_state(joint_state)
        bound = rollout.bound_constraint.forward(state.state_seq).clone()
        own = rollout.robot_self_collision_constraint.forward(state.robot_spheres).clone()
        primitive = rollout.primitive_collision_constraint
        if not primitive.enabled or not rollout.robot_self_collision_constraint.enabled:
            raise ValueError('enabled_world_and_self_constraints_required')
        values, sphere_data = query_both_modes(
            primitive, state.robot_spheres, rollout._goal_buffer.batch_world_idx,
            lambda: CollisionQueryBuffer.initialize_from_shape(
                state.robot_spheres.shape, primitive.tensor_args,
                primitive.world_coll_checker.collision_types))
        world = values['nonnegative_violation']
        finite = torch.isfinite(bound) & torch.isfinite(own) & torch.isfinite(world)
        valid = finite & (bound >= 0) & (own >= 0) & (world >= 0)
        tensors = {
            'q_unsorted': joint_state.position,
            'bound': bound, 'self': own, 'robot_spheres_xyzw': state.robot_spheres,
            'contact_bias': primitive._get_contact_stage(1.0),
            **{'world_' + name: value for name, value in values.items()},
            **sphere_data,
            'nonnegative_semantics_valid': valid,
            'original_sum_equal_zero': (bound + own + values['signed_esdf']) == 0,
            'componentwise_nonnegative_feasible': valid & (bound == 0) & (own == 0) & (world == 0),
        }
        return {name: value.detach().cpu().numpy().copy() for name, value in tensors.items()}


def worker(job_path):
    """One bounded GPU solve. CPU geometry qualification runs after GPU exit."""
    import torch
    import trimesh
    from curobo.rollout.arm_base import ArmBase
    from curobo.wrap.reacher.grasp_solver import GraspSolver, GraspSolverConfig
    job_path = Path(job_path).resolve()
    if os.environ.get('RESIMPAIR_CUDEX_ALIGNMENT_JOB') != str(job_path):
        raise ValueError('exclusive_bounded_supervisor_required')
    job = read(job_path)
    out = Path(job['output'])
    if (out / 'worker_started.json').exists():
        raise ValueError('previous_worker_started_recover_without_resubmission')
    write(out / 'worker_started.json', {'unix': time.time(), 'pid': os.getpid(), 'job': binding(job_path)})
    started = time.monotonic()
    descriptor = read(resolve(job['asset']))
    mesh = trimesh.load(resolve(descriptor['visual']), force='scene', process=False).to_geometry()
    mesh_path = out / 'object.obj'
    mesh.export(mesh_path)
    z = float(descriptor['collision_bounds_m'][0][2])
    world = {'mesh': {'payload': {'file_path': str(mesh_path), 'pose': [0, 0, 0, 1, 0, 0, 0]}},
             'cuboid': {'table': {'dims': [2, 2, .10], 'pose': [0, 0, z - .05, 1, 0, 0, 0]}}}
    manip = yaml.safe_load(resolve(job['manip']).read_text())
    manip['seed_num'] = 256
    torch.manual_seed(job['seed'])
    np.random.seed(job['seed'])
    write(out / 'effective_input.json', {'world': world, 'manip': manip, 'job': binding(job_path),
        'table_support_from': 'native_collision_bounds',
        'visual_table_z_difference_m': float(mesh.bounds[0, 2] - z),
        'no_postsolve_pose_translation': True})
    config = GraspSolverConfig.load_from_robot_config(world_model=[world], manip_name_list=['payload'],
        obj_gravity_center=[mesh.center_mass.tolist()],
        obj_obb_length=[[float(np.linalg.norm(mesh.bounding_box_oriented.primitive.extents) / 2)]],
        manip_config_data=manip, use_cuda_graph=False)

    class DiagnosticSolver(GraspSolver):
        diagnostic_error = None

        def get_result(self, num_seeds, result, goal_pose, return_seeds):
            retained = {'q_unsorted': result.action.position.detach().cpu().numpy().copy()}
            for name in ('feasible', 'constraint', 'grasp_error', 'dist_error',
                         'contact_point', 'contact_frame', 'contact_force'):
                value = getattr(result.metrics, name, None)
                if value is not None:
                    retained['upstream_' + name] = value.detach().cpu().numpy().copy()
            # This unsorted result can be recovered without a second GPU solve.
            np.savez_compressed(out / 'solver_state_before_diagnostics.npz', **retained)
            try:
                data = component_snapshot(self.rollout_fn, result.action)
                np.savez_compressed(out / 'constraint_components_unsorted.npz', **{**data, **retained})
                if not data['nonnegative_semantics_valid'].all():
                    raise ValueError('invalid_nonnegative_constraint_semantics_see_saved_components')
            except Exception as exc:
                self.diagnostic_error = str(exc)
                write(out / 'diagnostic_failure.json', {
                    'error': str(exc), 'traceback': traceback.format_exc(),
                    'raw_state': binding(out / 'solver_state_before_diagnostics.npz')})
            # Always save the normal sorted output too, even if diagnostics fail.
            return super().get_result(num_seeds, result, goal_pose, return_seeds)

    write(out / 'runtime_sources.json', {
        'grasp_solver': binding(inspect.getfile(GraspSolver)),
        'rollout': binding(inspect.getfile(ArmBase)), 'worker': binding(__file__),
        'query_buffers': 'fresh_independent_zero_initialized_buffer_per_mode',
        'comparison_order': 'both_predicates_on_same_unsorted_q; raw_poses_use_upstream_sorted_order',
        'physical_success': 'unmeasured'})
    solver = DiagnosticSolver(config)
    setup = time.monotonic() - started
    torch.cuda.synchronize()
    solving = time.monotonic()
    result = solver.solve_batch_env(return_seeds=256, num_seeds=256)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - solving
    arrays = {name: getattr(result, name).detach().cpu().numpy() for name in
              ('solution', 'success', 'grasp_error', 'dist_error', 'contact_point', 'contact_frame')}
    q = arrays['solution'][0]
    final = q[:, -1, :] if q.ndim == 3 else q
    arrays['poses'] = np.asarray([hand_to_tcp(row) for row in final])
    if arrays['poses'].shape != (256, 4, 4) or not np.isfinite(arrays['poses']).all():
        raise ValueError('expected_256_finite_native_tcp_poses')
    np.savez_compressed(out / 'raw.npz', **arrays)
    report = {'schema': VERSION, 'status': 'optimizer_complete' if solver.diagnostic_error is None else 'diagnostic_failed',
        'job': binding(job_path), 'asset': job['asset'], 'raw': binding(out / 'raw.npz'),
        'raw_unsorted': binding(out / 'solver_state_before_diagnostics.npz'),
        'upstream_success_count': int(arrays['success'].sum()),
        'setup_seconds': setup, 'solve_seconds': elapsed, 'worker_seconds': time.monotonic() - started,
        'seed_count': 256, 'raw_poses_posttranslated': False,
        'physical_success': 'unmeasured', 'api_calls': 0,
        'nonnegative_semantics_valid': solver.diagnostic_error is None}
    component_path = out / 'constraint_components_unsorted.npz'
    if component_path.exists():
        data = np.load(component_path, allow_pickle=False)
        old, new = data['original_sum_equal_zero'], data['componentwise_nonnegative_feasible']
        report.update(components=binding(component_path),
            old_infeasible_but_nonnegative_feasible=int((~old & new).sum()),
            old_feasible_but_nonnegative_infeasible=int((old & ~new).sum()))
    write(out / 'optimizer_result.json', report)
    if solver.diagnostic_error is not None:
        raise ValueError('diagnostic_failed_raw_outputs_preserved:' + solver.diagnostic_error)
    return report


def export(job_path):
    """Apply the frozen common native qualification to all 256 original poses.

    A successful optimizer process can legitimately have an empty cache. The
    native open-hand/table sweep and finger closure rules are not weakened.
    No dynamics success is implied by an offline cache entry.
    """
    from resimpair_benchmark.proposals import qualify
    job_path = Path(job_path).resolve()
    job = read(job_path)
    out = Path(job['output'])
    optimizer = read(out / 'optimizer_result.json')
    if (optimizer['job'] != binding(job_path) or optimizer['asset'] != job['asset']
            or optimizer['status'] != 'optimizer_complete'
            or optimizer['nonnegative_semantics_valid'] is not True):
        raise ValueError('optimizer_integrity_or_diagnostic_gate_failed')
    raw_path = resolve(optimizer['raw'])
    with np.load(raw_path, allow_pickle=False) as raw:
        if raw['poses'].shape != (256, 4, 4) or not np.isfinite(raw['poses']).all():
            raise ValueError('invalid_raw_pose_shape_or_values')
    result_path = out / 'result.json'
    if result_path.exists():
        existing = read(result_path)
        if existing['optimizer'] != binding(out / 'optimizer_result.json'):
            raise ValueError('export_input_changed')
        resolve(existing['cache'])
        return existing
    started = time.monotonic()
    qualified = qualify(job['asset'], optimizer['raw'], out / 'native_qualification', geometry='native', maximum=256)
    report = {**optimizer, 'status': 'completed', 'optimizer': binding(out / 'optimizer_result.json'),
        'cache': qualified['cache'], 'decisions': binding(out / 'native_qualification/decisions.json'),
        'checked': qualified['checked'], 'retained': qualified['retained'],
        'rejections': qualified['rejections'], 'export_seconds': time.monotonic() - started,
        'cache_policy': 'optimized_proposals_common_native_hand_6d_v1_full256',
        'physical_success': 'unmeasured', 'default_promoted': False}
    write(result_path, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['worker', 'export'])
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    print(worker(args.job) if args.command == 'worker' else export(args.job))
