"""Native cuDex constraint diagnostics and an opt-in radius-aware candidate.

Only the *constraint* query changes from signed ESDF to nonnegative violation.
Optimization costs, robot, contact strategy, seeds and convergence thresholds
remain unchanged. The original solver is never edited. Every returned state
also records both predicates, component violations and per-sphere distances.
This is not a physical grasp certification and is not the default cache policy.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import time

import numpy as np
import yaml

from .common import binding, read, resolve, write
from .cudex_native import UPSTREAM, hand_to_tcp

VERSION = 'native_cudex_constraint_v2'


def prepare(manip_path, output):
    """CPU preparation, preserving the exact native robot and all cost files."""
    manip_path, output = Path(manip_path).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('preserve_existing_constraint_candidate')
    source = yaml.safe_load(manip_path.read_text())
    base = Path(source['base_cfg_file'])
    if not base.is_absolute():
        base = UPSTREAM / 'src/curobo/content/configs/task' / base
    original = yaml.safe_load(base.read_text())
    if original['constraint']['primitive_collision_cfg']['compute_esdf'] is not True:
        raise ValueError('expected_retained_signed_ESDF_configuration')
    fixed = deepcopy(original)
    fixed['constraint']['primitive_collision_cfg']['compute_esdf'] = False
    output.mkdir(parents=True)
    base_out = output / 'base_grasp_nonnegative.yml'
    base_out.write_text(yaml.safe_dump(fixed))
    candidate = deepcopy(source)
    candidate['base_cfg_file'] = str(base_out)
    manip_out = output / 'manip_nonnegative.yml'
    manip_out.write_text(yaml.safe_dump(candidate))
    inverse = deepcopy(fixed)
    inverse['constraint']['primitive_collision_cfg']['compute_esdf'] = True
    assert inverse == original
    write(output / 'protocol.json', {
        'schema': VERSION, 'original_manip': binding(manip_path),
        'original_base': binding(base), 'candidate_manip': binding(manip_out),
        'candidate_base': binding(base_out), 'implementation': binding(__file__),
        'only_base_config_change': 'constraint.primitive_collision_cfg.compute_esdf:true->false',
        'same_costs_contact_strategy_robot_seeds_and_thresholds': True,
        'sum_less_equal_zero_shortcut': False, 'physical_success': 'unmeasured',
        'gpu_jobs': 0, 'api_calls': 0, 'default_promoted': False})
    return manip_out


def component_snapshot(rollout, joint_state):
    """Evaluate both world metrics on exactly the same optimized state.

    Clone every buffer before the next query; cuRobo reuses CUDA buffers.
    Component-wise nonnegativity prevents a signed world term from cancelling
    a joint-limit or self-collision violation. Radius-aware collision queries
    are used rather than interpreting point ESDF as sphere clearance.
    """
    import torch
    with torch.no_grad():
        state = rollout._get_augmented_state(joint_state)
        bound = rollout.bound_constraint.forward(state.state_seq).clone()
        own = rollout.robot_self_collision_constraint.forward(state.robot_spheres).clone()
        primitive = rollout.primitive_collision_constraint
        previous = primitive.compute_esdf
        if not primitive.enabled or not rollout.robot_self_collision_constraint.enabled:
            raise ValueError('enabled_world_and_self_constraints_required')
        data = {'q_unsorted': joint_state.position.detach().cpu().numpy().copy(),
                'bound': bound.cpu().numpy().copy(), 'self': own.cpu().numpy().copy(),
                'robot_spheres_xyzw': state.robot_spheres.cpu().numpy().copy(),
                'contact_bias': primitive._get_contact_stage(1.0).cpu().numpy().copy()}
        values = {}
        try:
            for name, flag in [('signed_esdf', True), ('nonnegative_violation', False)]:
                primitive.compute_esdf = flag
                values[name] = primitive.forward(state.robot_spheres,
                    env_query_idx=rollout._goal_buffer.batch_world_idx, opt_progress=1.0).clone()
                data['world_' + name] = values[name].cpu().numpy().copy()
                query = primitive._collision_query_buffer
                for kind in ('mesh', 'primitive'):
                    buffer = getattr(query, kind + '_collision_buffer', None)
                    if buffer is not None:
                        data[kind + '_' + name + '_per_sphere'] = buffer.distance_buffer.cpu().numpy().copy()
        finally:
            primitive.compute_esdf = previous
        original = (bound + own + values['signed_esdf']) == 0
        world = values['nonnegative_violation']
        finite = torch.isfinite(bound) & torch.isfinite(own) & torch.isfinite(world)
        no_negative_components = (bound >= 0) & (own >= 0) & (world >= 0)
        data['nonnegative_semantics_valid'] = (finite & no_negative_components).cpu().numpy().copy()
        corrected = (bound == 0) & (own == 0) & (world == 0)
        data['original_sum_equal_zero'] = original.cpu().numpy().copy()
        data['componentwise_nonnegative_feasible'] = corrected.cpu().numpy().copy()
        return data


def worker(job_path):
    """Finite supervisor-only GPU entry; raw states and failures stay durable."""
    import os
    import torch
    import trimesh
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
    # Preserve native tabletop support; never translate a solved grasp afterward.
    z = float(descriptor['collision_bounds_m'][0][2])
    world = {'mesh': {'payload': {'file_path': str(mesh_path), 'pose': [0, 0, 0, 1, 0, 0, 0]}},
             'cuboid': {'table': {'dims': [2, 2, .10], 'pose': [0, 0, z - .05, 1, 0, 0, 0]}}}
    manip = yaml.safe_load(resolve(job['manip']).read_text())
    manip['seed_num'] = 256
    torch.manual_seed(job['seed']); np.random.seed(job['seed'])
    write(out / 'effective_input.json', {'world': world, 'manip': manip, 'job': binding(job_path),
        'table_support_from': 'native_collision_bounds', 'visual_table_z_difference_m': float(mesh.bounds[0, 2]-z)})
    config = GraspSolverConfig.load_from_robot_config(world_model=[world], manip_name_list=['payload'],
        obj_gravity_center=[mesh.center_mass.tolist()],
        obj_obb_length=[[float(np.linalg.norm(mesh.bounding_box_oriented.primitive.extents) / 2)]],
        manip_config_data=manip, use_cuda_graph=False)

    class DiagnosticSolver(GraspSolver):
        def get_result(self, num_seeds, result, goal_pose, return_seeds):
            retained = {'q_unsorted': result.action.position.detach().cpu().numpy().copy()}
            for name in ('feasible', 'constraint', 'grasp_error', 'dist_error'):
                retained['upstream_' + name] = getattr(result.metrics, name).detach().cpu().numpy().copy()
            # Preserve the expensive result even if the diagnostic itself fails.
            np.savez_compressed(out / 'solver_state_before_diagnostics.npz', **retained)
            data = component_snapshot(self.rollout_fn, result.action)
            np.savez_compressed(out / 'constraint_components_unsorted.npz', **{**data, **retained})
            if not data['nonnegative_semantics_valid'].all():
                raise ValueError('invalid_nonnegative_constraint_semantics_see_saved_components')
            return super().get_result(num_seeds, result, goal_pose, return_seeds)

    import inspect
    from curobo.rollout.arm_base import ArmBase
    write(out / 'runtime_sources.json', {'grasp_solver': binding(inspect.getfile(GraspSolver)),
        'rollout': binding(inspect.getfile(ArmBase)), 'worker': binding(__file__),
        'comparison_order': 'same_unsorted_q_for_both_predicates; raw.npz_has_separate_upstream_sorted_order',
        'historical_139_population_recomputed': False})
    solver = DiagnosticSolver(config)
    setup = time.monotonic() - started
    torch.cuda.synchronize(); solving = time.monotonic()
    result = solver.solve_batch_env(return_seeds=256, num_seeds=256)
    torch.cuda.synchronize(); elapsed = time.monotonic() - solving
    arrays = {name: getattr(result, name).detach().cpu().numpy() for name in
              ('solution', 'success', 'grasp_error', 'dist_error', 'contact_point', 'contact_frame')}
    q = arrays['solution'][0]
    final = q[:, -1, :] if q.ndim == 3 else q
    arrays['poses'] = np.asarray([hand_to_tcp(row) for row in final])
    np.savez_compressed(out / 'raw.npz', **arrays)
    components = np.load(out / 'constraint_components_unsorted.npz', allow_pickle=False)
    old, new = components['original_sum_equal_zero'], components['componentwise_nonnegative_feasible']
    report = {'schema': VERSION, 'raw': binding(out / 'raw.npz'),
        'components': binding(out / 'constraint_components_unsorted.npz'),
        'upstream_success_count': int(arrays['success'].sum()),
        'old_infeasible_but_nonnegative_feasible': int((~old & new).sum()),
        'old_feasible_but_nonnegative_infeasible': int((old & ~new).sum()),
        'setup_seconds': setup, 'solve_seconds': elapsed, 'worker_seconds': time.monotonic()-started,
        'raw_poses_posttranslated': False, 'seed_count': 256,
        'physical_success': 'unmeasured', 'native_mesh_closure_and_physics_still_required': True}
    write(out / 'result.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare'); p.add_argument('--manip', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('worker'); p.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.manip, args.output) if args.command == 'prepare' else worker(args.job))
