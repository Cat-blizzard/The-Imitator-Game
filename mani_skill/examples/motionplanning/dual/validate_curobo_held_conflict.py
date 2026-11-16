"""Connect complete TIG backend prefixes to the fixed held-traffic fixture.

Both physical grasps and the unique collision attachment are supplied by the
unchanged complete runner. The default fixture plans native outbound paths to
full TCP poses derived from the archived fixed middle joints, then derives the
return by exact time reversal. Return paths are explicitly NOT native solves.
Every candidate retains independent/full paired payload checks, zero-step
executor rejection and bounded start-delay coordination before real execution.
Controlled success is held traffic plus20 common stable ticks within200 total
actions; it is separate from official fruit-in-basket task completion.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import faulthandler
import hashlib
import inspect
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import time
import traceback
from typing import Any

import numpy as np

from .combined_collision import validate_schedule
from .conflict_coordination import delay_trajectory
from .curobo_backend import urdf_tcp_pose
from .curobo_scene import _matrix, _pose
from .held_conflict_fixture import (FIXTURE, METADATA, VERSION, RESERVE_STEPS,
    _context, _load_fixed, _qualify, _sha, _unchanged)
from .trajectory_execution import as_numpy
from .trajectory_schedule import ArmTrajectory, build_hold_schedule
from .validate_curobo import git_identity, save_json
from .validate_grasp import json_value
from .validate_held_path_conflict import save_traffic_plan
from .validate_sync import held_substep_audit, hold_joint_substep_audit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[6]
TASK = 'TwoRobotPickAppleBananaToBaskets-v1'
FINAL_HOVER, SETTLE_BUDGET, TERMINAL_MARGIN = 20, 5, 1


def reverse_round_trip(outbound: ArmTrajectory) -> tuple[ArmTrajectory, dict[str, Any]]:
    """Keep source times/positions and append their exact time-reversed return.

    Joint positions [N,7] remain radians. The source end appears only once;
    return timestamps are ``2*T - t[::-1]`` after dropping that duplicate end.
    This preserves path geometry and segment times; it performs no retiming.
    Finite-difference checks include reversal and stationary outer boundaries.
    """
    if not isinstance(outbound, ArmTrajectory) or len(outbound.times_s) < 2 or outbound.duration_s <= 0:
        raise ValueError('Native outbound must have positive duration and at least two samples')
    if outbound.gripper_command != -1.:
        raise ValueError('Both real held hands must remain closed')
    times, positions = outbound.times_s, outbound.positions_rad
    joined_time = np.r_[times, 2*outbound.duration_s-times[-2::-1]]
    joined_positions = np.vstack([positions, positions[-2::-1]])
    assert joined_positions.shape == (len(joined_time), 7)
    trajectory = ArmTrajectory(outbound.action_key, outbound.joint_names,
                               joined_time, joined_positions, -1.)
    # Include the executor's holds before t0 and after the common round trip.
    extended_time = np.r_[-(times[1]-times[0]), joined_time,
                          joined_time[-1]+times[1]-times[0]]
    extended_q = np.vstack([joined_positions[0], joined_positions, joined_positions[-1]])
    intervals = np.diff(extended_time)
    velocity = np.diff(extended_q, axis=0)/intervals[:, None]
    acceleration = np.diff(velocity, axis=0)/((intervals[:-1]+intervals[1:])/2.)[:, None]
    if not np.isfinite(velocity).all() or not np.isfinite(acceleration).all():
        raise ValueError('Nonfinite derived round-trip derivatives')
    max_v, max_a = float(np.max(np.abs(velocity))), float(np.max(np.abs(acceleration)))
    receipt = {'source': 'selected backend native outbound plus exact derived time reversal',
        'outbound_is_native_planner_output': True, 'return_is_native_planner_output': False,
        'return_derivation': 'same joint positions in reverse order; timestamps2T-minus-t',
        'outbound_samples': len(times), 'round_trip_samples': len(joined_time),
        'outbound_duration_s': outbound.duration_s, 'round_trip_duration_s': trajectory.duration_s,
        'source_positions_changed': False, 'source_time_scale_changed': False,
        'round_trip_endpoint_equals_measured_start': bool(np.array_equal(joined_positions[0], joined_positions[-1])),
        'max_sampled_position_difference_velocity_rad_s': max_v,
        'max_sampled_position_difference_acceleration_rad_s2': max_a,
        'velocity_limit_rad_s': 1.5, 'acceleration_limit_rad_s2': 2.6,
        'numeric_tolerance': 1e-6,
        'sampled_derivative_gate_passed': bool(max_v <= 1.5+1e-6 and max_a <= 2.6+1e-6),
        'derivative_scope': 'sampled positions including reversal/hold boundaries; original outbound native derivatives remain separately archived'}
    return trajectory, receipt


def fixed_middle_poses(raw: Any, context: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Resolve one immutable archived qmid into complete world TCP poses[2,7]."""
    metadata = json.loads(METADATA.read_text())
    if (metadata.get('fixture_version') != VERSION or _sha(FIXTURE) != metadata.get('fixture_npz_sha256')
            or metadata.get('fixture_source_sha256') != _sha(HERE/'held_conflict_fixture.py')):
        raise ValueError('Frozen held fixture version/source identity changed')
    with np.load(FIXTURE, allow_pickle=False) as source:
        middles, bases = source['middles'].copy(), source['base_poses'].copy()
    if (middles.shape != (2, 7) or bases.shape != (2, 7) or not np.isfinite(middles).all()
            or not np.allclose(bases, context['base_poses'], rtol=0, atol=1e-6)
            or metadata['joint_names'] != [list(names) for names in context['names']]
            or metadata['action_keys'] != list(context['keys'])):
        raise ValueError('Fixed midpoint physical base/joint/action identities differ')
    poses, model_hashes = [], {}
    for side, agent in enumerate(raw.agent.agents):
        actual_names = tuple(joint.get_name() for joint in agent.robot.get_active_joints())
        joints = dict(zip(actual_names, context['full_states'][side], strict=True))
        joints.update(zip(context['names'][side], middles[side], strict=True))
        model = Path(agent.urdf_path).resolve()
        tcp_base = urdf_tcp_pose(model, joints)
        poses.append(_pose(_matrix(context['base_poses'][side]) @ _matrix(tcp_base)))
        model_hashes[str(model)] = _sha(model)
    array = np.asarray(poses, dtype=float)
    assert array.shape == (2, 7) and np.isfinite(array).all()
    return array, {'fixture_version': VERSION, 'fixture_npz': str(FIXTURE),
        'fixture_npz_sha256': _sha(FIXTURE), 'fixture_json_sha256': _sha(METADATA),
        'fixed_middle_joints_rad': middles.tolist(), 'fixed_world_TCP_poses_XYZ_WXYZ': poses,
        'world_poses_source': 'independent actual URDF FK at archived fixed qmid and measured unchanged base',
        'urdf_source_sha256': model_hashes, 'per_case_IK_or_geometry_selection': False,
        'goal_pose_identical_for_both_backends': True}


def conflict_hook(**context: Any) -> bool:
    """Run controlled traffic after the complete runner's actual grasp/liftoff prefix."""
    from .dual_arm_execution import execute_dual_held_move
    args, output, report = context['args'], context['output'], context['report']
    raw, checker, solvers = context['raw'], context['checker'], context['solvers']
    capture, targets, references = context['capture'], context['targets'], context['references']
    begin, finish, plan = context['begin'], context['finish'], context['plan']
    substeps, ticks = context['substeps'], context['ticks']
    prefix = int(as_numpy(raw.elapsed_steps)[0])
    remaining = 200-prefix
    maximum = remaining-RESERVE_STEPS
    report.update(prefix_completed_at_step=prefix, controlled_outcome='not_completed',
        traffic_budget={'total_hard_budget': 200, 'prefix_steps': prefix, 'remaining_steps': remaining,
            'maximum_schedule_steps': maximum, 'settle_max_steps': SETTLE_BUDGET,
            'final_hover_steps': FINAL_HOVER, 'terminal_margin': TERMINAL_MARGIN})
    if maximum < 1:
        raise RuntimeError('Held conflict cannot fit the unchanged final20/settle/margin budget')
    physical = _context(solvers, raw, checker)
    anchors = tuple(np.asarray(value).copy() for value in checker.attachments)
    support_ids = tuple(id(support) for support in checker._supports)
    initial_z = np.asarray([support._initial_object[2, 3] for support in checker._supports])
    before_digest = hashlib.sha256(b''.join(physical['state'][key].tobytes() for key in sorted(physical['state']))).hexdigest()
    poses, fixture = fixed_middle_poses(raw, physical)
    report['fixed_fixture'] = fixture
    report['held_start_state_sha256'] = before_digest
    begin('carry_conflict_geometry_query')
    if args.fixture_source == 'native':
        if 'require_safe_pair' not in inspect.signature(plan).parameters:
            raise RuntimeError('Complete plan() must expose require_safe_pair=False for read-only unsafe candidate acquisition')
        import sapien
        outbound = plan('carry_conflict_outbound', tuple(sapien.Pose(pose[:3], pose[3:]) for pose in poses),
            closed=True, profile='transport', budget_phase='retreat', require_safe_pair=False)
        trips, receipts = zip(*(reverse_round_trip(source) for source in outbound.trajectories))
        report['traffic_trajectory_provenance'] = {'mode': 'native-outbound-derived-return',
            'backend': args.backend, 'arms': list(receipts),
            'native_outbound_artifacts': 'carry_conflict_outbound-arm*-native-result.json and native parameterization files',
            'no_analytic_fallback': True}
        for receipt in receipts:
            if not receipt['sampled_derivative_gate_passed']:
                save_json(output/'traffic-derived-derivative-failure.json', report['traffic_trajectory_provenance'])
                raise RuntimeError('Derived time-reversal/hold join exceeds unchanged velocity/acceleration limits')
        left, right = trips
    else:
        left, right, analytic = _load_fixed(physical, maximum)
        report['traffic_trajectory_provenance'] = {'mode': 'archived-analytic-coordinator-only',
            'prefix_backend': args.backend, 'traffic_backend': None,
            'outbound_is_native_planner_output': False, 'return_is_native_planner_output': False,
            'archived_analytic_receipt': analytic, 'native_backend_conflict_qualification': 'not_tested'}
    _unchanged(physical, raw, checker)
    np.savez(output/'traffic-source-trajectories.npz', left_time=left.times_s,
        left_position=left.positions_rad, right_time=right.times_s,
        right_position=right.positions_rad,
        gripper_commands=np.array([left.gripper_command, right.gripper_command]))
    sync = build_hold_schedule(left, right, control_timestep_s=.05, max_steps=maximum)
    save_traffic_plan(output/'traffic-plan.npz', left, right, sync)
    qualification = _qualify(left, right, checker, maximum)
    report['qualification'] = qualification
    save_json(output/'traffic-qualification.json', qualification)
    _unchanged(physical, raw, checker)
    if qualification['status'] != 'passed':
        report['controlled_outcome'] = 'fixture_not_qualified'
        raise RuntimeError(f"Selected backend paths do not qualify the fixed held fixture: {qualification.get('reason')}")
    begin('carry_conflict_synchronous_rejected')
    rejected = execute_dual_held_move(capture, sync, checker, targets)
    report['blocked_execution'] = rejected
    _unchanged(physical, raw, checker)
    if (rejected.get('status') != 'rejected' or rejected.get('reason') != 'candidate_collision'
            or rejected.get('environment_steps') != 0 or int(as_numpy(raw.elapsed_steps)[0]) != prefix):
        raise RuntimeError('Unsafe synchronous candidate was not rejected before physics')
    delays = qualification['selected_delay_ticks']
    selected = build_hold_schedule(*(delay_trajectory(source, int(delay), .05)
        for source, delay in zip((left, right), delays, strict=True)),
        control_timestep_s=.05, max_steps=maximum)
    final_check = validate_schedule(selected, *physical['starts'], checker)
    report['coordinated_validation'] = final_check
    _unchanged(physical, raw, checker)
    if final_check['status'] != 'passed':
        raise RuntimeError('Selected fixed-delay candidate is no longer collision-free')
    save_traffic_plan(output/'traffic-plan-coordinated.npz', left, right, sync, selected)
    report['selected_schedule'] = {'planned_steps': selected.num_steps, 'delay_ticks': delays,
        'source_positions_changed': False, 'source_time_scale_changed': False,
        'gripper_commands': [-1., -1.]}
    begin('carry_conflict_execute')
    executed = execute_dual_held_move(capture, selected, checker, targets,
        hold_steps=FINAL_HOVER, settle_max_steps=SETTLE_BUDGET, stable_steps=3)
    finish('carry_conflict_execute', executed)
    report['held_physics_substep_audit'] = held_substep_audit(substeps, references)
    hover = [int(executed['initial_elapsed_steps'][0])+row['step']
             for row in executed['trace'] if row['phase'] == 'hover']
    report['physics_final_hover_joint_audit'] = hold_joint_substep_audit(substeps, hover)
    traffic_rows = [row for row in substeps if row['phase'] == 'carry_conflict_execute']
    if not traffic_rows:
        raise RuntimeError('No actual held-traffic substeps were recorded')
    minimum_height = np.min(np.asarray([np.asarray(row['object_poses'])[:, 2]-initial_z
                                       for row in traffic_rows]), axis=0)
    report['traffic_original_attachment_height_gain_m'] = {'minimum_at100Hz': minimum_height.tolist(),
        'minimum_required_m': [.10, .10], 'original_actor_origin_z_m': initial_z.tolist()}
    anchors_unchanged = (checker.attached and tuple(id(support) for support in checker._supports) == support_ids
        and all(np.array_equal(actual, original) for actual, original in zip(checker.attachments, anchors)))
    final_ticks = ticks[-FINAL_HOVER:]
    final_grasps = len(final_ticks) == FINAL_HOVER and all(obj['is_grasping']
        for tick in final_ticks for obj in tick['objects'])
    checks = {'independent_paths_safe_and_payload_conflict_qualified': qualification['status'] == 'passed',
        'unsafe_synchronous_zero_step_rejection': rejected['environment_steps'] == 0,
        'same_original_attachment_capture': anchors_unchanged and report['attachment_capture']['count'] == 1,
        'actual100Hz_payload_slip_within_existing_limits': report['held_physics_substep_audit']['status'] == 'passed',
        'actual_final20_hover_joints_stable': report['physics_final_hover_joint_audit']['status'] == 'passed'
            and len(hover) == FINAL_HOVER,
        'both_still_grasped_final20_ticks': bool(final_grasps),
        'original_grasp_height_gain_at_least10cm': bool(np.all(minimum_height >= .10)),
        'all_control_steps_before200': int(as_numpy(raw.elapsed_steps)[0]) < 200}
    report['controlled_checks'] = checks
    report['controlled_outcome'] = 'passed' if all(checks.values()) else 'failed'
    report.update(status=report['controlled_outcome'], controlled_200_step_complete=all(checks.values()),
        official_200_step_complete=False, project_complete_physical_closure=False,
        physical_task_outcome='controlled_held_traffic_and20tick_hover', stage='controlled_finished')
    return False


def self_test() -> dict[str, Any]:
    """CPU-only reversal, derivative boundaries, fixture identity and coordinator tests."""
    names = tuple(f'panda_joint{index}' for index in range(1, 8))
    times = np.linspace(0., 1., 101)
    u = times
    law = 10*u**3-15*u**4+6*u**5
    positions = np.zeros((len(times), 7)); positions[:, 0] = .1*law
    outbound = ArmTrajectory('left', names, times, positions, -1.)
    trip, receipt = reverse_round_trip(outbound)
    assert len(trip.times_s) == 201 and trip.duration_s == 2.
    assert np.array_equal(trip.positions_rad[:101], outbound.positions_rad)
    assert np.array_equal(trip.positions_rad[101:], outbound.positions_rad[-2::-1])
    assert np.array_equal(trip.times_s[101:], 2.-outbound.times_s[-2::-1])
    assert receipt['sampled_derivative_gate_passed'] and not receipt['return_is_native_planner_output']
    assert not trip.positions_rad.flags.writeable and not outbound.times_s.flags.writeable
    irregular = ArmTrajectory('left', names, [0., .13, .6], np.zeros((3, 7)), -1.)
    irregular_trip, _ = reverse_round_trip(irregular)
    assert np.allclose(irregular_trip.times_s, [0., .13, .6, 1.07, 1.2], atol=1e-15, rtol=0)
    abrupt = ArmTrajectory('left', names, [0., .05], np.array([[0.]*7, [.1]*7]), -1.)
    _, bad = reverse_round_trip(abrupt)
    assert not bad['sampled_derivative_gate_passed']
    rejected = 0
    for invalid in (None, ArmTrajectory('left', names, [0.], np.zeros((1, 7)), -1.),
                    ArmTrajectory('left', names, [0., 1.], np.zeros((2, 7)), 1.)):
        try:
            reverse_round_trip(invalid)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError('Invalid held outbound was accepted')
    fixture = json.loads(METADATA.read_text())
    assert _sha(FIXTURE) == fixture['fixture_npz_sha256']
    assert fixture['fixture_source_sha256'] == _sha(HERE/'held_conflict_fixture.py')
    from types import SimpleNamespace as NS
    physical_names = (*names, 'panda_finger_joint1', 'panda_finger_joint2')
    actual_urdf = HERE.parents[2]/'assets/robots/panda/panda_v3.urdf'
    raw = NS(agent=NS(agents=[NS(urdf_path=str(actual_urdf), robot=NS(get_active_joints=lambda: [
        NS(get_name=lambda name=name: name) for name in physical_names])) for _ in (0, 1)]))
    with np.load(FIXTURE, allow_pickle=False) as data:
        bases = data['base_poses'].copy()
    pose_context = {'names': (names, names), 'keys': tuple(fixture['action_keys']),
                    'base_poses': bases, 'full_states': np.zeros((2, 9))}
    fixed_poses, source = fixed_middle_poses(raw, pose_context)
    assert fixed_poses.shape == (2, 7) and np.allclose(np.linalg.norm(fixed_poses[:, 3:], axis=1), 1.)
    assert source['goal_pose_identical_for_both_backends']
    from .conflict_coordination import self_test as coordinator_test
    coordinator = coordinator_test()
    assert coordinator['status'] == 'passed' and coordinator['environment_steps'] == 0
    return {'status': 'passed', 'outbound_shape': [101, 7], 'round_trip_shape': [201, 7],
        'irregular_round_trip_time_shape': [5], 'rejected_inputs': rejected,
        'source_times_and_positions_preserved': True, 'derived_return_is_not_native': True,
        'coordinator_cpu_status': coordinator['status'], 'fixture_identity_verified': True,
        'fixed_world_TCP_pose_shape': [2, 7],
        'environment_steps': 0, 'gpu_initialized': False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--backend', choices=('curobo', 'mplib'), default='curobo')
    parser.add_argument('--fixture-source', choices=('native', 'archived-analytic'), default='native')
    parser.add_argument('--seed', type=int, default=401)
    parser.add_argument('--planner-seed', type=int, default=0)
    parser.add_argument('--approach-recipe', choices=('standard-8cm', 'compact-4cm'), default='compact-4cm')
    parser.add_argument('--worker-script', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), allow_nan=False)); return 0
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if (not gpu.startswith('GPU-') or ',' in gpu or args.output_dir is None
            or args.seed < 0 or args.planner_seed < 0
            or any(os.environ.get(f'MANI_SKILL_L{i}') for i in (1, 2, 3))):
        parser.error('One explicit GPU UUID, L0, nonnegative seeds and unused output directory required')
    if args.worker_script and (args.backend != 'curobo' or not args.worker_script.is_file()):
        parser.error('--worker-script requires cuRobo and an existing worker file')
    faulthandler.enable(); resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    from .validate_multitask import termination_handler
    from .validate_curobo_complete import execution_configuration, native_run
    signal.signal(signal.SIGTERM, termination_handler)
    args.task, args.mode = TASK, 'positive'
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    archive = output/'source'; archive.mkdir()
    sources = list(HERE.glob('*.py'))+[ROOT/f'infrastructure/{name}' for name in
        ('curobo_tig_worker.py', 'curobo_tig_session_worker.py', 'curobo_tig_payload_worker.py',
         'curobo_tig_fast_payload_worker.py', 'curobo_tig_task_payload_worker.py')]
    if args.worker_script:
        sources.append(args.worker_script.resolve())
    sources = list(dict.fromkeys(sources))
    for source in sources:
        shutil.copy2(source, archive/source.name)
    shutil.copy2(FIXTURE, output/'fixed-fixture-source.npz')
    shutil.copy2(METADATA, output/'fixed-fixture-source.json')
    report: dict[str, Any] = {'status': 'running', 'mode': 'positive', 'task': TASK,
        'backend': args.backend, 'fixture_source': args.fixture_source, 'seed': args.seed,
        'planner_seed': args.planner_seed, 'approach_recipe': args.approach_recipe,
        'worker_script': str(args.worker_script.resolve()) if args.worker_script else (
            str(ROOT/'infrastructure/curobo_tig_task_payload_worker.py') if args.backend == 'curobo' else None),
        'started_at': datetime.now(timezone.utc).isoformat(), 'gpu_uuid': gpu, 'max_episode_steps': 200,
        'scope': 'actual confirmed held traffic and common20tick hover; separate from official placement',
        'execution_configuration': execution_configuration(TASK, approach_recipe=args.approach_recipe),
        'official_200_step_complete': False, 'project_complete_physical_closure': False,
        'controlled_200_step_complete': False, 'physical_pose_assignment': False,
        'physical_attachment_created': False, 'physical_collision_masks_modified': False,
        'native_timeopt_payload_inertia_applied': False,
        'source_sha256': {str(source): _sha(source) for source in sources},
        'official_tig': git_identity(ROOT/'The-Imitator-Game'),
        'official_curobo': git_identity(ROOT/'infrastructure/backends/curobo')}
    save_json(output/'report.json', report)
    started = time.monotonic()
    try:
        native_run(args, output, report, phase_hook=conflict_hook)
        contacts = json.loads((output/'all-robot-contacts.json').read_text())
        steps = report.get('total_environment_steps', 0)
        full_contact_check = len(contacts) == 5*steps and not any(contact['physical_contact']
            and not contact['allowed'] for row in contacts for contact in row['contacts'])
        report['full_robot100Hz_contact_audit'] = {'status': 'passed' if full_contact_check else 'failed',
            'physical_substeps': len(contacts), 'expected_substeps': 5*steps,
            'physical_contact_policy': 'unchanged complete runner all-robot/world policy'}
        if not full_contact_check:
            report.update(status='failed', controlled_200_step_complete=False)
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    report.update(wall_s=time.monotonic()-started, completed_at=datetime.now(timezone.utc).isoformat())
    save_json(output/'report.json', report)
    print(json.dumps({'status': report['status'], 'report': str(output/'report.json')}), flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
