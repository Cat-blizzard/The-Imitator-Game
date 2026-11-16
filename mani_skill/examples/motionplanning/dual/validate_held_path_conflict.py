"""Actual dual grasp, one attachment capture, held conflict and initial waiting.

Prefix is the frozen compact manipulation prefix with its native TOPP arrays
and analytic retiming. Controlled success means both objects remain grasped
through traffic and a common twenty-tick hover, not basket placement success.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import time
import traceback
from typing import Any
import numpy as np
from .validate_grasp import TASK, grasp_poses, json_value, recording_summary, save_report
from .validate_banana import banana_grasp_poses
from .validate_sync import continue_success_only, applied_action, held_substep_audit, hold_joint_substep_audit
from .validate_sync_compact import (compact_configuration, compact_phase_limits, compact_set_phase_limits,
    compact_urdf_velocity_limits, compact_topp_reconstruction, compact_retime_pair,
    compact_kinematic_audit, compact_route_positions, SOLVER_LIMIT_FRACTION)
from .validate_path_conflict import interarm_contact_rows, _state_arrays, _assert_unchanged
from .combined_collision import validate_schedule, NativeDualCollisionChecker
from .conflict_coordination import search_start_delays
from .trajectory_schedule import ArmTrajectory, build_hold_schedule

HERE=Path(__file__).resolve().parent
ROOT=Path(__file__).resolve().parents[7]
ACTOR_NAMES=('apple_out','banana_out')
RESERVE_SETTLE, FINAL_HOVER, TERMINAL_MARGIN = 5,20,1


def physical_contact_points(points: Any) -> bool:
    return any(p['separation_m']<=0 or any(x!=0 for x in p['impulse_Ns']) for p in points)


def classify_interarm_contacts(rows: list[dict[str,Any]]) -> list[dict[str,Any]]:
    for row in rows:
        actual=physical_contact_points(row['points'])
        row.update(physical_contact=actual,
            positive_separation_zero_impulse_proximity=bool(row['points']) and not actual)
    return rows


def object_contact_rows(contacts: Any, identities: dict[Any,dict[str,Any]],
                        support_released: tuple[Any,Any]) -> list[dict[str,Any]]:
    """All manifolds involving a target, including zero impulses and both orders."""
    rows=[]
    for contact in contacts:
        sides=[dict(identities.get(body.entity,{'kind':'unknown','actor':str(getattr(body.entity,'name','unknown'))}))
               for body in contact.bodies]
        targets=[side['target_index'] for side in sides if side.get('target_index') is not None]
        if len(sides)!=2 or not targets: continue
        target_side=next(i for i,side in enumerate(sides) if side.get('target_index') is not None)
        target=sides[target_side]; other=sides[1-target_side]; index=target['target_index']
        if (other.get('robot_index')==index and other.get('link_name') in ('panda_leftfinger','panda_rightfinger')):
            classification='own_finger_allowed'
        elif other.get('actor')=='table' and support_released[index] is not True:
            classification='initial_table_support'
        else:
            classification='forbidden'
        points=[{'position_m':np.asarray(p.position,dtype=float).tolist(),
                 'normal':np.asarray(p.normal,dtype=float).tolist(),
                 'impulse_Ns':np.asarray(p.impulse,dtype=float).tolist(),
                 'separation_m':float(p.separation)} for p in contact.points]
        physical_contact=physical_contact_points(points)
        rows.append({'bodies':sides,'target_indices':targets,'class':classification,
                     'support_released_snapshot':list(support_released),'points':points,
                     'physical_contact':physical_contact,
                     'positive_separation_zero_impulse_proximity':bool(points) and not physical_contact})
    return rows


def payload_related_collision(validation: dict[str,Any], checker: Any) -> bool:
    from .held_conflict_fixture import _cross_payload_witness
    first=validation.get('first_collision')
    if not first: return False
    return bool(_cross_payload_witness(first['collisions'],checker))


def require_robot_only_safe(validation: dict[str,Any]) -> None:
    if validation.get('status')!='passed':
        raise RuntimeError('Synchronous robot-only route is unsafe; payload contribution is not isolated')


def save_traffic_plan(path: Path, left: ArmTrajectory, right: ArmTrajectory, sync: Any, selected: Any=None) -> None:
    arrays={'left_time':left.times_s,'left_position':left.positions_rad,
            'right_time':right.times_s,'right_position':right.positions_rad,
            'gripper_commands':np.array([left.gripper_command,right.gripper_command]),
            'sync_target_times':sync.target_times_s,'sync_left_commands':sync.left_commands,'sync_right_commands':sync.right_commands}
    if selected is not None:
        arrays.update(selected_left_time=selected.trajectories[0].times_s,
            selected_left_position=selected.trajectories[0].positions_rad,
            selected_right_time=selected.trajectories[1].times_s,
            selected_right_position=selected.trajectories[1].positions_rad,
            selected_target_times=selected.target_times_s,selected_left_commands=selected.left_commands,
            selected_right_commands=selected.right_commands)
    np.savez(path,**arrays)


def native_run(args: argparse.Namespace, output: Path, report: dict[str, Any]) -> None:
    import gymnasium as gym
    import imageio.v2 as imageio
    import sapien
    import torch
    import mani_skill.envs
    from mani_skill.utils.wrappers.record import RecordEpisode
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
    from .trajectory_execution import as_numpy
    from .dual_payload_collision import DualPayloadCollisionChecker
    from .dual_arm_execution import (plan_dual_pose, execute_dual_reach, close_dual_and_confirm,
                                    execute_dual_held_move, release_dual_and_confirm, verify_dual_placement)
    from .trajectory_schedule import ArmTrajectory, build_hold_schedule
    from mani_skill.envs.tasks.tabletop.dual_tasks._008_pick_apple_banana_baskets import world_aabb_from_local_mesh

    env, raw, original_hook = None, None, None
    checker = None
    attachment_count = 0
    frames, ticks, substeps, keys = [], [], [], []
    report["fault_events"] = []
    try:
        base = gym.make(TASK, obs_mode="none", reward_mode="none", control_mode="pd_joint_pos", sim_backend="cpu",
                        render_backend="gpu", render_mode="rgb_array", max_episode_steps=args.max_episode_steps)
        env = RecordEpisode(base, output_dir=str(output), trajectory_name="trajectory", save_video=False,
                            save_on_reset=False, clean_on_close=False, record_reward=False, record_env_state=True,
                            source_type="motionplanning", source_desc=f"compact synchronized apple and banana; route={args.route}; profile={args.execution_profile}; mode={args.mode}")
        env.reset(seed=args.seed)
        raw = env.unwrapped
        agents, targets = (raw.left_agent, raw.right_agent), (raw.apple_out, raw.banana_out)
        keys = list(raw.agent.agents_dict)
        if len(keys) != 2 or any(raw.agent.agents_dict[k] is not a for k, a in zip(keys, agents)):
            raise ValueError("Physical left/right action identity mismatch")
        report.update(backend={"physics": raw.backend.sim_backend, "render": raw.backend.render_backend},
                      control_timestep_s=float(raw.control_timestep), stages=[], action_key_order=keys)

        identities={obj.entity:{'kind':'robot','robot_index':i,'link_name':link.name,'actor':link.name}
                    for i,agent in enumerate(agents) for link in agent.robot.get_links() for obj in link._objs}
        for name,actor in raw.scene.actors.items():
            for obj in actor._objs:
                if obj not in identities:
                    identities[obj]={'kind':'actor','actor':name}
        for index,actor in enumerate(targets):
            for obj in actor._objs:
                identities[obj]={'kind':'payload','actor':ACTOR_NAMES[index],'target_index':index}
        robot_identities={entity:(value['robot_index'],value['link_name']) for entity,value in identities.items()
                          if value.get('kind')=='robot'}

        def measurement() -> dict[str,Any]:
            objects=[]
            for name,agent,actor in zip(ACTOR_NAMES,agents,targets):
                forces=[as_numpy(raw.scene.get_pairwise_contact_forces(link,actor))[0]
                        for link in (agent.finger1_link,agent.finger2_link)]
                objects.append({'actor':name,'pose':as_numpy(actor.pose.raw_pose)[0].tolist(),
                    'linear_velocity_m_s':as_numpy(actor.linear_velocity)[0].tolist(),
                    'angular_velocity_rad_s':as_numpy(actor.angular_velocity)[0].tolist(),
                    'is_grasping':bool(as_numpy(agent.is_grasping(actor))[0]),
                    'finger_contact_forces_N':[f.tolist() for f in forces],
                    'finger_contact_force_norms_N':[float(np.linalg.norm(f)) for f in forces]})
            return {'left_qpos':as_numpy(agents[0].robot.get_qpos())[0].tolist(),
                    'right_qpos':as_numpy(agents[1].robot.get_qpos())[0].tolist(),
                    'tcp_poses':[as_numpy(a.tcp.pose.raw_pose)[0].tolist() for a in agents],'objects':objects}

        class Capture:
            unwrapped=raw
            phase='preparation'
            observe_substeps=True
            pending_control_step=0
            pending_substep=0
            def frame(self) -> None:
                frame=as_numpy(env.render())
                if frame.ndim==4 and frame.shape[0]==1: frame=frame[0]
                if frame.ndim!=3 or frame.shape[-1]!=3 or frame.dtype!=np.uint8: raise ValueError('Expected HWC RGB uint8')
                frames.append(frame.copy())
            def step(self,action: dict[str,Any]) -> tuple[Any,...]:
                actual,inject=applied_action(action,mode='positive',phase=self.phase,keys=keys)
                assert not inject
                self.pending_control_step=int(as_numpy(raw.elapsed_steps)[0])+1
                self.pending_substep=0
                if self.pending_control_step>200: raise RuntimeError('Never step beyond native200')
                first=len(substeps)
                obs,reward,terminated,truncated,info=env.step(actual)
                done,continued=continue_success_only(terminated,truncated,info)
                ticks.append({'phase':self.phase,'environment_step':int(as_numpy(raw.elapsed_steps)[0]),
                    'requested_actions':json_value(action),'actions':json_value(actual),'fault_injected':False,
                    'native_terminated':bool(as_numpy(terminated).reshape(-1)[0]),
                    'native_truncated':bool(as_numpy(truncated).reshape(-1)[0]),
                    'success_only_termination_continued':continued,'official_evaluation':json_value(info),**measurement()})
                self.frame()
                if any(any(c['physical_contact'] for c in s['interarm_contacts']) or any(c['class']=='forbidden' and c['physical_contact']
                       for c in s['object_contact_rows']) for s in substeps[first:]):
                    raise RuntimeError('Forbidden physical contact manifold observed; stop both arms')
                return obs,reward,done,truncated,info
        captured=Capture()
        original_hook=raw._after_simulation_step
        def record_substep() -> None:
            original_hook()
            captured.pending_substep+=1
            supports=tuple(s.support_released for s in checker._supports) if checker is not None and checker.attached else (None,None)
            contacts=raw.scene.px.get_contacts()
            substeps.append({'substep':len(substeps)+1,'phase':captured.phase,'substep_index':captured.pending_substep,
                'control_step':captured.pending_control_step,'sim_timestep_s':float(raw.sim_timestep),
                'object_poses':[as_numpy(a.pose.raw_pose)[0].tolist() for a in targets],
                'tcp_poses':[as_numpy(a.tcp.pose.raw_pose)[0].tolist() for a in agents],
                'left_qpos':as_numpy(agents[0].robot.get_qpos())[0].tolist(),
                'right_qpos':as_numpy(agents[1].robot.get_qpos())[0].tolist(),
                'left_qvel':as_numpy(agents[0].robot.get_qvel())[0].tolist(),
                'right_qvel':as_numpy(agents[1].robot.get_qvel())[0].tolist(),
                'object_linear_velocities':[as_numpy(a.linear_velocity)[0].tolist() for a in targets],
                'object_angular_velocities':[as_numpy(a.angular_velocity)[0].tolist() for a in targets],
                'objects':measurement()['objects'],
                'interarm_contacts':classify_interarm_contacts(interarm_contact_rows(contacts,robot_identities)),
                'object_contact_rows':object_contact_rows(contacts,identities,supports)})

        from .dual_arm_execution import _canonical
        arms = _canonical(captured)
        report["urdf_joint_velocity_checks"] = [compact_urdf_velocity_limits(a.urdf_path, arm.names,
            max(args.joint_vel_limit, report["execution_configuration"]["approach_joint_vel_limit_rad_s"]))
            for a, arm in zip(agents, arms)]
        if (not np.isclose(raw.control_timestep, .05, rtol=0, atol=1e-12)
                or not np.isclose(raw.sim_timestep, .01, rtol=0, atol=1e-12)):
            raise ValueError("Compact runs require the unchanged 20Hz control/100Hz physics clock")
        report["physics_timestep_s"] = float(raw.sim_timestep)
        raw._after_simulation_step = record_substep
        captured.frame()
        initial_hold = {k: np.r_[as_numpy(a.robot.get_qpos())[0, :7], 1].astype(np.float32) for k, a in zip(keys, agents)}
        for _ in range(10):
            _, _, done, truncated, _ = captured.step(initial_hold)
            if done or bool(as_numpy(truncated).reshape(-1)[0]): raise RuntimeError("Environment ended during preparation")
        report["preparation_steps"] = 10
        class AuditedSolver(PandaArmMotionPlanningSolver):
            """Retain original native TOPP derivative arrays before conversion."""
            def setup_planner(self) -> Any:
                planner = super().setup_planner()
                native_topp = planner.TOPP

                def traced_topp(path: Any, step: float = .1, verbose: bool = False) -> Any:
                    raw_output = native_topp(path, step, verbose)
                    payload = {"native_path": np.asarray(path).copy(),
                               "solver_velocity_limits": np.asarray(planner.joint_vel_limits).copy(),
                               "solver_acceleration_limits": np.asarray(planner.joint_acc_limits).copy()}
                    for field, value in zip(("time", "position", "velocity", "acceleration", "duration"), raw_output):
                        if value is not None:
                            payload["raw_" + field] = np.asarray(value).copy()
                    # Persist the original output even if reconstruction rejects.
                    np.savez(output / f"{captured.phase}-arm{self.multi_robot_id}-native-topp-raw.npz", **payload)
                    self.last_native_parameterization = compact_topp_reconstruction(
                        path, planner.joint_vel_limits, planner.joint_acc_limits, step, raw_output)
                    return raw_output

                planner.TOPP = traced_topp
                return planner

            def move_to_pose_with_screw(self, pose: Any, dry_run: bool = False,
                                        refine_steps: int = 0, other_gripper_state: Any = None) -> Any:
                if not dry_run:
                    raise ValueError("Compact execution requires plan-only solver calls")
                self.last_native_parameterization = None
                result = super().move_to_pose_with_screw(pose, dry_run=True,
                    refine_steps=refine_steps, other_gripper_state=other_gripper_state)
                self.last_native_result = result
                return result
        solvers = [AuditedSolver(env, base_pose=a.robot.pose, multi_robot_id=i, debug=False,
                    vis=False, visualize_target_grasp_pose=False, print_env_info=False,
                    joint_vel_limits=report["execution_configuration"]["solver_joint_vel_limit_rad_s"],
                    joint_acc_limits=report["execution_configuration"]["solver_joint_acc_limit_rad_s2"])
                   for i, a in enumerate(agents)]
        checker = DualPayloadCollisionChecker.from_solvers(raw, *solvers, targets=targets)
        report["initial_collision_coverage"] = json_value(checker.coverage)

        def begin(name: str) -> None:
            report["stage"] = captured.phase = name
            save_report(output / "report.json", report)

        def finish(name: str, result: dict[str, Any]) -> bool:
            report["stages"].append({"name": name, **result})
            if result["status"] != "passed": report.update(status="failed", reason=f"{name}:{result['status']}")
            save_report(output / "report.json", report)
            return result["status"] == "passed"

        def plan(name: str, poses: tuple[Any, Any], *, closed: bool) -> Any:
            phase_limits = compact_phase_limits(report["execution_configuration"], name)
            compact_set_phase_limits(solvers, phase_limits)
            report.setdefault("phase_limit_configurations", []).append(phase_limits)
            before = int(as_numpy(raw.elapsed_steps)[0])
            raw_schedule = plan_dual_pose(*solvers, poses, gripper_commands=(-1, -1) if closed else (1, 1))
            if int(as_numpy(raw.elapsed_steps)[0]) != before: raise RuntimeError("Planning advanced environment")
            raw_results = tuple(solver.last_native_result if target is not None else None for solver, target in zip(solvers, poses))
            traces = tuple(solver.last_native_parameterization if target is not None else None for solver, target in zip(solvers, poses))
            retimed, retiming = compact_retime_pair(raw_results, traces,
                velocity_limit=phase_limits["nominal_velocity_limit_rad_s"],
                acceleration_limit=phase_limits["nominal_acceleration_limit_rad_s2"])
            trajectories = tuple(source if result is None else ArmTrajectory(source.action_key, source.joint_names,
                                  result["time"], result["position"], source.gripper_command)
                                 for source, result in zip(raw_schedule.trajectories, retimed))
            schedule = build_hold_schedule(*trajectories, control_timestep_s=raw_schedule.control_timestep_s)
            report.setdefault("phase_time_parameterization", []).append({"phase": name, **retiming,
                "raw_scheduled_steps": raw_schedule.num_steps, "retimed_scheduled_steps": schedule.num_steps})
            derivatives = []
            derivative_arrays = {}
            for index, (solver, target) in enumerate(zip(solvers, poses)):
                if target is None:
                    derivatives.append({"status": "passed", "stationary_hold": True})
                    continue
                result = retimed[index]
                audit = compact_kinematic_audit(result,
                    velocity_limit=phase_limits["nominal_velocity_limit_rad_s"],
                    acceleration_limit=phase_limits["nominal_acceleration_limit_rad_s2"])
                audit.update(solver_velocity_limit_rad_s=solver.joint_vel_limits,
                    solver_acceleration_limit_rad_s2=solver.joint_acc_limits,
                    solver_limit_fraction=SOLVER_LIMIT_FRACTION,
                    source_timestamp_scaling_applied=retiming["source_timestamp_scaling_applied"],
                    raw_native_derivative_audit=compact_kinematic_audit(raw_results[index],
                        velocity_limit=phase_limits["nominal_velocity_limit_rad_s"],
                        acceleration_limit=phase_limits["nominal_acceleration_limit_rad_s2"]))
                derivatives.append(audit)
                arm_name = "left" if index == 0 else "right"
                derivative_arrays[f"{arm_name}_velocity"] = np.asarray(result["velocity"])
                derivative_arrays[f"{arm_name}_acceleration"] = np.asarray(result["acceleration"])
                for field in ("time", "position", "velocity", "acceleration", "duration"):
                    derivative_arrays[f"{arm_name}_raw_{field}"] = np.asarray(raw_results[index][field])
                    derivative_arrays[f"{arm_name}_retimed_{field}"] = np.asarray(result[field])
                for field in ("native_path", "path_knots_s", "path_coefficients", "grid_s", "grid_speed_squared",
                              "grid_acceleration", "grid_time", "solver_velocity_limits", "solver_acceleration_limits"):
                    derivative_arrays[f"{arm_name}_{field}"] = traces[index][field]
            report.setdefault("kinematic_audits", []).append({"phase": name, "phase_limits": phase_limits, "arms": derivatives})
            np.savez(output / f"{name}-plan.npz", **derivative_arrays, left_time=schedule.trajectories[0].times_s,
                     right_time=schedule.trajectories[1].times_s, left_position=schedule.trajectories[0].positions_rad,
                     right_position=schedule.trajectories[1].positions_rad, target_times=schedule.target_times_s,
                     left_commands=schedule.left_commands, right_commands=schedule.right_commands)
            if any(audit["status"] != "passed" for audit in derivatives):
                raise RuntimeError(f"Native planner velocity/acceleration limit violation in {name}")
            report.setdefault("planning", []).append({"phase": name, "phase_limits": phase_limits,
                "environment_steps": 0, "scheduled_steps": schedule.num_steps,
                "source_durations_s": [t.duration_s for t in schedule.trajectories],
                "target_tcp_poses": [None if p is None else np.r_[p.p, p.q].tolist() for p in poses]})
            return schedule

        grasp_pairs = (grasp_poses(raw), banana_grasp_poses(raw))
        for index, name, contact in ((0, "pregrasp", False), (1, "descend", True)):
            begin(name)
            schedule = plan(name, (grasp_pairs[0][index], grasp_pairs[1][index]), closed=False)
            if not finish(name, execute_dual_reach(captured, schedule, checker, allow_target_contacts=(contact, contact))): return
        begin("close_and_confirm")
        if not finish("close_and_confirm", close_dual_and_confirm(captured, checker, targets)): return
        begin("attach_collision_prediction")
        checker.attach()
        attachment_count += 1
        assert attachment_count == 1
        references = np.asarray(checker.attachments).copy()
        report["attachment_T_tcp_objects"] = references.tolist()
        report["attachment_capture"] = {"capture_count":attachment_count,"environment_step":int(as_numpy(raw.elapsed_steps)[0]),
            "T_tcp_objects":references.tolist(),"world_object_poses":[as_numpy(a.pose.raw_pose)[0].tolist() for a in targets],
            "world_tcp_poses":[as_numpy(a.tcp.pose.raw_pose)[0].tolist() for a in agents],
            "initial_actor_origin_z_m":[float(as_numpy(a.pose.p)[0,2]) for a in targets],
            "both_grasps_confirmed_by_previous_phase":True,"physical_attachment_created":False}
        report["attachment_initial_coverage"] = json_value(checker.coverage)
        baskets = (raw.basket_apple, raw.basket_banana)
        release_offsets = report["execution_configuration"]["release_tcp_offsets_basket_m"]
        release_xyz = [as_numpy(b.pose.p)[0] + offset for b, offset in zip(baskets, release_offsets)]
        route = compact_route_positions([as_numpy(a.tcp.pose.p)[0] for a in agents], release_xyz, route=args.route)
        original_q = [as_numpy(a.tcp.pose.q)[0].copy() for a in agents]
        carry_q = [original_q[0], (sapien.Pose(q=[2**-.5, 0, 0, 2**-.5]) * sapien.Pose(q=original_q[1])).q]
        report["compact_route_world_positions_m"] = json_value(route)
        report["selected_release_tcp_xyz_m"] = [p.tolist() for p in release_xyz]
        lift_poses = tuple(sapien.Pose(xyz, q) for xyz, q in zip(route["lift"], original_q))
        begin("lift_and_hold")
        schedule = plan("lift_and_hold", lift_poses, closed=True)
        captured.observe_substeps = True
        hover = report["execution_configuration"]["airborne_hover_steps"]
        lifted = execute_dual_held_move(captured, schedule, checker, targets,
            min_height_gain_m=(.10, .10), hold_steps=hover)
        if not finish("lift_and_hold", lifted): return
        if hover:
            hover_steps = [int(lifted["initial_elapsed_steps"][0]) + row["step"] for row in lifted["trace"] if row["phase"] == "hover"]
            report["physics_hold_joint_audit"] = hold_joint_substep_audit(substeps, hover_steps)
            if report["physics_hold_joint_audit"]["status"] != "passed":
                report.update(status="failed", reason="physics_hold_joint_stability"); return
        else:
            report["physics_hold_joint_audit"] = {"status": "not_applicable", "observed_hover_steps": 0,
                "reason": "execution profile omits the separate airborne diagnostic hover; old hover regression preserved"}
        report['prefix_completed_at_step']=int(as_numpy(raw.elapsed_steps)[0])
        prefix_end=report['prefix_completed_at_step']
        remaining=200-prefix_end
        report['traffic_budget']={'prefix_steps':prefix_end,'remaining_steps':remaining,
            'settle_max_steps':RESERVE_SETTLE,'final_hover_steps':FINAL_HOVER,'terminal_margin':TERMINAL_MARGIN,
            'maximum_schedule_steps':remaining-RESERVE_SETTLE-FINAL_HOVER-TERMINAL_MARGIN}
        if remaining<=RESERVE_SETTLE+FINAL_HOVER+TERMINAL_MARGIN: raise RuntimeError('No remaining traffic budget')
        from .held_conflict_fixture import build_held_fixture
        before=_state_arrays(raw.get_state_dict())
        begin('held_geometry_query')
        left,right,metadata=build_held_fixture(solvers,raw,checker,output,remaining,develop=args.develop_fixture)
        _assert_unchanged(before,raw)
        report['fixture_metadata']=metadata
        maximum=report['traffic_budget']['maximum_schedule_steps']
        sync=build_hold_schedule(left,right,control_timestep_s=.05,max_steps=maximum)
        save_traffic_plan(output/'traffic-plan.npz',left,right,sync)
        starts=(left.positions_rad[0],right.positions_rad[0])
        holds=tuple(ArmTrajectory(t.action_key,t.joint_names,[0.],[t.positions_rad[0]],-1) for t in (left,right))
        report['independent_validation']={}
        for name,pair in (('left',(left,holds[1])),('right',(holds[0],right))):
            report['independent_validation'][name]=validate_schedule(build_hold_schedule(*pair,control_timestep_s=.05,max_steps=maximum),*starts,checker)
        report['synchronous_validation']=validate_schedule(sync,*starts,checker)
        bare=NativeDualCollisionChecker.from_solvers(*solvers)
        report['robot_only_synchronous_validation']=validate_schedule(sync,*starts,bare)
        save_report(output/'report.json',report)
        require_robot_only_safe(report['robot_only_synchronous_validation'])
        if any(v['status']!='passed' for v in report['independent_validation'].values()): raise RuntimeError('Unsafe individual carried route')
        if not payload_related_collision(report['synchronous_validation'],checker): raise RuntimeError('No cross-arm payload-related synchronous collision')
        begin('synchronous_rejected')
        blocked=execute_dual_held_move(captured,sync,checker,targets)
        report['blocked_execution']=blocked
        _assert_unchanged(before,raw)
        if blocked['status']!='rejected' or blocked.get('reason')!='candidate_collision' or blocked['environment_steps']!=0:
            raise RuntimeError('Conflicting synchronous route did not stop before physical execution')
        selected,search=search_start_delays(left,right,checker,control_timestep_s=.05,
            max_delay_ticks=min(40,maximum),delay_step_ticks=5,max_steps=maximum)
        report['search']=search
        _assert_unchanged(before,raw)
        if selected is None: raise RuntimeError('No initial-wait solution inside fixed remaining budget')
        save_traffic_plan(output/'traffic-plan.npz',left,right,sync,selected)
        report['selected_schedule']={'planned_steps':selected.num_steps,'delay_ticks':search['selected_delay_ticks'],
            'source_timestamp_scale_changed':False,'source_positions_changed':False,'gripper_commands':[-1,-1]}
        begin('carry_conflict_execute')
        result=execute_dual_held_move(captured,selected,checker,targets,hold_steps=FINAL_HOVER,
            settle_max_steps=RESERVE_SETTLE,stable_steps=3)
        if not finish('carry_conflict_execute',result): return
        report['attachment_final_coverage']=json_value(checker.coverage)
        report['held_physics_substep_audit']=held_substep_audit(substeps,references)
        hover_steps=[int(result['initial_elapsed_steps'][0])+row['step']
                     for row in result['trace'] if row['phase']=='hover']
        report['physics_final_hover_joint_audit']=hold_joint_substep_audit(substeps,hover_steps)
        if report['held_physics_substep_audit']['status']!='passed' or report['physics_final_hover_joint_audit']['status']!='passed':
            raise RuntimeError('Unchanged100Hz slip/final-hover audit failed')
        attachment_z=np.asarray(report['attachment_capture']['initial_actor_origin_z_m'])
        held_rows=[s for s in substeps if s['phase']=='carry_conflict_execute']
        minimum_gain=np.min(np.array([np.array(s['object_poses'])[:,2]-attachment_z for s in held_rows]),axis=0)
        report['traffic_original_attachment_height_gain_m']={'minimum_at100Hz':minimum_gain.tolist(),
            'minimum_required_m':[.10,.10],'scope':'actor origin height relative to the unique original grasp attachment'}
        report['attachment_capture']['capture_count_at_finish']=attachment_count
        if attachment_count!=1 or np.any(minimum_gain<.10): raise RuntimeError('Original grasp height/capture contract failed')
        report['controlled_outcome']='passed'
        report['controlled_200_step_complete']=int(as_numpy(raw.elapsed_steps)[0])<200
        report['physical_task_outcome']='controlled_held_traffic_passed'
        report['status']='passed' if report['controlled_200_step_complete'] else 'failed'
        report['stage']='finished'
    finally:
        errors = report.setdefault("artifact_errors", [])
        if raw is not None:
            for name, read in (("total_environment_steps", lambda: int(as_numpy(raw.elapsed_steps)[0])),
                               ("torch_cuda_initialized", torch.cuda.is_initialized),
                               ("official_task_evaluation", lambda: json_value(raw.evaluate()))):
                try: report[name] = read()
                except Exception as exc: errors.append(f"final {name}: {type(exc).__name__}: {exc}")
            if original_hook is not None: raw._after_simulation_step = original_hook
        for filename, rows in (("ticks.jsonl", ticks), ("physics-substeps.jsonl", substeps)):
            try:
                with (output / filename).open("w") as f:
                    for row in rows: f.write(json.dumps(json_value(row), allow_nan=False) + "\n")
            except Exception as exc: errors.append(f"{filename}: {type(exc).__name__}: {exc}")
        if env is not None:
            try:
                if getattr(env, "_trajectory_buffer", None) is not None: env.flush_trajectory()
            except Exception as exc: errors.append(f"trajectory: {type(exc).__name__}: {exc}")
            finally:
                try: env.close()
                except Exception as exc: errors.append(f"close: {type(exc).__name__}: {exc}")
            try:
                actions = np.asarray([np.concatenate([t["actions"][k] for k in keys]) for t in ticks]) if ticks else None
                report["recording"] = recording_summary(output / "trajectory.h5", len(ticks), actions)
                report["recording"]["action_key_order"] = keys
            except Exception as exc: errors.append(f"record check: {type(exc).__name__}: {exc}")
        if frames:
            try:
                imageio.mimsave(output / "execution.mp4", [frames[0]] * 10 + frames + [frames[-1]] * 20, fps=20, macro_block_size=1)
                imageio.imwrite(output / "preview.png", frames[-1])
                report["video"] = {"path": str(output / "execution.mp4"), "fps": 20, "padding_frames_before": 10, "padding_frames_after": 20}
            except Exception as exc: errors.append(f"video: {type(exc).__name__}: {exc}")
        report["success_only_termination_continuation_ticks"] = sum(t["success_only_termination_continued"] for t in ticks)
        report["phase_step_counts"] = {phase: sum(t["phase"] == phase for t in ticks) for phase in dict.fromkeys(t["phase"] for t in ticks)}
        report["first_official_success_step"] = next((t["environment_step"] for t in ticks
            if bool(np.asarray(t["official_evaluation"].get("success", False)).all())), None)
        report["official_200_step_complete"] = bool(report.get("official_200_step_complete", False))
        if errors: report.update(status="failed", official_200_step_complete=False)


def self_test() -> dict[str,Any]:
    from types import SimpleNamespace as NS
    target,finger,other,table=object(),object(),object(),object()
    point=NS(position=np.zeros(3),normal=np.array([0,0,1.]),impulse=np.zeros(3),separation=0.)
    def contact(a: Any,b: Any) -> Any: return NS(bodies=[NS(entity=a),NS(entity=b)],points=[point])
    ids={target:{'kind':'payload','actor':'apple_out','target_index':0},
        finger:{'kind':'robot','robot_index':0,'link_name':'panda_leftfinger','actor':'panda_leftfinger'},
        other:{'kind':'robot','robot_index':1,'link_name':'panda_hand','actor':'panda_hand'},
        table:{'kind':'actor','actor':'table'}}
    rows=object_contact_rows([contact(target,finger),contact(other,target),contact(table,target)],ids,(False,False))
    assert [r['class'] for r in rows]==['own_finger_allowed','forbidden','initial_table_support']
    assert rows[1]['points'][0]['impulse_Ns']==[0.,0.,0.]
    assert rows[1]['physical_contact']
    point.separation=.04
    proximity=object_contact_rows([contact(other,target)],ids,(False,False))[0]
    assert not proximity['physical_contact'] and proximity['positive_separation_zero_impulse_proximity']
    point.impulse=np.array([0.,0.,1e-9])
    assert object_contact_rows([contact(other,target)],ids,(False,False))[0]['physical_contact']
    point.separation=0.;point.impulse=np.zeros(3)
    classified=classify_interarm_contacts([{'bodies':[{'robot_index':0},{'robot_index':1}],
        'points':[{'separation_m':separation,'impulse_Ns':impulse}]} for separation,impulse in
        [(.04,[0.,0.,0.]),(.001,[0.,0.,1e-9]),(0.,[0.,0.,0.]),(-.001,[0.,0.,0.])]])
    assert [row['physical_contact'] for row in classified]==[False,True,True,True]
    assert classified[0]['positive_separation_zero_impulse_proximity']
    assert object_contact_rows([contact(target,table)],ids,(True,False))[0]['class']=='forbidden'
    checker=NS(_target_names=({'scene::apple_out::0'},{'scene::banana_out::0'}))
    assert payload_related_collision({'first_collision':{'collisions':[{'collision_type':'payload_payload'}]}},checker)
    assert payload_related_collision({'first_collision':{'collisions':[{'collision_type':'articulation',
        'object_name1':'right','object_name2':'scene::apple_out::0'}]}},checker)
    assert not payload_related_collision({'first_collision':{'collisions':[{'collision_type':'articulation',
        'object_name1':'left','object_name2':'scene::apple_out::0'}]}},checker)
    assert not payload_related_collision({'first_collision':{'collisions':[{'collision_type':'articulation'}]}},checker)
    require_robot_only_safe({'status':'passed'})
    try:require_robot_only_safe({'status':'failed'})
    except RuntimeError:pass
    else:raise AssertionError('Formal robot-only failure was not rejected')
    args=argparse.Namespace(route='merge-lift',execution_profile='execution',max_episode_steps=200,
        joint_vel_limit=1.5,joint_acc_limit=2.6,approach_vel_limit=.9,approach_acc_limit=.9)
    configuration=compact_configuration(args)
    assert configuration['joint_vel_limit_rad_s']==1.5 and configuration['approach_joint_vel_limit_rad_s']==.9
    assert 101+(99-RESERVE_SETTLE-FINAL_HOVER-TERMINAL_MARGIN)+RESERVE_SETTLE+FINAL_HOVER==199
    return {'status':'passed','checks':['contact identity and zero impulses','support-before/after release',
        'payload collision classification','frozen compact configuration','strict200 reservation'],
        'environment_constructions':0,'physics_steps':0}


def main() -> int:
    faulthandler.enable(); resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--seed',type=int,default=0);parser.add_argument('--develop-fixture',action='store_true')
    parser.add_argument('--output-dir',type=Path);args=parser.parse_args()
    if args.self_test: print(json.dumps(self_test(),indent=2));return 0
    if args.develop_fixture and args.seed!=0: parser.error('Development grid is restricted to seed0')
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not re.fullmatch(r'GPU-[0-9a-fA-F-]+',gpu): parser.error('Select GPU UUID')
    if any(os.environ.get(k)=='1' for k in ('MANI_SKILL_L1','MANI_SKILL_L2','MANI_SKILL_L3')): parser.error('L0 task only')
    args.mode='positive';args.route='merge-lift';args.execution_profile='execution';args.max_episode_steps=200
    args.joint_vel_limit=1.5;args.joint_acc_limit=2.6;args.approach_vel_limit=.9;args.approach_acc_limit=.9
    output=args.output_dir or ROOT/'infrastructure/validation/dual-arm/held-path-conflict-20261116'/datetime.now(timezone.utc).strftime('dev-seed0-%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True,exist_ok=False)
    files=[p for p in HERE.glob('*.py') if p.name!='__init__.py'];sources={p.name:p.read_bytes() for p in files}
    snapshot=output/'source';snapshot.mkdir()
    for name,data in sources.items():(snapshot/name).write_bytes(data)
    report={'status':'running','started_at':datetime.now(timezone.utc).isoformat(),'seed':args.seed,'task':TASK,
        'develop_fixture':args.develop_fixture,'gpu_uuid':gpu,'scope':'controlled actual-grasp held traffic and twenty-tick hold; no placement claim',
        'execution_configuration':compact_configuration(args),'controlled_outcome':'pending','max_episode_steps':200,
        'official_default_max_episode_steps':200,'official_200_step_complete':False,
        'source_snapshot':str(snapshot),'source_sha256':{n:hashlib.sha256(d).hexdigest() for n,d in sources.items()},
        'physical_attachment_created':False,'physical_pose_assignment':False,'continuous_collision_detection':False,
        'grasp_false_policy':'existing control-frame two-consecutive-loss abort;100Hz forces and grasp flags telemetry retained without new force gate'}
    report['object_contact_semantics']={'physical_contact':'any point has separation<=0 OR nonzero impulse; zero-impulse nonpositive separation retained as contact',
        'proximity':'positive separation and all impulses zero is retained proximity telemetry, not physical contact',
        'all_pair_manifolds_retained':True,'native_fcl_collision_gate_unchanged':True}
    report['contact_monitor_revision']={'version':'uniform-object-interarm-physical-contact-v2',
        'physical_contact':'any point separation<=0 OR any nonzero impulse component',
        'changed_from_old_interarm_gate':'old gate stopped on every returned manifold, including positive separation with exactly zero impulse; this entry now retains those as proximity telemetry',
        'native_fcl_and_physics_contact_offsets_unchanged':True,'all_original_interarm_manifold_fields_retained':True,
        '03_nonzero_impulse_contact_would_still_abort':True}
    save_report(output/'report.json',report);started=time.monotonic()
    try:native_run(args,output,report)
    except Exception as exc:report.update(status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    report.update(completed_at=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-started)
    if any(hashlib.sha256((HERE/n).read_bytes()).hexdigest()!=digest for n,digest in report['source_sha256'].items()):
        report.update(status='failed',source_identity_violation=True)
    save_report(output/'report.json',report);print(json.dumps({'status':report['status'],'report':str(output/'report.json')}),flush=True)
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
