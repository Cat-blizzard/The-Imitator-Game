"""Official-route serial baseline with the shared M1 physical recorder.

The official _pick_and_place functions define every grasp/transport waypoint.
Their original follow_path commands one arm and holds the measured peer. Added
checks/confirmation waits, task-matched limits, hard200 and final20-step physical
closure are recorded explicitly. This is an end-to-end official-route baseline,
not an isolated scheduling ablation. No module advances physics at import time.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import faulthandler
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import time
import traceback
from types import SimpleNamespace as NS
from typing import Any

import numpy as np

from .dual_arm_execution import verify_dual_placement
from .held_collision import _transform
from .lift_execution import _rotation_distance
from .multitask_adapters import TASK_IDS, get_task_adapter, task_execution_parameters, task_phase_limits
from .multitask_recording import make_physics_capture, finalize_physics_capture, save_task_geometry
from .trajectory_execution import as_numpy
from .validate_grasp import json_value, save_report
from .validate_path_conflict import _state_arrays
from .validate_sync_compact import compact_configuration, compact_phase_limits, compact_set_phase_limits, compact_kinematic_audit

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[7]
EVIDENCE = ROOT / 'infrastructure/validation/dual-arm/m1-multitask-20261116'
SOLUTION_MODULES = dict(zip(TASK_IDS, ('_008_pick_apple_banana_baskets', '_031_pick_pillbottle_to_regions', '_016_pick_fruits_to_plate')))
MOTION_PHASES = {
    TASK_IDS[0]: ('pregrasp','descend','lift_and_hold','carry_orient_and_transport','retreat'),
    TASK_IDS[1]: ('pregrasp','descend','lift_and_hold','carry_orient_and_transport','carry_lower','retreat','retreat'),
    TASK_IDS[2]: ('pregrasp','descend','lift_and_hold','carry_orient_and_transport','retreat'),
}


# Compatibility exports: implementations live in shared runtime modules.
from .runtime_io import (ControlledTermination, termination_handler)





def termination_self_test() -> None:
    previous=signal.signal(signal.SIGTERM,termination_handler);events=[]
    try:
        try:signal.raise_signal(signal.SIGTERM)
        finally:events.append('finalized')
    except ControlledTermination:events.append('caught')
    finally:signal.signal(signal.SIGTERM,previous)
    assert events==['finalized','caught']


def serial_phase_limits(configuration: dict[str, Any], task_id: str, phase: str) -> dict[str, Any]:
    """Keep common numeric limits; describe the original unretimed serial output."""
    limits=task_phase_limits(configuration,task_id,phase)
    limits.update(source_timestamp_scaling=False,time_parameterization_policy='original-native-TOPP-no-output-retiming')
    return limits


def single_pose7_for_report(pose: Any) -> np.ndarray:
    """Serialize one unchanged Pose to finite world XYZ/WXYZ[7].

    SAPIEN exposes position[3]/quaternion[4]; ManiSkill's single-environment
    Pose exposes position[1,3]/quaternion[1,4]. Accept only those two complete
    shapes, preserving values without modifying or normalizing the input.
    """
    position = np.asarray(as_numpy(pose.p), dtype=float)
    quaternion = np.asarray(as_numpy(pose.q), dtype=float)
    if (position.shape, quaternion.shape) not in (((3,), (4,)), ((1, 3), (1, 4))):
        raise ValueError('One Pose position[3]/quaternion[4] or batch-one [1,3]/[1,4] required')
    result = np.concatenate((position.reshape(3), quaternion.reshape(4)))
    if result.shape != (7,) or not np.isfinite(result).all():
        raise ValueError('Finite XYZ/WXYZ Pose values required for serialization')
    return result


def interpolate_pair(start: Any, command: Any, *, resolution: float = .02) -> np.ndarray:
    """Finite paired joint samples[S,14], including current and next command."""
    first, last = np.asarray(start,dtype=float), np.asarray(command,dtype=float)
    if (first.shape != (14,) or last.shape != (14,) or not np.isfinite(np.r_[first,last]).all()
            or not np.isfinite(resolution) or resolution <= 0):
        raise ValueError('Finite paired fourteen-joint states and positive resolution required')
    count = max(1, math.ceil(float(np.max(np.abs(last-first)))/resolution))
    if count > 10000: raise ValueError('Joint interpolation exceeds finite sampling budget')
    result = first[None] + np.linspace(0,1,count+1)[:,None]*(last-first)[None]
    assert result.shape == (count+1,14) and np.max(np.abs(np.diff(result,axis=0))) <= resolution+1e-12
    return result


def held_sample(reference: Any, object_pose: Any, tcp_pose: Any) -> dict[str, float]:
    """Recompute one measured rigid relation, translation/rotation slip and drop."""
    ref = np.asarray(reference,dtype=float)
    if ref.shape != (4,4) or not np.isfinite(ref).all(): raise ValueError('Finite captured TCP/object matrix required')
    obj, tcp = _transform(object_pose), _transform(tcp_pose)
    relation = np.linalg.inv(tcp) @ obj
    return {'position_slip_m':float(np.linalg.norm(relation[:3,3]-ref[:3,3])),
            'rotation_slip_rad':_rotation_distance(ref[:3,:3],relation[:3,:3]),
            'drop_below_predicted_attachment_m':max(0.,float((tcp@ref)[2,3]-obj[2,3]))}


def held_sample_passes(sample: dict[str,float]) -> bool:
    return bool(sample['position_slip_m'] <= .005 and sample['rotation_slip_rad'] <= .15
                and sample['drop_below_predicted_attachment_m'] <= .01)


class SerialCollisionGate:
    """Same dual world, with at most one real grasp's rigid collision relation."""
    scope = 'official_serial_dual_world_one_measured_or_rigid_payload_with_all_other_actors_measured'

    def __init__(self, base: Any) -> None:
        self.base, self.targets = base, base.targets
        self.supports: list[Any] = [None,None]
        self.active_index: int | None = None
        self.coverage = base.coverage

    @property
    def attached(self) -> bool: return self.active_index is not None

    def refresh(self) -> dict:
        self.base.refresh()
        if self.active_index is not None: self.supports[self.active_index].refresh()
        self.coverage = self.base.coverage
        return self.coverage

    def set_target_contacts_allowed(self, permissions: tuple[bool,bool]) -> None:
        self.base.set_target_contacts_allowed(permissions)

    def assert_scene_unchanged(self) -> None:
        self.base.assert_scene_unchanged()
        if self.active_index is not None: self.supports[self.active_index].assert_scene_unchanged()

    def assert_robot_geometry_unchanged(self) -> None:
        self.base.assert_robot_geometry_unchanged()
        if self.active_index is not None: self.supports[self.active_index].assert_robot_geometry_unchanged()

    def check(self, left: np.ndarray, right: np.ndarray) -> list:
        return (self.base if self.active_index is None else self.supports[self.active_index]).check(left,right)

    def measured_check(self) -> list:
        return (self.base if self.active_index is None else self.supports[self.active_index]).measured_check()

    def released(self) -> tuple[bool,bool]:
        return tuple(False if s is None else bool(s.support_released) for s in self.supports)


def native_run(args: argparse.Namespace, output: Path, report: dict[str,Any]) -> None:
    import gymnasium as gym
    import sapien
    import mani_skill.envs
    from mani_skill.utils.wrappers.record import RecordEpisode
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
    from .dual_payload_collision import DualPayloadCollisionChecker
    from .table_support_collision import NumericalTableSupportCollisionChecker
    from .support_surface_collision import SourceSurfaceSupportCollisionChecker
    from .pill_support_collision import PillTableSupportCollisionChecker

    env = raw = captured = original_hook = gate = None
    frames,ticks,substeps,keys = [],[],[],[]
    counts, phases, active_rows = [0,0],['preparation','preparation'],[[],[]]
    references: list[Any] = [None,None]
    initial_heights: list[Any] = [None,None]
    release_tcp: list[Any] = [None,None]
    losses = [0,0]
    report.update(fault_events=[],stages=[],native_planning_calls=[],planning=[])
    try:
        base = gym.make(args.task_id,obs_mode='none',reward_mode='none',control_mode='pd_joint_pos',sim_backend='cpu',
                        render_backend='gpu',render_mode='rgb_array',max_episode_steps=200)
        env = RecordEpisode(base,output_dir=str(output),trajectory_name='trajectory',save_video=False,save_on_reset=False,
            clean_on_close=False,record_reward=False,record_env_state=True,source_type='motionplanning',
            source_desc=f'official-route serial with shared M1 physical gates; task={args.task_id}')
        env.reset(seed=args.seed);raw=env.unwrapped
        adapter=get_task_adapter(args.task_id,raw);agents=(raw.left_agent,raw.right_agent);targets=adapter.targets
        report.update(task_configuration=adapter.configuration(),target_names=list(adapter.target_names),
                      official_predicate_keys=list(adapter.official_predicate_keys))
        report['task_geometry']=save_task_geometry(output,raw,adapter)
        report['target_state_names']=[report['task_geometry']['actors'][f'target{i}']['state_name'] for i in range(2)]
        np.savez(output/'initial-state.npz',**_state_arrays(raw.get_state_dict()))
        report['initial_state_sha256']=hashlib.sha256((output/'initial-state.npz').read_bytes()).hexdigest()
        keys=list(raw.agent.agents_dict)
        if len(keys)!=2 or any(raw.agent.agents_dict[k] is not a for k,a in zip(keys,agents)):
            raise ValueError('Canonical physical action identity mismatch')
        if not np.isclose(raw.control_timestep,.05,rtol=0,atol=1e-12) or not np.isclose(raw.sim_timestep,.01,rtol=0,atol=1e-12):
            raise ValueError('Serial comparison requires identical20Hz control/100Hz physics')
        report.update(action_key_order=keys,control_timestep_s=float(raw.control_timestep),physics_timestep_s=float(raw.sim_timestep),
                      backend={'physics':raw.backend.sim_backend,'render':raw.backend.render_backend})
        captured=make_physics_capture(env,raw,adapter,args,report,frames,ticks,substeps,keys,
            checker_provider=lambda:None if gate is None else gate.base,
            support_release_provider=lambda:(False,False) if gate is None else gate.released())
        original_hook=captured.original_hook;raw._after_simulation_step=captured.record_substep
        captured.target_phases=tuple(phases);captured.frame()

        def begin(index: int | None, phase: str) -> None:
            if index is not None: phases[index]=phase
            captured.target_phases=tuple(phases)
            captured.phase=phase if index is None else f'{("left","right")[index]}_{phase}'
            report['stage']=captured.phase;save_report(output/'report.json',report)

        def held_monitor(rows: list[dict]) -> None:
            if gate is None or gate.active_index is None: return
            index=gate.active_index
            for row in rows:
                sample=held_sample(references[index],row['object_poses'][index],row['tcp_poses'][index])
                sample.update(substep=row['substep'],control_step=row['control_step'],phase=row['phase'])
                active_rows[index].append(sample)
                if not held_sample_passes(sample): raise RuntimeError(f'Physical held envelope exceeded:arm{index}:{sample}')
            grasp=bool(as_numpy(agents[index].is_grasping(targets[index]))[0])
            losses[index]=0 if grasp else losses[index]+1
            if losses[index]>=2: raise RuntimeError(f'Actual grasp lost for two control ticks:arm{index}')

        class MonitoredEnv:
            unwrapped=raw
            def __getattr__(self,name: str) -> Any: return getattr(raw,name)
            def step(self,action: dict) -> tuple:
                if gate is not None:
                    gate.refresh()
                    start=np.r_[as_numpy(agents[0].robot.get_qpos())[0,:7],as_numpy(agents[1].robot.get_qpos())[0,:7]]
                    command=np.r_[action[keys[0]][:7],action[keys[1]][:7]]
                    for point in interpolate_pair(start,command):
                        blocked=gate.check(point[:7],point[7:])
                        if blocked:
                            report['last_blocked_candidate']={'phase':captured.phase,'collisions':json_value(blocked)}
                            raise RuntimeError('Serial command rejected by combined collision gate')
                first=len(substeps);result=captured.step(action)
                if gate is not None:
                    gate.refresh();blocked=gate.measured_check()
                    if blocked:
                        report['last_measured_collision']={'phase':captured.phase,'collisions':json_value(blocked)}
                        raise RuntimeError('Serial measured collision')
                    held_monitor(substeps[first:])
                stop_reason=filtered_step_stop_reason(result)
                if stop_reason is not None: raise RuntimeError(stop_reason)
                return result
        monitored=MonitoredEnv()
        solution=importlib.import_module('.solutions.'+SOLUTION_MODULES[args.task_id],__package__)
        solution._settle_env(monitored,steps=solution.SETTLE_STEPS)
        report['preparation_steps']=solution.SETTLE_STEPS

        def hold_action() -> dict:
            return {k:np.r_[as_numpy(a.robot.get_qpos())[0,:7],s.gripper_state].astype(np.float32)
                    for k,a,s in zip(keys,agents,solvers)}

        class AuditedSolver(PandaArmMotionPlanningSolver):
            def setup_planner(self) -> Any:
                planner=super().setup_planner();original_topp=planner.TOPP
                def traced(path: Any,step: float=.1,verbose: bool=False) -> Any:
                    result=original_topp(path,step,verbose)
                    arrays={'native_path':np.asarray(path),'solver_velocity_limits':np.asarray(planner.joint_vel_limits),
                            'solver_acceleration_limits':np.asarray(planner.joint_acc_limits)}
                    for field,value in zip(('time','position','velocity','acceleration','duration'),result):
                        if value is not None: arrays['raw_'+field]=np.asarray(value)
                    number=len(report['native_planning_calls'])
                    np.savez(output/f'{number:03d}-{captured.phase}-arm{self.multi_robot_id}-native-topp.npz',**arrays)
                    return result
                planner.TOPP=traced;return planner

            def move_to_pose_with_screw(self,pose: Any,dry_run: bool=False,refine_steps: int=0,other_gripper_state: Any=None) -> Any:
                index=self.multi_robot_id;position=self.motion_count;recipe=MOTION_PHASES[args.task_id]
                if position>=len(recipe): raise ValueError('Unexpected extra motion outside frozen official recipe')
                phase=recipe[position];self.motion_count+=1;begin(index,phase)
                allowed=[False,False];allowed[index]=phase=='descend';gate.set_target_contacts_allowed(tuple(allowed))
                limits=serial_phase_limits(report['execution_configuration'],args.task_id,phase);compact_set_phase_limits(solvers,limits)
                receipt={'phase':captured.phase,'arm_index':index,'status':'running'};started=time.perf_counter()
                before=int(as_numpy(raw.elapsed_steps)[0])
                try:
                    result=super().move_to_pose_with_screw(pose,dry_run=True,refine_steps=refine_steps,other_gripper_state=other_gripper_state)
                    receipt['status']='returned'
                except Exception as exc:
                    receipt.update(status='raised',error_type=type(exc).__name__,error=str(exc));raise
                finally:
                    receipt['wall_seconds']=time.perf_counter()-started;report['native_planning_calls'].append(receipt)
                if int(as_numpy(raw.elapsed_steps)[0])!=before: raise RuntimeError('Planning advanced physical environment')
                if not isinstance(result,dict) or result.get('status')!='Success': raise RuntimeError('Official serial planner failed')
                arrays={k:np.asarray(v) for k,v in result.items() if k in ('time','position','velocity','acceleration','duration')}
                np.savez(output/f'{len(report["planning"]):03d}-{captured.phase}-plan.npz',**arrays)
                audit=compact_kinematic_audit(result,velocity_limit=limits['nominal_velocity_limit_rad_s'],acceleration_limit=limits['nominal_acceleration_limit_rad_s2'])
                report['planning'].append({'phase':captured.phase,'arm_index':index,'native_backend_wall_seconds':receipt['wall_seconds'],
                    'environment_steps':0,'phase_limits':limits,'raw_kinematic_audit':audit,'target_tcp_pose':single_pose7_for_report(pose).tolist(),
                    'official_follow_path_rows':len(result['position']),'source_timestamps_retimed':False})
                if audit['status']!='passed': raise RuntimeError('Official source derivative limits exceeded')
                save_report(output/'report.json',report)
                if dry_run: return result
                moved=super().follow_path(result,other_gripper_state,refine_steps=refine_steps)
                if phase=='lift_and_hold':
                    gain=float(as_numpy(targets[index].pose.p)[0,2]-initial_heights[index])
                    report.setdefault('actual_lift_height_gains_m',{})[str(index)]=gain
                    if gain<.10: raise RuntimeError('Actual serial payload lift below10cm')
                report['stages'].append({'name':captured.phase,'status':'passed','final_control_step':int(as_numpy(raw.elapsed_steps)[0])})
                return moved

            def close_gripper(self,other_gripper_state: Any=None,t: int=6) -> Any:
                index=self.multi_robot_id;begin(index,'close_and_confirm')
                allowed=[False,False];allowed[index]=True;gate.set_target_contacts_allowed(tuple(allowed))
                start=len(ticks);result=super().close_gripper(other_gripper_state,t=t)
                def stable_count() -> int:
                    count=0;previous=None
                    for tick in ticks[start:]:
                        finger=np.asarray(tick[('left_qpos','right_qpos')[index]])[7:9]
                        count=count+1 if tick['objects'][index]['is_grasping'] and previous is not None and np.max(np.abs(finger-previous))<=.0002 else 0
                        previous=finger
                    return count
                while stable_count()<5 and len(ticks)-start<40: result=monitored.step(hold_action())
                if stable_count()<5: raise RuntimeError(f'Serial grasp not continuously stable:arm{index}')
                if counts[index]!=0 or gate.active_index is not None: raise RuntimeError('Serial attachment recapture or overlapping active payload')
                live=gate.base.lives[index]
                if args.task_id==TASK_IDS[2]:
                    support=SourceSurfaceSupportCollisionChecker.from_live(live,targets[index],active_robot=('left','right')[index],support_actor=adapter.initial_supports[index])
                elif args.task_id==TASK_IDS[1]:
                    support=PillTableSupportCollisionChecker.from_live(live,targets[index],active_robot=('left','right')[index])
                else:
                    support=NumericalTableSupportCollisionChecker.from_live(live,targets[index],active_robot=('left','right')[index])
                gate.supports[index]=support;gate.active_index=index;counts[index]+=1
                references[index]=support.attachment_T_tcp_object.copy();initial_heights[index]=float(as_numpy(targets[index].pose.p)[0,2])
                report.setdefault('attachment_captures',[]).append({'arm_index':index,'count':counts[index],
                    'control_step':int(as_numpy(raw.elapsed_steps)[0]),'T_tcp_object':references[index].tolist(),
                    'initial_actor_origin_z_m':initial_heights[index],'physical_weld':False,'coverage':json_value(support.coverage)})
                report['stages'].append({'name':captured.phase,'status':'passed','stable_steps':stable_count(),'environment_steps':len(ticks)-start})
                return result

            def open_gripper(self,other_gripper_state: Any=None,scale: float=1.) -> Any:
                index=self.multi_robot_id;was_held=gate is not None and gate.active_index==index
                begin(index,'release' if was_held else phases[index])
                if was_held:
                    allowed=[False,False];allowed[index]=True
                    gate.set_target_contacts_allowed(tuple(allowed));gate.active_index=None
                start=len(ticks);result=super().open_gripper(other_gripper_state,scale=scale)
                if was_held:
                    def released_count() -> int:
                        count=0
                        for tick in ticks[start:]:
                            finger=np.asarray(tick[('left_qpos','right_qpos')[index]])[7:9]
                            valid=not tick['objects'][index]['is_grasping'] and np.all(np.abs(finger-.04)<=.002)
                            count=count+1 if valid else 0
                        return count
                    while released_count()<3 and len(ticks)-start<30: result=monitored.step(hold_action())
                    if released_count()<3: raise RuntimeError(f'Serial release not continuously confirmed:arm{index}')
                    release_tcp[index]=as_numpy(agents[index].tcp.pose.p)[0].copy()
                    report['stages'].append({'name':captured.phase,'status':'passed','stable_steps':released_count(),'environment_steps':len(ticks)-start})
                    gate.set_target_contacts_allowed((False,False))
                return result

        solvers=[AuditedSolver(monitored,base_pose=a.robot.pose,multi_robot_id=i,debug=False,vis=False,
                 visualize_target_grasp_pose=False,print_env_info=False,
                 joint_vel_limits=report['execution_configuration']['solver_joint_vel_limit_rad_s'],
                 joint_acc_limits=report['execution_configuration']['solver_joint_acc_limit_rad_s2']) for i,a in enumerate(agents)]
        for solver in solvers: solver.motion_count=0
        gate=SerialCollisionGate(DualPayloadCollisionChecker.from_solvers(raw,*solvers,targets=targets))
        report['initial_collision_coverage']=json_value(gate.coverage)
        home=[a.tcp.pose for a in agents]
        solvers[0].open_gripper(other_gripper_state=solvers[1].gripper_state)
        solvers[1].open_gripper(other_gripper_state=solvers[0].gripper_state)
        for index in range(2):
            planner,peer=solvers[index],solvers[1-index]
            center=adapter.destination_actors[index].pose.sp.p
            if args.task_id==TASK_IDS[0]:
                solution._pick_and_place(raw,planner,peer,agent_id=index,obj=targets[index],place_center_xyz=center,init_q=None)
            elif args.task_id==TASK_IDS[1]:
                solution._pick_and_place(raw,planner,peer,agent_id=index,obj=targets[index],place_center_xyz=center)
            else:
                offset=np.array([-.05,.05,0.]) if index==0 else np.array([.05,-.05,0.])
                solution._pick_and_place(raw,planner,peer,agent_id=index,obj=targets[index],place_center_xyz=center+offset,vis=False)
            planner.move_to_pose_with_screw(home[index],other_gripper_state=peer.gripper_state)
            retreat=as_numpy(agents[index].tcp.pose.p)[0]-release_tcp[index]
            report.setdefault('actual_retreat_displacement_m',[]).append(retreat.tolist())
            if retreat[2]<.08: raise RuntimeError('Serial actual retreat vertical gain below8cm')
            phases[index]='verify_placement';captured.target_phases=tuple(phases)
        solvers[1].open_gripper(other_gripper_state=solvers[0].gripper_state)
        solvers[0].open_gripper(other_gripper_state=solvers[1].gripper_state)
        begin(None,'verify_placement')
        verified=verify_dual_placement(monitored,gate,targets,official_predicate_keys=adapter.official_predicate_keys,
                                      region_predicates=adapter.region_predicates(),stable_steps=20)
        report['stages'].append({'name':'verify_placement',**verified})
        report['held_physics_substep_audit']={'status':'passed','objects':[{'actor':name,'physical_substeps':len(rows),
            'max_position_slip_m':max((r['position_slip_m'] for r in rows),default=0.),
            'max_rotation_slip_rad':max((r['rotation_slip_rad'] for r in rows),default=0.),
            'max_drop_below_predicted_attachment_m':max((r['drop_below_predicted_attachment_m'] for r in rows),default=0.)}
            for name,rows in zip(adapter.target_names,active_rows)],'position_limit_m':.005,'rotation_limit_rad':.15,'drop_limit_m':.01}
        report['status']='passed' if verified['status']=='passed' else 'failed'
        report['physical_task_outcome']=report['status']
        report['official_200_step_complete']=report['status']=='passed' and int(as_numpy(raw.elapsed_steps)[0])<200
        report['stage']='finished'
    finally:
        report['attachment_capture_counts']=counts
        report['serial_held_physical_sample_counts']=[len(rows) for rows in active_rows]
        report['final_target_phases']=phases
        if gate is not None:
            report['attachment_final_coverage']=[None if support is None else json_value(support.coverage) for support in gate.supports]
        for index,rows in enumerate(active_rows):
            with (output/f'arm{index}-held-physics-audit.jsonl').open('w') as stream:
                for row in rows: stream.write(json.dumps(json_value(row),allow_nan=False)+'\n')
        finalize_physics_capture(env,raw,captured,output,report,frames,ticks,substeps,keys,original_hook)


def filtered_step_stop_reason(result: tuple) -> str | None:
    """Stop on post-capture termination; success-only termination is filtered upstream."""
    if bool(as_numpy(result[3]).reshape(-1)[0]): return 'Official200 truncation reached'
    if bool(as_numpy(result[2]).reshape(-1)[0]): return 'Native failure termination reached'
    return None


def self_test() -> dict[str,Any]:
    termination_self_test()
    import sapien
    from mani_skill.utils.structs.pose import Pose
    expected_pose = np.array([.123, -.456, .789, 1., 0., 0., 0.])
    unbatched = sapien.Pose(expected_pose[:3], expected_pose[3:])
    batched = Pose.create(unbatched)
    pose_serialization_cases = 0
    for supplied in (unbatched, batched):
        before_p, before_q = as_numpy(supplied.p).copy(), as_numpy(supplied.q).copy()
        serialized = single_pose7_for_report(supplied)
        assert serialized.shape == (7,)
        assert np.array_equal(serialized, np.concatenate((before_p.reshape(3), before_q.reshape(4))))
        assert np.array_equal(as_numpy(supplied.p), before_p) and np.array_equal(as_numpy(supplied.q), before_q)
        assert np.allclose(serialized, expected_pose, rtol=0, atol=1e-7)
        json.dumps(serialized.tolist(), allow_nan=False)
        pose_serialization_cases += 1
    for bad_p, bad_q in ((np.zeros((2, 3)), np.ones((2, 4))),
                          (np.zeros((1, 3)), np.ones(4)),
                          (np.zeros(3), np.ones((1, 4))),
                          (np.zeros(4), np.ones(4)),
                          (np.zeros((0, 3)), np.ones((0, 4))),
                          (np.full(3, np.nan), np.ones(4)),
                          (np.zeros(3), np.full(4, np.inf))):
        try: single_pose7_for_report(NS(p=bad_p, q=bad_q))
        except ValueError: pass
        else: raise AssertionError('Malformed, multi-environment or nonfinite Pose accepted')
        pose_serialization_cases += 1
    configuration={'joint_vel_limit_rad_s':1.5,'joint_acc_limit_rad_s2':2.6,
                   'approach_joint_vel_limit_rad_s':1.,'approach_joint_acc_limit_rad_s2':1.3,'solver_limit_fraction':.99}
    original=task_phase_limits(configuration,TASK_IDS[1],'pregrasp')
    serial_limits=serial_phase_limits(configuration,TASK_IDS[1],'pregrasp')
    assert serial_limits['source_timestamp_scaling'] is False
    assert serial_limits['time_parameterization_policy']=='original-native-TOPP-no-output-retiming'
    assert all(serial_limits[k]==v for k,v in original.items() if k not in ('source_timestamp_scaling','time_parameterization_policy'))
    zero=np.zeros(14);last=zero.copy();last[0]=.05
    samples=interpolate_pair(zero,last)
    assert samples.shape==(4,14) and np.array_equal(samples[0],zero) and np.array_equal(samples[-1],last)
    for invalid in (np.zeros(13),np.full(14,np.nan)):
        try: interpolate_pair(zero,invalid)
        except ValueError: pass
        else: raise AssertionError('Invalid paired joint vector accepted')
    identity=np.eye(4);pose=np.array([0.,0,0,1,0,0,0.])
    assert held_sample_passes(held_sample(identity,pose,pose))
    shifted=pose.copy();shifted[0]=.006
    assert not held_sample_passes(held_sample(identity,shifted,pose))
    dropped=pose.copy();dropped[2]=-.012
    assert not held_sample_passes(held_sample(identity,dropped,pose))
    calls=[]
    fake=NS(targets=(object(),object()),coverage={},refresh=lambda:calls.append('base_refresh'),
            check=lambda l,r:['base'],measured_check=lambda:['base_measured'],set_target_contacts_allowed=lambda p:calls.append(p))
    gate=SerialCollisionGate(fake);assert gate.check(zero[:7],zero[7:])==['base'] and gate.released()==(False,False)
    active=NS(support_released=False,refresh=lambda:calls.append('held_refresh'),check=lambda l,r:['held'],measured_check=lambda:['held_measured'])
    gate.supports[1]=active;gate.active_index=1;gate.refresh()
    assert gate.check(zero[:7],zero[7:])==['held'] and calls==['base_refresh','held_refresh']
    active.support_released=True;gate.active_index=None
    assert gate.released()==(False,True) and gate.measured_check()==['base_measured']
    assert len(SOLUTION_MODULES)==3 and len(MOTION_PHASES[TASK_IDS[1]])==7
    for done,truncated,expected in ((False,False,None),(True,False,'Native failure termination reached'),(False,True,'Official200 truncation reached'),(True,True,'Official200 truncation reached')):
        assert filtered_step_stop_reason((None,0,np.array([done]),np.array([truncated]),{}))==expected
    return {'status':'passed','paired_interpolation_shape':list(samples.shape),'held_pose_shape':[7],
            'boundary_cases':14 + pose_serialization_cases,'pose_serialization_boundary_cases':pose_serialization_cases,
            'pose_serialization_output_shape':[7],'pose_inputs_tested':['sapien.Pose','ManiSkill.Pose(batch=1)'],
            'environment_creations':0,'native_physics_steps':0,'native_validation':'NOT_RUN'}


def main() -> int:
    faulthandler.enable();resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--task-id',choices=TASK_IDS,default=TASK_IDS[1]);parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--output-dir',type=Path);args=parser.parse_args()
    if args.self_test: print(json.dumps(self_test(),indent=2));return 0
    signal.signal(signal.SIGTERM,termination_handler)
    if args.output_dir is None: parser.error('Explicit unused output directory required')
    if not re.fullmatch(r'GPU-[0-9a-fA-F-]+',os.environ.get('CUDA_VISIBLE_DEVICES','')): parser.error('GPU UUID required')
    if any(os.environ.get(k)=='1' for k in ('MANI_SKILL_L1','MANI_SKILL_L2','MANI_SKILL_L3')): parser.error('L0 only')
    args.route='merge-lift';args.execution_profile='execution';args.mode='positive'
    for key,value in task_execution_parameters(args.task_id).items(): setattr(args,key,value)
    output=args.output_dir;output.mkdir(parents=True,exist_ok=False)
    baseline=json.loads((EVIDENCE/'prechange-guard.json').read_text())
    files=[Path(p) for p in baseline['baseline_source_sha256']]+[Path(__file__),HERE/'multitask_adapters.py',HERE/'multitask_recording.py',HERE/'support_surface_collision.py',HERE.parent/'panda/motionplanner.py']
    files += [HERE/'solutions'/f'{name}.py' for name in SOLUTION_MODULES.values()]
    files += [HERE/'pill_support_collision.py']
    unique={str(p):p for p in files};sources={p.name:p.read_bytes() for p in unique.values()}
    if len(sources)!=len(unique): raise ValueError('Ambiguous source basenames')
    snapshot=output/'source';snapshot.mkdir()
    for name,data in sources.items(): (snapshot/name).write_bytes(data)
    official_matches={}
    for name in SOLUTION_MODULES.values():
        relative=Path('mani_skill/examples/motionplanning/dual/solutions')/f'{name}.py'
        official=ROOT/'The-Imitator-Game'/relative
        official_matches[name]=hashlib.sha256(official.read_bytes()).hexdigest()==hashlib.sha256((HERE/'solutions'/f'{name}.py').read_bytes()).hexdigest()
    if not all(official_matches.values()): raise RuntimeError('Serial reference solution differs from official fixed checkout')
    configuration=compact_configuration(args)
    configuration.update(source_timestamp_scaling=False,time_parameterization_policy='original-native-TOPP-no-output-retiming',
                         retiming_scope='original official follow_path consumes unretimed native TOPP rows; raw derivative limits are checked')
    for key in ('release_tcp_offsets_basket_m','release_pose_policy','release_pose_evidence_scope'): configuration.pop(key,None)
    report={'status':'running','started_at':datetime.now(timezone.utc).isoformat(),'task':args.task_id,'seed':args.seed,
        'method':'serial','mode':'positive','execution_configuration':configuration,'stages':[],
        'source_sha256':{name:hashlib.sha256(data).hexdigest() for name,data in sources.items()},
        'source_original_paths':{p.name:str(p) for p in unique.values()},'source_snapshot':str(snapshot),
        'official_reference_matches':official_matches,'official_default_max_episode_steps':200,'max_episode_steps':200,
        'official_200_step_complete':False,'physical_pose_assignment':False,'physical_attachment_created':False,'collision_masks_modified':False,
        'physical_contact_definition':'separation<=0 OR any nonzero impulse; positive-gap zero-impulse manifolds retained as proximity',
        'comparison_scope':'end_to_end_official_route_with_common_physical_gates_not_scheduling_only',
        'route_ledger':{'official_solution':SOLUTION_MODULES[args.task_id],'official_pick_and_place_called_directly':True,
            'official_initial_settle_steps':20 if args.task_id==TASK_IDS[2] else 10,
            'official_solver_follow_path_unchanged':True,'official_waypoints_unchanged':True,
            'retiming':'original native TOPP rows; task-matched solver limits; raw derivative audit',
            'added_grasp_confirmation':'5 consecutive stable controls, max40 including original close6',
            'added_release_confirmation':'3 consecutive open/ungrasped controls, max30 including original open6',
            'added_final_closure':'20 continuous shared placement checks and >=8cm measured retreat',
            'sync_route_differences':'sync uses task adapter grasps (pill model1+.12m/model5+.13m vs official both+.13m), measured release geometry, merged lift and side-preserving fruit goal slots; serial retains official grasps, fixed offsets and diagonal slots'}}
    save_report(output/'report.json',report);started=time.monotonic()
    try: native_run(args,output,report)
    except Exception as exc: report.update(status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    report.update(completed_at=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-started,
        planning_timing_scope='Instrumented move_to_pose call including native TOPP raw NPZ export; original TOPP rows without sync reconstruction; not pure MPLib backend time')
    if any(hashlib.sha256(Path(report['source_original_paths'][name]).read_bytes()).hexdigest()!=digest for name,digest in report['source_sha256'].items()):
        report.update(status='failed',source_identity_violation=True)
    save_report(output/'report.json',report)
    print(json.dumps({'status':report['status'],'report':str(output/'report.json')}),flush=True)
    return 0 if report['status']=='passed' else 1


if __name__=='__main__': raise SystemExit(main())
