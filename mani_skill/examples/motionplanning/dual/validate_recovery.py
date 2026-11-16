"""Thin recovery acceptance from saved real failures; native execution NOT_RUN.

Root schedules native development/acceptance only after the120 paired M1 cases.
This separate400-tick capability never replaces official200-step M1 results.
Hooks change planned grasp targets, physically open/retract/regrasp, or perform
checked held detours. They never reset after the fault or assign physical state.
Source/force/state/action evidence remains complete, including the original fault.
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
import signal
import time
import traceback
from typing import Any

import numpy as np

from .combined_collision import validate_schedule
from .conflict_coordination import delay_trajectory
from .dual_arm_execution import (close_dual_and_confirm, execute_dual_held_move,
                                 execute_dual_reach, release_dual_and_confirm)
from .multitask_adapters import task_execution_parameters
from .recovery_execution import (CAPABILITY_BUDGET, actual_progress_certificate,
    array_identity, default_recipe, proposed_progress_point, source_release_certificate,
    source_start_certificate, validate_recipe, self_test as pure_self_test,
    connect_actual_start_to_tail, connected_blockage_certificate, RECOVERY_INTERFACE_VERSION)
from .spatial_replanning import (ReadOnlyQueryGuard, controller_context, native_snapshot,
    plan_stationary_peer_route, route_summary, save_schedule, schedule_fk_xyz,
    spatial_change_certificate, stationary_pair, validate_frozen_blocker,
    validate_occupied_endpoint, write_json)
from .spatial_replanning import generic_hook_call_keywords
from .trajectory_execution import as_numpy
from .trajectory_schedule import ArmTrajectory, build_hold_schedule
from .validate_grasp import json_value, save_report
from .validate_multitask import native_run as generic_native_run, termination_handler
from .validate_path_conflict import _state_arrays
from .validate_spatial_replanning import validate_geometry_recipe
from .validate_sync_compact import compact_configuration

HERE=Path(__file__).resolve().parent
ROOT=Path(__file__).resolve().parents[7]
EVIDENCE=ROOT/'infrastructure/validation/dual-arm/m1-multitask-20261116'
TASK='TwoRobotPickAppleBananaToBaskets-v1'
CASES=('grasp-failure-recover','transport-blockage-recover','unrecoverable-negative')


def _poses(actors: Any) -> np.ndarray:
    return np.stack([as_numpy(actor.pose.raw_pose)[0].copy() for actor in actors])


def _checkpoint(context: dict[str,Any], label: str) -> dict[str,Any]:
    raw,output,report=context['raw'],context['output'],context['report']
    arrays={name:np.asarray(value).copy() for name,value in _state_arrays(raw.get_state_dict()).items()}
    path=output/f'{label}.npz'
    if path.exists():raise FileExistsError(path)
    np.savez(path,**arrays)
    step=int(as_numpy(raw.elapsed_steps)[0]);ticks=context['ticks']
    if step!=len(ticks):raise RuntimeError('Checkpoint control tick does not match physical action history')
    targets=context['adapter'].targets
    if ticks and any(not np.array_equal(as_numpy(actor.pose.raw_pose)[0],ticks[-1]['objects'][i]['pose']) for i,actor in enumerate(targets)):
        raise RuntimeError('Checkpoint does not equal actual last recorded target poses')
    value={'state_file':str(path),'state_file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'arrays_sha256':array_identity(arrays),'control_step':step,'state_leaf_count':len(arrays),
        'target_poses_xyz_wxyz':_poses(targets).tolist(),
        'tcp_poses_xyz_wxyz':_poses([agent.tcp for agent in raw.agent.agents]).tolist(),
        'measured_arm_qpos_rad':[as_numpy(agent.robot.get_qpos())[0,:7].tolist() for agent in raw.agent.agents]}
    report['recovery'].setdefault('checkpoints',{})[label]=value
    write_json(output/f'{label}.json',value)
    return value


def _query_guard(context: dict[str,Any],label: str) -> ReadOnlyQueryGuard:
    return ReadOnlyQueryGuard(lambda:native_snapshot(context['solvers'],context['raw'],context['checker']),context['output'],label)


def _validate_route(context: dict[str,Any],schedule: Any,label: str) -> dict[str,Any]:
    starts=controller_context(context['solvers'],context['raw'],context['checker'])['starts']
    with _query_guard(context,label):
        value=validate_schedule(schedule,*starts,context['checker'],max_joint_step_rad=.02,max_samples=20000)
    write_json(context['output']/f'{label}-collision-validation.json',value)
    return value


def _waits_from_actual_checkpoint(context: dict[str,Any],schedule: Any,
                                  connector_receipt: dict[str,Any]) -> dict[str,Any]:
    """Check all17 waits on the explicit connector plus unchanged old tail.

    The left held arm can drift slightly while the blocker moves. This keeps
    the exact old spatial tail and proves the first commanded connector from
    measured fault joints for every wait, rather than pretending drift is zero.
    """
    starts=controller_context(context['solvers'],context['raw'],context['checker'])['starts']
    pairs=[(0,0)]+[pair for delay in range(5,41,5) for pair in ((delay,0),(0,delay))]
    start_certificate=source_start_certificate(np.stack([source.positions_rad[0] for source in schedule.trajectories]),starts)
    if start_certificate['status']!='passed':raise RuntimeError('Actual-state wait source omitted its measured-start connector')
    candidates=[]
    for number,(left_wait,right_wait) in enumerate(pairs):
        sources=tuple(delay_trajectory(source,delay,.05) for source,delay in zip(schedule.trajectories,(left_wait,right_wait)))
        candidate=build_hold_schedule(*sources,control_timestep_s=.05,max_steps=1000)
        with _query_guard(context,f'recovery-actual-wait-{number:02d}'):
            validation=validate_schedule(candidate,*starts,context['checker'],max_joint_step_rad=.02,max_samples=20000)
        collision=validation.get('first_collision')
        unchanged_tail_start=left_wait*.05+connector_receipt['connector_duration_s']
        in_tail=bool(collision is not None and collision['time_s']>unchanged_tail_start)
        candidates.append({'delay_ticks':[left_wait,right_wait],'validation':validation,
                           'first_collision_strictly_in_unchanged_tail':in_tail,
                           'unchanged_tail_starts_at_s':unchanged_tail_start})
    result={'status':'passed' if all(item['validation']['status']=='collision'
                and item['first_collision_strictly_in_unchanged_tail'] for item in candidates) else 'failed',
        'candidate_count':17,'checked_candidate_count':len(candidates),'actual_start_qpos_rad':np.asarray(starts).tolist(),
        'actual_source_start_certificate':start_certificate,'connector':connector_receipt,
        'candidates':candidates,'environment_steps':0,'physics_steps':0,
        'scope':'explicit actual-state connector then unchanged saved left spatial tail; all17 waits geometrically checked from actual measured fault joints'}
    write_json(context['output']/'recovery-all17-waits-from-actual-fault-state.json',result)
    return result


def _native_route(context: dict[str,Any],label: str,side: int,waypoints: Any) -> tuple[Any|None,dict[str,Any]]:
    context['begin'](label)
    measured=controller_context(context['solvers'],context['raw'],context['checker'])
    before=_checkpoint(context,label+'-planning-input')
    schedule,receipt=plan_stationary_peer_route(context['solvers'],context['raw'],context['checker'],moving_arm=side,
        world_waypoints=waypoints,configuration=context['report']['execution_configuration'],output=context['output'],
        label=label,report=context['report'])
    after={name:np.asarray(value) for name,value in _state_arrays(context['raw'].get_state_dict()).items()}
    if array_identity(after)!=before['arrays_sha256']:raise RuntimeError('Native planning changed physical failure/recovery state')
    if schedule is not None:
        certificate=source_start_certificate(np.stack([source.positions_rad[0] for source in schedule.trajectories]),measured['starts'])
        if certificate['status']!='passed':raise RuntimeError('Recovery source does not start at actual measured joints')
        receipt['actual_state_start_certificate']=certificate
        receipt['actual_planning_input_checkpoint']=before
        write_json(context['output']/f'{label}-actual-source-start.json',receipt)
    return schedule,receipt


def _execute_held(context: dict[str,Any],label: str,schedule: Any,*,hold_steps: int=0) -> bool:
    context['begin'](label)
    result=execute_dual_held_move(context['capture'],schedule,context['checker'],context['adapter'].targets,
                                  hold_steps=hold_steps,settle_max_steps=15,stable_steps=3)
    return bool(context['finish'](label,result))


def _fail(context: dict[str,Any],reason: str) -> bool:
    context['report'].update(status='failed',reason=reason)
    context['report']['recovery'].update(status='failed',reason=reason)
    save_report(context['output']/'report.json',context['report'])
    return False


class RecoveryController:
    """One actual fault and at most one recovery; final task closure stays generic."""
    def __init__(self,args: Any,recipe: dict[str,Any],spatial_recipe: dict[str,Any]|None=None):
        self.args=args;self.recipe=validate_recipe(recipe);self.spatial_recipe=spatial_recipe
        self.initial_origins: np.ndarray|None=None
        self.injected=False;self.regrasp_attempts=0

    def grasp_pose_hook(self,*,grasp_pairs: Any,**context: Any) -> Any:
        report=context['report']
        report['recovery']={'status':'prefix_running','case':self.args.case,'recipe':self.recipe,'spatial_world_recipe':self.spatial_recipe,
            'engineering_budget_control_ticks':400,'official_default_budget_control_ticks':200,
            'post_fault_reset_calls':0,'physical_state_assignments':0,'physical_welds':0,'collision_mask_changes':0,
            'native_acceptance':'NOT_PROVEN','checkpoints':{},'faults':[]}
        self.initial_origins=_poses(context['adapter'].targets)[:,2].copy()
        report['recovery']['initial_source_origin_z_m']=self.initial_origins.tolist()
        if self.args.case!='grasp-failure-recover':return grasp_pairs
        import sapien
        offsets=np.asarray(self.recipe['grasp_miss_offsets_world_m'])
        result=tuple(tuple(sapien.Pose(p=np.asarray(pose.p)+offsets[side],q=pose.q) for pose in pair)
                     for side,pair in enumerate(grasp_pairs))
        self.injected=True
        report['recovery']['fault_injection']={'kind':'fixed_planned_TCP_horizontal_offset',
            'world_offsets_m':offsets.tolist(),'original_target_poses':[[np.r_[p.p,p.q].tolist() for p in pair] for pair in grasp_pairs],
            'fault_target_poses':[[np.r_[p.p,p.q].tolist() for p in pair] for pair in result],
            'physical_pose_assignment':False,'environment_steps':0}
        return result

    def close_failure_hook(self,*,close_result: dict[str,Any],**context: Any) -> bool:
        if self.args.case!='grasp-failure-recover' or not self.injected or self.regrasp_attempts:
            return False
        if close_result.get('status')!='grasp_timeout':return _fail(context,'grasp_recovery_requires_real_confirmation_timeout')
        raw=context['raw'];targets=context['adapter'].targets;agents=raw.agent.agents
        actual=[bool(as_numpy(agent.is_grasping(actor))[0]) for agent,actor in zip(agents,targets)]
        if actual[0]:return _fail(context,'offset_did_not_create_actual_left_grasp_failure')
        fault=_checkpoint(context,'fault_state')
        source_names=context['adapter'].initial_support_names
        source_contacts=[False,False]
        for contact in context['substeps'][-1]['object_contact_rows']:
            if contact['physical_contact']:
                for index in contact['target_indices']:
                    source_contacts[index]|=any(body.get('kind')=='actor' and body.get('actor')==source_names[index] for body in contact['bodies'])
        safety=source_release_certificate(actual,source_contacts,self.initial_origins,_poses(targets)[:,2])
        context['report']['recovery'].update(fault_detection={'actual_grasped':actual,'close_status':close_result['status'],
            'close_environment_steps':close_result['environment_steps'],'fault_checkpoint':fault},safe_source_open_certificate=safety)
        if safety['status']!='passed':return _fail(context,'grasp_recovery_peer_not_proven_still_on_source')
        self.regrasp_attempts+=1
        context['begin']('recovery_source_open')
        if not context['finish']('recovery_source_open',release_dual_and_confirm(context['capture'],context['checker'],targets)):
            return False
        _checkpoint(context,'recovery_after_open')
        import sapien
        retreat=tuple(sapien.Pose(p=as_numpy(agent.tcp.pose.p)[0]+[0,0,.08],q=as_numpy(agent.tcp.pose.q)[0]) for agent in agents)
        context['begin']('recovery_source_retreat')
        schedule=context['plan']('recovery_source_retreat',retreat,closed=False)
        if not context['finish']('recovery_source_retreat',execute_dual_reach(context['capture'],schedule,context['checker'])):return False
        _checkpoint(context,'recovery_before_regrasp')
        pairs=context['adapter'].grasp_poses()
        context['report']['recovery']['current_state_regrasp_target_poses']=[[np.r_[pose.p,pose.q].tolist() for pose in pair] for pair in pairs]
        for index,local,contact in ((0,'pregrasp',False),(1,'descend',True)):
            label='recovery_regrasp_'+local;context['begin'](label)
            schedule=context['plan'](label,(pairs[0][index],pairs[1][index]),closed=False,limit_phase=local)
            if not context['finish'](label,execute_dual_reach(context['capture'],schedule,context['checker'],allow_target_contacts=(contact,contact))):return False
        context['begin']('recovery_regrasp_close')
        closed=close_dual_and_confirm(context['capture'],context['checker'],targets)
        if not context['finish']('recovery_regrasp_close',closed):return False
        context['report']['recovery'].update(status='regrasp_passed_awaiting_same_task_closure',
            regrasp_attempts=self.regrasp_attempts,attach_after_regrasp='generic single actual capture follows this hook')
        _checkpoint(context,'recovery_after_regrasp')
        return True

    def phase_hook(self,**context: Any) -> bool|None:
        if self.args.case=='grasp-failure-recover':
            if self.regrasp_attempts != 1 or 'fault_state' not in context['report']['recovery']['checkpoints']:
                return _fail(context,'grasp_recovery_positive_requires_observed_actual_fault_and_regrasp')
            return True
        if self.spatial_recipe is None:return _fail(context,'transport_recovery_requires_pre_frozen_spatial_recipe')
        return self._transport_recovery(context)

    def _transport_recovery(self,context: dict[str,Any]) -> bool|None:
        recipe=self.spatial_recipe;raw=context['raw'];agents=raw.agent.agents;checker=context['checker']
        initial_tcp=_poses([agent.tcp for agent in agents]);initial_objects=_poses(context['adapter'].targets)
        q=initial_tcp[:,3:].copy();start_step=int(as_numpy(raw.elapsed_steps)[0])
        prefix=proposed_progress_point(initial_tcp[0,:3],recipe['nominal_middle_xyz_m'])
        schedule,_=_native_route(context,'carry_recovery_progress_prefix',0,[np.r_[prefix,q[0]]])
        if schedule is None:return _fail(context,'native_progress_prefix_no_solution')
        if _validate_route(context,schedule,'recovery_progress_prefix_check')['status']!='passed':return _fail(context,'progress_prefix_not_clear')
        if not _execute_held(context,'carry_recovery_progress_prefix',schedule):return False
        now_tcp=_poses([agent.tcp for agent in agents]);now_objects=_poses(context['adapter'].targets)
        progress=actual_progress_certificate(initial_tcp[0,:3],now_tcp[0,:3],initial_objects[0,:3],now_objects[0,:3],
            int(as_numpy(raw.elapsed_steps)[0])-start_step)
        context['report']['recovery']['actual_progress_before_blocker']=progress
        if progress['status']!='passed':return _fail(context,'transport_fault_cannot_be_claimed_without_actual_5tick_5cm_progress')
        _checkpoint(context,'recovery_actual_progress_state')
        nominal_points=np.array([np.r_[recipe['nominal_middle_xyz_m'],q[0]],np.r_[recipe['goal_xyz_m'],q[0]]])
        tail,tail_receipt=_native_route(context,'carry_recovery_original_tail',0,nominal_points)
        if tail is None or _validate_route(context,tail,'recovery_original_tail_initially_clear')['status']!='passed':
            return _fail(context,'original_tail_not_valid_before_blocker')
        save_schedule(context['output']/'recovery-original-tail-exact-source.npz',tail)
        original_source=route_summary(tail)['sources'][0]
        setup,_=_native_route(context,'carry_recovery_blocker_setup',1,[np.r_[recipe['blocker_xyz_m'],q[1]]])
        if setup is None or _validate_route(context,setup,'recovery_blocker_setup_check')['status']!='passed':return _fail(context,'blocker_setup_not_clear')
        if not _execute_held(context,'carry_recovery_blocker_setup',setup):return False
        fault=_checkpoint(context,'fault_state')
        actual_context=controller_context(context['solvers'],raw,checker)
        blocked=stationary_pair(tail.trajectories[0],actual_context,0,paired_starts=np.stack([tail.trajectories[0].positions_rad[0],actual_context['starts'][1]]))
        if route_summary(blocked)['sources'][0]!=original_source:raise RuntimeError('Blocker fault silently replaced original left tail')
        # Preserve the original tail byte-for-byte, but do not pretend its
        # pre-blocker first joint state equals the actual post-blocker hold.
        configuration=context['report']['execution_configuration']
        joined_source,connector_source,connector_receipt=connect_actual_start_to_tail(
            tail.trajectories[0],actual_context['starts'][0],
            velocity_limit=configuration['joint_vel_limit_rad_s'],acceleration_limit=configuration['joint_acc_limit_rad_s2'])
        connector=stationary_pair(connector_source,actual_context,0)
        actual_blocked=stationary_pair(joined_source,actual_context,0)
        save_schedule(context['output']/'recovery-actual-to-old-tail-connector.npz',connector)
        save_schedule(context['output']/'recovery-actual-connected-old-tail.npz',actual_blocked)
        write_json(context['output']/'recovery-actual-to-old-tail-connector.json',connector_receipt)
        connector_check=_validate_route(context,connector,'recovery_actual_connector_full_check')
        if connector_check['status']!='passed':return _fail(context,'actual_to_old_tail_connector_not_clear')
        current_check=_validate_route(context,actual_blocked,'recovery_original_tail_after_real_blocker')
        current_proof=connected_blockage_certificate(connector_check,current_check,connector_receipt)
        context['report']['recovery']['fault_detection']={'kind':'previously_clear_tail_blocked_after_real_transport_progress',
            'fault_checkpoint':fault,'original_left_source':original_source,'same_left_source':True,
            'measured_left_hold_drift_rad':(np.asarray(actual_context['starts'][0])-tail.trajectories[0].positions_rad[0]).tolist(),
            'candidate_after_blocker':current_check,'actual_start_connector':connector_receipt,
            'actual_connector_validation':connector_check,'connected_blockage_certificate':current_proof}
        if current_proof['status']!='passed':return _fail(context,'real_blocker_did_not_obstruct_unchanged_tail_from_actual_state')
        wait_proof=validate_frozen_blocker(blocked,checker,guard_factory=lambda label:_query_guard(context,label),label='recovery-all17-original-tail-waits')
        write_json(context['output']/'recovery-all17-original-tail-waits.json',wait_proof)
        if wait_proof['status']!='passed':return _fail(context,'waiting_exhaustion_or_internal_blocker_witness_not_proven')
        if _waits_from_actual_checkpoint(context,actual_blocked,connector_receipt)['status']!='passed':return _fail(context,'waiting_from_actual_fault_state_not_exhausted')
        if self.args.case=='unrecoverable-negative':
            occupied,_=_native_route(context,'carry_recovery_unsafe_endpoint_query',0,[np.r_[recipe['blocker_xyz_m'],q[0]]])
            if occupied is None:return _fail(context,'negative_native_ik_failure_does_not_prove_endpoint_occupied')
            with _query_guard(context,'recovery-occupied-endpoint-proof'):
                negative=validate_occupied_endpoint(occupied,checker)
            write_json(context['output']/'recovery-negative-endpoint.json',negative)
            if negative['status']!='passed':return _fail(context,'negative_endpoint_geometry_not_proven')
            hold_sources=tuple(ArmTrajectory(key,names,[0.],[start],-1.) for key,names,start in zip(actual_context['keys'],actual_context['names'],actual_context['starts']))
            hold=build_hold_schedule(*hold_sources,control_timestep_s=.05)
            if not _execute_held(context,'carry_recovery_safe_terminal_hold',hold,hold_steps=20):return False
            context['report'].update(status='passed',physical_task_outcome='incomplete_safe_terminal_refusal',official_200_step_complete=False)
            context['report']['recovery'].update(status='safe_terminal_refusal',recovery_completed_same_task=False,
                negative_scope='fixed unsafe endpoint geometry; no global impossibility claim',premature_release_commands=0)
            return None
        detour,detour_receipt=_native_route(context,'carry_recovery_spatial_detour',0,
            [np.r_[recipe['detour_middle_xyz_m'],q[0]],np.r_[recipe['goal_xyz_m'],q[0]]])
        if detour is None:return _fail(context,'spatial_recovery_no_native_route')
        if _validate_route(context,detour,'recovery_detour_full_combined_check')['status']!='passed':return _fail(context,'spatial_recovery_route_not_clear')
        with _query_guard(context,'recovery-spatial-change-proof'):
            original_xyz=schedule_fk_xyz(actual_blocked,checker,actual_context)
            detour_xyz=schedule_fk_xyz(detour,checker,actual_context)
            changed=spatial_change_certificate(original_xyz,detour_xyz,np.asarray(detour_receipt['native_waypoint_fk_xyz_wxyz'][0])[:3],nominal_points[-1],nominal_points[-1])
        write_json(context['output']/'recovery-spatial-change.json',changed)
        np.savez(context['output']/'recovery-spatial-FK.npz',original_xyz=original_xyz,detour_xyz=detour_xyz)
        if changed['status']!='passed':return _fail(context,'recovery_native_path_not_spatially_distinct_same_goal')
        if not _execute_held(context,'carry_recovery_spatial_detour',detour):return False
        _checkpoint(context,'recovery_after_detour')
        restore,_=_native_route(context,'carry_recovery_blocker_restore',1,[initial_tcp[1]])
        if restore is None or _validate_route(context,restore,'recovery_blocker_restore_full_check')['status']!='passed':return _fail(context,'blocker_cannot_safely_restore_for_original_task')
        if not _execute_held(context,'carry_recovery_blocker_restore',restore):return False
        _checkpoint(context,'recovery_before_original_task_continuation')
        context['report']['recovery'].update(status='detour_passed_awaiting_same_task_closure',
            attachment_reference_policy='original two measured grasps unchanged throughout transport recovery',
            recovery_completed_same_task=False,remaining_control_ticks=400-int(as_numpy(raw.elapsed_steps)[0]))
        return True


def self_test() -> dict[str,Any]:
    import inspect
    pure=pure_self_test();recipe=default_recipe()
    controller=RecoveryController(argparse.Namespace(case='grasp-failure-recover'),recipe)
    from types import SimpleNamespace
    import sapien
    targets=[SimpleNamespace(pose=SimpleNamespace(raw_pose=np.array([[0.,0,.02,1,0,0,0]]))) for _ in range(2)]
    poses=((sapien.Pose([0,0,.2]),sapien.Pose([0,0,.12])),(sapien.Pose([0,.2,.2]),sapien.Pose([0,.2,.12])))
    original=np.asarray(poses[0][1].p).copy()
    report={};adapted=controller.grasp_pose_hook(grasp_pairs=poses,report=report,adapter=SimpleNamespace(targets=targets))
    assert np.allclose(adapted[0][1].p,[.09,0,.12]) and np.array_equal(adapted[1][1].p,poses[1][1].p)
    assert np.array_equal(poses[0][1].p,original)
    assert report['recovery']['fault_injection']['physical_pose_assignment'] is False
    transport=RecoveryController(argparse.Namespace(case='transport-blockage-recover'),recipe)
    assert transport.grasp_pose_hook(grasp_pairs=poses,report={},adapter=SimpleNamespace(targets=targets)) is poses
    assert controller.regrasp_attempts==0 and pure['status']=='passed'
    callback_contracts={}
    for name in ('grasp_pose_hook','close_failure_hook','phase_hook'):
        keywords=generic_hook_call_keywords(name)
        inspect.signature(getattr(controller,name)).bind(**{key:object() for key in keywords})
        callback_contracts[name]=list(keywords)
    import ast
    tree=ast.parse(inspect.getsource(generic_native_run))
    continuation=[node for node in ast.walk(tree) if isinstance(node,ast.If)
        and isinstance(node.test,ast.Compare) and isinstance(node.test.left,ast.Name)
        and node.test.left.id=='continue_task' and len(node.test.ops)==1 and isinstance(node.test.ops[0],ast.IsNot)
        and len(node.test.comparators)==1 and isinstance(node.test.comparators[0],ast.Constant)
        and node.test.comparators[0].value is True and len(node.body)==1 and isinstance(node.body[0],ast.Return)]
    assert len(continuation)==1, 'Real generic True-return continuation contract changed'
    # Exercise the actual17-query function with synthetic collision geometry
    # only. A clear connector + blocked old tail passes; a prefix collision
    # fails even when all17 candidates report some collision.
    import tempfile
    from contextlib import nullcontext
    names=tuple(f'j{i}' for i in range(7));left=np.zeros(7);right=np.zeros(7);right[0]=2.
    end=left.copy();end[0]=1.;actual=left.copy();actual[0]=-.005
    old=ArmTrajectory('left',names,[0.,1.],np.stack([left,end]),-1.)
    joined,connector,certificate=connect_actual_start_to_tail(old,actual,velocity_limit=1.5,acceleration_limit=2.6)
    peer=ArmTrajectory('right',names,[0.],[right],-1.)
    source=build_hold_schedule(joined,peer,control_timestep_s=.05)
    class SyntheticChecker:
        prefix_blocked=False
        def check(self,a: Any,b: Any) -> list:
            hit=.4<=a[0]<=.6 or self.prefix_blocked and -.004<=a[0]<=-.001
            return [{'object_name1':'left','object_name2':'right'}] if hit else []
    actual_context={'starts':np.stack([actual,right])}
    original_context,original_guard=globals()['controller_context'],globals()['_query_guard']
    try:
        globals()['controller_context']=lambda *args,**kwargs:actual_context
        globals()['_query_guard']=lambda *args,**kwargs:nullcontext()
        for blocked_prefix,expected in ((False,'passed'),(True,'failed')):
            with tempfile.TemporaryDirectory() as directory:
                fake_checker=SyntheticChecker();fake_checker.prefix_blocked=blocked_prefix
                checked=_waits_from_actual_checkpoint({'solvers':None,'raw':None,'checker':fake_checker,'output':Path(directory)},source,certificate)
                assert checked['status']==expected and checked['checked_candidate_count']==17
    finally:
        globals()['controller_context'],globals()['_query_guard']=original_context,original_guard
    return {'status':'passed','pure_contract_checks':pure['checks'],'hook_checks':4,
        'recovery_interface_version':RECOVERY_INTERFACE_VERSION,'actual_generic_callback_signature_bind':callback_contracts,
        'actual_start_connector_boundaries':pure['actual_start_connector_boundaries'],
        'actual_state_wait_candidates_checked':34,'generic_true_return_continuation_AST':'passed',
        'wait_test_scope':'synthetic checker only; no native simulation or physical state',
        'environment_creations':0,'physics_steps':0,'native_acceptance':'NOT_RUN',
        'runtime_prerequisite':'root schedules all native capability runs only after120 formal M1 allocations'}


def main() -> int:
    faulthandler.enable();resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--case',choices=CASES,default=CASES[0]);parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--output-dir',type=Path);parser.add_argument('--recovery-recipe',type=Path)
    parser.add_argument('--spatial-recipe',type=Path);args=parser.parse_args()
    if args.self_test:print(json.dumps(self_test(),indent=2));return 0
    signal.signal(signal.SIGTERM,termination_handler)
    if args.output_dir is None:parser.error('An unused explicit output directory is required')
    if not re.fullmatch(r'GPU-[0-9a-fA-F-]+',os.environ.get('CUDA_VISIBLE_DEVICES','')):parser.error('One rendering GPU UUID required')
    if args.seed not in (*range(5),*range(320,325)):parser.error('Development0..4 or pre-frozen independent320..324 only')
    recipe=default_recipe() if args.recovery_recipe is None else validate_recipe(json.loads(args.recovery_recipe.read_text()))
    spatial=None
    if args.case!='grasp-failure-recover':
        if args.spatial_recipe is None:parser.error('Transport/negative requires a previously frozen spatial recipe')
        spatial=validate_geometry_recipe(json.loads(args.spatial_recipe.read_text()))
    if any(os.environ.get(key)=='1' for key in ('MANI_SKILL_L1','MANI_SKILL_L2','MANI_SKILL_L3')):parser.error('Frozen L0 only')
    args.task_id=TASK;args.route='merge-lift';args.execution_profile='execution';args.mode='positive'
    for key,value in task_execution_parameters(TASK).items():setattr(args,key,value)
    args.max_episode_steps=CAPABILITY_BUDGET
    output=args.output_dir;output.mkdir(parents=True,exist_ok=False)
    baseline=json.loads((EVIDENCE/'prechange-guard.json').read_text())
    sources=[Path(path) for path in baseline['baseline_source_sha256']]
    sources += [HERE/name for name in ('validate_recovery.py','recovery_execution.py','validate_multitask.py',
        'multitask_adapters.py','multitask_recording.py','spatial_replanning.py','validate_spatial_replanning.py')]
    source_bytes={path.name:path.read_bytes() for path in sources};snapshot=output/'source';snapshot.mkdir()
    for name,data in source_bytes.items():(snapshot/name).write_bytes(data)
    configuration=compact_configuration(args)
    for key in ('release_tcp_offsets_basket_m','release_pose_policy','release_pose_evidence_scope'):configuration.pop(key,None)
    report={'status':'running','started_at':datetime.now(timezone.utc).isoformat(),'task':TASK,'seed':args.seed,
        'recovery_interface_version':RECOVERY_INTERFACE_VERSION,
        'method':'recovery-capability','case':args.case,'mode':'positive','execution_configuration':configuration,
        'official_default_max_episode_steps':200,'max_episode_steps':400,'official_200_step_complete':False,
        'physical_pose_assignment':False,'physical_attachment_created':False,'collision_masks_modified':False,
        'source_sha256':{name:hashlib.sha256(data).hexdigest() for name,data in source_bytes.items()},'source_snapshot':str(snapshot),
        'formal_M1_comparison_scope':'separate400-tick engineering capability; not an official200 M1 replacement',
        'recipe_inputs':{'recovery_recipe_sha256':hashlib.sha256(json.dumps(recipe,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            'spatial_recipe_file':None if args.spatial_recipe is None else str(args.spatial_recipe),
            'spatial_recipe_file_sha256':None if args.spatial_recipe is None else hashlib.sha256(args.spatial_recipe.read_bytes()).hexdigest()}}
    save_report(output/'report.json',report);controller=RecoveryController(args,recipe,spatial);started=time.monotonic()
    try:
        generic_native_run(args,output,report,grasp_pose_hook=controller.grasp_pose_hook,
            close_failure_hook=controller.close_failure_hook,phase_hook=controller.phase_hook)
        if report.get('status')=='passed' and report['recovery']['status']!='safe_terminal_refusal':
            complete=report.get('physical_task_outcome')=='passed' and report.get('stages',[])[-1].get('name')=='verify_placement'
            report['recovery'].update(status='passed' if complete else 'failed',recovery_completed_same_task=complete)
            if not complete:report.update(status='failed',reason='recovery_did_not_finish_original_full_task')
    except Exception as exc:report.update(status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    if any(hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=digest for name,digest in report['source_sha256'].items()):
        report.update(status='failed',source_identity_violation=True)
    report.update(completed_at=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-started)
    save_report(output/'report.json',report);print(json.dumps({'status':report['status'],'report':str(output/'report.json')}),flush=True)
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
