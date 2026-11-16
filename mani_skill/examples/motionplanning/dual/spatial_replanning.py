"""Native MPLib spatial replanning and static held-blocker query evidence.

All planning starts from explicit copied nine-joint states. Canonical public
trajectories contain seven controller-order radians and CLOSED grippers. This
module changes only private planner/checker models; actual state, masks, elapsed
ticks, support-release state and original attachments are guarded across every
query. Analytic fake paths appear only in the __main__ CPU boundary self-test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
from transforms3d.quaternions import mat2quat

from .combined_collision import validate_schedule
from .conflict_coordination import search_start_delays
from .held_collision import _world_tcp
from .held_conflict_fixture import _context
from .grasp_collision import _signature
from .trajectory_schedule import ArmTrajectory, build_hold_schedule
from .trajectory_execution import as_numpy
from .validate_grasp import json_value
from .validate_sync_compact import compact_retime_pair, compact_topp_reconstruction, compact_phase_limits, compact_set_phase_limits, compact_kinematic_audit

DT = .05
DELAY_TICKS = tuple(range(0,41,5))
RESERVE_STEPS = 26
CAPABILITY_INTERFACE_VERSION = 'm1-spatial-recovery-interface-v02'


def generic_hook_call_keywords(hook_name: str) -> tuple[str, ...]:
    """Read the actual generic function AST, without constructing an environment.

    A fixed synthetic payload could keep passing while its real call site grows
    new fields. Extract both hook_context's actual dict keys and that hook's
    additional explicit keyword arguments for inspect.signature.bind tests.
    """
    import ast
    import inspect
    from .validate_multitask import native_run
    if hook_name not in ('phase_hook', 'grasp_pose_hook', 'close_failure_hook'):
        raise ValueError('Only real generic callback names are supported')
    tree = ast.parse(inspect.getsource(native_run))
    contexts = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'hook_context']
    if len(contexts) != 1:
        raise RuntimeError('The generic runner must expose one actual hook_context')
    returns = [node.value for node in contexts[0].body if isinstance(node, ast.Return)]
    if len(returns) != 1 or not isinstance(returns[0], ast.Call) or not isinstance(returns[0].func, ast.Name) or returns[0].func.id != 'dict':
        raise RuntimeError('Generic hook_context dict contract changed')
    keys = [keyword.arg for keyword in returns[0].keywords]
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == hook_name]
    if len(calls) != 1 or any(key is None for key in keys):
        raise RuntimeError('One explicit real callback call site is required')
    for keyword in calls[0].keywords:
        if keyword.arg is None:
            if not (isinstance(keyword.value, ast.Call) and isinstance(keyword.value.func, ast.Name)
                    and keyword.value.func.id == 'hook_context'):
                raise RuntimeError('Unknown generic callback keyword expansion')
        else:
            keys.append(keyword.arg)
    if len(set(keys)) != len(keys):
        raise RuntimeError('Duplicate actual generic callback keyword')
    return tuple(keys)


def write_json(path: Path, value: Any) -> None:
    """Write a new evidence artifact; an existing attempt is never replaced."""
    with path.open('x',encoding='utf-8') as stream:
        json.dump(json_value(value),stream,indent=2,allow_nan=False)
        stream.write('\n')


def _array_digest(value: Any) -> str:
    array=np.asarray(value)
    if array.dtype.kind not in 'biuf' or not np.isfinite(array).all(): raise ValueError('Finite numeric query arrays required')
    digest=hashlib.sha256();digest.update(str((array.shape,array.dtype.str)).encode());digest.update(array.tobytes())
    return digest.hexdigest()


def snapshot_identity(snapshot: dict[str,Any]) -> dict[str,Any]:
    arrays={key:_array_digest(value) for key,value in sorted(snapshot['arrays'].items())}
    invariants=json_value(snapshot['invariants'])
    content=json.dumps({'arrays':arrays,'invariants':invariants},sort_keys=True,allow_nan=False).encode()
    return {'sha256':hashlib.sha256(content).hexdigest(),'array_sha256':arrays,'invariants':invariants}


def native_snapshot(solvers: Any, raw: Any, checker: Any) -> dict[str,Any]:
    """Read actual physical arrays and the once-captured collision relations."""
    context=_context(solvers,raw,checker)
    arrays={f'state{key}':value.copy() for key,value in context['state'].items()}
    arrays['elapsed_steps']=context['elapsed'].copy()
    for index,attachment in enumerate(context['attachments']): arrays[f'attachment{index}']=attachment.copy()
    arrays['root_poses']=context['base_poses'].copy()
    for index,(agent,target) in enumerate(zip(raw.agent.agents,checker.targets)):
        arrays[f'actual_grasp{index}']=as_numpy(agent.is_grasping(target)).copy()
        for finger,link in enumerate((agent.finger1_link,agent.finger2_link)):
            arrays[f'actual_finger_force{index}_{finger}']=as_numpy(raw.scene.get_pairwise_contact_forces(link,target)).copy()
    masks={};scene_geometry={}
    for name,(_,shape) in checker.lives[0]._shape_refs.items():
        scene_geometry[name]=_signature(shape)
        getter=getattr(shape,'get_collision_groups',None)
        if callable(getter): masks[name]=list(getter())
    robot_masks={};robot_geometry={}
    for index,agent in enumerate(raw.agent.agents):
        for link in agent.robot.get_links():
            for component in link._objs:
                for shape_index,shape in enumerate(component.get_collision_shapes()):
                    key=f'{index}/{link.name}/{shape_index}'
                    robot_geometry[key]=_signature(shape)
                    getter=getattr(shape,'get_collision_groups',None)
                    if callable(getter): robot_masks[key]=list(getter())
    return {'arrays':arrays,'invariants':{'support_ids':list(context['support_ids']),
        'support_released':[bool(s.support_released) for s in checker._supports],
        'target_contact_permissions':list(context['permissions']),
        'scene_shape_geometry_signatures':json_value(scene_geometry),'robot_shape_geometry_signatures':json_value(robot_geometry),
        'scene_collision_groups':masks,'robot_collision_groups':robot_masks}}


class ReadOnlyQueryGuard:
    """Hash and retain before/after query state; planner failures still get proof."""
    def __init__(self,snapshot: Callable[[],dict[str,Any]],output: Path | None,label: str):
        self.snapshot=snapshot;self.output=output;self.label=label;self.receipt: dict[str,Any]={}

    def __enter__(self) -> ReadOnlyQueryGuard:
        self.before=self.snapshot();self.before_identity=snapshot_identity(self.before)
        if self.output is not None:
            path=self.output/f'{self.label}-query-before.npz'
            if path.exists(): raise FileExistsError(path)
            np.savez(path,**self.before['arrays'])
        return self

    def __exit__(self,error_type: Any,error: Any,tb: Any) -> bool:
        after=self.snapshot();identity=snapshot_identity(after)
        same=identity['sha256']==self.before_identity['sha256']
        self.receipt={'status':'passed' if same else 'failed','physical_state_unchanged':same,
            'before':self.before_identity,'after':identity,'environment_steps':0,'physics_steps':0,
            'query_exception':None if error is None else f'{type(error).__name__}: {error}'}
        if self.output is not None: write_json(self.output/f'{self.label}-query-proof.json',self.receipt)
        if not same: raise RuntimeError('A query changed physical state, attachment, support state, mask or permission')
        return False


def controller_context(solvers: Any,raw: Any,checker: Any) -> dict[str,Any]:
    checker.refresh()
    return _context(solvers,raw,checker)


def stationary_pair(moving: ArmTrajectory,context: dict[str,Any],moving_arm: int,*,
                    paired_starts: Any | None=None) -> Any:
    if moving_arm not in (0,1): raise ValueError('Canonical moving arm index required')
    starts=np.asarray(context['starts'] if paired_starts is None else paired_starts,dtype=float)
    if starts.shape!=(2,7) or not np.isfinite(starts).all(): raise ValueError('Finite starts[2,7] required')
    if not np.allclose(moving.positions_rad[0],starts[moving_arm],rtol=0,atol=1e-6): raise ValueError('Native source start differs from query start')
    peer=1-moving_arm
    hold=ArmTrajectory(context['keys'][peer],context['names'][peer],[0.],[starts[peer]],-1.)
    sources=(moving,hold) if moving_arm==0 else (hold,moving)
    return build_hold_schedule(*sources,control_timestep_s=DT,max_steps=1000)


def route_summary(schedule: Any) -> dict[str,Any]:
    return {'scheduled_steps':schedule.num_steps,'sources':[{
        'action_key':source.action_key,'joint_names':list(source.joint_names),'duration_s':source.duration_s,
        'time_sha256':_array_digest(source.times_s),'position_sha256':_array_digest(source.positions_rad),
        'start_qpos_rad':source.positions_rad[0].tolist(),'end_qpos_rad':source.positions_rad[-1].tolist(),
        'gripper_command':source.gripper_command} for source in schedule.trajectories]}


def save_schedule(path: Path,schedule: Any) -> None:
    if path.exists(): raise FileExistsError(path)
    np.savez(path,left_time=schedule.trajectories[0].times_s,right_time=schedule.trajectories[1].times_s,
        left_position=schedule.trajectories[0].positions_rad,right_position=schedule.trajectories[1].positions_rad,
        target_times=schedule.target_times_s,left_commands=schedule.left_commands,right_commands=schedule.right_commands)


def plan_stationary_peer_route(solvers: Any,raw: Any,checker: Any,*,moving_arm: int,
                               world_waypoints: Any,configuration: dict[str,Any],output: Path,
                               label: str,report: dict[str,Any],paired_starts: Any | None=None) -> tuple[Any | None,dict[str,Any]]:
    """Call original native waypoint IK/TOPP, retain raw traces, then retime q unchanged."""
    waypoints=np.asarray(world_waypoints,dtype=float)
    if (moving_arm not in (0,1) or waypoints.ndim!=2 or waypoints.shape[1]!=7 or len(waypoints)<1
            or not np.isfinite(waypoints).all() or not np.allclose(np.linalg.norm(waypoints[:,3:],axis=1),1.,atol=1e-6,rtol=0)):
        raise ValueError('Finite native world waypoints[K,7] with normalized WXYZ required')
    context=controller_context(solvers,raw,checker)
    starts=np.asarray(context['starts'] if paired_starts is None else paired_starts,dtype=float)
    if starts.shape!=(2,7) or not np.isfinite(starts).all(): raise ValueError('Paired private starts[2,7] required')
    solver=solvers[moving_arm];planner=solver.planner
    actual=tuple(j.get_name() for j in solver.robot.get_active_joints());planner_names=tuple(planner.user_joint_names)
    arm_names=tuple(context['names'][moving_arm]);group_names=tuple(planner_names[i] for i in planner.move_group_joint_indices)
    if set(group_names)!=set(arm_names) or len(group_names)!=7: raise ValueError('Native move group and controller joint names differ')
    permutation=[group_names.index(name) for name in arm_names]
    full=context['full_states'][moving_arm][[actual.index(name) for name in planner_names]].copy()
    for name,value in zip(arm_names,starts[moving_arm]): full[planner_names.index(name)]=value
    limits=compact_phase_limits(configuration,'carry_orient_and_transport');compact_set_phase_limits(solvers,limits)
    records=[];topp_records=[];native_topp=planner.TOPP;native_ik=planner.IK
    result=None;schedule=None;error=None;started=time.perf_counter()
    receipt={'status':'running','source':'original_MPLib_Planner.plan_waypoints_smooth','moving_arm':moving_arm,
        'world_waypoints_xyz_wxyz':waypoints.tolist(),'controller_joint_names':list(arm_names),
        'planner_group_joint_names':list(group_names),'planner_to_controller_permutation':permutation,
        'current_planner_qpos_copy':full.tolist(),'private_paired_starts_rad':starts.tolist(),'phase_limits':limits,
        'mask_non_arm_joints':[name not in arm_names for name in planner_names],
        'points_per_segment':10,'use_straight_line':False,'wrt_world':True,'environment_steps':0}
    def traced_ik(goal: Any,start: Any,mask: Any=None,*args: Any,**kwargs: Any) -> Any:
        value=native_ik(goal,np.asarray(start).copy(),mask,*args,**kwargs)
        records.append({'goal_base_xyz_wxyz':np.asarray(goal).tolist(),'start_planner_qpos':np.asarray(start).tolist(),
            'status':str(value[0]),'all_native_solutions_planner_order':[np.asarray(q).tolist() for q in value[1]]})
        return value
    def traced_topp(path: Any,step: float=.1,verbose: bool=False) -> Any:
        value=native_topp(np.asarray(path).copy(),step,verbose)
        arrays={'native_path':np.asarray(path).copy(),'solver_velocity_limits':np.asarray(planner.joint_vel_limits).copy(),
                'solver_acceleration_limits':np.asarray(planner.joint_acc_limits).copy()}
        for field,item in zip(('time','position','velocity','acceleration','duration'),value):
            if item is not None: arrays['raw_'+field]=np.asarray(item).copy()
        file=output/f'{label}-topp-{len(topp_records):02d}-raw.npz'
        if file.exists(): raise FileExistsError(file)
        np.savez(file,**arrays)
        trace=compact_topp_reconstruction(path,planner.joint_vel_limits,planner.joint_acc_limits,step,value)
        topp_records.append(trace);return value
    guard=ReadOnlyQueryGuard(lambda:native_snapshot(solvers,raw,checker),output,label)
    try:
        with guard:
            planner.IK=traced_ik;planner.TOPP=traced_topp
            try:
                result=planner.plan_waypoints_smooth(waypoints.copy(),full.copy(),time_step=DT,
                    mask=receipt['mask_non_arm_joints'],points_per_segment=10,use_straight_line=False,wrt_world=True)
            finally:
                planner.IK=native_ik;planner.TOPP=native_topp
            receipt['native_status']=str(result.get('status'))
            if result.get('status')!='Success':
                receipt['status']='native_no_solution'
            else:
                if len(topp_records)!=1: raise RuntimeError('Successful native source requires one complete TOPP trace')
                trace=topp_records[0]
                raw_pair=(result,None) if moving_arm==0 else (None,result)
                trace_pair=(trace,None) if moving_arm==0 else (None,trace)
                retimed,certificate=compact_retime_pair(raw_pair,trace_pair,
                    velocity_limit=limits['nominal_velocity_limit_rad_s'],acceleration_limit=limits['nominal_acceleration_limit_rad_s2'])
                converted=retimed[moving_arm]
                # Same nominal sample gate as the generic runner, alongside
                # compact_retime_pair's existing hard analytic-curve checks.
                nominal_audit=compact_kinematic_audit(converted,
                    velocity_limit=limits['nominal_velocity_limit_rad_s'],acceleration_limit=limits['nominal_acceleration_limit_rad_s2'])
                receipt['nominal_kinematic_audit']=nominal_audit
                receipt['raw_native_kinematic_audit']=compact_kinematic_audit(result,
                    velocity_limit=limits['nominal_velocity_limit_rad_s'],acceleration_limit=limits['nominal_acceleration_limit_rad_s2'])
                if nominal_audit['status']!='passed':raise RuntimeError('Native spatial source violates unchanged nominal kinematic gate')
                source=ArmTrajectory(context['keys'][moving_arm],arm_names,converted['time'],
                    np.asarray(converted['position'])[:,permutation],-1.)
                schedule=stationary_pair(source,context,moving_arm,paired_starts=starts)
                arrays={k:np.asarray(v).copy() for k,v in trace.items()}
                for field in ('time','position','velocity','acceleration','duration'):
                    arrays['raw_'+field]=np.asarray(result[field]).copy();arrays['retimed_'+field]=np.asarray(converted[field]).copy()
                arrays['waypoint_configs_planner_order']=np.asarray(result['waypoint_configs'])
                np.savez(output/f'{label}-native-plan.npz',**arrays)
                save_schedule(output/f'{label}-schedule.npz',schedule)
                fk=[]
                for config in result['waypoint_configs']:
                    matrix=_world_tcp(checker.models[moving_arm],np.asarray(config).copy(),'panda_hand_tcp')
                    fk.append(np.r_[matrix[:3,3],mat2quat(matrix[:3,:3])].tolist())
                actual_fk=np.asarray(fk,dtype=float)
                position_errors=np.linalg.norm(actual_fk[:,:3]-waypoints[:,:3],axis=1)
                orientation_errors=2*np.arccos(np.clip(np.abs(np.sum(actual_fk[:,3:]*waypoints[:,3:],axis=1)),-1,1))
                fidelity=bool(np.all(position_errors<=.002) and np.all(orientation_errors<=.01))
                receipt.update(status='passed',retiming=certificate,schedule=route_summary(schedule),
                    native_waypoint_fk_xyz_wxyz=fk,
                    native_waypoint_controller_qpos_rad=[np.asarray(q)[[planner_names.index(n) for n in arm_names]].tolist()
                                                         for q in result['waypoint_configs']],
                    world_goal_fidelity={'status':'passed' if fidelity else 'failed',
                        'position_errors_m':position_errors.tolist(),'orientation_errors_rad':orientation_errors.tolist(),
                        'position_limit_m':.002,'orientation_limit_rad':.01})
                if not fidelity:
                    receipt['status']='native_goal_fidelity_failed';schedule=None
    except Exception as exc:
        error=exc;receipt.update(status='raised',error_type=type(exc).__name__,error=str(exc))
    finally:
        elapsed=time.perf_counter()-started;receipt.update(wall_seconds=elapsed,IK=records,query_state_proof=guard.receipt)
        report.setdefault('native_planning_calls',[]).append({'phase':label,'arm_index':moving_arm,'status':receipt['status'],'wall_seconds':elapsed})
        write_json(output/f'{label}-native-plan.json',receipt)
    if error is not None: raise error
    return schedule,receipt


def blocker_witnesses(rows: Any,checker: Any) -> list[dict]:
    right_names=set(checker._target_names[1]) if hasattr(checker,'_target_names') else {'banana'}
    result=[]
    for row in rows:
        objects={row.get('object_name1'),row.get('object_name2')}
        if 'right' in objects or objects & right_names: result.append(row)
    return result


def validate_frozen_blocker(schedule: Any,checker: Any,*,guard_factory: Callable[[str],Any] | None=None,
                            label: str='static-blocker') -> dict[str,Any]:
    """Require clear ends/interior conflict and all17 truly checked wait rejections."""
    left,right=schedule.trajectories
    if (right.duration_s!=0. or len(right.positions_rad)!=1 or left.gripper_command!=-1. or right.gripper_command!=-1.):
        raise ValueError('The right held blocker must be a constant CLOSED source')
    starts=(left.positions_rad[0],right.positions_rad[0]);ends=(left.positions_rad[-1],right.positions_rad[-1])
    def evaluate() -> dict[str,Any]:
        initial=checker.check(*starts);final=checker.check(*ends)
        nominal=validate_schedule(schedule,*starts,checker,max_joint_step_rad=.02,max_samples=20000)
        selected,waits=search_start_delays(left,right,checker,control_timestep_s=DT,max_delay_ticks=40,
            delay_step_ticks=5,max_steps=schedule.num_steps+40,max_joint_step_rad=.02,max_samples=20000)
        collision=nominal.get('first_collision')
        interior=bool(collision is not None and 0.<collision['time_s']<left.duration_s)
        named=[] if collision is None else blocker_witnesses(collision['collisions'],checker)
        all_checked=(selected is None and waits['candidate_count']==17 and waits['checked_candidate_count']==17
            and all(row['reason']=='candidate_collision' and row['validation'] is not None for row in waits['candidates']))
        return {'status':'passed' if not initial and not final and interior and named and all_checked else 'failed',
            'initial_collisions':initial,'final_collisions':final,'nominal_validation':nominal,
            'interior_conflict':interior,'named_right_blocker_witnesses':named,'wait_search':waits,
            'all17_delays_geometrically_checked_and_rejected':all_checked,'source':route_summary(schedule),
            'proof_scope':'stationary right occupancy and unchanged left spatial curve; bounded17-query evidence, no moving-obstacle universal claim',
            'environment_steps':0,'physics_steps':0}
    if guard_factory is None: return evaluate()
    with guard_factory(label): return evaluate()


def schedule_fk_xyz(schedule: Any,checker: Any,context: dict[str,Any],moving_arm: int=0) -> np.ndarray:
    """FK of the same <=.02rad sampled command polygon used by the safety gate."""
    source=schedule.trajectories[moving_arm];model=checker.models[moving_arm]
    # The private model is built in solver planner order. Map the captured
    # physical active joints explicitly into that same nine-joint order.
    normal_names=tuple(checker.solvers[moving_arm].planner.user_joint_names)
    if len(normal_names)!=9: raise ValueError('Expected nine named native joints')
    actual=tuple(j.get_name() for j in checker.solvers[moving_arm].robot.get_active_joints())
    full=context['full_states'][moving_arm][[actual.index(n) for n in normal_names]].copy()
    indices=[normal_names.index(n) for n in source.joint_names]
    points=[context['starts'][moving_arm].copy()]
    commands=(schedule.left_commands,schedule.right_commands)[moving_arm]
    for command in commands:
        count=max(1,math.ceil(float(np.max(np.abs(command[:7]-points[-1])))/.02))
        start=points[-1].copy()
        points.extend(start+(command[:7]-start)*j/count for j in range(1,count+1))
    xyz=[]
    for joints in points:
        state=full.copy();state[indices]=joints
        xyz.append(_world_tcp(model,state,'panda_hand_tcp')[:3,3])
    return np.asarray(xyz)


def validate_occupied_endpoint(schedule: Any,checker: Any) -> dict[str,Any]:
    """Reject a fixed exact endpoint; every declared wait must receive geometry."""
    left,right=schedule.trajectories
    if right.duration_s!=0. or len(right.positions_rad)!=1:
        raise ValueError('Endpoint negative requires stationary right occupancy')
    initial=checker.check(left.positions_rad[0],right.positions_rad[0])
    final=checker.check(left.positions_rad[-1],right.positions_rad[-1])
    selected,waits=search_start_delays(left,right,checker,control_timestep_s=DT,max_delay_ticks=40,delay_step_ticks=5,
        max_steps=schedule.num_steps+40,max_joint_step_rad=.02,max_samples=20000)
    all_checked=selected is None and waits['checked_candidate_count']==waits['candidate_count']==17 and all(
        row['reason']=='candidate_collision' for row in waits['candidates'])
    named=blocker_witnesses(final,checker)
    return {'status':'passed' if not initial and final and named and all_checked else 'failed',
        'initial_collisions':initial,'endpoint_collisions':final,'named_right_blocker_witnesses':named,
        'wait_search':waits,'all17_delays_geometrically_checked_and_rejected':all_checked,
        'claim_scope':'unsafe fixed endpoint configuration with stationary peer; no global impossibility claim',
        'environment_steps':0,'physics_steps':0}


def spatial_change_certificate(nominal_xyz: Any,detour_xyz: Any,detour_waypoint_fk: Any,
                               nominal_goal: Any,detour_goal: Any) -> dict[str,Any]:
    """Same world goal, nontrivial travel, and actual native-FK spatial change."""
    original,changed=np.asarray(nominal_xyz,float),np.asarray(detour_xyz,float)
    middle=np.asarray(detour_waypoint_fk,float);first_goal=np.asarray(nominal_goal,float);last_goal=np.asarray(detour_goal,float)
    if (any(v.ndim!=2 or v.shape[1]!=3 or len(v)<2 for v in (original,changed))
            or middle.shape!=(3,) or first_goal.shape!=(7,) or last_goal.shape!=(7,)
            or not all(np.isfinite(v).all() for v in (original,changed,middle,first_goal,last_goal))):
        raise ValueError('Finite sampled XYZ paths, middle[3] and world goals[7] required')
    high=float(middle[2]-original[:,2].max())
    lateral=float(max(np.max(middle[:2]-original[:,:2].max(axis=0)),np.max(original[:,:2].min(axis=0)-middle[:2])))
    waypoint_distance=float(np.linalg.norm(changed-middle,axis=1).min())
    displacement=float(np.linalg.norm(first_goal[:3]-original[0]))
    nominal_goal_error=float(np.linalg.norm(original[-1]-first_goal[:3]))
    detour_goal_error=float(np.linalg.norm(changed[-1]-last_goal[:3]))
    same=bool(np.array_equal(first_goal,last_goal))
    passed=(same and displacement>=.08 and max(high,lateral)>=.05 and waypoint_distance<=.015
            and max(nominal_goal_error,detour_goal_error)<=.003)
    return {'status':'passed' if passed else 'failed','same_exact_world_goal':same,'goal_xyz_wxyz':first_goal.tolist(),
        'start_to_goal_m':displacement,'detour_middle_fk_xyz_m':middle.tolist(),'above_nominal_peak_m':high,
        'outside_nominal_xy_envelope_m':lateral,'executed_polygon_nearest_middle_m':waypoint_distance,
        'nominal_endpoint_goal_error_m':nominal_goal_error,'detour_endpoint_goal_error_m':detour_goal_error,
        'nominal_fk_bounds_m':[original.min(axis=0).tolist(),original.max(axis=0).tolist()],
        'spatial_change_threshold_m':.05,'scope':'native waypoint FK and <=.02rad sampled command polygon; no continuous Cartesian certificate'}


def self_test() -> dict[str,Any]:
    names=tuple(f'j{i}' for i in range(7));left=np.zeros((2,7));left[1,0]=1.;right=np.zeros((1,7));right[0,0]=2.
    sources=(ArmTrajectory('left',names,[0.,1.],left,-1.),ArmTrajectory('right',names,[0.],right,-1.))
    schedule=build_hold_schedule(*sources,control_timestep_s=DT)
    class Checker:
        _target_names=({'apple'},{'banana'})
        def check(self,a: Any,b: Any) -> list:
            return [{'collision_type':'synthetic_only','object_name1':'apple','object_name2':'right'}] if .4<=a[0]<=.6 and b[0]==2 else []
    result=validate_frozen_blocker(schedule,Checker())
    assert result['status']=='passed' and result['wait_search']['checked_candidate_count']==17
    end_left=ArmTrajectory('left',names,[0.,1.],np.vstack([left[0],np.array([.5,0,0,0,0,0,0])]),-1.)
    negative=validate_occupied_endpoint(build_hold_schedule(end_left,sources[1],control_timestep_s=DT),Checker())
    assert negative['status']=='passed' and negative['wait_search']['checked_candidate_count']==17
    state={'q':np.zeros(3)}
    snapshot=lambda:{'arrays':{'q':state['q'].copy()},'invariants':{'support_released':[True,True]}}
    with ReadOnlyQueryGuard(snapshot,None,'unchanged') as guard: pass
    assert guard.receipt['physical_state_unchanged']
    try:
        with ReadOnlyQueryGuard(snapshot,None,'changed'): state['q'][0]=1.
    except RuntimeError: pass
    else: raise AssertionError('Physical query mutation accepted')
    try:
        with ReadOnlyQueryGuard(snapshot,None,'exception') as guard: raise ValueError('sentinel')
    except ValueError: pass
    assert guard.receipt['physical_state_unchanged'] and 'sentinel' in guard.receipt['query_exception']
    nominal=np.array([[0,0,.1],[.1,0,.1],[.2,0,.1]])
    detour=np.array([[0,0,.1],[.1,0,.2],[.2,0,.1]])
    goal=np.array([.2,0,.1,1,0,0,0.])
    assert spatial_change_certificate(nominal,detour,detour[1],goal,goal)['status']=='passed'
    assert spatial_change_certificate(nominal,nominal,nominal[1],goal,goal)['status']=='failed'
    changed=goal.copy();changed[0]=.3
    assert spatial_change_certificate(nominal,detour,detour[1],goal,changed)['status']=='failed'
    wrong_endpoint=detour.copy();wrong_endpoint[-1,0]+=.01
    assert spatial_change_certificate(nominal,wrong_endpoint,detour[1],goal,goal)['status']=='failed'
    native_arrays={'status':'Success','time':np.array([0.,1.]),'position':np.zeros((2,7)),
        'velocity':np.zeros((2,7)),'acceleration':np.zeros((2,7))}
    assert compact_kinematic_audit(native_arrays,velocity_limit=1.5,acceleration_limit=2.6)['status']=='passed'
    for field,limit in (('velocity',1.5),('acceleration',2.6)):
        excessive={**native_arrays,field:np.full((2,7),limit+.001)}
        assert compact_kinematic_audit(excessive,velocity_limit=1.5,acceleration_limit=2.6)['status']=='failed'
    return {'status':'passed','boundary_cases':9,'checked_wait_candidates':34,'environment_creations':0,
        'capability_interface_version':CAPABILITY_INTERFACE_VERSION,'unchanged_nominal_kinematic_gate_boundaries':3,
        'physics_steps':0,'native_planner_status':'NOT_RUN','synthetic_paths_only_in_self_test':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if not args.self_test: parser.error('use --self-test')
    print(json.dumps(self_test(),indent=2))
