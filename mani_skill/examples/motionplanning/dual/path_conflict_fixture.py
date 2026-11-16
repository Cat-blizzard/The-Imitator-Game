"""Fixed empty-hand round trips with a reachable shared middle configuration.

Contract: world metres/WXYZ, controller-order arm radians[N,7], seconds[N].
MPLib IK chooses middle joints during development; a quintic joint time law is
the source trajectory. No native planner trajectory is claimed. build_fixture
loads the selected immutable NPZ and never plans, changes state, or steps.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .trajectory_schedule import ArmTrajectory, build_hold_schedule
from .combined_collision import validate_schedule

ROOT = Path(__file__).resolve().parents[7]
EVIDENCE = ROOT / 'infrastructure/validation/dual-arm/path-conflict-20261116'
FIXTURE = EVIDENCE / 'fixture.npz'
METADATA = EVIDENCE / 'fixture.json'
GRID = ((-.10,.22),(-.10,.30),(-.20,.22),(-.20,.30),(-.15,.26),
        (-.10,.38),(-.20,.38),(0.,.30),(0.,.38))
DT, SOURCE_DT, VMAX, AMAX = .05, .01, 1.5, 2.6


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save(path: Path, value: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def quintic_round_trip(starts: Any, middles: Any) -> tuple[np.ndarray,np.ndarray,dict[str,Any]]:
    """Two analytic zero-velocity/acceleration endpoint round trips [2,N,7]."""
    start, middle = np.asarray(starts,dtype=float), np.asarray(middles,dtype=float)
    assert start.shape == middle.shape == (2,7) and np.isfinite(np.r_[start,middle]).all()
    delta = middle-start; magnitude = float(np.max(np.abs(delta)))
    required = max(1.875*magnitude/VMAX,np.sqrt((10/np.sqrt(3))*magnitude/AMAX),DT)
    half = np.ceil((required*1.05)/DT)*DT
    times = np.linspace(0.,2*half,int(round(2*half/SOURCE_DT))+1)
    u = np.where(times<=half,times/half,(2*half-times)/half)
    law = 10*u**3-15*u**4+6*u**5
    positions = start[:,None,:] + delta[:,None,:]*law[None,:,None]
    positions[:,0] = positions[:,-1] = start
    maxima_v = 1.875*np.max(np.abs(delta),axis=1)/half
    maxima_a = (10/np.sqrt(3))*np.max(np.abs(delta),axis=1)/half**2
    assert np.max(maxima_v) <= VMAX and np.max(maxima_a) <= AMAX
    assert positions.shape == (2,len(times),7)
    return times,positions,{'source':'analytic quintic joint round trip after MPLib IK; not native planner output',
        'outbound_duration_s':float(half),'round_trip_duration_s':float(2*half),
        'max_velocity_rad_s':maxima_v.tolist(),'max_acceleration_rad_s2':maxima_a.tolist(),
        'velocity_limit_rad_s':VMAX,'acceleration_limit_rad_s2':AMAX,
        'endpoint_velocity_and_acceleration':0.,'source_dt_s':SOURCE_DT,
        'motion_control_steps':int(round(2*half/DT)),
        'serial_motion_wait_control_steps':int(round(4*half/DT)),
        'preparation_steps':10,'settling_budget_steps':25,
        'root_stride10_serial_wait_steps':int(np.ceil(round(2*half/DT)/10)*10),
        'root_stride10_motion_wait_steps':int(round(2*half/DT))+int(np.ceil(round(2*half/DT)/10)*10),
        'total_serial_preparation_settle_budget_steps':int(round(4*half/DT))+35}


def delayed(trajectory: ArmTrajectory, delay_s: float) -> ArmTrajectory:
    if delay_s <= 0 or not np.isfinite(delay_s): raise ValueError('Positive finite start wait required')
    return ArmTrajectory(trajectory.action_key,trajectory.joint_names,
        np.r_[0.,trajectory.times_s+delay_s],
        np.vstack([trajectory.positions_rad[0],trajectory.positions_rad]),trajectory.gripper_command)


def _trajectories(raw: Any, times: np.ndarray, positions: np.ndarray) -> tuple[ArmTrajectory,ArmTrajectory]:
    keys = list(raw.agent.agents_dict)
    if len(keys)!=2: raise ValueError('Two canonical arms required')
    return tuple(ArmTrajectory(key,tuple(agent.controller.controllers['arm'].config.joint_names),
        times,positions[i],1.) for i,(key,agent) in enumerate(zip(keys,raw.agent.agents)))


def build_fixture(solvers: Any, raw: Any, checker: Any) -> tuple[ArmTrajectory,ArmTrajectory,dict[str,Any]]:
    """Use fixed middle joints and exact measured starts after ten open-hand ticks.

    New scenes receive the same fixed qmid and analytic timing rule, without IK,
    candidate selection, or relaxed source/start checks. Scene and combined
    collision validation remain the caller's per-case responsibility.
    """
    from .trajectory_execution import as_numpy
    if len(solvers)!=2 or checker.attached or checker.target_contacts_allowed!=(False,False):
        raise ValueError('Empty hands and original collision permissions required')
    metadata = json.loads(METADATA.read_text(encoding='utf-8'))
    if _sha(FIXTURE)!=metadata['fixture_npz_sha256']: raise ValueError('Fixture source identity changed')
    with np.load(FIXTURE,allow_pickle=False) as data:
        middles = data['middles'].copy()
        bases,fingers = data['base_poses'].copy(),data['finger_qpos'].copy()
    starts=[]
    for i,agent in enumerate(raw.agent.agents):
        q = as_numpy(agent.robot.get_qpos())[0]
        starts.append(np.asarray(q[:7],dtype=float).copy())
        if not np.allclose(q[7:],fingers[i],rtol=0,atol=1e-5): raise ValueError('Prepared finger geometry differs')
        if not np.allclose(as_numpy(agent.robot.pose.raw_pose)[0],bases[i],rtol=0,atol=1e-6):
            raise ValueError('Robot root differs from fixed fixture')
    starts=np.asarray(starts)
    times,positions,kinematics=quintic_round_trip(starts,middles)
    left,right = _trajectories(raw,times,positions)
    assert np.array_equal(left.positions_rad[0],starts[0]) and np.array_equal(right.positions_rad[0],starts[1])
    metadata['development_reference_seed']=metadata.pop('seed')
    metadata['development_reference_checks']=metadata.pop('checks')
    metadata['development_reference_kinematics']=metadata.pop('kinematics')
    metadata.update(source_factory='fixed archived qmid plus exact measured per-case q0; same quintic rule; no per-case IK or selection',
        current_start_joints_rad=starts.tolist(),fixed_middle_joints_rad=middles.tolist(),
        kinematics=kinematics,right_start_wait_s=left.duration_s,
        per_case_collision_validation_required=True)
    return left,right,metadata


def prepare_exploration() -> dict[str,Any]:
    """Create one task instance, perform exactly ten ordinary hold ticks, then guard."""
    import gymnasium as gym
    import mani_skill.envs
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
    from .dual_payload_collision import DualPayloadCollisionChecker
    from .trajectory_execution import as_numpy
    from .task_targets import TASK
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    if (EVIDENCE/'exploration.json').exists() or FIXTURE.exists():
        raise ValueError('Existing exploration evidence must be preserved; do not create a second task instance')
    env = gym.make(TASK,obs_mode='none',reward_mode='none',control_mode='pd_joint_pos',
        sim_backend='cpu',render_backend='gpu',render_mode='rgb_array',max_episode_steps=200)
    env.reset(seed=0); raw = env.unwrapped
    assert np.isclose(raw.control_timestep,DT) and np.isclose(raw.sim_timestep,.01)
    report={'status':'preparing','created_at':datetime.now(timezone.utc).isoformat(),
        'purpose':'development geometric fixture discovery; no formal denominator',
        'task':TASK,'seed':0,'environment_constructions':1,'explicit_reset_calls':1,
        'preparation_steps':0,'physics_steps_after_preparation':0,'attempts':[],
        'source_sha256':{str(Path(__file__)):_sha(Path(__file__))}}
    _save(EVIDENCE/'exploration.json',report)
    hold={key:np.r_[as_numpy(agent.robot.get_qpos())[0,:7],1.].astype(np.float32)
          for key,agent in raw.agent.agents_dict.items()}
    for _ in range(10):
        _,_,term,trunc,_=env.step(hold)
        if bool(as_numpy(term).reshape(-1)[0]) or bool(as_numpy(trunc).reshape(-1)[0]):
            raise RuntimeError('Preparation terminated')
        report['preparation_steps']+=1
    elapsed=as_numpy(raw.elapsed_steps).copy(); assert elapsed.tolist()==[10]
    original_step=raw.step
    def forbidden_step(*args:Any,**kwargs:Any) -> Any:
        raise RuntimeError('Exploration cannot step after ten preparation ticks')
    raw.step=forbidden_step
    solvers=[PandaArmMotionPlanningSolver(env,base_pose=a.robot.pose,multi_robot_id=i,
        debug=False,vis=False,visualize_target_grasp_pose=False,print_env_info=False,
        joint_vel_limits=VMAX,joint_acc_limits=AMAX) for i,a in enumerate(raw.agent.agents)]
    checker=DualPayloadCollisionChecker.from_solvers(raw,*solvers,targets=(raw.apple_out,raw.banana_out))
    assert checker.target_contacts_allowed==(False,False) and not checker.attached
    report.update(status='geometry_ready',prepared_elapsed_steps=elapsed.tolist(),
        initial_tcp_poses=[as_numpy(a.tcp.pose.raw_pose)[0].tolist() for a in raw.agent.agents],
        base_poses=[as_numpy(a.robot.pose.raw_pose)[0].tolist() for a in raw.agent.agents])
    _save(EVIDENCE/'exploration.json',report)
    return {'env':env,'raw':raw,'solvers':solvers,'checker':checker,'report':report,
            'prepared_elapsed_steps':elapsed,'original_step':original_step}


def explore(context: dict[str,Any]) -> dict[str,Any]:
    """Bounded IK grid and independent full-scene candidate checks; zero steps."""
    from .trajectory_execution import as_numpy
    from .held_collision import _world_tcp
    raw,solvers,checker,report = (context[k] for k in ('raw','solvers','checker','report'))
    starts=np.array([as_numpy(a.robot.get_qpos())[0,:7] for a in raw.agent.agents])
    full=[as_numpy(a.robot.get_qpos())[0].copy() for a in raw.agent.agents]
    initial=checker.check(*starts)
    if initial: raise ValueError(f'Initial full-scene collision: {initial}')
    report['initial_collisions']=initial
    for index,(x,z) in enumerate(GRID):
        attempt={'index':index,'shared_middle_tcp_xyz_m':[x,0.,z],
                 'ik_status':[],'ik_solution_counts':[],'status':'running'}
        report['attempts'].append(attempt); _save(EVIDENCE/'exploration.json',report)
        goals=[]
        for i,solver in enumerate(solvers):
            world=np.r_[x,0.,z,as_numpy(raw.agent.agents[i].tcp.pose.q)[0]]
            local=solver.planner.transform_goal_to_wrt_base(world)
            status,solutions=solver.planner.IK(local,full[i].copy(),mask=[False]*7+[True]*2,
                n_init_qpos=4,threshold=1e-4)
            attempt['ik_status'].append(status); attempt['ik_solution_counts'].append(len(solutions))
            if status!='Success' or not solutions: break
            selected=min(solutions,key=lambda q:float(np.max(np.abs(np.asarray(q)[:7]-starts[i]))))
            goals.append(np.asarray(selected)[:7].copy())
        if len(goals)!=2:
            attempt['status']='ik_failed'; _save(EVIDENCE/'exploration.json',report); continue
        times,positions,bounds=quintic_round_trip(starts,goals)
        attempt.update(middle_joints_rad=np.asarray(goals).tolist(),kinematics=bounds)
        if (bounds['total_serial_preparation_settle_budget_steps']>200
                or bounds['root_stride10_motion_wait_steps']>165):
            attempt['status']='serial_budget_exceeded'; _save(EVIDENCE/'exploration.json',report); continue
        left,right=_trajectories(raw,times,positions)
        holds=[ArmTrajectory(t.action_key,t.joint_names,[0.],[t.positions_rad[0]],1.) for t in (left,right)]
        checks={}
        for name,pair in (('left_other_at_start',(left,holds[1])),('right_other_at_start',(holds[0],right)),
                          ('synchronous',(left,right)),('right_waits_at_start',(left,delayed(right,left.duration_s)))):
            schedule=build_hold_schedule(*pair,control_timestep_s=DT,max_steps=200)
            checks[name]=validate_schedule(schedule,*starts,checker)
        attempt['checks']=checks
        conflict=checks['synchronous']['first_collision']
        interarm=bool(conflict and any({c['object_name1'],c['object_name2']}=={'left','right'} for c in conflict['collisions']))
        accepted=(checks['left_other_at_start']['status']=='passed' and checks['right_other_at_start']['status']=='passed'
            and checks['right_waits_at_start']['status']=='passed' and interarm)
        attempt['status']='selected' if accepted else 'collision_contract_failed'
        _save(EVIDENCE/'exploration.json',report)
        if not accepted: continue
        assert np.array_equal(as_numpy(raw.elapsed_steps),context['prepared_elapsed_steps'])
        assert not FIXTURE.exists() and not METADATA.exists()
        bases=np.asarray([as_numpy(a.robot.pose.raw_pose)[0] for a in raw.agent.agents])
        fingers=np.asarray([as_numpy(a.robot.get_qpos())[0,7:] for a in raw.agent.agents])
        np.savez(FIXTURE,times_s=times,positions_rad=positions,starts=starts,middles=np.asarray(goals),
            base_poses=bases,finger_qpos=fingers)
        middle_world=[]
        for i,model in enumerate(checker.models):
            qq=checker.native.full_qpos[i].copy(); qq[checker.native.arm_indices[i]]=goals[i]
            middle_world.append(_world_tcp(model,qq,'panda_hand_tcp').tolist())
        metadata={'fixture_version':'shared-middle-quintic-round-trip-v1','created_at':datetime.now(timezone.utc).isoformat(),
            'selected_attempt':index,'task':report['task'],'seed':0,'preparation_steps':10,
            'physics_steps_after_preparation':0,'source':bounds['source'],'kinematics':bounds,
            'fixture_npz_sha256':_sha(FIXTURE),'shared_middle_tcp_xyz_m':[x,0.,z],
            'middle_tcp_world_T':middle_world,'right_start_wait_s':left.duration_s,
            'joint_names':[list(t.joint_names) for t in (left,right)],'action_keys':[t.action_key for t in (left,right)],
            'gripper_commands':[1.,1.],'checks':checks,
            'scene_contact_permissions':[False,False],'physical_execution_performed':False,
            'collision_scope':'native full scene sampled at <=.02rad; not continuous collision or physical tracking',
            'environment_modifications':'none: original task/assets/roots/collision masks retained'}
        _save(METADATA,metadata); report.update(status='fixture_selected',selected_attempt=index,
            selected_fixture=str(FIXTURE),prepared_elapsed_steps_after_geometry=as_numpy(raw.elapsed_steps).tolist())
        _save(EVIDENCE/'exploration.json',report)
        return metadata
    report['status']='bounded_grid_exhausted'; _save(EVIDENCE/'exploration.json',report)
    raise RuntimeError('No fixture met all geometric/budget conditions in the bounded grid')


def self_test() -> dict[str,Any]:
    starts=np.zeros((2,7)); goals=np.zeros((2,7));goals[0,0]=.4;goals[1,1]=-.7
    times,positions,bounds=quintic_round_trip(starts,goals)
    assert times[0]==0 and np.all(np.diff(times)>0) and positions.shape==(2,len(times),7)
    assert np.array_equal(positions[:,0],starts) and np.array_equal(positions[:,-1],starts)
    assert np.allclose(positions[:,len(times)//2],goals)
    assert np.max(np.abs(np.diff(positions,axis=1)/np.diff(times)[None,:,None]))<=VMAX
    velocity=np.diff(positions,axis=1)/np.diff(times)[None,:,None]
    assert np.max(np.abs(np.diff(velocity,axis=1)/SOURCE_DT))<=AMAX
    trajectory=ArmTrajectory('left',tuple(f'j{i}' for i in range(7)),times,positions[0],1.)
    shifted=delayed(trajectory,trajectory.duration_s)
    assert np.array_equal(shifted.positions_rad[1:],trajectory.positions_rad)
    assert shifted.duration_s==2*trajectory.duration_s
    changed_start=starts+np.array([[.02]*7,[-.015]*7])
    new_times,new_positions,new_bounds=quintic_round_trip(changed_start,goals)
    assert np.array_equal(new_positions[:,0],changed_start)
    assert np.array_equal(new_positions[:,-1],changed_start)
    assert np.allclose(new_positions[:,len(new_times)//2],goals,rtol=0,atol=1e-15)
    assert max(new_bounds['max_velocity_rad_s'])<=VMAX and max(new_bounds['max_acceleration_rad_s2'])<=AMAX
    zero_times,zero_positions,_=quintic_round_trip(starts,starts)
    assert len(zero_times)>1 and np.all(zero_positions==0)
    return {'status':'passed','checks':['quintic source endpoint/turnaround','analytic and sampled derivative bounds',
        'start-wait preserves positions','fixed middle with changed exact measured start',
        'zero-displacement boundary'],'environment_constructions':0,'physics_steps':0}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test: print(json.dumps(self_test(),indent=2))
    else: parser.error('Use prepare_exploration()/explore() explicitly; no implicit environment execution')
