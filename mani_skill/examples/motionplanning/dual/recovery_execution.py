"""Pure evidence rules for recovery from actual dual-Panda failures.

Positions are world XYZ metres, rotations WXYZ, and pair index0/1 is left/right.
No function creates an environment, changes a pose/state, or advances physics.
A400-tick engineering capability budget is separate from official200 M1 results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any

import numpy as np

SCHEMA = 'm1-actual-state-recovery-recipe-v1'
CAPABILITY_BUDGET = 400
RECOVERY_INTERFACE_VERSION = 'm1-actual-state-recovery-interface-v02'
MAX_ACTUAL_TO_TAIL_GAP_RAD = .03


def finite(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f'{label}: expected finite{shape}')
    return result


def validate_recipe(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get('schema') != SCHEMA:
        raise ValueError('Explicit frozen recovery recipe schema required')
    offsets = finite(value.get('grasp_miss_offsets_world_m'), (2, 3), 'fixed grasp offsets')
    if not np.array_equal(offsets[1], np.zeros(3)) or not np.isclose(offsets[0, 2], 0.):
        raise ValueError('Only the left horizontal grasp target may be deliberately offset')
    if not .06 <= np.linalg.norm(offsets[0, :2]) <= .12:
        raise ValueError('Miss offset must have a fixed6..12cm horizontal norm')
    if value.get('budget_control_ticks') != CAPABILITY_BUDGET or value.get('max_regrasp_attempts') != 1:
        raise ValueError('Frozen400-tick budget and one actual regrasp attempt required')
    if value.get('upward_recovery_retreat_m') != .08:
        raise ValueError('Recovery retains an8cm upward commanded retreat')
    if value.get('progress_target_distance_m') != .06 or value.get('minimum_progress_ticks') != 5 or value.get('minimum_actual_progress_m') != .05:
        raise ValueError('Frozen6cm prefix proposal and real5tick/5cm evidence required')
    return {'schema': SCHEMA, 'grasp_miss_offsets_world_m': offsets.tolist(),
            'budget_control_ticks': CAPABILITY_BUDGET, 'max_regrasp_attempts': 1,
            'upward_recovery_retreat_m': .08, 'progress_target_distance_m': .06,
            'minimum_progress_ticks': 5, 'minimum_actual_progress_m': .05}


def default_recipe() -> dict[str, Any]:
    return validate_recipe({'schema': SCHEMA, 'grasp_miss_offsets_world_m': [[.09, 0., 0.], [0., 0., 0.]],
        'budget_control_ticks':400, 'max_regrasp_attempts':1, 'upward_recovery_retreat_m':.08,
        'progress_target_distance_m':.06, 'minimum_progress_ticks':5, 'minimum_actual_progress_m':.05})


def array_identity(values: dict[str, Any]) -> str:
    """Exact sorted leaf/dtype/shape/bytes identity; no ZIP timestamp dependence."""
    digest = hashlib.sha256()
    if not values: raise ValueError('A real checkpoint requires nonempty state arrays')
    for name, value in sorted(values.items()):
        array = np.asarray(value)
        if array.dtype.kind not in 'biuf' or not np.isfinite(array).all():
            raise ValueError('Checkpoint arrays must be finite and numeric')
        header = json.dumps([name,array.dtype.str,list(array.shape)],separators=(',',':')).encode()
        payload = np.ascontiguousarray(array).tobytes()
        digest.update(len(header).to_bytes(8,'big'));digest.update(header)
        digest.update(len(payload).to_bytes(8,'big'));digest.update(payload)
    return digest.hexdigest()


def source_release_certificate(grasped: Any, source_contacts: Any, original_z: Any, measured_z: Any) -> dict[str, Any]:
    """Opening a still-grasped peer requires actual native source contact.

    A peer is never released in the air. The3mm limit is an additional recovery
    eligibility condition, not a collision/contact tolerance. Original native
    source contact uses separation<=0 OR nonzero impulse, with no new threshold.
    """
    grasps, contacts = np.asarray(grasped), np.asarray(source_contacts)
    if grasps.shape != (2,) or contacts.shape != (2,) or not np.isin(grasps,[False,True]).all() or not np.isin(contacts,[False,True]).all():
        raise ValueError('Two actual grasp/source-contact predicates required')
    gains = finite(measured_z,(2,),'actual origin heights')-finite(original_z,(2,),'pre-grasp source origin heights')
    eligible = [bool(not grasps[i] or contacts[i] and gains[i] <= .003) for i in range(2)]
    return {'status':'passed' if all(eligible) else 'refused', 'actual_grasped':grasps.astype(bool).tolist(),
            'actual_source_contacts':contacts.astype(bool).tolist(), 'origin_gains_m':gains.tolist(),
            'eligible_to_open':eligible, 'maximum_unlifted_origin_gain_m':.003,
            'scope':'only still-source-supported objects can be intentionally opened before regrasp'}


def proposed_progress_point(start_xyz: Any, nominal_middle_xyz: Any) -> np.ndarray:
    start = finite(start_xyz,(3,),'measured post-lift TCP');middle = finite(nominal_middle_xyz,(3,),'fixed nominal middle')
    distance = float(np.linalg.norm(middle-start))
    if distance < .10: raise ValueError('Frozen nominal middle must be at least10cm from post-lift start')
    result = start+.06*(middle-start)/distance
    assert result.shape == (3,) and np.isclose(np.linalg.norm(result-start),.06)
    return result


def actual_progress_certificate(initial_tcp: Any, measured_tcp: Any, initial_object: Any,
                                measured_object: Any, control_ticks: int) -> dict[str, Any]:
    if isinstance(control_ticks,bool) or not isinstance(control_ticks,(int,np.integer)) or control_ticks < 0:
        raise ValueError('Actual nonnegative integer progress tick count required')
    tcp = float(np.linalg.norm(finite(measured_tcp,(3,),'measured TCP')-finite(initial_tcp,(3,),'initial TCP')))
    obj = float(np.linalg.norm(finite(measured_object,(3,),'measured object')-finite(initial_object,(3,),'initial object')))
    return {'status':'passed' if control_ticks >= 5 and min(tcp,obj) >= .05 else 'failed',
            'executed_control_ticks':int(control_ticks), 'actual_tcp_displacement_m':tcp,
            'actual_object_displacement_m':obj, 'minimum_control_ticks':5,'minimum_actual_displacement_m':.05,
            'scope':'both actual carried object and TCP must progress before the new blocker is introduced'}


def source_start_certificate(source_qpos: Any, measured_qpos: Any) -> dict[str, Any]:
    source = finite(source_qpos,(2,7),'planned source start');measured = finite(measured_qpos,(2,7),'measured source start')
    error = float(np.max(np.abs(source-measured)))
    return {'status':'passed' if error <= 1e-6 else 'failed','max_joint_difference_rad':error,
            'limit_rad':1e-6,'scope':'new native replan starts at actual measured arm joints'}


def connect_actual_start_to_tail(source: Any, measured_qpos: Any, *,
                                 velocity_limit: float, acceleration_limit: float,
                                 control_timestep_s: float=.05) -> tuple[Any, Any, dict[str, Any]]:
    """An explicit query connector followed by the EXACT saved original tail.

    The connector is never silently substituted for a native spatial replan.
    It only makes old-tail rejection/wait queries start at the real held state.
    Its <=.03rad discrepancy bound is the existing endpoint acceptance bound.
    The native tail's positions, relative times, joint names and gripper remain
    intact, with one explicit constant absolute-time offset. Callers must check
    the whole connector and connected route before claiming actual blockage.
    """
    from .trajectory_schedule import ArmTrajectory
    measured = finite(measured_qpos, (7,), 'actual post-blocker arm joints')
    if (source.gripper_command != -1. or len(source.times_s) < 2
            or not np.isfinite([velocity_limit,acceleration_limit,control_timestep_s]).all()
            or min(velocity_limit,acceleration_limit) <= 0
            or not np.isclose(control_timestep_s,.05,rtol=0,atol=1e-12)):
        raise ValueError('A saved moving CLOSED tail and the frozen20Hz positive limits are required')
    original_times = np.asarray(source.times_s).copy()
    original_positions = np.asarray(source.positions_rad).copy()
    gap = original_positions[0] - measured
    maximum = float(np.abs(gap).max())
    if maximum > MAX_ACTUAL_TO_TAIL_GAP_RAD:
        raise ValueError('Actual-to-old-tail join exceeds the original .03rad endpoint bound')
    # Quintic zero-speed/zero-acceleration endpoints; these exact extrema bound
    # its generating curve. The checker still samples the executed command
    # polygon, without a continuous collision/dynamics claim.
    minimum_duration = max(1.875 * maximum / velocity_limit,
                           np.sqrt((10*np.sqrt(3)/3) * maximum / acceleration_limit))
    steps = max(1,int(np.ceil(minimum_duration/control_timestep_s)))
    times = control_timestep_s*np.arange(steps+1)
    u = times/times[-1]
    blending = 10*u**3 - 15*u**4 + 6*u**5
    positions = measured + blending[:,None]*gap
    positions[0] = measured; positions[-1] = original_positions[0]
    connector = ArmTrajectory(source.action_key,source.joint_names,times,positions,source.gripper_command)
    offset = connector.duration_s
    joined = ArmTrajectory(source.action_key,source.joint_names,
        np.r_[times[:-1],offset+original_times],
        np.vstack([positions[:-1],original_positions]),source.gripper_command)
    tail_positions_exact = np.array_equal(joined.positions_rad[steps:],original_positions)
    tail_times_exact_with_offset = np.array_equal(joined.times_s[steps:],offset+original_times)
    if not tail_positions_exact or not tail_times_exact_with_offset:
        raise RuntimeError('Connector construction altered the saved native spatial tail')
    receipt = {'version':RECOVERY_INTERFACE_VERSION,'status':'passed','query_only_not_executed':True,
        'execution_permitted':False,
        'proof_scope':'read-only counterfactual geometry; connector is never dispatched and is not a whole-join executable kinematic certificate',
        'actual_start_qpos_rad':measured.tolist(),'saved_tail_start_qpos_rad':original_positions[0].tolist(),
        'max_actual_to_old_tail_gap_rad':maximum,'maximum_join_gap_rad':MAX_ACTUAL_TO_TAIL_GAP_RAD,
        'connector_control_ticks':steps,'connector_duration_s':offset,'tail_absolute_time_offset_s':offset,
        'connector_duration_policy':'ceil exact quintic velocity/acceleration minimum duration to20Hz ticks; unchanged native tail not retimed',
        'tail_start_array_index':steps,'tail_positions_exact':tail_positions_exact,
        'tail_times_exact_with_constant_offset':tail_times_exact_with_offset,
        'saved_tail_arrays_sha256':array_identity({'time':original_times,'position':original_positions}),
        'joint_names':list(source.joint_names),'gripper_command':source.gripper_command,
        'connector_curve_velocity_peak_rad_s':float(1.875*maximum/offset),
        'connector_curve_acceleration_peak_rad_s2':float((10*np.sqrt(3)/3)*maximum/offset**2),
        'frozen_velocity_limit_rad_s':float(velocity_limit),'frozen_acceleration_limit_rad_s2':float(acceleration_limit),
        'sampling_scope':'all connector and unchanged-tail command segments require the common <=.02rad combined collision checker'}
    return joined,connector,receipt


def connected_blockage_certificate(connector_validation: dict, connected_validation: dict,
                                    connector_receipt: dict) -> dict[str, Any]:
    """A connector collision or unblocked tail cannot count as a blocker fault."""
    collision = connected_validation.get('first_collision')
    duration = float(connector_receipt['connector_duration_s'])
    after_connector = bool(collision is not None and float(collision['time_s']) > duration)
    passed = bool(connector_validation.get('status') == 'passed'
                  and connected_validation.get('status') == 'collision' and after_connector
                  and connector_receipt.get('tail_positions_exact') is True
                  and connector_receipt.get('tail_times_exact_with_constant_offset') is True)
    return {'status':'passed' if passed else 'failed','connector_clear':connector_validation.get('status')=='passed',
            'query_only_not_executed':True,'execution_permitted':False,
            'first_collision_strictly_in_unchanged_tail':after_connector,
            'connector_duration_s':duration,'first_collision_time_s':None if collision is None else collision['time_s'],
            'claim':'saved tail blocked from actual fault state through an explicitly clear connector' if passed else 'blocker fault from actual state not proven'}


def self_test() -> dict[str, Any]:
    recipe=default_recipe();assert recipe['budget_control_ticks']==400
    a={'actor':np.zeros((1,13),np.float32),'arm':np.ones((1,31),np.float32)}
    assert array_identity(a)==array_identity(dict(reversed(list(a.items()))))
    b={name:value.copy() for name,value in a.items()};b['actor'][0,0]=1e-8
    assert array_identity(a)!=array_identity(b)
    assert source_release_certificate([False,True],[False,True],[0,0],[0,.002])['status']=='passed'
    assert source_release_certificate([False,True],[False,False],[0,0],[0,.002])['status']=='refused'
    assert source_release_certificate([False,True],[False,True],[0,0],[0,.004])['status']=='refused'
    point=proposed_progress_point([0,0,.2],[.2,0,.2]);assert np.allclose(point,[.06,0,.2])
    assert actual_progress_certificate([0]*3,[.06,0,0],[0]*3,[.055,0,0],5)['status']=='passed'
    assert actual_progress_certificate([0]*3,[.06,0,0],[0]*3,[.049,0,0],5)['status']=='failed'
    assert actual_progress_certificate([0]*3,[.06,0,0],[0]*3,[.055,0,0],4)['status']=='failed'
    assert source_start_certificate(np.zeros((2,7)),np.zeros((2,7)))['status']=='passed'
    assert source_start_certificate(np.zeros((2,7)),np.ones((2,7))*.001)['status']=='failed'
    rejected=0
    for call in (lambda:validate_recipe({**recipe,'budget_control_ticks':200}),
                 lambda:validate_recipe({**recipe,'grasp_miss_offsets_world_m':[[.09,0,0],[.01,0,0]]}),
                 lambda:proposed_progress_point([0]*3,[.01,0,0]),
                 lambda:array_identity({'invalid':np.array([np.nan])}),
                 lambda:actual_progress_certificate([0]*3,[0]*3,[0]*3,[0]*3,True)):
        try:call()
        except ValueError:rejected+=1
        else:raise AssertionError('Malformed recovery evidence accepted')
    assert rejected==5
    from .trajectory_schedule import ArmTrajectory,build_hold_schedule
    from .combined_collision import validate_schedule
    names=tuple(f'j{i}' for i in range(7));q0=np.zeros(7);q1=q0.copy();q1[0]=1.
    tail=ArmTrajectory('left',names,[0.,1.],np.stack([q0,q1]),-1.)
    actual=q0.copy();actual[0]=-.005
    joined,connector,receipt=connect_actual_start_to_tail(tail,actual,velocity_limit=1.5,acceleration_limit=2.6)
    assert np.array_equal(tail.positions_rad,np.stack([q0,q1])) and np.array_equal(tail.times_s,[0.,1.])
    assert np.array_equal(joined.positions_rad[receipt['tail_start_array_index']:],tail.positions_rad)
    assert np.array_equal(joined.times_s[receipt['tail_start_array_index']:],tail.times_s+connector.duration_s)
    assert receipt['connector_curve_velocity_peak_rad_s']<=1.5 and receipt['connector_curve_acceleration_peak_rad_s2']<=2.6
    zero_joined,zero_connector,zero_receipt=connect_actual_start_to_tail(tail,q0,velocity_limit=1.5,acceleration_limit=2.6)
    assert zero_connector.duration_s==.05 and zero_receipt['connector_curve_acceleration_peak_rad_s2']==0.
    assert np.array_equal(zero_joined.positions_rad[1:],tail.positions_rad)
    limit_start=q0.copy();limit_start[0]=-.03
    _,_,high_acceleration=connect_actual_start_to_tail(tail,limit_start,velocity_limit=1.5,acceleration_limit=2.6)
    naive_one_tick_acceleration=(10*np.sqrt(3)/3)*.03/.05**2
    assert naive_one_tick_acceleration>2.6 and high_acceleration['connector_control_ticks']>1
    assert high_acceleration['connector_curve_acceleration_peak_rad_s2']<=2.6
    assert high_acceleration['connector_curve_velocity_peak_rad_s']<=1.5
    assert high_acceleration['execution_permitted'] is False
    peer=ArmTrajectory('right',names,[0.],[q0],-1.)
    class Collision:
        def check(self,a: Any,b: Any) -> list:
            return [{'object_name1':'apple','object_name2':'right'}] if .4<=a[0]<=.6 else []
    prefix_check=validate_schedule(build_hold_schedule(connector,peer,control_timestep_s=.05),actual,q0,Collision())
    complete_check=validate_schedule(build_hold_schedule(joined,peer,control_timestep_s=.05),actual,q0,Collision())
    assert connected_blockage_certificate(prefix_check,complete_check,receipt)['status']=='passed'
    try:validate_schedule(build_hold_schedule(tail,peer,control_timestep_s=.05),actual,q0,Collision())
    except ValueError:pass
    else:raise AssertionError('The old tail incorrectly accepted drifted actual start without an explicit connector')
    assert connected_blockage_certificate({'status':'collision'},complete_check,receipt)['status']=='failed'
    assert connected_blockage_certificate(prefix_check,{'status':'passed'},receipt)['status']=='failed'
    assert connected_blockage_certificate(prefix_check,{'status':'collision','first_collision':{'time_s':connector.duration_s/2}},receipt)['status']=='failed'
    try:connect_actual_start_to_tail(tail,actual-.04,velocity_limit=1.5,acceleration_limit=2.6)
    except ValueError:pass
    else:raise AssertionError('An excessive actual-to-saved-tail jump was accepted')
    for velocity,acceleration,dt in ((0.,2.6,.05),(1.5,float('nan'),.05),(1.5,2.6,.1)):
        try:connect_actual_start_to_tail(tail,actual,velocity_limit=velocity,acceleration_limit=acceleration,control_timestep_s=dt)
        except ValueError:pass
        else:raise AssertionError('Invalid connector limits/control clock accepted')
    return {'status':'passed','recipe_schema':SCHEMA,'checks':7,'malformed_inputs_rejected':rejected,
            'recovery_interface_version':RECOVERY_INTERFACE_VERSION,'actual_start_connector_boundaries':11,
            'high_acceleration_connector_explicit_duration_extension':high_acceleration,
            'shapes':{'paired_qpos':[2,7],'paired_offsets':[2,3],'single_XYZ':[3]},
            'environment_creations':0,'physics_steps':0,'native_recovery_acceptance':'NOT_RUN'}


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-test',action='store_true',required=True)
    parser.parse_args();print(json.dumps(self_test(),indent=2));return 0


if __name__=='__main__': raise SystemExit(main())
