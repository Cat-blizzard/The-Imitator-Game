"""Mode-shape compatibility repair for the frozen v1 closure policy.

Validate the actual two Panda controllers and resolve MultiAgent's per-UID
mode dict. Temporarily adapt only the Solver's dispatch attribute for v1;
restore the same object in finally. Never change native controller modes,
configs, gains, physics, the policy schedule or acceptance thresholds.
"""
from contextlib import contextmanager
from copy import deepcopy
import numpy as np
from resimpair_grasp.common import binding,write
from . import contact_latched_closure_v1 as retained

VERSION='cached_width_contact_latched_closure_mode_adapter_v2'
POLICY=retained.POLICY


def validate_mode_binding(rec,motion):
    if motion.env is not rec or type(motion.multi_robot_id) is not int or motion.multi_robot_id not in (0,1):
        raise ValueError('closure_v2_requires_bound_recorder_and_dual_arm_index')
    group=rec.agent
    if len(group.agents)!=2:
        raise ValueError('closure_v2_requires_exactly_two_agents')
    expected={f'{agent.uid}-{i}':agent for i,agent in enumerate(group.agents)}
    if set(expected)!={'panda_wristcam-0','panda_wristcam-1'}:
        raise ValueError('closure_v2_requires_known_Panda_action_channel_UIDs')
    if set(group.agents_dict)!=set(expected) or any(group.agents_dict[k] is not a for k,a in expected.items()):
        raise ValueError('MultiAgent_UID_identity_mapping_mismatch')
    actual=group.control_mode
    if type(actual) is not dict or set(actual)!=set(expected):
        raise ValueError('actual_MultiAgent_per_UID_control_modes_required')
    solver_mode=motion.control_mode
    if type(solver_mode) is dict:
        if solver_mode != actual:
            raise ValueError('solver_control_mode_dict_differs_from_native_agents')
        representation='MultiAgent_per_UID_dict'
    elif type(solver_mode) is str:
        if solver_mode!='pd_joint_pos' or any(v!=solver_mode for v in actual.values()):
            raise ValueError('scalar_solver_mode_must_match_both_native_agents')
        representation='uniform_scalar_compatibility'
    else:
        raise ValueError('unsupported_solver_control_mode_type')
    if any(v!='pd_joint_pos' for v in actual.values()):
        raise ValueError('both_native_arms_must_use_absolute_pd_joint_pos')
    selected=group.agents[motion.multi_robot_id]
    if motion.env_agent is not selected or motion.robot is not selected.robot:
        raise ValueError('solver_arm_or_robot_identity_mismatch')
    rows=[]
    for uid,agent in expected.items():
        if agent.control_mode!=actual[uid]:raise ValueError('native_agent_mode_mismatch')
        components=agent.controller.controllers
        if set(components)!={'arm','gripper'}:
            raise ValueError('expected_native_arm_and_gripper_controller_components')
        arm,grip=components['arm'].config,components['gripper'].config
        if arm.use_delta or arm.normalize_action or len(arm.joint_names)!=7:
            raise ValueError('arm_requires_absolute_unnormalized_seven_joint_positions')
        if grip.use_delta or not grip.normalize_action or len(grip.joint_names)!=2:
            raise ValueError('gripper_requires_absolute_normalized_two_joint_mimic_positions')
        lo,hi=np.asarray(grip.lower,float),np.asarray(grip.upper,float)
        if lo.shape!=() or hi.shape!=() or not np.isfinite([lo,hi]).all() or not lo<=0 or not hi>=.04 or hi<=lo:
            raise ValueError('native_gripper_scalar_bounds_do_not_cover_Panda_aperture')
        rows.append({'uid':uid,'control_mode':agent.control_mode,
            'controller_types':{k:type(v).__name__ for k,v in components.items()},
            'arm':{'joint_names':list(arm.joint_names),'normalize_action':bool(arm.normalize_action),'use_delta':bool(arm.use_delta)},
            'gripper':{'joint_names':list(grip.joint_names),'lower':float(lo),'upper':float(hi),
                'normalize_action':bool(grip.normalize_action),'use_delta':bool(grip.use_delta),
                'stiffness':grip.stiffness,'damping':grip.damping,'force_limit':grip.force_limit,'friction':grip.friction}})
    return {'schema':VERSION,'adapter':binding(__file__),'retained_policy_implementation':binding(retained.__file__),
        'solver_mode_representation':representation,'solver_mode_original':deepcopy(solver_mode),
        'actual_native_control_modes':dict(actual),'selected_arm':motion.multi_robot_id,
        'native_controller_configuration':rows,'native_controller_mutated':False,
        'policy_identical_to_frozen_v1':True}


def call_validated(rec,motion,callback,*args,**kwargs):
    evidence=validate_mode_binding(rec,motion)
    prior=motion.control_mode
    try:
        motion.control_mode='pd_joint_pos'
        return callback(*args,**kwargs),evidence
    finally:
        motion.control_mode=prior


@contextmanager
def install(rec,task,policy=POLICY):
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver as Solver
    with retained.install(rec,task,policy):
        v1_close=Solver.close_gripper
        def close(motion,other_gripper_state=None,t=6):
            opid=rec.stage.rsplit(':',1)[0]
            folder=rec.directory/'contact_latched_closure_v2'
            before=motion.control_mode
            try:
                proof=validate_mode_binding(rec,motion)
                write(folder/(opid+'_entry.json'),proof)
                result,_=call_validated(rec,motion,lambda:v1_close(motion,other_gripper_state=other_gripper_state,t=t))
                write(folder/(opid+'_entry_result.json'),{'schema':VERSION,'status':'returned',
                    'solver_mode_object_restored':motion.control_mode is before})
                return result
            except Exception as exc:
                write(folder/(opid+'_entry_result.json'),{'schema':VERSION,'status':'raised',
                    'exception':{'type':type(exc).__name__,'message':str(exc)},
                    'solver_mode_object_restored':motion.control_mode is before})
                raise
        Solver.close_gripper=close
        try:yield
        finally:Solver.close_gripper=v1_close
