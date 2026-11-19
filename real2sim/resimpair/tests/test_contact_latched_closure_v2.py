from types import SimpleNamespace
from copy import deepcopy
import pytest
from resimpair_pickplace import contact_latched_closure_v2 as c
from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.panda.panda import Panda


def fixture():
    # Exercise the real MultiAgent.control_mode property and real Panda config
    # dataclasses; no actor, simulation, GPU or fake success trajectory.
    template=object.__new__(Panda)
    agents=[]
    for _ in range(2):
        cfg=deepcopy(template._controller_configs['pd_joint_pos'])
        agents.append(SimpleNamespace(uid='panda_wristcam',scene=None,control_mode='pd_joint_pos',robot=object(),
            controller=SimpleNamespace(controllers={k:SimpleNamespace(config=v) for k,v in cfg.items()})))
    group=MultiAgent(agents);rec=SimpleNamespace(agent=group)
    motion=SimpleNamespace(env=rec,multi_robot_id=0,control_mode=group.control_mode,env_agent=agents[0],robot=agents[0].robot)
    return rec,motion


def test_real_MultiAgent_dict_property_matches_each_bound_native_agent_config():
    rec,m=fixture();proof=c.validate_mode_binding(rec,m)
    assert proof['solver_mode_representation']=='MultiAgent_per_UID_dict'
    assert proof['actual_native_control_modes']=={'panda_wristcam-0':'pd_joint_pos','panda_wristcam-1':'pd_joint_pos'}
    m.multi_robot_id=1;m.env_agent=rec.agent.agents[1];m.robot=m.env_agent.robot
    assert c.validate_mode_binding(rec,m)['selected_arm']==1


@pytest.mark.parametrize('arm',[0,1])
def test_incompatible_mode_on_either_arm_is_rejected(arm):
    rec,m=fixture();rec.agent.agents[arm].control_mode='pd_joint_delta_pos';m.control_mode=rec.agent.control_mode
    with pytest.raises(ValueError,match='both_native_arms'):c.validate_mode_binding(rec,m)


def test_scalar_mode_is_legal_only_when_native_modes_and_configs_match():
    rec,m=fixture();m.control_mode='pd_joint_pos'
    assert c.validate_mode_binding(rec,m)['solver_mode_representation']=='uniform_scalar_compatibility'
    m.control_mode='pd_joint_pos_vel'
    with pytest.raises(ValueError,match='scalar_solver_mode'):c.validate_mode_binding(rec,m)


@pytest.mark.parametrize('field,value',[('normalize_action',False),('use_delta',True),('upper',.02),('lower',float('nan'))])
def test_gripper_configuration_is_validated_instead_of_bypassed(field,value):
    rec,m=fixture();setattr(rec.agent.agents[0].controller.controllers['gripper'].config,field,value)
    with pytest.raises(ValueError):c.validate_mode_binding(rec,m)


def test_mode_object_and_native_modes_restore_on_return_and_exception():
    rec,m=fixture();prior=m.control_mode;native=dict(rec.agent.control_mode)
    def ok():
        assert m.control_mode=='pd_joint_pos';assert rec.agent.control_mode==native;return 'executed-interface-only'
    result,proof=c.call_validated(rec,m,ok)
    assert result=='executed-interface-only' and m.control_mode is prior
    def bad():
        assert m.control_mode=='pd_joint_pos';raise RuntimeError('deliberate_before_any_physics')
    with pytest.raises(RuntimeError,match='deliberate_before_any_physics'):c.call_validated(rec,m,bad)
    assert m.control_mode is prior and rec.agent.control_mode==native


def test_real_dict_property_enters_frozen_v1_context_and_restores_it(tmp_path,monkeypatch):
    import numpy as np
    from resimpair_grasp.common import binding,write,read
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver as Solver
    rec,m=fixture();prior=m.control_mode;original=Solver.close_gripper
    for agent in rec.agent.agents:agent.robot=SimpleNamespace(qpos=np.r_[np.zeros(7),.04,.04])
    m.robot=m.env_agent.robot;m.elapsed_steps=0;m.gripper_state=1
    rec.directory=tmp_path;rec.stage='op:close';rec.object_ids=['obj'];rec.objects=[object()];rec.rows=[]
    candidate={'id':'p','width_m':.02,'closing_sweep':{'contact_pair_reachable':True,'width_tolerance_m':.004,
       'fingers':[{'first_contact_width_m':.02},{'first_contact_width_m':.02}]}}
    write(tmp_path/'cache.json',{'candidates':[candidate]});write(tmp_path/'op_grasp_selection.json',{'selected':'p'})
    rec.scene_spec={'grasp_caches':{'obj':binding(tmp_path/'cache.json')}}
    def step(action):rec.rows.append({});return (None,None,None,None,None)
    rec.step=step
    monkeypatch.setattr(c.retained,'_array',np.asarray)
    def feedback(agent,obj):
        w=.08 if not rec.rows else (.0198 if len(rec.rows)>=8 else .024)
        return {'aperture_m':w,'qpos':np.r_[np.zeros(7),w/2,w/2].tolist(),'native_grasp':len(rec.rows)>=8,
                'finger_object_force_world_N':[[0,1,0],[0,-1,0]]}
    monkeypatch.setattr(c.retained,'_native_feedback',feedback)
    with c.install(rec,{'operations':[{'id':'op','object':'obj'}]}):
        Solver.close_gripper(m,other_gripper_state=1,t=12)
        assert len(rec.rows)==24 and m.control_mode is prior
    assert rec.step is step and Solver.close_gripper is original
    assert read(tmp_path/'contact_latched_closure_v2/op_entry.json')['solver_mode_representation']=='MultiAgent_per_UID_dict'
    assert read(tmp_path/'contact_latched_closure_v2/op_entry_result.json')['solver_mode_object_restored']
