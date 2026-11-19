from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from resimpair_grasp.common import binding,write,read
from resimpair_pickplace import contact_latched_closure_v1 as c


def candidate(width=.02):
    return {'id':'p0000','width_m':width,'closing_sweep':{'contact_pair_reachable':True,
        'width_tolerance_m':.004,'fingers':[{'first_contact_width_m':width},{'first_contact_width_m':width}]}}


def test_width_schedule_stops_at_native_bounded_floor_never_negative_full_close():
    s=c.ClosureSchedule(candidate(),.08,-.01,.04)
    commands=[s.next_command(.08,False) for _ in range(24)]
    assert all(x['full_aperture_target_m']>=.016-1e-12 for x in commands)
    assert all(x['normalized_command']>-1 for x in commands)
    with pytest.raises(RuntimeError,match='no_persistent_native_grasp'):
        s.finish(.016,False)


def test_first_native_contact_latches_then_never_resqueezes_on_contact_loss():
    s=c.ClosureSchedule(candidate(),.08,-.01,.04)
    for _ in range(6):s.next_command(.08,False)
    first=s.next_command(.0215,True)
    later=[s.next_command(.0215,False) for _ in range(17)]
    assert {x['full_aperture_target_m'] for x in later}=={first['full_aperture_target_m']}
    with pytest.raises(RuntimeError,match='no_persistent_native_grasp'):
        s.finish(.0215,False)


def test_invalid_cache_and_nonfinite_feedback_fail_closed():
    bad=candidate();bad['closing_sweep']['fingers'][0]['first_contact_width_m']=float('nan')
    with pytest.raises(ValueError):c.ClosureSchedule(bad,.08,-.01,.04)
    s=c.ClosureSchedule(candidate(),.08,-.01,.04)
    with pytest.raises(ValueError):s.next_command(float('nan'),False)
    with pytest.raises(RuntimeError,match='below_bounded_cache_squeeze'):
        s.next_command(.013,True)


def test_context_keeps_original_hold_checks_but_preserves_latched_action_until_open(tmp_path,monkeypatch):
    # Interface test only: this fake feedback is never reported as physics.
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver as Solver
    original_close=Solver.close_gripper
    cache=tmp_path/'cache.json';write(cache,{'candidates':[candidate()]})
    write(tmp_path/'op_grasp_selection.json',{'selected':'p0000'})
    config=SimpleNamespace(lower=-.01,upper=.04,stiffness=1000.,damping=100.,force_limit=100.,friction=0.,
        normalize_action=True,use_delta=False,drive_mode='force')
    agents=[SimpleNamespace(robot=SimpleNamespace(qpos=np.r_[np.zeros(7),.04,.04]),controller=SimpleNamespace(controllers={'gripper':SimpleNamespace(config=config)})) for _ in range(2)]
    class Recorder:
        def __init__(self):
            self.stage='op:close';self.directory=tmp_path;self.object_ids=['obj'];self.objects=[object()]
            self.scene_spec={'grasp_caches':{'obj':binding(cache)}};self.agent=SimpleNamespace(agents=agents)
            self.rows=[];self.actions=[]
        def step(self,action):
            self.actions.append(deepcopy(action));self.rows.append({});return (None,None,None,None,None)
    rec=Recorder();motion=SimpleNamespace(env=rec,control_mode='pd_joint_pos',multi_robot_id=0,elapsed_steps=0,gripper_state=1)
    monkeypatch.setattr(c,'_array',np.asarray)
    def feedback(agent,actor):
        w=.08 if len(rec.rows)==0 else (.0198 if len(rec.rows)>=8 else .024)
        return {'aperture_m':w,'qpos':np.r_[np.zeros(7),w/2,w/2].tolist(),
            'native_grasp':len(rec.rows)>=8,'finger_object_force_world_N':[[0,1,0],[0,-1,0]]}
    monkeypatch.setattr(c,'_native_feedback',feedback)
    with c.install(rec,{'operations':[{'id':'op','object':'obj'}]}):
        Solver.close_gripper(motion,other_gripper_state=1,t=12)
        assert len(rec.rows)==24 and motion.gripper_state>-1
        rec.stage='op:lift_settle';rec.step({'panda_wristcam-0':np.r_[np.zeros(7),-1.], 'panda_wristcam-1':np.r_[np.zeros(7),1.]})
        assert rec.actions[-1]['panda_wristcam-0'][-1]==motion.gripper_state
        rec.stage='op:open';rec.step({'panda_wristcam-0':np.r_[np.zeros(7),1.], 'panda_wristcam-1':np.r_[np.zeros(7),1.]})
        assert rec.actions[-1]['panda_wristcam-0'][-1]==1
    assert Solver.close_gripper is original_close and 'step' not in rec.__dict__
    assert read(tmp_path/'contact_latched_closure/op_hold.json')['legacy_full_close_commands_replaced']==1
