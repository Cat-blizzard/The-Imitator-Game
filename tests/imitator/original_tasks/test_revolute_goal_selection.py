"""Limit-valid periodic IK choices must get FK checked and fully replanned."""
from types import SimpleNamespace as NS
from pathlib import Path

import numpy as np
import pytest

from imitator.planning._tig_sync.planning_endpoint import (
    checked_screw_plan, nearest_revolute_goal,
)


def test_revolute_equivalents_obey_finite_limits_and_never_wrap_prismatic_joints():
    period = 2*np.pi
    selected = nearest_revolute_goal([.2-period, -3.13, 6.4], [.1, 3.13, .1],
        ("revolute", "revolute", "prismatic"), [[-period, period], [-3.14, 3.14], [-10, 10]])
    assert np.allclose(selected, [.2, -3.13, 6.4])
    # Piper's legal -3.13 endpoint has no equivalent inside the +3.14 limit.
    assert selected[1] < 0 and abs(selected[1]-3.13) > 6
    with pytest.raises(ValueError, match="equivalent"):
        nearest_revolute_goal([3.5], [0.], ("revolute",), [[-1., 1.]])
    with pytest.raises(ValueError, match="exceeds"):
        nearest_revolute_goal([6.4], [0.], ("prismatic",), [[-1., 1.]])


class PeriodicSolver:
    def __init__(self, tmp_path, *, enabled=True, invalid_fk=False):
        self.state = np.array([.1, .031])
        self.period, self.goal = 2*np.pi, .2
        self.invalid_fk = invalid_fk
        self.paths, self.replans = [], []
        self.env_agent = NS(prefer_nearby_revolute_goals=enabled)
        self.base_env = NS(control_timestep=.05)
        self.use_point_cloud = False
        urdf = tmp_path/'arm.urdf'
        urdf.write_text('<robot name="fixture"><joint name="hinge" type="revolute"/></robot>')
        self.robot = NS(get_qpos=lambda: self.state[None].copy(),
            get_qlimits=lambda: np.array([[[-self.period,self.period],[0.,.04]]]),
            get_active_joints=lambda: [NS(get_name=lambda name=name:name) for name in ('hinge','finger')])
        self.planner = NS(urdf=str(urdf), user_joint_names=['hinge','finger'],
            move_group_joint_indices=[0], move_group='tcp', link_name_2_idx={'tcp':0},
            transform_goal_to_wrt_base=lambda pose:pose.copy(), TOPP=self.topp,
            planning_world=NS(use_attach=False), plan_qpos_to_qpos=self.replan,
            pinocchio_model=NS(compute_forward_kinematics=self.fk,get_link_pose=lambda _:self.actual))

    def fk(self, full):
        assert full[1] == self.state[1]
        q=full[0]
        offset=.01 if self.invalid_fk and q > -np.pi else 0.
        self.actual=np.array([offset,0.,0.,np.cos(q/2),0.,0.,np.sin(q/2)])

    def topp(self,path,step,verbose=False):
        path=np.array(path,copy=True)
        self.paths.append(path)
        return np.arange(len(path))*.05,path,np.zeros_like(path),np.zeros_like(path),.05*(len(path)-1)

    def move_to_pose_with_screw(self,pose,*,dry_run):
        assert dry_run
        path=np.array([[self.state[0]],[self.goal-self.period]])
        raw=self.planner.TOPP(path,.05)
        return dict(status='Success',**dict(zip(('time','position','velocity','acceleration','duration'),raw)))

    def replan(self,goals,start,**options):
        assert np.array_equal(start,self.state)
        assert goals[0][1] == self.state[1]
        assert options['fix_joint_limits'] is False
        self.replans.append(goals[0].copy())
        # A real native geometric path can have intermediate vertices: the
        # adapter must retain these rather than just rewrite an old endpoint.
        path=np.array([[start[0]],[.15],[goals[0][0]]])
        return self.planner.TOPP(path,.05)

    def pose(self):
        return np.array([0.,0.,0.,np.cos(self.goal/2),0.,0.,np.sin(self.goal/2)])


def test_changed_periodic_goal_replans_entire_native_path_and_preserves_other_joints(tmp_path):
    solver=PeriodicSolver(tmp_path)
    result=checked_screw_plan(solver,solver.pose())
    audit=result['endpoint_audit']
    assert len(solver.replans) == 1 and len(solver.paths) == 2
    assert np.isclose(solver.replans[0][0],.2)
    assert np.allclose(result['position'][:,0],[.1,.15,.2])
    assert result['position'].shape == (3,1)
    assert audit['fresh_native_joint_replan'] and audit['corrections'] == 0
    assert audit['periodic_goal_selection']['actual_urdf_types_and_limits_checked']
    assert audit['periodic_goal_selection']['changes'][0]['joint'] == 'hinge'
    assert audit['final_error']['rotation_rad'] < 1e-7
    assert np.array_equal(solver.state,[.1,.031])


def test_planner_advertises_periodic_selection_without_mutating_native_agent(tmp_path):
    solver = PeriodicSolver(tmp_path, enabled=False)
    solver.prefer_nearby_revolute_goals = True
    result = checked_screw_plan(solver, solver.pose())
    assert result['endpoint_audit']['periodic_goal_selection']['enabled']
    assert len(solver.replans) == 1 and np.isclose(solver.replans[0][0], .2)
    assert not solver.env_agent.prefer_nearby_revolute_goals


def test_periodic_goal_fk_mismatch_is_rejected_before_replanning(tmp_path):
    solver=PeriodicSolver(tmp_path,invalid_fk=True)
    with pytest.raises(ValueError,match='Equivalent revolute goal exceeds FK'):
        checked_screw_plan(solver,solver.pose())
    assert solver.replans == []
    assert solver.last_endpoint_audit['status'] == 'failed'


def test_opt_out_preserves_existing_native_result(tmp_path):
    solver=PeriodicSolver(tmp_path,enabled=False)
    result=checked_screw_plan(solver,solver.pose())
    assert np.isclose(result['position'][-1,0],.2-2*np.pi)
    assert solver.replans == [] and len(solver.paths) == 1
    assert 'periodic_goal_selection' not in result['endpoint_audit']


def test_urdf_joint_type_is_used_instead_of_treating_every_coordinate_as_periodic(tmp_path):
    solver=PeriodicSolver(tmp_path)
    Path(solver.planner.urdf).write_text(
        '<robot name="fixture"><joint name="hinge" type="prismatic"/></robot>')
    result=checked_screw_plan(solver,solver.pose())
    assert np.isclose(result['position'][-1,0],.2-2*np.pi)
    assert solver.replans == []
    assert result['endpoint_audit']['periodic_goal_selection']['changes'] == []
