"""CPU kinematics and rejection-path tests; no simulated or claimed success."""
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from resimpair_grasp.common import read
from resimpair_grasp.motion import MeshMotion
from resimpair_pickplace import cartesian_transfer as candidate
from resimpair_pickplace.cartesian_reachability import ArmSelector, INITIAL
from data_generation_methods.human_to_robot_pipeline.strict_motion import world_fk, validate_joint_path


def test_native_ik_descent_preserves_local_corridor_and_requested_orientation():
    selector = ArmSelector([[-.6, .35, 0.]])
    root, planner = selector.roots[0], selector.planners[0]
    goal = world_fk(INITIAL[None, :7], root)[0]
    goal[2, 3] -= .06
    goal[:3, :3] = goal[:3, :3] @ Rotation.from_rotvec([.002, 0., 0.]).as_matrix()
    path = candidate.local_cartesian_path(planner, root, INITIAL, goal, 'grasp_reach')
    joints = np.asarray(path['position']); poses = world_fk(joints, root)
    assert validate_joint_path(joints, planner.joint_limits, INITIAL)['passed']
    assert np.linalg.norm(poses[-1, :3, 3] - goal[:3, 3]) < .002
    assert np.linalg.norm(poses[:, :2, 3] - poses[0, :2, 3], axis=1).max() < .002
    assert np.max(np.diff(poses[:, 2, 3])) < .001
    assert Rotation.from_matrix(poses[-1, :3, :3].T @ goal[:3, :3]).magnitude() < .01
    assert path['requires_retained_full_path_guard']


@pytest.mark.parametrize('stage', ['lift', 'retreat'])
def test_native_upward_proposals_reuse_local_chain(stage):
    selector = ArmSelector([[-.6, .35, 0.]])
    root, planner = selector.roots[0], selector.planners[0]
    goal = world_fk(INITIAL[None, :7], root)[0]; goal[2, 3] += .06
    path = candidate.local_cartesian_path(planner, root, INITIAL, goal, stage)
    poses = world_fk(np.asarray(path['position']), root)
    assert np.linalg.norm(poses[-1, :3, 3] - goal[:3, 3]) < .002
    assert np.min(np.diff(poses[:, 2, 3])) > -.001
    assert path['requires_retained_full_path_guard']


@pytest.mark.parametrize('problem,message', [
    ('sideways', 'vertical_grasp_reach'),
    ('upward', 'vertical_grasp_reach'),
    ('rotation', 'orientation_must_remain_fixed'),
    ('unsupported', 'unsupported_local_cartesian_stage'),
    ('nonfinite', 'invalid_cartesian_input'),
])
def test_bad_local_requests_reject_before_solver(problem, message):
    root = np.eye(4); goal = world_fk(INITIAL[None, :7], root)[0]; goal[2, 3] -= .06
    stage = 'grasp_reach'
    if problem == 'sideways': goal[0, 3] += .03
    elif problem == 'upward': goal[2, 3] += .12
    elif problem == 'rotation': goal[:3, :3] = goal[:3, :3] @ Rotation.from_rotvec([.1, 0., 0.]).as_matrix()
    elif problem == 'unsupported': stage = 'above_bowl'
    elif problem == 'nonfinite': goal[0, 3] = np.nan
    with pytest.raises(ValueError, match=message):
        candidate.local_cartesian_path(None, root, INITIAL, goal, stage)


def test_retained_guard_rejection_cannot_return_cartesian_proposal(tmp_path, monkeypatch):
    guard = candidate.CartesianTransferMotion.__new__(candidate.CartesianTransferMotion)
    pose = SimpleNamespace(to_transformation_matrix=lambda: np.eye(4))
    guard.agent = SimpleNamespace(robot=SimpleNamespace(qpos=INITIAL.copy()))
    guard.selected_actor = SimpleNamespace(pose=pose)
    guard.use_attach = False; guard.directory = tmp_path; guard.counter = 0; guard.root = np.eye(4); guard.planner = None
    guard.rec = SimpleNamespace(rows=[], control_timestep=.05,
                               performance=SimpleNamespace(section=lambda _: nullcontext()))
    guard.check = lambda *_: ({'passed': False, 'collision_failures': ['target']}, None, None)
    monkeypatch.setattr(candidate, 'local_cartesian_path', lambda *_args, **_kwargs:
                        {'status': 'Success', 'position': np.vstack([INITIAL[:7], INITIAL[:7]])})
    def reject_fallback(*_args, **_kwargs):
        raise RuntimeError('retained_fallback_rejected')
    monkeypatch.setattr(MeshMotion, 'plan', reject_fallback)
    with pytest.raises(RuntimeError, match='retained_fallback_rejected'):
        guard.plan(pose, 'grasp_reach')
    decision = read(tmp_path / '000_grasp_reach_cartesian/decision.json')
    assert not decision['accepted'] and decision['report']['collision_failures'] == ['target']
