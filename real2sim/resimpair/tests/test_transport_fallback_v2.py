"""Native CPU kinematics plus no-op success/fail-closed fallback boundaries."""
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from resimpair_grasp.common import read, write
from resimpair_pickplace import transport_fallback_v2 as candidate
from resimpair_pickplace.cartesian_transfer import CartesianTransferMotion
from resimpair_pickplace.cartesian_reachability import ArmSelector, INITIAL
from data_generation_methods.human_to_robot_pipeline.strict_motion import world_fk, validate_joint_path


def test_native_se3_ik_has_small_endpoint_error_and_no_large_joint_jump():
    selector = ArmSelector([[-.6, .35, 0.]])
    root, planner = selector.roots[0], selector.planners[0]
    start = world_fk(INITIAL[None, :7], root)[0]
    goal = start.copy(); goal[:3, 3] += [.08, -.03, .04]
    goal[:3, :3] = start[:3, :3] @ Rotation.from_rotvec([0., 0., .20]).as_matrix()
    path = candidate.se3_transport_path(planner, root, INITIAL, goal)
    joints = np.asarray(path['position']); poses = world_fk(joints, root)
    assert validate_joint_path(joints, planner.joint_limits, INITIAL)['passed']
    assert np.linalg.norm(poses[-1, :3, 3] - goal[:3, 3]) < .002
    assert Rotation.from_matrix(poses[-1, :3, :3].T @ goal[:3, :3]).magnitude() < .01
    delta = goal[:3, 3] - start[:3, 3]
    fraction = (poses[:, :3, 3] - start[:3, 3]) @ delta / (delta @ delta)
    line = start[:3, 3] + fraction[:, None] * delta
    assert np.max(np.linalg.norm(poses[:, :3, 3] - line, axis=1)) < .002
    assert path['cartesian_waypoints'] <= 512
    assert path['requires_retained_full_path_guard'] is True


@pytest.mark.parametrize('stage', ['above_bowl', 'lift', 'grasp_reach', 'release_above_rim', 'retreat'])
def test_existing_success_returns_identical_path_with_zero_fallback_calls(stage, monkeypatch):
    guard = candidate.TransportFallbackMotion.__new__(candidate.TransportFallbackMotion)
    guard.counter = 7
    original = {'position': np.zeros((2, 7)), 'guard_evidence': 'retained-path'}
    calls = []
    def success(self, pose, passed_stage, current=None):
        calls.append((self, pose, passed_stage, current)); return original
    def forbidden(*_args, **_kwargs):
        pytest.fail('successful retained path must not invoke fallback IK')
    monkeypatch.setattr(CartesianTransferMotion, 'plan', success)
    monkeypatch.setattr(candidate, 'se3_transport_path', forbidden)
    pose = object(); current = INITIAL.copy()
    assert guard.plan(pose, stage, current) is original
    assert len(calls) == 1 and calls[0][1] is pose and calls[0][3] is current
    assert guard.counter == 7


def make_rejected_guard(tmp_path, monkeypatch, *, guarded=True, attached=True):
    guard = candidate.TransportFallbackMotion.__new__(candidate.TransportFallbackMotion)
    guard.directory = tmp_path; guard.counter = 0; guard.use_attach = attached
    guard.root = np.eye(4); guard.planner = None
    guard.rec = SimpleNamespace(control_timestep=.05, performance=SimpleNamespace(section=lambda _: nullcontext()))
    request = {'stage': 'above_bowl', 'current': INITIAL.tolist(), 'goal': np.eye(4).tolist(), 'frame': 10, 'attached': attached}
    calls = []
    def rejected(self, _pose, stage, current=None):
        calls.append(stage)
        for i, mode in enumerate(('screw', 'rrt')):
            folder = tmp_path / f'{i:03d}_above_bowl_{mode}'
            write(folder / 'request.json', request)
            decision = {'accepted': False, 'report': {'passed': False, 'collision_failures': ['obstacle']}}
            if not guarded and mode == 'rrt': decision = {'accepted': False, 'planner_status': 'Failure'}
            write(folder / 'decision.json', decision)
        self.counter = 2
        raise RuntimeError('independent_guard_rejected:above_bowl')
    monkeypatch.setattr(CartesianTransferMotion, 'plan', rejected)
    return guard, calls


@pytest.mark.parametrize('guarded,attached', [(False, True), (True, False)])
def test_solver_failure_or_unattached_request_does_not_trigger_new_branch(tmp_path, monkeypatch, guarded, attached):
    guard, calls = make_rejected_guard(tmp_path, monkeypatch, guarded=guarded, attached=attached)
    monkeypatch.setattr(candidate, 'se3_transport_path', lambda *_args: pytest.fail('out-of-scope fallback'))
    with pytest.raises(RuntimeError, match='independent_guard_rejected:above_bowl'):
        guard.plan(object(), 'above_bowl')
    assert len(calls) == 1 and guard.counter == 2


@pytest.mark.parametrize('passed', [False, True])
def test_one_fallback_must_pass_original_guard_before_return(tmp_path, monkeypatch, passed):
    guard, calls = make_rejected_guard(tmp_path, monkeypatch)
    planned = []
    def proposal(*args):
        planned.append(args)
        return {'position': np.vstack([INITIAL[:7], INITIAL[:7]]), 'status': 'Success'}
    monkeypatch.setattr(candidate, 'se3_transport_path', proposal)
    checked = []
    def check(*args):
        checked.append(args)
        return {'passed': passed, 'collision_failures': [] if passed else ['inactive_robot']}, None, None
    guard.check = check
    if passed:
        result = guard.plan(object(), 'above_bowl')
        assert result['guard_evidence'].endswith('002_above_bowl_se3/decision.json')
    else:
        with pytest.raises(RuntimeError, match='independent_guard_rejected:above_bowl'):
            guard.plan(object(), 'above_bowl')
    assert len(calls) == len(planned) == len(checked) == 1
    assert guard.counter == 3
    decision = read(tmp_path / '002_above_bowl_se3' / 'decision.json')
    assert decision['accepted'] is passed


@pytest.mark.parametrize('problem', ['nonfinite', 'nonrigid', 'too_far', 'no_motion', 'timestep', 'joint_shape'])
def test_invalid_transport_rejects_before_solver(problem):
    root = np.eye(4); current = INITIAL.copy(); goal = world_fk(current[None, :7], root)[0]
    timestep = .05
    if problem != 'no_motion': goal[0, 3] += .04
    if problem == 'nonfinite': goal[0, 3] = np.nan
    elif problem == 'nonrigid': goal[0, 0] += .5
    elif problem == 'too_far': goal[0, 3] += 2.
    elif problem == 'timestep': timestep = 0.
    elif problem == 'joint_shape': current = current[:7]
    with pytest.raises(ValueError):
        candidate.se3_transport_path(None, root, current, goal, timestep)


def test_original_point_zero_eight_joint_branch_limit_is_retained():
    root = np.eye(4); goal = world_fk(INITIAL[None, :7], root)[0]; goal[0, 3] += .04
    def jump(_pose, q, **_kwargs):
        q[0] += .081
        return 'Success', [q]
    planner = SimpleNamespace(IK=jump)
    with pytest.raises(RuntimeError, match='transport_waypoint_branch_jump:1'):
        candidate.se3_transport_path(planner, root, INITIAL, goal)
