import copy
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from resimpair_pickplace.above_rim_release_v1 import raised_pose, policy, install


def config():
    return {'bottom_above_rim_m': .03, 'hand_above_rim_m': .01, 'maximum_fall_m': .15}


def test_payload_bottom_not_tcp_or_center_controls_release():
    final = np.eye(4); final[2, 3] = .04
    local = np.eye(4); local[2, 3] = .03
    vertices = np.array([[-.02, -.02, -.04], [.02, .02, .04]])
    points = np.array([[0., 0., -.02], [0., 0., .08]])
    desired, evidence = raised_pose(final, local, vertices, points, .05, config())
    assert desired[2, 3] == pytest.approx(.12)
    assert evidence['payload_bottom_above_rim_m'] == pytest.approx(.03)
    assert evidence['open_hand_bottom_above_rim_m'] >= .01
    assert final[2, 3] == .04


def test_low_grasp_requires_extra_height_for_fingers():
    final = np.eye(4); final[2, 3] = .05
    vertices = np.array([[0., 0., -.05], [0., 0., .05]])
    points = np.array([[0., 0., -.07], [0., 0., 0.]])
    desired, evidence = raised_pose(final, np.eye(4), vertices, points, .04, config())
    assert evidence['open_hand_bottom_above_rim_m'] == pytest.approx(.01)
    assert evidence['payload_bottom_above_rim_m'] >= .03 - 1e-10


def test_tilted_grasp_preserves_full_orientation():
    final = np.eye(4); final[:3, :3] = Rotation.from_euler('xy', [.4, .2]).as_matrix()
    vertices = np.array([[-.02, 0, -.01], [.02, 0, .01]])
    desired, _ = raised_pose(final, np.eye(4), vertices, vertices, .02, config())
    np.testing.assert_array_equal(desired[:3, :3], final[:3, :3])
    np.testing.assert_array_equal(desired[:2, 3], final[:2, 3])


def test_deep_container_cannot_silently_make_high_drop():
    with pytest.raises(ValueError, match='exceeds_declared'):
        raised_pose(np.eye(4), np.eye(4), np.array([[0., 0., 0.]]), np.array([[0., 0., 0.]]), .3, config())


def test_fragile_and_unknown_drop_never_implicitly_enabled():
    op = {'target': {'support': 'container_0'}, 'release': {'mode': 'drop_from_above', 'allow_drop': True}}
    with pytest.raises(ValueError, match='nonfragile'): policy(op)
    op['release']['fragile'] = True
    with pytest.raises(ValueError, match='nonfragile'): policy(op)
    op['release']['fragile'] = False
    assert policy(op)['bottom_above_rim_m'] == .03
    op['target']['support'] = 'table'
    with pytest.raises(ValueError, match='requires_a_container'): policy(op)


def test_adapter_restores_frozen_worker_after_exception():
    from resimpair_pickplace import whole_path_motion_v1 as old, rolling_pickplace_v3 as rolling
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import release_control
    original = old.prepare, old.WholePathMotion, rolling.PlanningView, release_control.readiness
    task = {'operations': []}; snapshot = copy.deepcopy(task)
    with pytest.raises(RuntimeError):
        with install(task): raise RuntimeError('probe')
    assert original == (old.prepare, old.WholePathMotion, rolling.PlanningView, release_control.readiness)
    assert snapshot == task
