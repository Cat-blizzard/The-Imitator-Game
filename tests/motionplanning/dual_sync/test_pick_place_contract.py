"""Native action archive order, actual cube geometry and collision negatives."""

from types import SimpleNamespace as NS

import numpy as np
import pytest
import sapien
import torch

from mani_skill.envs.tasks.tabletop.dual_pick_place import DualArmPickPlaceEnv
from mani_skill.examples.embodiments.check_sync_collision import overlap_private_models
from mani_skill.examples.motionplanning.dual.sync.pick_place import native_action
from mani_skill.examples.motionplanning.dual.sync import arm_planner


@pytest.mark.parametrize("sizes", [(7, 7), (8, 8), (7, 8)])
def test_native_action_archive_preserves_each_controller_size_and_order(sizes):
    order = [{"key": "physical_right", "size": sizes[0]},
             {"key": "physical_left", "size": sizes[1]}]
    source = np.arange(sum(sizes), dtype=np.float32)
    result = native_action(source, order)
    assert list(result) == ["physical_right", "physical_left"]
    assert np.array_equal(result["physical_right"], source[:sizes[0]])
    assert np.array_equal(result["physical_left"], source[sizes[0]:])
    result["physical_right"][0] = -100
    assert source[0] == 0


def test_native_action_archive_rejects_wrong_width_nonfinite_and_duplicate_keys():
    order = [{"key": "piper_left", "size": 7}, {"key": "piper_right", "size": 7}]
    for source in (np.zeros(16), np.full(14, np.nan), np.zeros((1, 14))):
        with pytest.raises(ValueError, match="Action shape"):
            native_action(source, order)
    with pytest.raises(ValueError, match="unique"):
        native_action(np.zeros(14), [{"key": "left", "size": 7}] * 2)
    with pytest.raises(ValueError, match="positive"):
        native_action(np.zeros(7), [{"key": "left", "size": 0}, {"key": "right", "size": 7}])


def contained_cube(x, y, z, angle=0.):
    pose = torch.eye(4)[None]
    pose[0, :2, :2] = torch.tensor([[np.cos(angle), -np.sin(angle)],
                                   [np.sin(angle), np.cos(angle)]], dtype=torch.float32)
    pose[0, :3, 3] = torch.tensor([x, y, z])
    fixture = NS(cube_half_size=.018, region_half_size=.055, device="cpu",
                 destination_xy=[[0., 0.]], targets=[NS(pose=NS(
                     p=pose[..., :3, 3], to_transformation_matrix=lambda: pose))])
    return bool(DualArmPickPlaceEnv.region_contains(fixture, 0).item())


def test_region_requires_whole_rotated_cube_and_table_height():
    assert contained_cube(0., 0., .018)
    assert contained_cube(.0369, 0., .018)
    assert not contained_cube(.0371, 0., .018)
    # A rotated cube has a larger exact support along the region axes. The
    # centre alone being inside is insufficient; all its corners must fit.
    assert contained_cube(.029, 0., .018, np.pi/4)
    assert not contained_cube(.030, 0., .018, np.pi/4)
    assert not contained_cube(0., 0., .025)
    assert not contained_cube(0., 0., .011)


@pytest.mark.parametrize("count", [6, 7])
def test_arm_planner_passes_named_full_state_without_executing(tmp_path, monkeypatch, count):
    names = tuple(f"joint_{i}" for i in range(count))
    full_names = ("finger_a", *names, "finger_b")
    qpos = np.r_[.027, np.arange(count) / 10, .031]
    urdf = tmp_path / "robot.urdf"
    urdf.write_text("fixture")
    urdf.with_suffix(".srdf").write_text("fixture")
    base_pose = np.array([[.2, -.3, .1, 1., 0., 0., 0.]])
    calls = []

    class Planner:
        def __init__(self, **kwargs):
            self.user_joint_names = tuple(kwargs["user_joint_names"])
            self.move_group_joint_indices = list(range(1, count+1))
            assert self.user_joint_names == full_names
            assert kwargs["move_group"] == "grasp_tcp"
            assert kwargs["joint_vel_limits"].shape == (count,)
            assert kwargs["joint_acc_limits"].shape == (count,)

        def set_base_pose(self, value):
            assert np.array_equal(value, base_pose[0])

        def plan_screw(self, target, measured, **kwargs):
            assert np.array_equal(measured, qpos)
            assert kwargs == {"time_step": .05, "use_point_cloud": False}
            calls.append(target.copy())
            return {"status": "Success"}

    monkeypatch.setattr(arm_planner, "Planner", Planner)
    robot = NS(get_qpos=lambda: qpos[None].copy(), pose=NS(raw_pose=base_pose),
        get_links=lambda: [NS(get_name=lambda: "grasp_tcp")],
        get_active_joints=lambda: [NS(get_name=lambda name=name: name) for name in full_names])
    agent = NS(robot=robot, urdf_path=str(urdf), ee_link_name="grasp_tcp",
        controller=NS(controllers={"arm": NS(config=NS(joint_names=names))}))
    env = NS(unwrapped=NS(control_timestep=.05))  # No step() exists on this planning-only fixture.
    solver = arm_planner.ArmMotionPlanner(env, agent)
    target = sapien.Pose([.1, .2, .3])
    assert solver.move_to_pose_with_screw(target, dry_run=True)["status"] == "Success"
    assert len(calls) == 1 and np.allclose(calls[0][:3], [.1, .2, .3])
    with pytest.raises(ValueError, match="shared executor"):
        solver.move_to_pose_with_screw(target, dry_run=False)
    assert len(calls) == 1
    agent.controller.controllers["arm"].config.joint_names = names[::-1]
    with pytest.raises(ValueError, match="orders differ"):
        arm_planner.ArmMotionPlanner(env, agent)


def test_private_collision_negative_restores_models_and_requires_mutual_pair():
    class Model:
        def __init__(self, x):
            self.base = np.array([x, 0., 0., 1., 0., 0., 0.])

        def get_base_pose(self):
            return self.base.copy()

        def set_base_pose(self, value):
            self.base = np.array(value, copy=True)

    models = (Model(-.3), Model(.3))
    before = tuple(model.get_base_pose() for model in models)

    def check(left, right):
        assert left.shape == (6,) and right.shape == (7,)
        if np.array_equal(models[0].base, models[1].base):
            return [{"object_name1": "left", "object_name2": "right", "collision_type": "self"}]
        return []

    checker = NS(models=models, check=check)
    result = overlap_private_models(checker, np.zeros(6), np.zeros(7))
    assert result["passed"] and result["private_model_bases_restored"]
    assert all(np.array_equal(model.base, pose) for model, pose in zip(models, before))
    checker.check = lambda *_: [{"object_name1": "left", "object_name2": "left"}]
    with pytest.raises(AssertionError, match="mutual"):
        overlap_private_models(checker, np.zeros(6), np.zeros(7))
    assert all(np.array_equal(model.base, pose) for model, pose in zip(models, before))


def test_private_collision_negative_restores_models_after_query_exception():
    values = [np.array([i, 0, 0, 1, 0, 0, 0.]) for i in (-1., 1.)]
    originals = [value.copy() for value in values]
    models = tuple(NS(get_base_pose=lambda i=i: values[i].copy(),
                      set_base_pose=lambda value, i=i: values.__setitem__(i, np.array(value, copy=True)))
                   for i in range(2))
    calls = []

    def check(*_):
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError("native query failure")
        return []

    with pytest.raises(RuntimeError, match="native query failure"):
        overlap_private_models(NS(models=models, check=check), np.zeros(6), np.zeros(6))
    assert all(np.array_equal(value, original) for value, original in zip(values, originals))
