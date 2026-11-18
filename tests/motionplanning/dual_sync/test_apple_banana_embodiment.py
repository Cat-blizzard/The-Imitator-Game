"""Original basket-task contracts for the optional Piper integration."""
from __future__ import annotations

from types import SimpleNamespace as NS
from itertools import product

import numpy as np
import pytest
import torch

from mani_skill.envs.tasks.tabletop.dual_arm_profiles import get_task_profile
from mani_skill.envs.tasks.tabletop.dual_tasks import _008_pick_apple_banana_baskets as task_env
from mani_skill.examples.motionplanning.dual import two_robot_run as runner
from mani_skill.examples.motionplanning.dual.sync.apple_banana_task import (
    AppleBananaTask, TASK_ID, basket_release_center, basket_contents_geometry_proof)
from mani_skill.examples.motionplanning.dual.sync.solve import task_components


def test_piper_dispatch_selects_original_task_without_replacing_its_horizon(monkeypatch):
    calls = []
    class ReachedEnvironment(Exception):
        pass
    def make(env_id, **kwargs):
        calls.append((env_id, kwargs))
        raise ReachedEnvironment
    monkeypatch.setattr(runner.gym, "make", make)
    args = runner.parse_args(["-e", TASK_ID, "--robot", "piper", "--execution-mode", "sync"])
    with pytest.raises(ReachedEnvironment):
        runner._main(args)
    assert calls[0][0] == TASK_ID
    assert calls[0][1]["robot_uids"] == ("piper", "piper")
    assert calls[0][1]["control_mode"] == "pd_joint_pos"
    assert "max_episode_steps" not in calls[0][1]
    assert task_components(TASK_ID)[0] is AppleBananaTask


@pytest.mark.parametrize("robot_uids,wrist_sensor,reason", [
    (("piper", "panda_wristcam"), False, "homogeneous"),
    (("piper", "piper"), True, "no wrist camera"),
])
def test_invalid_pair_and_missing_wrist_sensor_fail_before_environment_loading(robot_uids, wrist_sensor, reason):
    with pytest.raises(ValueError, match=reason):
        task_env.TwoRobotPickAppleBananaToBasketsEnv(robot_uids=robot_uids, wrist_sensor=wrist_sensor)


def test_original_success_uses_both_basket_predicates_and_static_remains_diagnostic(monkeypatch):
    env = task_env.TwoRobotPickAppleBananaToBasketsEnv.__new__(task_env.TwoRobotPickAppleBananaToBasketsEnv)
    agent = NS(is_static=lambda _: torch.tensor([False]), is_grasping=lambda _: torch.tensor([False]))
    env.agent = NS(agents=[agent, agent])
    env.apple_out, env.banana_out = NS(position=torch.tensor([[.05, 0., .03]])), NS(position=torch.tensor([[.05, 0., .03]]))
    env.basket_apple, env.basket_banana = NS(), NS()
    env._get_actor_geom_center = lambda actor: actor.position
    monkeypatch.setattr(task_env, "world_aabb_from_local_mesh", lambda actor, device:
        torch.tensor([[[0., -.1, 0.], [.1, .1, .07]]]))
    env.device = torch.device("cpu")
    result = env.evaluate()
    assert result["success"].item() and not result["is_robot_static"].item()
    env.banana_out.position[0, 0] = .11
    assert not env.evaluate()["success"].item()
    assert task_env.TwoRobotPickAppleBananaToBasketsEnv.apple_scale == .7
    assert task_env.TwoRobotPickAppleBananaToBasketsEnv.basket_scale == 1.5


def test_measured_grasp_uses_bounds_center_instead_of_vertex_density():
    objects = (NS(), NS())
    adapter = AppleBananaTask.__new__(AppleBananaTask)
    adapter.raw = NS(embodiment_profile=get_task_profile(TASK_ID, "piper"))
    adapter.targets = objects
    # Repeating one vertex changes its mean, but must not move the grasp center.
    vertices = np.array([[0., 0., 0.], [0., 0., 0.], [.06, .04, .08]])
    adapter.vertices = lambda actor: vertices
    for pregrasp, grasp in adapter.grasp_poses():
        np.testing.assert_allclose(grasp.p, [.03, .02, .052], atol=1e-8)
        np.testing.assert_allclose(pregrasp.p-grasp.p, [0., 0., .08], atol=1e-8)


def test_landing_proposal_preserves_measured_container_and_prefers_free_space():
    basket = np.array([[-.15, -.15, 0.], [.15, .15, .08]])
    contents = np.array([[-.03, -.03, 0.], [.03, .03, .07]])
    before = basket.copy(), contents.copy()
    center = basket_release_center(basket, [.025, .025], contents, [-.2, -.2])
    # The basket center is occupied. Propose nearby free space rather than
    # dropping a second fruit onto it merely to minimize centrality.
    assert np.any((center+.025 <= contents[0, :2]) |
                  (center-.025 >= contents[1, :2]))
    assert np.linalg.norm(center) < .08
    np.testing.assert_array_equal(basket, before[0])
    np.testing.assert_array_equal(contents, before[1])
    # A long banana can retain its original dimensions when just inside a basket.
    center = basket_release_center([[-.105, -.145, 0], [.105, .145, .08]],
                                   [.099, .035], contents, [-.2, 0.])
    assert np.all(np.isfinite(center)) and abs(center[0]) <= .006


def test_overlapping_proposal_remains_available_when_no_free_footprint_exists():
    # Bounds are a goal heuristic, not another containment/contact rejection.
    # This basket admits a payload but cannot separate it from the contents.
    center = basket_release_center([[-.04, -.04, 0.], [.04, .04, .08]],
        [.025, .025], [[-.03, -.03, 0.], [.03, .03, .07]], [-.2, -.2])
    assert center.shape == (2,) and np.isfinite(center).all()


@pytest.mark.parametrize("half", [[0., .01], [.2, .01], [np.nan, .01]])
def test_invalid_or_noncontained_release_proposal_is_a_planning_error(half):
    with pytest.raises(ValueError):
        basket_release_center([[-.1, -.1, 0.], [.1, .1, .08]], half,
                              [[-.03, -.03, 0.], [.03, .03, .07]], [0., 0.])


def test_contents_contact_is_bound_to_current_geometry_and_original_contact_points():
    basket = np.array(list(product((-.15, .15), (-.15, .15), (0., .08))))
    fruit = np.array(list(product((-.03, .03), (-.03, .03), (.02, .07))))
    assert basket_contents_geometry_proof(fruit, fruit, basket, [[0., 0., .04]])["allowed_geometry"]
    for moved, points in ((fruit+[.2, 0., 0.], [[0., 0., .04]]),
                          (fruit+[0., 0., .2], [[0., 0., .04]]),
                          (fruit, [[.16, 0., .04]]), (fruit, [[0., 0., .181]])):
        assert not basket_contents_geometry_proof(moved, fruit, basket, points)["allowed_geometry"]
    # Original success uses the geometric center. An overhanging bound alone
    # must not turn an otherwise normal same-basket contact into a collision.
    overhanging = fruit.copy()
    overhanging[:, 0] *= 6
    proof = basket_contents_geometry_proof(overhanging, fruit, basket, [[0., 0., .04]])
    assert proof['allowed_geometry'] and not proof['full_bounds_are_acceptance_gate']


@pytest.mark.parametrize("index,other,phase", [(0, "banana_in", "release"),
    (1, "apple_in", "release"), (0, "apple_in", "carry_lower"), (1, "banana_in", "descend")])
def test_contents_contact_rejects_cross_basket_and_held_phases(index, other, phase):
    task = AppleBananaTask.__new__(AppleBananaTask)
    # The early identity/phase refusal must not depend on missing scene geometry.
    assert not task.destination_contents_contact_proof(index, other, phase, [])['allowed']
