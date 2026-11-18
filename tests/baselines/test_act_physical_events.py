"""Reported grasp/placement facts cannot turn a lost grasp into placed release."""
from types import SimpleNamespace as NS
import json

import numpy as np
import pytest
import torch
from gymnasium import spaces

from examples.baselines.act import eval_act_single as evaluation


def sample(step, grasped, in_region):
    return {"step": step, "left_grasped": grasped, "right_grasped": grasped,
            "left_in_region": in_region, "right_in_region": in_region}


@pytest.mark.parametrize("last_grasped,last_in_region,expected_placed,outside_loss", [
    (False, False, False, [12]),  # Lost contact outside both destinations.
    (False, True, True, []),      # An observed placed/ungrasped state, no causal claim.
    (True, True, False, []),      # Region success can precede actual opening.
])
def test_grasp_transition_and_placement_are_separate_facts(
        last_grasped, last_in_region, expected_placed, outside_loss):
    events = [sample(10, False, False), sample(11, True, False),
              sample(12, last_grasped, last_in_region)]
    original = json.dumps(events)
    result = evaluation.grasp_placement_measurements(events)
    assert result["both_final_placed_and_ungrasped"] is expected_placed
    for arm in result["arms"].values():
        assert arm["first_grasp_step"] == 11
        assert arm["grasp_to_ungrasped_steps"] == ([] if last_grasped else [12])
        assert arm["grasp_to_ungrasped_outside_region_steps"] == outside_loss
        assert arm["first_placed_and_ungrasped_step"] == (12 if expected_placed else None)
    assert "not_controlled_release" in result["semantics"]
    assert json.dumps(events) == original


def test_missing_samples_do_not_invent_a_transition_or_negative_placement():
    events = [sample(1, True, False), {"step": 2}, sample(3, False, True)]
    result = evaluation.grasp_placement_measurements(events)
    assert result["arms"]["left"]["grasp_to_ungrasped_steps"] == []
    assert result["arms"]["left"]["grasp_measurements_complete"] is False
    empty = evaluation.grasp_placement_measurements([])
    assert empty["both_final_placed_and_ungrasped"] is None
    assert empty["arms"]["left"]["ever_grasped"] is None


@pytest.mark.parametrize("cube", [False, True])
def test_original_pill_and_shared_cube_expose_the_same_grasp_region_facts(cube):
    target = NS(pose=NS(p=torch.tensor([[0., 0., .1]])),
                get_linear_velocity=lambda: torch.zeros(1, 3),
                get_angular_velocity=lambda: torch.zeros(1, 3))
    if cube:
        agent = NS(is_grasping=lambda actor: torch.tensor([False]),
                   is_static=lambda threshold: torch.tensor([True]))
        raw = NS(targets=[target, target], agent=NS(agents=[agent, agent]))
        info = {"left_in_region": True, "right_in_region": True}
    else:
        raw = NS(pill_a=target, pill_b=target)
        info = {"pill_a_in_region": True, "pill_b_in_region": True,
                "is_pill_a_grasped": False, "is_pill_b_grasped": False,
                "is_robot_static": True}
    event = evaluation._task_events(raw, info)
    event["step"] = 1
    assert evaluation.grasp_placement_measurements([event])["both_final_placed_and_ungrasped"]


def test_new_report_does_not_change_actions_score_or_legacy_release(monkeypatch):
    trace = [sample(1, True, False), sample(2, False, False), sample(3, False, False)]
    submitted = []
    class Environment:
        unwrapped = NS(control_freq=20, action_space=spaces.Dict({
            key: spaces.Box(-1., 1., shape=(7,), dtype=np.float32)
            for key in ("left", "right")}))
        def reset(self, seed):
            return {}, {}
        def step(self, action):
            submitted.append(action)
            info = {"success": False, "event": trace[len(submitted)-1]}
            return {}, 0., False, len(submitted) == len(trace), info
    monkeypatch.setattr(evaluation, "live_observation", lambda *args:
                        {"rgb": torch.zeros(3, 2, 2), "state": torch.zeros(16)})
    monkeypatch.setattr(evaluation, "_task_events", lambda raw, info: dict(info["event"]))
    agent = NS(get_action=lambda rgb, state: torch.zeros(1, 1, 14))
    normalizer = NS(resolve_idx=lambda env_id: 0, denormalize_action=lambda action, idx: action)
    args = NS(fps=20, execute_steps=1, max_episode_steps=3, native_contract={})
    train = NS(cameras=["base_camera"], state_type="qpos", single_arm=False,
               action_dim=14, image_size=[2, 2], obs_horizon=1)
    result = evaluation.evaluate_episode(agent, Environment(), normalizer, args,
                                         train, 1, "fixture", torch.device("cpu"))
    assert result["steps"] == len(submitted) == 3
    assert all(np.array_equal(action["left"], np.zeros(7)) for action in submitted)
    assert not result["official_success_at_end"] and result["timeout"]
    assert result["physical"]["release_observed"] is True  # Preserve legacy value.
    assert "legacy" in result["physical"]["release_observed_semantics"]
    assert result["physical"]["grasp_placement"]["both_final_placed_and_ungrasped"] is False
