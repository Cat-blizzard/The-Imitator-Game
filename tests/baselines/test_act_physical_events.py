"""Physical-event reporting stays separate from policy control and task success."""
import copy
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


def test_drop_regrasp_success_and_native_rollout_are_unchanged_by_event_reporting(monkeypatch):
    """An observed drop cannot veto later recovery or change native step results.

    Run the actual event sampler and a disabled sampler against the same scripted
    policy/environment. The fixture supplies native task facts; it does not model
    the physics of dropping or regrasping. This tests the evaluation boundary.
    """
    script = [
        (True, False, False),   # Grasp outside the destination.
        (False, False, False),  # Drop outside the destination.
        (True, False, False),   # Regrasp and keep executing the policy.
        (True, True, True),     # Native success may precede opening.
        (False, True, True),    # Release in the destination, native time limit.
    ]

    class Actor:
        def __init__(self):
            self.pose = NS(p=torch.zeros(1, 3))
            self.velocity = torch.zeros(1, 3)

        def get_linear_velocity(self):
            return self.velocity

        def get_angular_velocity(self):
            return self.velocity

    class Environment:
        def __init__(self):
            self.unwrapped = NS(control_freq=20, script_index=0,
                action_space=spaces.Dict({key: spaces.Box(-1., 1., shape=(7,), dtype=np.float32)
                                         for key in ("left", "right")}),
                pill_a=Actor(), pill_b=Actor())
            self.submitted = []
            self.native_results = []

        def reset(self, seed):
            assert seed == 73
            return {}, {}

        def step(self, action):
            self.submitted.append({key: value.copy() for key, value in action.items()})
            index = self.unwrapped.script_index
            grasped, in_region, success = script[index]
            self.unwrapped.script_index += 1
            for actor in (self.unwrapped.pill_a, self.unwrapped.pill_b):
                actor.pose.p.fill_(index / 10.)
                actor.velocity.fill_(0. if in_region else .01)
            info = {"success": success, "pill_a_in_region": in_region,
                    "pill_b_in_region": in_region, "is_pill_a_grasped": grasped,
                    "is_pill_b_grasped": grasped, "is_robot_static": in_region}
            reward = float(sum(value.sum() for value in action.values())) + index
            terminated, truncated = success, index == len(script) - 1
            self.native_results.append((reward, terminated, truncated, copy.deepcopy(info)))
            return {}, reward, terminated, truncated, info

    def observation(raw, *args):
        return {"rgb": torch.zeros(3, 2, 2),
                "state": torch.full((16,), float(raw.script_index))}

    def predict(rgb, state):
        # Vary actions each step, including one clipped prediction, so an early
        # stop, extra policy call, or change to action processing is observable.
        return (torch.arange(14, dtype=torch.float32).reshape(1, 1, 14) / 10.
                + state[0, -1, 0] / 100.)

    monkeypatch.setattr(evaluation, "live_observation", observation)
    event_sampler = evaluation._task_events
    sampled_steps = []

    def checked_event_sampler(raw, info):
        before_info = copy.deepcopy(info)
        before_step = raw.script_index
        before_state = [(actor.pose.p.clone(), actor.velocity.clone())
                        for actor in (raw.pill_a, raw.pill_b)]
        result = event_sampler(raw, info)
        assert info == before_info
        assert raw.script_index == before_step
        for actor, (position, velocity) in zip((raw.pill_a, raw.pill_b), before_state):
            torch.testing.assert_close(actor.pose.p, position, rtol=0, atol=0)
            torch.testing.assert_close(actor.velocity, velocity, rtol=0, atol=0)
        sampled_steps.append(before_step)
        return result

    monkeypatch.setattr(evaluation, "_task_events", checked_event_sampler)
    agent = NS(get_action=predict)
    normalizer = NS(resolve_idx=lambda env_id: 0, denormalize_action=lambda action, idx: action)
    args = NS(fps=20, execute_steps=1, max_episode_steps=7, native_contract={})
    train = NS(cameras=["base_camera"], state_type="qpos", single_arm=False,
               action_dim=14, image_size=[2, 2], obs_horizon=1)

    observed_env = Environment()
    observed = evaluation.evaluate_episode(agent, observed_env, normalizer, args,
                                            train, 73, "fixture", torch.device("cpu"))
    # Reporting must not mutate the native info object returned by env.step.
    expected_info = [{"success": success, "pill_a_in_region": in_region,
                      "pill_b_in_region": in_region, "is_pill_a_grasped": grasped,
                      "is_pill_b_grasped": grasped, "is_robot_static": in_region}
                     for grasped, in_region, success in script]
    assert [row[3] for row in observed_env.native_results] == expected_info
    assert observed["steps"] == len(script)
    assert sampled_steps == list(range(1, len(script) + 1))
    assert observed["official_success_at_end"] and observed["official_success_once"]
    assert observed["first_success_step"] == 4
    assert observed["failure_stage"] is None and not observed["timeout"]
    assert observed["terminated"] and observed["truncated"]
    for arm in observed["physical"]["grasp_placement"]["arms"].values():
        assert arm["grasp_to_ungrasped_outside_region_steps"] == [2]
        assert arm["grasp_to_ungrasped_steps"] == [2, 5]
        assert arm["final_placed_and_ungrasped"] is True
    assert observed["events"][2]["left_grasped"] is True

    monkeypatch.setattr(evaluation, "_task_events", lambda raw, info: {})
    unobserved_env = Environment()
    unobserved = evaluation.evaluate_episode(agent, unobserved_env, normalizer, args,
                                              train, 73, "fixture", torch.device("cpu"))
    assert observed_env.native_results == unobserved_env.native_results
    assert len(observed_env.submitted) == len(unobserved_env.submitted) == len(script)
    for actual, reference in zip(observed_env.submitted, unobserved_env.submitted):
        for side in ("left", "right"):
            np.testing.assert_array_equal(actual[side], reference[side])
    for key in ("steps", "official_success_once", "official_success_at_end", "first_success_step",
                "terminated", "truncated", "timeout", "action_clipped_steps"):
        assert observed[key] == unobserved[key]
    assert observed["action_clipped_steps"] == len(script)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
