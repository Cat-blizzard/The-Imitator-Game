"""CPU contracts for the single-task ACT training and observation path.

The visual encoder is the real, randomly initialized SimpleCNN, and the ACT
transformer and CVAE are exercised directly without assets or pretrained weights.
"""

import copy
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest
import numpy as np
import torch
from torch.utils.data import Dataset

from examples.baselines.act import train_act_single as training


def _args(**overrides):
    values = dict(
        cuda=False,
        obs_encoder_type="simple_cnn",
        image_size=[32, 32],
        state_dim=18,
        action_dim=16,
        obs_latent_dim=256,
        state_latent_dim=64,
        hidden_dim=256,
        state_hidden_dim=32,
        pred_horizon=4,
        nheads=4,
        enc_layers=1,
        dec_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        batch_size=2,
        num_workers=0,
    )
    values.update(overrides)
    return training.TrainingArgs(**values)


def _batch(batch_size=2):
    generator = torch.Generator().manual_seed(31)
    return {
        "robot_obs": {
            "rgb": torch.rand(batch_size, 3, 32, 32, generator=generator),
            "state": torch.randn(batch_size, 18, generator=generator),
        },
        "robot_actions": torch.randn(batch_size, 4, 16, generator=generator),
        "action_is_pad": torch.tensor([[False, False, True, True]]).repeat(batch_size, 1),
    }


@pytest.fixture
def agent():
    torch.manual_seed(17)
    return training.ACTSingleTaskAgent(_args(), torch.device("cpu"))


@pytest.mark.parametrize("batch_size", [1, 2])
def test_real_act_conditioning_forward_backward(agent, batch_size):
    batch = _batch(batch_size)
    assert agent.model.__class__.__module__.endswith("detr_video.detr_vae")
    assert agent.obs_proj.in_features == 320
    assert agent.obs_proj.out_features == 256
    assert agent.model.encoder_state_proj.in_features == 256
    assert agent.model.input_proj_robot_state.in_features == 256

    agent.train()
    loss = agent.compute_loss(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    # Both observation conditioning and the genuine CVAE/decoder receive gradients.
    for parameter in (
        agent.obs_proj.weight,
        agent.state_encoder.encoder.cls_proj.weight,
        agent.obs_encoder.encoder.features[0].weight,
        agent.model.encoder_action_proj.weight,
        agent.model.action_head.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0

    agent.eval()
    prediction = agent.get_action(**batch["robot_obs"])
    assert prediction.shape == (batch_size, 4, 16)
    assert torch.isfinite(prediction).all()


def test_masked_l1_averages_real_elements_and_blocks_padded_gradients():
    prediction = torch.zeros(2, 3, 2, requires_grad=True)
    target = torch.tensor([
        [[1.0, 3.0], [100.0, 100.0], [100.0, 100.0]],
        [[5.0, 7.0], [9.0, 11.0], [100.0, 100.0]],
    ])
    is_pad = torch.tensor([[False, True, True], [False, False, True]])
    loss = training.masked_l1(prediction, target, is_pad)
    torch.testing.assert_close(loss, torch.tensor(6.0))
    loss.backward()
    torch.testing.assert_close(prediction.grad[is_pad], torch.zeros(3, 2))
    torch.testing.assert_close(prediction.grad[~is_pad], torch.full((3, 2), -1.0 / 6.0))


def test_masked_l1_all_real_is_legacy_mean_and_all_padding_is_rejected():
    prediction = torch.tensor([[[1.0, 4.0], [3.0, 2.0]]], requires_grad=True)
    target = torch.zeros_like(prediction)
    torch.testing.assert_close(
        training.masked_l1(prediction, target, torch.zeros(1, 2, dtype=torch.bool)),
        torch.nn.functional.l1_loss(prediction, target),
    )
    with pytest.raises(ValueError, match="real|valid|padding|padded"):
        training.masked_l1(prediction, target, torch.ones(1, 2, dtype=torch.bool))


def _forward(agent, obs, actions, mask="omitted"):
    kwargs = dict(actions=actions, video_feature=torch.zeros(actions.shape[0], 1))
    if not isinstance(mask, str):
        kwargs["action_is_pad"] = mask
    torch.manual_seed(53)
    return agent.model(obs, **kwargs)


def test_cvae_padding_does_not_change_posterior_or_reconstruction(agent):
    agent.eval()
    batch = _batch()
    actions = batch["robot_actions"]
    mask = batch["action_is_pad"]
    changed = actions.clone()
    changed[mask] = torch.linspace(-500.0, 500.0, int(mask.sum()) * 16).reshape(-1, 16)
    with torch.no_grad():
        obs = agent._encode_obs(**batch["robot_obs"])
        prediction, posterior = _forward(agent, obs, actions, mask)
        changed_prediction, changed_posterior = _forward(agent, obs, changed, mask)
        for actual, expected in zip(changed_posterior, posterior):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(changed_prediction, prediction, atol=1e-6, rtol=1e-6)

        # The same action perturbation must be visible when padding is not masked.
        _, unmasked_posterior = _forward(agent, obs, actions)
        _, changed_unmasked_posterior = _forward(agent, obs, changed)
        assert not torch.allclose(unmasked_posterior[0], changed_unmasked_posterior[0])


def test_cvae_optional_mask_preserves_existing_calls(agent):
    agent.eval()
    batch = _batch()
    actions = batch["robot_actions"]
    with torch.no_grad():
        obs = agent._encode_obs(**batch["robot_obs"])
        original, original_posterior = _forward(agent, obs, actions)
        explicit, explicit_posterior = _forward(agent, obs, actions, torch.zeros(2, 4, dtype=torch.bool))
    torch.testing.assert_close(explicit, original, atol=0, rtol=0)
    for actual, expected in zip(explicit_posterior, original_posterior):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_compute_loss_uses_the_same_padding_for_posterior_and_l1(agent):
    agent.eval()
    batch = _batch()
    changed = copy.deepcopy(batch)
    changed["robot_actions"][batch["action_is_pad"]] += 1000.0
    with torch.no_grad():
        torch.manual_seed(79)
        original_loss = agent.compute_loss(batch)
        torch.manual_seed(79)
        changed_loss = agent.compute_loss(changed)
    torch.testing.assert_close(changed_loss, original_loss, atol=1e-6, rtol=1e-6)


def test_strict_checkpoint_reload_reproduces_fixed_action(agent):
    agent.eval()
    batch = _batch(1)
    expected = agent.get_action(**batch["robot_obs"])
    checkpoint = io.BytesIO()
    torch.save({"args": vars(agent.args), "agent_state_dict": agent.state_dict()}, checkpoint)
    checkpoint.seek(0)
    stored = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restored = training.ACTSingleTaskAgent(training.TrainingArgs(**stored["args"]), torch.device("cpu"))
    restored.load_state_dict(stored["agent_state_dict"], strict=True)
    restored.eval()
    actual = restored.get_action(**batch["robot_obs"])
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


class _FiveSamples(Dataset):
    state_dim = 18
    action_dim = 16

    def __len__(self):
        return 5

    def __getitem__(self, index):
        return {"sample_id": index}


@pytest.mark.parametrize("augmentation", [False, True])
def test_dataloader_preserves_camera_recorded_fps_and_all_samples(tmp_path, monkeypatch, augmentation):
    dataset_root = tmp_path / "pills" / "meta"
    dataset_root.mkdir(parents=True)
    (dataset_root / "info.json").write_text(json.dumps({"fps": 20}))
    sub_config = tmp_path / "dataset.json"
    sub_config.write_text(json.dumps([{"repo_id": "pills", "root": "pills"}]))
    captured = []

    def build(config):
        captured.append(config)
        return _FiveSamples()

    # Native training preserves the explicitly selected recorded camera.
    monkeypatch.setattr(training, "build_lerobot_dataset", build)
    args = _args(
        sim_root=str(tmp_path), sub_config=str(sub_config), cameras=["zed2i"],
        enable_augmentation=augmentation,
    )
    loader, dataset = training.build_dataloader(args)
    assert len(dataset) == 5
    assert len(captured) == 1
    assert captured[0].cameras == ["zed2i"]
    assert captured[0].fps == 20
    assert captured[0].horizon == 4
    assert captured[0].enable_augmentation is augmentation
    assert loader.drop_last is False
    batches = list(loader)
    assert sorted(len(batch["sample_id"]) for batch in batches) == [1, 2, 2]
    assert sorted(torch.cat([batch["sample_id"] for batch in batches]).tolist()) == list(range(5))


def test_native_training_augmentation_is_explicit():
    assert training.TrainingArgs().enable_augmentation is False
    assert training.TrainingArgs(enable_augmentation=True).enable_augmentation is True



def _evaluation_module():
    from examples.baselines.act import eval_act_single
    return eval_act_single


def _raw_observation(state_width=9):
    return {
        "sensor_data": {"zed2i": {"rgb": np.full((1, 6, 10, 3), 127, np.uint8)}},
        # Deliberately different from the native controller insertion order.
        "agent": {
            "left": {"qpos": torch.arange(state_width, dtype=torch.float32)[None]},
            "right": {"qpos": torch.arange(100, 100 + state_width, dtype=torch.float32)[None]},
        },
    }


@pytest.mark.parametrize("state_width", [8, 9, 12])
@pytest.mark.parametrize("image_size", [(6, 10), (3, 7)])
def test_native_selected_camera_qpos_order_and_read_only_observation(state_width, image_size):
    from examples.baselines.act.single_task_io import live_observation
    raw_observation = _raw_observation(state_width)
    original = copy.deepcopy(raw_observation)
    raw = SimpleNamespace(agent=SimpleNamespace(agents=[None, None],
                                                agents_dict={"right": None, "left": None}))
    identity = SimpleNamespace(normalize_state=lambda value, idx: value)
    observation = live_observation(raw, raw_observation, "zed2i", identity,
                                   image_size=image_size)
    assert set(observation) == {"rgb", "state"}
    assert observation["rgb"].shape == (3, *image_size)
    assert observation["rgb"].dtype == torch.float32
    torch.testing.assert_close(observation["rgb"], torch.full((3, *image_size), 127 / 255.))
    expected_state = torch.cat([original["agent"][key]["qpos"].reshape(-1)
                                for key in ("right", "left")])
    torch.testing.assert_close(observation["state"], expected_state, atol=0, rtol=0)
    assert np.array_equal(raw_observation["sensor_data"]["zed2i"]["rgb"],
                          original["sensor_data"]["zed2i"]["rgb"])
    for key in original["agent"]:
        torch.testing.assert_close(raw_observation["agent"][key]["qpos"],
                                   original["agent"][key]["qpos"], atol=0, rtol=0)


def test_native_policy_rejects_control_rate_different_from_training():
    evaluation = _evaluation_module()
    env = SimpleNamespace(reset=lambda seed: ({}, {}), unwrapped=SimpleNamespace(control_freq=30))
    with pytest.raises(ValueError, match="Hz"):
        evaluation.evaluate_episode(None, env, None, SimpleNamespace(fps=20), None,
                                    401, "pills", torch.device("cpu"))


def test_native_normalizer_uses_checkpoint_stats_and_does_not_reload_disk(monkeypatch):
    evaluation = _evaluation_module()
    from examples.baselines.lerobot_dataset import lerobot_dataset
    state_key = "observation.qpos_gripper_states"
    action_key = "action.qpos_gripper_actions"
    stats = {state_key: {"q01": [-2.] * 18, "q99": [2.] * 18},
             action_key: {"q01": [-4.] * 16, "q99": [4.] * 16}}

    def read_disk(*args, **kwargs):
        pytest.fail("Policy evaluation must reuse saved training statistics")

    monkeypatch.setattr(lerobot_dataset, "LeRobotDatasetMetadata", read_disk)
    normalizer = evaluation.SingleTaskNormalizer([
        {"repo_id": "pills", "dataset_idx": 0, "stats": stats,
         "state_key": state_key, "action_key": action_key}])
    torch.testing.assert_close(normalizer.normalize_state(torch.ones(2, 18)), torch.full((2, 18), .5))
    torch.testing.assert_close(normalizer.denormalize_action(torch.full((2, 16), .5)), torch.full((2, 16), 2.))
    with pytest.raises(ValueError, match="saved training statistics"):
        evaluation.SingleTaskNormalizer([])


@pytest.mark.parametrize("state_dim,action_dim", [(18, 16), (16, 14), (24, 14)])
def test_seeded_native_policy_rollout_keeps_action_width_and_official_recovery(state_dim, action_dim):
    """Use the public observation adapter and the real evaluator, without physics.

    The recorded native episode drops, regrips and reaches native success before
    its time limit. Event reporting must neither truncate nor alter that episode.
    This replaces the superseded vector/Panda-only evaluator contract.
    """
    from gymnasium import spaces
    evaluation = _evaluation_module()
    script = [(True, False, False), (False, False, False), (True, False, False),
              (True, True, True), (False, True, True)]

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.queries = 0

        def get_action(self, rgb, state):
            assert rgb.shape == (1, 1, 3, 6, 10)
            assert state.shape == (1, 1, state_dim)
            self.queries += 1
            return (torch.arange(4 * action_dim, dtype=torch.float32).reshape(1, 4, action_dim)
                    / 100. + (self.queries - 1) / 10.)

    class Environment:
        def __init__(self, report_physical_facts):
            self.report_physical_facts = report_physical_facts
            self.steps = 0
            self.actions, self.native_results, self.reset_seeds = [], [], []
            self.unwrapped = SimpleNamespace(control_freq=20,
                agent=SimpleNamespace(agents=[None, None], agents_dict={"left": None, "right": None}),
                action_space=spaces.Dict({key: spaces.Box(-1., 1., shape=(action_dim // 2,), dtype=np.float32)
                                          for key in ("left", "right")}))

        def reset(self, seed):
            self.reset_seeds.append(seed)
            self.steps = 0
            return _raw_observation(state_dim // 2), {}

        def step(self, action):
            self.actions.append(copy.deepcopy(action))
            grasped, in_region, success = script[self.steps]
            self.steps += 1
            terminated, truncated = success, self.steps == len(script)
            reward = float(sum(value.sum() for value in action.values())) + self.steps
            self.native_results.append((reward, terminated, truncated, success))
            info = {"success": success}
            if self.report_physical_facts:
                info.update(is_pill_a_grasped=grasped, is_pill_b_grasped=grasped,
                            pill_a_in_region=in_region, pill_b_in_region=in_region)
            return _raw_observation(state_dim // 2), reward, terminated, truncated, info

    normalizer = SimpleNamespace(resolve_idx=lambda env_id: 0,
                                 normalize_state=lambda value, idx: value,
                                 denormalize_action=lambda value, idx: value)
    args = SimpleNamespace(fps=20, execute_steps=4, max_episode_steps=7, native_contract={})
    train = _args(cameras=["zed2i"], image_size=[6, 10], state_dim=state_dim, action_dim=action_dim)
    outputs = []
    for report in (True, False):
        env, policy = Environment(report), Policy()
        result = evaluation.evaluate_episode(policy, env, normalizer, args, train,
                                             401, "pills", torch.device("cpu"))
        assert policy.queries == 2
        assert env.reset_seeds == [401]
        assert result["steps"] == len(script)
        assert result["official_success_once"] and result["official_success_at_end"]
        assert result["first_success_step"] == 4
        assert result["truncated"] and result["terminated"]
        assert result["failure_stage"] is None
        assert all(value.shape == (action_dim // 2,) for action in env.actions for value in action.values())
        outputs.append((env, result))
    observed, unobserved = (row[0] for row in outputs)
    assert observed.native_results == unobserved.native_results
    for actual, expected in zip(observed.actions, unobserved.actions):
        for key in actual:
            np.testing.assert_array_equal(actual[key], expected[key])
    assert outputs[0][1]["physical"]["grasp_placement"]["arms"]["left"][
        "grasp_to_ungrasped_outside_region_steps"] == [2]
    for key in ("steps", "official_success_once", "official_success_at_end", "first_success_step",
                "terminated", "truncated", "timeout", "action_clipped_steps"):
        assert outputs[0][1][key] == outputs[1][1][key]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
