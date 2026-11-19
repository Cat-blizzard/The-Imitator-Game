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

    # Auto-discovery must not replace the camera that the user explicitly selected.
    monkeypatch.setattr(training, "_auto_detect_cameras", lambda *unused: ["unselected_camera"])
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


def test_augmentation_default_preserves_existing_training_behavior():
    assert training.TrainingArgs().enable_augmentation is True


def _evaluation_module():
    from examples.baselines.act import eval_act_single
    return eval_act_single


def _raw_observation(batch_size=2):
    image = torch.empty(batch_size, 6, 10, 3, dtype=torch.uint8)
    for index in range(batch_size):
        image[index] = torch.tensor([index * 40, 127, 255], dtype=torch.uint8)
    return {
        "sensor_data": {
            "unselected": {"rgb": torch.full_like(image, 12)},
            "zed2i": {"rgb": image, "depth": torch.ones(batch_size, 6, 10, 1)},
        },
        "sensor_param": {"calibration": torch.eye(4)},
        # Deliberately different from the agent's insertion order.
        "agent": {
            "left": {"qpos": torch.arange(9, dtype=torch.float64).repeat(batch_size, 1),
                     "qvel": torch.full((batch_size, 9), -500.)},
            "right": {"qpos": torch.arange(100, 109, dtype=torch.float64).repeat(batch_size, 1),
                      "qvel": torch.full((batch_size, 9), 500.)},
        },
        "extra": {"task_only": torch.full((batch_size, 13), 900.)},
    }


def _fake_rgb_env(observation, control_freq=20):
    import gymnasium as gym

    class TinyEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.agent = SimpleNamespace(agents_dict={"right": None, "left": None})
            self.control_freq = control_freq
            self._init_raw_obs = observation
            self.updated_observation = None

        def update_obs_space(self, result):
            self.updated_observation = result

    return TinyEnv()


def _assert_same_observation(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, dict):
            _assert_same_observation(actual[key], value)
        else:
            torch.testing.assert_close(actual[key], value, atol=0, rtol=0)


@pytest.mark.parametrize("image_size", [(6, 10), (3, 7)])
def test_rgb_wrapper_selects_camera_qpos_order_and_preserves_raw_observation(image_size):
    evaluation = _evaluation_module()
    raw = _raw_observation()
    original = copy.deepcopy(raw)
    env = _fake_rgb_env(raw)
    wrapper = evaluation.SingleTaskRGBObservationWrapper(env, camera="zed2i", image_size=image_size)
    observation = wrapper.observation(raw)
    assert set(observation) == {"rgb", "state"}
    assert observation["rgb"].shape == (2, 3, *image_size)
    assert observation["rgb"].dtype == torch.float32
    expected_rgb = raw["sensor_data"]["zed2i"]["rgb"][:, 0, 0].float() / 255.
    torch.testing.assert_close(
        observation["rgb"], expected_rgb[:, :, None, None].expand(2, 3, *image_size),
        atol=0, rtol=0,
    )
    expected_state = torch.cat([raw["agent"][key]["qpos"] for key in ("right", "left")], dim=-1).float()
    assert observation["state"].shape == (2, 18)
    assert observation["state"].dtype == torch.float32
    torch.testing.assert_close(observation["state"], expected_state, atol=0, rtol=0)
    assert env.updated_observation is not None
    _assert_same_observation(raw, original)


def test_rgb_wrapper_rejects_control_rate_different_from_training():
    evaluation = _evaluation_module()
    with pytest.raises(ValueError, match="frequency"):
        evaluation.SingleTaskRGBObservationWrapper(_fake_rgb_env(_raw_observation(), 30), "zed2i")


def _normalizer_inputs(tmp_path):
    config = tmp_path / "dataset.json"
    config.write_text(json.dumps([{"repo_id": "pills", "root": "pills"}]))
    stats = {
        "observation.qpos_gripper_states": {"q01": [-2.] * 18, "q99": [2.] * 18},
        "action.qpos_gripper_actions": {"q01": [-4.] * 16, "q99": [4.] * 16},
    }
    return config, stats


def test_single_camera_normalizer_uses_checkpoint_stats_without_imagenet(tmp_path, monkeypatch):
    evaluation = _evaluation_module()
    from examples.baselines.lerobot_dataset import lerobot_dataset

    config, stats = _normalizer_inputs(tmp_path)

    def read_disk(*args, **kwargs):
        raise AssertionError("Saved training statistics must take precedence over disk metadata")

    monkeypatch.setattr(lerobot_dataset, "LeRobotDatasetMetadata", read_disk)
    normalizer = evaluation.SingleTaskNormalizer(
        str(config), str(tmp_path),
        dataset_stats=[{"repo_id": "pills", "dataset_idx": 0, "stats": stats}],
        normalize_images=False,
    )
    rgb = torch.rand(2, 3, 8, 8)
    torch.testing.assert_close(normalizer.normalize_rgb(rgb), rgb, atol=0, rtol=0)
    torch.testing.assert_close(normalizer.normalize_state(torch.ones(2, 18)), torch.full((2, 18), .5))
    torch.testing.assert_close(normalizer.denormalize_action(torch.full((2, 16), .5)), torch.full((2, 16), 2.))


def test_legacy_normalizer_keeps_disk_stats_and_imagenet_path(tmp_path, monkeypatch):
    evaluation = _evaluation_module()
    from examples.baselines.lerobot_dataset import lerobot_dataset

    config, stats = _normalizer_inputs(tmp_path)
    calls = []

    def read_disk(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(stats=stats)

    monkeypatch.setattr(lerobot_dataset, "LeRobotDatasetMetadata", read_disk)
    normalizer = evaluation.SingleTaskNormalizer(str(config), str(tmp_path))
    assert calls == [{"repo_id": "pills", "root": str(tmp_path / "pills")}]
    rgb = torch.full((1, 3, 4, 4), .5)
    expected = (rgb - evaluation.IMAGENET_MEAN[None, :, None, None]) / evaluation.IMAGENET_STD[None, :, None, None]
    torch.testing.assert_close(normalizer.normalize_rgb(rgb), expected)


@pytest.mark.parametrize("info_layout", ["episode_numpy", "final_info_torch"])
def test_cpu_policy_rollout_reaches_budget_and_reads_episode_metrics(info_layout):
    evaluation = _evaluation_module()

    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.queries = 0

        def get_action(self, rgb, state):
            assert rgb.shape == (1, 3, 8, 8)
            assert state.shape == (1, 18)
            self.queries += 1
            return torch.arange(64, dtype=torch.float32).reshape(1, 4, 16) / 100.

    class TinyVectorEnv:
        num_envs = 1

        def __init__(self):
            self.actions = []
            self.reset_seeds = []
            self.steps = 0

        def reset(self, seed=None):
            self.reset_seeds.append(seed)
            self.steps = 0
            return {"rgb": np.zeros((1, 3, 8, 8), dtype=np.float32),
                    "state": np.zeros((1, 18), dtype=np.float32)}, {}

        def step(self, action):
            assert set(action) == {"panda_wristcam-0", "panda_wristcam-1"}
            assert all(isinstance(value, np.ndarray) and value.shape == (1, 8) for value in action.values())
            self.actions.append(copy.deepcopy(action))
            self.steps += 1
            info = {}
            if self.steps == 3:
                metrics = {"success_once": np.array([True]),
                           "success_at_end": np.array([True]),
                           "episode_len": np.array([3])}
                if info_layout == "episode_numpy":
                    info = {"episode": metrics}
                else:
                    info = {"final_info": {"episode": {key: torch.as_tensor(value) for key, value in metrics.items()}}}
            observation = {"rgb": np.zeros((1, 3, 8, 8), dtype=np.float32),
                           "state": np.zeros((1, 18), dtype=np.float32)}
            return observation, np.zeros(1), np.array([False]), np.array([self.steps == 3]), info

    policy = TinyPolicy()
    env = TinyVectorEnv()
    identity_normalizer = SimpleNamespace(
        normalize_rgb=lambda value: value,
        normalize_state=lambda value: value,
        denormalize_action=lambda value: value,
    )
    result = evaluation.evaluate_episode(
        1, policy, env, identity_normalizer,
        SimpleNamespace(seed=401, sim_backend="physx_cpu", camera="zed2i"),
        torch.device("cpu"), _args(),
    )
    assert policy.queries == 1
    assert len(env.actions) == 3
    assert env.reset_seeds == [401, None]
    expected = np.arange(64, dtype=np.float32).reshape(4, 16) / 100.
    for index, action in enumerate(env.actions):
        np.testing.assert_allclose(action["panda_wristcam-0"], expected[index:index + 1, :8])
        np.testing.assert_allclose(action["panda_wristcam-1"], expected[index:index + 1, 8:])
    np.testing.assert_array_equal(result["success_once"], [True])
    np.testing.assert_array_equal(result["success_at_end"], [True])
    np.testing.assert_array_equal(result["episode_len"], [3])


def test_policy_drop_regrasp_success_is_not_gated_by_reported_physical_facts():
    """Native task facts cannot veto recovery or alter the policy rollout.

    This vectorized evaluator has no physical-event observer. Pair the same
    scripted native episode with and without optional grasp/region facts in info;
    the real evaluator must preserve actions and the native final success metric.
    The fixture tests that boundary, not physical grasp recovery in simulation.
    """
    evaluation = _evaluation_module()
    script = [(True, False, False), (False, False, False), (True, False, False),
              (True, True, True), (False, True, True)]

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.queries = 0

        def get_action(self, rgb, state):
            self.queries += 1
            return (torch.arange(64, dtype=torch.float32).reshape(1, 4, 16) / 100.
                    + (self.queries - 1) / 10.)

    class Environment:
        num_envs = 1

        def __init__(self, report_physical_facts):
            self.report_physical_facts = report_physical_facts
            self.steps = 0
            self.actions = []
            self.native_results = []
            self.physical_info = []

        def observation(self):
            return {"rgb": np.full((1, 3, 8, 8), self.steps / 10., dtype=np.float32),
                    "state": np.full((1, 18), self.steps / 10., dtype=np.float32)}

        def reset(self, seed=None):
            self.steps = 0
            return self.observation(), {}

        def step(self, action):
            self.actions.append(copy.deepcopy(action))
            grasped, in_region, success = script[self.steps]
            self.steps += 1
            terminated, truncated = success, self.steps == len(script)
            reward = float(sum(value.sum() for value in action.values())) + self.steps
            self.native_results.append((reward, terminated, truncated, success))
            info = {"success": np.array([success])}
            if self.report_physical_facts:
                info.update(is_pill_a_grasped=np.array([grasped]),
                            is_pill_b_grasped=np.array([grasped]),
                            pill_a_in_region=np.array([in_region]),
                            pill_b_in_region=np.array([in_region]))
                self.physical_info.append(info)
            if truncated:
                info["final_info"] = {"episode": {
                    "success_once": np.array([True]), "success_at_end": np.array([success]),
                    "episode_len": np.array([self.steps])}}
            return (self.observation(), np.array([reward]), np.array([terminated]),
                    np.array([truncated]), info)

    normalizer = SimpleNamespace(normalize_rgb=lambda value: value,
                                 normalize_state=lambda value: value,
                                 denormalize_action=lambda value: value)
    outputs = []
    for report in (True, False):
        env, policy = Environment(report), Policy()
        result = evaluation.evaluate_episode(
            1, policy, env, normalizer,
            SimpleNamespace(seed=401, sim_backend="physx_cpu", camera="zed2i"),
            torch.device("cpu"), _args())
        assert policy.queries == 2
        assert len(env.actions) == len(script)
        np.testing.assert_array_equal(result["success_once"], [True])
        np.testing.assert_array_equal(result["success_at_end"], [True])
        np.testing.assert_array_equal(result["episode_len"], [len(script)])
        outputs.append((env, result))

    observed, unobserved = (row[0] for row in outputs)
    assert [bool(row["is_pill_a_grasped"][0]) for row in observed.physical_info] == [
        True, False, True, True, False]
    assert observed.native_results == unobserved.native_results
    for actual, expected in zip(observed.actions, unobserved.actions):
        for key in ("panda_wristcam-0", "panda_wristcam-1"):
            np.testing.assert_array_equal(actual[key], expected[key])
    for key in outputs[0][1]:
        np.testing.assert_array_equal(outputs[0][1][key], outputs[1][1][key])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
