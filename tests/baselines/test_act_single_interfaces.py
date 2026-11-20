"""CPU regressions for ACT construction, conditioning, and evaluation entry points.

Models use real encoders/CVAE/transformers. The ResNet test only disables its
pretrained weight download; simulator rollout is replaced at the factory boundary.
"""

import inspect
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.baselines.act import train_act_single as training


@pytest.mark.parametrize("batch_size,hidden_dim", [(1, 256), (2, 32)])
def test_real_conditioning_forward_backward_and_inference(batch_size, hidden_dim):
    torch.manual_seed(17)
    args = training.TrainingArgs(cuda=False, obs_encoder_type="simple_cnn",
        image_size=[32, 32], pred_horizon=4, hidden_dim=hidden_dim,
        enc_layers=1, dec_layers=1, dim_feedforward=64, dropout=0.0)
    agent = training.ACTSingleTaskAgent(args, torch.device("cpu"))
    rgb = torch.rand(batch_size, 3, 32, 32)
    state = torch.zeros(batch_size, args.state_dim)
    actions = torch.randn(batch_size, args.pred_horizon, args.action_dim)
    assert args.obs_latent_dim + args.state_latent_dim != hidden_dim

    loss = agent.compute_loss({"robot_obs": {"rgb": rgb, "state": state},
                               "robot_actions": actions})
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    for parameter in (agent.obs_proj.weight, agent.model.encoder_state_proj.weight,
                      agent.model.encoder_action_proj.weight, agent.model.action_head.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0

    agent.eval()
    predicted = agent.get_action(rgb, state)
    assert predicted.shape == (batch_size, args.pred_horizon, args.action_dim)
    assert torch.isfinite(predicted).all()


def test_default_resnet_model_backward_and_strict_checkpoint_reload(monkeypatch, tmp_path):
    import torchvision.models as models
    from examples.baselines.act.eval_act_single import load_checkpoint

    resnet18 = models.resnet18

    def offline_resnet18(*args, **kwargs):
        kwargs["weights"] = None
        return resnet18(*args, **kwargs)

    monkeypatch.setattr(models, "resnet18", offline_resnet18)
    torch.manual_seed(31)
    args = training.TrainingArgs(cuda=False, dropout=0.0)
    agent = training.ACTSingleTaskAgent(args, torch.device("cpu"))
    rgb, state = torch.rand(1, 1, 3, 224, 224), torch.zeros(1, 1, args.state_dim)
    actions = torch.randn(1, args.pred_horizon, args.action_dim)
    loss = agent.compute_loss({"view_1": rgb, "states": state, "actions": actions})
    assert torch.isfinite(loss)
    loss.backward()
    assert agent.obs_proj.weight.grad is not None
    assert torch.isfinite(agent.obs_proj.weight.grad).all()
    assert agent.obs_proj.weight.grad.abs().sum() > 0

    agent.eval()
    predicted = agent.get_action(rgb, state)
    assert predicted.shape == (1, 16, 16) and torch.isfinite(predicted).all()
    checkpoint = tmp_path / "act.pt"
    torch.save({"args": vars(args), "agent_state_dict": agent.state_dict()}, checkpoint)
    loaded, loaded_args = load_checkpoint(str(checkpoint), SimpleNamespace(), torch.device("cpu"))
    assert vars(loaded_args) == vars(args)
    torch.testing.assert_close(loaded.get_action(rgb, state), predicted, rtol=0, atol=0)


def evaluation_fixture(monkeypatch, output_dir, *, factory_error=False, rollout_error=False):
    from examples.baselines.act import eval_act_single as evaluation

    events, factory_calls = [], []
    signature = inspect.signature(evaluation.make_eval_envs)
    env = SimpleNamespace(close=lambda: events.append("close"))

    def factory(**kwargs):
        signature.bind(**kwargs)
        factory_calls.append(kwargs)
        if factory_error:
            raise RuntimeError("factory failure")
        return env

    def rollout(**kwargs):
        assert kwargs["eval_envs"] is env
        if rollout_error:
            raise RuntimeError("rollout failure")
        return {"success_once": np.array([1.0])}

    monkeypatch.setattr(evaluation, "make_eval_envs", factory)
    monkeypatch.setattr(evaluation, "evaluate_episode", rollout)
    monkeypatch.setattr(evaluation, "set_l_level", lambda level: events.append(level))
    monkeypatch.setattr(evaluation, "clear_l_level", lambda: events.append("clear"))
    args = SimpleNamespace(include_depth=False, num_envs=1, sim_backend="physx_cpu",
        control_mode="pd_joint_pos", max_episode_steps=500, num_episodes=1)

    def evaluate(env_id):
        return evaluation.evaluate_single_env(object(), env_id, args, object(),
            SimpleNamespace(obs_horizon=1), output_dir, "test", device=torch.device("cpu"))

    return evaluate, events, factory_calls


@pytest.mark.parametrize("level", ["L0", "L1", "L2", "L3"])
def test_evaluation_uses_existing_factory_and_level_mapping(monkeypatch, tmp_path, level):
    evaluate, events, calls = evaluation_fixture(monkeypatch, tmp_path)
    result = evaluate(f"{level}_TwoRobotCleanCup-v1")
    assert result["status"] == "success" and result["success_once_mean"] == 1.0
    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["env_id"] == ("TwoRobotCleanCupL3-v1" if level == "L3" else "TwoRobotCleanCup-v1")
    assert kwargs["l_level"] == level
    assert kwargs["sim_backend"] == "physx_cpu" and kwargs["num_envs"] == 1
    assert kwargs["env_kwargs"] == dict(control_mode="pd_joint_pos", reward_mode="dense",
        obs_mode="rgb", render_mode="rgb_array", max_episode_steps=500,
        sensor_configs=dict(shader_pack="rt-fast"), human_render_camera_configs=dict(shader_pack="rt-fast"))
    assert kwargs["other_kwargs"] == {"obs_horizon": 1}
    assert kwargs["video_dir"] == str(tmp_path / "videos" / f"{level}_TwoRobotCleanCup-v1")
    assert [wrapper.__name__ for wrapper in kwargs["wrappers"]] == ["FlattenRGBDObservationWrapper"]
    assert events == [level, "close", "clear"]


@pytest.mark.parametrize("failure", ["factory", "rollout"])
def test_evaluation_preserves_error_reporting_and_cleanup(monkeypatch, tmp_path, failure):
    evaluate, events, calls = evaluation_fixture(monkeypatch, tmp_path,
        factory_error=failure == "factory", rollout_error=failure == "rollout")
    result = evaluate("L3_TwoRobotCleanCup-v1")
    assert result["status"] == "error" and result["error"] == f"{failure} failure"
    assert len(calls) == 1
    assert events == (["L3", "clear"] if failure == "factory" else ["L3", "close", "clear"])


@pytest.mark.parametrize("module", ["train_act_single", "eval_act_single"])
def test_module_cli_help_starts_without_data_or_checkpoint(module):
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", NO_ALBUMENTATIONS_UPDATE="1")
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run([sys.executable, "-B", "-m", f"examples.baselines.act.{module}", "--help"],
        cwd=Path(__file__).resolve().parents[2], env=environment, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "--sub-config" in completed.stdout


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
