"""Real visual ACT with a fixture only at frozen video feature extraction."""
from dataclasses import replace
import runpy
import sys

import pytest
import torch
from torch import nn

from examples.baselines.act import native_video_model as native
from examples.baselines.act.act.detr_video import backbone as visual_backbone
from examples.baselines.encoders.task_encoder.video_backbone import FrozenVideoBackbone


@pytest.fixture
def video_fixture(monkeypatch):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    # Keep the real ResNet, DETRVAE, Transformer and FrozenVideoBackbone adapters;
    # prevent internet access for pretrained weights in a CPU interface test.
    monkeypatch.setattr(visual_backbone, "is_main_process", lambda: False)

    def load(encoder):
        if encoder._loaded:
            return
        encoder._backbone = nn.Linear(3, 768, bias=False)
        encoder._backbone.requires_grad_(False).eval()
        encoder._loaded = True
        encoder.observed_clips = []

    def encode(encoder, video):
        encoder.observed_clips.append(video.detach().cpu().clone())
        raw = encoder._backbone(video.mean(dim=(1, 3, 4)))
        return raw, raw.unsqueeze(1).expand(-1, 4, -1)

    monkeypatch.setattr(FrozenVideoBackbone, "_load_backbone", load)
    monkeypatch.setattr(FrozenVideoBackbone, "_encode_dino", encode)
    yield
    torch.set_num_threads(previous_threads)


def arguments(**kwargs):
    args = native.TrainingArgs(hidden_dim=32, nheads=4, dim_feedforward=64,
                               enc_layers=1, dec_layers=1, dropout=0.0,
                               task_latent_dim=16, pred_horizon=4,
                               state_dim=16, action_dim=14,
                               frozen_backbone_type="dinov2_vitb14")
    return replace(args, **kwargs)


def batch(args, count=1):
    return {"view_1": torch.rand(count, 1, 3, 64, 64),
            "states": torch.rand(count, 1, args.state_dim),
            "actions": torch.rand(count, args.pred_horizon, args.action_dim),
            "action_is_pad": torch.tensor([[False, False, True, True]]).expand(count, -1),
            "human_video": torch.rand(1, 3, 3, 32, 32).expand(count, -1, -1, -1, -1),
            "human_demo_id": ["sha256-clip-A-frames-0-15-30"] * count}


@pytest.mark.parametrize("state_dim,action_dim", [(16, 14), (24, 14), (18, 16)])
def test_real_visual_act_padding_adapter_gradient_and_strict_reload(video_fixture, state_dim, action_dim):
    torch.manual_seed(13)
    args = arguments(state_dim=state_dim, action_dim=action_dim)
    agent = native.NativeVideoACTAgent(args, torch.device("cpu"))
    assert isinstance(agent, native.ACTAgent)
    assert isinstance(agent.task_encoder.backbone, FrozenVideoBackbone)
    data = batch(args)
    data["actions"].requires_grad_()
    agent.eval()
    torch.manual_seed(21)
    first = agent.compute_loss(data)
    changed = dict(data, actions=data["actions"].detach().clone())
    changed["actions"][:, 2:] += 1000
    torch.manual_seed(21)
    second = agent.compute_loss(changed)
    torch.testing.assert_close(first, second, atol=1e-6, rtol=0)
    first.backward()
    assert first.ndim == 0 and torch.isfinite(first)
    assert torch.count_nonzero(data["actions"].grad[:, 2:]) == 0
    assert agent.task_encoder.backbone.cls_adapter[0].weight.grad.abs().sum() > 0
    assert agent.model.video_feature_proj.weight.grad.abs().sum() > 0
    assert agent.model.backbones[0][0].body["conv1"].weight.grad.abs().sum() > 0
    assert all(not parameter.requires_grad for parameter in agent.task_encoder.backbone._backbone.parameters())
    assert set(agent.last_loss_terms) == {"l1", "kl"}
    first_summary = agent.set_demonstration(data["human_video"], data["human_demo_id"][0])
    expected = agent.get_action(data["view_1"], data["states"])
    assert expected.shape == (1, 4, action_dim)
    saved = agent.state_dict()
    assert any("task_encoder.backbone._backbone" in key for key in saved)
    restored = native.NativeVideoACTAgent(args, torch.device("cpu"))
    restored.load_state_dict(saved, strict=True)
    restored_summary = restored.set_demonstration(data["human_video"], data["human_demo_id"][0])
    actual = restored.get_action(data["view_1"], data["states"])
    assert torch.equal(expected, actual)
    assert first_summary == restored_summary


def test_raw_cache_deduplicates_clips_without_detaching_adapter(video_fixture):
    torch.manual_seed(5)
    args = arguments()
    agent = native.NativeVideoACTAgent(args, torch.device("cpu"))
    data = batch(args, count=2)
    agent.train()
    assert not agent.task_encoder.backbone._backbone.training
    optimizer = torch.optim.Adam(agent.task_encoder.backbone.cls_adapter.parameters(), lr=0.01)
    loss = agent.compute_loss(data)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    assert len(agent.task_encoder.backbone.observed_clips) == 1
    assert agent.task_encoder.backbone.observed_clips[0].shape[0] == 1
    first_z = agent._task_z(data["human_video"], data["human_demo_id"]).detach().clone()
    agent.compute_loss(data).backward()
    assert agent.task_encoder.backbone.cls_adapter[0].weight.grad.abs().sum() > 0
    optimizer.step()
    second_z = agent._task_z(data["human_video"], data["human_demo_id"]).detach()
    assert not torch.equal(first_z, second_z)
    assert len(agent.task_encoder.backbone.observed_clips) == 1
    cached = agent.task_encoder._raw_cache._store[data["human_demo_id"][0]]
    assert not cached["cls"].requires_grad and not cached["seq"].requires_grad


def test_changed_demonstration_reaches_backbone_and_changes_conditioning(video_fixture):
    torch.manual_seed(9)
    agent = native.NativeVideoACTAgent(arguments(), torch.device("cpu"))
    image, state = torch.rand(1, 3, 64, 64), torch.zeros(1, 16)
    with pytest.raises(RuntimeError, match="set_demonstration"):
        agent.get_action(image, state)
    video_a, video_b = torch.zeros(3, 3, 32, 32), torch.zeros(3, 3, 32, 32)
    video_a[:, 0] = 1
    video_b[:, 2] = 1
    summary_a = agent.set_demonstration(video_a, "clip-A-content-hash")
    action_a = agent.get_action(image, state)
    summary_b = agent.set_demonstration(video_b, "clip-B-content-hash")
    action_b = agent.get_action(image, state)
    assert summary_a["task_z_sha256"] != summary_b["task_z_sha256"]
    assert not torch.equal(action_a, action_b)
    assert torch.equal(agent.task_encoder.backbone.observed_clips[0][0], video_a)
    assert torch.equal(agent.task_encoder.backbone.observed_clips[1][0], video_b)
    assert agent.set_demonstration(video_a, "clip-A-content-hash") == summary_a
    assert torch.equal(agent.get_action(image, state), action_a)
    assert len(agent.task_encoder.backbone.observed_clips) == 2


def test_load_clears_features_and_current_conditioning(video_fixture):
    agent = native.NativeVideoACTAgent(arguments(), torch.device("cpu"))
    agent.set_demonstration(torch.rand(2, 3, 32, 32), "clip-A")
    assert agent.task_encoder._raw_cache.is_populated
    agent.load_state_dict(agent.state_dict(), strict=True)
    assert not agent.task_encoder._raw_cache.is_populated
    assert agent._cached_task_z is None


def test_constant_condition_removes_clip_information_but_keeps_visual_policy(video_fixture):
    torch.manual_seed(7)
    args = arguments()
    agent = native.NativeVideoACTAgent(args, torch.device("cpu"), conditioning_mode="constant")
    data = batch(args)
    loss = agent.compute_loss(data)
    loss.backward()
    assert torch.isfinite(loss)
    assert agent.model.backbones[0][0].body["conv1"].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in agent.task_encoder.parameters())
    first = agent.set_demonstration(torch.zeros(3, 3, 32, 32), "clip-A")
    actions = agent.get_action(data["view_1"], data["states"])
    second = agent.set_demonstration(torch.ones(3, 3, 32, 32), "clip-B")
    assert torch.equal(actions, agent.get_action(data["view_1"], data["states"]))
    assert first["task_z_sha256"] == second["task_z_sha256"]
    assert first["conditioning_mode"] == "constant" and first["task_z_norm"] == 0
    assert not agent.task_encoder.backbone.observed_clips
    restored = native.NativeVideoACTAgent(args, torch.device("cpu"), conditioning_mode="constant")
    restored.load_state_dict(agent.state_dict(), strict=True)
    restored.set_demonstration(torch.ones(3, 3, 32, 32), "clip-C")
    assert torch.equal(actions, restored.get_action(data["view_1"], data["states"]))


def test_native_mask_is_required_and_all_padding_is_rejected(video_fixture):
    args = arguments()
    agent = native.NativeVideoACTAgent(args, torch.device("cpu"))
    data = batch(args)
    missing = dict(data)
    del missing["action_is_pad"]
    with pytest.raises(KeyError, match="action_is_pad"):
        agent.compute_loss(missing)
    data["action_is_pad"] = torch.ones(1, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="real action"):
        agent.compute_loss(data)


def test_module_sanity_entry(video_fixture, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["native_video_model", "--device", "cpu"])
    runpy.run_module("examples.baselines.act.native_video_model", run_name="__main__")


@pytest.mark.parametrize("change", [{"obs_horizon": 2}, {"include_depth": True},
                                    {"input_mode": "language_only"}, {"frozen_backbone_lora_rank": 8}])
def test_explicit_initial_scope(change):
    with pytest.raises(ValueError, match="Native video ACT"):
        native.NativeVideoACTAgent(arguments(**change), torch.device("cpu"))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
