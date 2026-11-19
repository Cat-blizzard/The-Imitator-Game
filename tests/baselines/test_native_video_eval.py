"""CPU checks for strict native-video reload and the shared policy rollout."""
from dataclasses import asdict
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium import spaces

from examples.baselines.act import eval_act_native_video as video_eval
from examples.baselines.act import eval_act_single
from examples.baselines.act import native_video_data
from examples.baselines.act.train_act_single import TrainingArgs as NativeArgs
from examples.baselines.act.train_act_imitator import TrainingArgs as VideoArgs

TASK = "TwoRobotPickPillToRegions-v1"


def _checkpoint():
    native = NativeArgs(state_dim=16, action_dim=14, cameras=["base_camera"],
                        image_size=[8, 8], pred_horizon=2)
    model = VideoArgs(state_dim=16, action_dim=14, cameras=["base_camera"],
                      image_size=(8, 8), pred_horizon=2)
    return {"policy_type": "native_video_act", "args": asdict(native),
            "model_args": asdict(model), "human_manifest": {"saved": True},
            "data_contract": {"fps": 20, "native": {"robot_uids": ["piper"] * 2}},
            "dataset_stats": []}


class _ReloadPolicy(torch.nn.Linear):
    def __init__(self, args, device):
        super().__init__(args.state_dim, args.action_dim, device=device)
        self.received_args = args


def test_checkpoint_loader_uses_saved_native_dimensions_and_strict_weights(tmp_path, monkeypatch):
    from examples.baselines.act import native_video_model
    monkeypatch.setattr(native_video_model, "NativeVideoACTAgent", _ReloadPolicy)
    checkpoint = _checkpoint()
    original = _ReloadPolicy(VideoArgs(**checkpoint["model_args"]), torch.device("cpu"))
    checkpoint["agent_state_dict"] = original.state_dict()
    path = tmp_path / "video.pt"
    torch.save(checkpoint, path)
    restored, settings = video_eval.load_checkpoint(str(path), SimpleNamespace(), torch.device("cpu"))
    assert settings.state_dim == 16 and settings.action_dim == 14
    assert not restored.training
    assert torch.equal(original.weight, restored.weight)
    assert restored.received_args.image_size == (8, 8)
    relocated, _ = video_eval.load_checkpoint(
        str(path), SimpleNamespace(hf_cache_dir="/relocated/hf/hub"), torch.device("cpu"))
    assert relocated.received_args.hf_cache_dir == "/relocated/hf/hub"
    with pytest.raises(ValueError, match="pred_horizon must match"):
        video_eval.load_checkpoint(str(path), SimpleNamespace(pred_horizon=3), torch.device("cpu"))
    checkpoint["agent_state_dict"]["unrecognized_weight"] = torch.zeros(1)
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        video_eval.load_checkpoint(str(path), SimpleNamespace(), torch.device("cpu"))


@pytest.mark.parametrize("key,value", [("action_dim", 16), ("cameras", ["zed2i"]),
                                      ("image_size", [224, 224])])
def test_checkpoint_rejects_native_model_input_disagreement(tmp_path, key, value):
    checkpoint = _checkpoint()
    checkpoint["model_args"][key] = value
    path = tmp_path / "video.pt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match=f"native/model {key}"):
        video_eval.load_checkpoint(str(path), SimpleNamespace(), torch.device("cpu"))


def test_checkpoint_requires_the_explicit_video_policy_type(tmp_path):
    checkpoint = _checkpoint()
    checkpoint["policy_type"] = "single_task_act"
    path = tmp_path / "single.pt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="native_video_act checkpoint"):
        video_eval.load_checkpoint(str(path), SimpleNamespace(), torch.device("cpu"))


class _ConditionedPolicy:
    def __init__(self):
        self.selected = []
        self.observed = []

    def set_demonstration(self, video, demo_id):
        self.selected.append((video.clone(), demo_id))
        self.condition = float(video.mean())
        return {"human_demo_id": demo_id,
                "task_z_sha256": hashlib.sha256(video.numpy().tobytes()).hexdigest(),
                "task_z_norm": self.condition, "task_z_shape": [1, 1]}

    def get_action(self, rgb, state):
        self.observed.append((rgb.clone(), state.clone()))
        return torch.full((1, 2, 14), self.condition) + state.mean() / 10


def _install_demo_loader(monkeypatch, *, task_id=TASK):
    calls = []

    def load(source, *, role, demo_id, image_size, video_backend):
        calls.append((source, role, demo_id, image_size, video_backend))
        replacement = demo_id == "replacement"
        return {"human_video": torch.full((3, 3, *image_size), .4 if replacement else .2),
                "human_demo_id": "content:replacement" if replacement else "content:correct",
                "metadata": {"id": demo_id or "evaluation-demo", "task_id": task_id,
                             "video_sha256": "b" * 64 if replacement else "a" * 64,
                             "frame_indices": [1, 3, 5], "repo_id": "human_H31",
                             "episode_id": int(replacement), "role": role}}

    monkeypatch.setattr(native_video_data, "load_human_demo", load)
    return calls


def test_each_episode_sets_condition_and_preserves_video_identity(monkeypatch):
    calls = _install_demo_loader(monkeypatch)
    conditioner, agent = video_eval.HumanVideoConditioning(), _ConditionedPolicy()
    checkpoint = _checkpoint()
    checkpoint["args"]["video_backend"] = "opencv"
    first = conditioner(agent, checkpoint, {}, None, 6100, TASK, torch.device("cpu"))
    second = conditioner(agent, checkpoint, {}, None, 6101, f"L3_{TASK}", torch.device("cpu"))
    assert len(calls) == 1 and len(agent.selected) == 2
    assert calls[0] == ({"saved": True}, "evaluation", None, (8, 8), "opencv")
    assert agent.selected[0][0].shape == (3, 3, 8, 8)
    assert first["human_demo_id"] == "content:correct"
    assert first["human_demonstration"]["frame_indices"] == [1, 3, 5]
    assert first["human_demonstration"]["video_sha256"] == "a" * 64
    assert second["human_demonstration"]["task_matches_environment"]


def test_explicit_replacement_changes_effective_condition_and_reports_override(monkeypatch):
    calls = _install_demo_loader(monkeypatch)
    agent, checkpoint = _ConditionedPolicy(), _checkpoint()
    kwargs = (agent, checkpoint, {}, None, 6100, TASK, torch.device("cpu"))
    correct = video_eval.HumanVideoConditioning()(*kwargs)
    replaced = video_eval.HumanVideoConditioning("other.json", demo_id="replacement")(*kwargs)
    assert len(calls) == 2
    assert calls[-1][0] == "other.json"
    assert correct["conditioning"]["task_z_sha256"] != replaced["conditioning"]["task_z_sha256"]
    assert correct["human_demo_id"] != replaced["human_demo_id"]
    assert replaced["human_demonstration"]["explicit_demo_override"]


def test_wrong_task_requires_explicit_diagnostic_selection(monkeypatch):
    _install_demo_loader(monkeypatch, task_id="TwoRobotCleanCup-v1")
    agent, checkpoint = _ConditionedPolicy(), _checkpoint()
    kwargs = (agent, checkpoint, {}, None, 6100, TASK, torch.device("cpu"))
    with pytest.raises(ValueError, match="intentional replacement requires"):
        video_eval.HumanVideoConditioning()(*kwargs)
    assert not agent.selected
    result = video_eval.HumanVideoConditioning(demo_id="replacement")(*kwargs)
    assert not result["human_demonstration"]["task_matches_environment"]


def test_missing_manifest_requires_an_explicit_input():
    checkpoint = _checkpoint()
    del checkpoint["human_manifest"]
    with pytest.raises(ValueError, match="supply --human-manifest"):
        video_eval.HumanVideoConditioning()(_ConditionedPolicy(), checkpoint, {}, None,
                                           6100, TASK, torch.device("cpu"))


def test_multitask_condition_cache_selects_each_task_and_returns_to_first(monkeypatch):
    calls = []
    other = "TwoRobotPickAppleBananaToBaskets-v1"

    def load(source, *, task_id, **kwargs):
        calls.append(task_id)
        return {"human_video": torch.full((3, 3, 8, 8), .2 if task_id == TASK else .4),
                "human_demo_id": f"content:{task_id}", "metadata": {"task_id": task_id}}

    monkeypatch.setattr(native_video_data, "load_human_demo", load)
    checkpoint = _checkpoint()
    checkpoint["human_manifest"] = {"schema_version": 2, "tasks": []}
    agent, conditioner = _ConditionedPolicy(), video_eval.HumanVideoConditioning()
    reports = [conditioner(agent, checkpoint, {}, None, 1, task, torch.device("cpu"))
               for task in (TASK, other, TASK)]
    assert calls == [TASK, other]
    assert reports[0]["human_demo_id"] == reports[2]["human_demo_id"]
    assert reports[0]["human_demo_id"] != reports[1]["human_demo_id"]
    assert all(row["human_demonstration"]["task_matches_environment"] for row in reports)


def test_per_dataset_contract_routes_mounts_image_size_and_rejects_wrong_task():
    basket = "TwoRobotPickAppleBananaToBaskets-v1"
    checkpoint = {"data_contract": {"fps": 20, "datasets": [
        {"repo_id": "piper-pill", "env_id": TASK,
         "native": {"task_profile": {"mount_poses": [-.30]}} , "source_image_size": [224, 224]},
        {"repo_id": "piper-basket", "env_id": basket,
         "native": {"task_profile": {"mount_poses": [-.25]}}, "source_image_size": [256, 256]},
    ]}}
    for repo, task, mount, size in [("piper-pill", TASK, -.30, 224), ("piper-basket", basket, -.25, 256)]:
        selected = eval_act_single.dataset_contract(checkpoint, {"repo_id": repo, "env_id": task})
        assert selected["native"]["task_profile"]["mount_poses"] == [mount]
        assert selected["source_image_size"] == [size, size] and selected["fps"] == 20
    with pytest.raises(ValueError, match="saved dataset contract"):
        eval_act_single.dataset_contract(checkpoint, {"repo_id": "piper-basket", "env_id": TASK})


def test_basket_reporting_reads_native_predicates_without_inventing_grasp():
    info = {"apple_in_basket": True, "banana_in_basket": False,
            "is_apple_grasped": False, "is_banana_grasped": True, "is_robot_static": False}
    original = dict(info)
    events = eval_act_single._task_events(SimpleNamespace(), info)
    assert events["left_in_region"] and not events["right_in_region"]
    assert not events["left_grasped"] and events["right_grasped"]
    assert info == original


def test_native_main_uses_shared_live_rollout_and_records_condition(tmp_path, monkeypatch):
    _install_demo_loader(monkeypatch)
    checkpoint, agent = _checkpoint(), _ConditionedPolicy()
    path, config = tmp_path / "policy.pt", tmp_path / "config.json"
    torch.save(checkpoint, path)
    config.write_text(json.dumps([{"repo_id": "native-piper", "env_id": TASK}]))
    environments = []

    class Environment:
        def __init__(self):
            self.unwrapped = self
            self.robot_uids = ["piper"] * 2
            self.control_freq = 20
            self.action_space = spaces.Dict({f"piper-{side}": spaces.Box(-1., 1., (7,), np.float32)
                                             for side in range(2)})
            self.agent = SimpleNamespace(agents=[None, None],
                                         agents_dict={"piper-0": None, "piper-1": None})
            self.actions, self.seeds = [], []
            self.closed = False

        def observation(self):
            return {"sensor_data": {"base_camera": {"rgb": np.full((1, 8, 8, 3), self.steps, np.uint8)}},
                    "agent": {f"piper-{side}": {"qpos": torch.full((1, 8), self.steps / 10.)}
                              for side in range(2)}}

        def reset(self, seed):
            self.seeds.append(seed)
            self.steps = 0
            return self.observation(), {}

        def step(self, action):
            self.actions.append({key: value.copy() for key, value in action.items()})
            self.steps += 1
            # Early official termination is a report, not a new physical gate.
            return self.observation(), 0., True, self.steps == 3, {"success": True}

        def close(self):
            self.closed = True

    def make(env_id, **kwargs):
        assert env_id == TASK and kwargs["robot_uids"] == ("piper", "piper")
        assert kwargs["reward_mode"] == "none" and kwargs["control_mode"] == "pd_joint_pos"
        env = Environment()
        environments.append(env)
        return env

    monkeypatch.setattr(eval_act_single.gym, "make", make)
    normalizer = SimpleNamespace(resolve_idx=lambda _: 0, normalize_state=lambda value, _: value,
                                 denormalize_action=lambda value, _: value)
    monkeypatch.setattr(eval_act_single, "SingleTaskNormalizer", lambda *args, **kwargs: normalizer)
    monkeypatch.setattr(video_eval, "load_checkpoint", lambda *args: (agent, NativeArgs(**checkpoint["args"])))
    output = tmp_path / "out"
    summary = video_eval.main(["--checkpoint", str(path), "--sub-config", str(config),
                              "--seeds", "6100", "6101", "--max-episode-steps", "3",
                              "--no-save-trajectory", "--output-dir", str(output), "--device", "cpu"])
    env = environments[0]
    assert env.closed and env.seeds == [6100, 6101]
    assert len(env.actions) == 6 and len(agent.selected) == 2
    assert summary["episodes"] == summary["official_success_at_end"] == 2
    assert all(list(row) == ["piper-0", "piper-1"] for row in env.actions)
    assert all(block.shape == (7,) for row in env.actions for block in row.values())
    assert not np.array_equal(env.actions[0]["piper-0"], env.actions[1]["piper-0"])
    records = json.loads((output / summary["results"]).read_text())
    assert all(row["steps"] == 3 and row["human_demo_id"] == "content:correct" for row in records)
    assert all(row["conditioning"]["task_z_shape"] == [1, 1] for row in records)
    assert (output / summary["results"]).name.startswith("act_native_video_")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
