"""Training configuration must preserve native dimensions and task pairing."""
import json
from types import SimpleNamespace

import pytest

from examples.baselines.act.train_act_native_video import (
    TrainingArgs, model_configuration, robot_task_id, train,
)


def test_native_dimensions_own_the_policy_and_json_selects_video_encoder(tmp_path):
    path = tmp_path / "model.json"
    path.write_text(json.dumps({"state_dim": 18, "action_dim": 16,
                                "frozen_backbone_type": "dinov2_vitl14",
                                "hf_cache_dir": "/models/hf/hub"}))
    args = TrainingArgs(model_config=str(path), state_dim=24, action_dim=14,
                        cameras=["zed2i"], pred_horizon=16, video_backend="opencv")
    model = model_configuration(args)
    assert (model.state_dim, model.action_dim, model.pred_horizon) == (24, 14, 16)
    assert model.frozen_backbone_type == "dinov2_vitl14"
    assert model.hf_cache_dir == "/models/hf/hub"
    assert model.video_backend == "opencv" and model.input_mode == "video_only"


def test_task_pairing_uses_recorded_task_and_rejects_silent_multitask_mixing():
    dataset = SimpleNamespace(main_dataset=SimpleNamespace(meta=SimpleNamespace(
        info={"native_data_contract": {"env_id": "TwoRobotPickPillToRegions-v1"}})))
    assert robot_task_id([{"repo_id": "arx-pills"}], dataset) == "TwoRobotPickPillToRegions-v1"
    with pytest.raises(ValueError, match="one robot dataset/task"):
        robot_task_id([{"env_id": "task-a"}, {"env_id": "task-b"}], dataset)
    with pytest.raises(ValueError, match="recorded native task"):
        robot_task_id([{"env_id": "another-task"}], dataset)
    dataset.main_dataset.meta.info.clear()
    with pytest.raises(ValueError, match="identify its task"):
        robot_task_id([{"repo_id": "unidentified"}], dataset)


def test_training_refuses_to_overwrite_an_existing_run_before_loading_data(tmp_path):
    (tmp_path / "metrics.jsonl").write_text("preserve this run\n")
    with pytest.raises(FileExistsError):
        train(TrainingArgs(output_dir=str(tmp_path), human_manifest="unused.json", cuda=False))
    assert (tmp_path / "metrics.jsonl").read_text() == "preserve this run\n"
