"""Single-task ACT reads the sampling rate recorded in local dataset metadata."""

import json
from pathlib import Path

import pytest

from examples.baselines.act import train_act_single as training


def _write_config(tmp_path: Path, rates: list, *, override_root: bool = False) -> Path:
    """Write real metadata files; no dataset download, video decode or simulator."""
    configs = []
    for index, rate in enumerate(rates):
        repo_id = f"dataset-{index}"
        dataset_root = f"custom-{index}" if override_root else repo_id
        metadata = tmp_path / dataset_root / "meta"
        metadata.mkdir(parents=True)
        (metadata / "info.json").write_text(json.dumps({"fps": rate}))
        config = {"repo_id": repo_id}
        if override_root:
            config["root"] = dataset_root
        configs.append(config)
    path = tmp_path / "datasets.json"
    path.write_text(json.dumps(configs))
    return path


@pytest.mark.parametrize("fps", [20, 30, 29.97])
def test_reads_real_metadata_rate_without_rounding(tmp_path: Path, fps: float) -> None:
    config = _write_config(tmp_path, [fps, fps])
    assert training._dataset_fps(str(tmp_path), str(config)) == fps


def test_dataset_root_override_is_respected(tmp_path: Path) -> None:
    config = _write_config(tmp_path, [20], override_root=True)
    assert training._dataset_fps(str(tmp_path), str(config)) == 20


def test_mixed_rates_are_rejected_instead_of_silently_retimed(tmp_path: Path) -> None:
    config = _write_config(tmp_path, [20, 30])
    with pytest.raises(ValueError, match="one dataset sampling rate"):
        training._dataset_fps(str(tmp_path), str(config))


@pytest.mark.parametrize("fps", [0, -1, True, "20", float("nan"), float("inf")])
def test_invalid_metadata_rates_are_rejected(tmp_path: Path, fps: object) -> None:
    config = _write_config(tmp_path, [fps])
    with pytest.raises(ValueError, match="Invalid dataset fps"):
        training._dataset_fps(str(tmp_path), str(config))


def test_empty_dataset_configuration_is_rejected(tmp_path: Path) -> None:
    config = _write_config(tmp_path, [])
    with pytest.raises(ValueError, match="configuration is empty"):
        training._dataset_fps(str(tmp_path), str(config))


@pytest.mark.parametrize("fps", [20, 30])
def test_build_dataloader_forwards_recorded_rate(tmp_path: Path, monkeypatch, fps: int) -> None:
    config = _write_config(tmp_path, [fps])
    captured = []
    dataset = [0]

    def capture_config(data_config):
        captured.append(data_config)
        return dataset

    monkeypatch.setattr(training, "_auto_detect_cameras", lambda *args: ["cam1"])
    monkeypatch.setattr(training, "build_lerobot_dataset", capture_config)
    args = training.TrainingArgs(
        sim_root=str(tmp_path), sub_config=str(config), batch_size=1, num_workers=0
    )
    loader, returned_dataset = training.build_dataloader(args)
    assert returned_dataset is dataset
    assert loader.dataset is dataset
    assert len(captured) == 1
    assert captured[0].fps == fps
    assert captured[0].horizon == args.pred_horizon
    assert captured[0].obs_horizon == args.obs_horizon


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
