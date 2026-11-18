"""Finite collection preserves failures and keeps evaluation seeds out of training."""
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.collect_pick_place import collection_state, read_splits
from scripts import collect_pick_place


def test_overlapping_seed_pools_are_rejected(tmp_path):
    path = tmp_path / "splits.json"
    path.write_text(json.dumps({"schema_version": 1, "seed_splits": {
        "train": [1], "development": [2], "evaluation": [1]}}))
    with pytest.raises(ValueError, match="unique"):
        read_splits(str(path))


def test_interrupted_launch_consumes_attempt_and_is_not_a_sample(tmp_path):
    pools = {"train": [1, 2], "development": [3], "evaluation": [4]}
    summary, manifest, attempted = collection_state(tmp_path, pools,
        [{"event": "launch", "seed": 1, "split": "train"}])
    assert summary["splits"]["train"]["attempts"] == 1
    assert summary["completed_attempts"] == 0
    assert attempted == {1}
    assert manifest["episodes"] == []


def test_task_success_and_training_usability_are_independent(tmp_path):
    pools = {"train": [1, 2], "development": [3], "evaluation": [4]}
    source = tmp_path / "train-2" / "trajectory.h5"
    source.parent.mkdir()
    source.touch()
    source.with_suffix(".json").write_text('{"episodes": [{"episode_id": 0}]}')
    events = [{"event": "launch", "seed": seed, "split": "train"} for seed in (1, 2)]
    for seed in (1, 2):
        events.append({"event": "finish", "seed": seed, "split": "train",
            "task_success": True, "training_usable": seed == 2,
            "failure_reason": "recording_failed" if seed == 1 else None,
            "h5_path": "train-2/trajectory.h5", "episode_id": 0})
    summary, manifest, attempted = collection_state(tmp_path, pools, events)
    assert summary["splits"]["train"]["task_successes"] == 2
    assert summary["splits"]["train"]["training_usable"] == 1
    assert summary["failure_reasons"] == {"recording_failed": 1}
    assert [row["seed"] for row in manifest["episodes"]] == [2]
    assert attempted == {1, 2}


def test_missing_accepted_source_stops_resume(tmp_path):
    pools = {"train": [1], "development": [2], "evaluation": [3]}
    with pytest.raises(FileNotFoundError, match="Accepted source"):
        collection_state(tmp_path, pools, [{"event": "finish", "seed": 1,
            "split": "train", "training_usable": True, "task_success": True,
            "h5_path": "missing.h5", "episode_id": 0}])


def test_saved_split_must_match_frozen_pool(tmp_path):
    pools = {"train": [1], "development": [2], "evaluation": [3]}
    with pytest.raises(ValueError, match="frozen seed split"):
        collection_state(tmp_path, pools,
            [{"event": "launch", "seed": 3, "split": "train"}])


@pytest.mark.parametrize("profile_kind", ["missing", "uncoupled"])
def test_resume_does_not_mix_unknown_or_different_gripper_dynamics(tmp_path, monkeypatch, profile_kind):
    from mani_skill.agents.robots.embodiment_profiles import get_robot_profile
    from mani_skill.envs.tasks.tabletop.dual_pick_place import PROFILES
    from mani_skill.examples.motionplanning.dual.sync.pick_place import POLICY

    seeds = tmp_path / "splits.json"
    seeds.write_text(json.dumps({"schema_version": 1, "seed_splits": {
        "train": [1], "development": [2], "evaluation": [3]}}))
    root = tmp_path / "previous-data"
    root.mkdir()
    settings = {"task": "DualArmPickPlace-v1", "robot": "piper", "fps": 20,
                "profile": asdict(PROFILES["piper"]), "policy": POLICY.to_dict()}
    if profile_kind == "uncoupled":
        settings["robot_profile"] = asdict(replace(get_robot_profile("piper"),
                                                  coupled_gripper_joints=None))
    config = root / "collection_config.json"
    config.write_text(json.dumps(settings))
    before = config.read_bytes()
    monkeypatch.setattr(collect_pick_place.subprocess, "Popen",
                        lambda *a, **k: pytest.fail("launched into an incompatible data version"))
    args = SimpleNamespace(seed_splits=str(seeds), target_episodes=1, max_attempts=1,
                           tasks=None, demos_dir=str(root), dry_run=False, robot="piper")
    with pytest.raises(ValueError, match="Collection configuration changed.*new data directory"):
        collect_pick_place.collect(args)
    assert config.read_bytes() == before
    assert not (root / "attempts.jsonl").exists()
