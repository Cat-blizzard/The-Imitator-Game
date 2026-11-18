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


def test_original_task_uses_one_bounded_official_attempt_and_native_recording(tmp_path):
    task = collect_pick_place.selected_task(["TwoRobotPickPillToRegions"])
    command, source = collect_pick_place.worker_spec(task, "panda", 6000, tmp_path)
    assert "mani_skill.examples.motionplanning.dual.two_robot_run" in command
    assert command[command.index("--num-traj")+1] == "1"
    assert command[command.index("--seed")+1] == "6000"
    assert "--only-count-success" not in command
    assert source == tmp_path/task/"motionplanning"/"trajectory.h5"
    source.parent.mkdir(parents=True)
    source.with_suffix(".json").write_text(json.dumps({"episodes": [{"episode_id": 0,
        "dual_sync": {"official": {"success": True, "elapsed_steps": 171},
                      "execution": {"status": "incomplete"},
                      "quality": {"status": "failed"},
                      "recording": {"status": "passed"}}}]}))
    result = collect_pick_place.worker_result(task, tmp_path, source)
    assert result["task_success"] and result["steps"] == 171
    assert result["execution"]["status"] == "incomplete"
    assert result["quality"]["status"] == "failed"


def test_task_selection_keeps_shared_default_and_requires_one_supported_task():
    assert collect_pick_place.selected_task(None) == "DualArmPickPlace-v1"
    assert collect_pick_place.selected_task(["DualArmPickPlace"]) == "DualArmPickPlace-v1"
    with pytest.raises(ValueError, match="exactly one"):
        collect_pick_place.selected_task(["DualArmPickPlace", "TwoRobotPickPillToRegions"])
    with pytest.raises(ValueError, match="Unsupported"):
        collect_pick_place.selected_task(["Other"])


@pytest.mark.parametrize("recording", ["passed", "saved"])
def test_both_native_recording_success_labels_are_accepted(recording):
    result = {"task_success": True, "execution": {"status": "passed"},
              "recording": {"status": recording}}
    assert collect_pick_place.training_usable(result, 0)
    assert not collect_pick_place.training_usable(result, 1)
    result["execution"]["status"] = "incomplete"
    assert not collect_pick_place.training_usable(result, 0)


def test_selected_quality_failure_is_not_a_training_episode_or_an_official_failure():
    result = {"task_success": True, "execution": {"status": "passed"},
              "recording": {"status": "saved"}}
    assert collect_pick_place.training_usable(result, 0)  # Legacy no-quality report.
    result["quality"] = {"status": "not_requested"}
    assert collect_pick_place.training_usable(result, 0)
    result["quality"] = {"status": "failed", "reason": "retreat distance"}
    assert not collect_pick_place.training_usable(result, 0)
    assert collect_pick_place.failure_reason(result, 0) == "retreat distance"
    assert result["task_success"] and result["execution"]["status"] == "passed"


@pytest.mark.parametrize("category,key,cause", [
    ("execution", "reason", "lift_and_hold: measured_collision"),
    ("quality", "reason", "Measured retreat below selected distance"),
    ("recording", "error", "trajectory_flush_failed"),
])
def test_specific_failure_reason_survives_official_outcome(category, key, cause):
    result = {"task_success": category != "execution", category: {"status": "failed", key: cause}}
    assert collect_pick_place.failure_reason(result, 0) == cause


@pytest.mark.parametrize("kind", ["timeout", "startup"])
def test_worker_primary_failure_survives_missing_result(tmp_path, monkeypatch, kind):
    # Finish optional dependency imports before replacing the shared subprocess module.
    from mani_skill.examples.motionplanning.dual.sync.pick_place import POLICY

    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"schema_version": 1, "seed_splits": {
        "train": [1], "development": [2], "evaluation": [3]}}))
    root = tmp_path / "data"
    class TimedOutWorker:
        def wait(self, timeout):
            raise collect_pick_place.subprocess.TimeoutExpired("expert", timeout)
    def launch(*args, **kwargs):
        if kind == "startup":
            raise OSError("startup unavailable")
        return TimedOutWorker()
    monkeypatch.setattr(collect_pick_place.subprocess, "Popen", launch)
    monkeypatch.setattr(collect_pick_place, "_stop", lambda *args: None)
    args = SimpleNamespace(seed_splits=str(splits), target_episodes=1, max_attempts=1,
        tasks=["TwoRobotPickPillToRegions"], demos_dir=str(root), dry_run=False,
        robot="piper", split="train", task_timeout=1)
    assert collect_pick_place.collect(args) == 2
    finish = collect_pick_place.read_events(root / "attempts.jsonl")[-1]
    assert finish["failure_reason"].startswith("worker_timeout" if kind == "timeout" else "worker_startup")
    assert finish["worker_error"] == finish["failure_reason"]
    assert finish["result_read_error"].startswith("worker_result_invalid")


@pytest.mark.parametrize("profile_kind", ["missing", "uncoupled", "old_acceptance"])
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
                "acceptance_rule_version": collect_pick_place.ACCEPTANCE_RULE_VERSION,
                "profile": asdict(PROFILES["piper"]), "policy": POLICY.to_dict()}
    if profile_kind == "uncoupled":
        settings["robot_profile"] = asdict(replace(get_robot_profile("piper"),
                                                  coupled_gripper_joints=None))
    elif profile_kind == "old_acceptance":
        settings["robot_profile"] = asdict(get_robot_profile("piper"))
        settings.pop("acceptance_rule_version")
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
