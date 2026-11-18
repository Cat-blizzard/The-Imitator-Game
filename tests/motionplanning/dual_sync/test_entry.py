"""CPU coverage for optional dispatch and independent recording facts."""

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from mani_skill.examples.motionplanning.dual import two_robot_run as runner


TASK = "TwoRobotPickPillToRegions-v1"


def test_serial_remains_default_and_seed_is_explicit():
    args = runner.parse_args(["-e", TASK, "--seed", "401"])
    assert args.execution_mode == "serial"
    assert args.seed == 401
    assert runner._solver_for(args) is runner.MP_SOLUTIONS[TASK]


def test_unsupported_sync_rejected_before_environment_creation(monkeypatch):
    monkeypatch.setattr(runner.gym, "make", lambda *a, **k: pytest.fail("created env"))
    args = runner.parse_args(["-e", "TwoRobotPickTennisBallGolfBall-v1", "--execution-mode", "sync"])
    with pytest.raises(ValueError, match="not supported"):
        runner._main(args)


def test_sync_dispatch_loads_policy_only_when_requested(monkeypatch):
    module = ModuleType("mani_skill.examples.motionplanning.dual.sync.solve")
    policy = object()
    calls = []
    module.load_policy = lambda path: calls.append(path) or policy
    module.solve = lambda env, **kwargs: kwargs
    monkeypatch.setitem(sys.modules, module.__name__, module)
    args = runner.parse_args(["-e", TASK, "--execution-mode", "sync"])
    assert runner._solver_for(args) is module.solve
    assert calls == []
    args.sync_config = "task.json"
    result = runner._solver_for(args)(None, seed=401)
    assert result == {"seed": 401, "policy": policy}
    assert calls == ["task.json"]


def test_sync_config_requires_sync_mode():
    args = runner.parse_args(["-e", TASK, "--sync-config", "task.json"])
    with pytest.raises(ValueError, match="requires"):
        runner._solver_for(args)


def test_sync_metadata_keeps_official_success(tmp_path):
    env = SimpleNamespace(
        _json_data={"episodes": []}, _json_path=tmp_path / "trajectory.json",
        unwrapped=SimpleNamespace(dual_sync_result={"quality": {"passed": False}}),
    )
    env.flush_trajectory = lambda save: env._json_data["episodes"].append({"success": True}) if save else None
    runner._flush_trajectory(env, "sync")
    recorded = json.loads(env._json_path.read_text())["episodes"][0]
    assert recorded["success"] is True
    assert recorded["dual_sync"]["quality"] == {"passed": False}
    assert recorded["dual_sync"]["recording"]["status"] == "saved"
    runner._flush_trajectory(env, "sync", save=False)
    assert len(env._json_data["episodes"]) == 1


def test_main_passes_seed(monkeypatch):
    seen = []
    monkeypatch.setattr(runner, "_main", lambda args, **kwargs: seen.append(kwargs))
    runner.main(runner.parse_args(["-e", TASK, "--seed", "401"]))
    assert seen == [{"start_seed": 401}]



@pytest.mark.parametrize("failure_at", ["flush", "metadata"])
def test_recording_failure_does_not_change_official_success(tmp_path, monkeypatch, failure_at):
    facts = {"official": {"success": True}, "quality": {"status": "passed"}}
    env = SimpleNamespace(_json_data={"episodes": []}, _json_path=tmp_path / "trajectory.json",
        unwrapped=SimpleNamespace(dual_sync_result=facts))

    def fail(*args, **kwargs):
        raise OSError("disk full")

    def flush(save):
        env._json_data["episodes"].append({"success": True})

    env.flush_trajectory = fail if failure_at == "flush" else flush
    if failure_at == "metadata":
        monkeypatch.setattr(runner, "dump_json", fail)
    with pytest.raises(OSError, match="disk full"):
        runner._flush_trajectory(env, "sync")
    assert facts["official"]["success"] is True
    assert facts["quality"]["status"] == "passed"
    assert facts["recording"] == {"status": "failed", "reason": "OSError: disk full"}
    if env._json_data["episodes"]:
        episode = env._json_data["episodes"][0]
        assert episode["success"] is True
        assert episode["dual_sync"]["recording"]["status"] == "failed"
