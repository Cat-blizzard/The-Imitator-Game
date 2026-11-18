"""CPU coverage for optional dispatch and independent recording facts."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest
import numpy as np

from mani_skill.examples.motionplanning.dual import two_robot_run as runner


TASK = "TwoRobotPickPillToRegions-v1"


def test_serial_remains_default_and_seed_is_explicit():
    args = runner.parse_args(["-e", TASK, "--seed", "401"])
    assert args.execution_mode == "serial"
    assert args.planner_backend == "mplib"
    assert args.curobo_device == "cuda:0"
    assert args.seed == 401
    assert runner._solver_for(args) is runner.MP_SOLUTIONS[TASK]


def test_unsupported_sync_rejected_before_environment_creation(monkeypatch):
    monkeypatch.setattr(runner.gym, "make", lambda *a, **k: pytest.fail("created env"))
    args = runner.parse_args(["-e", "TwoRobotPickAppleBananaToBaskets-v1", "--execution-mode", "sync"])
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


def test_curobo_requires_sync_before_environment_creation(monkeypatch):
    monkeypatch.setattr(runner.gym, "make", lambda *a, **k: pytest.fail("created env"))
    args = runner.parse_args(["-e", TASK, "--planner-backend", "curobo"])
    with pytest.raises(ValueError, match="requires --execution-mode sync"):
        runner._main(args)


@pytest.mark.parametrize("option,value", [
    ("--curobo-python", "python-curobo"),
    ("--curobo-device", "cuda:0"),
    ("--curobo-cache-dir", "robot-cache"),
])
def test_explicit_curobo_options_require_selected_backend(monkeypatch, option, value):
    monkeypatch.setattr(runner.gym, "make", lambda *a, **k: pytest.fail("created env"))
    args = runner.parse_args(["-e", TASK, "--execution-mode", "sync", option, value])
    with pytest.raises(ValueError, match="require --planner-backend curobo"):
        runner._main(args)


@pytest.mark.parametrize("cli_python,env_python,expected", [
    ("python-explicit", "python-environment", "python-explicit"),
    (None, "python-environment", "python-environment"),
    (None, None, sys.executable),
])
def test_curobo_route_resolves_interpreter_and_keeps_policy(
        monkeypatch, cli_python, env_python, expected):
    module = ModuleType("mani_skill.examples.motionplanning.dual.sync.solve")
    policy = object()
    module.load_policy = lambda path: policy
    module.solve = lambda env, **kwargs: (env, kwargs)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    if env_python is None:
        monkeypatch.delenv("TIG_CUROBO_PYTHON", raising=False)
    else:
        monkeypatch.setenv("TIG_CUROBO_PYTHON", env_python)
    argv = ["-e", TASK, "--execution-mode", "sync", "--planner-backend", "curobo",
            "--sync-config", "policy.json", "--curobo-device", "cuda:3",
            "--curobo-cache-dir", "robot-cache"]
    if cli_python:
        argv.extend(["--curobo-python", cli_python])
    env = object()
    actual_env, result = runner._solver_for(runner.parse_args(argv))(env, seed=401)
    assert actual_env is env
    assert result == {"seed": 401, "policy": policy, "planner_backend": "curobo",
                      "curobo_options": {"python": expected, "device": "cuda:3",
                                         "cache_dir": "robot-cache"}}


def test_curobo_route_preserves_backend_error_without_mplib_fallback(monkeypatch):
    module = ModuleType("mani_skill.examples.motionplanning.dual.sync.solve")
    seen = []
    def fail(env, **kwargs):
        seen.append(kwargs)
        raise RuntimeError("cuRobo unavailable")
    module.solve = fail
    module.load_policy = lambda path: pytest.fail("loaded unrequested policy")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    args = runner.parse_args(["-e", TASK, "--execution-mode", "sync",
                              "--planner-backend", "curobo"])
    with pytest.raises(RuntimeError, match="cuRobo unavailable"):
        runner._solver_for(args)(None)
    assert seen[0]["planner_backend"] == "curobo"
    assert seen[0]["curobo_options"]["device"] == "cuda:0"
    assert seen[0]["curobo_options"]["cache_dir"] is None


@pytest.mark.parametrize("backend,label", [("mplib", "MPLib"), ("curobo", "cuRobo")])
def test_recorder_metadata_uses_selected_backend(monkeypatch, backend, label):
    module = ModuleType("mani_skill.examples.motionplanning.dual.sync.solve")
    def stop_before_physics(env, **kwargs):
        raise RuntimeError("stop before physics")
    module.solve = stop_before_physics
    module.load_policy = lambda path: pytest.fail("loaded unrequested policy")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    env = SimpleNamespace(unwrapped=SimpleNamespace(control_freq=20),
        _json_data={}, _h5_file=SimpleNamespace(filename="synthetic.h5"), close=lambda: None)
    if backend == "curobo":
        backend_module = ModuleType("mani_skill.examples.motionplanning.dual.sync.curobo_backend")
        backend_module.CuroboSession = lambda **kwargs: SimpleNamespace(close=lambda: None)
        monkeypatch.setitem(sys.modules, backend_module.__name__, backend_module)
    monkeypatch.setattr(runner.gym, "make", lambda *a, **k: env)
    recorded = []
    monkeypatch.setattr(runner, "RecordEpisode", lambda env, **kwargs: recorded.append(kwargs) or env)
    args = runner.parse_args(["-e", TASK, "--execution-mode", "sync",
                              "--planner-backend", backend, "--seed", "401"])
    with pytest.raises(RuntimeError, match="stop before physics"):
        runner._main(args)
    assert recorded[0]["source_desc"] == f"synchronous {label} dual-Panda solution"
    assert recorded[0]["video_fps"] == 20
    assert env._json_data["motion_planning"] == {
        "execution_mode": "sync", "planner_backend": backend, "control_freq": 20,
        "action_order": ["panda_wristcam-0", "panda_wristcam-1"],
    }


def test_real_mplib_routes_import_without_curobo_or_cuda_initialization():
    script = """
import importlib.abc
import sys

class NoCurobo(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'curobo' or fullname.startswith('curobo.'):
            raise AssertionError('default route tried to import cuRobo')

sys.meta_path.insert(0, NoCurobo())
from mani_skill.examples.motionplanning.dual import two_robot_run as runner
for mode in ('serial', 'sync'):
    args = runner.parse_args(['-e', 'TwoRobotPickPillToRegions-v1', '--execution-mode', mode])
    assert callable(runner._solver_for(args))
assert not any(name == 'curobo' or name.startswith('curobo.') for name in sys.modules)
import torch
assert not torch.cuda.is_initialized()
"""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TIG_CUROBO_PYTHON="unused-python")
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


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


def _entry_harness(monkeypatch: pytest.MonkeyPatch, *, failure_at: str | None = None,
                   outcomes: list[dict[str, Any]] | None = None) -> SimpleNamespace:
    """Exercise the episode loop with owned CPU fakes and no renderer/worker."""
    state = SimpleNamespace(sessions=[], calls=[], postfix=[], flushed=[], env_closes=0,
                            progress_closes=0)
    outcomes = outcomes or [
        {"official": False, "execution": {"status": "failed", "category": "geometry"},
         "quality": {"status": "not_evaluated"}},
        {"official": True, "execution": {"status": "passed"},
         "quality": {"status": "failed"}},
    ]

    def close_env() -> None:
        state.env_closes += 1

    env = SimpleNamespace(control_freq=20, _json_data={},
        _h5_file=SimpleNamespace(filename="synthetic.h5"), close=close_env)
    env.unwrapped = env
    env.evaluate = lambda: {"success": np.bool_(state.current["official"])}
    state.env = env

    class Session:
        def __init__(self, *, options: dict[str, Any]) -> None:
            self.options, self.close_count = options, 0
            state.sessions.append(self)

        def close(self) -> None:
            self.close_count += 1

    class Progress:
        def update(self, count: int) -> None:
            pass

        def set_postfix(self, values: dict[str, Any]) -> None:
            state.postfix.append(values)

        def close(self) -> None:
            state.progress_closes += 1

    def solve(actual_env: Any, **kwargs: Any) -> tuple[Any, Any]:
        state.calls.append(kwargs)
        if failure_at == "solve":
            raise RuntimeError("synthetic solve failure")
        state.current = outcomes[len(state.calls)-1]
        actual_env.dual_sync_result = {
            "execution": state.current["execution"], "quality": state.current["quality"]}
        if state.current["execution"]["status"] != "passed":
            return -1, -1
        result = (None, None, None, None,
                  {"success": np.bool_(state.current["official"]), "elapsed_steps": np.int32(171)})
        return result, result

    def flush(actual_env: Any, mode: str, save: bool = True) -> None:
        state.flushed.append(save)
        if failure_at == "flush":
            raise OSError("synthetic flush failure")

    module = ModuleType("mani_skill.examples.motionplanning.dual.sync.curobo_backend")
    module.CuroboSession = Session
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(runner, "_solver_for", lambda args: solve)
    monkeypatch.setattr(runner.gym, "make", lambda *args, **kwargs: env)
    monkeypatch.setattr(runner, "RecordEpisode", lambda actual_env, **kwargs: actual_env)
    monkeypatch.setattr(runner, "tqdm", lambda *args, **kwargs: Progress())
    monkeypatch.setattr(runner, "_flush_trajectory", flush)
    return state


def test_curobo_batch_owns_one_session_and_reports_independent_outcomes(monkeypatch):
    state = _entry_harness(monkeypatch)
    args = runner.parse_args(["-e", TASK, "--execution-mode", "sync",
        "--planner-backend", "curobo", "--curobo-device", "cuda:3", "-n", "2"])
    assert runner._main(args, start_seed=541) == "synthetic.h5"
    assert len(state.sessions) == 1
    assert [call["seed"] for call in state.calls] == [541, 542]
    assert all(call["curobo_session"] is state.sessions[0] for call in state.calls)
    assert state.sessions[0].options["device"] == "cuda:3"
    assert state.sessions[0].close_count == state.env_closes == state.progress_closes == 1
    final = state.postfix[-1]
    assert final["official_success_rate"] == .5
    assert final["execution_failure_rate"] == .5
    assert final["failure_categories"] == {"geometry": 1}
    assert final["quality_failures"] == 1
    assert "failed_motion_plan_rate" not in final


def test_single_curobo_episode_keeps_backend_without_shared_pool(monkeypatch):
    dispatch = runner._solver_for
    state = _entry_harness(monkeypatch, outcomes=[{
        "official": True, "execution": {"status": "passed"},
        "quality": {"status": "passed"}}])
    module = ModuleType("mani_skill.examples.motionplanning.dual.sync.solve")
    module.solve = runner._solver_for(None)
    module.load_policy = lambda path: pytest.fail("loaded unrequested policy")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(runner, "_solver_for", dispatch)
    backend = sys.modules["mani_skill.examples.motionplanning.dual.sync.curobo_backend"]
    monkeypatch.setattr(backend, "CuroboSession", lambda **kwargs: pytest.fail("created shared pool"))
    args = runner.parse_args(["-e", TASK, "--execution-mode", "sync",
        "--planner-backend", "curobo", "-n", "1"])
    assert runner._main(args) == "synthetic.h5"
    assert len(state.calls) == 1 and state.sessions == []
    assert state.calls[0]["planner_backend"] == "curobo"
    assert "curobo_session" not in state.calls[0]
    assert state.env_closes == state.progress_closes == 1


@pytest.mark.parametrize("failure_at,error", [("solve", RuntimeError), ("flush", OSError)])
def test_owned_session_and_environment_close_on_episode_error(monkeypatch, failure_at, error):
    state = _entry_harness(monkeypatch, failure_at=failure_at)
    args = runner.parse_args(["-e", TASK, "--execution-mode", "sync",
        "--planner-backend", "curobo", "-n", "2"])
    with pytest.raises(error, match=f"synthetic {failure_at} failure"):
        runner._main(args)
    assert len(state.calls) == len(state.sessions) == 1
    assert state.sessions[0].close_count == state.env_closes == state.progress_closes == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
