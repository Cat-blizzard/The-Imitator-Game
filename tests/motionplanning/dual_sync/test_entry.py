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
from mani_skill.examples.motionplanning.dual.sync import entry


TASK = "TwoRobotPickPillToRegions-v1"


BASKET_TASK = "TwoRobotPickAppleBananaToBaskets-v1"
ROBOTS = ("panda_wristcam", "piper", "ur5_wsg", "arx_x5", "xarm6_robotiq")


def _collection_env(robot: str = "panda_wristcam", *, close=lambda: None) -> SimpleNamespace:
    """Expose live native orders without constructing a physical environment."""
    arm_count = 7 if robot == "panda_wristcam" else 6
    agents = {}
    for side in (0, 1):
        # State order deliberately differs from both arm order and the peer arm.
        state_names = (["gripper", *[f"joint{i}" for i in range(arm_count)]] if side == 0
                       else [*[f"joint{i}" for i in reversed(range(arm_count))], "gripper"])
        joints = [SimpleNamespace(get_name=lambda name=name: name)
                  for name in state_names]
        arm_joints = [SimpleNamespace(get_name=lambda index=index: f"joint{index}")
                      for index in range(arm_count)]
        agents[f"{robot}-{side}"] = SimpleNamespace(
            single_action_space=SimpleNamespace(shape=(arm_count+1,)),
            controller=SimpleNamespace(controllers={"arm": SimpleNamespace(joints=arm_joints)}),
            robot=SimpleNamespace(get_active_joints=lambda joints=joints: joints),
            gripper_coordinate_unit="rad" if robot == "xarm6_robotiq" else "m",
            gripper_open_action=0. if robot == "xarm6_robotiq" else 1.,
            gripper_closed_action=.81 if robot == "xarm6_robotiq" else -1.)
    env = SimpleNamespace(control_freq=20, robot_uids=(robot, robot),
        agent=SimpleNamespace(agents_dict=agents, agents=list(agents.values())),
        _json_data={}, _h5_file=SimpleNamespace(filename="synthetic.h5"), close=close)
    env.unwrapped = env
    return env


def test_serial_remains_default_and_seed_is_explicit():
    args = runner.parse_args(["-e", TASK, "--seed", "401"])
    assert args.execution_mode == "serial"
    assert args.planner_backend == "mplib"
    assert args.curobo_device == "cuda:0"
    assert args.seed == 401
    assert entry.resolve_solver(args, runner.MP_SOLUTIONS) is runner.MP_SOLUTIONS[TASK]


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
    assert entry.resolve_solver(args, runner.MP_SOLUTIONS) is module.solve
    assert calls == []
    args.sync_config = "task.json"
    result = entry.resolve_solver(args, runner.MP_SOLUTIONS)(None, seed=401)
    assert result == {"seed": 401, "policy": policy}
    assert calls == ["task.json"]


def test_sync_config_requires_sync_mode():
    args = runner.parse_args(["-e", TASK, "--sync-config", "task.json"])
    with pytest.raises(ValueError, match="requires"):
        entry.resolve_solver(args, runner.MP_SOLUTIONS)


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
    actual_env, result = entry.resolve_solver(runner.parse_args(argv), runner.MP_SOLUTIONS)(env, seed=401)
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
        entry.resolve_solver(args, runner.MP_SOLUTIONS)(None)
    assert seen[0]["planner_backend"] == "curobo"
    assert seen[0]["curobo_options"]["device"] == "cuda:0"
    assert seen[0]["curobo_options"]["cache_dir"] is None


@pytest.mark.parametrize("backend,label,robot", [
    ("mplib", "MPLib", "panda_wristcam"), ("curobo", "cuRobo", "panda_wristcam"),
    *[("mplib", "MPLib", robot) for robot in ROBOTS[1:]],
])
def test_recorder_metadata_uses_selected_backend_and_native_layout(monkeypatch, backend, label, robot):
    module = ModuleType("mani_skill.examples.motionplanning.dual.sync.solve")
    def stop_before_physics(env, **kwargs):
        raise RuntimeError("stop before physics")
    module.solve = stop_before_physics
    module.load_policy = lambda path: pytest.fail("loaded unrequested policy")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    env = _collection_env(robot)
    if backend == "curobo":
        backend_module = ModuleType("mani_skill.examples.motionplanning.dual.sync.curobo_backend")
        backend_module.CuroboSession = lambda **kwargs: SimpleNamespace(close=lambda: None)
        monkeypatch.setitem(sys.modules, backend_module.__name__, backend_module)
    monkeypatch.setattr(runner.gym, "make", lambda *a, **k: env)
    recorded = []
    monkeypatch.setattr(runner, "RecordEpisode", lambda env, **kwargs: recorded.append(kwargs) or env)
    args = runner.parse_args(["-e", TASK, "--execution-mode", "sync", "--robot", robot,
                              "--planner-backend", backend, "--seed", "401"])
    with pytest.raises(RuntimeError, match="stop before physics"):
        runner._main(args)
    pair = "dual-Panda" if robot == "panda_wristcam" else "dual-arm"
    assert recorded[0]["source_desc"] == f"synchronous {label} {pair} solution"
    assert recorded[0]["video_fps"] == 20
    arm_count = 7 if robot == "panda_wristcam" else 6
    expected_layout = [{"key": f"{robot}-{side}", "size": arm_count+1,
        "arm_joint_names": [f"joint{i}" for i in range(arm_count)],
        "state_joint_names": (["gripper", *[f"joint{i}" for i in range(arm_count)]] if side == 0
                              else [*[f"joint{i}" for i in reversed(range(arm_count))], "gripper"]),
        "arm_coordinate_unit": "rad",
        "gripper_coordinate_unit": "rad" if robot == "xarm6_robotiq" else "m",
        "gripper_open_action": 0. if robot == "xarm6_robotiq" else 1.,
        "gripper_closed_action": .81 if robot == "xarm6_robotiq" else -1.}
        for side in (0, 1)]
    assert env._json_data["motion_planning"] == {
        "execution_mode": "sync", "planner_backend": backend, "control_freq": 20,
        "action_order": [f"{robot}-0", f"{robot}-1"],
        "action_sizes": [arm_count+1, arm_count+1], "robot_uids": [robot, robot],
        "native_action_layout": expected_layout,
    }


@pytest.mark.parametrize("task,robot,backend", [
    *[(TASK, robot, "mplib") for robot in ROBOTS],
    (TASK, "panda", "mplib"), (TASK, "panda_wristcam", "curobo"),
    (BASKET_TASK, "piper", "mplib"),
])
def test_supported_task_robot_backend_matrix_reaches_environment(monkeypatch, task, robot, backend):
    class ReachedEnvironment(Exception):
        pass

    seen = []
    def make(env_id, **kwargs):
        seen.append((env_id, kwargs))
        raise ReachedEnvironment

    # Selection is tested separately from planning and environment construction.
    for name in ("solve", "apple_banana"):
        module = ModuleType(f"mani_skill.examples.motionplanning.dual.sync.{name}")
        module.solve = lambda *args, **kwargs: pytest.fail("ran a policy")
        module.load_policy = lambda path: pytest.fail("loaded unrequested policy")
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(runner.gym, "make", make)
    args = runner.parse_args(["-e", task, "--execution-mode", "sync", "--robot", robot,
                              "--planner-backend", backend])
    with pytest.raises(ReachedEnvironment):
        runner._main(args)
    expected_robot = "panda_wristcam" if robot == "panda" else robot
    assert len(seen) == 1 and seen[0][0] == task
    assert seen[0][1]["robot_uids"] == (expected_robot, expected_robot)
    assert seen[0][1]["control_mode"] == "pd_joint_pos"
    assert "max_episode_steps" not in seen[0][1]


@pytest.mark.parametrize("task,robot", [
    *[(TASK, robot) for robot in ROBOTS[1:]], (BASKET_TASK, "piper"),
])
def test_unsupported_curobo_combination_rejected_before_environment_or_worker(monkeypatch, task, robot):
    monkeypatch.setattr(runner.gym, "make", lambda *a, **k: pytest.fail("created env"))
    backend = ModuleType("mani_skill.examples.motionplanning.dual.sync.curobo_backend")
    backend.CuroboSession = lambda **kwargs: pytest.fail("started a cuRobo worker")
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    args = runner.parse_args(["-e", task, "--execution-mode", "sync", "--robot", robot,
                              "--planner-backend", "curobo"])
    with pytest.raises(ValueError, match="only the Panda pill task"):
        runner._main(args)


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
    assert callable(runner.sync_entry.resolve_solver(args, runner.MP_SOLUTIONS))
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
    entry.flush_trajectory(env, "sync")
    recorded = json.loads(env._json_path.read_text())["episodes"][0]
    assert recorded["success"] is True
    assert recorded["dual_sync"]["quality"] == {"passed": False}
    assert recorded["dual_sync"]["recording"]["status"] == "saved"
    entry.flush_trajectory(env, "sync", save=False)
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
        monkeypatch.setattr(entry, "dump_json", fail)
    with pytest.raises(OSError, match="disk full"):
        entry.flush_trajectory(env, "sync")
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

    env = _collection_env(close=close_env)
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
    monkeypatch.setattr(entry, "resolve_solver", lambda args, serial_solutions: solve)
    monkeypatch.setattr(runner.gym, "make", lambda *args, **kwargs: env)
    monkeypatch.setattr(runner, "RecordEpisode", lambda actual_env, **kwargs: actual_env)
    monkeypatch.setattr(runner, "tqdm", lambda *args, **kwargs: Progress())
    monkeypatch.setattr(entry, "flush_trajectory", flush)
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
    dispatch = entry.resolve_solver
    state = _entry_harness(monkeypatch, outcomes=[{
        "official": True, "execution": {"status": "passed"},
        "quality": {"status": "passed"}}])
    module = ModuleType("mani_skill.examples.motionplanning.dual.sync.solve")
    module.solve = entry.resolve_solver(None, {})
    module.load_policy = lambda path: pytest.fail("loaded unrequested policy")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(entry, "resolve_solver", dispatch)
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


def test_serial_collection_keeps_original_recording_and_progress(monkeypatch):
    state = _entry_harness(monkeypatch)
    recorded = []
    monkeypatch.setattr(runner, "RecordEpisode", lambda env, **kwargs: recorded.append(kwargs) or env)
    args = runner.parse_args(["-e", TASK, "-n", "2"])
    assert runner._main(args, start_seed=541) == "synthetic.h5"
    assert state.sessions == []
    assert [call["seed"] for call in state.calls] == [541, 542]
    assert all("curobo_session" not in call for call in state.calls)
    assert recorded[0]["video_fps"] == 30
    assert recorded[0]["source_desc"] == "official motion planning solution from ManiSkill contributors"
    assert "motion_planning" not in state.env._json_data
    assert state.postfix[-1]["failed_motion_plan_rate"] == .5
    assert "execution_failure_rate" not in state.postfix[-1]
    assert state.env_closes == state.progress_closes == 1


def test_success_only_collection_reuses_session_across_filtered_attempts(monkeypatch):
    state = _entry_harness(monkeypatch)
    args = runner.parse_args(["-e", TASK, "--execution-mode", "sync",
        "--planner-backend", "curobo", "-n", "1", "--only-count-success"])
    runner._main(args, start_seed=541)
    assert state.flushed == [False, True]
    assert [call["seed"] for call in state.calls] == [541, 542]
    assert len(state.sessions) == 1
    assert all(call["curobo_session"] is state.sessions[0] for call in state.calls)
    # Filtering follows official success, even when the saved episode fails quality.
    assert state.postfix[-1]["official_success_rate"] == .5
    assert state.postfix[-1]["quality_failures"] == 1
    assert state.sessions[0].close_count == state.env_closes == state.progress_closes == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
