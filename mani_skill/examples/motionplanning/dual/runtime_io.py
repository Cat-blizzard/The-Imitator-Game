"""Recording IO, finite JSON conversion, source identity and state hashing.

Actions are recorded as T by 16 finite controller values; state leaves retain
T+1 frames. Units and joint order belong to the caller's recording contract.
No helper constructs, steps or restores an environment. Writes target only the
explicitly supplied path, and hashing never changes the observed state.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import signal
import subprocess
from typing import Any

import numpy as np

# Direct-script setup is test-only; normal package imports need no path changes.
if __name__ == "__main__" and not __package__:
    import mani_skill.examples.motionplanning.dual as _dual_package
    _dual_package.__path__ = [str(Path(__file__).parent), *_dual_package.__path__]
    __package__ = _dual_package.__name__

from .trajectory_execution import as_numpy

def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def git_identity(path: Path) -> dict[str, Any]:
    def read(*args: str) -> str:
        return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()
    return {'path': str(path), 'commit': read('rev-parse', 'HEAD'), 'status': read('status', '--porcelain')}


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def recording_summary(path: Path, expected_steps: int,
                      expected_actions: np.ndarray | None = None) -> dict[str, Any]:
    import h5py
    with h5py.File(path, "r") as h5:
        if expected_steps == 0:
            return {"path": str(path), "recorded_steps": 0}
        group = h5["traj_0"]
        lengths: dict[str, int] = {}
        group.visititems(lambda name, obj: lengths.update({name: len(obj)})
                         if isinstance(obj, h5py.Dataset) else None)
        actions = group["actions"][()]
        states = {k: n for k, n in lengths.items() if k.startswith("env_states/")}
        if actions.shape != (expected_steps, 16) or not np.isfinite(actions).all() or not states or any(
                n != expected_steps + 1 for n in states.values()):
            raise RuntimeError(f"Recording lengths disagree with actual steps: {lengths}")
        if expected_actions is not None and (expected_actions.shape != actions.shape or
                not np.allclose(actions, expected_actions, rtol=0, atol=1e-7)):
            raise RuntimeError("Recorded arm action order/values differ from submitted actions")
        return {"path": str(path), "recorded_steps": expected_steps,
                "actions_shape": list(actions.shape), "submitted_actions_matched": expected_actions is not None,
                "state_lengths": states,
                "state_playback_tested": False}


def save_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_value(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")


class ControlledTermination(RuntimeError):
    """External stop that unwinds native_run and preserves its partial evidence."""


def termination_handler(signum: int, frame: Any) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise ControlledTermination(f'External signal {signum}; finalize partial episode')


def _state_arrays(value: Any, prefix: str = "") -> dict[str, np.ndarray]:
    from .trajectory_execution import as_numpy
    if isinstance(value, dict):
        result: dict[str, np.ndarray] = {}
        for name, child in value.items():
            result.update(_state_arrays(child, prefix + "/" + str(name)))
        return result
    return {prefix: as_numpy(value).copy()}


def state_digest(raw: Any) -> str:
    """Hash every actual actor/articulation state and elapsed step, without motion."""
    values = _state_arrays(raw.get_state_dict())
    digest = hashlib.sha256(as_numpy(raw.elapsed_steps).tobytes())
    for name, value in sorted(values.items()):
        array = np.asarray(value)
        if not np.isfinite(array).all():
            raise ValueError('Nonfinite actual simulator state')
        digest.update(name.encode())
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def self_test() -> dict[str, Any]:
    """Exercise actual file IO and finite-state contracts without physics."""
    import tempfile
    from types import SimpleNamespace
    import h5py

    def rejects(function: Any, *args: Any, **kwargs: Any) -> None:
        try:
            function(*args, **kwargs)
        except (ValueError, RuntimeError):
            return
        raise AssertionError(f"Expected rejection from {function.__name__}")

    nested = {"actor": {"pose": np.arange(7, dtype=np.float64)[None]},
              "robot": np.arange(9, dtype=np.float32)[None]}
    flat = _state_arrays(nested)
    assert set(flat) == {"/actor/pose", "/robot"}
    flat["/robot"][0, 0] = -100
    assert nested["robot"][0, 0] == 0
    converted = json_value({1: (np.array([1, 2]), np.float32(.5), np.int64(3))})
    assert converted == {"1": [[1, 2], .5, 3]}
    raw = SimpleNamespace(elapsed_steps=np.array([3]), get_state_dict=lambda: nested)
    digest = state_digest(raw)
    assert len(digest) == 64 and digest == state_digest(raw)
    nested["robot"][0, 0] = 1
    assert digest != state_digest(raw)
    nested["robot"][0, 0] = np.nan
    rejects(state_digest, raw)
    nested["robot"][0, 0] = 0
    with tempfile.TemporaryDirectory(prefix="tig-runtime-io-selftest-") as directory:
        folder = Path(directory)
        save_json(folder / "one.json", {"ok": True})
        assert json.loads((folder / "one.json").read_text()) == {"ok": True}
        save_report(folder / "two.json", {"value": np.array([1, 2]), "name": "药瓶"})
        assert json.loads((folder / "two.json").read_text()) == {"value": [1, 2], "name": "药瓶"}
        rejects(save_report, folder / "nan.json", {"bad": float("nan")})
        empty = folder / "empty.h5"
        with h5py.File(empty, "w"):
            pass
        assert recording_summary(empty, 0)["recorded_steps"] == 0
        path = folder / "recording.h5"
        actions = np.zeros((3, 16), dtype=np.float32)
        with h5py.File(path, "w") as h5:
            group = h5.create_group("traj_0")
            group.create_dataset("actions", data=actions)
            group.create_dataset("env_states/actors/a", data=np.zeros((4, 13)))
            group.create_dataset("env_states/articulations/r", data=np.zeros((4, 18)))
        recording = recording_summary(path, 3, actions)
        assert recording["submitted_actions_matched"] is True
        rejects(recording_summary, path, 2)
        wrong = actions.copy(); wrong[0, 0] = .1
        rejects(recording_summary, path, 3, wrong)
        with h5py.File(path, "a") as h5:
            del h5["traj_0/env_states/actors/a"]
            h5.create_dataset("traj_0/env_states/actors/a", data=np.zeros((3, 13)))
        rejects(recording_summary, path, 3)
    previous = signal.getsignal(signal.SIGTERM)
    try:
        try:
            termination_handler(signal.SIGTERM, None)
        except ControlledTermination as error:
            assert "finalize partial episode" in str(error)
        else:
            raise AssertionError("Expected ControlledTermination")
        assert signal.getsignal(signal.SIGTERM) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGTERM, previous)
    identity = git_identity(Path.cwd())
    assert len(identity["commit"]) == 40 and isinstance(identity["status"], str)
    return {"status": "passed", "physical_environment_steps": 0,
            "recording_actions": 3, "recording_states": 4,
            "state_digest_finite_and_change_checks": True,
            "signal_handler_restored": signal.getsignal(signal.SIGTERM) == previous,
            "git_identity_read": True}


if __name__ == "__main__":
    print(json.dumps(self_test(), indent=2, allow_nan=False))
