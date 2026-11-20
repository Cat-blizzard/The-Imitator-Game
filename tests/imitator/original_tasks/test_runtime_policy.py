"""Configuration failures happen before a simulation is constructed."""
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from tempfile import TemporaryDirectory

from imitator.planning._tig_sync.runtime_policy import (
    DEFAULT_POLICY, RuntimePolicy, load_policy
)


def test_optional_quality_settings_keep_official_horizon():
    with TemporaryDirectory() as folder:
        path = Path(folder) / "sync.json"
        path.write_text(json.dumps({"require_retreat": False, "final_stable_steps": 7}), encoding="utf-8")
        selected = load_policy(path)
        assert not selected.require_retreat and selected.final_stable_steps == 7
        assert selected.max_episode_steps == 200 and selected.last_completion_step == 199
        assert DEFAULT_POLICY.require_retreat and DEFAULT_POLICY.final_stable_steps == 20
        assert RuntimePolicy.from_dict(selected.to_dict()) == selected
        try:
            selected.final_stable_steps = 3
        except FrozenInstanceError:
            pass
        else:
            raise AssertionError("Runtime settings must be immutable")


def test_invalid_or_fixed_settings_are_rejected():
    invalid = [[], {"unknown": 1}, {"max_episode_steps": 201},
               {"last_completion_step": 200}, {"policy_id": "different"},
               {"grasp_stable_steps": True}, {"position_slip_m": -1},
               {"require_final_stability": 0}]
    with TemporaryDirectory() as folder:
        path = Path(folder) / "sync.json"
        for value in invalid:
            path.write_text(json.dumps(value), encoding="utf-8")
            try:
                load_policy(path)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Invalid sync settings accepted: {value}")
