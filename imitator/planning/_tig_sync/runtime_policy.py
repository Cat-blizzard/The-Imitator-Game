"""Immutable numeric execution and optional placement-quality settings."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
import math
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class RuntimePolicy:
    """Complete resolved settings; distances are metres, rotations radians.

    Counts are control ticks. Frequencies remain the caller's declared runtime
    configuration. Window lengths do not by themselves prove grasp, stability
    or sufficient future budget; physical measurements are still required.
    """

    policy_id: str = "dual-panda-sync-v1"
    schema_version: int = 1
    grasp_stable_steps: int = 5
    position_slip_m: float = .005
    rotation_slip_rad: float = .15
    release_stable_steps: int = 3
    reach_stable_steps: int = 3
    retreat_command_m: float = .081
    minimum_retreat_m: float = .08
    final_stable_steps: int = 20
    min_lift_m: float = .10
    max_settle_steps: int = 15
    max_episode_steps: int = 200
    last_completion_step: int = 199
    require_retreat: bool = True
    require_final_stability: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.policy_id, str) or not self.policy_id.strip():
            raise ValueError("policy_id must be a nonempty string")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported runtime policy schema_version")
        positive_counts = ("grasp_stable_steps", "release_stable_steps", "reach_stable_steps",
                           "final_stable_steps", "max_settle_steps", "max_episode_steps")
        for name in positive_counts:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer control-tick count")
        for name in ("last_completion_step",):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer control-tick count")
        for name in ("position_slip_m", "rotation_slip_rad", "retreat_command_m",
                     "minimum_retreat_m", "min_lift_m"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative scalar in its declared unit")
            object.__setattr__(self, name, float(value))
        for name in ("require_retreat", "require_final_stability"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean quality-selection flag")
        if self.last_completion_step >= self.max_episode_steps:
            raise ValueError("last_completion_step must precede max_episode_steps")

    def to_dict(self) -> dict[str, Any]:
        """Return every resolved field, without a hash or implicit defaults."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RuntimePolicy:
        """Read a complete schema; unknown or omitted fields are errors."""
        if not isinstance(value, Mapping):
            raise ValueError("Resolved runtime policy must be a mapping")
        expected = {field.name for field in fields(cls)}
        actual = set(value)
        if actual != expected:
            raise ValueError(f"Runtime policy fields differ: missing={sorted(expected-actual)}, "
                             f"unknown={sorted(map(str, actual-expected))}")
        return cls(**dict(value))


DEFAULT_POLICY = RuntimePolicy()


def load_policy(path: str | Path) -> RuntimePolicy:
    """Apply a JSON object of optional quality/physical settings to the defaults.

    The official episode horizon and policy identity remain fixed. Invalid or
    unknown parameters are reported before creating an environment.
    """
    with Path(path).open(encoding="utf-8") as stream:
        overrides = json.load(stream)
    if not isinstance(overrides, dict):
        raise ValueError("Sync configuration must be a JSON object")
    fixed = {"policy_id", "schema_version", "max_episode_steps", "last_completion_step"}
    available = {field.name for field in fields(RuntimePolicy)} - fixed
    unsupported = set(overrides) - available
    if unsupported:
        raise ValueError(f"Unsupported sync configuration fields: {sorted(unsupported)}")
    return replace(DEFAULT_POLICY, **overrides)
