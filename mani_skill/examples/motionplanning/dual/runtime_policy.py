"""Versioned resolved execution settings, independent of simulator/GPU imports.

This first schema centralizes the legacy strict protocol. Retreat and final
stability describe selected quality requirements; episode/completion limits
remain independent time limits. This module stores policy and does not execute
or relax any runtime gate, select a new quality strategy, or expose a CLI.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError, asdict, dataclass, fields, replace
import hashlib
import json
import math
from typing import Any


@dataclass(frozen=True, slots=True)
class RuntimePolicy:
    """Complete resolved settings; distances are metres, rotations radians.

    Counts are control ticks. Frequencies remain the caller's declared runtime
    configuration. Window lengths do not by themselves prove grasp, stability
    or sufficient future budget; physical measurements are still required.
    """

    policy_id: str = "legacy-strict-v1"
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
    minimum_settle_budget: int = 3
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
        for name in ("minimum_settle_budget", "last_completion_step"):
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
        if self.minimum_settle_budget > self.max_settle_steps:
            raise ValueError("minimum_settle_budget cannot exceed max_settle_steps")
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


LEGACY_STRICT = RuntimePolicy()


def canonical_sha256(policy: RuntimePolicy | Mapping[str, Any]) -> str:
    """Hash validated resolved values as sorted compact finite JSON (UTF-8)."""
    resolved = policy if isinstance(policy, RuntimePolicy) else RuntimePolicy.from_dict(policy)
    payload = json.dumps(resolved.to_dict(), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolved_policy_record(policy: RuntimePolicy = LEGACY_STRICT) -> dict[str, Any]:
    """Record all resolved values and their canonical identity in a report."""
    if not isinstance(policy, RuntimePolicy):
        raise ValueError("A validated RuntimePolicy is required")
    return {"resolved": policy.to_dict(), "sha256": canonical_sha256(policy)}


def policy_from_report(report: Mapping[str, Any]) -> RuntimePolicy:
    """Use legacy settings only when the runtime_policy key is absent.

    An explicit malformed record, incomplete resolved configuration or changed
    hash must fail rather than silently reinterpret a recorded trajectory.
    """
    if not isinstance(report, Mapping):
        raise ValueError("Report must be a mapping")
    if "runtime_policy" not in report:
        return LEGACY_STRICT
    record = report["runtime_policy"]
    if not isinstance(record, Mapping) or set(record) != {"resolved", "sha256"}:
        raise ValueError("runtime_policy must contain exactly resolved and sha256")
    policy = RuntimePolicy.from_dict(record["resolved"])
    digest = record["sha256"]
    if not isinstance(digest, str) or digest != canonical_sha256(policy):
        raise ValueError("runtime_policy canonical SHA256 disagrees with resolved values")
    return policy


def self_test() -> dict[str, Any]:
    """Exercise legacy round trips, immutability and malformed configurations."""
    original = LEGACY_STRICT.to_dict()
    assert len(original) == 17 and original["last_completion_step"] == 199
    assert RuntimePolicy.from_dict(original) == LEGACY_STRICT
    record = resolved_policy_record()
    assert policy_from_report({}) is LEGACY_STRICT
    assert policy_from_report({"runtime_policy": json.loads(json.dumps(record))}) == LEGACY_STRICT
    assert canonical_sha256(dict(reversed(list(original.items())))) == record["sha256"]
    assert len(record["sha256"]) == 64
    record["resolved"]["grasp_stable_steps"] = 7
    assert LEGACY_STRICT.grasp_stable_steps == 5
    try:
        LEGACY_STRICT.grasp_stable_steps = 6
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("Runtime policy must remain frozen")
    # Quality selection never changes time limits implicitly.
    flags = replace(LEGACY_STRICT, require_retreat=False, require_final_stability=False)
    assert flags.max_episode_steps == 200 and flags.last_completion_step == 199
    assert canonical_sha256(flags) != canonical_sha256(LEGACY_STRICT)
    assert RuntimePolicy(position_slip_m=0).position_slip_m == 0.
    invalid = [dict(original, unknown=1), {k:v for k,v in original.items() if k != "min_lift_m"},
               dict(original, schema_version=2), dict(original, schema_version=True),
               dict(original, policy_id="  "), dict(original, grasp_stable_steps=True),
               dict(original, release_stable_steps=0), dict(original, reach_stable_steps=3.),
               dict(original, minimum_settle_budget=-1), dict(original, minimum_settle_budget=16),
               dict(original, max_episode_steps=0), dict(original, last_completion_step=200),
               dict(original, require_retreat=1), dict(original, require_final_stability=None)]
    for name in ("position_slip_m", "rotation_slip_rad", "retreat_command_m", "minimum_retreat_m", "min_lift_m"):
        invalid += [dict(original, **{name:value}) for value in (float("nan"), float("inf"), -.01, True, ".1")]
    rejected = 0
    for item in invalid:
        try:
            RuntimePolicy.from_dict(item)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError("Malformed resolved policy accepted")
    for bad in (None, {}, {"resolved": original}, {"resolved": original, "sha256": "0"*64},
                {"resolved": original, "sha256": canonical_sha256(LEGACY_STRICT), "unknown": 1}, record):
        try:
            policy_from_report({"runtime_policy": bad})
        except ValueError:
            rejected += 1
        else:
            raise AssertionError("Malformed policy report silently accepted")
    assert LEGACY_STRICT.to_dict() == original
    return {"status":"passed", "schema_version":1, "policy_id":LEGACY_STRICT.policy_id,
            "resolved_field_count":len(original), "canonical_sha256":canonical_sha256(LEGACY_STRICT),
            "rejected_invalid_cases":rejected, "environment_creations":0, "physics_steps":0}


if __name__ == "__main__":
    print(json.dumps(self_test(), sort_keys=True, allow_nan=False))
