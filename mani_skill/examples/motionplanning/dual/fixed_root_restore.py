"""Restore recorded CPU Panda states while retaining already-exact fixed roots.

The fixed root pose is a precondition, compared exactly rather than with a
tolerance. Actors and all root velocity/joint fields are still restored. This
avoids a redundant root-pose setter; it is not a general PhysX checkpoint and
does not restore hidden contact, sleep or solver state. Unsupported scenes fail.
"""
from __future__ import annotations

import argparse
import copy
import json
from typing import Any

import numpy as np

from .sync_replay_metrics import LEAVES
from .replay_support import batch_state, unbatch_state


def restore_fixed_roots(raw: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Restore actor13/Panda31 leaves; skip only already-identical fixed poses.

    Input leaves are unbatched float arrays [13]/[31]. All roots and schemas
    are checked before the first mutation. Returns an explicit setter receipt.
    """
    if raw.gpu_sim_enabled or raw.num_envs != 1:
        raise ValueError("Fixed-root restoration supports one CPU environment")
    if set(state) != set(LEAVES) or any(set(state[c]) != names for c, names in LEAVES.items()):
        raise ValueError("Expected seven canonical state leaves")
    current = unbatch_state(raw.get_state_dict())
    if set(current) != set(state) or any(set(current[c]) != set(state[c]) for c in state):
        raise ValueError("Current and requested state schemas differ")
    for category in state:
        width = 13 if category == "actors" else 31
        for name, value in state[category].items():
            array = np.asarray(value)
            if array.shape != (width,) or array.dtype.kind != "f" or not np.isfinite(array).all():
                raise ValueError("Expected finite floating unbatched state leaves")
            if abs(float(np.linalg.norm(array[3:7])) - 1.) > 1e-3:
                raise ValueError("Invalid state quaternion")
    registry = raw.scene.state_dict_registry.articulations
    for name, requested in state["articulations"].items():
        articulation = registry[name]
        if articulation.max_dof != 9 or not bool(np.asarray(articulation.fixed_root_link).all()):
            raise ValueError("All requested Panda roots must be fixed")
        actual = current["articulations"][name]
        if actual.dtype != requested.dtype or not np.array_equal(actual[:7], requested[:7]):
            raise ValueError("Requested fixed root pose is not already exactly present")
    raw.set_state_dict({"actors": batch_state(state["actors"])})
    events = ["set_state_dict(actors)"]
    for name, requested in state["articulations"].items():
        articulation = registry[name]
        articulation.set_root_linear_velocity(requested[7:10].copy())
        articulation.set_root_angular_velocity(requested[10:13].copy())
        articulation.set_qpos(requested[13:22].copy())
        articulation.set_qvel(requested[22:31].copy())
        events.append(f"{name}:root_linear_velocity,root_angular_velocity,qpos,qvel")
    observed = unbatch_state(raw.get_state_dict())
    roots_exact = all(np.array_equal(observed["articulations"][n][:7], state["articulations"][n][:7])
                      for n in state["articulations"])
    if not roots_exact:
        raise RuntimeError("Root pose changed during the non-pose setters")
    return {"policy": "already_exact_fixed_root_pose", "root_pose_precondition_exact": True,
            "root_pose_postcondition_exact": roots_exact, "skipped_root_pose_setters": list(state["articulations"]),
            "setter_events": events, "hidden_physics_state_restored": False}


def self_test() -> dict[str, Any]:
    from types import SimpleNamespace as NS
    pose = np.array([0., 0., 0., 1., 0., 0., 0.], dtype=np.float32)
    state = {"actors": {n: np.r_[pose, np.zeros(6, np.float32)] for n in LEAVES["actors"]},
             "articulations": {n: np.r_[pose, np.zeros(24, np.float32)] for n in LEAVES["articulations"]}}
    class Raw:
        gpu_sim_enabled = False
        num_envs = 1
        def __init__(self) -> None:
            self.state = copy.deepcopy(state)
            self.calls: list[str] = []
            registry = {}
            for name in state["articulations"]:
                def setter(value: Any, *, key: str, section: slice) -> None:
                    self.calls.append(key); self.state["articulations"][key][section] = value
                from functools import partial
                registry[name] = NS(max_dof=9, fixed_root_link=np.array([True]),
                    set_root_linear_velocity=partial(setter, key=name, section=slice(7, 10)),
                    set_root_angular_velocity=partial(setter, key=name, section=slice(10, 13)),
                    set_qpos=partial(setter, key=name, section=slice(13, 22)),
                    set_qvel=partial(setter, key=name, section=slice(22, 31)))
            self.scene = NS(state_dict_registry=NS(articulations=registry))
        def get_state_dict(self) -> dict[str, Any]:
            return batch_state(self.state)
        def set_state_dict(self, value: dict[str, Any]) -> None:
            self.calls.append("actors"); self.state["actors"] = unbatch_state(value)["actors"]
    raw = Raw(); requested = copy.deepcopy(state)
    requested["articulations"][next(iter(LEAVES["articulations"]))][13] = .4
    result = restore_fixed_roots(raw, requested)
    assert result["root_pose_postcondition_exact"] and len(raw.calls) == 9
    assert all(np.array_equal(raw.state[c][n], requested[c][n]) for c in requested for n in requested[c])
    for mode in ("root", "movable", "nonfinite", "gpu"):
        raw = Raw(); requested = copy.deepcopy(state)
        name = next(iter(LEAVES["articulations"]))
        if mode == "root": requested["articulations"][name][2] = .1
        elif mode == "movable": raw.scene.state_dict_registry.articulations[name].fixed_root_link[:] = False
        elif mode == "nonfinite": requested["actors"]["apple_out"][0] = np.nan
        else: raw.gpu_sim_enabled = True
        try: restore_fixed_roots(raw, requested)
        except ValueError: pass
        else: raise AssertionError(f"Accepted invalid {mode}")
        assert not raw.calls
    return {"status": "passed", "checks": ["all non-root fields restored", "fixed root exact pre/postconditions",
        "root mismatch/movable root/nonfinite/GPU rejected before mutation"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(self_test(), indent=2))
