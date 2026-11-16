"""Independent CPU fixtures for compact recording loader evidence boundaries.

Synthetic fixtures exercise format/protocol validation only. They do not claim
physical execution, collision safety, a real grasp or benchmark success. All
temporary fixtures are removed after testing; source recordings are read only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Callable

import h5py
import numpy as np

from .compact_replay_metrics import load_compact_recording
from .replay_metrics import ACTION_KEYS
from .sync_replay_metrics import LEAVES
from .validate_grasp import TASK

HERE = Path(__file__).resolve().parent
OPEN_PHASES = {"preparation", "pregrasp", "descend", "release", "retreat", "verify_placement"}
SOURCE_NAMES = ("validate_sync_compact.py", "dual_arm_execution.py", "dual_payload_collision.py")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_ticks(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def make_fixture(directory: Path, *, budget: int = 200, route: str = "merge-lift",
                 total_steps: int | None = None) -> dict[str, Any]:
    """Write T×16 actions, T+1×13 actors and T+1×31 Panda synthetic leaves."""
    if route not in ("merge-lift", "merge-transport"):
        raise ValueError("Unknown synthetic route")
    parts = [("preparation", 10), ("pregrasp", 3), ("descend", 3),
             ("close_and_confirm", 5), ("lift_and_hold", 5)]
    if route == "merge-transport":
        parts.append(("carry_raise", 3))
    parts += [("carry_orient_and_transport", 10), ("carry_lower", 5),
              ("release", 7), ("retreat", 5), ("verify_placement", 20)]
    if total_steps is not None:
        additional = total_steps - sum(count for _, count in parts)
        if additional < 0:
            raise ValueError("Requested synthetic trajectory is too short")
        parts = [(name, count + additional if name == "carry_orient_and_transport" else count)
                 for name, count in parts]
    phases = [phase for phase, count in parts for _ in range(count)]
    count = len(phases)
    directory.mkdir()
    actions = np.zeros((count, 16), dtype=np.float32)
    states = {category: {} for category in LEAVES}
    for category, names in LEAVES.items():
        for name in names:
            width = 13 if category == "actors" else 31
            array = np.zeros((count + 1, width), dtype=np.float32)
            array[:, 3] = 1
            if category == "actors":
                array[:, 2] = .1
            states[category][name] = array
    success = np.zeros(count, dtype=bool)
    success[-20:] = True
    ticks = []
    for index, phase in enumerate(phases):
        grip = 1 if phase in OPEN_PHASES else -1
        actions[index, [7, 15]] = grip
        command = {key: actions[index, side * 8:(side + 1) * 8].tolist()
                   for side, key in enumerate(ACTION_KEYS)}
        objects = []
        for side, name in enumerate(("apple_out", "banana_out")):
            actor = states["actors"][name][index + 1]
            objects.append({"actor": name, "pose": actor[:7].tolist(),
                "linear_velocity_m_s": actor[7:10].tolist(),
                "angular_velocity_rad_s": actor[10:13].tolist(),
                "is_grasping": grip == -1, "finger_contact_force_norms_N": [0., 0.]})
        ticks.append({"phase": phase, "environment_step": index + 1, "fault_injected": False,
            "actions": command, "requested_actions": command,
            "native_terminated": bool(success[index]), "native_truncated": False,
            "success_only_termination_continued": bool(success[index]),
            "official_evaluation": {"success": bool(success[index]), "fail": False},
            "objects": objects, "tcp_poses": [[0., 0., .2, 1., 0., 0., 0.]] * 2,
            "left_qpos": [0.] * 9, "right_qpos": [0.] * 9})
    with h5py.File(directory / "trajectory.h5", "w") as handle:
        group = handle.create_group("traj_0")
        group.create_dataset("actions", data=actions)
        for name, array in (("success", success), ("terminated", success),
                            ("truncated", np.zeros(count, dtype=bool))):
            group.create_dataset(name, data=array)
        for category, leaves in states.items():
            for name, array in leaves.items():
                group.create_dataset(f"env_states/{category}/{name}", data=array)
    snapshot = directory / "source"
    snapshot.mkdir()
    hashes = {}
    for name in SOURCE_NAMES:
        contents = (HERE / name).read_bytes()
        (snapshot / name).write_bytes(contents)
        hashes[name] = hashlib.sha256(contents).hexdigest()
    report = {"status": "passed", "mode": "positive", "task": TASK, "seed": 0,
        "max_episode_steps": budget, "control_timestep_s": .05,
        "backend": {"physics": "physx_cpu"}, "preparation_steps": 10,
        "total_environment_steps": count, "recording": {"recorded_steps": count,
        "action_key_order": list(ACTION_KEYS)}, "execution_configuration": {"route": route},
        "stages": [{"name": phase, "status": "passed", "environment_steps": length}
                   for phase, length in parts[1:]], "source_sha256": hashes,
        "success_only_termination_continuation_ticks": int(success.sum())}
    metadata = {"env_info": {"env_id": TASK, "max_episode_steps": budget,
        "env_kwargs": {"control_mode": "pd_joint_pos", "sim_backend": "cpu", "max_episode_steps": budget}},
        "episodes": [{"episode_id": 0, "episode_seed": 0, "reset_kwargs": {"seed": 0},
            "elapsed_steps": count, "control_mode": {key: "pd_joint_pos" for key in ACTION_KEYS}}]}
    write_json(directory / "report.json", report)
    write_json(directory / "trajectory.json", metadata)
    write_ticks(directory / "ticks.jsonl", ticks)
    return {"count": count, "actions_shape": list(actions.shape), "state_frames": count + 1}


def reject(directory: Path, mutation: Callable[[Path], None], label: str) -> dict[str, Any]:
    mutation(directory)
    try:
        load_compact_recording(directory)
    except (ValueError, KeyError, TypeError) as exc:
        return {"case": label, "rejected": True, "reason": str(exc)}
    raise AssertionError(f"Loader accepted invalid synthetic fixture: {label}")


def self_test() -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="tig-compact-loader-audit-") as temporary:
        root = Path(temporary)
        for route in ("merge-lift", "merge-transport"):
            for budget in (200, 400):
                directory = root / f"valid-{route}-{budget}"
                shapes = make_fixture(directory, route=route, budget=budget)
                loaded = load_compact_recording(directory)
                assert loaded["actions"].shape == (shapes["count"], 16)
                assert len(loaded["states"]) == shapes["state_frames"]
                results.append({"case": directory.name, "accepted": True, **shapes})

        def json_change(path: Path, filename: str, callback: Callable[[Any], None]) -> None:
            value = read_json(path / filename); callback(value); write_json(path / filename, value)

        def report_count(path: Path) -> None:
            json_change(path, "report.json", lambda value: value.update(total_environment_steps=value["total_environment_steps"] + 1))

        def short_state(path: Path) -> None:
            with h5py.File(path / "trajectory.h5", "r+") as handle:
                key = "traj_0/env_states/actors/apple_in"
                array = handle[key][:-1]; del handle[key]; handle.create_dataset(key, data=array)

        def truncation(path: Path) -> None:
            with h5py.File(path / "trajectory.h5", "r+") as handle:
                handle["traj_0/truncated"][-1] = True
            rows = [json.loads(line) for line in (path / "ticks.jsonl").read_text().splitlines()]
            rows[-1].update(native_truncated=True, success_only_termination_continued=False)
            write_ticks(path / "ticks.jsonl", rows)

        def action_mapping(path: Path) -> None:
            rows = [json.loads(line) for line in (path / "ticks.jsonl").read_text().splitlines()]
            rows[0]["actions"] = dict(reversed(list(rows[0]["actions"].items())))
            write_ticks(path / "ticks.jsonl", rows)

        def missing_close(path: Path) -> None:
            rows = [json.loads(line) for line in (path / "ticks.jsonl").read_text().splitlines()]
            with h5py.File(path / "trajectory.h5", "r+") as handle:
                for index, row in enumerate(rows):
                    if row["phase"] != "close_and_confirm": continue
                    row["phase"] = "descend"
                    for field in ("actions", "requested_actions"):
                        for key in ACTION_KEYS: row[field][key][-1] = 1.
                    handle["traj_0/actions"][index, 7] = 1.
                    handle["traj_0/actions"][index, 15] = 1.
            write_ticks(path / "ticks.jsonl", rows)
            counts = Counter(row["phase"] for row in rows)
            phases = list(dict.fromkeys(row["phase"] for row in rows))
            json_change(path, "report.json", lambda value: value.update(stages=[
                {"name": phase, "status": "passed", "environment_steps": counts[phase]} for phase in phases[1:]]))

        def wrong_closed_command(path: Path) -> None:
            rows = [json.loads(line) for line in (path / "ticks.jsonl").read_text().splitlines()]
            index = next(i for i, row in enumerate(rows) if row["phase"] == "lift_and_hold")
            for field in ("actions", "requested_actions"): rows[index][field][ACTION_KEYS[1]][-1] = 1.
            with h5py.File(path / "trajectory.h5", "r+") as handle: handle["traj_0/actions"][index, 15] = 1.
            write_ticks(path / "ticks.jsonl", rows)

        def empty_source(path: Path) -> None:
            json_change(path, "report.json", lambda value: value.update(source_sha256={}))

        def corrupt_source(path: Path) -> None:
            with (path / "source" / SOURCE_NAMES[0]).open("ab") as handle: handle.write(b"\n# audit corruption\n")

        cases = [("report_count", report_count, None), ("T_plus_one_shape", short_state, None),
                 ("native_tick200_truncation", truncation, 200), ("physical_action_mapping", action_mapping, None),
                 ("missing_close_phase", missing_close, None), ("right_hand_open_during_held_phase", wrong_closed_command, None),
                 ("missing_source_hashes", empty_source, None), ("source_snapshot_corruption", corrupt_source, None)]
        for label, mutation, steps in cases:
            directory = root / label
            make_fixture(directory, total_steps=steps)
            results.append(reject(directory, mutation, label))
    return {"status": "passed", "fixture_only": True, "physics_executed": False,
        "loader_sha256": hashlib.sha256((HERE / "compact_replay_metrics.py").read_bytes()).hexdigest(),
        "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "positive_cases": 4, "negative_cases": 8, "cases": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if not arguments.self_test: parser.error("Provide --self-test")
    print(json.dumps(self_test(), indent=2, allow_nan=False))
