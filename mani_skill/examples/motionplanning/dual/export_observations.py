"""Render policy observations from an immutable accepted dual-Panda recording.

This is state playback for observation extraction, not a new physical execution
or a task re-evaluation. Each stored frame is restored without a physics step.
The original T actions are unchanged; the terminal observation has no action.
CPU-only ``--self-test`` covers the data contract and state-playback loop.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import traceback
from typing import Any, Callable

import h5py
import numpy as np

SCHEMA = "tig-dual-arm-observations-v1"
TASK = "TwoRobotPickPillToRegions-v1"
AGENT_KEYS = ("panda_wristcam-0", "panda_wristcam-1")
ARTICULATION_KEYS = ("panda_wristcam-agent-0", "panda_wristcam-agent-1")
CAMERA = "zed2i"
CONTROL_HZ = 20
IMAGE_SHAPE = (224, 224, 3)
STATE_NAMES = ([f"left_joint_{i}" for i in range(7)] + ["left_gripper1", "left_gripper2"]
               + [f"right_joint_{i}" for i in range(7)] + ["right_gripper1", "right_gripper2"])
ACTION_NAMES = ([f"left_joint_{i}" for i in range(7)] + ["left_gripper1"]
                + [f"right_joint_{i}" for i in range(7)] + ["right_gripper1"])


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_identity(directory: Path) -> dict[str, dict[str, Any]]:
    """Hash every existing source file, including archived code and scores."""
    root = Path(directory).resolve()
    require(root.is_dir(), "An existing source directory is required")
    result = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "Source inventory must not contain symlinks")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = {
                "size": path.stat().st_size, "sha256": file_sha256(path)}
    require(bool(result), "Source directory is empty")
    return result


def save_json(path: Path, value: Any) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def qpos_from_state(state: dict[str, Any]) -> np.ndarray:
    """Stored Panda31 layout is root13, physical qpos9, qvel9, left then right."""
    joints = []
    for key in ARTICULATION_KEYS:
        leaf = np.asarray(state["articulations"][key])
        require(leaf.shape == (31,) and leaf.dtype == np.float32 and np.isfinite(leaf).all(),
                "Expected a finite float32 Panda31 state")
        joints.append(leaf[13:22])
    return np.concatenate(joints).copy()


def extract_policy_observation(raw: Any) -> tuple[np.ndarray, np.ndarray]:
    """Read standard sensor RGB and physical joints, without contact/task queries.

    _get_obs_sensor_data is BaseEnv's standard render/capture/texture pipeline.
    Calling get_obs would also invoke task extras that depend on fresh contact
    evaluation; those extras are deliberately outside this policy input.
    """
    require(tuple(raw.agent.agents_dict) == AGENT_KEYS, "Canonical left/right agent order differs")
    sensor_data = raw._get_obs_sensor_data()
    require(CAMERA in sensor_data and "rgb" in sensor_data[CAMERA], "Missing zed2i RGB sensor")
    rgb = as_numpy(sensor_data[CAMERA]["rgb"])
    if rgb.shape == (1,) + IMAGE_SHAPE:
        rgb = rgb[0]
    require(rgb.shape == IMAGE_SHAPE and rgb.dtype == np.uint8, "Expected native uint8 zed2i224 RGB")
    qpos = []
    for key in AGENT_KEYS:
        value = as_numpy(raw.agent.agents_dict[key].robot.get_qpos())
        require(value.shape == (1, 9) and value.dtype == np.float32 and np.isfinite(value).all(),
                "Expected actual float32 physical Panda qpos [1,9]")
        qpos.append(value[0])
    return rgb.copy(), np.concatenate(qpos).copy()


def validate_observation_arrays(rgb: np.ndarray, qpos: np.ndarray, actions: np.ndarray,
                                timestamps_s: np.ndarray) -> int:
    """Require o[0:T+1], q[0:T+1], original a[0:T], t=i/20; no terminal action."""
    count = len(actions)
    require(count > 0 and actions.shape == (count, 16) and actions.dtype == np.float32
            and np.isfinite(actions).all(), "Actions must be finite float32 [T,16]")
    require(rgb.shape == (count + 1,) + IMAGE_SHAPE and rgb.dtype == np.uint8,
            "RGB must be uint8 [T+1,224,224,3]")
    require(qpos.shape == (count + 1, 18) and qpos.dtype == np.float32 and np.isfinite(qpos).all(),
            "Physical qpos must be finite float32 [T+1,18]")
    require(timestamps_s.shape == (count + 1,) and timestamps_s.dtype == np.float64
            and np.array_equal(timestamps_s, np.arange(count + 1, dtype=np.float64) / CONTROL_HZ),
            "Timestamps must be exact float64 frame_index/20 seconds")
    return count


def capture_state_observations(raw: Any, states: list[dict[str, Any]],
                               restore: Callable[[Any, dict[str, Any]], dict[str, Any]],
                               write_frame: Callable[[int, np.ndarray, np.ndarray], None]
                               ) -> dict[str, Any]:
    """Restore each source state, render once, and never call env.step."""
    require(len(states) >= 2, "At least two state frames are required")
    starting_steps = as_numpy(raw.elapsed_steps).copy()
    receipts = []
    for index, state in enumerate(states):
        receipt = restore(raw, state)
        image, joints = extract_policy_observation(raw)
        require(np.array_equal(joints, qpos_from_state(state)),
                f"Restored physical qpos differs from source frame {index}")
        require(np.array_equal(as_numpy(raw.elapsed_steps), starting_steps),
                "Observation export must not advance control steps")
        write_frame(index, image, joints)
        receipts.append({"frame_index": index, "qpos_exact": True,
                         "root_pose_precondition_exact": receipt["root_pose_precondition_exact"],
                         "root_pose_postcondition_exact": receipt["root_pose_postcondition_exact"],
                         "hidden_physics_state_restored": receipt["hidden_physics_state_restored"]})
    return {"state_frames": len(states), "actual_control_steps": 0,
            "root_pose_setters": 0, "restoration_receipts": receipts,
            "physical_execution_revalidated": False,
            "note": "State playback supplies policy observations; physical validity remains source evidence."}


def exporter_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[4]
    def git(*args: str) -> str:
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                                text=True, check=True)
        return result.stdout.strip()
    files = [Path(__file__), Path(__file__).with_name("fixed_root_restore.py"),
             Path(__file__).with_name("validate_curobo_complete_replay.py"),
             root / "mani_skill/envs/sapien_env.py",
             root / "mani_skill/envs/tasks/tabletop/dual_tasks/_031_pick_pillbottle_to_regions.py"]
    return {"git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
            "files": {path.relative_to(root).as_posix(): file_sha256(path) for path in files}}


def load_initial_state(source_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    """Read the immutable original frame zero for online policy evaluation."""
    source_dir = Path(source_dir).resolve()
    metadata = json.loads((source_dir / "trajectory.json").read_text())
    require(len(metadata["episodes"]) == 1, "Expected one source episode")
    episode = metadata["episodes"][0]["episode_id"]
    with h5py.File(source_dir / "trajectory.h5", "r") as handle:
        group = handle[f"traj_{episode}/env_states"]
        return {category: {name: leaf[0].copy() for name, leaf in group[category].items()}
                for category in ("actors", "articulations")}


def load_export(input_dir: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Authenticate the derived export before conversion or training."""
    directory = Path(input_dir).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    require(manifest.get("schema") == SCHEMA and manifest.get("status") == "passed",
            "Expected completed dual-arm observation export")
    require(manifest.get("control_hz") == CONTROL_HZ and manifest.get("camera") == CAMERA
            and manifest.get("control_mode") == "pd_joint_pos", "Export policy contract differs")
    path = directory / "observations.h5"
    require(file_sha256(path) == manifest["observations_h5_sha256"], "Observation archive hash differs")
    with h5py.File(path, "r") as handle:
        require(set(handle) == {"rgb", "qpos", "actions", "timestamps_s"}, "Export H5 keys differ")
        arrays = {key: value[:] for key, value in handle.items()}
    count = validate_observation_arrays(**arrays)
    require(count == manifest["action_steps"] and count + 1 == manifest["state_frames"],
            "Export dimensions and manifest disagree")
    return manifest, arrays


def export_observations(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Create a new derived archive; refuse overwrite or writes beneath source."""
    from .validate_curobo_complete_replay import load_recording, restore_initial_fixed_roots
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    require(source_dir != output_dir and source_dir not in output_dir.parents,
            "Derived output must be outside the source recording")
    require(not output_dir.exists(), "Output already exists; choose a new derived directory")
    before = directory_identity(source_dir)
    recording = load_recording(source_dir, replay_mode="cross-version")
    metadata, source_report = recording["metadata"], recording["report"]
    require(recording["task"] == TASK and source_report["seed"] == 401,
            "First bridge is fixed to accepted pill seed401")
    actions = recording["actions"]
    require(actions.shape == (171, 16) and actions.dtype == np.float32,
            "First bridge requires the accepted 171x16 float32 source actions")
    states = recording["states"]
    require(len(states) == 172, "First bridge requires all172 physical state frames")
    output_dir.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "schema": SCHEMA, "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": TASK, "seed": 401, "control_mode": "pd_joint_pos", "control_hz": CONTROL_HZ,
        "action_dim": 16, "state_dim": 18, "camera": CAMERA, "image_shape": list(IMAGE_SHAPE),
        "action_steps": len(actions), "state_frames": len(states), "source_dir": str(source_dir),
        "source_h5_sha256": before["trajectory.h5"]["sha256"],
        "source_report_sha256": before["report.json"]["sha256"],
        "source_metadata_sha256": before["trajectory.json"]["sha256"],
        "source_identity": {"commit_info": metadata.get("commit_info"),
            "archived_runtime_sha256": source_report.get("source_sha256"),
            "admission": recording["receipt"], "source_inventory": before},
        "source_result_reference": {key: source_report.get(key) for key in
            ("status", "official_200_step_complete", "results", "selected_policy_complete")},
        "config": {"agent_order": list(AGENT_KEYS), "articulation_order": list(ARTICULATION_KEYS),
            "state_names": STATE_NAMES, "action_names": ACTION_NAMES,
            "qpos_units": "arm radians; two physical finger positions per arm in metres",
            "action_semantics": "pd_joint_pos: seven arm targets and one original gripper command per arm",
            "rgb_semantics": "native zed2i sensor uint8 RGB; no resize or image augmentation",
            "environment_kwargs_source": metadata["env_info"]["env_kwargs"],
            "terminal_policy": "o_T and q_T retained; no synthetic a_T; training rows t=0..T-1",
            "sampling": "o_t,q_t before original action a_t; timestamp_s=t/20"},
        "version": exporter_identity(), "physical_execution_revalidated": False}
    save_json(output_dir / "manifest.json", manifest)
    env = None
    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
        kwargs = dict(metadata["env_info"]["env_kwargs"])
        kwargs.update(obs_mode="rgb", max_episode_steps=200, hi_res=False, wrist_sensor=False)
        env = gym.make(TASK, **kwargs)
        env.reset(**metadata["episodes"][0]["reset_kwargs"])
        raw = env.unwrapped
        require(not raw.gpu_sim_enabled and raw.num_envs == 1
                and np.isclose(raw.control_timestep, .05, rtol=0, atol=1e-10),
                "Live extraction requires one CPU20Hz environment")
        manifest["config"]["environment_kwargs_export"] = kwargs
        parameters = raw.get_sensor_params()[CAMERA]
        manifest["camera_parameters"] = {key: as_numpy(value).tolist() for key, value in parameters.items()}
        temporary = output_dir / "observations.h5.partial"
        with h5py.File(temporary, "w") as handle:
            handle.attrs["schema"] = SCHEMA
            rgb_ds = handle.create_dataset("rgb", (len(states),) + IMAGE_SHAPE, dtype="uint8",
                                           chunks=(1,) + IMAGE_SHAPE, compression="lzf")
            qpos_ds = handle.create_dataset("qpos", (len(states), 18), dtype="float32")
            handle.create_dataset("actions", data=actions)
            handle.create_dataset("timestamps_s", data=np.arange(len(states), dtype=np.float64) / CONTROL_HZ)
            restoration_errors = []
            def write(index: int, image: np.ndarray, joints: np.ndarray) -> None:
                from .replay_metrics import compare_states, summarize_errors
                from .replay_support import STATE_LIMITS, unbatch_state, within_limits
                errors = compare_states(states[index], unbatch_state(raw.get_state_dict()))
                require(within_limits(summarize_errors([errors])["global"], STATE_LIMITS),
                        f"Source state restoration differs at observation frame {index}")
                restoration_errors.append(errors)
                rgb_ds[index] = image
                qpos_ds[index] = joints
            manifest["extraction"] = capture_state_observations(raw, states, restore_initial_fixed_roots, write)
            from .replay_metrics import summarize_errors
            from .replay_support import STATE_LIMITS
            manifest["extraction"]["restored_state_errors"] = summarize_errors(restoration_errors)
            manifest["extraction"]["restoration_limits"] = dict(STATE_LIMITS)
            validate_observation_arrays(rgb_ds[:], qpos_ds[:], handle["actions"][:], handle["timestamps_s"][:])
        temporary.rename(output_dir / "observations.h5")
        manifest["observations_h5_sha256"] = file_sha256(output_dir / "observations.h5")
        manifest["status"] = "passed"
    except BaseException as error:
        manifest.update(status="failed", error_type=type(error).__name__, error=str(error),
                        traceback=traceback.format_exc())
        raise
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            after = directory_identity(source_dir)
            manifest["source_unchanged"] = before == after
            manifest["source_inventory_after"] = after
            if before != after:
                manifest["status"] = "failed"
                manifest["source_integrity_error"] = "Immutable source changed during extraction"
            save_json(output_dir / "manifest.json", manifest)
            require(before == after, "Immutable source changed during extraction")
    return manifest


def self_test() -> dict[str, Any]:
    """Synthetic data only: no simulator, assets, render or CUDA initialization."""
    from types import SimpleNamespace as NS
    states = []
    for i in range(3):
        states.append({"articulations": {name: np.r_[np.zeros(13, np.float32),
            np.arange(9, dtype=np.float32) + i + side, np.zeros(9, np.float32)]
            for side, name in enumerate(ARTICULATION_KEYS)}})
    raw = NS(elapsed_steps=np.array([0]), state=states[0])
    raw.agent = NS(agents_dict={key: NS(robot=NS(get_qpos=lambda side=side:
        qpos_from_state(raw.state)[9*side:9*(side+1)][None])) for side, key in enumerate(AGENT_KEYS)})
    raw._get_obs_sensor_data = lambda: {CAMERA: {"rgb": np.zeros((1,) + IMAGE_SHAPE, np.uint8)}}
    def restore(target: Any, state: dict[str, Any]) -> dict[str, Any]:
        target.state = state
        return {"root_pose_precondition_exact": True, "root_pose_postcondition_exact": True,
                "hidden_physics_state_restored": False}
    frames = []
    result = capture_state_observations(raw, states, restore, lambda i, im, q: frames.append((im, q)))
    rgb, qpos = np.stack([x[0] for x in frames]), np.stack([x[1] for x in frames])
    actions = np.arange(32, dtype=np.float32).reshape(2, 16)
    times = np.arange(3, dtype=np.float64) / CONTROL_HZ
    assert validate_observation_arrays(rgb, qpos, actions, times) == 2
    assert result["actual_control_steps"] == 0 and np.array_equal(qpos[1], qpos_from_state(states[1]))
    # T=1 boundary keeps two observations and exactly one genuine action.
    assert validate_observation_arrays(rgb[:2], qpos[:2], actions[:1], times[:2]) == 1
    for bad in (times + .001, times.astype(np.float32)):
        try:
            validate_observation_arrays(rgb, qpos, actions, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Accepted incorrect time alignment")
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        (directory / "report.json").write_text('{"status":"passed"}\n')
        identity = directory_identity(directory)
        assert identity == directory_identity(directory)
        (directory / "report.json").write_text('{"status":"failed"}\n')
        assert identity != directory_identity(directory)
    return {"status": "passed", "rgb_shape": list(rgb.shape), "qpos_shape": list(qpos.shape),
            "actions_shape": list(actions.shape), "checks": ["pre-action T/T+1 alignment", "T=1",
            "physical left/right qpos order", "zero control steps", "invalid timestamps rejected",
            "source hash change detected"], "native_render": "not_run"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
    else:
        if not args.source_dir or not args.output_dir:
            parser.error("--source-dir and --output-dir are required")
        report = export_observations(args.source_dir, args.output_dir)
        print(json.dumps({key: report[key] for key in ("status", "action_steps", "state_frames", "source_unchanged")}, indent=2))


if __name__ == "__main__":
    main()
