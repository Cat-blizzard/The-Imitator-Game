"""Bounded controlled diagnostics of initial-state setters and replay observers.

Contract: one seed-0 reset, original 323 paired float32 actions at 20 Hz, 400
engineering ticks, no planning/controller reset/corrections. The seven leaves
are actor13 or Panda31; outputs contain T+1 states, actual command copies and
raw native flags. Existing strict ACTION_LIMITS are unchanged. These are
single-episode diagnostics, not acceptance repairs or general determinism tests.
"""
from __future__ import annotations

import argparse
import ast
import builtins
from datetime import datetime, timezone
import faulthandler
import hashlib
import inspect
import json
import os
from pathlib import Path
import resource
import symtable
import textwrap
import time
import traceback
from typing import Any, Callable

import numpy as np

from . import validate_sync_replay as original
from .replay_metrics import compare_states, pose_errors, summarize_errors
from .runtime_io import json_value, save_report
from .replay_support import ACTION_LIMITS, split_action, within_limits

HERE = Path(__file__).resolve().parent
ROOT = original.ROOT
INPUT_NAMES = ("trajectory.h5", "trajectory.json", "ticks.jsonl", "report.json", "physics-substeps.jsonl", "execution.mp4")
RESTORE_MODES = ("full", "none", "actors-only", "articulations-only", "articulations-without-root-pose")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_comparison(reference: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    """Report raw dtype/shape/byte identity, first scalar/byte differences."""
    if set(reference) != set(actual) or any(set(reference[c]) != set(actual[c]) for c in reference):
        raise ValueError("State schemas differ")
    leaves = {}
    for category in reference:
        for name, expected in reference[category].items():
            expected, observed = np.asarray(expected), np.asarray(actual[category][name])
            schema = expected.shape == observed.shape and expected.dtype == observed.dtype
            identical = schema and expected.tobytes() == observed.tobytes()
            entry: dict[str, Any] = {"exact_bytes": identical, "shape": list(observed.shape), "dtype": str(observed.dtype)}
            if schema and not identical:
                scalar = np.flatnonzero(expected.reshape(-1) != observed.reshape(-1))
                byte = np.flatnonzero(np.frombuffer(expected.tobytes(), dtype=np.uint8) != np.frombuffer(observed.tobytes(), dtype=np.uint8))
                entry.update(first_different_byte=int(byte[0]), first_different_scalar=int(scalar[0]) if len(scalar) else None)
                if len(scalar):
                    j = scalar[0]
                    entry.update(reference_scalar=float(expected.reshape(-1)[j]), actual_scalar=float(observed.reshape(-1)[j]))
            leaves[f"{category}/{name}"] = entry
    return {"all_exact_bytes": all(x["exact_bytes"] for x in leaves.values()), "per_entity": leaves}


def observer_ast() -> ast.FunctionDef:
    tree = ast.parse(textwrap.dedent(inspect.getsource(original.native_run)))
    found = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "observe"]
    if len(found) != 1:
        raise ValueError("Expected one original nested observe()")
    return found[0]


def full_observer(context: dict[str, Any]) -> Callable[[int], None]:
    """Execute the unchanged original observe AST with its original closure data.

    This reuses all original get_state/evaluate/grasp/contact/geometry/render
    calls and ordering. Only outer orchestration and artifact export differ.
    """
    node = observer_ast()
    symbols = symtable.symtable(ast.unparse(node), "original-observe", "exec").get_children()[0].get_symbols()
    required = {symbol.get_name() for symbol in symbols if symbol.is_global() and symbol.is_referenced() and not hasattr(builtins, symbol.get_name())}
    missing = required - set(context)
    if missing:
        raise ValueError(f"Original observer missing globals: {sorted(missing)}")
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(HERE / "validate_sync_replay.py"), "exec"), context)
    return context["observe"]


def apply_restore(raw: Any, state: dict[str, Any], mode: str) -> list[str]:
    """Perform selected original setters once; no controller state is touched."""
    batched = original.batch_state(state)
    if mode == "none":
        return []
    if mode == "full":
        raw.set_state_dict(batched)
        return ["set_state_dict(actors+articulations)"]
    if mode in ("actors-only", "articulations-only"):
        category = "actors" if mode == "actors-only" else "articulations"
        raw.set_state_dict({category: batched[category]})
        return [f"set_state_dict({category})"]
    if mode != "articulations-without-root-pose":
        raise ValueError("Unknown restore intervention")
    events = []
    # Exact CPU Articulation.set_state field sequence, omitting only root pose.
    for name, state_row in state["articulations"].items():
        articulation = raw.scene.state_dict_registry.articulations[name]
        if articulation.max_dof != 9:
            raise ValueError("Expected Panda9")
        articulation.set_root_linear_velocity(state_row[7:10].copy())
        articulation.set_root_angular_velocity(state_row[10:13].copy())
        articulation.set_qpos(state_row[13:22].copy())
        articulation.set_qvel(state_row[22:31].copy())
        events.append(f"{name}:root_linear_velocity,root_angular_velocity,qpos,qvel")
    return events


def run(args: argparse.Namespace, recording: dict[str, Any], output: Path, report: dict[str, Any]) -> None:
    import gymnasium as gym
    import h5py
    import imageio.v2 as imageio
    import mani_skill.envs
    from .trajectory_execution import as_numpy
    from .sync_replay_metrics import action_outcome
    from mani_skill.envs.tasks.tabletop.dual_tasks._008_pick_apple_banana_baskets import world_aabb_from_local_mesh

    env = raw = None
    rows: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    signals: list[dict[str, Any]] = []
    applied: list[np.ndarray] = []
    exact: list[dict[str, Any]] = []
    try:
        metadata = recording["metadata"]
        kwargs = dict(metadata["env_info"]["env_kwargs"])
        kwargs["max_episode_steps"] = metadata["env_info"]["max_episode_steps"]
        env = gym.make(metadata["env_info"]["env_id"], **kwargs)
        env.reset(**metadata["episodes"][0]["reset_kwargs"])
        raw = env.unwrapped
        agents = (raw.left_agent, raw.right_agent)
        if list(raw.agent.agents_dict) != recording["action_keys"] or any(raw.agent.agents_dict[k] is not a for k, a in zip(recording["action_keys"], agents)):
            raise ValueError("Physical action-key mismatch")
        if raw.backend.sim_backend != "physx_cpu" or not np.isclose(float(raw.control_timestep), .05, atol=1e-10, rtol=0):
            raise ValueError("Expected original CPU20Hz")
        for agent in agents:
            if agent.control_mode != "pd_joint_pos":
                raise ValueError("Expected absolute joint targets")
            for controller in agent.controller.controllers.values():
                if any(bool(getattr(controller.config, k)) for k in ("use_delta", "use_target", "interpolate")):
                    raise ValueError("Unrecorded cumulative controller state")
        report["fresh_reset_initial"] = exact_comparison(recording["states"][0], original.unbatch_state(raw.get_state_dict()))
        if not report["fresh_reset_initial"]["all_exact_bytes"]:
            raise ValueError("Fresh seed0 initial state must equal source bytewise")
        report["setter_events"] = apply_restore(raw, recording["states"][0], args.restore)
        report["post_intervention_initial"] = exact_comparison(recording["states"][0], original.unbatch_state(raw.get_state_dict()))
        if args.observers == "full":
            context = dict(np=np, Any=Any, raw=raw, env=env, recording=recording, args=argparse.Namespace(mode="action"),
                           agents=agents, targets=(raw.apple_out, raw.banana_out), baskets=(raw.basket_apple, raw.basket_banana),
                           ACTORS=original.ACTORS, rows=rows, frames=frames, measured_states=states, errors=errors,
                           signals=signals, previous=[None, None], anchors=[None, None], as_numpy=as_numpy,
                           compare_states=compare_states, pose_errors=pose_errors, json_value=json_value,
                           unbatch_state=original.unbatch_state,
                           physical_region=original.physical_region, world_aabb_from_local_mesh=world_aabb_from_local_mesh)
            observe = full_observer(context)
        else:
            def observe(index: int) -> None:
                state = original.unbatch_state(raw.get_state_dict())
                states.append(state)
                error = compare_states(recording["states"][index], state)
                errors.append(error)
                rows.append({"frame": index, "environment_step": int(as_numpy(raw.elapsed_steps)[0]), "errors": error})
        observe(0)
        exact.append(exact_comparison(recording["states"][0], states[0]))
        for index, source_action in enumerate(recording["actions"], 1):
            command = split_action(source_action, recording["action_keys"])
            command_copy = np.concatenate([command[k] for k in recording["action_keys"]]).copy()
            before = int(as_numpy(raw.elapsed_steps)[0])
            _, _, terminated, truncated, info = env.step(command)
            applied.append(command_copy)
            signal = original.replay_signal(terminated, truncated, info)
            signals.append(signal)
            observe(index)
            exact.append(exact_comparison(recording["states"][index], states[-1]))
            if int(as_numpy(raw.elapsed_steps)[0]) != before + 1 or signal["unexpected_end"]:
                raise RuntimeError("Unexpected step count or native end; triggering state retained")
        summary = summarize_errors(errors)
        violations = [{"frame": i, "entity": entity, "metric": metric, "value": v[metric], "limit": limit}
                      for i, error in enumerate(errors) for entity, v in error["per_entity"].items()
                      for metric, limit in ACTION_LIMITS.items() if metric in v and v[metric] > limit]
        first_by_entity = {}
        for i, comparison in enumerate(exact):
            for entity, value in comparison["per_entity"].items():
                if not value["exact_bytes"] and entity not in first_by_entity:
                    first_by_entity[entity] = {"frame": i, **value}
        report.update(status="completed", trajectory_errors=summary,
                      strict_all7_gate_passed=within_limits(summary["global"], ACTION_LIMITS), violations=violations,
                      all7_states_exact_bytes=all(x["all_exact_bytes"] for x in exact),
                      first_difference_by_entity=first_by_entity,
                      actions_applied_exactly=np.array_equal(np.asarray(applied), recording["actions"]),
                      native_flags_match_source=all(np.array_equal([s[k] for s in signals], recording["native_flags"][k]) for k in ("success", "terminated", "truncated")),
                      native_success_ticks=sum(s["success"] for s in signals),
                      control_timestep_s=float(raw.control_timestep), physical_environment_steps=int(as_numpy(raw.elapsed_steps)[0]))
        if args.observers == "full":
            report["physical_outcome"] = action_outcome(rows)
    finally:
        if env is not None:
            env.close()
        report.update(observed_state_frames=len(states), applied_action_steps=len(applied), rendered_frames=len(frames))
        for index, row in enumerate(rows):
            if index < len(exact):
                row["exact_comparison"] = exact[index]
        with (output / "frames.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(json_value(row), allow_nan=False) + "\n")
        with h5py.File(output / "observed-states.h5", "w") as h:
            h.attrs["diagnostic_only"] = True
            h.create_dataset("applied_actions", data=np.asarray(applied, dtype=np.float32).reshape(-1, 16))
            for k in ("success", "terminated", "truncated", "fail"):
                h.create_dataset(f"native_flags/{k}", data=np.asarray([s[k] for s in signals], dtype=bool))
            if states:
                for category in states[0]:
                    for name in states[0][category]:
                        h.create_dataset(f"env_states/{category}/{name}", data=np.stack([s[category][name] for s in states]))
        if frames:
            imageio.mimsave(output / "observation.mp4", frames, fps=20, macro_block_size=1)


def self_test() -> dict[str, Any]:
    state = {"actors": {"a": np.arange(13, dtype=np.float32)}}
    assert exact_comparison(state, state)["all_exact_bytes"]
    different = {"actors": {"a": state["actors"]["a"].copy()}}
    different["actors"]["a"][2] += 1
    result = exact_comparison(state, different)
    assert not result["all_exact_bytes"] and result["per_entity"]["actors/a"]["first_different_scalar"] == 2
    assert not exact_comparison(state, {"actors": {"a": np.arange(13, dtype=np.float64)}})["all_exact_bytes"]
    class FakeRaw:
        def __init__(self) -> None:
            self.calls: list[Any] = []
        def set_state_dict(self, value: Any) -> None:
            self.calls.append(value)
    fake = FakeRaw()
    assert apply_restore(fake, state, "none") == [] and not fake.calls
    apply_restore(fake, state, "actors-only")
    assert len(fake.calls) == 1 and set(fake.calls[0]) == {"actors"} and fake.calls[0]["actors"]["a"].shape == (1, 13)
    symbols = symtable.symtable(ast.unparse(observer_ast()), "selftest-observe", "exec").get_children()[0].get_symbols()
    required = {s.get_name() for s in symbols if s.is_global() and s.is_referenced() and not hasattr(builtins, s.get_name())}
    observer = full_observer({key: object() for key in required})
    assert callable(observer) and isinstance(observer_ast(), ast.FunctionDef)
    try:
        full_observer({})
    except ValueError as exc:
        assert "unbatch_state" in str(exc)
    else:
        raise AssertionError("Missing observer globals must reject before native execution")
    assert within_limits({k: 0.0 for k in ACTION_LIMITS}, ACTION_LIMITS)
    assert not within_limits({k: v + .001 for k, v in ACTION_LIMITS.items()}, ACTION_LIMITS)
    return {"status": "passed", "checks": ["dtype/shape/byte identity and first scalar", "restore selection and batching", "unchanged original observer compiles", "original limit boundary"], "synthetic_actor_shape": [13]}


def main() -> int:
    faulthandler.enable()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--restore", choices=RESTORE_MODES, default="full")
    p.add_argument("--observers", choices=("minimal", "full"), default="minimal")
    p.add_argument("--input-dir", type=Path, default=original.DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path)
    args = p.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return 0
    from .sync_replay_metrics import load_sync_recording
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "GPU-1f8a72b4-0821-9275-e36f-49ab74f13572":
        p.error("This bounded diagnostic is assigned GPU0 UUID only")
    recording = load_sync_recording(args.input_dir)
    if len(recording["actions"]) != 323 or recording["metadata"]["episodes"][0]["reset_kwargs"]["seed"] != 0:
        raise ValueError("Expected frozen seed0 323-action source")
    hashes = {n: digest(args.input_dir / n) for n in INPUT_NAMES}
    output = args.output_dir or ROOT / "infrastructure/validation/dual-arm" / datetime.now(timezone.utc).strftime(f"restore-diagnostic-{args.restore}-{args.observers}-%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    observer_source = ast.unparse(observer_ast()).encode()
    paths = [Path(__file__).resolve(), HERE / "validate_sync_replay.py", HERE / "validate_replay.py", HERE / "replay_metrics.py", HERE / "sync_replay_metrics.py"]
    paths += [HERE / n for n in recording["source_sha256"] if HERE / n not in paths]
    snapshot_hashes = {}
    for path in paths:
        relative = str(path.relative_to(HERE.parents[3]))
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        snapshot_hashes[relative] = digest(path)
    for name, sha in recording["source_sha256"].items():
        if digest(HERE / name) != sha:
            raise ValueError(f"Frozen source changed: {name}")
    report: dict[str, Any] = {"status": "running", "kind": "controlled diagnostic only", "restore_intervention": args.restore,
        "observers": args.observers, "input_dir": str(args.input_dir), "input_sha256": hashes,
        "source_sha256": snapshot_hashes, "full_observer_ast_sha256": hashlib.sha256(observer_source).hexdigest(),
        "full_observer_reuses_original_ast": True, "limits": ACTION_LIMITS, "max_episode_steps": 400,
        "official_default_max_episode_steps": 200, "controller_reset_after_restore": False,
        "warmup_steps": 0, "replanning_performed": False, "later_state_correction": False,
        "collision_revalidated": False, "contact_internal_state_restored": False,
        "static_scene_reconstructed_from_seed": True, "started_at": datetime.now(timezone.utc).isoformat()}
    save_report(output / "report.json", report)
    started = time.monotonic()
    try:
        run(args, recording, output, report)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    report["input_unchanged"] = all(digest(args.input_dir / n) == sha for n, sha in hashes.items())
    report["execution_sources_unchanged"] = all(digest(HERE / n) == sha for n, sha in recording["source_sha256"].items())
    report.update(completed_at=datetime.now(timezone.utc).isoformat(), wall_seconds=time.monotonic() - started)
    save_report(output / "report.json", report)
    print(json.dumps({k: report.get(k) for k in ("status", "strict_all7_gate_passed", "all7_states_exact_bytes", "observed_state_frames", "applied_action_steps", "input_unchanged")}))
    print(str(output / "report.json"))
    return 0 if report["status"] == "completed" and report["input_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
