"""Optional CPU call profiling of cuRobo configuration construction.

TIG_CUROBO_PROFILE_CONFIG=1 profiles only MotionPlannerCfg.create. It does not
change arguments, reuse native objects, synchronize CUDA, or record arguments.
Its wall times contain profiling overhead and are diagnostic, not a performance
baseline. GPU kernel attribution requires a separate GPU profiler.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import os
from pathlib import Path
import time
from typing import Any, TypeVar


_T = TypeVar("_T")
_ENVIRONMENT_KEY = "TIG_CUROBO_PROFILE_CONFIG"
_TRACKED_ENTRIES = {
    "robot_cfg_create": ("curobo/_src/types/robot.py", "create"),
    "kinematics_cfg_build": ("curobo/_src/robot/kinematics/kinematics_cfg.py", "from_config"),
    "urdf_parser_construct": ("curobo/_src/robot/parser/parser_urdf.py", "__init__"),
    "urdf_file_load": ("yourdfpy/urdf.py", "load"),
    "scene_cfg_create": ("curobo/_src/geom/types.py", "create"),
    "ik_cfg_create": ("curobo/_src/solver/solver_ik_cfg.py", "create"),
    "trajopt_cfg_create": ("curobo/_src/solver/solver_trajopt_cfg.py", "create"),
    "rrt_cfg_create": ("curobo/_src/graph_planner/rrt/config_rrt_graph_planner.py", "create"),
    "dynamics_construct": ("curobo/_src/robot/dynamics/dynamics.py", "__init__"),
    "timeopt_construct": ("curobo/_src/motion/motion_trajectory_timeopt.py", "__init__"),
}


def config_profiling_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Enable profiling only for the explicit diagnostic environment value 1."""
    return (os.environ if environ is None else environ).get(_ENVIRONMENT_KEY) == "1"


def _display_filename(filename: str, roots: tuple[tuple[str, Path], ...]) -> str:
    """Retain package/repository locations without exposing absolute host paths."""
    normalized = filename.replace("\\", "/")
    for package in ("curobo", "torch", "warp", "yourdfpy", "numpy", "sympy"):
        marker = f"/{package}/"
        if marker in normalized:
            return f"{package}/{normalized.split(marker, 1)[1]}"
    path = Path(filename)
    for label, root in roots:
        try:
            return f"{label}/{path.relative_to(root).as_posix()}"
        except ValueError:
            pass
    if path.is_absolute():
        return f"external/{path.name}"
    return normalized


def summarize_config_profile(
    statistics: Mapping[tuple[str, int, str], tuple[Any, ...]],
    *, roots: tuple[tuple[str, Path], ...] = (), top_n: int = 20,
) -> dict[str, Any]:
    """Summarize pstats rows (primitive calls, total calls, self s, cumulative s).

    Nested cumulative times overlap. Tracked entries show repeated factory
    invocations separately from expensive model/URDF builds: RobotCfg.create
    can return an existing typed object without rebuilding it.
    """
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer")
    rows = []
    tracked: dict[str, list[dict[str, Any]]] = {name: [] for name in _TRACKED_ENTRIES}
    for (filename, line, function), values in statistics.items():
        primitive, total, self_s, cumulative_s = values[:4]
        displayed = _display_filename(filename, roots)
        row = {"file": displayed, "line": line, "function": function,
               "primitive_calls": primitive, "total_calls": total,
               "self_s": self_s, "cumulative_s": cumulative_s}
        rows.append(row)
        for label, (suffix, expected_function) in _TRACKED_ENTRIES.items():
            if displayed.endswith(suffix) and function == expected_function:
                tracked[label].append(row)
    order = lambda row: (-row["cumulative_s"], row["file"], row["line"], row["function"])
    return {
        "function_count": len(rows),
        "primitive_calls": sum(row["primitive_calls"] for row in rows),
        "total_calls": sum(row["total_calls"] for row in rows),
        "total_self_s": sum(row["self_s"] for row in rows),
        "top_cumulative": sorted(rows, key=order)[:top_n],
        "top_self": sorted(rows, key=lambda row: (-row["self_s"], *order(row)[1:]))[:top_n],
        "tracked_calls": {
            label: {"total_calls": sum(row["total_calls"] for row in entries),
                    "entries": sorted(entries, key=order)}
            for label, entries in tracked.items()
        },
        "interpretation": (
            "CPU/Python call attribution; includes native waits but not GPU kernel attribution. "
            "Cumulative times overlap and must not be summed. Factory call counts do not "
            "imply repeated model parsing; compare kinematics and URDF build entries."
        ),
    }


def create_profiled_config(
    create: Callable[..., _T], kwargs: dict[str, Any], *,
    source_root: Path, code_root: Path, environ: Mapping[str, str] | None = None,
) -> tuple[_T, dict[str, Any]]:
    """Call the factory once with unchanged kwargs, optionally returning a profile.

    cfg_create_s is elapsed host wall time around the factory, without a new
    CUDA synchronization. The worker's existing synchronized initialization_s
    remains the overall boundary. Profile summary generation is timed separately.
    No profile instrumentation overhead is subtracted or guessed.
    """
    enabled = config_profiling_enabled(environ)
    if not enabled:
        begin = time.perf_counter()
        result = create(**kwargs)
        return result, {"cfg_create_s": time.perf_counter() - begin,
                        "config_profile_summary_s": 0.0,
                        "config_profile": {"enabled": False}}
    import cProfile
    import pstats

    profiler = cProfile.Profile()
    begin = time.perf_counter()
    result = profiler.runcall(create, **kwargs)
    wall_s = time.perf_counter() - begin
    summary_begin = time.perf_counter()
    summary = summarize_config_profile(
        pstats.Stats(profiler).stats,
        roots=(("curobo-source", source_root), ("checkout", code_root)),
    )
    summary_s = time.perf_counter() - summary_begin
    summary.update(enabled=True, profiled_call_wall_s=wall_s,
                   instrumentation_overhead_s=None,
                   performance_baseline_valid=False,
                   timing_note="cfg_create_s includes cProfile overhead; overhead is not independently measured")
    return result, {"cfg_create_s": wall_s,
                    "config_profile_summary_s": summary_s, "config_profile": summary}


def self_test() -> None:
    """Check diagnostic opt-in, one-call behavior, attribution and path redaction."""
    assert not config_profiling_enabled({})
    assert not config_profiling_enabled({_ENVIRONMENT_KEY: "0"})
    assert not config_profiling_enabled({_ENVIRONMENT_KEY: "true"})
    assert config_profiling_enabled({_ENVIRONMENT_KEY: "1"})
    rows = {
        ("/private/host/curobo/_src/types/robot.py", 59, "create"): (4, 4, 0.01, 0.5, {}),
        ("/private/host/curobo/_src/robot/parser/parser_urdf.py", 31, "__init__"): (1, 1, 0.01, 0.3, {}),
        ("/private/host/site-packages/yourdfpy/urdf.py", 8, "load"): (1, 1, 0.02, 0.2, {}),
        ("/private/host/application/other.py", 3, "build"): (1, 1, 0.3, 0.3, {}),
    }
    summary = summarize_config_profile(rows, top_n=2)
    assert summary["tracked_calls"]["robot_cfg_create"]["total_calls"] == 4
    assert summary["tracked_calls"]["urdf_parser_construct"]["total_calls"] == 1
    assert summary["tracked_calls"]["urdf_file_load"]["total_calls"] == 1
    assert summary["tracked_calls"]["rrt_cfg_create"]["total_calls"] == 0
    assert summary["top_self"][0]["file"] == "external/other.py"
    assert summary["top_cumulative"][0]["function"] == "create"
    assert "/private/host" not in json.dumps(summary)
    assert _display_filename("/checkout/infrastructure/worker.py", (("checkout", Path("/checkout")),)) == "checkout/infrastructure/worker.py"
    calls: list[int] = []
    token = object()
    def create(*, value: object) -> object:
        calls.append(len(calls))
        return value
    for value in ("0", "1"):
        returned, timings = create_profiled_config(
            create, {"value": token}, source_root=Path("/source"), code_root=Path("/checkout"),
            environ={_ENVIRONMENT_KEY: value},
        )
        assert returned is token
        assert timings["cfg_create_s"] >= 0.0
        assert timings["config_profile"]["enabled"] == (value == "1")
        json.dumps(timings, allow_nan=False)
    assert len(calls) == 2
    assert timings["config_profile"]["instrumentation_overhead_s"] is None
    assert not timings["config_profile"]["performance_baseline_valid"]
    try:
        create_profiled_config(lambda: (_ for _ in ()).throw(ValueError("factory failure")), {},
                              source_root=Path("/source"), code_root=Path("/checkout"),
                              environ={_ENVIRONMENT_KEY: "1"})
    except ValueError as error:
        assert str(error) == "factory failure"
    else:
        raise AssertionError("profiling must preserve factory errors")
    print(json.dumps({"self_test": "PASS", "config_profile_opt_in": True,
                      "one_call_per_factory": True, "paths_redacted": True}))


if __name__ == "__main__":
    self_test()
