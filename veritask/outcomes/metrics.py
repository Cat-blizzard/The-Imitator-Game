"""Physical outcome metrics. Pure numpy; inputs are end-state geometry (from a simulator's privileged state or from
real-world perception). Each metric is registered by name so task contracts can refer to it.
"""
from __future__ import annotations

import numpy as np

REGISTRY: dict[str, callable] = {}


def metric(name):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


def _hull_area(xy: np.ndarray) -> float:
    from scipy.spatial import ConvexHull
    return float(ConvexHull(xy).volume)


# ------------------------------------------------------------------------------------------------ cloth
@metric("cloth.corner_distance")
def corner_distance(verts: np.ndarray, a: int, c: int, **_) -> float:
    """Horizontal distance between two cloth vertices (e.g. the two corners a diagonal fold should bring together) [m]."""
    return float(np.linalg.norm(verts[a, :2] - verts[c, :2]))


@metric("cloth.footprint_ratio")
def footprint_ratio(verts: np.ndarray, verts_init: np.ndarray, **_) -> float:
    """Convex-hull area of the cloth's top-down projection, relative to its initial flat state."""
    return _hull_area(verts[:, :2]) / _hull_area(verts_init[:, :2])


@metric("cloth.frac_above")
def frac_above(verts: np.ndarray, height: float = 0.05, ground_z: float = 0.0, **_) -> float:
    """Fraction of cloth vertices higher than `height` above the support surface (proxy for 'still lifted / not released')."""
    return float(np.mean(verts[:, 2] - ground_z > height))


# ------------------------------------------------------------------------------------------------ liquid / granular
@metric("particles.frac_in_cylinder")
def frac_in_cylinder(particles: np.ndarray, center_xy, radius: float, z_bottom: float, z_top: float, **_) -> float:
    """Fraction of particles inside a vertical cylinder (a cup / container approximated by its inner volume)."""
    p = np.asarray(particles)
    r = np.linalg.norm(p[:, :2] - np.asarray(center_xy)[None], axis=1)
    inside = (r <= radius) & (p[:, 2] >= z_bottom) & (p[:, 2] <= z_top)
    return float(inside.mean()) if len(p) else 0.0


@metric("particles.frac_spilled")
def frac_spilled(particles: np.ndarray, source_inside: np.ndarray, target_inside: np.ndarray, **_) -> float:
    """Fraction of particles that are in neither the source nor the target container at the end (spill)."""
    return float(np.mean(~(source_inside | target_inside))) if len(particles) else 0.0


# ------------------------------------------------------------------------------------------------ rigid
@metric("rigid.distance_to_goal")
def distance_to_goal(pos: np.ndarray, goal: np.ndarray, **_) -> float:
    return float(np.linalg.norm(np.asarray(pos) - np.asarray(goal)))


@metric("rigid.tilt_deg")
def tilt_deg(up_axis_world: np.ndarray, **_) -> float:
    """Angle between an object's up axis (unit vector in world frame) and world +z [deg]."""
    u = np.asarray(up_axis_world, float); u = u / np.linalg.norm(u)
    return float(np.degrees(np.arccos(np.clip(u[2], -1.0, 1.0))))


def measure(name: str, **kwargs) -> float:
    return REGISTRY[name](**kwargs)


# ------------------------------------------------------------------------------------------------ traces
# Event-type tasks (press, poke, pour) need conditions over the episode, not only the final state. A trace is a
# per-step array recorded by an adapter (see adapters/maniskill.py). These metrics turn traces into numbers that
# contract terms can threshold. 'final' uses the MEAN of the last `window` steps; prefer 'window_max' / 'window_min'
# (every step of the window must satisfy the bound) for end-state conditions: a mean can hide a late violation.
@metric("trace.final")
def trace_final(traces: dict, signal: str, window: int = 10, **_) -> float:
    x = np.asarray(traces[signal], float); return float(x[-window:].mean())


@metric("trace.last")
def trace_last(traces: dict, signal: str, **_) -> float:
    """Value at the last recorded step."""
    return float(np.asarray(traces[signal], float)[-1])


@metric("trace.window_max")
def trace_window_max(traces: dict, signal: str, window: int = 10, **_) -> float:
    """Worst (largest) value over the last `window` steps. Use with '<' / '<=' to require that EVERY step of the
    settled window satisfies the bound. Unlike trace.final (a mean), a transient violation cannot hide in it
    (E10 red team: a cube pushed out of the goal in the last 3 steps passed a mean-based window)."""
    return float(np.asarray(traces[signal], float)[-window:].max())


@metric("trace.window_min")
def trace_window_min(traces: dict, signal: str, window: int = 10, **_) -> float:
    """Worst (smallest) value over the last `window` steps. Use with '>' / '>=' (e.g. on-tray flags must be 1 at
    every step of the window)."""
    return float(np.asarray(traces[signal], float)[-window:].min())


@metric("trace.min")
def trace_min(traces: dict, signal: str, **_) -> float:
    return float(np.min(traces[signal]))


@metric("trace.max")
def trace_max(traces: dict, signal: str, **_) -> float:
    return float(np.max(traces[signal]))


@metric("trace.any")
def trace_any(traces: dict, signal: str, **_) -> float:
    return float(np.any(np.asarray(traces[signal], bool)))


@metric("trace.sum")
def trace_sum(traces: dict, signal: str, **_) -> float:
    """Sum over the episode (e.g. total accumulated contact impulse)."""
    return float(np.sum(np.asarray(traces[signal], float)))


@metric("trace.count_true")
def trace_count_true(traces: dict, signal: str, **_) -> float:
    return float(np.sum(np.asarray(traces[signal], bool)))


@metric("trace.window_disp")
def trace_window_disp(traces: dict, signal: str, window: int = 10, signals: list | None = None, **_) -> float:
    """Net displacement over the last `window` steps: largest distance of any step's point (the given coordinate
    signals, e.g. [obj_x, obj_y, obj_z]; default [signal]) from the window's first step. Use with '<=' to require that an
    object stays put during the judged window (a slow creep under a speed bound shows up here; solver jitter does not)."""
    X = np.stack([np.asarray(traces[s], float)[-window:] for s in (signals or [signal])], axis=1)
    return float(np.max(np.linalg.norm(X - X[0], axis=1))) if len(X) else 0.0


# ------------------------------------------------------------------------------------------------ topology (D2, appended)
# Topological primitives (winding number, disk crossings + insertion margin, wrap angle) and their geometry and trace
# metrics live in outcomes/topology.py. Importing it here registers them whenever this registry is used (contract
# validation, protocol v3). Nothing above is changed.
from . import topology as _topology  # noqa: E402,F401
