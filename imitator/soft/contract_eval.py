"""Contract evaluation for the soft envs through veritask (protocol v3 traces + the original v1 end-state contract).

TraceRecorder collects one row of per-world signals per control step (policy phase 0, evaluator settle phase 1), adds
towel_footprint_ratio with veritask's own metric (scipy ConvexHull), and keeps the vertex positions needed for v1.
evaluate() returns, per world:
  v3  verdict of imsim/soft/contracts/towel_fold_diagonal_v3.json via veritask.adapters.base.evaluate_traces
      (window located by protocol_v3.find_window from the traces alone)
  v1  verdict of veritask/tasks/towel_fold_diagonal.json (cloth.* terms on the vertices settle_seconds after the policy
      ended, released = majority of the last 10 'released' flags, as adapters/base.py does for v1)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import veritask  # noqa: E402

VERITASK = os.path.dirname(os.path.abspath(veritask.__file__))

from veritask.adapters.base import evaluate_traces, released_from_trace, save_traces  # noqa: E402
from veritask.outcomes.metrics import corner_distance, footprint_ratio, frac_above  # noqa: E402
from veritask.spec.contract import TaskContract  # noqa: E402

V3_PATH = os.path.join(HERE, "contracts", "towel_fold_diagonal_v3.json")
V1_PATH = os.path.join(VERITASK, "tasks", "towel_fold_diagonal.json")
TRACE_SIGNALS = ("towel_corner_dist", "towel_footprint_ratio", "towel_frac_above_5cm", "robot_cloth_impulse",
                 "robot_cloth_fmax", "finger_cloth_enclosure", "cloth_n_enclosed", "cloth_speed_p95", "cloth_speed_max",
                 "cloth_zmin", "cloth_nonfinite", "pad_penetration", "robot_qvel_max", "released", "tcp_x", "tcp_y",
                 "tcp_z", "finger_q", "towel_A_x", "towel_A_y", "towel_A_z", "towel_C_x", "towel_C_y", "towel_C_z")


def load_contracts():
    return TaskContract.load(V3_PATH), TaskContract.load(V1_PATH)


class TraceRecorder:
    def __init__(self, num_worlds: int, control_freq: float = 20.0, keep_frames_every: int = 0):
        self.n = num_worlds
        self.control_freq = control_freq
        self.rows: list[dict] = []
        self.verts: list[np.ndarray] = []          # every step (float16 would lose mm; keep float32, small towels)
        self.verts_init = None
        self.keep_every = keep_frames_every
        self.frames: list[np.ndarray] = []

    def set_init(self, verts_init):
        self.verts_init = np.asarray(verts_init, np.float64)

    def record(self, signals: dict, verts, phase: int):
        v = verts.detach().cpu().numpy() if hasattr(verts, "detach") else np.asarray(verts)
        row = {k: (signals[k].detach().cpu().numpy() if hasattr(signals[k], "detach") else np.asarray(signals[k])).astype(
            np.float64).reshape(-1) for k in TRACE_SIGNALS if k in signals}
        row["towel_footprint_ratio"] = np.array([self._foot(v[w], w) for w in range(self.n)])
        row["phase"] = np.full(self.n, float(phase))
        self.rows.append(row)
        self.verts.append(v.astype(np.float32))

    def _foot(self, x, w):
        if not np.isfinite(x).all():
            return np.inf
        try:
            return footprint_ratio(x.astype(np.float64), self.verts_init[w])
        except Exception:
            return np.inf

    def traces(self, w: int) -> dict:
        keys = self.rows[0].keys()
        return {k: np.array([r[k][w] for r in self.rows]) for k in keys}

    def evaluate(self, contract_v3=None, contract_v1=None) -> list[dict]:
        c3, c1 = (contract_v3, contract_v1) if contract_v3 is not None else load_contracts()
        out = []
        ph = np.array([r["phase"][0] for r in self.rows])
        n_policy = int((ph < 0.5).sum())
        k_v1 = min(len(self.rows) - 1, n_policy - 1 + int(round(c1.settle_seconds * self.control_freq)))
        for w in range(self.n):
            tr = self.traces(w)
            v3, bench = evaluate_traces(c3, tr, control_freq=self.control_freq)
            info = bench.get("protocol_info", {})
            x = self.verts[k_v1][w].astype(np.float64)
            finite = bool(np.isfinite(x).all())
            m = {}
            if finite:
                m = {"cloth.corner_distance": corner_distance(x, 0, len(x) - 1),
                     "cloth.footprint_ratio": footprint_ratio(x, self.verts_init[w]),
                     "cloth.frac_above": frac_above(x, 0.05)}
            rel = released_from_trace(tr["released"][: k_v1 + 1], 10)
            v1 = c1.verdict(m, released=rel) if finite else dict(success=False, terms=[], released=rel)
            out.append(dict(world=w, v3_success=v3["success"], v3_binding=v3.get("binding_term"),
                            v3_terms={t["metric"]: t["value"] for t in v3["terms"]}, v3_released=v3.get("released"),
                            v3_window=info.get("window"), v3_window_status=info.get("window_status"),
                            v1_success=v1["success"], v1_terms={t["metric"]: t["value"] for t in v1.get("terms", [])},
                            v1_released=rel, finite_all=bool(all(np.isfinite(vv[w]).all() for vv in self.verts))))
        return out

    def end_verts(self, w: int, step: int = -1) -> np.ndarray:
        return self.verts[step][w]

    def robustness(self) -> list[dict]:
        """Per world: NaN, explosion (speed > 5 m/s or a vertex > 1 m from the towel), penetration statistics."""
        res = []
        for w in range(self.n):
            tr = self.traces(w)
            V = np.stack([v[w] for v in self.verts])                # (T, V, 3)
            finite = bool(np.isfinite(V).all())
            far = float(np.nanmax(np.abs(V[..., :2] - self.verts_init[w][None, :, :2]))) if finite else np.inf
            res.append(dict(world=w, nonfinite=not finite, max_speed=float(np.nanmax(tr["cloth_speed_max"])),
                            explosion=bool((not finite) or np.nanmax(tr["cloth_speed_max"]) > 5.0 or far > 1.0),
                            min_particle_z_mm=1000 * float(np.nanmin(tr["cloth_zmin"])),
                            below_table=bool(np.nanmin(tr["cloth_zmin"]) < 0.0),
                            max_pad_penetration_mm=1000 * float(np.nanmax(tr["pad_penetration"])),
                            max_fmax_N=float(np.nanmax(tr["robot_cloth_fmax"]))))
        return res

    def save(self, path: str, w: int, meta: dict):
        save_traces(path, self.traces(w), meta)
