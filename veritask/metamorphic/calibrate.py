"""M5 nudge calibration (card B1: "先标定：静止在桌面的物体位移应小于 1 cm").

For a benchmark, objects resting on the table (normally the task objects in the episodes' initial states) are given
a horizontal velocity change dv in each of n_dirs directions; the robot then holds still for nudge_wait_s and the
object's displacement is measured.  The calibrated dv is the LARGEST grid value for which the MEDIAN displacement
over all (object, direction) samples is below nudge_calib_max_disp_m (1 cm).  Every sample is returned so the
calibration can be audited (objects that roll or tip are visible as outliers).
"""
from __future__ import annotations

import math

import numpy as np

from .protocol import Protocol

GRID = (0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)


def nudge_samples(ad, ep, ro, t: int, names: list, grid=GRID, n_dirs: int = 8, proto: Protocol | None = None,
                  stop_disp: float | None = None) -> list:
    """Displacement samples from the state after step t of the episode (t=-1: initial state).
    stop_disp: once an object moved more than stop_disp in EVERY direction at some dv, the larger grid values are
    recorded as censored (disp_m None, censored True) without simulating; choose_dv treats them as +inf, which
    cannot change its choice (the median at a larger dv can only be larger)."""
    proto = proto or Protocol()
    out = []
    n_wait = proto.steps(proto.nudge_wait_s, ad.control_freq)
    for name in names:
        stopped = False
        for dv in grid:
            if stopped:
                for k in range(n_dirs):
                    out.append(dict(object=name, dv=dv, dir_deg=round(360.0 * k / n_dirs, 1), disp_m=None, censored=True))
                continue
            this = []
            for k in range(n_dirs):
                ang = 2 * math.pi * k / n_dirs
                d = np.array([math.cos(ang), math.sin(ang), 0.0]) * dv
                ad.goto(ep, ro, t)
                p0 = ad.object_poses()
                if name not in p0:
                    continue
                ad.set_hold_reference()
                if not ad.nudge([name], d):
                    return out
                for _ in range(n_wait):
                    ad.step(ad.hold_action("open"))
                p1 = ad.object_poses()
                disp = float(np.linalg.norm(np.asarray(p1[name][:3]) - np.asarray(p0[name][:3])))
                out.append(dict(object=name, dv=dv, dir_deg=round(math.degrees(ang), 1), disp_m=round(disp, 5)))
                this.append(disp)
            if stop_disp is not None and this and min(this) > stop_disp:
                stopped = True
    return out


def choose_dv(samples: list, max_disp: float = 0.01) -> dict:
    """Largest dv whose median displacement < max_disp (and every smaller dv also satisfies it)."""
    by = {}
    for s in samples:
        by.setdefault(s["dv"], []).append(float("inf") if s.get("disp_m") is None else s["disp_m"])
    table = {dv: dict(n=len(v), median=float(np.median(v)), p90=float(np.percentile(v, 90, method="higher")),
                      max=float(np.max(v)), censored=int(sum(1 for x in v if x == float("inf"))))
             for dv, v in sorted(by.items())}
    chosen = None
    for dv in sorted(table):
        if table[dv]["median"] < max_disp:
            chosen = dv
        else:
            break
    return dict(dv=chosen, table={str(k): v for k, v in table.items()}, max_disp_m=max_disp)
