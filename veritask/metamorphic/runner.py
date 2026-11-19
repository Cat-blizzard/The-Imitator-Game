"""Run the metamorphic test items on episodes through an Adapter and write one JSONL row per episode.

Row schema 'b1.row.v1' (all keys always present unless noted):
  benchmark, source, task, episode, task_class, instruction, official_semantics, primary_point
  official            the benchmark's official verdict of the original episode
  n_steps, success_any, success_end, t_first     from the original episode's per-step predicate trace
  points: {first|end: {t, same_as, objects, M1, M2, M3, M5, M4}}   PhaseLog dicts; 'end' only if the predicate holds
          at the last step; an item that does not apply is absent; 'same_as' marks a test point identical to another
  M0: {success_any, success_end, n_steps, t_first}
  M6: {success_any, success_end, frozen, n_steps, t_first, error}
  M7: {ref: {success_any, success_end, n_steps}, half: {...}, ref_is_original, error}
  nudge_dir, protocol (version), wall_s, errors
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
import traceback
from typing import Any, Callable, Iterable

import numpy as np

from .adapter import Adapter, PhaseLog, Rollout
from .protocol import PLACE_ONLY, Protocol

ROW_SCHEMA = "b1.row.v1"


def nudge_direction(key: str) -> list:
    """Deterministic horizontal unit vector from an episode key (sha256 -> angle)."""
    h = int(hashlib.sha256(key.encode()).hexdigest()[:12], 16)
    ang = (h % 3600000) / 3600000.0 * 2 * math.pi
    return [round(math.cos(ang), 6), round(math.sin(ang), 6), 0.0]


def _first(seq) -> int | None:
    for i, s in enumerate(seq):
        if s:
            return i
    return None


def _trace_summary(succ: list) -> dict:
    return dict(success_any=bool(any(succ)), success_end=bool(succ[-1]) if succ else False,
                n_steps=len(succ), t_first=_first(succ))


def _safe(fn: Callable, errors: list, tag: str):
    try:
        return fn()
    except Exception as ex:  # one failing item must not lose the episode
        errors.append(dict(item=tag, error=f"{type(ex).__name__}: {ex}"[:300], tb=traceback.format_exc()[-800:]))
        return None


def run_episode(ad: Adapter, ep, proto: Protocol, *, task: str, episode_id: str, task_class: str,
                instruction: str = "", source: str = "", primary_point: str | None = None,
                extra: dict | None = None, items: Iterable[str] | None = None,
                controls_on_nonplace: bool = True) -> dict:
    """All requested items for one episode. `controls_on_nonplace`: also execute M2/M3 on hold/event tasks as an
    expected-failure control (reported separately, never pooled with place-type statistics)."""
    t0 = time.time()
    items = tuple(items or proto.items)
    errors: list = []
    row: dict[str, Any] = dict(schema=ROW_SCHEMA, protocol=proto.version, benchmark=ad.benchmark, source=source,
                               task=task, episode=episode_id, task_class=task_class, instruction=instruction)
    if extra:
        row.update(extra)
    ro: Rollout | None = _safe(lambda: ad.rollout(ep), errors, "rollout")
    if ro is None or not ro.success:
        row.update(rollout_ok=False, errors=errors, wall_s=round(time.time() - t0, 2))
        return row
    succ = [bool(x) for x in ro.success]
    row.update(rollout_ok=True, official=bool(ro.official), official_semantics=ro.official_semantics,
               primary_point=primary_point or ("first" if ro.official_semantics == "any" else "end"),
               **_trace_summary(succ), rollout_meta=ro.meta)
    ndir = nudge_direction(f"{ad.benchmark}|{task}|{episode_id}")
    row["nudge_dir"] = ndir
    # ---------------------------------------------------------------- post-success items at the test points
    pts = {}
    if "first" in proto.test_points and row["t_first"] is not None:
        pts["first"] = row["t_first"]
    if "end" in proto.test_points and row["success_end"]:
        pts["end"] = row["n_steps"] - 1
    row["points"] = {}
    run_place = task_class == "place" or controls_on_nonplace
    for name, t in pts.items():
        same = [k for k, v in row["points"].items() if v.get("t") == t and not v.get("same_as")]
        if same:
            row["points"][name] = dict(t=t, same_as=same[0])
            continue
        P: dict[str, Any] = dict(t=t, same_as=None)
        def go():
            ad.goto(ep, ro, t)
            return True
        if _safe(go, errors, f"{name}.goto"):
            P["objects"] = {k: [round(float(x), 5) for x in v] for k, v in ad.object_poses().items()}
            P["predicate_at_point"] = bool(ad.success())
            if hasattr(ad, "grasped"):
                g = _safe(ad.grasped, errors, f"{name}.grasped")
                if g is not None:
                    P["grasped"] = {k: bool(v) for k, v in g.items()}
            if "M1" in items:
                lg = _safe(lambda: ad.phase_hold(proto), errors, f"{name}.M1")
                if lg is not None:
                    P["M1"] = lg.to_dict()
            if "M2" in items and run_place:
                if _safe(go, errors, f"{name}.goto2"):
                    lg = _safe(lambda: ad.phase_release(proto), errors, f"{name}.M2")
                    if lg is not None:
                        P["M2"] = lg.to_dict()
                        if "M3" in items:
                            lg3 = _safe(lambda: ad.phase_retreat(proto), errors, f"{name}.M3")
                            if lg3 is not None:
                                P["M3"] = lg3.to_dict()
                                if "M5" in items and proto.nudge_dv_mps and task_class == "place":
                                    lg5 = _safe(lambda: ad.phase_nudge(proto, ndir), errors, f"{name}.M5")
                                    if lg5 is not None:
                                        P["M5"] = lg5.to_dict()
            if "M4" in items and ad.has_policy:
                if _safe(go, errors, f"{name}.goto4"):
                    lg = _safe(lambda: ad.phase_policy(proto), errors, f"{name}.M4")
                    if lg is not None:
                        P["M4"] = lg.to_dict()
        row["points"][name] = P
    # ---------------------------------------------------------------- M0: do nothing from the initial state
    if "M0" in items:
        def m0():
            ad.reset_initial(ep, ro)
            n = int(ad.horizon(ep, ro))
            lg = ad.phase_idle(n)
            return dict(success_any=bool(lg.success_any), success_end=bool(lg.success), n_steps=lg.n_steps,
                        success_frac=lg.success_frac)
        r = _safe(m0, errors, "M0")
        if r is not None:
            row["M0"] = r
    # ---------------------------------------------------------------- M6: goal objects frozen, same actions
    if "M6" in items:
        r = _safe(lambda: ad.replay(ep, ro, freeze=True), errors, "M6")
        if r is not None:
            s = [bool(x) for x in r.get("success", [])]
            row["M6"] = dict(**_trace_summary(s), frozen=r.get("frozen"), error=r.get("error"),
                             info={k: v for k, v in r.items() if k not in ("success", "frozen", "error")})
    # ---------------------------------------------------------------- M7: half physics step, same actions
    if "M7" in items:
        def m7():
            out = {}
            if ro.meta.get("replay_is_original"):
                out["ref"] = _trace_summary(succ); out["ref_is_original"] = True
            else:
                r1 = ad.replay(ep, ro, dt_scale=1.0)
                out["ref"] = dict(**_trace_summary([bool(x) for x in r1.get("success", [])]),
                                  info={k: v for k, v in r1.items() if k not in ("success",)})
                out["ref_is_original"] = False
            r2 = ad.replay(ep, ro, dt_scale=proto.half_step_scale)
            out["half"] = dict(**_trace_summary([bool(x) for x in r2.get("success", [])]),
                               info={k: v for k, v in r2.items() if k not in ("success",)})
            return out
        r = _safe(m7, errors, "M7")
        if r is not None:
            row["M7"] = r
    row["errors"] = errors
    row["wall_s"] = round(time.time() - t0, 2)
    return row


def _jsonable(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (set, tuple)):
        return list(x)
    return str(x)


def append_row(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=_jsonable) + "\n")


def done_episodes(path: str) -> set:
    out = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                    out.add((r.get("task"), str(r.get("episode"))))
                except Exception:
                    pass
    return out
