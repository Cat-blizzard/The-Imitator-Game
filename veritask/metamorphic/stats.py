"""Summaries of metamorphic-test rows (schema b1.row.v1): rates with Wilson 95% intervals.

Definitions (pre-registered in research/results/b1_metamorphic/prereg.json):
  N            episodes whose original rollout completed
  official     k = episodes the benchmark's official evaluation credits
  test point   P = the row's primary point ('first' for any-step scoring, 'end' for final-step scoring); the base set
               S_P = episodes with the predicate true at P (equals the official set for both scoring rules)
  item Mi      still-success share = #(success at the end of Mi) / #(S_P with Mi executed, applicable, not excluded)
  excluded     M3 / M5: episodes with an arm-object contact during the retreat (lift + wait) are not counted as
               failures; they are reported separately (n_contact, of which still successful)
  persistent   place-type: success after M1, M2 and M3; hold/event-type: success after M1
  M0 / M6      share of episodes whose predicate fires at any step / at the end (expected 0)
  M7           share of episodes whose official verdict flips between the reference replay and the half-step replay
"""
from __future__ import annotations

import math
from collections import OrderedDict


def wilson(k: int, n: int, z: float = 1.96) -> list:
    if n <= 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def count(k: int, n: int) -> dict:
    return dict(k=int(k), n=int(n), rate=(round(k / n, 4) if n else None), ci95=wilson(k, n))


def _point(r: dict, name: str) -> dict | None:
    pts = r.get("points") or {}
    p = pts.get(name)
    if p and p.get("same_as"):
        p = pts.get(p["same_as"])
    return p


def _ok(log) -> bool | None:
    if not isinstance(log, dict) or log.get("error"):
        return None
    return log.get("success")


def summarize_group(rows: list, point: str | None = None) -> dict:
    rows = [r for r in rows if r.get("rollout_ok")]
    N = len(rows)
    out = OrderedDict(N=N, official=count(sum(bool(r.get("official")) for r in rows), N))
    classes = sorted({r.get("task_class") for r in rows})
    out["task_classes"] = classes
    for pt in ([point] if point else ["primary", "first", "end"]):
        res = OrderedDict()
        base = []
        for r in rows:
            name = r.get("primary_point") if pt == "primary" else pt
            p = _point(r, name)
            if p is not None and ("M1" in p or "M2" in p):
                base.append((r, p))
        res["n_point"] = len(base)
        for item in ("M1", "M2", "M3", "M5", "M4"):
            k = n = n_contact = k_contact = 0
            for r, p in base:
                if item not in p:
                    continue
                if item in ("M2", "M3", "M5") and r.get("task_class") != "place":
                    continue
                s = _ok(p[item])
                if s is None:
                    continue
                contact = item in ("M3", "M5") and bool((p.get("M3") or {}).get("contact_task") or
                                                       (p.get("M3") or {}).get("contact_other") or
                                                       (item == "M5" and ((p.get("M5") or {}).get("contact_task") or
                                                                          (p.get("M5") or {}).get("contact_other"))))
                if contact:
                    n_contact += 1; k_contact += bool(s); continue
                n += 1; k += bool(s)
            if n or n_contact:
                res[item] = count(k, n)
                if item in ("M3", "M5"):
                    res[item]["excluded_contact"] = dict(n=n_contact, still_success=k_contact)
        # persistent success at this point
        k = n = 0
        for r, p in base:
            need = ("M1", "M2", "M3") if r.get("task_class") == "place" else ("M1",)
            vals = [_ok(p.get(i)) for i in need]
            if any(v is None for v in vals):
                continue
            if r.get("task_class") == "place" and ((p.get("M3") or {}).get("contact_task") or
                                                   (p.get("M3") or {}).get("contact_other")):
                continue
            n += 1; k += all(vals)
        res["persistent"] = count(k, n)
        res["not_persistent"] = count(n - k, n)
        # controls: M2/M3 on hold/event tasks (expected to fail by task definition)
        for item in ("M2", "M3"):
            kk = nn = 0
            for r, p in base:
                if r.get("task_class") == "place" or item not in p:
                    continue
                s = _ok(p[item])
                if s is None:
                    continue
                nn += 1; kk += bool(s)
            if nn:
                res[f"{item}_nonplace_control"] = count(kk, nn)
        out[pt] = res
    for item, key in (("M0", "success_any"), ("M0", "success_end"), ("M6", "success_any"), ("M6", "success_end")):
        xs = [r[item][key] for r in rows if isinstance(r.get(item), dict) and not r[item].get("error")]
        if xs:
            out[f"{item}_{key}"] = count(sum(bool(x) for x in xs), len(xs))
    flips = n7 = 0
    for r in rows:
        m7 = r.get("M7")
        if not isinstance(m7, dict) or "half" not in m7 or "ref" not in m7:
            continue
        key = "success_any" if r.get("official_semantics") == "any" else "success_end"
        n7 += 1; flips += bool(m7["half"][key]) != bool(m7["ref"][key])
    if n7:
        out["M7_flip"] = count(flips, n7)
    return out


def summarize(rows: list, group_by=("benchmark", "source", "task_class")) -> list:
    groups: dict = OrderedDict()
    for r in rows:
        groups.setdefault(tuple(r.get(k) for k in group_by), []).append(r)
    return [dict(group=dict(zip(group_by, g)), **summarize_group(rs)) for g, rs in groups.items()]


def fmt(c: dict | None) -> str:
    if not c or c.get("n", 0) == 0:
        return "-"
    lo, hi = c["ci95"]
    return f"{c['k']}/{c['n']} {c['rate']:.3f} [{lo:.2f},{hi:.2f}]"
