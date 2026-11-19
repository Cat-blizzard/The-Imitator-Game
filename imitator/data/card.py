"""Dataset card (markdown + json) for an imsim dataset directory: counts, certification rate (Wilson 95 % CI),
rejection causes, episode lengths, margins, benchmark-flag agreement, noise actually injected, physical draws, the
perturbation spec, contracts (sha256), code hash and throughput.

    imitator.data.card.write_card(dataset_dir)       ->  <dir>/card.md, <dir>/card.json
    python -m imitator card <dataset_dir>
"""
from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _q(x, qs=(0.0, 0.5, 0.9, 1.0)) -> dict:
    x = np.asarray([v for v in x if v is not None], float)
    if not len(x):
        return {}
    d = {("min" if q == 0 else "max" if q == 1 else f"p{int(q * 100)}"): float(np.quantile(x, q)) for q in qs}
    d["mean"] = float(x.mean())
    return d


def summarize(root: str) -> dict:
    meta = json.load(open(os.path.join(root, "dataset.json")))
    rows = [json.loads(l) for l in open(os.path.join(root, "attempts.jsonl")) if l.strip()]
    im = [r["imsim"] for r in rows]
    n = len(rows)
    cert = [r for r in rows if r["imsim"]["certified"]]
    rej = [r for r in rows if not r["imsim"]["certified"]]
    k = len(cert)
    causes = Counter(r["imsim"]["rejection_cause"] for r in rej)
    all_causes = Counter(c for r in rej for c in r["imsim"].get("causes", []))
    cids = list((meta.get("contracts") or {}).keys())
    margins = {cid: _q([(r["imsim"].get("verdicts") or {}).get(cid, {}).get("margin") for r in cert]) for cid in cids}
    contract_rate = {}
    for cid in cids:
        v = [(r["imsim"].get("verdicts") or {}).get(cid, {}).get("success") for r in rows]
        contract_rate[cid] = dict(accepted=sum(1 for x in v if x is True), rejected=sum(1 for x in v if x is False),
                                  unknown=sum(1 for x in v if x is None))
    bench_c = [r["imsim"].get("bench", {}).get("final") for r in cert]
    bench_r = [r["imsim"].get("bench", {}).get("final") for r in rej if r["imsim"]["split"] == "rejected"]
    noise = {}
    for r in rows:
        for a, d in (r["imsim"].get("noise") or {}).items():
            if isinstance(d, dict) and d.get("noised_frac", 1.0) > 0 and d.get("rms_rad", 0) > 0:
                noise.setdefault(a, dict(rms_noised=[], noised_frac=[]))
                noise[a]["rms_noised"].append(d.get("rms_noised_rad", d.get("rms_rad")))
                noise[a]["noised_frac"].append(d.get("noised_frac"))
    phys_keys = Counter()
    phys_vals = {}
    for r in rows:
        ph = r["imsim"].get("phys") or {}
        if "cloth" in ph:                                   # cloth draw: one dict per world
            ph = {"towel": ph}
        for name, d in ph.items():
            if isinstance(d, dict):
                phys_keys[name] += 1
                for kk in ("dx", "dy", "dyaw_deg", "mass_scale", "friction_scale"):
                    if d.get(kk) is not None:
                        phys_vals.setdefault(f"{name}.{kk}", []).append(d[kk])
    s = meta.get("summary") or {}
    out = dict(
        dataset=os.path.abspath(root), task=meta.get("task"), embodiment=meta.get("embodiment"),
        perturb=meta.get("perturb"), perturb_text=meta.get("perturb_text"), devmodel=meta.get("devmodel"),
        certify=meta.get("certify", True), keep_rule=meta.get("keep_rule"),
        counts=dict(attempted=n, certified=k, rejected=sum(1 for i in im if i["split"] == "rejected"),
                    errors=sum(1 for i in im if i["split"] == "error")),
        certification_rate=k / max(n, 1), rate_ci95=wilson(k, n),
        rejection_causes=dict(causes.most_common()), all_failed_checks=dict(all_causes.most_common()),
        per_contract=contract_rate,
        length_certified=_q([r["elapsed_steps"] for r in cert]), length_all=_q([r["elapsed_steps"] for r in rows]),
        transitions_certified=int(sum(r["elapsed_steps"] for r in cert)),
        margins_certified=margins,
        bench=dict(certified_bench_true=sum(1 for b in bench_c if b), certified=len(bench_c),
                   rejected_bench_true=sum(1 for b in bench_r if b), rejected=len(bench_r),
                   note="benchmark-style info['success'] at the last policy step; rejected_bench_true = episodes the "
                        "benchmark flag would accept but a contract rejects"),
        noise_rms_rad={a: _q(v["rms_noised"]) for a, v in noise.items()},
        noise_frac={a: _q(v["noised_frac"]) for a, v in noise.items()},
        phys_draws={kk: _q(v) for kk, v in phys_vals.items()}, phys_actors=dict(phys_keys),
        contracts=meta.get("contracts"), code_hash=meta.get("code_hash"), generator=meta.get("generator"),
        created=meta.get("created"), format=meta.get("format"), label_key=meta.get("label_key"),
        phase_codes=meta.get("phase_codes"),
        throughput=dict(wall_s=s.get("wall_s"), workers=s.get("workers"), episodes_per_hour=s.get("episodes_per_hour"),
                        certified_per_hour=s.get("certified_per_hour"), cpu_s_per_episode=s.get("cpu_s_per_episode"),
                        num_envs=meta.get("num_envs")),
        seeds=meta.get("seeds"))
    return out


def render_md(c: dict) -> str:
    lo, hi = c["rate_ci95"]
    L = [f"# Dataset card: {c['task']} / {c['embodiment']} / {(c['perturb'] or {}).get('name')}", "",
         f"`{c['dataset']}`", "",
         "## Summary", "",
         f"- episodes attempted: {c['counts']['attempted']} (seeds {_seeds(c['seeds'])}); certified: {c['counts']['certified']}; "
         f"rejected: {c['counts']['rejected']}; errors: {c['counts']['errors']}",
         f"- certification rate: {c['certification_rate']:.1%} (Wilson 95% CI {lo:.1%} - {hi:.1%})",
         f"- keep rule: {c['keep_rule']}",
         f"- certified transitions (labelled steps): {c['transitions_certified']}",
         f"- labels: `{c['label_key']}` (clean expert actions); `actions` = executed (noisy) actions for replay; "
         f"phases per step and arm in `phase` ({c['phase_codes']})", "",
         "## Perturbation", "", f"- {c['perturb_text']}", f"- spec: `{json.dumps(c['perturb'], sort_keys=True)}`"]
    if c.get("devmodel"):
        L.append(f"- deviation model: `{json.dumps(c['devmodel'], sort_keys=True)[:600]}`")
    if c["noise_rms_rad"]:
        L.append("- injected noise (executed - clean arm joint targets) in episodes where the arm was noised: RMS over "
                 "the noised steps [rad] " + "; ".join(
                     f"{a}: median {d.get('p50', 0):.4f} (max {d.get('max', 0):.4f}), noised on "
                     f"{(c.get('noise_frac') or {}).get(a, {}).get('p50', float('nan')):.0%} of the steps (median)"
                     for a, d in c["noise_rms_rad"].items() if isinstance(d, dict) and d))
    if c["phys_draws"]:
        L.append("- physical draws: " + "; ".join(f"{k} in [{d['min']:.3g}, {d['max']:.3g}]"
                                                  for k, d in c["phys_draws"].items() if d))
    L += ["", "## Rejection causes (first failing check)", "", "| cause | episodes |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in c["rejection_causes"].items()] or ["| (none) | 0 |"]
    if c["all_failed_checks"] and c["all_failed_checks"] != c["rejection_causes"]:
        L += ["", "All failed checks (an episode can fail several contracts):", "", "| check | episodes |", "|---|---|"]
        L += [f"| {k} | {v} |" for k, v in c["all_failed_checks"].items()]
    L += ["", "## Contracts", "", "| contract | sha256 | accepted | rejected | unknown | certified min norm. margin (median) |",
          "|---|---|---|---|---|---|"]
    for cid, d in (c["contracts"] or {}).items():
        pc = c["per_contract"].get(cid, {})
        m = c["margins_certified"].get(cid) or {}
        L.append(f"| {cid} | {d['sha256'][:12]} | {pc.get('accepted')} | {pc.get('rejected')} | {pc.get('unknown')} | "
                 f"{m.get('p50', float('nan')):.3f} |")
    b = c["bench"]
    L += ["", "## Benchmark flag", "",
          f"- certified episodes with the benchmark flag set: {b['certified_bench_true']} / {b['certified']}",
          f"- rejected episodes the benchmark flag would accept: {b['rejected_bench_true']} / {b['rejected']}",
          "", "## Episode length (policy steps)", "",
          f"- certified: {_fmt(c['length_certified'])}", f"- all attempts: {_fmt(c['length_all'])}",
          "", "## Provenance and throughput", "",
          f"- generator: {c['generator']}; format: {c['format']}; created {c['created']}",
          f"- code hash: {c['code_hash']}",
          f"- throughput: {c['throughput']}"]
    return "\n".join(L) + "\n"


def _seeds(s) -> str:
    if isinstance(s, (list, tuple)) and len(s) == 3:
        return f"{s[0]}..{s[1]}, {s[2]} seeds"
    return str(s)


def _fmt(d: dict) -> str:
    if not d:
        return "-"
    return f"min {d['min']:.0f}, median {d['p50']:.0f}, mean {d['mean']:.1f}, p90 {d['p90']:.0f}, max {d['max']:.0f}"


def write_card(root: str) -> dict:
    c = summarize(root)
    md, js = os.path.join(root, "card.md"), os.path.join(root, "card.json")
    with open(md, "w") as f:
        f.write(render_md(c))
    from .store import jsonable
    json.dump(jsonable(c), open(js, "w"), indent=1)
    return dict(md=md, json=js, card=c)
