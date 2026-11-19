"""Task credibility card: what evidence stands behind a contract, and how far it can be trusted.

Levels (cumulative; each states what has been *shown*, not assumed):
  L0 declared          the contract validates; nothing executed yet
  L1 counterexample    >= 3 distinct designed counterexample behaviours executed and all rejected, and every
                       designed positive control accepted in >= 90% of its episodes
  L2 engine-robust     L1, and the same behaviours (>= 1 control and >= 1 counterexample) evaluated in >= 2 physics
                       engines with the same majority verdict in each engine
  L3 real-validated    L2, and >= 20 real-world episodes ('engine' == 'real') whose contract verdict agrees with the
                       observed real outcome in >= 90% of cases

The card also reports what the benchmark's own predicate did on the same episodes, the fragility of the verdict
(how close accepted controls sit to the threshold), and declared counterexamples that were never executed.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .evidence import EvidenceRow

LEVELS = {0: "L0 declared", 1: "L1 counterexample-tested", 2: "L2 engine-robust", 3: "L3 real-validated"}


def _rate(k, n):
    return (k / n) if n else None


def summarize(contract, rows: list[EvidenceRow]) -> dict:
    rows = [r for r in rows if r.contract == contract.task_id]
    beh = defaultdict(list)
    for r in rows:
        beh[(r.behaviour, r.role)].append(r)
    per = []
    for (b, role), rs in sorted(beh.items(), key=lambda kv: (["control", "counterexample", "near_miss", "natural"].index(kv[0][1]), kv[0][0])):
        known = [r for r in rs if r.verdict is not None]
        bk = [r for r in rs if r.bench is not None]
        margins = [r.margin for r in rs if r.margin is not None]
        per.append(dict(behaviour=b, role=role, n=len(rs), unknown=len(rs) - len(known),
                        contract_accepted=sum(r.verdict for r in known), contract_rate=_rate(sum(r.verdict for r in known), len(known)),
                        bench_n=len(bk), bench_accepted=sum(r.bench for r in bk), bench_rate=_rate(sum(r.bench for r in bk), len(bk)),
                        engines=sorted({r.engine for r in rs}), experiments=sorted({r.experiment for r in rs if r.experiment}),
                        margin_min=min(margins) if margins else None, margin_max=max(margins) if margins else None))

    def group(role):
        g = [p for p in per if p["role"] == role]
        n = sum(p["n"] - p["unknown"] for p in g)
        return dict(behaviours=len(g), episodes=sum(p["n"] for p in g), known=n,
                    contract_accepted=sum(p["contract_accepted"] for p in g),
                    bench_known=sum(p["bench_n"] for p in g), bench_accepted=sum(p["bench_accepted"] for p in g))

    cx, ctl, nat = group("counterexample"), group("control"), group("natural")
    nat_rows = [r for r in rows if r.role == "natural" and r.verdict is not None and r.bench is not None]
    nat["bench_only"] = sum(r.bench and not r.verdict for r in nat_rows)
    nat["contract_only"] = sum(r.verdict and not r.bench for r in nat_rows)

    # cross-engine agreement on behaviours run in >= 2 engines
    by_be = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.verdict is not None and r.engine and r.role in ("control", "counterexample"):
            by_be[(r.behaviour, r.role)][r.engine].append(bool(r.verdict))
    xeng = []
    for (b, role), engs in sorted(by_be.items()):
        if len(engs) >= 2:
            maj = {e: (np.mean(v) >= 0.5) for e, v in engs.items()}
            xeng.append(dict(behaviour=b, role=role, engines=sorted(engs), majority={e: bool(v) for e, v in maj.items()},
                             agree=len(set(maj.values())) == 1))

    # independent truth (human / real outcomes)
    tr = [r for r in rows if r.truth is not None and r.verdict is not None]
    real = [r for r in tr if r.engine == "real"]
    truth = dict(n=len(tr), agree=sum(r.verdict == r.truth for r in tr),
                 bench_agree=sum(r.bench == r.truth for r in tr if r.bench is not None),
                 real_n=len(real), real_agree=sum(r.verdict == r.truth for r in real))

    ctl_margins = [r.margin for r in rows if r.role == "control" and r.verdict and r.margin is not None]
    cx_margins = [r.margin for r in rows if r.role == "counterexample" and r.verdict is False and r.margin is not None]
    indep = [r for r in rows if r.extra.get("independent") and r.truth is not None]
    red_team = dict(n=len(indep), attacks=len({r.behaviour for r in indep}),
                    false_accepts=sum(1 for r in indep if r.truth is False and r.verdict is True),
                    false_rejects=sum(1 for r in indep if r.truth is True and r.verdict is False),
                    bench_false_accepts=sum(1 for r in indep if r.truth is False and r.bench is True),
                    bench_false_rejects=sum(1 for r in indep if r.truth is True and r.bench is False))
    bench_sem = sorted({str(r.extra.get("bench_semantics")) for r in rows if r.bench is not None and r.extra.get("bench_semantics")})
    declared = {c.name for c in contract.counterexamples}
    executed = {p["behaviour"] for p in per}
    s = dict(task_id=contract.task_id, n_rows=len(rows), behaviours=per, counterexamples=cx, controls=ctl, natural=nat,
             cross_engine=xeng, truth=truth,
             fragility=dict(control_min_margin=min(ctl_margins) if ctl_margins else None,
                            counterexample_max_margin=max(cx_margins) if cx_margins else None),
             declared_not_executed=sorted(declared - executed), bench_semantics=bench_sem, red_team=red_team)
    s["level"], s["level_reasons"] = level(s)
    return s


def level(s: dict) -> tuple[int, list[str]]:
    why = []
    cx_beh = [p for p in s["behaviours"] if p["role"] == "counterexample" and p["n"] > p["unknown"]]
    ctl_beh = [p for p in s["behaviours"] if p["role"] == "control" and p["n"] > p["unknown"]]
    l1 = True
    if len(cx_beh) < 3:
        l1 = False; why.append(f"only {len(cx_beh)} distinct counterexample behaviours executed (need >= 3)")
    acc = [p["behaviour"] for p in cx_beh if p["contract_accepted"] > 0]
    if acc:
        l1 = False; why.append(f"counterexamples accepted by the contract: {acc}")
    if not ctl_beh:
        l1 = False; why.append("no positive control executed")
    low = [p["behaviour"] for p in ctl_beh if (p["contract_rate"] or 0) < 0.9]
    if low:
        l1 = False; why.append(f"positive controls accepted in < 90% of episodes: {low}")
    if not l1:
        return 0, why
    xe = s["cross_engine"]
    roles = {x["role"] for x in xe}
    l2 = bool(xe) and roles >= {"control", "counterexample"} and all(x["agree"] for x in xe)
    if not l2:
        why.append("no cross-engine evidence" if not xe else
                   ("cross-engine disagreement: " + str([x["behaviour"] for x in xe if not x["agree"]]) if not all(x["agree"] for x in xe)
                    else "cross-engine evidence lacks a control or a counterexample"))
        return 1, why
    t = s["truth"]
    if t["real_n"] >= 20 and t["real_agree"] / t["real_n"] >= 0.9:
        return 3, why
    why.append(f"real-world validation: {t['real_n']} episodes (need >= 20 with >= 90% agreement)")
    return 2, why


# ---------------------------------------------------------------------------------------------------- rendering
def _f(x, nd=2):
    if x is None: return "-"
    if isinstance(x, float): return f"{x:.{nd}f}"
    return str(x)


def render_markdown(contract, s: dict) -> str:
    L = []
    L.append(f"# 可信度卡片：`{contract.task_id}`\n")
    L.append(f"- **指令**：{contract.instruction}")
    L.append(f"- **可信度等级**：**{LEVELS[s['level']]}**")
    for w in s["level_reasons"]:
        L.append(f"  - 未达到下一级的原因：{w}")
    src = contract.source or {}
    if src:
        L.append(f"- **来源**：{src.get('benchmark', '')} {src.get('task', '')}".rstrip())
        if src.get("released_code_predicate"):
            L.append(f"- **原判据**：{src['released_code_predicate']}")
    L.append(f"- **要求松手**：{'是' if contract.require_released else '否'}；**静止时间**：{contract.settle_seconds} s\n")
    L.append("## 合同条款\n")
    L.append("| 测量 | 条件 | 单位 | 真机怎么测（摘要） |\n|---|---|---|---|")
    for t in contract.outcome:
        rm = (t.real_measurement or "").replace("|", "/").replace("\n", " ")
        L.append(f"| `{t.key}` | {t.op} {t.threshold} | {t.units} | {rm[:120] + ('…' if len(rm) > 120 else '')} |")
    cx, ctl, nat = s["counterexamples"], s["controls"], s["natural"]
    L.append("\n## 证据汇总\n")
    bench_txt = (f"benchmark 自带判据接受 {cx['bench_accepted']}/{cx['bench_known']}" if cx["bench_known"]
                 else "benchmark 没有对应判据或没有记录")
    L.append(f"- **反例**：{cx['behaviours']} 种行为、{cx['episodes']} 个 episode。合同接受 **{cx['contract_accepted']}/{cx['known']}**；{bench_txt}")
    L.append(f"- **正对照**：{ctl['behaviours']} 种行为、{ctl['episodes']} 个 episode。合同接受 **{ctl['contract_accepted']}/{ctl['known']}**")
    if nat["episodes"]:
        L.append(f"- **自然 episode**（策略、专家、数据引擎）：{nat['episodes']} 个。benchmark 判成功、合同判失败 {nat['bench_only']} 个；"
                 f"合同判成功、benchmark 判失败 {nat['contract_only']} 个")
    rt = s.get("red_team") or {}
    if rt.get("n"):
        L.append(f"- **独立红队**：{rt['attacks']} 种攻击、{rt['n']} 个裁决过的 episode。合同误接受 **{rt['false_accepts']}**、误拒绝 "
                 f"**{rt['false_rejects']}**；同一批 episode 上 benchmark 误接受 {rt['bench_false_accepts']}、误拒绝 {rt['bench_false_rejects']}")
    else:
        L.append("- **独立红队**：尚未攻击（目前的等级只基于我们自己设计的反例）")
    for b in s.get("bench_semantics", []):
        L.append(f"  - benchmark 判定口径：{b}")
    fr = s["fragility"]
    L.append(f"- **脆弱度**：正对照的最小归一化余量 {_f(fr['control_min_margin'])}；被拒反例的最大归一化余量 "
             f"{_f(fr['counterexample_max_margin'])}（越接近 0，判定越容易被物理参数或噪声翻转）")
    if s["cross_engine"]:
        L.append(f"- **跨引擎**：" + "；".join(
            f"{x['behaviour']}（{', '.join(f'{e}={v}' for e, v in x['majority'].items())}）{'一致' if x['agree'] else '**不一致**'}"
            for x in s["cross_engine"]))
    t = s["truth"]
    if t["n"]:
        L.append(f"- **独立真值**：{t['n']} 个 episode，合同一致 {t['agree']}，benchmark 一致 {t['bench_agree']}；其中真机 {t['real_n']} 个")
    if s["declared_not_executed"]:
        L.append(f"- **合同里声明了、但还没执行过的反例**：{', '.join(s['declared_not_executed'])}")
    L.append("\n## 逐行为\n")
    L.append("| 行为 | 角色 | n | 合同接受 | benchmark 接受 | 余量范围 | 引擎 | 实验 |\n|---|---|---|---|---|---|---|---|")
    role_zh = {"counterexample": "反例", "control": "正对照", "natural": "自然", "near_miss": "近失败"}
    for p in s["behaviours"]:
        L.append(f"| {p['behaviour']} | {role_zh[p['role']]} | {p['n']} | {p['contract_accepted']}/{p['n'] - p['unknown']} | "
                 f"{(str(p['bench_accepted']) + '/' + str(p['bench_n'])) if p['bench_n'] else '-'} | {_f(p['margin_min'])} … {_f(p['margin_max'])} | "
                 f"{', '.join(p['engines'])} | {', '.join(p['experiments'])} |")
    if contract.notes:
        L.append(f"\n## 备注\n\n{contract.notes}")
    L.append("\n*本卡片由 `python -m veritask card` 根据证据账本自动生成；等级只反映已经执行过的证据。*")
    return "\n".join(L) + "\n"
