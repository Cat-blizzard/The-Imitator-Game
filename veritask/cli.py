"""veritask command line.

  python -m veritask validate tasks/*.json
  python -m veritask lint PATH [PATH...] [--json out.json]
  python -m veritask score CONTRACT TRACES.npz [...] [--released-signal released] [--json out.json]
  python -m veritask rescore EPISODES.jsonl --field success_trace [--modes any,final,window:10,persist:20]
                                            [--group-by model,suite] [--json out.json]
  python -m veritask credibility EPISODES.jsonl [--verdict verdict --truth truth --behaviour behaviour --margin margin]
  python -m veritask card CONTRACT EVIDENCE.jsonl [...] [--out card.md] [--json card.json]
  python -m veritask primitives [--weakness W | --flags F1,F2]
  python -m veritask metrics

Every subcommand prints a human-readable table; --json writes the full machine-readable result.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import warnings


def _dump(obj, path):
    if path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=1, ensure_ascii=False, default=_jsonable)
        print(f"wrote {path}")


def _jsonable(x):
    try:
        import numpy as np
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return float(x)
        if isinstance(x, np.ndarray): return x.tolist()
    except ImportError:
        pass
    return str(x)


def _expand(paths, pattern="*.py"):
    out = []
    for p in paths:
        out += sorted(glob.glob(os.path.join(p, pattern))) if os.path.isdir(p) else [p]
    return out


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _fmt(x, nd=3):
    if x is None: return "-"
    if isinstance(x, float):
        return "nan" if math.isnan(x) else f"{x:.{nd}f}"
    return str(x)


# ------------------------------------------------------------------------------------------------ validate
def cmd_validate(a) -> int:
    from .spec.contract import validate_dict
    bad = 0
    for f in _expand(a.files, "*.json"):
        try:
            with open(f, encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception as e:     # malformed JSON
            print(f"ERROR {f}: cannot parse JSON: {e}"); bad += 1; continue
        probs = validate_dict(d)
        errs = [p for p in probs if not p.startswith("warning:")]
        warns = [p for p in probs if p.startswith("warning:")]
        print(f"{'ERROR' if errs else 'OK   '} {f}" + (f"  ({len(warns)} warnings)" if warns else ""))
        for p in errs + (warns if a.verbose else []):
            print("      ", p)
        bad += bool(errs) or (a.strict and bool(warns))
    return 1 if bad else 0


# ------------------------------------------------------------------------------------------------ lint
def cmd_lint(a) -> int:
    from .audit import static_lint
    from .audit.primitives import suggest
    if a.kind == "robotwin":
        raise SystemExit("the RoboTwin linter is not part of Imitator 2.0 (see the research archive)")
        from .audit.robotwin_lint import lint_robotwin  # noqa: F401  (unreachable; kept for reference)
        if len(a.paths) != 1:
            raise SystemExit("--kind robotwin takes the RoboTwin repository root")
        res = lint_robotwin(a.paths[0])
        for r in res["tasks"]:
            r["suggested_primitives"] = suggest(r["flags"])
            shown = " ".join(x for x in r["flags"] if x != "B_NO_STATIC")
            print(f"{r['task']:28s} {r['release']:34s} {shown}")
        print(f"\n{res['n_tasks']} tasks; flag counts: {res['flag_counts']}")
        poll = res["eval_semantics"].get("poll_loop")
        if poll:
            print(f"evaluation: check_success() polled inside take_action's physics loop ({poll['file']}:{poll['while_line']}); "
                  f"first True latches eval_success -> 'any' semantics at sub-step granularity")
        print("Flags are candidates: confirm with an executed counterexample (python -m veritask primitives --flags ...).")
        _dump(res, a.json)
        return 0
    rows = []
    for f in _expand(a.paths):
        try:
            r = static_lint.lint(f)
        except SyntaxError as e:
            print(f"skip {f}: {e}"); continue
        if r:
            r["file"] = f; r["flags"] = static_lint.flags(r); r["suggested_primitives"] = suggest(r["flags"]); rows.append(r)
    for r in rows:
        extra = (" unused=" + ",".join(r["unused_conditions"])) if r["unused_conditions"] else ""
        print(f"{r['task']:40s} {' '.join(r['flags']) or '-':40s}{extra}")
    n = len(rows)
    print(f"\n{n} tasks with evaluate(): UNUSED_COND {sum('UNUSED_COND' in r['flags'] for r in rows)} | "
          f"EVER_LATCH {sum('EVER_LATCH' in r['flags'] for r in rows)} | NO_STATIC_CHECK {sum('NO_STATIC_CHECK' in r['flags'] for r in rows)}")
    print("Flags are candidates: confirm with an executed counterexample (python -m veritask primitives --flags ...).")
    _dump(rows, a.json)
    return 0


# ------------------------------------------------------------------------------------------------ score
def cmd_score(a) -> int:
    from .spec.contract import TaskContract
    from .adapters.base import evaluate_traces, load_traces
    c = TaskContract.load(a.contract)
    extra = json.loads(a.extra) if a.extra else None
    rows = []
    for f in a.traces:
        tr, meta = load_traces(f)
        hz = a.control_freq or meta.get("control_freq") or 20.0
        v, bench = evaluate_traces(c, tr, a.released_signal or None, extra, a.release_window,
                                   protocol=a.protocol, release=json.loads(a.release) if a.release else None, control_freq=hz,
                                   **({"hold": a.hold} if getattr(a, "hold", None) else {}))
        rows.append(dict(file=f, meta=meta, contract_success=v["success"], **bench, binding_term=v["binding_term"],
                         min_norm_margin=v["min_norm_margin"], verdict=v))
    print(f"{'file':50s} {'contract':>9s} {'bench_any':>9s} {'bench_end':>9s}  binding term (norm margin)")
    for r in rows:
        print(f"{os.path.basename(r['file'])[:50]:50s} {_fmt(r['contract_success']):>9s} {_fmt(r['bench_success_any']):>9s} "
              f"{_fmt(r['bench_success_final']):>9s}  {r['binding_term']} ({_fmt(r['min_norm_margin'])})")
    disagree = sum(1 for r in rows if r["bench_success_any"] and r["contract_success"] is False)
    print(f"\n{len(rows)} rollouts | benchmark-accepted but contract-rejected: {disagree}")
    _dump(rows, a.json)
    return 0


# ------------------------------------------------------------------------------------------------ rescore
def cmd_rescore(a) -> int:
    from .spec.semantics import rescore_many
    eps = _read_jsonl(a.episodes)
    keys = [k for k in (a.group_by.split(",") if a.group_by else []) if k]
    groups: dict[tuple, list] = {}
    for e in eps:
        if a.field not in e:
            raise SystemExit(f"episode without field '{a.field}': {list(e)[:10]}")
        groups.setdefault(tuple(e.get(k) for k in keys), []).append(e[a.field])
    modes = a.modes.split(",")
    out = []
    print(f"{'group':30s} {'n':>5s} " + " ".join(f"{m:>18s}" for m in modes) + f"   over-credit by '{a.reference}'")
    for g, traces in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        r = rescore_many(traces, modes, reference=a.reference)
        out.append(dict(group=dict(zip(keys, g)), **r))
        cells = " ".join(f"{r['modes'][m]['rate']:.3f} [{r['modes'][m]['ci95'][0]:.2f},{r['modes'][m]['ci95'][1]:.2f}]".rjust(18)
                         for m in modes)
        over = ", ".join(f"{m}:{r['modes'][m]['over_credit_by_reference']}" for m in modes if m != a.reference)
        print(f"{'/'.join(str(x) for x in g) or 'all':30s} {r['n']:>5d} {cells}   {over}")
    _dump(out, a.json)
    return 0


# ------------------------------------------------------------------------------------------------ credibility
def cmd_credibility(a) -> int:
    from .audit.credibility import Episode, by_margin, confusion, counterexample_pass_rate
    raw = _read_jsonl(a.episodes)
    eps = [Episode(behaviour=str(e.get(a.behaviour, "")), verdict=e.get(a.verdict), truth=e.get(a.truth),
                   margin=e.get(a.margin)) for e in raw]
    c = confusion(eps); cx = counterexample_pass_rate(eps)
    res = dict(confusion=c, per_behaviour=cx)
    print("confusion:", {k: _fmt(v) for k, v in c.items()})
    print("\nper behaviour (accepted / n):")
    for b, r in cx.items():
        print(f"  {b:32s} {r['accepted']:>4d} / {r['n']:<4d} ({_fmt(r['rate'])})")
    if any(e.margin is not None for e in eps):
        res["by_margin"] = by_margin(eps)
        print("\nby |margin|:")
        for r in res["by_margin"]:
            print(f"  [{r['abs_margin_lo']}, {r['abs_margin_hi']}) n={r['n']:<4d} bal_acc={_fmt(r['balanced_accuracy'])} "
                  f"FSR={_fmt(r['false_success_rate'])}")
    _dump(res, a.json)
    return 0


# ------------------------------------------------------------------------------------------------ card
def cmd_card(a) -> int:
    from .spec.contract import TaskContract
    from .audit import evidence
    from .audit.card import LEVELS, render_markdown, summarize
    c = TaskContract.load(a.contract)
    rows = []
    for f in a.evidence:
        rows += evidence.load(f)
    s = summarize(c, rows)
    md = render_markdown(c, s)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"wrote {a.out}")
    else:
        print(md)
    print(f"{c.task_id}: {LEVELS[s['level']]}" + (f"  (next level blocked by: {s['level_reasons'][0]})" if s["level_reasons"] else ""))
    _dump(s, a.json)
    return 0


# ------------------------------------------------------------------------------------------------ primitives / metrics
def cmd_primitives(a) -> int:
    from .audit.primitives import PRIMITIVES, WEAKNESSES, for_weakness, suggest
    if a.flags:
        names = suggest(a.flags.split(","))
        sel = [PRIMITIVES[n] for n in names]
    elif a.weakness:
        if a.weakness not in WEAKNESSES:
            raise SystemExit(f"unknown weakness; one of {sorted(WEAKNESSES)}")
        sel = for_weakness(a.weakness)
    else:
        sel = list(PRIMITIVES.values())
    for p in sel:
        print(f"{p.name:28s} [{', '.join(p.weaknesses)}]\n    {p.description}")
        for ev in p.evidence:
            print(f"    evidence: {ev}")
    return 0


def cmd_metrics(a) -> int:
    from .outcomes.metrics import REGISTRY
    for name, fn in sorted(REGISTRY.items()):
        doc = (fn.__doc__ or "").strip().splitlines()
        print(f"{name:28s} {doc[0] if doc else ''}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="veritask", description="credible physical task verdicts")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("validate", help="check contract files against the schema")
    p.add_argument("files", nargs="+"); p.add_argument("--strict", action="store_true", help="warnings count as errors")
    p.add_argument("-v", "--verbose", action="store_true"); p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("lint", help="static success-predicate lint")
    p.add_argument("paths", nargs="+")
    p.add_argument("--kind", choices=["maniskill", "robotwin"], default="maniskill",
                   help="maniskill: task files with evaluate() (ManiSkill, Imitator); robotwin: repository root")
    p.add_argument("--json"); p.set_defaults(fn=cmd_lint)

    p = sub.add_parser("score", help="contract verdicts for recorded trace files (.npz, adapters/base.save_traces)")
    p.add_argument("contract"); p.add_argument("traces", nargs="+")
    p.add_argument("--released-signal", default="released"); p.add_argument("--release-window", type=int, default=10)
    p.add_argument("--extra", help="JSON dict of non-trace measurements keyed by term key")
    p.add_argument("--protocol", choices=["v1", "v2", "v3"], help="override the contract's evaluation protocol")
    p.add_argument("--hold", choices=["A", "B"], help="protocol v3: evaluator hold (A: robot held where it stopped; "
                                                      "B: robot must come to rest by itself); default: the contract's settle.hold")
    p.add_argument("--release", help='v2 release rule as JSON, e.g. {"contact_signals": [...], "gap_signals": [...]}')
    p.add_argument("--control-freq", type=float, help="control frequency [Hz] (default: trace metadata, else 20)")
    p.add_argument("--json"); p.set_defaults(fn=cmd_score)

    p = sub.add_parser("rescore", help="success rates of recorded per-step success traces under several semantics")
    p.add_argument("episodes"); p.add_argument("--field", default="success_trace")
    p.add_argument("--modes", default="any,final,window:10,persist:20"); p.add_argument("--reference", default="any")
    p.add_argument("--group-by", default=""); p.add_argument("--json"); p.set_defaults(fn=cmd_rescore)

    p = sub.add_parser("credibility", help="false-success rate etc. of a verifier against ground truth")
    p.add_argument("episodes"); p.add_argument("--verdict", default="verdict"); p.add_argument("--truth", default="truth")
    p.add_argument("--behaviour", default="behaviour"); p.add_argument("--margin", default="margin")
    p.add_argument("--json"); p.set_defaults(fn=cmd_credibility)

    p = sub.add_parser("card", help="task credibility card from a contract and evidence ledgers (.jsonl)")
    p.add_argument("contract"); p.add_argument("evidence", nargs="+"); p.add_argument("--out"); p.add_argument("--json")
    p.set_defaults(fn=cmd_card)

    p = sub.add_parser("primitives", help="counterexample behaviour library")
    g = p.add_mutually_exclusive_group(); g.add_argument("--weakness"); g.add_argument("--flags")
    p.set_defaults(fn=cmd_primitives)

    p = sub.add_parser("metrics", help="list registered outcome metrics"); p.set_defaults(fn=cmd_metrics)

    a = ap.parse_args(argv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore" if a.cmd != "validate" else "default")
        return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())


# ================================================================================================================
# metamorphic (card B1, appended 2026-11-16; nothing above this line was changed)
#
#   python -m veritask metamorphic items
#   python -m veritask metamorphic classify "put the bowl on the plate" ["pick up the cube" ...]
#   python -m veritask metamorphic run --adapter MODULE:FACTORY [--adapter-args JSON] --out rows.jsonl
#                                      [--items M0,M1,...] [--protocol JSON] [--limit N]
#       FACTORY(**adapter_args) yields (adapter, episode, meta) with meta keys task, episode_id, task_class
#       [, instruction, source, primary_point]; rows (schema b1.row.v1) are appended, finished episodes skipped.
#   python -m veritask metamorphic summarize rows.jsonl [...] [--group-by benchmark,source,task_class] [--json out]
# ================================================================================================================
def _metamorphic_main(argv) -> int:
    ap = argparse.ArgumentParser(prog="veritask metamorphic",
                                 description="post-success metamorphic tests of a benchmark's own success predicate")
    sub = ap.add_subparsers(dest="mcmd", required=True)
    p = sub.add_parser("items", help="list the test items and their expected outcome")
    p = sub.add_parser("classify", help="place / hold / event class of instructions (keyword rule)")
    p.add_argument("texts", nargs="+")
    p = sub.add_parser("run", help="run the items on episodes provided by an adapter factory")
    p.add_argument("--adapter", required=True, help="MODULE:FACTORY (module importable from sys.path or a .py path)")
    p.add_argument("--adapter-args", default="{}", help="JSON kwargs for the factory")
    p.add_argument("--out", required=True); p.add_argument("--items", default=None)
    p.add_argument("--protocol", default="{}", help="JSON overrides of veritask.metamorphic.Protocol fields")
    p.add_argument("--limit", type=int, default=None)
    p = sub.add_parser("summarize", help="per-group rates with Wilson 95% intervals from rows (b1.row.v1)")
    p.add_argument("rows", nargs="+"); p.add_argument("--group-by", default="benchmark,source,task_class")
    p.add_argument("--json")
    a = ap.parse_args(argv)
    from . import metamorphic as MM
    if a.mcmd == "items":
        from .metamorphic import protocol as P
        doc = (P.__doc__ or "").split("Test points:")[0]
        print(doc.strip())
        return 0
    if a.mcmd == "classify":
        for t in a.texts:
            c, why = MM.classify(t)
            print(f"{c:6s} {t}   ({why})")
        return 0
    if a.mcmd == "run":
        import importlib, importlib.util
        mod_name, fac_name = a.adapter.rsplit(":", 1)
        if mod_name.endswith(".py"):
            spec = importlib.util.spec_from_file_location(os.path.basename(mod_name)[:-3], mod_name)
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        else:
            mod = importlib.import_module(mod_name)
        factory = getattr(mod, fac_name)
        proto = MM.Protocol(**json.loads(a.protocol))
        items = tuple(a.items.split(",")) if a.items else None
        done = MM.done_episodes(a.out)
        n = 0
        for ad, ep, meta in factory(**json.loads(a.adapter_args)):
            if (meta["task"], str(meta["episode_id"])) in done:
                continue
            row = MM.run_episode(ad, ep, proto, items=items, **meta)
            MM.append_row(a.out, row)
            pp = (row.get("points") or {}).get(row.get("primary_point") or "", {}) or {}
            print(f"{row['task']} {row['episode']}: official={row.get('official')} "
                  + " ".join(f"{i}={(pp.get(i) or {}).get('success')}" for i in ("M1", "M2", "M3", "M5", "M4") if i in pp)
                  + f" M0={(row.get('M0') or {}).get('success_any')} M6={(row.get('M6') or {}).get('success_any')}"
                  + f" errors={len(row.get('errors', []))} {row.get('wall_s')}s", flush=True)
            n += 1
            if a.limit and n >= a.limit:
                break
        print(f"{n} episodes -> {a.out}")
        return 0
    if a.mcmd == "summarize":
        rows = []
        for f in a.rows:
            rows += _read_jsonl(f)
        res = MM.summarize(rows, tuple(k for k in a.group_by.split(",") if k))
        from .metamorphic.stats import fmt
        cols = ("M1", "M2", "M3", "M5", "M4", "persistent")
        print(f"{'group':48s} {'N':>5s} {'official':>24s} " + " ".join(f"{c:>24s}" for c in cols)
              + f" {'M0 any':>22s} {'M6 any':>22s} {'M7 flip':>22s}")
        for r in res:
            g = "/".join(str(v) for v in r["group"].values())
            pr = r.get("primary", {})
            print(f"{g[:48]:48s} {r['N']:>5d} {fmt(r['official']):>24s} "
                  + " ".join(f"{fmt(pr.get(c)):>24s}" for c in cols)
                  + f" {fmt(r.get('M0_success_any')):>22s} {fmt(r.get('M6_success_any')):>22s} {fmt(r.get('M7_flip')):>22s}")
        _dump(res, a.json)
        return 0
    return 1


_main_before_metamorphic = main


def main(argv=None) -> int:  # noqa: F811  (wraps the v0.1 entry point; adds the 'metamorphic' subcommand)
    args = sys.argv[1:] if argv is None else list(argv)
    if args and args[0] == "metamorphic":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _metamorphic_main(args[1:])
    if not args or args[0] in ("-h", "--help"):
        try:
            return _main_before_metamorphic(args)
        finally:
            print("\nadditional subcommand:\n  metamorphic         post-success metamorphic tests of a benchmark's "
                  "success predicate (python -m veritask metamorphic -h)")
    return _main_before_metamorphic(args)
