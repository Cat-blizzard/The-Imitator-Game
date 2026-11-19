"""Tests for veritask v0.1 additions (schema validation, evaluation semantics, shared adapter logic, CLI).
No simulator needed.  usage: python -B tests/test_v01.py
"""
import contextlib, copy, glob, io, json, math, os, sys, tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import REPO  # noqa: E402
ROOT = os.path.join(REPO, "veritask")   # package dir (tasks/ lives here)
sys.path.insert(0, REPO)
from veritask.spec.contract import TaskContract, ContractError, validate_dict, SCHEMA_VERSION
from veritask.spec import semantics as S
from veritask.adapters.base import evaluate_traces, save_traces, load_traces
from veritask.audit.primitives import PRIMITIVES, suggest
from veritask import cli
from _paths import DATA, REPO, TMP as _TMP, repo_path  # noqa: E402,F401
TMP = _TMP

TMP = _TMP
os.makedirs(TMP, exist_ok=True)


def _run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cli.main(argv)
    return rc, buf.getvalue()


def test_all_contracts_validate():
    files = sorted(glob.glob(f"{ROOT}/tasks/*.json"))
    assert len(files) >= 5
    for f in files:
        with open(f) as fh:
            errs = [p for p in validate_dict(json.load(fh)) if not p.startswith("warning:")]
        assert not errs, (f, errs)
        TaskContract.load(f)


def test_validation_catches_errors():
    with open(f"{ROOT}/tasks/poke_cube_v1.json") as fh:
        base = json.load(fh)
    cases = {
        "trace metric '": lambda d: d["outcome"][0]["args"].pop("signal"),
        "op '": lambda d: d["outcome"][0].__setitem__("op", "=="),
        "duplicate measurement keys": lambda d: d["outcome"].append(copy.deepcopy(d["outcome"][0])),
        "finite number": lambda d: d["outcome"][0].__setitem__("threshold", float("nan")),
        "unknown field": lambda d: d["outcome"][0].__setitem__("tolerance", 0.1),
        "missing required field 'outcome'": lambda d: d.pop("outcome"),
        "unsupported schema_version": lambda d: d.__setitem__("schema_version", "9.9"),
        "counterexample names must be unique": lambda d: d["counterexamples"].append(copy.deepcopy(d["counterexamples"][0])),
    }
    for needle, mutate in cases.items():
        d = copy.deepcopy(base); mutate(d)
        probs = validate_dict(d)
        assert any(needle in p and not p.startswith("warning:") for p in probs), (needle, probs)
        try:
            TaskContract.from_dict(d)
        except (ContractError, TypeError):
            pass
        else:
            raise AssertionError(f"from_dict accepted invalid contract ({needle})")
    d = copy.deepcopy(base); d["outcome"][0]["metric"] = "trace.does_not_exist"
    assert any(p.startswith("warning:") and "registry" in p for p in validate_dict(d))
    try:
        TaskContract.from_dict(d, strict=True)
    except ContractError:
        pass
    else:
        raise AssertionError("strict mode should reject unknown metric")


def test_roundtrip_save_load():
    c = TaskContract.load(f"{ROOT}/tasks/press_juicer_v1.json")
    assert c.schema_version == SCHEMA_VERSION
    p = os.path.join(TMP, "roundtrip_contract.json"); c.save(p)
    c2 = TaskContract.load(p)
    assert c2.to_dict() == c.to_dict()
    os.remove(p)


def test_semantics():
    tr = [0, 0, 1, 1, 0, 0]                      # transient success (e.g. held inside region, then lifted out)
    r = S.rescore(tr, ["any", "final", "window:2", "persist:2", "persist:3"])
    assert r == {"any": True, "final": False, "window:2": False, "persist:2": True, "persist:3": False}, r
    assert S.first_true(tr) == 2 and S.first_true([0, 0]) is None and S.longest_run(tr) == 2
    settled = [0, 1, 1, 1, 1]
    assert S.rescore(settled, ["any", "final", "window:3"]) == {"any": True, "final": True, "window:3": True}
    m = S.rescore_many([tr, settled, [0, 0, 0]], ["any", "window:3"])
    assert m["n"] == 3 and m["modes"]["any"]["successes"] == 2 and m["modes"]["window:3"]["successes"] == 1
    assert m["modes"]["window:3"]["over_credit_by_reference"] == 1
    lo, hi = S.wilson(5, 10); assert 0.2 < lo < 0.5 < hi < 0.8
    for bad in ("window", "window:0", "persist:x", "sometimes"):
        try:
            S.parse_mode(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def _old_maniskill_evaluate(contract, traces, released_signal="released"):
    """The pre-v0.1 TraceRecorder.evaluate body, kept here to check the refactor is behaviour-preserving."""
    from veritask.outcomes.metrics import measure
    m = {}
    for t in contract.outcome:
        if t.metric.startswith("trace."):
            m[t.key] = measure(t.metric, traces=traces, **t.args)
    rel = traces.get(released_signal)
    released = bool(np.mean(rel[-10:]) > 0.5) if rel else None
    bs = traces.get("bench_success") or []
    return contract.verdict(m, released=released), dict(bench_success_final=bool(bs[-1]) if bs else None,
                                                          bench_success_any=bool(np.any(bs)) if bs else None)


def test_shared_evaluation_matches_old_adapter():
    rng = np.random.default_rng(0)
    for name in ("poke_cube_v1", "press_juicer_v1", "place_cup_plate_v1"):
        c = TaskContract.load(f"{ROOT}/tasks/{name}.json")
        sigs = {t.args["signal"] for t in c.outcome if "signal" in t.args} | {"released", "bench_success"}
        for _ in range(20):
            n = int(rng.integers(12, 60))
            traces = {s: list((rng.random(n) * rng.choice([0.05, 1.0, 30.0]) - rng.choice([0, 0.3])).round(3)) for s in sigs}
            traces["released"] = list((rng.random(n) > rng.random()).astype(float))
            traces["bench_success"] = list((rng.random(n) > 0.7).astype(float))
            assert evaluate_traces(c, traces) == _old_maniskill_evaluate(c, traces), name
    # new behaviour: a missing signal is 'unknown', not a crash
    c = TaskContract.load(f"{ROOT}/tasks/poke_cube_v1.json")
    v, _ = evaluate_traces(c, {"released": [1.0] * 10}, released_signal=None)
    assert v["success"] is None


def test_trace_files_and_cli_score():
    c = TaskContract.load(f"{ROOT}/tasks/press_juicer_v1.json")
    n = 40
    good = {"knob_qpos": np.linspace(0, -0.5, n), "robot_knob_contact": np.r_[np.zeros(10), np.ones(15), np.zeros(15)],
            "released": np.r_[np.zeros(25), np.ones(15)], "bench_success": np.ones(n)}
    hover = {"knob_qpos": np.zeros(n), "robot_knob_contact": np.zeros(n), "released": np.ones(n), "bench_success": np.ones(n)}
    paths = []
    for nm, tr in (("good", good), ("hover", hover)):
        p = os.path.join(TMP, f"trace_{nm}.npz"); save_traces(p, tr, {"behaviour": nm}); paths.append(p)
        tr2, meta = load_traces(p)
        assert meta == {"behaviour": nm} and set(tr2) == set(tr)
    rc, out = _run(["score", f"{ROOT}/tasks/press_juicer_v1.json", *paths, "--json", os.path.join(TMP, "score.json")])
    with open(os.path.join(TMP, "score.json")) as fh:
        rows = json.load(fh)
    assert rc == 0 and [r["contract_success"] for r in rows] == [True, False], out
    assert "benchmark-accepted but contract-rejected: 1" in out
    for p in paths + [os.path.join(TMP, "score.json")]:
        os.remove(p)


def test_cli_validate_lint_rescore_primitives():
    rc, out = _run(["validate", f"{ROOT}/tasks"])
    assert rc == 0 and out.count("OK") >= 5, out
    imitator = os.path.join(REPO, "mani_skill/envs/tasks/tabletop/dual_tasks")
    if os.path.isdir(imitator):
        j = os.path.join(TMP, "lint.json")
        rc, out = _run(["lint", *sorted(glob.glob(f"{imitator}/_0*.py")), "--json", j])
        with open(j) as fh:
            rows = json.load(fh)
        os.remove(j)
        assert sum("UNUSED_COND" in r["flags"] for r in rows) == 12, out[-400:]
        pk = [r for r in rows if r["task"] == "_020_pour_kettle"][0]
        assert "present_without_effect" in pk["suggested_primitives"]
    eps = os.path.join(TMP, "eps.jsonl")
    with open(eps, "w") as f:
        for model, tr in (("a", [0, 1, 1, 1]), ("a", [0, 1, 0, 0]), ("b", [0, 0, 0, 1]), ("b", [0, 0, 0, 0])):
            f.write(json.dumps({"model": model, "success_trace": tr}) + "\n")
    j = os.path.join(TMP, "rescore.json")
    rc, out = _run(["rescore", eps, "--group-by", "model", "--modes", "any,window:2", "--json", j])
    with open(j) as fh:
        res = {r["group"]["model"]: r for r in json.load(fh)}
    assert res["a"]["modes"]["any"]["successes"] == 2 and res["a"]["modes"]["window:2"]["successes"] == 1
    assert res["b"]["modes"]["window:2"]["over_credit_by_reference"] == 1   # success only at the very last step
    os.remove(eps); os.remove(j)
    assert suggest(["EVER_LATCH"])[0] in PRIMITIVES and "hold_in_goal_never_release" in suggest(["SEM_IN_NO_CONTACT"])
    rc, out = _run(["primitives", "--flags", "UNUSED_COND"]); assert rc == 0 and "bypass_required_tool" in out
    rc, out = _run(["metrics"]); assert "trace.final" in out and "cloth.corner_distance" in out



def test_robotwin_lint_matches_e5():
    root = os.path.join(os.path.dirname(ROOT), "research", "ext_data", "clones", "RoboTwin")
    ref = os.path.join(os.path.dirname(ROOT), "research", "results", "e5", "robotwin_lint.json")
    if not (os.path.isdir(root) and os.path.exists(ref)):
        print("  (skipped: RoboTwin clone or E5 result missing)"); return
    from veritask.audit.robotwin_lint import lint_robotwin
    new = lint_robotwin(root)
    with open(ref) as fh:
        old = json.load(fh)
    assert new["flag_counts"] == old["flag_counts"]
    assert {t["task"]: sorted(t["flags"]) for t in new["tasks"]} == {t["task"]: sorted(t["flags"]) for t in old["tasks"]}
    assert "poll_loop" in new["eval_semantics"]


def test_evidence_and_card():
    from veritask.audit.evidence import EvidenceRow, save, load
    from veritask.audit.card import summarize, render_markdown
    c = TaskContract.load(f"{ROOT}/tasks/press_juicer_v1.json")
    rows = []
    for eng in ("sapien-physx", "other-engine"):
        rows += [EvidenceRow(c.task_id, "press_knob", "control", True, bench=True, margin=1.4, engine=eng, seed=i) for i in range(5)]
        for b in ("hover_front_10cm", "hover_above_4cm", "touch_and_leave"):
            rows += [EvidenceRow(c.task_id, b, "counterexample", False, bench=True, margin=-1.0, engine=eng, seed=i) for i in range(5)]
    p = os.path.join(TMP, "ev.jsonl"); save(p, rows); back = load(p); os.remove(p)
    assert back == rows
    s = summarize(c, rows)
    assert s["level"] == 2 and s["counterexamples"]["contract_accepted"] == 0 and s["counterexamples"]["bench_accepted"] == 30
    assert "expert" in s["declared_not_executed"] and "no_op" in s["declared_not_executed"]
    # one accepted counterexample drops the card to L0
    s0 = summarize(c, rows + [EvidenceRow(c.task_id, "touch_and_leave", "counterexample", True, engine="sapien-physx")])
    assert s0["level"] == 0 and any("accepted" in w for w in s0["level_reasons"])
    # a single engine caps it at L1
    s1 = summarize(c, [r for r in rows if r.engine == "sapien-physx"])
    assert s1["level"] == 1
    md = render_markdown(c, s)
    assert "L2 engine-robust" in md and "hover_front_10cm" in md
    try:
        EvidenceRow(c.task_id, "x", "bogus_role", True)
    except ValueError:
        pass
    else:
        raise AssertionError("bad role accepted")


def test_protocol_v2_dispatch():
    """evaluate_traces / `veritask score` dispatch to protocol v2 and reproduce E11's stored verdicts."""
    from veritask.adapters.protocol_v2 import evaluate_traces_v2
    res = os.path.join(os.path.dirname(ROOT), "research", "results", "e11_contract_v2")
    ev = os.path.join(res, "evidence_poke_cube_v2.jsonl")
    if not os.path.exists(ev):
        print("  (skipped: E11 results missing)"); return
    c = TaskContract.load(f"{ROOT}/tasks/poke_cube_v2.json")
    assert c.protocol == "v1"            # the frozen E11 files predate the 'protocol' field: pass protocol="v2" explicitly
    rows = [json.loads(l) for l in open(ev)][:40]
    n = 0
    for r in rows:
        tr_path = r["extra"].get("trace")
        if tr_path and not os.path.isabs(tr_path):
            tr_path = os.path.join(os.path.dirname(ROOT), tr_path)      # stored relative to the project root
        if not tr_path or not os.path.exists(tr_path):
            continue
        tr, meta = load_traces(tr_path)
        v, flags = evaluate_traces(c, tr, protocol="v2", control_freq=meta.get("control_freq", 20.0))
        assert v["success"] == r["verdict"], (tr_path, v["success"], r["verdict"])
        assert flags["protocol_info"]["protocol"] == "v2"
        v2, _, _ = evaluate_traces_v2(c, tr, None, meta.get("control_freq", 20.0))
        assert v2 == v
        n += 1
    assert n >= 10, n
    # a v2 trace without a settle phase is 'unknown', never silently judged
    short = {k: np.asarray(x)[:5] for k, x in tr.items()}; short["phase"] = np.zeros(5)
    assert evaluate_traces(c, short, protocol="v2")[0]["success"] is None
    # schema 0.2: protocol / release fields validate; trace.final is an error under v2
    d = c.to_dict(); d["protocol"] = "v2"; d["release"] = {"contact_signals": ["a"], "gap_signals": ["b"], "gap_m": 0.005}
    assert not [p for p in validate_dict(d) if not p.startswith("warning:")]
    assert TaskContract.from_dict(d).protocol == "v2"
    d2 = copy.deepcopy(d); d2["outcome"][0]["metric"] = "trace.final"
    assert any("not allowed under protocol v2" in p for p in validate_dict(d2))
    d3 = copy.deepcopy(d); d3["release"]["bogus"] = 1
    assert any("release: unknown field" in p for p in validate_dict(d3))
    rc, out = _run(["score", f"{ROOT}/tasks/poke_cube_v2.json", tr_path, "--protocol", "v2"])
    assert rc == 0 and "unknown" not in out.lower().split("binding term")[1][:80], out

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
