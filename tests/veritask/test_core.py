"""Smoke tests for the veritask kernel (no simulator needed).  usage: python -B tests/test_core.py"""
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import REPO  # noqa: E402
ROOT = os.path.join(REPO, "veritask")   # package dir (tasks/ lives here)
sys.path.insert(0, REPO)
from veritask.spec.contract import TaskContract
from veritask.outcomes.metrics import measure
from veritask.audit.credibility import Episode, confusion, counterexample_pass_rate, by_margin, auroc, cross_engine_agreement
from veritask.calib.agreement import rank_agreement, mmrv, ppi_mean
from veritask.audit import static_lint
from veritask.judges.harness import parse_yes_no

R = os.environ.get("IMITATOR_RESEARCH_DIR", "")   # cross-engine towel results (research archive, optional)


def test_contract_on_real_engine_results():
    import pytest
    if not os.path.exists(f"{R}/results/xengine"):
        pytest.skip("needs the cross-engine results of the research archive (IMITATOR_RESEARCH_DIR)")
    c = TaskContract.load(f"{ROOT}/tasks/towel_fold_diagonal.json")
    out = {}
    for eng in ("newton", "mujoco4", "superdex"):
        for var in ("expert", "drop_midway"):
            z = np.load(f"{R}/results/xengine/{eng}/{var}.npz")
            m = {"cloth.corner_distance": measure("cloth.corner_distance", verts=z["end"], a=0, c=624),
                 "cloth.footprint_ratio": measure("cloth.footprint_ratio", verts=z["end"], verts_init=z["init"]),
                 "cloth.frac_above": measure("cloth.frac_above", verts=z["end"], height=0.05)}
            v = c.verdict(m, released=True)
            out[(eng, var)] = v
            assert v["success"] == (var == "expert"), (eng, var, v)
    margins = {f"{k[0]}/{k[1]}": (round(v["min_norm_margin"], 3), v["binding_term"]) for k, v in out.items()}
    print("contract verdicts OK; min margins per (engine, variant):", margins)


def test_unknown_when_missing():
    c = TaskContract.load(f"{ROOT}/tasks/towel_fold_diagonal.json")
    assert c.verdict({"cloth.corner_distance": 0.01}, released=True)["success"] is None


def test_credibility_metrics():
    eps = [Episode("expert", True, True, margin=0.03)] * 8 + [Episode("hover", True, False, margin=-0.2)] * 10 + [Episode("no_op", False, False, margin=-0.3)] * 10
    c = confusion(eps); assert abs(c["false_success_rate"] - 0.5) < 1e-9
    assert counterexample_pass_rate(eps)["hover"]["rate"] == 1.0
    assert auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert cross_engine_agreement({"a": {"x": True, "y": True}, "b": {"x": True, "y": False}})["sim_uncertain"] == ["b"]
    by_margin(eps)


def test_calibration_stats():
    sim = {"p1": 0.9, "p2": 0.6, "p3": 0.3}; real = {"p1": 0.7, "p2": 0.5, "p3": 0.2}
    r = rank_agreement(sim, real); assert r["kendall_tau"] == 1.0 and r["mmrv"] == 0.0
    real2 = {"p1": 0.4, "p2": 0.5, "p3": 0.2}; assert mmrv(sim, real2) > 0
    rng = np.random.default_rng(0)
    f_u = rng.binomial(1, 0.8, 2000); f_p = rng.binomial(1, 0.8, 40); y_p = np.where(rng.random(40) < 0.85, f_p, 1 - f_p) * (rng.random(40) < 0.9)
    p = ppi_mean(f_u, f_p, y_p); assert p["ci"][0] < p["estimate"] < p["ci"][1]
    print("PPI example:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items() if k != "ci"}, "ci", tuple(round(x, 3) for x in p["ci"]))


def test_static_lint_on_imitator():
    rows = [static_lint.lint(f) for f in sorted(glob.glob(os.path.join(REPO, "mani_skill/envs/tasks/tabletop/dual_tasks/_0*.py")))]
    flagged = {r["task"] for r in rows if r and r["unused_conditions"]}
    assert "_020_pour_kettle" in flagged and "_050_press_juicer" in flagged and len(flagged) == 12, flagged


def test_parse_yes_no():
    assert parse_yes_no("YES.") is True and parse_yes_no("no") is False and parse_yes_no("maybe") is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
