"""E29b fork B1 tests: DualLiftBox-v1 and DualStack-v1 (registration, contracts, carry signal, env step; slow: one expert
and one counterexample episode per task on CPU).

    research/envs/imsim/bin/python -B -m pytest -q -p no:cacheprovider tests/imitator/test_e29b_dual_b1.py
    IMSIM_SLOW=1 research/envs/imsim/bin/python -B -m pytest -q -p no:cacheprovider tests/imitator/test_e29b_dual_b1.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import OrderedDict

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # repository root
PROJECT = ROOT
sys.path.insert(0, ROOT)

import imitator  # noqa: E402
from imitator.verify import dual_tasks_b1 as B1  # noqa: E402
from _paths import DATA, REPO, TMP as _TMP, repo_path  # noqa: E402,F401
TMP = _TMP

SLOW = os.environ.get("IMSIM_SLOW", "") not in ("", "0")
MANIFEST = os.path.join(DATA, "e29b_contract_manifest.json")
TASKS = ("DualLiftBox-v1", "DualStack-v1")


def test_registered_with_contract_behaviours():
    for t in TASKS:
        assert t in imitator.list_tasks()
        spec = imitator.get_task(t)
        assert len(spec.embodiments) >= 3 and spec.default_embodiment in spec.embodiments
        c = spec.load_contract()
        names = {x.name: x.expected for x in c.counterexamples}
        assert set(spec.scripts) == set(names), t
        assert sum(1 for v in names.values() if v is False) >= 3 and names["expert"] is True
        assert c.settle["max_seconds"] == 5.0 and c.settle["quiet_max"] == {"objs_speed": 0.02}


def test_contracts_frozen_hash():
    if not os.path.exists(MANIFEST):
        pytest.skip("no manifest yet")
    m = json.load(open(MANIFEST))
    for t in TASKS:
        spec = imitator.get_task(t)
        c = spec.load_contract()
        if c.task_id not in m:
            pytest.skip(f"{c.task_id} not frozen yet")
        h = hashlib.sha256(open(spec.contract_path(), "rb").read()).hexdigest()
        assert m[c.task_id]["sha256"] == h, f"{c.task_id} changed after its freeze"


def _tr(n, **sig):
    base = {k: np.zeros(n) for k in ("arm0_cubeA_impulse_Ns", "arm1_cubeA_impulse_Ns", "table_cubeA_impulse_Ns",
                                      "arm0_cubeB_impulse_Ns", "arm1_cubeB_impulse_Ns", "table_cubeB_impulse_Ns")}
    base["cubeA_bottom_dz_m"] = np.zeros(n); base["cubeB_bottom_dz_m"] = np.zeros(n)
    base.update({k: np.asarray(v, float) for k, v in sig.items()})
    return base


def test_carry_signal_cases():
    n = 30
    air = np.r_[np.zeros(5), np.full(10, 0.05), np.zeros(15)]
    touch = (air > 0).astype(float)
    tr = _tr(n, cubeA_bottom_dz_m=air, arm0_cubeA_impulse_Ns=touch)
    s = B1.carry_signal(tr, "cubeA", 0)
    assert s[:9].max() == 0 and s[9:].min() == 1.0              # 5th consecutive carry step = index 9
    assert B1.carry_signal(tr, "cubeA", 1).max() == 0           # carried by the other arm
    tr2 = _tr(n, cubeA_bottom_dz_m=air, arm0_cubeA_impulse_Ns=touch, arm1_cubeA_impulse_Ns=touch)
    assert B1.carry_signal(tr2, "cubeA", 0).max() == 0          # both arms touch it: not 'by the left arm alone'
    short = np.r_[np.zeros(5), np.full(4, 0.05), np.zeros(21)]
    tr3 = _tr(n, cubeA_bottom_dz_m=short, arm0_cubeA_impulse_Ns=(short > 0).astype(float))
    assert B1.carry_signal(tr3, "cubeA", 0).max() == 0          # 4 steps < carry_steps
    tr4 = _tr(n, cubeA_bottom_dz_m=air, arm0_cubeA_impulse_Ns=touch, table_cubeA_impulse_Ns=touch)
    assert B1.carry_signal(tr4, "cubeA", 0).max() == 0          # dragged along the table


def test_stack_contract_rejects_one_arm_traces():
    """synthetic traces: perfect end state, but B carried by the left arm -> rejected by carry_B_by_arm1"""
    spec = imitator.get_task("DualStack-v1")
    c = spec.load_contract()
    der = B1.make_stack_derive(c)
    tr = der(_tr(10, cubeB_bottom_dz_m=np.full(10, 0.05), arm0_cubeB_impulse_Ns=np.ones(10)))
    assert tr["carry_B_by_arm1"].max() == 0 and tr["carry_A_by_arm0"].max() == 0


@pytest.mark.parametrize("task", TASKS)
def test_make_and_step_cpu(task):
    env = imitator.make(task, "dual_ur5_wsg", sim="cpu", render="none")
    obs, info = env.reset(seed=3)
    u = env.unwrapped
    acts = OrderedDict((k, ag.home_action()) for k, ag in zip(u.single_action_space.spaces, u.agent.agents))
    for _ in range(3):
        obs, rew, term, trunc, info = env.step(acts)
    assert "success" in info and not bool(info["success"][0])
    env.close()


@pytest.mark.skipif(not SLOW, reason="set IMSIM_SLOW=1 (scripted episodes)")
@pytest.mark.parametrize("task,cx", [("DualLiftBox-v1", "one_end_lift"), ("DualStack-v1", "side_by_side")])
def test_slow_expert_and_counterexample(task, cx):
    out = os.path.join(TMP, "test_e29b_b1", task)
    r = imitator.evaluate(task, "dual_ur5_wsg", "expert", seeds=[0], out_dir=out, verbose=False, experiment="E29b-test")
    assert r["rows"][0]["verdict"] is True, r["rows"][0].get("binding")
    r = imitator.evaluate(task, "dual_ur5_wsg", cx, seeds=[0], out_dir=out, verbose=False, experiment="E29b-test")
    assert r["rows"][0]["verdict"] is False
