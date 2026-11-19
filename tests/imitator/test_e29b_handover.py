"""E29b fork A: DualHandover-v1 with contract revision imsim_dual_handover_v2 (protocol v3 + imsim v3.2 signals).

    source research/env.sh; source research/scripts/e29b/env_e29b.sh
    $PY -B -m pytest -q -p no:cacheprovider tests/imitator/test_e29b_handover.py               # fast
    IMSIM_SLOW=1 $PY -B -m pytest -q -p no:cacheprovider tests/imitator/test_e29b_handover.py  # + scripted CPU episodes
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # repository root
PROJECT = ROOT
sys.path.insert(0, ROOT)

import imitator  # noqa: E402
from imitator.verify import handover as HO  # noqa: E402
from imitator.verify import v31  # noqa: E402
from _paths import DATA, REPO, TMP as _TMP, repo_path  # noqa: E402,F401
TMP = _TMP

SLOW = os.environ.get("IMSIM_SLOW", "") not in ("", "0")
MANIFEST = os.path.join(DATA, "e29b_contract_manifest.json")
V2 = "imsim_dual_handover_v2"


def _contract():
    return imitator.get_task("DualHandover-v1").load_contract()


def test_v2_is_default_and_valid():
    from veritask.adapters.protocol_v3 import check_contract_v3
    spec = imitator.get_task("DualHandover-v1")
    assert spec.default_contract == V2 and "dual_handover_v1" in spec.contracts
    c = _contract()
    assert c.task_id == V2 and check_contract_v3(c) == []
    assert c.settle["max_seconds"] == 5.0 and c.require_released
    names = {x.name for x in c.counterexamples}
    assert set(spec.scripts) <= names and {x.name for x in c.counterexamples if x.expected} == {"handover"}
    assert v31.retreat_cfg(c) is not None
    assert spec.recorder.endswith("make_recorder_v32") and spec.judge == "imitator.verify.v32:judge"


def test_v2_frozen_in_manifest():
    m = json.load(open(MANIFEST))
    assert V2 in m and m[V2].get("frozen_at"), "imsim_dual_handover_v2 not frozen"
    p = imitator.get_task("DualHandover-v1").contract_path()
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == m[V2]["sha256"]
    assert m[V2].get("revision_of") == "imsim_dual_handover_v1"


def _ho_traces(L, R, T, dz):
    n = len(L)
    return dict(armL_obj_impulse_Ns=np.asarray(L, float), armR_obj_impulse_Ns=np.asarray(R, float),
                table_obj_impulse_Ns=np.asarray(T, float), obj_bottom_dz_m=np.asarray(dz, float), phase=np.zeros(n))


def test_v2_derive_uses_v1_signal_names():
    c = _contract()
    assert HO.handover_args(c) == dict(carry_steps=5, airborne_dz_m=0.003)
    on, off = np.ones(10), np.zeros(10)
    L = np.r_[off, on, on, off]; R = np.r_[off, off, on, on]; T = np.r_[on, off, off, off]   # left, both, right
    dz = np.r_[off, np.full(30, 0.05)]
    tr = HO.make_derive(c)(_ho_traces(L, R, T, dz))
    assert tr["handover_done"].max() == 1 and tr["handover_done"][:30].max() == 0
    L2 = np.r_[off, on, off, off]; R2 = np.r_[off, off, off, on]
    T2 = T.copy(); T2[25] = 1; dz2 = dz.copy(); dz2[25] = 0.0  # set down on the table between the carries
    assert HO.make_derive(c)(_ho_traces(L2, R2, T2, dz2))["handover_done"].max() == 0


def test_recorder_v32_signals_cover_contract():
    env = imitator.make("DualHandover-v1", "dual_ur5_wsg", sim="cpu", render="none")
    rec = HO.make_recorder_v32(env)
    rec.reset(seed=0)
    u = env.unwrapped
    assert abs(u.block_density - 250.0) < 1e-9 and u.block_friction == 1.0
    acts = {k: ag.home_action() for k, ag in zip(u.single_action_space.spaces.keys(), u.agent.agents)}
    for _ in range(2):
        rec.step(acts)
    c = _contract()
    need = {"phase", "armL_obj_impulse_Ns", "armR_obj_impulse_Ns", "table_obj_impulse_Ns", "obj_bottom_dz_m"}
    for t in c.outcome:
        if t.metric.startswith("trace.") and t.args.get("signal") != "handover_done":
            need |= set(t.args.get("signals", [t.args["signal"]]))
    need |= set(c.release["contact_signals"]) | set(c.release["enclosure_signals"])
    need |= set(c.settle["quiet_max"]) | {s for g in c.settle["quiet_disp"] for s in g["signals"]} | {c.settle["robot_speed_signal"]}
    missing = need - set(rec.traces)
    assert not missing, missing
    assert all(len(rec.traces[k]) == 2 for k in need)
    assert rec.traces["obj_enclosure_m"][-1] >= 0.9 and rec.traces["grip_cmd_gap_m"][-1] == 0.0   # arms at home
    env.close()


@pytest.mark.skipif(not SLOW, reason="set IMSIM_SLOW=1 (scripted episodes)")
def test_slow_expert_accepted_with_phases_and_table_transfer_rejected():
    from imitator.data.phase import CONTACT, FREE, get_phase
    from imitator.planning import handover as H
    env = imitator.make("DualHandover-v1", "dual_ur5_wsg", sim="cpu", render="none")
    rec = HO.make_recorder_v32(env)
    rec.reset(seed=0)
    seen = []
    orig = rec.step

    def step(a):
        seen.append((get_phase(rec, 0), get_phase(rec, 1)))
        return orig(a)
    rec.step = step
    log = H.run(rec, "handover")
    rec.step = orig
    assert log["status"] == "ok", log
    assert {p for p, _ in seen} == {FREE, CONTACT} and {p for _, p in seen} == {FREE, CONTACT}
    c = _contract()
    j = rec.judge(c, derive=HO.make_derive(c))
    assert j["success"] is True, (j["binding_term"], j["terms"])
    assert j["terms"]["trace.max:handover_done"] == 1.0
    env.close()
    out = os.path.join(TMP, "test_e29b_handover")
    r = imitator.evaluate("DualHandover-v1", "dual_ur5_wsg", "table_transfer", seeds=[0], out_dir=out, verbose=False)
    row = r["rows"][0]
    assert row["executed"] and row["verdict"] is False and row["terms"]["trace.max:handover_done"] == 0.0
