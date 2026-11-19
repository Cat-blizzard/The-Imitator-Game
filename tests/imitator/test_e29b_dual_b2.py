"""E29b fork-B2 tasks: DualPegInHole-v1, DualDrawer-v1 (registration, frozen contracts, recorder signals, one smoke
episode per task on CPU).

    source research/env.sh; source research/scripts/e29b/env_e29b.sh
    $PY -B -m pytest -q -p no:cacheprovider tests/imitator/test_e29b_dual_b2.py                 # fast
    IMSIM_SLOW=1 $PY -B -m pytest -q -p no:cacheprovider tests/imitator/test_e29b_dual_b2.py    # + one expert episode each
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # repository root
PROJECT = ROOT
sys.path.insert(0, ROOT)

import imitator  # noqa: E402
from imitator.tasks import get_task  # noqa: E402
from _paths import DATA, REPO, TMP as _TMP, repo_path  # noqa: E402,F401
TMP = _TMP

SLOW = os.environ.get("IMSIM_SLOW", "") not in ("", "0")
MANIFEST = os.path.join(DATA, "e29b_contract_manifest.json")
TASKS = {"DualPegInHole-v1": ("imsim_dual_peg_in_hole_v1", {"expert", "hold_peg", "peg_at_entrance", "block_on_table", "no_op"}),
         "DualDrawer-v1": ("imsim_dual_drawer_v1", {"expert", "pull_without_stabilising", "open_small", "hold_handle", "no_op"})}


@pytest.mark.parametrize("task", sorted(TASKS))
def test_registered(task):
    assert task in imitator.list_tasks()
    spec = get_task(task)
    cid, behs = TASKS[task]
    assert spec.default_contract == cid
    assert set(spec.scripts) == behs
    assert len(spec.embodiments) >= 3
    C = spec.load_contract()
    assert C.task_id == cid
    assert {c.name for c in C.counterexamples} == behs
    assert sum(1 for c in C.counterexamples if not c.expected) >= 3
    assert C.settle["max_seconds"] == 5.0


@pytest.mark.parametrize("task", sorted(TASKS))
def test_contract_frozen(task):
    """the contract file matches its SHA-256 in the E29b manifest (frozen before the final evaluation)"""
    spec = get_task(task)
    cid = TASKS[task][0]
    if not os.path.exists(MANIFEST) or cid not in json.load(open(MANIFEST)):
        pytest.skip("contract not frozen yet")
    h = hashlib.sha256(open(spec.contract_path(), "rb").read()).hexdigest()
    assert json.load(open(MANIFEST))[cid]["sha256"] == h


@pytest.mark.parametrize("task", sorted(TASKS))
def test_env_and_signals(task):
    """the env builds on CPU, the recorder produces every signal the contract reads, the bench flag is off at reset"""
    from imitator.tasks import _ref
    spec = get_task(task)
    env = imitator.make(task, "dual_ur5_wsg", sim="cpu", render="none")
    rec = _ref(spec.recorder)(env)
    rec.reset(seed=0)
    log = spec.script("no_op")(rec)
    assert log["status"] == "ok"
    C = spec.load_contract()
    for t in C.outcome:
        s = (t.args or {}).get("signal")
        if s:
            assert s in rec.traces, s
    for s in C.release["contact_signals"] + C.release["enclosure_signals"]:
        assert s in rec.traces, s
    assert rec.traces["bench_success"][-1] == 0.0
    env.close()


@pytest.mark.skipif(not SLOW, reason="slow: set IMSIM_SLOW=1")
@pytest.mark.parametrize("task", sorted(TASKS))
def test_expert_episode(task):
    res = imitator.evaluate(task, "dual_ur5_wsg", "expert", seeds=[0], out_dir=os.path.join(
        TMP, "e29b_b2_test_out", task), verbose=False, experiment="E29b-test")
    r = [x for x in res["rows"] if "error" not in x][0]
    assert r["plan_status"] == "ok"
    assert r["verdict"] is True
