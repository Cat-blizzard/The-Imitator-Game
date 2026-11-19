"""E29b tests: v3.2 verification layer and the ManiSkill Panda tasks.

    source research/env.sh; source research/scripts/e29b/env_e29b.sh
    $PY -B -m pytest -q -p no:cacheprovider tests/imitator/test_e29b_panda.py            # fast
    IMSIM_SLOW=1 $PY -B -m pytest -q -p no:cacheprovider tests/imitator/test_e29b_panda.py   # + one expert episode per task
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
from imitator.verify import v31, v32  # noqa: E402
from _paths import DATA, REPO, TMP as _TMP, repo_path, needs_stock_tasks  # noqa: E402,F401
TMP = _TMP

SLOW = os.environ.get("IMSIM_SLOW", "") not in ("", "0")
MANIFEST = os.path.join(DATA, "e29b_contract_manifest.json")
PANDA_TASKS = ("PickCube-v1", "StackCube-v1", "PushCube-v1", "PullCube-v1", "PlaceSphere-v1", "StackPyramid-v1",
               "LiftPegUpright-v1", "PlugCharger-v1", "PegInsertionSide-v1", "PullCubeTool-v1")
E27_REF = {"PegInsertionSide-v1": ("peg_insertion_side_v1", "peg_insertion_side_v2.json"),
           "PullCubeTool-v1": ("pull_cube_tool_v1", "pull_cube_tool_v2.json")}


# ------------------------------------------------------------------------------------------------ registry
@needs_stock_tasks()
def test_panda_tasks_registered_and_contracts_match():
    from veritask.adapters.protocol_v3 import check_contract_v3
    for t in PANDA_TASKS:
        spec = imitator.get_task(t)
        assert spec.embodiments == ("panda",) and spec.kind == "rigid"
        c = spec.load_contract()
        assert check_contract_v3(c) == [], t
        names = {x.name for x in c.counterexamples}
        assert set(spec.scripts) == names, (t, set(spec.scripts) ^ names)
        assert sum(1 for x in c.counterexamples if x.expected) == 1
        assert sum(1 for x in c.counterexamples if not x.expected) >= 3
        assert c.settle["max_seconds"] == 5.0                      # E24c fix 4
        quiet = set(c.settle["quiet_max"])
        assert quiet == {"objs_speed"}                             # E24c fix 1: position-based speed only


@needs_stock_tasks()
def test_panda_contracts_frozen_hashes():
    if not os.path.exists(MANIFEST):
        pytest.skip("no manifest yet")
    m = json.load(open(MANIFEST))
    for t in PANDA_TASKS:
        spec = imitator.get_task(t)
        c = spec.load_contract()
        if c.task_id not in m:
            pytest.skip(f"{c.task_id} not frozen yet")
        h = hashlib.sha256(open(spec.contract_path(), "rb").read()).hexdigest()
        assert h == m[c.task_id]["sha256"], t


@needs_stock_tasks()
def test_e27_tasks_reference_e27_contracts():
    """E27's frozen v2 contracts are recorded (path + prereg hash) as the reference and do not pass imsim's v3 check"""
    from veritask.adapters.protocol_v3 import check_contract_v3
    from veritask.spec.contract import TaskContract
    e27 = os.path.join(os.environ.get("IMITATOR_RESEARCH_DIR", ""), "results/e27_unbundle")
    if not os.path.exists(os.path.join(e27, "prereg.json")):
        pytest.skip("needs the E27 pre-registration of the research archive (IMITATOR_RESEARCH_DIR)")
    pre = json.load(open(os.path.join(e27, "prereg.json")))["part2"]["contracts"]
    for task, (key, fname) in E27_REF.items():
        c = imitator.get_task(task).load_contract()
        h = pre[key]["sha256"]
        assert h in c.source["reference"] and fname in c.source["reference"]
        p = os.path.join(e27, "contracts", fname)
        assert hashlib.sha256(open(p, "rb").read()).hexdigest() == h
        assert check_contract_v3(TaskContract.load(p)) != []          # case 2: not usable unchanged under protocol v3


# ------------------------------------------------------------------------------------------------ v3.2 logic
def _traces(n_pol=30, n_set=40, **over):
    n = n_pol + n_set
    tr = {k: np.zeros(n) for k in ("cube_x", "cube_y", "cube_z", "objs_speed", "cube_bottom_dz_m", "robot_qvel_max",
                                    "robot_cube_impulse_Ns", "grip_cmd_gap_m", "cube_tilt_deg")}
    tr["goal_dist_xy_m"] = np.full(n, 0.05)
    tr["cube_enclosure_m"] = np.ones(n)
    tr["phase"] = np.r_[np.zeros(n_pol), np.ones(n_set)]
    tr["bench_success"] = np.ones(n)
    for k, v in over.items():
        tr[k] = np.asarray(v, float) if np.ndim(v) else np.full(n, float(v))
    return tr


@needs_stock_tasks()
def test_v32_push_contract_logic():
    c = imitator.get_task("PushCube-v1").load_contract()
    ok = {v31.RETREAT_METRIC: 0.0}
    assert v31.judge_traces(c, _traces(), 20.0, ok)["success"] is True
    assert v31.judge_traces(c, _traces(), 20.0, {v31.RETREAT_METRIC: 0.02})["success"] is False   # retreat moved it
    assert v31.judge_traces(c, _traces(), 20.0, {})["success"] is None                            # aftermath missing
    lifted = _traces(cube_bottom_dz_m=np.r_[np.zeros(10), np.full(5, 0.05), np.zeros(55)])
    assert v31.judge_traces(c, lifted, 20.0, ok)["success"] is False                             # carried, not pushed
    assert v31.judge_traces(c, _traces(robot_cube_impulse_Ns=0.01), 20.0, ok)["success"] is False  # still touching
    assert v31.judge_traces(c, _traces(goal_dist_xy_m=0.12), 20.0, ok)["success"] is False
    # position-based speed: a moving cube never becomes quiescent within the 5 s cap -> reject
    n = 30 + 120
    moving = _traces(n_set=120, objs_speed=0.05)
    j = v31.judge_traces(c, moving, 20.0, ok)
    assert j["success"] is False and j["binding_term"] == "protocol.quiescence"


def test_v32_free_opening_and_calibration():
    class V:
        def __init__(self, k): self.k = k
        def calib_key(self): return self.k
    assert v32.free_opening(V("panda"), 0.04) == pytest.approx(0.08)
    assert v32.free_opening(V("panda"), -0.01) == 0.0
    cal = v32.grip_calib()
    for rec in ("ur5_wsg", "arx_x5", "piper", "xarm6_robotiq"):
        assert rec in cal
        o = np.asarray(cal[rec]["opening"])
        assert o.max() > 0.07 and o.min() < 0.005                 # spans open .. closed


def test_v32_contract_helpers():
    s = v32.settle_block(["a", "b"])
    assert s["max_seconds"] == 5.0 and s["quiet_max"] == {"objs_speed": 0.02} and len(s["quiet_disp"]) == 2
    r = v32.retreat_term(0.01)
    assert r["metric"] == v31.RETREAT_METRIC and r["args"]["retreat_m"] == 0.03


@needs_stock_tasks()
def test_panda_env_registered_and_step():
    env = imitator.make("PushCube-v1", "panda", sim="cpu", render="none")
    obs, info = env.reset(seed=0)
    u = env.unwrapped
    assert u.dual.mount_poses()[0].p[0] == pytest.approx(-0.615, abs=1e-3)
    a = np.r_[u.agent.robot.get_qpos()[0, :7].cpu().numpy(), 1.0]
    env.step(a)
    assert "success" in env.step(a)[-1]
    env.close()


# ------------------------------------------------------------------------------------------------ slow: episodes
@pytest.mark.skipif(not SLOW, reason="set IMSIM_SLOW=1 (scripted episodes)")
@pytest.mark.parametrize("task", PANDA_TASKS)
@needs_stock_tasks()
def test_slow_panda_expert_episode(task):
    out = os.path.join(TMP, "test_e29b_panda", task)
    r = imitator.evaluate(task, "panda", "expert", seeds=[0], out_dir=out, verbose=False, experiment="E29b-test")
    row = r["rows"][0]
    assert row.get("executed") and row["verdict"] is True, (task, row.get("binding"), row.get("plan_status"))
