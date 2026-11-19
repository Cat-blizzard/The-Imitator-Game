"""E29b fork C tests: the cloth tasks (DualFoldTowel-v1 v2 revision, DualClothCoverBox-v1).

    research/envs/imsim/bin/python -B -m pytest -q -p no:cacheprovider tests/imitator/test_e29b_cloth.py
    IMSIM_SLOW=1 ... tests/imitator/test_e29b_cloth.py      # + one smoke batch per task (GPU PhysX + Newton VBD)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (PROJECT,):
    if p not in sys.path:
        sys.path.insert(0, p)

import imitator  # noqa: E402
from _paths import DATA, REPO, TMP as _TMP, repo_path  # noqa: E402,F401
TMP = _TMP

SLOW = os.environ.get("IMSIM_SLOW", "") not in ("", "0")
CLOTH_TASKS = ("DualFoldTowel-v1", "DualClothCoverBox-v1")


def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # pragma: no cover
        return False


def test_cloth_tasks_registered():
    for t in CLOTH_TASKS:
        assert t in imitator.list_tasks()
        spec = imitator.get_task(t)
        assert spec.kind == "soft" and spec.has_cloth
        assert spec.embodiments and spec.default_embodiment in spec.embodiments
        d = spec.describe()
        assert "expert" in d["behaviours"]
    fold = imitator.get_task("DualFoldTowel-v1")
    assert fold.default_contract == "dual_towel_fold_v2" and "dual_towel_fold_v1" in fold.contracts


def test_cloth_contracts_load_and_v3_valid():
    import warnings
    from veritask.adapters.protocol_v3 import check_contract_v3
    for t in CLOTH_TASKS:
        spec = imitator.get_task(t)
        for name in spec.contracts:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                c = spec.load_contract(name)
            assert check_contract_v3(c) == [], name
        c = spec.load_contract()
        names = {x.name for x in c.counterexamples}
        assert set(spec.scripts) == names, (t, names)
        assert sum(1 for x in c.counterexamples if x.expected is False) >= 3
        if c.task_id != "dual_towel_fold_v1":
            assert c.settle["max_seconds"] == 5.0
            assert "cloth_speed_p95_pos" in c.settle["quiet_max"]
            assert any(o.metric == "aftermath.retreat_object_disp" for o in c.outcome)


def test_fold_waypoints_phases_and_e23_timeline():
    from imitator.planning.fold import BEHAVIOURS, SHAPES, fold_waypoints, phase_at
    from imitator.soft.expert import interp_waypoints, total_time
    A, C = np.array([0.465, 0.125]), np.array([0.22, 0.12])
    w = fold_waypoints("expert", A, C, 0.03)
    assert abs(total_time(w) - 6.4) < 0.01                     # E23 timeline with the default parameters
    T = total_time(w)
    ph = [phase_at(w, t) for t in np.arange(0.025, T, 0.05)]
    assert ph[0] == 1 and ph[-1] == 0                          # descent is CONTACT, the final retreat FREE
    g = [interp_waypoints(w, t)[3] for t in np.arange(0.025, T, 0.05)]
    i_open = next(i for i, x in enumerate(g) if i > 20 and x < 0.01)
    assert all(p == 0 for p in ph[i_open + 1:])                # after the gripper has opened: FREE
    for b in set(BEHAVIOURS) | set(SHAPES):
        ww = fold_waypoints(b, A, C, 0.03)
        assert all(len(x) == 6 for x in ww)
    hold = fold_waypoints("hold_cloth", A, C, 0.03)
    assert all(x[3] == 1 for x in hold[2:])                    # never opens


def test_cloth_aftermath_patch_rule():
    """the cloth retreat value counts only particles near a retracting gripper (0 when no arm is near)"""
    from imitator import soft_eval
    assert soft_eval.PATCH_M == 0.05
    import inspect
    src = inspect.getsource(soft_eval.cloth_aftermath)
    assert "near_m" in src and "mask" in src


def _subprocess_json(code: str, timeout: int = 1800) -> dict:
    out = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=timeout,
                         env=dict(os.environ))
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads([l for l in out.stdout.splitlines() if l.startswith("RESULT ")][-1][len("RESULT "):])


@pytest.mark.skipif(not (SLOW and _cuda()), reason="set IMSIM_SLOW=1 (GPU PhysX + Newton)")
@pytest.mark.parametrize("task", CLOTH_TASKS)
def test_slow_cloth_smoke_expert(task):
    out = os.path.join(TMP, f"test_e29b_cloth_{task}")
    v = _subprocess_json(f"""
import json, shutil, imitator
shutil.rmtree({out!r}, ignore_errors=True)
r = imitator.evaluate({task!r}, "dual_ur5_wsg", "expert", seeds=[0], num_envs=2, out_dir={out!r}, verbose=False)
print("RESULT " + json.dumps([dict(v=x["verdict"], win=x["window_status"], aft=(x.get("aftermath") or {{}}).get("value"))
                              for x in r["rows"]]))
""")
    assert len(v) == 2 and all(x["win"] in ("ok", "not_quiescent") for x in v), v
    assert any(x["v"] is True for x in v), v                   # an expert world is accepted (rate >= 90 % in E29b)
