"""A2 tests: the expert parameter spaces (imitator.planning.params) and the tuned-set hook.

    source research/env.sh; source research/scripts/e29b/env_e29b.sh
    $PY -B -m pytest -q -p no:cacheprovider tests/imitator/test_a2_params.py              # fast
    IMSIM_SLOW=1 $PY -B -m pytest -q -p no:cacheprovider tests/imitator/test_a2_params.py # + default-identity episodes
"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import needs_stock_tasks  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # repository root
PROJECT = ROOT
sys.path.insert(0, ROOT)

from imitator.planning import params as P  # noqa: E402

SLOW = os.environ.get("IMSIM_SLOW", "") not in ("", "0")
A2_SPACES = {"PlugCharger-v1", "PullCubeTool-v1", "DualCubePlace-v1:gripper", "DualCubePlace-v1:hand", "DualDrawer-v1",
             "DualFoldTowel-v1", "FoldTowel-v1", "StackCube-v1", "PushCube-v1"}


# ------------------------------------------------------------------------------------------------ spaces
def test_spaces_registered_and_bounded():
    assert set(P.SPACES) >= A2_SPACES
    for key, s in P.SPACES.items():
        assert 5 <= len(s.params) <= 15, key
        assert len(set(s.names)) == len(s.names), key
        for p in s.params:
            assert p.lo < p.hi and p.lo <= p.default <= p.hi, (key, p.name)
            if p.switch:
                assert (p.lo, p.hi, p.default) == (0.0, 1.0, 0.0), (key, p.name)
        d = s.defaults()
        assert s.clip(d) == d, key
        z = s.to_unit(d)
        assert np.all((z >= 0) & (z <= 1)), key
        back = s.from_unit(z)
        for n in s.names:
            assert back[n] == pytest.approx(d[n], abs=1e-9), (key, n)


def test_clip_rounds_integers_and_bounds():
    s = P.SPACES["PlugCharger-v1"]
    x = s.clip(dict(s.defaults(), refine_wait=7.6, insert_step=1.0))
    hi = {p.name: p.hi for p in s.params}
    assert x["refine_wait"] == 8 and x["insert_step"] == hi["insert_step"]


def test_space_for_embodiment_kinds():
    assert P.space_for("DualCubePlace-v1", "dual_ur5_wsg").key == "DualCubePlace-v1:gripper"
    assert P.space_for("DualCubePlace-v1", "dual_xarm6_allegro").key == "DualCubePlace-v1:hand"
    assert P.space_for("PlugCharger-v1", "panda").key == "PlugCharger-v1"
    assert P.space_for("PickCube-v1", "panda") is None


# ------------------------------------------------------------------------------------------------ tuned-set hook
def test_resolve_and_tuned_hook(monkeypatch):
    key = "StackCube-v1"
    fake = {f"{key}@panda": dict(params=dict(pre_wait=10), adopted=True),
            "PushCube-v1@panda": dict(params=dict(pre_wait=5), adopted=False)}
    monkeypatch.setattr(P, "_tuned_file", lambda: fake)
    monkeypatch.delenv("IMSIM_EXPERT_PARAMS", raising=False)
    assert P.resolve(key, None, "panda")["pre_wait"] == 10              # adopted set of this embodiment
    assert P.resolve(key, None, "dual_piper")["pre_wait"] == 0          # other embodiment: frozen defaults
    assert P.resolve(key)["pre_wait"] == 0                              # no embodiment, no bare entry
    assert P.resolve("PushCube-v1", None, "panda")["pre_wait"] == 0     # not adopted
    assert P.resolve(key, {"pre_wait": 3}, "panda")["pre_wait"] == 3    # explicit overrides win
    monkeypatch.setenv("IMSIM_EXPERT_PARAMS", "frozen")
    assert P.resolve(key, None, "panda")["pre_wait"] == 0
    monkeypatch.setenv("IMSIM_EXPERT_PARAMS", "tuned")
    assert P.resolve("PushCube-v1", None, "panda")["pre_wait"] == 5
    with pytest.raises(KeyError):
        P.resolve(key, {"no_such_parameter": 1})


def test_pv_reads_the_embodiment_of_the_env(monkeypatch):
    key = "PushCube-v1"
    monkeypatch.setattr(P, "_tuned_file", lambda: {f"{key}@panda": dict(params=dict(push_segs=3), adopted=True)})
    monkeypatch.delenv("IMSIM_EXPERT_PARAMS", raising=False)

    class U:
        imsim = dict(task=key, embodiment="panda")

    class Env:
        unwrapped = U()

    assert P.emb_of(Env()) == "panda"
    assert P.pv(None, key, Env())["push_segs"] == 3
    assert P.pv(None, key)["push_segs"] == 1
    full = P.SPACES[key].defaults()
    assert P.pv(full, key, Env()) is full                               # a full dict is used as given


def test_counterexample_variants_keep_frozen_parameters(monkeypatch):
    """a tuned set applies to the positive control only; counterexample variants keep the frozen parameters"""
    monkeypatch.setattr(P, "_tuned_file", lambda: {
        "StackCube-v1@panda": dict(params=dict(pre_wait=10), adopted=True),
        "DualCubePlace-v1:hand@dual_xarm6_allegro": dict(params=dict(close_cmd=0.9), adopted=True)})
    monkeypatch.delenv("IMSIM_EXPERT_PARAMS", raising=False)

    class Env:
        def __init__(self, emb):
            self.unwrapped = type("U", (), dict(imsim=dict(embodiment=emb)))()

    assert P.expert_variant("StackCube-v1") == "expert"
    assert P.expert_variant("DualCubePlace-v1:hand") == "release"
    assert P.pv(None, "StackCube-v1", Env("panda"), "expert")["pre_wait"] == 10
    assert P.pv(None, "StackCube-v1", Env("panda"), "cage")["pre_wait"] == 0
    assert P.pv(None, "DualCubePlace-v1:hand", Env("dual_xarm6_allegro"), "release")["close_cmd"] == 0.9
    assert P.pv(None, "DualCubePlace-v1:hand", Env("dual_xarm6_allegro"), "cage")["close_cmd"] == 1.0


def test_tuned_file_entries_valid():
    if not os.path.exists(P.TUNED_PATH):
        pytest.skip("no tuned_params.json yet")
    data = json.load(open(P.TUNED_PATH))
    for k, e in data.items():
        if k.startswith("_"):
            continue
        key = k.split("@")[0]
        assert key in P.SPACES, k
        s = P.SPACES[key]
        assert set(e["params"]) <= set(s.names), k
        assert isinstance(e.get("adopted"), bool), k
        for p in s.params:
            if p.name in e["params"]:
                assert p.lo <= e["params"][p.name] <= p.hi, (k, p.name)


# ------------------------------------------------------------------------------------------------ experts read the spaces
EXPERT_FUNCS = {("planning/panda_ms.py", "stack_cube"): ["StackCube-v1"],
                ("planning/panda_ms.py", "_push_like"): ["PushCube-v1"],
                ("planning/panda_ms.py", "plug_charger"): ["PlugCharger-v1"],
                ("planning/panda_ms.py", "pull_cube_tool"): ["PullCubeTool-v1"],
                ("planning/scripted.py", "pick_place"): ["DualCubePlace-v1:gripper", "DualCubePlace-v1:hand"],
                ("planning/dual_drawer.py", "*"): ["DualDrawer-v1"],
                ("planning/towel_params.py", "run_towel_p"): ["FoldTowel-v1"]}


def test_expert_parameter_names_exist():
    """every p["name"] / P["name"] / ctx.p["name"] read by a parameterised expert is a parameter of its space"""
    for (f, fn), keys in EXPERT_FUNCS.items():
        src = open(os.path.join(ROOT, "imitator", f)).read()
        for node in ast.walk(ast.parse(src)):
            if not isinstance(node, ast.FunctionDef) or (fn != "*" and node.name != fn):
                continue
            seg = ast.get_source_segment(src, node)
            used = set(re.findall(r'\b(?:p|P|ctx\.p)\["(\w+)"\]', seg))
            for k in keys:
                missing = used - set(P.SPACES[k].names)
                assert not missing, (f, node.name, k, sorted(missing))


def test_fold_space_defaults_equal_the_frozen_planner():
    from imitator.planning import fold
    d = P.SPACES["DualFoldTowel-v1"].defaults()
    for k, v in d.items():
        if k in fold.EXPERT:
            assert fold.EXPERT[k] == v, k
    assert d["grasp_dz"] == 0.0


# ------------------------------------------------------------------------------------------------ slow: identity
_IDENTITY = r'''
import hashlib, json, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import imitator
from imitator.tasks import get_task
from imitator.planning import params as P
task, mode = sys.argv[2], sys.argv[3]
env = imitator.make(task, "panda", sim="cpu", render="none", obs="state")
env.reset(seed=0)
acts = []
step = env.step
def rec(a):
    acts.append(np.asarray(a, dtype=np.float64).round(9).tolist())
    return step(a)
env.step = rec
if mode == "frozen":
    log = get_task(task).script("expert")(env)
else:
    log = P.expert_callable(task, "panda", P.space_for(task, "panda").defaults())(env)
print(json.dumps(dict(status=str(log.get("status")), n=len(acts),
                      h=hashlib.sha256(json.dumps(acts).encode()).hexdigest())))
'''


@pytest.mark.skipif(not SLOW, reason="set IMSIM_SLOW=1 (scripted episodes)")
@needs_stock_tasks()
@pytest.mark.parametrize("task", ["PushCube-v1", "StackCube-v1", "PullCubeTool-v1"])
def test_slow_defaults_reproduce_the_frozen_expert(task):
    """same actions from the TaskSpec expert (params=None, IMSIM_EXPERT_PARAMS=frozen) and from the explicit defaults,
    each in a fresh process (CPU PhysX carries state between episodes of one process)"""
    env = dict(os.environ, IMSIM_EXPERT_PARAMS="frozen")
    out = []
    for mode in ("frozen", "defaults"):
        r = subprocess.run([sys.executable, "-B", "-c", _IDENTITY, ROOT, task, mode], capture_output=True, text=True,
                           env=env, cwd=PROJECT, timeout=900)
        line = [x for x in r.stdout.splitlines() if x.startswith("{")]
        assert line, r.stderr[-2000:]
        out.append(json.loads(line[-1]))
    assert out[0]["status"] == "ok" and out[0] == out[1], out
