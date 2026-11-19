"""Tests for veritask.metamorphic (card B1). No simulator: a toy 2-D world with a gripper.

usage: python -B tests/test_metamorphic.py
"""
import contextlib, io, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import REPO  # noqa: E402
ROOT = os.path.join(REPO, "veritask")   # package dir (tasks/ lives here)
sys.path.insert(0, REPO)
from veritask import cli  # noqa: E402
from veritask.metamorphic import Adapter, Protocol, Rollout, classify, run_episode, summarize  # noqa: E402
from veritask.metamorphic.calibrate import choose_dv  # noqa: E402
from _paths import DATA, REPO, TMP as _TMP, repo_path  # noqa: E402,F401
TMP = _TMP

TMP = os.path.join(_TMP, "test_metamorphic")
os.makedirs(TMP, exist_ok=True)


class Toy(Adapter):
    """Object on a shelf at x in [0.45, 0.55], z = 0.10 (floor z = 0). Predicate: |x - 0.5| < 0.07 and |z - 0.10| < 0.02.
    A closed gripper within 3 cm of the object holds it; an unheld object not over the shelf falls to the floor;
    over the shelf it rests on it. dt_scale is irrelevant in this world (M7 must not flip)."""
    benchmark = "toy"
    control_freq = 10.0
    has_policy = False

    def __init__(self):
        self.s = None

    def _init(self, frozen=False):
        self.s = dict(obj=np.array([0.0, 0.0]), tcp=np.array([0.0, 0.3]), closed=False, held=False, frozen=frozen)
        self.last = dict(tcp=self.s["tcp"].copy(), grip=-1.0)

    def _phys(self):
        s = self.s
        if s["closed"] and np.linalg.norm(s["tcp"] - s["obj"]) < 0.03 and not s["frozen"]:
            s["held"] = True
        if not s["closed"]:
            s["held"] = False
        if s["held"]:
            s["obj"] = s["tcp"].copy()
        elif not s["frozen"]:
            s["obj"][1] = 0.10 if 0.45 <= s["obj"][0] <= 0.55 and s["obj"][1] >= 0.10 else 0.0

    def step(self, a):
        tgt = np.asarray(a["tcp"], float)
        d = tgt - self.s["tcp"]
        n = np.linalg.norm(d)
        self.s["tcp"] = tgt if n <= 0.05 else self.s["tcp"] + d / n * 0.05
        self.s["closed"] = a["grip"] > 0
        self.last = dict(tcp=tgt.copy(), grip=a["grip"])
        self._phys()

    def success(self):
        x, z = self.s["obj"]
        return bool(abs(x - 0.5) < 0.07 and abs(z - 0.10) < 0.02)

    def object_poses(self):
        return {"obj": np.array([self.s["obj"][0], 0.0, self.s["obj"][1], 1, 0, 0, 0])}

    def tcp_positions(self):
        return {"arm": np.array([self.s["tcp"][0], 0.0, self.s["tcp"][1]])}

    def set_hold_reference(self):
        self.ref = self.last["tcp"].copy()

    def hold_action(self, gripper="keep"):
        return dict(tcp=self.ref, grip=self.last["grip"] if gripper == "keep" else -1.0)

    def tcp_action(self, targets):
        p = targets["arm"]
        return dict(tcp=np.array([p[0], p[2]]), grip=-1.0)

    def nudge(self, names, dv):
        if not self.s["held"] and not self.s["frozen"]:
            self.s["obj"][0] += 0.005 * np.sign(dv[0] or 1)
        return True

    def rollout(self, ep):
        self._init()
        acts, succ = [], []
        for a in ep["actions"]:
            self.step(a); acts.append(a); succ.append(self.success())
        return Rollout(success=succ, official=any(succ), official_semantics="any", actions=acts,
                       meta=dict(replay_is_original=True))

    def goto(self, ep, ro, t):
        self._init()
        for a in ro.actions[: t + 1]:
            self.step(a)

    def replay(self, ep, ro, freeze=False, dt_scale=1.0):
        self._init(frozen=freeze)
        succ = []
        for a in ro.actions:
            self.step(a); succ.append(self.success())
        return dict(success=succ, frozen=["obj"] if freeze else [])


def script(place_x, release):
    """go to the object, grasp, lift, carry to (place_x, 0.10), optionally release and leave."""
    A = []
    for z in np.linspace(0.3, 0.0, 7):
        A.append(dict(tcp=[0.0, z], grip=-1.0))
    A += [dict(tcp=[0.0, 0.0], grip=1.0)] * 2
    for x in np.linspace(0.0, place_x, 12):
        A.append(dict(tcp=[x, 0.15], grip=1.0))
    A += [dict(tcp=[place_x, 0.15], grip=1.0)] * 2      # let the speed-limited TCP arrive before descending
    A += [dict(tcp=[place_x, 0.10], grip=1.0)] * 2
    if release:
        A += [dict(tcp=[place_x, 0.10], grip=-1.0)] * 2 + [dict(tcp=[place_x, 0.3], grip=-1.0)] * 4
    return A


def run(ep_actions, cls="place"):
    return run_episode(Toy(), dict(actions=ep_actions), Protocol(nudge_dv_mps=0.1, retreat_speed_mps=0.5),
                       task="toy", episode_id=str(len(ep_actions)), task_class=cls, source="test")


def test_genuine_placement_survives():
    r = run(script(0.50, release=True))
    assert r["official"] and r["errors"] == [], r["errors"]
    P = r["points"]["first"]
    for i in ("M1", "M2", "M3", "M5"):
        assert P[i]["success"] is True, (i, P[i])
    assert r["M0"]["success_any"] is False and r["M6"]["success_any"] is False
    assert r["M7"]["half"]["success_any"] == r["M7"]["ref"]["success_any"]


def test_held_at_goal_fails_release():
    r = run(script(0.565, release=False))        # held beside the shelf edge: predicate true only while held
    assert r["official"]
    P = r["points"]["first"]
    assert P["M1"]["success"] is True
    assert P["M2"]["success"] is False             # released object falls to the floor
    assert r["M6"]["success_any"] is False


def test_summary_and_cli():
    rows = [run(script(0.50, True)), run(script(0.565, False))]
    rows[1]["episode"] = "held"
    s = summarize(rows, ("benchmark",))[0]
    assert s["official"]["k"] == 2
    assert s["primary"]["M2"]["k"] == 1 and s["primary"]["M2"]["n"] == 2
    assert s["primary"]["persistent"]["k"] == 1
    path = os.path.join(TMP, "rows.jsonl")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cli.main(["metamorphic", "summarize", path])
    assert rc == 0 and "toy" in buf.getvalue()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cli.main(["metamorphic", "classify", "put the bowl on the plate", "press the button", "lift the pot"])
    out = buf.getvalue().split("\n")
    assert out[0].startswith("place") and out[1].startswith("event") and out[2].startswith("hold"), out


def test_classify_and_calibration_rule():
    assert classify("Place the sphere into the shallow bin")[0] == "place"
    assert classify("Shake the bottle with proper arm")[0] == "hold"
    smp = [dict(dv=0.1, disp_m=0.002), dict(dv=0.1, disp_m=0.004), dict(dv=0.2, disp_m=0.008),
           dict(dv=0.2, disp_m=0.2), dict(dv=0.2, disp_m=0.009), dict(dv=0.3, disp_m=0.02), dict(dv=0.3, disp_m=None)]
    assert choose_dv(smp)["dv"] == 0.2


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
