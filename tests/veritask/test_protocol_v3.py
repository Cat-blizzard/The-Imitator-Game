"""Tests for evaluation protocol v3 (adapters/protocol_v3.py, maniskill_v3.py, maniskill_v3_tasks.py), the v3 contracts
and the schema-0.3 / dispatch changes (spec/contract.py, adapters/base.py, outcomes/metrics.py).

Pure tests need numpy + scipy (+ gymnasium for the recorder pieces; skipped without it). Simulator tests (development
seed 100) run only where the right mani_skill is importable:
  research/envs/msup/bin/python -B tests/test_protocol_v3.py   -> PokeCube: hold A from pd_joint_vel stops the robot; window found
  research/envs/ms2/bin/python  -B tests/test_protocol_v3.py   -> TwoRobotPlaceCupPlate: enclosure clearance grasped vs open
"""
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import REPO  # noqa: E402
ROOT = os.path.join(REPO, "veritask")   # package dir (tasks/ lives here)
sys.path.insert(0, REPO)
from veritask.spec.contract import TaskContract, validate_dict
from veritask.adapters import protocol_v3 as P
from veritask.adapters.base import evaluate_traces
from veritask.outcomes.metrics import measure

V3 = ["press_juicer_v3", "place_cup_plate_v3", "poke_cube_v3"]


def _load(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TaskContract.load(f"{ROOT}/tasks/{name}.json")


SETTLE = dict(hold="A", max_seconds=3.0, window_seconds=1.0, quiet_steps=10, quiet_max={"speed": 0.02},
              quiet_disp=[{"signals": ["x"], "max": 0.001}], robot_speed_signal="rq", robot_speed_max=0.01)


def _contract(outcome, require_released=False, settle=None, release=None):
    d = dict(task_id="t", instruction="x", outcome=outcome, require_released=require_released, protocol="v3",
             settle=dict(SETTLE, **(settle or {})), counterexamples=[dict(name="c", description="d", expected=False)])
    if release is not None:
        d["release"] = release
    return TaskContract.from_dict(d, validate=False)


def _term(metric, op, thr, signal, **args):
    return dict(metric=metric, op=op, threshold=thr, args=dict(signal=signal, **args), real_measurement="test")


def _traces(n_policy, settle: dict, policy: dict | None = None):
    """settle: signal -> array over the settle steps; policy: signal -> array over the policy steps (default zeros)"""
    n_s = len(next(iter(settle.values())))
    tr = {}
    for k, v in settle.items():
        p = np.zeros(n_policy) if not policy or k not in policy else np.asarray(policy[k], float)
        tr[k] = np.r_[p, np.asarray(v, float)]
    tr["phase"] = np.r_[np.zeros(n_policy), np.ones(n_s)]
    tr["bench_success"] = np.zeros(n_policy + n_s)
    return tr


def _still(n, **over):
    s = dict(speed=np.zeros(n), x=np.zeros(n), rq=np.zeros(n), d=np.full(n, 0.01))
    s.update({k: np.asarray(v, float) for k, v in over.items()})
    return s


# ------------------------------------------------------------------------------------------------ window
def test_window_after_quiescence():
    c = _contract([_term("trace.window_max", "<", 0.05, "d", window=20)])
    cfg = P.SettleConfig.from_contract(c)
    # transient: object moving for 6 settle steps (C28 / P23), then still -> quiet at settle step 15, window 16..35
    sp = np.r_[np.full(6, 0.2), np.zeros(54)]
    x = np.r_[np.linspace(0, 0.03, 7)[1:], np.full(54, 0.03)]
    tr = _traces(50, _still(60, speed=sp, x=x))
    w = P.find_window(tr, cfg, 50)
    assert w["status"] == "ok" and w["quiet_at"] == 50 + 15 and w["window"] == (66, 85), w
    v, _, info = P.evaluate_traces_v3(c, tr)
    assert v["success"] is True and info["window"] == [66, 85] and info["settle_steps_used"] == 36
    # a violation during the transient does not count; one inside the window does
    tr["d"][50:56] = 0.2
    assert P.evaluate_traces_v3(c, tr)[0]["success"] is True
    tr["d"][70] = 0.051
    assert P.evaluate_traces_v3(c, tr)[0]["success"] is False


def test_not_quiescent_and_incomplete():
    c = _contract([_term("trace.window_max", "<", 0.05, "d", window=20)])
    # robot never at rest (creep under hold B): rejected, with the reason
    tr = _traces(30, _still(40, rq=np.full(40, 0.05)))
    v, _, info = P.evaluate_traces_v3(c, tr)
    assert v["success"] is False and info["window_status"] == "not_quiescent" and "protocol_reject" in v
    # a creeping object (speed under the bound, but moving > 1 mm per 10 steps): never quiet
    tr = _traces(30, _still(40, speed=np.full(40, 0.01), x=np.linspace(0, 0.02, 40)))
    assert P.evaluate_traces_v3(c, tr)[1:] and P.evaluate_traces_v3(c, tr)[0]["success"] is False
    # quiet only from settle step 31 on: the window would end after the cap -> rejected
    tr = _traces(30, _still(60, speed=np.r_[np.full(31, 0.5), np.zeros(29)]))
    assert P.find_window(tr, P.SettleConfig.from_contract(c), 30)["status"] == "not_quiescent"
    # quiet at the latest admissible point (settle index 39) -> accepted
    tr = _traces(30, _still(60, speed=np.r_[np.full(30, 0.5), np.zeros(30)]))
    w = P.find_window(tr, P.SettleConfig.from_contract(c), 30)
    assert w["status"] == "ok" and w["quiet_at"] == 30 + 39
    # recorder stopped early: unknown, not a verdict
    tr = _traces(30, _still(12))
    v, _, info = P.evaluate_traces_v3(c, tr)
    assert v["success"] is None and info["window_status"] == "incomplete"
    # no settle at all
    tr = dict(d=np.zeros(30), phase=np.zeros(30), bench_success=np.zeros(30))
    assert P.evaluate_traces_v3(c, tr)[0]["success"] is None


def test_window_disp_catches_creep_inside_window():
    c = _contract([_term("trace.window_disp", "<=", 0.002, "x", signals=["x"], window=20)])
    x = np.r_[np.zeros(10), np.linspace(0, 0.004, 20), np.full(10, 0.004)]   # quiet 10 steps, then a 4 mm creep in the window
    tr = _traces(20, _still(40, x=x))
    v = P.evaluate_traces_v3(c, tr)[0]
    assert v["success"] is False and abs(v["terms"][0]["value"] - 0.004) < 1e-9
    assert measure("trace.window_disp", traces=dict(a=[0, 0, 3], b=[0, 4, 4]), signal="a", signals=["a", "b"], window=3) == 5.0


def test_release_contact_and_enclosure():
    rel = dict(contact_signals=["imp"], enclosure_signals=["enc"], enclosure_margin_m=0.002, max_impulse_Ns=0.0)
    c = _contract([_term("trace.window_max", "<", 0.05, "d", window=20)], require_released=True, release=rel)
    far = np.full(40, P.NOT_ENCLOSED_M)
    ok = _traces(20, _still(40, imp=np.zeros(40), enc=far), policy=dict(imp=np.ones(20), enc=np.zeros(20)))
    v, _, info = P.evaluate_traces_v3(c, ok)
    assert v["success"] is True and v["released"] is True and info["released_steps"] == 20
    # open gripper around the object with 7.5 mm total clearance (E13 C23): released
    tr = dict(ok); tr["enc"] = np.r_[np.zeros(20), np.full(40, 0.0075)]
    assert P.evaluate_traces_v3(c, tr)[0]["released"] is True
    # fingers closed around it without force (E10 C1): enclosed -> not released
    tr = dict(ok); tr["enc"] = np.r_[np.zeros(20), np.full(40, 0.0001)]
    v = P.evaluate_traces_v3(c, tr)[0]
    assert v["released"] is False and v["success"] is False
    # one contact step in the window is enough; contact before the window is not
    tr = dict(ok); tr["imp"] = np.r_[np.ones(20), np.ones(5), np.zeros(35)]
    assert P.evaluate_traces_v3(c, tr)[0]["released"] is True
    tr["imp"][45] = 1e-6
    assert P.evaluate_traces_v3(c, tr)[0]["released"] is False
    # missing release signals -> unknown
    tr = _traces(20, _still(40))
    assert P.evaluate_traces_v3(c, tr)[0]["success"] is None
    # the release block can be overridden (re-scoring with another margin)
    tr = dict(ok); tr["enc"] = np.r_[np.zeros(20), np.full(40, 0.003)]
    assert P.evaluate_traces_v3(c, tr)[0]["released"] is True
    assert P.evaluate_traces_v3(c, tr, release=dict(rel, enclosure_margin_m=0.004))[0]["released"] is False


def test_hold_B_requires_robot_at_rest():
    c = _contract([_term("trace.window_max", "<", 0.05, "d", window=20)])
    rq = np.r_[np.zeros(15), np.full(5, 0.02), np.zeros(20)]        # robot moves inside the window (quiescence already passed)
    tr = _traces(20, _still(40, rq=rq))
    vA = P.evaluate_traces_v3(c, tr, hold="A")[0]; vB = P.evaluate_traces_v3(c, tr, hold="B")[0]
    assert vA["success"] is True and vB["success"] is False and vB["binding_term"] == "protocol.robot_at_rest"
    assert P.evaluate_traces_v3(c, _traces(20, _still(40)), hold="B")[0]["success"] is True


def test_event_terms_read_until_window_end():
    c = _contract([_term("trace.count_true", ">=", 1, "contact"), _term("trace.max", "<=", 0.005, "push")])
    contact = np.zeros(60); contact[5] = 1
    tr = _traces(20, _still(40, contact=contact[20:], push=np.zeros(40)), policy=dict(contact=contact[:20], push=np.zeros(20)))
    assert P.evaluate_traces_v3(c, tr)[0]["success"] is True
    # steps recorded after the window (a longer settle) are not read
    tr2 = _traces(20, _still(50, contact=np.zeros(50), push=np.r_[np.zeros(45), np.full(5, 0.01)]), policy=dict(contact=contact[:20]))
    assert P.evaluate_traces_v3(c, tr2)[0]["success"] is True
    tr2["push"][-25] = 0.01
    assert P.evaluate_traces_v3(c, tr2)[0]["success"] is False


def test_protocol_checks():
    assert P.check_contract_v3(_contract([_term("trace.final", "<", 1, "d", window=10)]))
    assert P.check_contract_v3(_contract([_term("trace.window_max", "<", 1, "d", window=10)]))       # window != judged window
    assert P.check_contract_v3(_contract([_term("trace.window_max", "<", 1, "d", window=20)], settle=dict(max_seconds=1.0)))
    assert any("enclosure" in p for p in P.check_contract_v3(_contract([_term("trace.window_max", "<", 1, "d")], require_released=True,
                                                                         release=dict(contact_signals=["imp"]))))
    try:
        P.evaluate_traces_v3(_contract([_term("trace.final", "<", 1, "d", window=10)]), _traces(5, _still(40)))
    except P.ProtocolError:
        pass
    else:
        raise AssertionError("protocol violation not raised")


def test_schema_and_dispatch():
    base = json.load(open(f"{ROOT}/tasks/poke_cube_v3.json"))
    assert validate_dict(base) == []
    bad = dict(base, settle=dict(base["settle"], bogus=1))
    assert any("unknown field" in p for p in validate_dict(bad))
    assert any("hold" in p for p in validate_dict(dict(base, settle=dict(base["settle"], hold="C"))))
    juicer = json.load(open(f"{ROOT}/tasks/press_juicer_v3.json"))
    assert any("enclosure_signals" in p for p in validate_dict(dict(juicer, release=dict(contact_signals=["x"]))))
    assert any("not one of" in p for p in validate_dict(dict(base, protocol="v9")))
    # older files and protocols still validate
    for name in ("press_juicer_v1", "place_cup_plate_v2", "poke_cube_v2"):
        assert not [p for p in validate_dict(json.load(open(f"{ROOT}/tasks/{name}.json"))) if not p.startswith("warning:")]
    # base.evaluate_traces dispatches on contract.protocol (and accepts a hold override)
    c = _contract([_term("trace.window_max", "<", 0.05, "d", window=20)])
    tr = _traces(20, _still(40, rq=np.r_[np.zeros(15), np.full(5, 0.02), np.zeros(20)]))
    v, bench = evaluate_traces(c, tr)
    assert v["success"] is True and bench["protocol_info"]["protocol"] == "v3"
    assert evaluate_traces(c, tr, hold="B")[0]["success"] is False


def test_v3_contracts():
    try:
        from veritask.adapters.maniskill_v3_tasks import BINDINGS_V3
    except ImportError:
        BINDINGS_V3 = None; print("  (bindings not importable: signal check skipped)")
    for name in V3:
        with open(f"{ROOT}/tasks/{name}.json") as f:
            d = json.load(f)
        assert validate_dict(d) == [], (name, validate_dict(d))
        c = _load(name)
        assert c.protocol == "v3" and c.task_id == name and c.settle["hold"] == "A"
        assert P.check_contract_v3(c) == [], (name, P.check_contract_v3(c))
        assert any(x.name.startswith("E13:") for x in c.counterexamples) and any(x.name.startswith("E10:") for x in c.counterexamples)
        if BINDINGS_V3:
            b = BINDINGS_V3[name]()
            produced = set(b.probes())
            for k, p in b.pairs().items():
                produced |= {f"{k}_impulse_Ns", f"{k}_contact_acc"} | ({f"{k}_gap_m"} if p.gap else set()) | \
                    ({f"{k}_impulse_cum_Ns"} if p.cumulative else set()) | ({f"{k}_impulse_xy_Ns", f"{k}_impulse_xy_cum_Ns"} if getattr(p, "inplane", False) else set())
            for t in b.trackers():
                produced |= set(t.signal_names())
            need = {t.args["signal"] for t in c.outcome} | {s for t in c.outcome for s in t.args.get("signals", [])}
            need |= set(P.SettleConfig.from_contract(c).signals())
            need |= set(c.release.get("contact_signals", [])) | set(c.release.get("enclosure_signals", []))
            assert need <= produced, (name, sorted(need - produced))


# ------------------------------------------------------------------------------------------------ recorder pieces
def _gym():
    try:
        import gymnasium  # noqa: F401
        return True
    except ImportError:
        return False


class _Vec:
    def __init__(self, v):
        self.v = np.asarray(v, float)

    def __getitem__(self, i):
        import types
        return types.SimpleNamespace(cpu=lambda: types.SimpleNamespace(numpy=lambda: self.v[i[1]] if isinstance(i, tuple) else self.v))


class _Obj:
    def __init__(self):
        self.p, self.v = np.zeros(3), np.zeros(3)

    @property
    def pose(self):
        import types
        o = self
        return types.SimpleNamespace(p=_P(o.p))

    @property
    def linear_velocity(self):
        return _P(self.v)


class _P:
    def __init__(self, a):
        self.a = np.asarray(a, float)

    def __getitem__(self, idx):
        import torch
        t = torch.as_tensor(self.a[None])
        return t[idx]


def test_push_tracker():
    if not _gym():
        print("  skip (gymnasium not importable)"); return
    try:
        import torch  # noqa: F401
    except ImportError:
        print("  skip (torch not importable)"); return
    from veritask.adapters.maniskill_v3 import PushTracker
    cube = _Obj()
    u = type("U", (), {})()
    t = PushTracker("push", "robot", obj=lambda _u: cube, others=("peg",), rest_speed=0.005)
    tr = {k: [] for k in t.signal_names()}

    def step(substeps):
        t.sync(u, tr)
        for (dx, v, robot, peg) in substeps:
            cube.p = cube.p + np.array([dx, 0, 0]); cube.v = np.array([v, 0, 0])
            t.on_substep(u, dict(robot=robot, peg=peg))
        for k, val in t.end_step().items():
            tr[k].append(val)
    step([(0, 0, 0, 0)] * 5)
    assert tr["push_disp_m"][-1] == 0.0
    # a vertical load: contact, no motion (E13 P24)
    step([(0, 0, 0.2, 0)] * 5)
    assert tr["push_disp_m"][-1] == 0.0 and tr["push_active"][-1] == 1.0
    # contact ends with the cube at rest: the episode closes at 0
    step([(0, 0, 0, 0)] * 5)
    assert tr["push_active"][-1] == 0.0 and tr["push_total_m"][-1] == 0.0
    # a flick: 1 mm while touching, then 5 mm of coasting without contact until it stops -> 6 mm
    step([(0.001, 0.3, 0.1, 0)] + [(0.0012, 0.2, 0, 0)] * 4)
    step([(0.0002, 0.0, 0, 0)])
    assert abs(tr["push_total_m"][-1] - 0.0060) < 1e-9 and tr["push_active"][-1] == 0.0
    # the peg takes over: coasting attribution stops when the peg touches the cube
    step([(0.002, 0.1, 0.1, 0), (0.05, 0.1, 0, 0.3)])
    assert abs(tr["push_total_m"][-1] - (0.0060 + 0.052)) < 1e-9
    # back-and-forth while touching: net 0, but trace.max keeps the largest excursion
    step([(0.003, 0.1, 0.1, 0), (-0.003, 0.1, 0.1, 0), (0, 0, 0, 0)])
    assert max(tr["push_disp_m"]) >= 0.058 + 0.003 - 1e-9 and abs(tr["push_total_m"][-1] - 0.058) < 1e-9


# ------------------------------------------------------------------------------------------------ simulator
def _ms_version():
    try:
        import mani_skill
        return mani_skill.__version__
    except Exception:
        return None


def test_sim_pokecube_hold_A_from_velocity_mode():
    if _ms_version() != "3.0.1":
        print("  skip (needs mani_skill 3.0.1: research/envs/msup/bin/python)"); return
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from veritask.adapters.maniskill_v3_tasks import BINDINGS_V3
    c = _load("poke_cube_v3")
    env = gym.make("PokeCube-v1", obs_mode="state", control_mode="pd_joint_vel", render_mode=None, sim_backend="cpu",
                   render_backend="none", num_envs=1)
    rec = BINDINGS_V3["poke_cube_v3"]().recorder(env)
    rec.reset(seed=100)
    a = np.zeros(8, np.float32); a[0] = 0.08; a[3] = -0.05; a[7] = 1.0
    for _ in range(15):
        rec.step(a)                         # a slow joint-velocity creep; the policy ends while still moving
    u = env.unwrapped
    q_end = u.agent.robot.get_qpos()[0, :7].cpu().numpy().copy()
    assert rec.traces["robot_qvel_max"][-1] > 0.03
    win = rec.settle_v3(c, "A")
    q_after = u.agent.robot.get_qpos()[0, :7].cpu().numpy()
    assert win["status"] == "ok", win
    assert np.abs(q_after - q_end).max() < 0.01, np.abs(q_after - q_end).max()        # held where it stopped (< 0.6 deg)
    assert max(rec.traces["robot_qvel_max"][-20:]) < 0.01 and u.agent.control_mode == "pd_joint_pos"
    v, _, info = rec.evaluate_v3(c)
    assert info["window_status"] == "ok" and v["success"] is False      # nothing was poked: rejected on the terms
    rec.reset(seed=101)
    assert u.agent.control_mode == "pd_joint_vel"                       # the original mode is restored at the next reset
    env.close()


def test_sim_cups_enclosure():
    v = _ms_version()
    if v is None or v == "3.0.1":
        print("  skip (needs the Imitator fork: research/envs/ms2/bin/python)"); return
    scripts = os.path.join(os.environ.get("IMITATOR_RESEARCH_DIR", ""), "scripts")
    if not os.path.exists(os.path.join(scripts, "e10_place_cup_plate.py")):
        print("  skip (needs research/scripts/e10_place_cup_plate.py from the research archive: IMITATOR_RESEARCH_DIR)"); return
    sys.path.insert(0, scripts)
    os.environ.setdefault("E15_DEV", "1")
    from veritask.data.maniskill.session import ManiSkillSession
    from veritask.data.maniskill.tasks import TASK_CLASSES
    import e10_place_cup_plate as E    # stage runner (writes nothing at import)
    from veritask.adapters.maniskill_v3_tasks import BINDINGS_V3
    s = ManiSkillSession(TASK_CLASSES["place_cup_plate_v1"]())
    b = BINDINGS_V3["place_cup_plate_v3"]()
    s.rec = s.env = b.recorder(s.rec.env, v1_probes=s.task.probes(s.C)); s.v2_binding = b
    s.reset(100); s.hold(2)
    tr = s.rec.traces
    assert tr["cups_enclosure_clearance_m"][-1] == P.NOT_ENCLOSED_M
    E.run_stages(s, E.cup_names(1, upto="close")); s.hold(5)
    grasped = tr["cups_enclosure_clearance_m"][-1]
    E.run_stages(s, E.cup_names(1, frm="lift", upto="release")); s.hold(20)
    opened = tr["cups_enclosure_clearance_m"][-1]
    print(f"   enclosure clearance: grasped {grasped * 1000:.2f} mm, released with the gripper fully open around the cup {opened * 1000:.2f} mm")
    assert grasped <= 0.002 < opened < 0.02


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
