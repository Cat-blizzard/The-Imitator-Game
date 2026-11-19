"""Tests for evaluation protocol v2 (adapters/protocol_v2.py, maniskill_v2.py, maniskill_v2_tasks.py) and the v2 contracts.

Pure tests need numpy + scipy (+ gymnasium for the modules that import the ManiSkill recorder; those tests are skipped
without it). Simulator tests are skipped unless the right mani_skill is importable:
  research/envs/msup/bin/python -B tests/test_protocol_v2.py     -> PokeCube tests (substep hook, scene-API equality,
                                                                    gap vs independent distance, settle holds the command)
  research/envs/ms2/bin/python  -B tests/test_protocol_v2.py     -> TwoRobotPlaceCupPlate tests (tray floor map, cup base)
Simulator tests use development seeds (>= 100) only.
"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import REPO  # noqa: E402
ROOT = os.path.join(REPO, "veritask")   # package dir (tasks/ lives here)
sys.path.insert(0, REPO)
from veritask.spec.contract import TaskContract, validate_dict
from veritask.adapters import protocol_v2 as P
from veritask.adapters.base import evaluate_traces

V2 = ["press_juicer_v2", "place_cup_plate_v2", "poke_cube_v2"]


def _load(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TaskContract.load(f"{ROOT}/tasks/{name}.json")


def _contract(outcome, require_released=False, settle=1.0):
    return TaskContract.from_dict(dict(task_id="t", instruction="x", outcome=outcome, require_released=require_released,
                                       settle_seconds=settle, counterexamples=[dict(name="c", description="d", expected=False)]),
                                  validate=False)


def _term(metric, op, thr, signal, **args):
    return dict(metric=metric, op=op, threshold=thr, args=dict(signal=signal, **args), real_measurement="test")


def _traces(n_policy, n_settle, **sig):
    tr = {k: np.asarray(v, float) for k, v in sig.items()}
    tr["phase"] = np.r_[np.zeros(n_policy), np.ones(n_settle)]
    tr.setdefault("bench_success", np.zeros(n_policy + n_settle))
    for k, v in tr.items():
        assert len(v) == n_policy + n_settle, k
    return tr


# ------------------------------------------------------------------------------------------------ pure: semantics
def test_late_violation_is_caught():
    """E10 P1: the cube leaves the goal in the last 3 steps. A 10-step mean (v1) accepts, v2 rejects."""
    d = np.r_[np.full(57, 0.01), [0.03, 0.06, 0.069]]                       # policy ends 6.9 cm out
    c1 = _contract([_term("trace.final", "<", 0.05, "d", window=10)])
    assert evaluate_traces(c1, dict(d=d), released_signal=None)[0]["success"] is True
    c2 = _contract([_term("trace.window_max", "<", 0.05, "d", window=20)])
    tr = _traces(60, 20, d=np.r_[d, np.full(20, 0.08)])                       # it rests at 8 cm during the settle
    v, _, info = P.evaluate_traces_v2(c2, tr)
    assert v["success"] is False and info["n_policy"] == 60 and info["n_settle"] == 20
    tr2 = _traces(60, 20, d=np.r_[d, np.full(19, 0.01), [0.051]])            # one bad settle step is enough
    assert P.evaluate_traces_v2(c2, tr2)[0]["success"] is False


def test_end_state_terms_read_only_the_settle_phase():
    """E10 C4: a correct but late finish. The policy phase still shows the placing motion; the settle phase is clean."""
    speed = np.r_[np.full(50, 0.3), np.zeros(20)]
    c2 = _contract([_term("trace.window_max", "<=", 0.02, "speed", window=20)])
    assert P.evaluate_traces_v2(c2, _traces(50, 20, speed=speed))[0]["success"] is True
    # the same data judged the v1 way (20-step mean ending at the policy's last step) fails
    c1 = _contract([_term("trace.final", "<=", 0.02, "speed", window=20)])
    assert evaluate_traces(c1, dict(speed=speed[:50]), released_signal=None)[0]["success"] is False
    # default window = the whole settle phase
    c3 = _contract([_term("trace.window_max", "<=", 0.02, "speed")])
    assert P.evaluate_traces_v2(c3, _traces(50, 20, speed=speed))[0]["success"] is True
    assert P.evaluate_traces_v2(c3, _traces(50, 20, speed=np.r_[np.zeros(50), [0.5], np.zeros(19)]))[0]["success"] is False


def test_no_settle_no_verdict():
    c2 = _contract([_term("trace.window_max", "<", 0.05, "d", window=20)])
    for tr in (dict(d=np.zeros(40), bench_success=np.zeros(40)),            # no phase signal at all
               _traces(40, 10, d=np.zeros(50))):                             # settle too short
        v, _, info = P.evaluate_traces_v2(c2, tr)
        assert v["success"] is None and "unknown_reason" in info
    # a longer settle than required is truncated to the required steps: a violation after them is not judged
    tr = _traces(40, 30, d=np.r_[np.zeros(60), np.full(10, 1.0)])
    v, _, info = P.evaluate_traces_v2(c2, tr)
    assert v["success"] is True and info["n_settle"] == 20


def test_protocol_errors():
    for bad in ([_term("trace.final", "<", 0.05, "d", window=10)],           # mean
                [_term("trace.window_max", "<", 0.05, "d", window=40)]):     # window reaches back into the policy phase
        c = _contract(bad)
        assert P.check_contract_v2(c, 20.0)
        try:
            P.evaluate_traces_v2(c, _traces(50, 20, d=np.zeros(70)))
        except P.ProtocolError:
            pass
        else:
            raise AssertionError("protocol violation not raised")
    c = _contract([_term("trace.window_max", "<", 0.05, "d", window=20)], require_released=True)
    assert any("ReleaseRule" in p for p in P.check_contract_v2(c, 20.0))
    rule = P.ReleaseRule(["imp"], ["gap"], gap_m=0.005)
    c = _contract([_term("trace.window_min", ">", 0.004, "gap", window=20)], require_released=True)
    assert any("differs" in p for p in P.check_contract_v2(c, 20.0, rule))    # the gap is written once


def test_event_terms_read_the_whole_episode():
    contact = np.zeros(70); contact[12] = 1.0                                 # one contact step, long before the end
    c = _contract([_term("trace.count_true", ">=", 1, "contact"), _term("trace.max", "<=", 0.005, "cum")])
    assert P.evaluate_traces_v2(c, _traces(50, 20, contact=contact, cum=np.zeros(70)))[0]["success"] is True
    cum = np.r_[np.zeros(60), np.full(10, 0.02)]                              # a touch DURING the settle phase counts too
    assert P.evaluate_traces_v2(c, _traces(50, 20, contact=contact, cum=cum))[0]["success"] is False


def test_release_rule():
    rule = P.ReleaseRule(contact_signals=["imp"], gap_signals=["gap"])
    assert rule.gap_m == 0.005 and rule.max_impulse_Ns == 0.0
    c = _contract([_term("trace.window_min", ">", 0.005, "gap", window=20)], require_released=True)
    far = np.full(70, 0.04)
    ok = _traces(50, 20, gap=far, imp=np.r_[np.full(50, 0.3), np.zeros(20)])  # contact during the policy phase is fine
    v, _, info = P.evaluate_traces_v2(c, ok, release=rule)
    assert v["success"] is True and v["released"] is True and info["released_steps"] == 20
    # E10 C1: fingers still closed around the object at ~0 N: gap ~0 -> not released
    held = _traces(50, 20, gap=np.r_[far[:50], np.full(20, 0.0)], imp=np.zeros(70))
    v = P.evaluate_traces_v2(c, held, release=rule)[0]
    assert v["released"] is False and v["success"] is False
    # one settle step inside the gap, or one settle step with any contact impulse, is enough
    g = far.copy(); g[55] = 0.0049
    assert P.evaluate_traces_v2(c, _traces(50, 20, gap=g, imp=np.zeros(70)), release=rule)[0]["released"] is False
    imp = np.zeros(70); imp[69] = 1e-6
    assert P.evaluate_traces_v2(c, _traces(50, 20, gap=far, imp=imp), release=rule)[0]["released"] is False
    assert rule.step_value(dict(imp=0.0, gap=0.0051)) == 1.0 and rule.step_value(dict(imp=0.0, gap=0.005)) == 0.0
    # require_released with the release signals missing -> unknown, not True
    v = P.evaluate_traces_v2(c, _traces(50, 20, gap=far), release=rule)[0]
    assert v["success"] is None


def test_bench_flags_and_phase_counts():
    bs = np.r_[np.zeros(30), np.ones(10), np.zeros(10), np.ones(20)]          # true mid-episode, false at the policy end, true after settle
    tr = _traces(50, 20, d=np.zeros(70), bench_success=bs)
    assert P.phase_counts(tr) == (50, 20)
    b = P.bench_flags_v2(tr, 50)
    assert b == dict(bench_success_final=False, bench_success_any=True, bench_success_after_settle=True)
    ph = np.r_[np.zeros(10), np.ones(5), np.zeros(10), np.ones(20)]           # only the TRAILING run is the settle phase
    assert P.phase_counts(dict(phase=ph)) == (25, 20)


# ------------------------------------------------------------------------------------------------ pure: v2 contracts
def test_v2_contracts_validate_and_follow_the_protocol():
    import json
    try:
        from veritask.adapters.maniskill_v2_tasks import BINDINGS
    except ImportError:
        BINDINGS = None; print("  (bindings not importable here: release-rule consistency not checked)")
    for name in V2:
        with open(f"{ROOT}/tasks/{name}.json") as f:
            d = json.load(f)
        assert validate_dict(d) == [], (name, validate_dict(d))               # no errors AND no warnings
        c = _load(name)
        assert c.task_id == name and c.settle_seconds == 1.0
        assert all(t.metric != "trace.final" for t in c.outcome)
        for t in c.outcome:                                                    # every end-state window = the whole settle phase
            if t.metric in P.END_STATE_METRICS:
                assert t.args.get("window") == 20, (name, t.key)
        rule = BINDINGS[name]().release() if BINDINGS else (P.ReleaseRule(["x"], ["y"]) if c.require_released else None)
        probs = P.check_contract_v2(c, 20.0, rule)
        assert probs == [], (name, probs)
        names = [x.name for x in c.counterexamples]
        assert any(n.startswith("E10:") for n in names) and any(not x.expected for x in c.counterexamples)
        if BINDINGS:                                                           # every signal a term reads is produced by the binding
            b = BINDINGS[name]()
            from veritask.adapters.maniskill_v2 import TraceRecorderV2

            class _E:                                                          # minimal stand-in so the recorder can be built
                unwrapped = None
            produced = set(b.probes())
            for k, p in b.pairs().items():
                produced |= {f"{k}_impulse_Ns", f"{k}_contact_acc"} | ({f"{k}_gap_m"} if p.gap else set()) | \
                    ({f"{k}_impulse_cum_Ns"} if p.cumulative else set())
            missing = [t.args["signal"] for t in c.outcome if t.args["signal"] not in produced]
            assert not missing, (name, missing)
            if c.require_released:
                assert set(rule.contact_signals + rule.gap_signals) <= produced


def test_v2_contracts_on_synthetic_traces():
    """Each E10 defect, replayed on synthetic traces through the real v2 contract files."""
    n, s = 60, 20
    z, far = np.zeros(n + s), np.full(n + s, 0.04)
    # --- juicer: pressed and left pressed (accept) vs pressed then turned back (E10 J1) vs finger left on the dial (J7)
    c = _load("press_juicer_v2"); rule = P.ReleaseRule(["robot_knob_impulse_Ns"], ["fingers_knob_gap_m"])
    contact = z.copy(); contact[20:30] = 1
    good = _traces(n, s, knob_qpos=np.r_[np.zeros(20), np.linspace(0, -0.6, 10), np.full(50, -0.6)], robot_knob_contact_acc=contact,
                   fingers_knob_gap_m=np.r_[np.full(20, 0.04), np.zeros(15), np.full(45, 0.04)],
                   robot_knob_impulse_Ns=np.r_[np.zeros(20), np.full(10, 0.2), np.zeros(50)])
    assert P.evaluate_traces_v2(c, good, rule)[0]["success"] is True
    undone = dict(good); undone["knob_qpos"] = np.r_[good["knob_qpos"][:40], np.linspace(-0.6, 0.0, 10), np.zeros(30)]
    v = P.evaluate_traces_v2(c, undone, rule)[0]
    assert v["success"] is False and v["binding_term"] == "trace.window_max:knob_qpos"
    resting = dict(good); resting["fingers_knob_gap_m"] = np.r_[good["fingers_knob_gap_m"][:60], np.full(20, 0.0009)]
    v = P.evaluate_traces_v2(c, resting, rule)[0]
    assert v["success"] is False and v["released"] is False
    untouched = dict(good); untouched["robot_knob_contact_acc"] = z
    assert P.evaluate_traces_v2(c, untouched, rule)[0]["success"] is False
    # --- poke: late exit (E10 P1), sustained hand touch, cube on top of the peg
    c = _load("poke_cube_v2")
    pc = z.copy(); pc[30] = 1
    good = _traces(n, s, cube_goal_xy=np.full(n + s, 0.01), cube_speed=z, cube_lift_m=z, peg_cube_contact_acc=pc, robot_cube_impulse_cum_Ns=z)
    assert P.evaluate_traces_v2(c, good)[0]["success"] is True
    for sig, bad in (("cube_goal_xy", np.r_[np.full(n + s - 1, 0.01), [0.05]]),          # on the radius at the last step: '<' fails
                     ("cube_speed", np.r_[np.zeros(n), [0.03], np.zeros(s - 1)]),
                     ("cube_lift_m", np.full(n + s, 0.05)),
                     ("peg_cube_contact_acc", z),
                     ("robot_cube_impulse_cum_Ns", np.r_[np.zeros(10), np.full(n + s - 10, 0.0051)])):
        tr = dict(good); tr[sig] = bad
        assert P.evaluate_traces_v2(c, tr)[0]["success"] is False, sig
    tol = dict(good); tol["robot_cube_impulse_cum_Ns"] = np.full(n + s, 0.005)             # exactly at the tolerance: accepted
    assert P.evaluate_traces_v2(c, tol)[0]["success"] is True
    # --- cups: rim perch (E10 C2), never let go (C1), late knock (C3)
    c = _load("place_cup_plate_v2"); rule = P.ReleaseRule(["robot_cups_impulse_Ns"], ["fingers_cups_gap_m"])
    one = np.ones(n + s)
    good = _traces(n, s, cup1_tilt_deg=one, cup2_tilt_deg=one, cup1_bottom_dz_m=z + 2e-4, cup2_bottom_dz_m=z + 2e-4,
                   cup1_base_in_floor_frac=one, cup2_base_in_floor_frac=one, cup_max_speed=np.r_[np.full(n, 0.2), np.zeros(s)],
                   fingers_cups_gap_m=np.r_[np.zeros(n), np.full(s, 0.04)], robot_cups_impulse_Ns=np.r_[np.ones(n), np.zeros(s)])
    assert P.evaluate_traces_v2(c, good, rule)[0]["success"] is True
    for sig, bad in (("cup2_bottom_dz_m", z + 0.025), ("cup1_base_in_floor_frac", one * 0.6),
                     ("fingers_cups_gap_m", z), ("cup1_tilt_deg", np.r_[one[:-1], [30.0]]),
                     ("cup_max_speed", np.r_[np.zeros(n + s - 1), [0.3]]), ("robot_cups_impulse_Ns", z + 0.01)):
        tr = dict(good); tr[sig] = bad
        assert P.evaluate_traces_v2(c, tr, rule)[0]["success"] is False, sig


# ------------------------------------------------------------------------------------------------ pure: recorder pieces
class _Pt:
    def __init__(self, impulse, separation):
        self.impulse, self.separation = np.asarray(impulse, float), separation


class _Body:
    def __init__(self, entity):
        self.entity = entity


class _Contact:
    def __init__(self, e0, e1, pts):
        self.bodies, self.points = [_Body(e0), _Body(e1)], pts


def _need_gym():
    try:
        import gymnasium  # noqa: F401
        return True
    except ImportError:
        return False


def test_substep_accumulation():
    """A contact that exists only in substeps 2-3 of 5 is invisible to last-substep sampling and visible to v2."""
    if not _need_gym():
        print("  skip (gymnasium not importable)"); return
    from veritask.adapters.maniskill_v2 import Pair, SubstepContacts
    acc = SubstepContacts({"peg_cube": Pair(None, None, gap=True), "robot_cube": Pair(None, None)})
    acc.sets = {"peg_cube": ({"peg"}, {"cube"}), "robot_cube": ({"f1", "f2", "hand"}, {"cube"})}
    acc.gap_far = {"peg_cube": 0.04, "robot_cube": float("nan")}; acc.dt = 0.01

    class U:
        class scene:
            class px:
                cur = []

                @staticmethod
                def get_contacts():
                    return U.scene.px.cur
    table = _Contact("cube", "table", [_Pt([0, 0, 0.006], 0.0)])
    hit = _Contact("cube", "peg", [_Pt([0.02, 0, 0], -1e-4), _Pt([0.01, 0, 0], 0.0)])   # bodies in the other order: sign flips, norm same
    near = _Contact("peg", "cube", [_Pt([0, 0, 0], 0.003)])
    seq = [[table, near], [table, hit], [table, hit], [table, near], [table]]
    acc._begin()
    for cs in seq:
        U.scene.px.cur = cs; acc.on_substep(U)
    out = acc.end_step()
    assert abs(out["peg_cube_impulse_Ns"] - 0.06) < 1e-12 and out["peg_cube_n_sub"] == 2
    assert abs(out["peg_cube_max_sub_force_N"] - 3.0) < 1e-9 and out["peg_cube_last_sub_force_N"] == 0.0   # v1 would have seen nothing
    assert abs(out["peg_cube_gap_m"] - (-1e-4)) < 1e-12 and out["robot_cube_impulse_Ns"] == 0.0 and out["_n_substeps"] == 5
    # no contact point at all -> the gap is reported as the contact range ("farther than PhysX looks")
    acc._begin(); U.scene.px.cur = [table]; acc.on_substep(U)
    assert acc.end_step()["peg_cube_gap_m"] == 0.04
    # two links of side A touching B: sum of the per-link norms (as v1's force_sum), not the norm of the sum
    acc._begin()
    U.scene.px.cur = [_Contact("f1", "cube", [_Pt([0, 0.3, 0], 0)]), _Contact("cube", "f2", [_Pt([0, 0.3, 0], 0)])]
    acc.on_substep(U)
    assert abs(acc.end_step()["robot_cube_impulse_Ns"] - 0.6) < 1e-12


def test_tray_floor_and_cup_base_geometry():
    if not _need_gym():
        print("  skip (gymnasium not importable)"); return
    from veritask.adapters.maniskill_v2_tasks import CupBase, TrayFloor, _upper_hull_height

    def box(lo, hi):
        return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)
    # exact upper hull of a wedge: height rises linearly from 0.02 to 0.05 across x in [0, 0.03]
    wedge = np.array([[0, 0, 0], [0, 0.1, 0], [0.03, 0, 0], [0.03, 0.1, 0], [0, 0, 0.02], [0, 0.1, 0.02], [0.03, 0, 0.05], [0.03, 0.1, 0.05]], float)
    X, Y = np.meshgrid([0.0, 0.015, 0.03, 0.04], [0.05], indexing="ij")
    h = _upper_hull_height(wedge, X, Y).ravel()
    assert np.allclose(h[:3], [0.02, 0.035, 0.05], atol=1e-9) and np.isnan(h[3])
    # synthetic tray: 30 x 20 cm floor slab (top z = 0.024), 2.5 cm wide rim (top z = 0.05) all around
    parts = [box([-0.15, -0.10, 0], [0.15, 0.10, 0.024]),
             box([-0.15, -0.10, 0.024], [-0.125, 0.10, 0.05]), box([0.125, -0.10, 0.024], [0.15, 0.10, 0.05]),
             box([-0.15, -0.10, 0.024], [0.15, -0.075, 0.05]), box([-0.15, 0.075, 0.024], [0.15, 0.10, 0.05])]
    tray = TrayFloor(parts)
    assert abs(tray.floor_ref - 0.024) < 1e-9 and abs(tray.rim_top - 0.05) < 1e-9
    assert abs(tray.floor_area_m2 - 0.25 * 0.15) < 0.25 * 0.15 * 0.05          # inner floor 25 x 15 cm (2 mm grid)
    assert tray.on_floor(np.array([[0.0, 0.0], [0.11, 0.06]])).all()
    assert not tray.on_floor(np.array([[0.14, 0.0], [0.0, 0.09], [0.3, 0.0], [-0.2, -0.2]])).any()   # rim, rim, outside, outside
    assert tray.floor_height([0.0, 0.0]) == tray.floor_height([0.14, 0.0]) == tray.floor_ref
    # synthetic cup (up = local +y): base radius 3 cm, flaring to 4 cm at 10 cm, plus a handle well above the base band
    th = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    ring = lambda r, y: np.c_[r * np.cos(th), np.full(48, y), r * np.sin(th)]  # noqa: E731
    cup = CupBase([np.vstack([ring(0.03, 0.0), ring(0.04, 0.10)]), box([-0.005, 0.03, 0.04], [0.005, 0.07, 0.06])])
    assert abs(cup.base_radius_m - 0.03) < 0.001 and abs(np.linalg.norm(cup.samples[:, [0, 2]], axis=1).max() - 0.03) < 1e-9
    assert np.allclose(cup.samples[:, 1], 0.0) and len(cup.samples) > 300

    def frac(cx, cy):       # upright cup standing with its base centre at (cx, cy)
        return float(tray.on_floor(cup.samples[:, [0, 2]] + [cx, cy]).mean())
    assert frac(0.0, 0.0) == 1.0 and frac(0.125 - 0.031, 0.0) == 1.0           # fully on the floor, 1 mm from the rim
    assert 0.4 < frac(0.125, 0.0) < 0.6                                        # centred on the floor edge: about half
    assert frac(0.1375, 0.0) < 0.3 and frac(0.3, 0.0) == 0.0                   # perched on the rim (E10 C2) / off the tray
    edge = frac(0.125 - 0.03 + 0.006, 0.0)                                     # 6 mm of a 3 cm-radius base over the rim: ~95%
    assert 0.9 < edge < 1.0, edge


# ------------------------------------------------------------------------------------------------ simulator: PokeCube
def _mani_skill_version():
    try:
        import mani_skill
        return mani_skill.__version__
    except Exception:
        return None


def _convex_distance(VA, VB, iters=400):
    x = VA[0] - VB[0]
    for _ in range(iters):
        s = VA[int(np.argmin(VA @ x))] - VB[int(np.argmax(VB @ x))]
        g = float(x @ (x - s))
        if g <= 1e-14:
            break
        x = x + min(1.0, g / float((x - s) @ (x - s))) * (s - x)
    return float(np.linalg.norm(x))


def _mesh_gap(objs_a, objs_b):
    from veritask.adapters.maniskill_v2_tasks import _pose_Rp, collision_parts_local

    def parts(objs):
        out = []
        for o in objs:
            R, p = _pose_Rp(o)
            out += [(R @ V.T).T + p for V in collision_parts_local(o)]
        return out
    return min(_convex_distance(A, B) for A in parts(objs_a) for B in parts(objs_b))


def test_sim_pokecube_recorder():
    """Substep hook, equality with scene.get_pairwise_contact_impulses, PhysX gap vs an independent convex distance,
    and the settle phase (dev seed 100)."""
    if _mani_skill_version() != "3.0.1":
        print("  skip (needs mani_skill 3.0.1: research/envs/msup/bin/python)"); return
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import sapien
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
    from veritask.adapters.maniskill_v2_tasks import BINDINGS
    b = BINDINGS["poke_cube_v2"](); c = _load("poke_cube_v2")
    env = gym.make("PokeCube-v1", obs_mode="state", control_mode="pd_joint_pos", render_mode=None, sim_backend="cpu",
                   render_backend="none", num_envs=1)
    rec = b.recorder(env)
    rec.reset(seed=100)
    u = env.unwrapped
    lm = u.agent.robot.links_map
    api_pairs = [(lm["panda_leftfinger"], u.cube), (lm["panda_rightfinger"], u.cube), (lm["panda_hand"], u.cube), (u.peg, u.cube)]
    chk = dict(n=0, nz=0, worst=0.0)
    prev = u._after_simulation_step

    def hook():
        prev()
        chk["worst"] = max(chk["worst"], rec.verify_against_scene_api(api_pairs)); chk["n"] += 1
        chk["nz"] += int(any(float(np.linalg.norm(u.scene.get_pairwise_contact_impulses(x, y)[0].cpu().numpy())) > 0 for x, y in api_pairs))
    u._after_simulation_step = hook
    mp = PandaArmMotionPlanningSolver(rec, debug=False, vis=False, base_pose=u.agent.robot.pose, visualize_target_grasp_pose=False,
                                      print_env_info=False)
    cube = u.cube.pose.p[0].cpu().numpy(); q = u.agent.tcp.pose.sp.q
    mp.close_gripper()
    mp.move_to_pose_with_screw(sapien.Pose([cube[0], cube[1], 0.14], q))
    fingers = [lm["panda_leftfinger"], lm["panda_rightfinger"]]
    robot = [l for l in u.agent.robot.links if l._bodies[0].get_collision_shapes()]
    gaps = []
    for z in (0.075, 0.06, 0.05, 0.045):               # fingertips come down to a few mm above the cube top, then touch it
        mp.move_to_pose_with_screw(sapien.Pose([cube[0], cube[1], z], q))
        qpos = u.agent.robot.get_qpos()[0, :-2].cpu().numpy()
        for _ in range(15):
            rec.step(np.hstack([qpos, -1.0]))
        gaps.append((z, rec.traces["robot_cube_gap_m"][-1], _mesh_gap(robot, [u.cube]), rec.traces["robot_cube_impulse_Ns"][-1]))
    print("   tcp z, PhysX gap [mm], independent gap [mm], step impulse [N*s]:",
          [(z, round(g * 1e3, 3), round(m * 1e3, 3), round(j, 5)) for z, g, m, j in gaps])
    n_sub = int(u._sim_steps_per_control)
    assert chk["n"] == n_sub * len(rec.traces["phase"]) and n_sub == 5
    assert chk["nz"] > 0 and chk["worst"] < 1e-7, chk
    for z, g, m, j in gaps:
        if g < 0.039 and g > 0.0005:                    # inside PhysX's contact range and not touching: must agree within 0.5 mm
            assert abs(g - m) < 5e-4, (z, g, m)
        if j > 0:
            assert g < 5e-4
    assert any(0.0005 < g < 0.039 for _, g, _, _ in gaps), "no hover configuration inside the contact range was reached"
    assert rec.traces["robot_cube_impulse_cum_Ns"][-1] > 0.005          # the fingertips pressed on the cube in the last pose
    assert abs(rec.traces["robot_cube_impulse_cum_Ns"][-1] - sum(rec.traces["robot_cube_impulse_Ns"])) < 1e-9
    # settle: the evaluator repeats the last command; the robot does not move; phase flags are set
    n_policy = len(rec.traces["phase"])
    q0 = u.agent.robot.get_qpos()[0].cpu().numpy().copy()
    n = rec.run_settle(c)
    assert n == 20 and P.phase_counts(rec.traces) == (n_policy, 20)
    assert np.abs(u.agent.robot.get_qpos()[0].cpu().numpy() - q0).max() < 2e-3
    v, bench, info = rec.evaluate_v2(c)
    assert v["success"] is False and info["n_settle"] == 20              # the hand pushed on the cube: over the tolerance
    assert not next(t for t in v["terms"] if t["metric"].endswith("robot_cube_impulse_cum_Ns"))["holds"]
    try:                                                                 # relative control modes cannot be held by repetition
        rec.control_modes = lambda: ["pd_ee_delta_pose"]; rec.settle(0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("delta control mode accepted for settle")
    env.close()


def test_sim_cupplate_geometry():
    """Tray floor map and cup base footprints from the real collision meshes (dev seed 100)."""
    v = _mani_skill_version()
    if v is None or v == "3.0.1":
        print("  skip (needs the Imitator fork: research/envs/ms2/bin/python)"); return
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from veritask.adapters.maniskill_v2_tasks import BINDINGS
    b = BINDINGS["place_cup_plate_v2"]()
    env = gym.make("TwoRobotPlaceCupPlate-v1", obs_mode="state", control_mode="pd_joint_pos", render_mode=None,
                   render_backend="pci:0", sim_backend="cpu", num_envs=1)
    rec = b.recorder(env)
    rec.reset(seed=100)
    u = env.unwrapped
    pre = b.preconditions(u); tray = pre["tray"]
    print("   tray:", {k: (np.round(v, 4).tolist() if not isinstance(v, float) else round(v, 4)) for k, v in tray.items()})
    assert 0.022 < tray["floor_ref_z"] < 0.025 and 0.049 < tray["rim_top_z"] < 0.053           # floor ~2.4 cm, rim top ~5 cm
    (x0, y0), (x1, y1) = tray["floor_bounds_xy"]
    assert 0.22 < x1 - x0 < 0.26 and 0.28 < y1 - y0 < 0.32                                    # inner floor ~24 x 30 cm
    assert tray["floor_height_range"][1] <= tray["floor_ref_z"] + 0.003 + 1e-9
    r1, r2 = (c["base_equiv_radius_m"] for c in pre["cups"])
    assert 0.025 < r1 < 0.029 and 0.032 < r2 < 0.036
    # cups start on the table: lowest point ~2.4 cm below the tray floor reference, no base over the floor
    dz1, f1 = b.cup_state(u, 1); dz2, f2 = b.cup_state(u, 2)
    assert abs(dz1 - tray["floor_ref_z"]) < 0.002 and abs(dz2 - tray["floor_ref_z"]) < 0.002 and f1 == 0.0 and f2 == 0.0
    # one step with both arms holding: hook count, signals present, released (fingers far from the cups)
    act = {k: np.hstack([ag.robot.get_qpos()[0, :7].cpu().numpy(), 1.0]) for k, ag in zip(u.action_space.spaces.keys(), u.agent.agents)}
    rec.step(act)
    assert rec.traces["released_gap"][-1] == 1.0 and rec.traces["fingers_cups_gap_m"][-1] == rec.acc.gap_far["fingers_cups"]
    assert rec.traces["cup1_base_in_floor_frac"][-1] == 0.0
    env.close()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
