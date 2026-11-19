"""Tests for card D2: topological success primitives (outcomes/topology.py), their registration as geometry and trace
metrics, the protocol-v3 classification, and the proof-test stage (contract schema 0.4, adapters/protocol_v3.py).
No simulator needed.  usage: python -B tests/test_topology.py   (or pytest)

Reference values are the numpy probe research/scripts/e30_probe_topology.py (results in
research/results/e30_probes/topology_sanity.json).
"""
import copy
import json
import math
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import REPO  # noqa: E402
ROOT = os.path.join(REPO, "veritask")   # package dir (tasks/ lives here)
sys.path.insert(0, REPO)
from veritask.outcomes import topology as T  # noqa: E402
from veritask.outcomes.metrics import REGISTRY, measure  # noqa: E402
from veritask.spec.contract import SCHEMA_VERSION, TaskContract, validate_dict, validate_proof_test  # noqa: E402
from veritask.adapters import protocol_v3 as P  # noqa: E402


# ------------------------------------------------------------------------------------------------ winding number
def probe_open_box():
    """the probe's open-top 20 x 20 x 15 cm box (5 faces, 12 x 12 grid each)"""
    return T.open_box_mesh(0.1, 0.1, 0.0, 0.15, n=12)


def test_winding_probe_values():
    V, F = probe_open_box()
    assert V.shape == (845, 3) and F.shape == (1440, 3)            # same tessellation as the probe
    pts = np.array([[0, 0, 0.02], [0, 0, 0.075], [0, 0, 0.15], [0, 0, 0.20], [0, 0, 0.45]])
    w = T.winding_number(V, F, pts)
    probe = [0.879, 0.779, 0.500, 0.295, 0.032]                   # topology_sanity.json (rounded to 3 decimals)
    assert np.allclose(w, probe, atol=6e-4), w
    assert np.round(w, 2).tolist() == [0.88, 0.78, 0.50, 0.30, 0.03]
    assert abs(T.winding_number(V, F, [0.3, 0, 0.07]) - (-0.0097)) < 5e-4          # beside the box
    assert abs(T.winding_number(V, F, [0, 0, -0.05]) - (-0.0641)) < 5e-4           # below it
    # exactly 0.5 at the centre of the opening, by symmetry
    assert abs(T.winding_number(V, F, [0, 0, 0.15]) - 0.5) < 1e-9


def test_winding_closed_surfaces():
    V, F = T.open_box_mesh(0.05, 0.04, 0.0, 0.06, n=4, closed=True)
    inside = np.array([[0, 0, 0.03], [0.04, 0.03, 0.01], [-0.049, 0, 0.059]])
    outside = np.array([[0.2, 0, 0.03], [0, 0, 0.07], [0, 0, -0.001], [1.0, 2.0, 3.0]])
    assert np.allclose(T.winding_number(V, F, inside), 1.0, atol=1e-9)
    assert np.allclose(T.winding_number(V, F, outside), 0.0, atol=1e-9)
    Vc, Fc = T.open_cylinder_mesh(0.04, 0.0, 0.1, n=24, closed=True)
    assert abs(T.winding_number(Vc, Fc, [0.01, 0.0, 0.05]) - 1.0) < 1e-9
    assert abs(T.winding_number(Vc, Fc, [0.06, 0.0, 0.05])) < 1e-9
    # open cup: deep inside > 0.5 > far above
    Vo, Fo = T.open_cylinder_mesh(0.04, 0.0, 0.1, n=24)
    assert T.winding_number(Vo, Fo, [0, 0, 0.01]) > 0.8 and T.winding_number(Vo, Fo, [0, 0, 0.3]) < 0.1
    assert abs(T.winding_number(Vo, Fo, [0, 0, 0.1]) - 0.5) < 1e-6


def test_winding_vectorised_chunked_and_oriented():
    V, F = probe_open_box()
    rng = np.random.default_rng(0)
    P_ = rng.uniform([-0.15, -0.15, -0.05], [0.15, 0.15, 0.3], size=(57, 3))
    full = T.winding_number(V, F, P_)
    one = np.array([T.winding_number(V, F, p) for p in P_])
    tiny = T.winding_number(V, F, P_, max_pairs=1000)           # chunks of < 1 point: one point per chunk
    assert full.shape == (57,) and isinstance(T.winding_number(V, F, P_[0]), float)
    assert np.allclose(full, one, atol=1e-12) and np.allclose(full, tiny, atol=1e-12)
    assert np.allclose(T.winding_number(V, F[:, ::-1], P_), -full, atol=1e-12)      # flipped orientation
    assert abs(T.containment_margin(0.75) - 0.5) < 1e-12 and np.allclose(T.containment_margin([0.5, 1.0]), [0, 1])


# ------------------------------------------------------------------------------------------------ crossings
def probe_ropes():
    t = np.linspace(0, 1, 60)[:, None]
    through = np.hstack([0.01 * np.sin(6 * t), 0.0 * t, -0.10 + 0.25 * t])        # 15 cm beyond the ring
    barely = np.hstack([0.0 * t, 0.0 * t, -0.20 + 0.21 * t])                       # tip only 1 cm beyond
    beside = np.hstack([0.06 + 0.0 * t, 0.0 * t, -0.10 + 0.25 * t])               # outside the ring
    back = np.vstack([through, through[::-1] + np.array([0.012, 0, 0])])          # through and back
    return dict(through=through, barely=barely, beside=beside, back=back)


def test_crossings_probe_cases():
    c, n, r = np.zeros(3), np.array([0, 0, 1.0]), 0.03
    R = probe_ropes()
    x = T.disk_crossings(R["through"], c, n, r)
    assert (x.net, x.n) == (1, 1) and abs(x.margin - 0.10113023481611774) < 1e-9 and x.threaded
    net, margin = T.disk_crossings(R["barely"], c, n, r)                # unpacks like the probe's (net, margin)
    assert net == 1 and abs(margin - 0.01) < 1e-9
    x = T.disk_crossings(R["beside"], c, n, r)
    assert (x.net, x.n, x.margin) == (0, 0, 0.0) and math.isnan(x.arc_before)
    x = T.disk_crossings(R["back"], c, n, r)
    # probe: (0, 0.101) -- it reported the first crossing's margin although net = 0; here net 0 means not threaded
    assert (x.net, x.n, x.margin) == (0, 2, 0.0) and x.signs == (1, -1)
    # direction: the reversed rope crosses against the normal
    assert T.disk_crossings(R["through"][::-1], c, n, r).net == -1


def test_crossings_edge_cases_and_free_end():
    c, n = np.zeros(3), np.array([0, 0, 1.0])
    # a vertex exactly on the disk plane is counted once
    C = np.array([[0, 0, -0.1], [0, 0, 0.0], [0, 0, 0.1]])
    assert T.disk_crossings(C, c, n, 0.01).net == 1
    C2 = np.array([[0, 0, -0.1], [0, 0, 0.0], [0, 0, -0.05]])          # touches the plane and goes back
    assert T.disk_crossings(C2, c, n, 0.01).net in (0,) and T.disk_crossings(C2, c, n, 0.01).n == 2
    # outside the radius: no crossing
    assert T.disk_crossings(np.array([[0.02, 0, -0.1], [0.02, 0, 0.1]]), c, n, 0.019).n == 0
    assert T.disk_crossings(np.array([[0.02, 0, -0.1], [0.02, 0, 0.1]]), c, n, 0.021).n == 1
    # a peg (base at z=0, top at z=0.10) through a ring at height 0.03: free end = the top
    peg = np.array([[0, 0, 0.0], [0, 0, 0.05], [0, 0, 0.10]])
    rc = np.array([0.005, 0, 0.03])
    assert abs(T.disk_crossings(peg, rc, n, 0.02, free_end="end").margin - 0.07) < 1e-12
    assert abs(T.disk_crossings(peg, rc, n, 0.02, free_end="start").margin - 0.03) < 1e-12
    assert abs(T.disk_crossings(peg, rc, n, 0.02).margin - 0.03) < 1e-12               # 'both' = the smaller
    # a tilted ring: normal 30 deg from vertical still threads the peg
    nt = np.array([np.sin(np.radians(30)), 0, np.cos(np.radians(30))])
    assert T.disk_crossings(peg, rc, nt, 0.02).net == 1
    # ring above the top: not threaded
    assert T.disk_crossings(peg, [0, 0, 0.11], n, 0.02).n == 0
    for bad in (dict(free_end="tip"),):
        try:
            T.disk_crossings(peg, rc, n, 0.02, **bad)
        except ValueError:
            continue
        raise AssertionError(bad)


# ------------------------------------------------------------------------------------------------ wrap angle
def test_wrap_angle():
    th = np.linspace(0, 2 * np.pi, 73)
    circle = np.stack([0.05 * np.cos(th), 0.05 * np.sin(th), 0 * th], 1)
    assert abs(T.wrap_angle(circle, [0, 0, 0], [0, 0, 1]) - 2 * np.pi) < 1e-9
    assert abs(T.wrap_angle(circle, [0, 0, 0], [0, 0, -1]) + 2 * np.pi) < 1e-9          # right-hand rule
    th2 = np.linspace(0, 5 * np.pi, 200)                                                  # helix, 2.5 turns
    helix = np.stack([0.03 * np.cos(th2), 0.03 * np.sin(th2), 0.01 * th2], 1)
    assert abs(T.wrap_turns(helix, [0, 0, 0], [0, 0, 1]) - 2.5) < 1e-9
    assert abs(T.wrap_angle(helix[::-1], [0, 0, 0], [0, 0, 1]) + 5 * np.pi) < 1e-9
    assert abs(T.wrap_angle(circle + [0.2, 0, 0], [0, 0, 0], [0, 0, 1])) < 1e-9          # loop not around the axis
    line = np.stack([np.linspace(-1, 1, 50), 0.02 + 0 * th[:50], 0 * th[:50]], 1)        # passes beside the axis
    assert abs(T.wrap_angle(line, [0, 0, 0], [0, 0, 1])) < np.pi
    # tilted axis: the same helix rotated stays 2.5 turns about the rotated axis
    R = T.quat_to_matrix([np.cos(0.3), np.sin(0.3), 0, 0])
    assert abs(T.wrap_turns(helix @ R.T, [0, 0, 0], R @ [0, 0, 1]) - 2.5) < 1e-9


# ------------------------------------------------------------------------------------------------ registry
def test_registered_geometry_metrics():
    for k in ("topology.winding_number", "topology.containment_margin", "topology.disk_crossings",
              "topology.insertion_margin", "topology.wrap_angle", "topology.wrap_turns") + T.TRACE_END_STATE + T.TRACE_EVENT:
        assert k in REGISTRY, k
    V, F = probe_open_box()
    pts = [[0, 0, 0.02], [0, 0, 0.45]]
    assert abs(measure("topology.winding_number", V=V, F=F, points=pts) - 0.032) < 6e-4          # min
    assert abs(measure("topology.winding_number", V=V, F=F, points=pts, reduce="max") - 0.879) < 6e-4
    assert measure("topology.winding_number", V=V, F=F, points=pts, reduce="frac_inside") == 0.5
    assert abs(measure("topology.containment_margin", V=V, F=F, points=[[0, 0, 0.02]]) - 0.758) < 2e-3
    R = probe_ropes()
    kw = dict(centre=[0, 0, 0], normal=[0, 0, 1], radius=0.03)
    assert measure("topology.disk_crossings", polyline=R["through"][::-1], **kw) == 1.0
    assert measure("topology.disk_crossings", polyline=R["through"][::-1], absolute=False, **kw) == -1.0
    assert abs(measure("topology.insertion_margin", polyline=R["barely"], **kw) - 0.01) < 1e-9
    th = np.linspace(0, 4 * np.pi, 100)
    coil = np.stack([0.03 * np.cos(th), 0.03 * np.sin(th), 0.005 * th], 1)
    assert abs(measure("topology.wrap_turns", polyline=coil, axis_point=[0, 0, 0], axis_dir=[0, 0, 1]) - 2) < 1e-9
    assert abs(measure("topology.wrap_angle", polyline=coil, axis_point=[0, 0, 0], axis_dir=[0, 0, 1]) - 720) < 1e-6


def _pose_traces(prefix, P, Q):
    out = {}
    for k, s in enumerate(("px", "py", "pz")):
        out[f"{prefix}_{s}"] = list(np.asarray(P, float)[:, k])
    for k, s in enumerate(("qw", "qx", "qy", "qz")):
        out[f"{prefix}_{s}"] = list(np.asarray(Q, float)[:, k])
    return out


def _ring_down_peg(n=30):
    """ring (core radius 3 cm) lowered from 5 cm above a 10 cm peg's top to 1 cm above its base, then resting"""
    z = np.concatenate([np.linspace(0.15, 0.01, 20), np.full(n - 20, 0.01)])
    P_ = np.stack([np.full(n, 0.004), np.zeros(n), z], 1)
    Q = np.tile([1.0, 0, 0, 0], (n, 1))
    tr = _pose_traces("ring", P_, Q)
    tr.update(_pose_traces("peg", np.tile([0, 0, 0.0], (n, 1)), Q))
    return tr, z


PEG_ARGS = dict(ring="ring", ring_radius=0.03, curve=[[0, 0, 0], [0, 0, 0.10]], curve_pose="peg")


def test_trace_metrics_crossings_and_insertion():
    tr, z = _ring_down_peg()
    s = T.series("crossings", tr, "ring_px", **PEG_ARGS)
    assert s.tolist() == [float(v < 0.10) for v in z]
    m = T.series("insertion", tr, "ring_px", free_end="end", **PEG_ARGS)
    assert np.allclose(m, np.where(z < 0.10, 0.10 - z, 0.0))
    assert measure("trace.crossings_window_min", traces=tr, signal="ring_px", window=10, **PEG_ARGS) == 1.0
    assert measure("trace.crossings_min", traces=tr, signal="ring_px", **PEG_ARGS) == 0.0          # event: whole trace
    v = measure("trace.insertion_window_min", traces=tr, signal="ring_px", window=10, free_end="end", **PEG_ARGS)
    assert abs(v - 0.09) < 1e-12
    # a static peg pose given in the args gives the same values
    a2 = dict(PEG_ARGS, curve_pose=[0, 0, 0, 1, 0, 0, 0])
    assert measure("trace.crossings_window_min", traces=tr, signal="ring_px", window=10, **a2) == 1.0
    # signed crossings: ring flipped upside down -> -1 with absolute=False
    trf = dict(tr)
    trf.update(_pose_traces("ring", np.stack([tr["ring_px"], tr["ring_py"], tr["ring_pz"]], 1),
                            np.tile([0.0, 1.0, 0, 0], (len(z), 1))))
    assert measure("trace.crossings_window_min", traces=trf, signal="ring_px", window=5, absolute=False, **PEG_ARGS) == -1.0
    try:
        measure("trace.crossings_window_min", traces={"ring_px": [0.0]}, signal="ring_px", window=1, **PEG_ARGS)
    except KeyError as e:
        assert "pose signals missing" in str(e)
    else:
        raise AssertionError("missing pose signals must raise")


def test_trace_metric_winding_with_moving_container():
    """ball (r = 2 cm) resting in a 4 x 4 cm bin 1 cm deep (ManiSkill PlaceSphere geometry): the ball's centre is above
    the rim (w < 0.5) but its lowest point, 2 mm inside the ball, is inside; the bin is then lifted 10 cm and tilted 20
    deg with the ball in it, and finally the ball is beside the bin"""
    n = 12
    mesh = dict(kind="open_box", half_x=0.02, half_y=0.02, z0=0.0025, z1=0.0125, n=8)    # bin frame (origin = plate centre)
    bin_p = np.tile([0.05, -0.02, 0.0025], (n, 1)); bin_q = np.tile([1.0, 0, 0, 0], (n, 1))
    ball = np.tile([0.05, -0.02, 0.025], (n, 1)); ball_q = np.tile([1.0, 0, 0, 0], (n, 1))
    a = np.radians(20)
    q_t = [np.cos(a / 2), np.sin(a / 2), 0, 0]                           # 20 deg about x
    for k in range(4, 8):                                                # lifted and tilted, ball carried along
        bin_p[k, 2] += 0.10; bin_q[k] = q_t
        R = T.quat_to_matrix(q_t)
        ball[k] = bin_p[k] + R @ np.array([0, 0, 0.0225])
    ball[8:] = [0.15, -0.02, 0.02]                                       # beside the bin, on the table
    tr = _pose_traces("bin", bin_p, bin_q); tr.update(_pose_traces("sphere", ball, ball_q))
    base = dict(mesh=mesh, container="bin", object="sphere")
    centre = T.series("winding", tr, "sphere_px", **base)
    low = T.series("winding", tr, "sphere_px", points=[[0, 0, -0.018]], points_frame="world", **base)
    assert np.all(centre[:4] < 0.5) and np.all(low[:4] > 0.6), (centre, low)
    assert np.all(low[4:8] > 0.6)                         # tilted 20 deg: the lowest point stays inside
    assert np.all(low[8:] < 0.05)
    assert measure("trace.winding_window_min", traces={k: v[:8] for k, v in tr.items()}, signal="sphere_px", window=8,
                   points=[[0, 0, -0.018]], points_frame="world", **base) > 0.6
    assert measure("trace.winding_min", traces=tr, signal="sphere_px", points=[[0, 0, -0.018]], points_frame="world",
                   **base) < 0.05


def test_trace_metric_wrap_turns():
    n = 5
    th = np.linspace(0, 4 * np.pi, 80)
    coil = np.stack([0.03 * np.cos(th), 0.03 * np.sin(th), 0.002 * th], 1).tolist()
    Q = np.tile([1.0, 0, 0, 0], (n, 1))
    tr = _pose_traces("rope", np.tile([0.1, 0, 0], (n, 1)), Q)
    tr.update(_pose_traces("post", np.tile([0.1, 0, 0], (n, 1)), Q))
    v = measure("trace.wrap_turns_window_min", traces=tr, signal="rope_px", window=3, curve=coil, curve_pose="rope",
                axis_pose="post")
    assert abs(v - 2.0) < 1e-9


# ------------------------------------------------------------------------------------------------ protocol v3 + proof test
SETTLE = dict(hold="A", max_seconds=3.0, window_seconds=1.0, quiet_steps=10, quiet_max={"speed": 0.02},
              robot_speed_signal="rq", robot_speed_max=0.01)


def _contract(outcome, proof_test=None):
    d = dict(schema_version="0.4", task_id="t", instruction="x", outcome=outcome, require_released=False, protocol="v3",
             settle=dict(SETTLE), counterexamples=[dict(name="c", description="d", expected=False)])
    if proof_test is not None:
        d["proof_test"] = proof_test
    return d


def _topo_term(metric, op, thr, **args):
    return dict(metric=metric, op=op, threshold=thr, units="", args=dict(signal="ring_px", **PEG_ARGS, **args),
                real_measurement="test")


def test_protocol_v3_classifies_and_measures_topology_terms():
    terms = [_topo_term("trace.crossings_window_min", ">=", 1, window=20),
             _topo_term("trace.insertion_window_min", ">=", 0.05, window=20, free_end="end")]
    c = TaskContract.from_dict(_contract(terms))
    assert P.check_contract_v3(c) == [], P.check_contract_v3(c)
    n_pol, n_set = 30, 40
    tr, _ = _ring_down_peg(n_pol + n_set)
    tr["phase"] = [0.0] * n_pol + [1.0] * n_set
    tr["speed"] = [0.0] * (n_pol + n_set)
    tr["rq"] = [0.0] * (n_pol + n_set)
    v, _, info = P.evaluate_traces_v3(c, tr)
    assert info["window_status"] == "ok" and v["success"] is True, (v, info)
    assert abs(v["terms"][1]["value"] - 0.09) < 1e-12
    # ring stuck above the top: crossings 0 in the window -> rejected
    tr2 = dict(tr); tr2["ring_pz"] = [0.12] * (n_pol + n_set)
    v2, _, _ = P.evaluate_traces_v3(c, tr2)
    assert v2["success"] is False and v2["binding_term"].startswith("trace.crossings_window_min")


def _pt_block(tests=None):
    return dict(robot="clear", pre_settle_seconds=0.5, window_seconds=1.0, tests=tests or [
        dict(name="push_x", kind="push", target="ring", params=dict(delta_v_mps=0.2, direction=[1, 0, 0]),
             rest_seconds=1.5, terms=[_topo_term("trace.crossings_min", ">=", 1),
                                      _topo_term("trace.insertion_window_min", ">=", 0.05, window=20, free_end="end")]),
        dict(name="table", kind="table_shake", target="table",
             params=dict(accel_mps2=0.3, freq_hz=2.0, duration_seconds=2.0),
             terms=[_topo_term("trace.crossings_window_min", ">=", 1, window=20)])])


def _pt_traces(test, n_settle=10, n_dist=20, n_rest=30, unthread_at=None, z=0.01):
    n = n_settle + n_dist + n_rest
    tr, _ = _ring_down_peg(n)
    zz = np.full(n, z)
    if unthread_at is not None:
        zz[unthread_at:] = 0.15
    tr["ring_pz"] = list(zz)
    tr[P.PT_STAGE_SIGNAL] = [0.0] * n_settle + [1.0] * n_dist + [2.0] * n_rest
    return {P.pt_key(test, k): v for k, v in tr.items()}


def test_proof_test_schema_validation():
    d = _contract([_topo_term("trace.crossings_window_min", ">=", 1, window=20)], _pt_block())
    assert not [p for p in validate_dict(d) if not p.startswith("warning:")], validate_dict(d)
    c = TaskContract.from_dict(d)
    assert c.proof_test["tests"][0]["kind"] == "push" and P.check_proof_test(c) == []
    assert "proof_test" in c.to_dict()
    bad = {
        "kind 'wiggle'": lambda b: b["tests"][0].__setitem__("kind", "wiggle"),
        "params.delta_v_mps must be a number": lambda b: b["tests"][0]["params"].pop("delta_v_mps"),
        "test names must be unique": lambda b: b["tests"].append(copy.deepcopy(b["tests"][0])),
        "must be trace metrics": lambda b: b["tests"][0]["terms"].append(dict(metric="rigid.tilt_deg", op="<",
                                                                              threshold=1.0, args={})),
        "proof_test: unknown field": lambda b: b.__setitem__("shake_everything", True),
        "robot must be 'clear'": lambda b: b.__setitem__("robot", "stay"),
        "name must be a non-empty string without ':'": lambda b: b["tests"][0].__setitem__("name", "a:b"),
    }
    for needle, mutate in bad.items():
        b = _pt_block(); mutate(b)
        probs = validate_proof_test(b, "0.4", "v3", REGISTRY)
        assert any(needle in p and not p.startswith("warning:") for p in probs), (needle, probs)
    assert any("needs protocol v3" in p for p in validate_proof_test(_pt_block(), "0.4", "v2"))
    assert any(p.startswith("warning:") and "schema-0.4" in p for p in validate_proof_test(_pt_block(), "0.3", "v3"))
    # a window that differs from the proof window is a protocol problem
    b = _pt_block(); b["tests"][1]["terms"][0]["args"]["window"] = 7
    assert any("differs from the proof window" in p for p in P.check_proof_test(TaskContract.from_dict(_contract(
        [_topo_term("trace.crossings_window_min", ">=", 1, window=20)], b))))


def test_proof_test_evaluation_and_combination():
    c = TaskContract.from_dict(_contract([_topo_term("trace.crossings_window_min", ">=", 1, window=20)], _pt_block()))
    tr = {}
    tr.update(_pt_traces("push_x")); tr.update(_pt_traces("table"))
    r = P.evaluate_proof_tests(c, tr)
    assert r["declared"] and r["success"] is True and [t["success"] for t in r["tests"]] == [True, True], r
    assert abs(r["tests"][0]["terms"]["trace.insertion_window_min:ring_px"] - 0.09) < 1e-12
    # unthreaded during the push's disturbance, back on (impossible physically, but the event term must see it)
    tr2 = dict(tr); tr2.update(_pt_traces("push_x", unthread_at=15))
    r2 = P.evaluate_proof_tests(c, tr2)
    assert r2["success"] is False and r2["tests"][0]["success"] is False and r2["tests"][1]["success"] is True
    assert r2["binding_test"] == "push_x"
    # unthreading during the pre-settle stage (stage 0) is not part of the test
    tr3 = dict(tr)
    t3 = _pt_traces("push_x"); z = np.asarray(t3[P.pt_key("push_x", "ring_pz")]); z[:5] = 0.15
    t3[P.pt_key("push_x", "ring_pz")] = list(z); tr3.update(t3)
    assert P.evaluate_proof_tests(c, tr3)["success"] is True
    # a declared test without traces -> unknown; a failed test decides False even if another is unknown
    r4 = P.evaluate_proof_tests(c, _pt_traces("push_x"))
    assert r4["success"] is None and r4["tests"][1]["status"] == "missing"
    r5 = P.evaluate_proof_tests(c, _pt_traces("push_x", unthread_at=12))
    assert r5["success"] is False
    # too short (fewer steps than the window after the disturbance starts) -> unknown
    r6 = P.evaluate_proof_tests(c, {**_pt_traces("push_x", n_dist=5, n_rest=5), **_pt_traces("table")})
    assert r6["success"] is None and r6["tests"][0]["status"] == "too_short"
    assert P.combine_with_proof(True, r) is True and P.combine_with_proof(True, r2) is False
    assert P.combine_with_proof(False, r) is False and P.combine_with_proof(None, r) is None
    assert P.combine_with_proof(True, r4) is None and P.combine_with_proof(False, r4) is False
    assert P.combine_with_proof(True, dict(declared=False)) is True and P.combine_with_proof(True, None) is True
    # contracts without a proof test: not declared
    c0 = TaskContract.from_dict(_contract([_topo_term("trace.crossings_window_min", ">=", 1, window=20)]))
    assert P.evaluate_proof_tests(c0, tr) == dict(declared=False, success=None, tests=[], binding_test=None)
    main, per = P.split_proof_traces({**tr, "phase": [0.0]})
    assert set(per) == {"push_x", "table"} and list(main) == ["phase"]


def test_backward_compatibility():
    """schema 0.4 is a superset: every existing contract validates and serialises without a proof_test key"""
    import glob
    files = sorted(glob.glob(f"{ROOT}/tasks/*.json"))
    imsim_dir = os.path.join(REPO, "imitator", "contracts")
    files += sorted(glob.glob(f"{imsim_dir}/*.json"))
    assert len(files) >= 20
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        errs = [p for p in validate_dict(d) if not p.startswith("warning:")]
        assert not errs, (f, errs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c = TaskContract.from_dict(d)
        declared = bool(d.get("proof_test"))            # only the D2 schema-0.4 contracts declare one
        assert ("proof_test" in c.to_dict()) == declared, f
        assert P.evaluate_proof_tests(c, {})["declared"] is declared
        if not declared:
            assert c.to_dict() == TaskContract.from_dict(c.to_dict(), validate=False).to_dict()
    assert SCHEMA_VERSION == "0.4"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
