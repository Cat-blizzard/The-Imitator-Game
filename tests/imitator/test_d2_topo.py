"""Tests for card D2 in imsim: the topological tasks (RingOnPeg-v1, MugOnHook-v1), their frozen contracts, the pose and
parts-based enclosure probes, and the proof-test drivers (imitator/verify/proof_test.py).

    source research/env.sh; export HOME=$R/ext_data/sapien_home
    research/envs/imsim/bin/python -B -m pytest -q -p no:cacheprovider tests/imitator/test_d2_topo.py            # fast
    IMSIM_SLOW=1 research/envs/imsim/bin/python -B -m pytest -q -p no:cacheprovider tests/imitator/test_d2_topo.py  # + episodes
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # repository root
PROJECT = ROOT
sys.path.insert(0, ROOT)

import imitator  # noqa: E402
from imitator.verify import proof_test as PT  # noqa: E402
from _paths import DATA, REPO, TMP as _TMP, repo_path, needs_stock_tasks  # noqa: E402,F401
TMP = _TMP

SLOW = os.environ.get("IMSIM_SLOW", "") not in ("", "0")
MANIFEST = os.path.join(DATA, "d2_contract_manifest.json")


class _Rec:
    def __init__(self, env):
        self.env = env

    def hold_action_A(self):
        raise RuntimeError("expected pd_joint_pos")


@pytest.fixture(scope="module")
def ring_env():
    env = imitator.make("RingOnPeg-v1", "panda", sim="cpu", render="none")
    yield env
    env.close()


def test_tasks_registered_and_contracts_frozen():
    import hashlib
    from veritask.adapters.protocol_v3 import check_contract_v3, check_proof_test
    m = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {}
    for task in ("RingOnPeg-v1", "MugOnHook-v1"):
        assert task in imitator.list_tasks()
        spec = imitator.get_task(task)
        c = spec.load_contract()
        assert c.schema_version == "0.4" and c.proof_test and c.protocol == "v3"
        assert check_contract_v3(c) == [] and check_proof_test(c) == []
        assert [x.name for x in c.counterexamples if x.expected is False], task
        assert set(spec.scripts) == {x.name for x in c.counterexamples}
        if c.task_id in m:                                     # frozen file unchanged on disk
            assert hashlib.sha256(open(spec.contract_path(), "rb").read()).hexdigest() == m[c.task_id]["sha256"]


def test_env_geometry_and_benchmark_predicate(ring_env):
    env = ring_env
    env.reset(seed=1000)
    u = env.unwrapped
    t = u.bench_terms()
    assert t["success"] is False and t["on_peg"] is False           # the ring starts on the table away from the peg
    r = u.ring.pose.p[0].numpy(); p = u.peg.pose.p[0].numpy()
    assert np.linalg.norm(r[:2] - p[:2]) >= u.RING_R + u.RING_A + u.PEG_R + u.MIN_SEP - 1e-6
    env2 = imitator.make("MugOnHook-v1", "panda", sim="cpu", render="none")
    env2.reset(seed=1000)
    u2 = env2.unwrapped
    assert abs(u2.mug._bodies[0].angular_damping - u2.MUG_ANGULAR_DAMPING) < 1e-6
    poly = np.asarray(u2.hook_polyline())
    assert abs(np.linalg.norm(poly[-1] - poly[1]) - (u2.HOOK_L + u2.HOOK_R)) < 1e-9
    assert u2.bench_terms()["success"] is False
    env2.close()


def test_pose_and_enclosure_probes(ring_env):
    from imitator.verify.topo_tasks import ring_on_peg
    rec = ring_on_peg(ring_env)
    rec.reset(seed=1001)
    u = ring_env.unwrapped
    rec.step(np.r_[u.agent.robot.get_qpos()[0, :7].numpy(), 1.0].astype(np.float32))
    tr = rec.traces
    p = u.ring.pose.p[0].numpy(); q = u.ring.pose.q[0].numpy()
    got = np.array([tr[f"ring_{k}"][-1] for k in ("px", "py", "pz")])
    gq = np.array([tr[f"ring_{k}"][-1] for k in ("qw", "qx", "qy", "qz")])
    assert np.allclose(got, p, atol=1e-6) and min(np.abs(gq - q).max(), np.abs(gq + q).max()) < 1e-6
    from imitator.verify.v32 import NOT_ENCLOSED_M
    assert tr["ring_enclosure_m"][-1] == NOT_ENCLOSED_M                # nothing between the fingers at the start


def _settle(env, act, n):
    for _ in range(n):
        env.step(act)


@needs_stock_tasks()
def test_proof_drivers_push_shake_lift_tilt_pull_table():
    """each driver does what its parameters say, on scene objects only (robot cleared)"""
    env = imitator.make("PlaceSphere-v1", "panda", sim="cpu", render="none")
    env.reset(seed=3)
    u = env.unwrapped
    act = PT.clear_action(_Rec(env))
    tcp = u.agent.tcp.pose.p[0].numpy()
    assert tcp[2] > 0.6 and tcp[0] < -0.5                         # robot out of the workspace
    _settle(env, act, 10)
    dt = 1.0 / float(u.sim_freq)
    snap = u.get_state_dict()

    def run(test, steps):
        u.set_state_dict(snap)
        d = PT.DRIVERS[test["kind"]](u, test)
        d.begin()
        off = PT._install(u, d, dt)
        try:
            for _ in range(steps):
                env.step(act)
        finally:
            off(); d.end()
        return d

    # lift and tilt the (kinematic) bin: +10 cm, 20 deg, ball carried along
    b0 = u.bin.pose.p[0].numpy().copy()
    d = run(dict(kind="lift_tilt", target="bin", params=dict(lift_m=0.10, tilt_deg=20.0, lift_seconds=1.0, tilt_seconds=0.5,
                                                             hold_seconds=0.5)), 40)
    R = u.bin.pose.sp.to_transformation_matrix()[:3, :3]
    assert abs(u.bin.pose.p[0].numpy()[2] - b0[2] - 0.10) < 1e-3
    assert abs(math.degrees(math.acos(R[2, 2])) - 20.0) < 0.5
    # shake: back at the start pose after the (ramped) shake, max excursion ~ amplitude
    xs = []
    u.set_state_dict(snap)
    d = PT.Shake(u, dict(kind="shake", target="bin", params=dict(amplitude_m=0.02, freq_hz=2.0, duration_seconds=2.0,
                                                                 axis=[1, 0, 0])))
    d.begin()
    off = PT._install(u, d, dt)
    for _ in range(40):
        env.step(act); xs.append(u.bin.pose.p[0].numpy()[0] - b0[0])
    off(); d.end()
    assert abs(xs[-1]) < 1e-4 and 0.018 < max(np.abs(xs)) <= 0.0201
    assert abs(d.info["peak_accel_mps2"] - 0.02 * (4 * math.pi) ** 2) < 1e-3
    # table shake: amplitude = accel / (2 pi f)^2, fixtures move with the table
    t0 = PT.pose_of(PT.resolve(u, "table")).p.copy()
    u.set_state_dict(snap)
    d = PT.TableShake(u, dict(kind="table_shake", target="table", params=dict(accel_mps2=0.3, freq_hz=2.0, duration_seconds=2.0,
                                                                              axis=[0, 1, 0], fixtures=["bin"])))
    d.begin()
    off = PT._install(u, d, dt)
    dy_t, dy_b = [], []
    for _ in range(40):
        env.step(act)
        dy_t.append(PT.pose_of(PT.resolve(u, "table")).p[1] - t0[1]); dy_b.append(u.bin.pose.p[0].numpy()[1] - b0[1])
    off(); d.end()
    A = 0.3 / (4 * math.pi) ** 2
    assert np.allclose(dy_t, dy_b, atol=1e-6) and abs(max(np.abs(dy_t)) - A) < 0.1 * A
    # push the ball (dynamic): velocity change applied once
    u.set_state_dict(snap)
    v0 = np.asarray(PT.body(u.obj).get_linear_velocity(), float)
    d = PT.Push(u, dict(kind="push", target="obj", params=dict(delta_v_mps=0.25, direction=[0, 1, 0])))
    d.begin()
    off = PT._install(u, d, dt)
    u._before_simulation_step()                                   # one substep's worth of driver call, no physics step
    off()
    v1 = np.asarray(PT.body(u.obj).get_linear_velocity(), float)
    assert np.allclose(v1 - v0, [0, 0.25, 0], atol=1e-6) and abs(d.info["impulse_Ns"] - 0.25 * d.info["mass_kg"]) < 1e-6
    # pull: 2 N upwards on the 33.5 g ball for 0.25 s -> net acceleration F/m - g
    u.set_state_dict(snap)
    z0 = u.obj.pose.p[0].numpy()[2]
    d = PT.Pull(u, dict(kind="pull", target="obj", params=dict(force_N=2.0, duration_seconds=0.25, axis=[0, 0, 1])))
    d.begin()
    off = PT._install(u, d, dt)
    for _ in range(5):
        env.step(act)
    off(); d.end()
    m = float(PT.body(u.obj).mass)
    expect = 0.5 * (2.0 / m - 9.81) * 0.25 ** 2
    assert abs((u.obj.pose.p[0].numpy()[2] - z0) - expect) < 0.15 * expect
    env.close()


# ------------------------------------------------------------------------------------------------ slow
@pytest.mark.skipif(not SLOW, reason="set IMSIM_SLOW=1 (scripted episodes with settle, retreat and proof test)")
@pytest.mark.parametrize("task,beh,expect", [("RingOnPeg-v1", "expert", True), ("RingOnPeg-v1", "on_top", False),
                                             ("MugOnHook-v1", "expert", True)])
def test_slow_episode(task, beh, expect):
    out = os.path.join(TMP, "test_d2", task, beh)           # project-local, never /tmp
    res = imitator.evaluate(task, "panda", beh, seeds=[1010], out_dir=out, experiment="D2-test", verbose=False)
    r = res["rows"][0]
    assert "error" not in r, r
    assert r["verdict"] is expect
    assert r["aftermath"]["proof_test"]["with_proof"] is expect                # (c) = (b) and the proof test
