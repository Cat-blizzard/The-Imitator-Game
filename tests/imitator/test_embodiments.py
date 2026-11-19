"""Tests for the imsim embodiment library (E19).

Run (msup venv, upstream ManiSkill 3.0.1):
    source research/env.sh
    research/envs/msup/bin/python -B tests/imitator/test_embodiments.py            # fast: no rendering, CPU sim, ~2-4 min
    IMSIM_SLOW=1 research/envs/msup/bin/python -B tests/imitator/test_embodiments.py   # + scripted pick-and-place judged by
                                                                                  #   the contract, one CPU render
The file is also pytest-compatible (test_* functions) if pytest is available.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from collections import OrderedDict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from imitator import compat  # noqa: E402
from imitator.embodiments import dual as D  # noqa: E402
from imitator.embodiments import registry as reg  # noqa: E402
from imitator.embodiments.urdf_tools import load_urdf, matrix_to_rpy, rpy_to_matrix  # noqa: E402

SLOW = os.environ.get("IMSIM_SLOW", "") not in ("", "0")
_ENVS: dict = {}


def _env(cfg):
    if cfg not in _ENVS:
        _ENVS[cfg] = D.make_dual_env(cfg)
    e = _ENVS[cfg]
    e.reset(seed=0)
    return e


# ------------------------------------------------------------------------------------------------ pure
def test_environment_vars_inside_project():
    assert compat.check_env() == [], compat.check_env()


def test_rpy_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        rpy = rng.uniform([-3, -1.5, -3], [3, 1.5, 3])
        R = rpy_to_matrix(rpy)
        assert np.allclose(rpy_to_matrix(matrix_to_rpy(R)), R, atol=1e-9)


def test_registry_records():
    for name, r in reg.ARMS.items():
        for p in (r.urdf_path, r.srdf_path, r.mplib_urdf_path, r.mplib_srdf_path):
            assert os.path.exists(p), (name, p)
        m = load_urdf(r.urdf_path)
        assert r.tcp_link in m.links and r.flange_link in m.links, name
        for l in r.grasp_links:
            assert l in m.links, (name, l)
        if r.pad_links:
            for side in r.sides:
                for l in side:
                    assert l in m.links, (name, l)
        for j in r.arm_joints + r.gripper_joints:
            J = m.joints[j]
            assert J.lower is not None and J.upper is not None and J.lower < J.upper, (name, j)
            assert abs(J.lower) < 7 and abs(J.upper) < 7, (name, j, "placeholder limits")
        home = r.home_qpos()
        for j, v in home.items():
            J = m.joints[j]
            if J.lower is not None:
                assert J.lower - 1e-6 <= v <= J.upper + 1e-6, (name, j, v)
        assert r.license and r.source, name
        assert r.action_dim == r.dof_arm + r.dof_gripper
        # TCP axes: the finger pads straddle the TCP along its y axis (closing) and lie ahead along z (approach)
        if r.pad_links:
            Ts = m.fk(home)
            Ti = np.linalg.inv(Ts[r.tcp_link])
            c = [((Ti @ Ts[l])[:3, :3] @ m.collision_points(l).T).T.mean(0) + (Ti @ Ts[l])[:3, 3] for l in r.pad_links]
            assert c[0][1] * c[1][1] < 0, (name, "pads not on both sides of the TCP along y", c)
            assert abs(c[0][0]) < 0.02 and abs(c[1][0]) < 0.02, (name, c)


def test_srdf_bit_budget():
    """simulator SRDFs stay well inside SAPIEN's 31-pair limit, so two arms fit into group 2 together"""
    import json
    for name, r in reg.ARMS.items():
        s = json.load(open(r.srdf_path.replace(".srdf", "_srdf.json")))
        assert s["n_sim_pairs"] <= 12, (name, s["n_sim_pairs"])


def test_dual_configs():
    assert len(D.CONFIGS) == 5
    from mani_skill.agents.registration import REGISTERED_AGENTS
    import imitator.embodiments.agents  # noqa: F401
    for c in D.CONFIGS.values():
        l, r = c.mount_poses()
        assert np.isclose(l.p[1], -r.p[1]) and np.isclose(l.p[1] - r.p[1], c.separation)
        for uid in c.robot_uids:
            assert uid in REGISTERED_AGENTS, uid


def test_contract_valid():
    from imitator.verify import smoke as V
    from veritask.adapters.protocol_v3 import check_contract_v3
    c = V.load_contract()
    assert c.validate() == [] and check_contract_v3(c) == []


def test_contract_on_synthetic_traces():
    """accept: cube resting in the square, no contact, not enclosed; reject: same but enclosed / touching"""
    from imitator.verify import smoke as V
    from veritask.adapters.protocol_v3 import evaluate_traces_v3
    c = V.load_contract()
    n_pol, n_set = 30, 40
    n = n_pol + n_set
    base = dict(cube_x=[0.3] * n, cube_y=[0.0] * n, cube_z=[0.025] * n, cube_speed=[0.0] * n, cube_target_margin_m=[0.04] * n,
                cube_bottom_dz_m=[0.0] * n, robot_qvel_max=[0.0] * n, robot_cube_impulse_Ns=[0.0] * n,
                cube_enclosure_clearance_m=[1.0] * n, phase=[0.0] * n_pol + [1.0] * n_set, bench_success=[1.0] * n)
    v, _, _ = evaluate_traces_v3(c, base)
    assert v["success"] is True, v
    enc = dict(base, cube_enclosure_clearance_m=[0.0005] * n)
    assert evaluate_traces_v3(c, enc)[0]["success"] is False
    touch = dict(base, robot_cube_impulse_Ns=[0.01] * n)
    assert evaluate_traces_v3(c, touch)[0]["success"] is False
    out = dict(base, cube_target_margin_m=[-0.01] * n)
    assert evaluate_traces_v3(c, out)[0]["success"] is False


# ------------------------------------------------------------------------------------------------ simulator (CPU)
def test_load_home_stable_all_configs():
    for cfg in D.CONFIGS:
        env = _env(cfg)
        u = env.unwrapped
        ags = list(u.agent.agents)
        q0 = [a.robot.get_qpos().clone() for a in ags]
        act = OrderedDict((k, a.home_action()) for k, a in zip(u.single_action_space.spaces.keys(), ags))
        for _ in range(20):
            env.step(act)
        for a, q in zip(ags, q0):
            assert float((a.robot.get_qpos() - q).abs().max()) < 0.02, (cfg, a.uid)
            lim = a.robot.get_qlimits()
            qq = a.robot.get_qpos()
            assert float(np.maximum((lim[..., 0] - qq).max(), (qq - lim[..., 1]).max())) < 1e-3, (cfg, a.uid)
        assert float(u.cube.linear_velocity.norm()) < 0.01, cfg


def test_group2_bits_disjoint_between_arms():
    for cfg in D.CONFIGS:
        u = _env(cfg).unwrapped
        a, b = [D.group2_bits(x) for x in u.agent.agents]
        assert (a & b) <= {D.TABLE_BIT}, (cfg, a, b)
        # the shared table bit is only on links rigidly attached to the bases
        for ag, links in zip(u.agent.agents, u.base_table_links):
            for l in ag.robot.get_links():
                bits = D.group2_bits_of_link(l)
                if D.TABLE_BIT in bits:
                    assert l.name in links, (cfg, l.name)


def test_measured_opening_open_gripper():
    for cfg in D.CONFIGS:
        u = _env(cfg).unwrapped
        for ag in u.agent.agents:
            r = ag.record
            o = float(ag.measured_opening()[0])
            if r.kind == "arm_gripper":
                assert abs(o - r.gripper_max_opening_m) < 0.006, (cfg, r.name, o)
                jo = reg.joint_opening(ag, r)
                if jo is not None:
                    assert abs(float(jo[0]) - o) < 0.002, (cfg, r.name, float(jo[0]), o)
            else:
                assert o > 0.05, (cfg, r.name, o)


def test_inter_arm_contact_is_real():
    """swing both arms' base joints toward each other: the two robots must exchange contact impulses (if their
    collision-group bits were shared, as ManiSkill's Robotiq bit 31 would make them, they would pass through)"""
    cfg = "dual_xarm6_robotiq"
    env = _env(cfg)
    u = env.unwrapped
    from imitator.planning import scripted as S
    ags = list(u.agent.agents)
    ex = S.DualExecutor(env)
    A = {l._bodies[0].entity for l in ags[0].robot.links}
    B = {l._bodies[0].entity for l in ags[1].robot.links}
    hit = 0.0
    q0 = [t[0] for t in ex.targets]
    for t in np.linspace(0, 1, 60):
        ex.targets[0][0] = q0[0] - 1.2 * t          # left arm (y > 0) swings toward -y
        ex.targets[1][0] = q0[1] + 1.2 * t          # right arm swings toward +y
        ex.step()
        for c in u.scene.px.get_contacts():
            e0, e1 = c.bodies[0].entity, c.bodies[1].entity
            if (e0 in A and e1 in B) or (e0 in B and e1 in A):
                hit += sum(float(np.linalg.norm(q.impulse)) for q in c.points)
    assert hit > 0, "no arm-arm contact: inter-arm collisions are filtered"


# ------------------------------------------------------------------------------------------------ slow
def test_slow_pick_place_contract():
    if not SLOW:
        return
    from imitator.planning import scripted as S
    from imitator.verify import smoke as V
    env = D.make_dual_env("dual_arx_x5")
    rec = V.make_recorder(env)
    rec.reset(seed=0)
    log = S.pick_place(rec, variant="release")
    j = V.judge(rec)
    assert log["status"] == "ok" and j["success"] is True, (log["status"], j)
    rec.reset(seed=0)
    S.pick_place(rec, variant="hold")
    j = V.judge(rec)
    assert j["success"] is False, j


def test_slow_cpu_render():
    if not SLOW:
        return
    env = D.make_dual_env("dual_piper", render=True)
    env.reset(seed=0)
    img = env.render()[0].cpu().numpy()
    assert img.shape[2] == 3 and img.std() > 5


# ------------------------------------------------------------------------------------------------ runner
if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    order = [n for n in names if not n.startswith("test_slow")] + [n for n in names if n.startswith("test_slow")]
    fails = 0
    t0 = time.time()
    for n in order:
        t = time.time()
        try:
            globals()[n]()
            print(f"PASS {n} ({time.time() - t:.1f}s)", flush=True)
        except Exception:  # noqa: BLE001
            fails += 1
            print(f"FAIL {n}", flush=True)
            traceback.print_exc()
    print(f"{len(order) - fails}/{len(order)} passed in {time.time() - t0:.0f}s (slow={'on' if SLOW else 'off'})")
    sys.exit(1 if fails else 0)
