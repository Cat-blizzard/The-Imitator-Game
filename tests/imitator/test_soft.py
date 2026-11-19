"""Tests for imitator.soft (E20). Fast tests run on CPU; GPU tests need CUDA + newton/warp (research/envs/imsim_soft);
env-level smoke tests (route A needs SAPIEN GPU PhysX, route B MuJoCo-Warp; minutes on a cold kernel cache) run only
with IMSIM_SLOW=1.

    source research/env.sh; HOME=$R/ext_data/sapien_home $R/envs/imsim_soft/bin/python -B -m pytest -q tests/imitator/test_soft.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "imsim"))
sys.path.insert(0, os.path.join(ROOT, "veritask"))


def _cuda():
    try:
        import torch
        import warp  # noqa: F401
        import newton  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


CUDA = _cuda()
SLOW = os.environ.get("IMSIM_SLOW") == "1"


# ------------------------------------------------------------------------------------------------ pure python / numpy
def test_waypoints_behaviours_differ_where_they_should():
    from imitator.soft.expert import towel_waypoints, interp_waypoints, total_time
    e, n, d, g = (towel_waypoints(b) for b in ("expert", "no_grip", "drop_midway", "drag_on_ground"))
    assert total_time(e) == pytest.approx(total_time(n))
    assert max(w[3] for w in n) == 0                              # no_grip never closes
    assert max(w[2] for w in g if w[3] == 1) < 0.02               # drag keeps the closed gripper at grasp height
    # drop_midway opens while the gripper is high and still far from C
    opened = [w for i, w in enumerate(d[1:], 1) if d[i - 1][3] == 1 and w[3] == 0][0]
    assert opened[2] > 0.05 and np.hypot(opened[0] - 0.12, opened[1] - 0.12) > 0.1
    x = interp_waypoints(e, 0.0)
    assert x[3] == 0 and x[2] == pytest.approx(0.08)


def test_panda_fk_matches_sapien_rest_tcp_and_ik_converges():
    from imitator.soft.panda_ik import UrdfChain, PANDA_REST, joint_trajectory
    from imitator.soft.expert import towel_waypoints, interp_waypoints
    c = UrdfChain()
    T, _, _ = c.fk(PANDA_REST)
    # SAPIEN/ManiSkill: TCP of the Panda 'rest' keyframe, robot base at (-0.615, 0, 0) -> world (0, 0, 0.1698), pointing down
    assert np.allclose(T[:3, 3], [0.615, 0.0, 0.1698], atol=2e-4)
    assert np.allclose(T[:3, :3], np.diag([1, -1, -1]), atol=1e-3)
    w = towel_waypoints("expert", center=(-0.05, 0.0))
    pts = np.array([interp_waypoints(w, 0.05 * k)[:3] for k in range(1, 120)])
    q, err = joint_trajectory(pts)
    assert err < 1e-4 and np.all(np.isfinite(q))
    assert np.abs(np.diff(q[1:], axis=0)).max() < 0.2            # smooth (warm-started) joint path


def test_gripper_link_poses_geometry():
    from imitator.soft.expert import panda_gripper_link_poses
    P = panda_gripper_link_poses([0.1, 0.2, 0.05], 0.01)
    hand, lf, rf = P
    assert hand[2] == pytest.approx(0.05 + 0.1034, abs=1e-6)       # hand above the TCP when pointing down
    assert np.linalg.norm(lf[:3] - rf[:3]) == pytest.approx(0.02, abs=1e-6)


def test_footprint_area_matches_scipy_hull():
    torch = pytest.importorskip("torch")
    from scipy.spatial import ConvexHull
    from imitator.soft.metrics import footprint_area
    rng = np.random.default_rng(0)
    xy = rng.normal(size=(5, 300, 2)).astype(np.float32)
    xy[0] = np.stack(np.meshgrid(np.linspace(-.15, .15, 25), np.linspace(-.15, .15, 25)), -1).reshape(-1, 2)[:300]
    a = footprint_area(torch.tensor(xy)).numpy()
    ref = np.array([ConvexHull(x).volume for x in xy])
    assert np.allclose(a, ref, rtol=2e-3)
    # towel-like states: flat grid and a diagonally folded grid (vertices reflected across the anti-diagonal)
    g = np.stack(np.meshgrid(np.linspace(-.15, .15, 25), np.linspace(-.15, .15, 25)), -1).reshape(-1, 2)
    f = g.copy(); m = f[:, 0] + f[:, 1] < 0; f[m] = -f[m][:, ::-1]
    for x in (g, f):
        assert footprint_area(torch.tensor(x[None], dtype=torch.float32)).item() == pytest.approx(ConvexHull(x).volume, rel=1e-3)


def test_v3_contract_is_valid_and_protocol_clean():
    from veritask.spec.contract import TaskContract
    from veritask.adapters.protocol_v3 import check_contract_v3
    from imitator.soft.contract_eval import V1_PATH, V3_PATH
    c3 = TaskContract.load(V3_PATH, strict=True)
    assert c3.protocol == "v3" and check_contract_v3(c3, 20.0) == []
    c1 = TaskContract.load(V1_PATH)
    # same thresholds as the frozen v1 towel contract
    thr1 = sorted(t.threshold for t in c1.outcome)
    thr3 = sorted(t.threshold for t in c3.outcome)
    assert thr1 == thr3


def _synthetic_traces(n_policy=40, n_settle=60, folded=True, released=True, quiet=True):
    tr = {k: np.zeros(n_policy + n_settle) for k in ("towel_corner_dist", "towel_footprint_ratio", "towel_frac_above_5cm",
                                                       "robot_cloth_impulse", "finger_cloth_enclosure",
                                                       "cloth_speed_p95", "robot_qvel_max", "phase")}
    tr["phase"][n_policy:] = 1
    tr["towel_corner_dist"][:] = 0.03 if folded else 0.2
    tr["towel_footprint_ratio"][:] = 0.55 if folded else 0.95
    tr["finger_cloth_enclosure"][:] = 1.0 if released else 0.0
    tr["robot_cloth_impulse"][:] = 0.0 if released else 0.5
    tr["cloth_speed_p95"][:] = 0.001 if quiet else 0.1
    n = n_policy + n_settle
    wobble = 0.0 if quiet else 0.01 * np.sin(np.arange(n))           # corners keep moving ~1 cm per step
    for c, (x, y) in (("A", (0.1, 0.1)), ("C", (0.12, 0.12))):
        tr[f"towel_{c}_x"] = np.full(n, x) + wobble
        tr[f"towel_{c}_y"] = np.full(n, y)
        tr[f"towel_{c}_z"] = np.full(n, 0.008)
    return tr


@pytest.mark.parametrize("folded,released,quiet,expect", [(True, True, True, True), (False, True, True, False),
                                                          (True, False, True, False), (True, True, False, False)])
def test_contract_eval_on_synthetic_traces(folded, released, quiet, expect):
    from veritask.adapters.base import evaluate_traces
    from imitator.soft.contract_eval import load_contracts
    c3, _ = load_contracts()
    v, bench = evaluate_traces(c3, _synthetic_traces(folded=folded, released=released, quiet=quiet), control_freq=20.0)
    assert v["success"] is expect


# ------------------------------------------------------------------------------------------------ GPU: cloth core
@pytest.mark.skipif(not CUDA, reason="needs CUDA + newton")
def test_cloth_rest_is_stable_and_flat():
    import torch
    from imitator.soft.cloth import NewtonClothWorlds, TowelSpec, ClothSolverSpec
    from imitator.soft.colliders import PANDA_COLLIDERS, PANDA_PADS
    from imitator.soft.expert import panda_gripper_link_poses
    cl = NewtonClothWorlds(2, PANDA_COLLIDERS, PANDA_PADS, TowelSpec(), ClothSolverSpec(substeps=10))
    P = torch.tensor(np.tile(panda_gripper_link_poses([0.0, 0.0, 0.4], 0.04)[None], (2, 1, 1)), device="cuda")
    cl.set_link_poses(P, first=True)
    for _ in range(10):
        cl.set_link_poses(P); cl.step()
    q = cl.particle_q().cpu().numpy(); s = {k: v.cpu().numpy() for k, v in cl.signals().items()}
    assert np.isfinite(q).all()
    assert q[..., 2].min() > 0.0 and q[..., 2].max() < 0.01
    assert (s["robot_cloth_impulse"] == 0).all() and (s["finger_cloth_enclosure"] == 1.0).all()
    assert s["towel_corner_dist"] == pytest.approx(np.hypot(0.3, 0.3), abs=2e-3)


@pytest.mark.skipif(not CUDA, reason="needs CUDA + newton")
@pytest.mark.parametrize("mode", ["friction", "attach"])
def test_kinematic_pinch_lifts_corner_only_when_closed(mode):
    import torch
    from imitator.soft.cloth import NewtonClothWorlds, TowelSpec, ClothSolverSpec
    from imitator.soft.colliders import PANDA_COLLIDERS, PANDA_PADS
    from imitator.soft.expert import panda_gripper_link_poses, towel_waypoints, interp_waypoints
    cl = NewtonClothWorlds(2, PANDA_COLLIDERS, PANDA_PADS, TowelSpec(), ClothSolverSpec(substeps=30, grasp_mode=mode))
    plans = [towel_waypoints("expert"), towel_waypoints("no_grip")]
    for k in range(0, 60):                                         # grasp + lift to 10 cm (3 s)
        P = []
        for w in plans:
            x, y, z, g = interp_waypoints(w, 0.05 * k)
            P.append(panda_gripper_link_poses([x, y, z], 0.04 * (1 - g) + 0.002 * g))
        P = torch.tensor(np.stack(P), device="cuda")
        if k == 0:
            cl.set_link_poses(P, first=True); continue
        cl.set_link_poses(P)
        if mode == "attach":
            cl.set_grip([int(interp_waypoints(w, 0.05 * k)[3] > 0.99) for w in plans])
        cl.step()
    q = cl.particle_q().cpu().numpy()
    assert np.isfinite(q).all()
    assert q[0, 0, 2] > 0.04, "closed gripper should have lifted corner A"
    assert q[1, :, 2].max() < 0.02, "open gripper must not lift the towel"


# ------------------------------------------------------------------------------------------------ slow: whole envs
@pytest.mark.skipif(not (CUDA and SLOW), reason="set IMSIM_SLOW=1 (builds a full env, minutes on a cold cache)")
def test_route_b_env_expert_vs_no_grip():
    import torch
    from imitator.soft.route_b import FoldTowelNewtonEnv, scripted_joint_action
    from imitator.soft.expert import joint_space_plan
    env = FoldTowelNewtonEnv(num_envs=2)
    env.reset(seed=0)
    qe, ge, _ = joint_space_plan("expert", center=env.towel_spec.center, n_steps=130)
    qn, gn, _ = joint_space_plan("no_grip", center=env.towel_spec.center, n_steps=130)
    for k in range(130):
        a = scripted_joint_action(env, torch.tensor(np.stack([qe[k], qn[k]])), torch.tensor([ge[k], gn[k]]))
        obs, rew, term, trunc, info = env.step(a)
    d = info["towel_corner_dist"].cpu().numpy()
    assert d[0] < 0.10 < d[1]


@pytest.mark.skipif(not (CUDA and SLOW), reason="set IMSIM_SLOW=1 (SAPIEN GPU PhysX + Newton, minutes)")
def test_route_a_env_link_poses_match_gripper_model():
    import gymnasium as gym
    import torch
    from imitator.soft.route_a import FoldTowelBridgeEnv  # noqa: F401
    from imitator.soft.expert import panda_gripper_link_poses
    env = gym.make("FoldTowelBridge-v1", num_envs=2, obs_mode="state", control_mode="pd_joint_pos",
                   sim_backend="physx_cuda", render_backend="none")
    u = env.unwrapped
    env.reset(seed=0)
    L = u.link_poses()[0].cpu().numpy()
    tcp = u.agent.tcp_pose.p[0].cpu().numpy(); qf = float(u.agent.robot.get_qpos()[0, 7])
    E = panda_gripper_link_poses(tcp, qf)
    assert np.allclose(L[:, :3], E[:, :3], atol=1e-3)
