"""Tests for the E1 tactile pads (imitator/sensors): solver, ray kernels, default pads, CPU PhysX end-to-end, API hook.

    source research/env.sh; export HOME=$R/ext_data/sapien_home
    research/envs/imsim/bin/python -B -m pytest -q -p no:cacheprovider tests/imitator/test_tactile.py              # fast, CPU
    IMSIM_SLOW=1 research/envs/imsim/bin/python -B -m pytest -q -p no:cacheprovider tests/imitator/test_tactile.py # + GPU
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # repository root
PROJECT = ROOT
sys.path.insert(0, ROOT)

from imitator.sensors import geometry as G  # noqa: E402
from imitator.sensors import pads as PD  # noqa: E402
from imitator.sensors.tactile import (DEFAULT_K, PadGrid, bin_taxels, parse_layout, parse_level, shear_field,  # noqa: E402
                                   winkler_solve)
from _paths import DATA, REPO, TMP as _TMP, repo_path, needs_stock_tasks  # noqa: E402,F401
TMP = _TMP

SLOW = os.environ.get("IMSIM_SLOW", "") not in ("", "0")
DERIVED = os.path.join(DATA, "pads_derived.json")


def _cuda():
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def _grid(nx=16, ny=16, W=0.024, L=0.024, ss=2):
    g = PadGrid(nx, ny, ss, W, L)
    XY = torch.tensor(g.coords(), dtype=torch.float32)
    c = DEFAULT_K * torch.tensor(g.weights(), dtype=torch.float32)[None]
    return g, XY[:, 0][None], XY[:, 1][None], c


# ------------------------------------------------------------------------------------------------ layout / grid
def test_layout_level_parsing():
    assert parse_layout("12x32") == (12, 32) and parse_layout("16x16") == (16, 16) and parse_layout("flexitac") == (12, 32)
    assert parse_layout((8, 4)) == (8, 4)
    assert parse_level("T2") == 2 and parse_level("t0") == 0 and parse_level(None) == -1
    with pytest.raises(ValueError):
        parse_level("T3")


def test_pad_grid_reaches_the_boundary_and_bins_exactly():
    g = PadGrid(12, 32, 2, 0.024, 0.064)
    XY = g.coords()
    assert XY.shape == (g.S, 2) and g.S == 24 * 64
    assert np.isclose(XY[:, 0].min(), -0.012) and np.isclose(XY[:, 0].max(), 0.012)      # CoP reachable up to the edge
    assert np.isclose(g.weights().sum(), 0.024 * 0.064)
    M = g.binning()
    assert np.allclose(M.sum(1), 1.0) and np.allclose(M.sum(0), 4.0)                      # 2 x 2 sub-points per taxel
    v = torch.ones((1, g.S))
    assert torch.allclose(bin_taxels(v, torch.tensor(M, dtype=torch.float32), 12, 32), torch.full((1, 12, 32), 4.0))


# ------------------------------------------------------------------------------------------------ solver
def test_flat_uniform_and_offcentre_gradient():
    g, xs, ys, c = _grid()
    gap = torch.zeros((1, 1, g.S))
    f, _ = winkler_solve(gap, xs, ys, c, torch.tensor([[5.0]]), torch.zeros(1), torch.zeros(1))
    p = f[0, 0] / torch.tensor(g.weights(), dtype=torch.float32)
    assert abs(float(f.sum()) - 5.0) < 1e-5 and float(p.std() / p.mean()) < 1e-5
    e = 0.002                                                    # inside the kern W / 6: linear gradient, full contact
    f, _ = winkler_solve(gap, xs, ys, c, torch.tensor([[5.0]]), torch.tensor([5.0 * e]), torch.zeros(1))
    assert abs(float(f.sum()) - 5.0) < 1e-5
    assert abs(float((f[0, 0] * xs[0]).sum()) / 5.0 - e) < 1e-7
    p = (f[0, 0] / torch.tensor(g.weights(), dtype=torch.float32)).numpy()
    A = np.polyfit(xs[0].numpy(), p, 1)
    assert A[0] > 0 and abs(A[0] / (12 * 5.0 * e / (0.024 ** 2 * 0.024 ** 2)) - 1) < 0.05


@pytest.mark.parametrize("F", [1.0, 5.0, 20.0])
def test_sphere_patch_radius_matches_winkler_closed_form(F):
    g, xs, ys, c = _grid()
    R = 0.010
    r2 = xs ** 2 + ys ** 2
    gap = torch.where(r2 < R * R, R - torch.sqrt((R * R - r2).clamp_min(0)), torch.full_like(r2, 1e9))[None]
    f, _ = winkler_solve(gap, xs, ys, c, torch.tensor([[F]]), torch.zeros(1), torch.zeros(1))
    assert abs(float(f.sum()) - F) / F < 1e-5
    area = float(torch.tensor(g.weights())[f[0, 0] > 0].sum())
    a_num, a_th = math.sqrt(area / math.pi), (4 * R * F / (DEFAULT_K * math.pi)) ** 0.25
    assert abs(a_num - a_th) < 0.0008                           # < one sub-point pitch (0.75 mm)


def test_two_bodies_shared_tilt_and_geometric_mode():
    g, xs, ys, c = _grid()
    R = 0.010

    def sph(cx, cy):
        r2 = (xs - cx) ** 2 + (ys - cy) ** 2
        return torch.where(r2 < R * R, R - torch.sqrt((R * R - r2).clamp_min(0)), torch.full_like(r2, 1e9))
    gap = torch.stack([sph(-0.006, 0.0), sph(0.005, 0.004)], 1)
    Fn = torch.tensor([[2.0, 3.0]])
    Mx = torch.tensor([2 * -0.006 + 3 * 0.005]); My = torch.tensor([3 * 0.004])
    f, info = winkler_solve(gap, xs, ys, c, Fn, Mx, My)
    assert torch.allclose(f.sum(-1), Fn, rtol=1e-5)
    assert abs(float((f * xs[:, None]).sum()) - float(Mx)) < 1e-6 and abs(float((f * ys[:, None]).sum()) - float(My)) < 1e-6
    f2, info2 = winkler_solve(gap, xs, ys, c, Fn, Mx, My, tilt=False)
    assert torch.allclose(f2.sum(-1), Fn, rtol=1e-5) and float(info2["a"].abs().max()) == 0.0


def test_infeasible_cop_is_bounded_and_force_exact():
    g, xs, ys, c = _grid()
    gap = torch.zeros((1, 1, g.S))
    f, info = winkler_solve(gap, xs, ys, c, torch.tensor([[5.0]]), torch.tensor([5 * 0.05]), torch.zeros(1),
                            amax=0.2, bmax=0.2)
    assert abs(float(f.sum()) - 5.0) < 1e-4 and torch.isfinite(f).all()
    assert float(info["a"].abs()) <= 0.2 + 1e-6
    cx = float((f * xs).sum()) / 5.0
    assert 0.009 < cx <= 0.012


def test_no_surface_body_gets_no_force():
    g, xs, ys, c = _grid()
    gap = torch.full((1, 1, g.S), 1e9)
    f, info = winkler_solve(gap, xs, ys, c, torch.tensor([[3.0]]), torch.zeros(1), torch.zeros(1))
    assert float(f.sum()) == 0.0 and bool(info["no_surface"][0, 0])


def test_shear_field_force_and_twist():
    g, xs, ys, c = _grid()
    f, _ = winkler_solve(torch.zeros((1, 1, g.S)), xs, ys, c, torch.tensor([[5.0]]), torch.tensor([0.004]), torch.zeros(1))
    t = shear_field(f[:, 0], xs, ys, torch.tensor([[0.7, -0.4]]), torch.tensor([0.003]))
    assert torch.allclose(t.sum(-2), torch.tensor([[0.7, -0.4]]), atol=1e-6)
    assert abs(float((xs * t[..., 1] - ys * t[..., 0]).sum()) - 0.003) < 1e-7


# ------------------------------------------------------------------------------------------------ ray kernels
def test_ray_kernels_match_analytic_gaps_cpu():
    import trimesh
    bank = G.GeomBank("cpu")
    I = np.array([1.0, 0, 0, 0])
    box = trimesh.creation.box((0.04, 0.06, 0.02))
    shapes = {"box": G.Shape(G.BOX, np.zeros(3), I, np.r_[0.02, 0.03, 0.01, 0]),
              "sphere": G.Shape(G.SPHERE, np.zeros(3), I, np.r_[0.01, 0, 0, 0]),
              "capsule": G.Shape(G.CAPSULE, np.zeros(3), I, np.r_[0.005, 0.02, 0, 0]),
              "cylinder": G.Shape(G.CYLINDER, np.zeros(3), I, np.r_[0.005, 0.02, 0, 0]),
              "plane": G.Shape(G.PLANE, np.zeros(3), np.r_[np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0], np.zeros(4)),
              "convex": G.Shape(G.CONVEX, np.zeros(3), I, np.zeros(4), planes=G.hull_planes(box.vertices)),
              "trimesh": G.Shape(G.TRIMESH, np.zeros(3), I, np.zeros(4), verts=box.vertices, tris=box.faces)}
    rows = {k: bank.add_body([s]) for k, s in shapes.items()}
    B = len(rows)
    body_pos = torch.tensor([[0, 0, 0.05]] * B, dtype=torch.float32)
    body_pos[list(rows).index("plane")] = torch.tensor([0, 0, 0.03])
    body_q = torch.tensor([[1.0, 0, 0, 0]] * B)
    xs = torch.tensor([0.0, 0.004, 0.009, 0.015, 0.024])
    sub = torch.stack([xs, torch.zeros_like(xs), torch.full_like(xs, -0.01)], -1)[None]
    t = bank.cast(torch.zeros((B, 3)), torch.tensor([[1.0, 0, 0, 0]] * B), sub, torch.arange(B, dtype=torch.int32)[:, None],
                  body_pos, body_q)
    gap = (t - 0.01).numpy()[:, 0, :]
    x = xs.numpy()
    inf = np.inf
    exp = {"box": np.where(np.abs(x) <= 0.02, 0.04, inf),
           "sphere": np.where(np.abs(x) <= 0.01, 0.05 - np.sqrt(np.maximum(0.01 ** 2 - x ** 2, 0)), inf),
           "capsule": np.where(np.abs(x) <= 0.02, 0.045, 0.05 - np.sqrt(np.maximum(0.005 ** 2 - (np.abs(x) - 0.02) ** 2, 0))),
           "cylinder": np.where(np.abs(x) <= 0.02, 0.045, inf), "plane": np.full_like(x, 0.03)}
    exp["convex"] = exp["trimesh"] = exp["box"]
    for i, k in enumerate(rows):
        got = np.where(gap[i] > 1e8, inf, gap[i])
        assert np.all((np.isinf(exp[k]) & np.isinf(got)) | (np.abs(exp[k] - got) < 2e-6)), (k, got, exp[k])


# ------------------------------------------------------------------------------------------------ pads
def test_default_pad_tables_are_orthonormal_and_cover_all_embodiments():
    for uid in ("panda", "imsim_ur5_wsg", "imsim_arx_x5", "imsim_piper", "imsim_xarm6_robotiq", "imsim_xarm6_allegro_left"):
        assert uid in PD.AGENT_PADS
        for p in PD.pads_for_table(PD.AGENT_PADS[uid]):
            R = p.rot()
            assert np.allclose(R.T @ R, np.eye(3), atol=1e-6) and np.linalg.det(R) > 0.999
            assert 0.005 < p.size[0] < 0.05 and 0.005 < p.size[1] < 0.08


@pytest.mark.skipif(not os.path.exists(DERIVED), reason="run research/scripts/e1_tactile_build_pads.py first")
def test_default_pads_match_the_hull_derivation():
    d = json.load(open(DERIVED))
    for rec, table in PD.RECORD_PADS.items():
        for link, (o, R, size) in table.items():
            ref = d[rec][link]
            n_ref = np.asarray(ref["R"])[:, 2]
            n = np.asarray(PD._orthonormal(R))[:, 2]
            assert np.degrees(np.arccos(np.clip(n @ n_ref / np.linalg.norm(n_ref), -1, 1))) < 0.5, (rec, link)
            assert np.allclose(o, ref["origin"], atol=6e-5), (rec, link)
            assert np.allclose(size, ref["size"], atol=6e-5), (rec, link)


# ------------------------------------------------------------------------------------------------ CPU PhysX end-to-end
def test_press_rig_cpu_sphere_and_edge():
    import gymnasium as gym
    from imitator.sensors import rigs
    from imitator.sensors.tactile import TactileSensor
    for ind in ([dict(kind="sphere", r=0.01, mass=0.4, xy=(0.003, -0.002), com_offset=[0, 0, -0.0005])],
                [dict(kind="cylinder", r=0.001, hl=0.006, mass=0.3, xy=(0.0, 0.001), yaw=math.pi / 2, com_offset=[0, 0, -0.0005])]):
        rig = dict(pad=dict(W=0.016, L=0.016, t=0.004), indenters=ind)
        env = gym.make("ImsimTactilePressRig-v1", num_envs=1, sim_backend="physx_cpu", rig=rig, render_backend="none",
                       obs_mode="state")
        s = TactileSensor(env, "T2", pads=[rigs.press_pad_spec(rig)], layout="16x16")
        env.reset(seed=0); s.on_reset()
        Fs = []
        for i in range(60):
            s.begin_step(); env.step(None); out = s.end_step()
            if i >= 40:
                Fs.append(float(out["force"][0, 0, 2]))
        w = ind[0]["mass"] * 9.81
        Fn = float(out["force"][0, 0, 2])
        assert abs(np.mean(Fs) - w) / w < 0.01            # a resting r = 1 mm cylinder keeps rocking: compare a 20-step mean
        cop_ref = env.unwrapped.indenters[0].pose.p[0, :2].numpy()           # contact under the (resting) indenter
        assert np.allclose(out["cop"][0, 0].numpy(), cop_ref, atol=1e-4)
        P = out["pressure"][0, 0]
        assert abs(float(P.sum()) * s.grids[0].taxel_area - Fn) / Fn < 1e-4
        assert bool(out["contact"][0, 0]) and float(out["shear"].abs().max()) < 1e-3 * float(P.max())
        env.close()


@needs_stock_tasks()
def test_make_tactile_opt_in_cpu():
    import imitator
    env = imitator.make("PickCube-v1", "panda", sim="cpu", render="none")
    obs, _ = env.reset(seed=0)
    assert torch.is_tensor(obs)                                  # default stays off: flat state
    env.close()
    env = imitator.make("PickCube-v1", "panda", sim="cpu", render="none", tactile="T2", tactile_kw=dict(layout="16x16"))
    obs, _ = env.reset(seed=0)
    assert set(obs) == {"state", "tactile"} and obs["tactile"]["pressure"].shape == (1, 2, 16, 16)
    a = torch.zeros(env.unwrapped.single_action_space.shape)
    obs, *_ = env.step(a)
    t = obs["tactile"]
    assert t["force"].shape == (1, 2, 3) and t["cop"].shape == (1, 2, 2) and t["shear"].shape == (1, 2, 16, 16, 2)
    assert env.unwrapped.imsim["tactile"]["level"] == "T2" and env.sensor.pad_names == ["panda_leftfinger", "panda_rightfinger"]
    env.close()


# ------------------------------------------------------------------------------------------------ GPU (subprocess)
GPU_SCRIPT = r"""
import math, numpy as np, torch, gymnasium as gym
from imitator import compat; compat.ensure_sapien_home()
from imitator.sensors import rigs
from imitator.sensors.tactile import TactileSensor
rig = dict(pad=dict(W=0.016, L=0.016, t=0.004), indenters=[dict(kind="sphere", r=0.008, mass=0.2, xy=(-0.004, -0.003), com_offset=[0, 0, -0.0005]),
                                                           dict(kind="sphere", r=0.006, mass=0.4, xy=(0.004, 0.004), com_offset=[0, 0, -0.0005])])
env = gym.make("ImsimTactilePressRig-v1", num_envs=3, sim_backend="physx_cuda", rig=rig, render_backend="none", obs_mode="state")
s = TactileSensor(env, "T2", pads=[rigs.press_pad_spec(rig)], layout="16x16")
env.reset(seed=0); s.on_reset()
for _ in range(40):
    s.begin_step(); env.step(None); out = s.end_step()
F = out["force"][:, 0, 2].cpu().numpy(); w = 0.6 * 9.81
assert np.all(np.abs(F - w) / w < 0.005), F
P = out["pressure"][:, 0].sum((-1, -2)).cpu().numpy() * s.grids[0].taxel_area
assert np.all(np.abs(P - F) / F < 1e-4), (P, F)
cop = out["cop"][:, 0].cpu().numpy()
P = [a.pose.p[:, :2].cpu().numpy() for a in env.unwrapped.indenters]       # resting positions (they settle ~1 mm)
ref = (P[0] * 0.2 + P[1] * 0.4) / 0.6
assert np.all(np.abs(cop - ref) < 3e-4), (cop, ref)
print("GPU_OK")
"""


@pytest.mark.skipif(not (SLOW and _cuda()), reason="set IMSIM_SLOW=1 (GPU PhysX, subprocess)")
def test_slow_gpu_press_rig_two_spheres():
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, "-B", "-c", GPU_SCRIPT], capture_output=True, text=True, env=env, timeout=900)
    assert "GPU_OK" in r.stdout, r.stdout[-2000:] + r.stderr[-4000:]
