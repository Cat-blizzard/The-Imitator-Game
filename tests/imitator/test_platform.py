"""Tests for the E23 platform layer: imitator.make / evaluate / CLI, contract v3.1, DualHandover-v1, DualFoldTowel-v1, GPU
batched evaluation, Newton rigid camera.

    source research/env.sh; export HOME=$R/ext_data/sapien_home
    research/envs/imsim/bin/python -B -m pytest -q -p no:cacheprovider tests/imitator/test_platform.py      # fast (~1-2 min)
    IMSIM_SLOW=1 research/envs/imsim/bin/python -B -m pytest -q -p no:cacheprovider tests/imitator/test_platform.py  # + episodes
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # repository root
PROJECT = ROOT
sys.path.insert(0, ROOT)

import imitator  # noqa: E402
from imitator.verify import handover as HO  # noqa: E402
from imitator.verify import v31  # noqa: E402
from _paths import DATA, REPO, TMP as _TMP, repo_path  # noqa: E402,F401
TMP = _TMP

SLOW = os.environ.get("IMSIM_SLOW", "") not in ("", "0")
FREEZE = os.path.join(DATA, "e23_contract_freeze.json")


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# ------------------------------------------------------------------------------------------------ registry / api
def test_list_tasks_and_embodiments():
    t = imitator.list_tasks()
    assert {"DualCubePlace-v1", "DualHandover-v1", "DualFoldTowel-v1", "FoldTowel-v1"} <= set(t)
    e = imitator.list_embodiments()
    assert {"dual_arx_x5", "dual_ur5_wsg", "dual_piper", "dual_xarm6_robotiq", "dual_xarm6_allegro"} <= set(e)
    d = imitator.list_embodiments(detail=True)
    assert d["dual_arx_x5"]["gpu_sim_freq"] == 200 and d["dual_piper"]["gpu_sim_freq"] == 500
    for name in t:
        spec = imitator.get_task(name)
        assert spec.default_embodiment in spec.embodiments
        for b in spec.scripts:
            c = spec.load_contract()
            assert any(x.name == b for x in c.counterexamples) or spec.name == "FoldTowel-v1", (name, b)


def test_choose_render_auto(monkeypatch):
    from imitator import api
    monkeypatch.setattr(api, "vulkan_info", lambda: dict(any=True, cuda=False))
    assert api.choose_render(True, "cpu", None) == "newton"          # cloth -> Newton camera
    assert api.choose_render(False, "cpu", None) == "sapien_cpu"     # lavapipe
    assert api.choose_render(False, "gpu", None) == "newton"         # GPU physics without CUDA-Vulkan
    monkeypatch.setattr(api, "vulkan_info", lambda: dict(any=False, cuda=False))
    assert api.choose_render(False, "cpu", None) == "newton"         # no Vulkan at all
    monkeypatch.setattr(api, "vulkan_info", lambda: dict(any=True, cuda=True))
    assert api.choose_render(False, "gpu", None) == "sapien_gpu"
    assert api.choose_render(False, "cpu", "none") == "none"
    with pytest.raises(ValueError):
        api.choose_render(False, "cpu", "vulkan")


def test_make_rejects_bad_args():
    with pytest.raises(ValueError):
        imitator.make("DualHandover-v1", "dual_xarm6_allegro")           # hands not supported for the handover
    with pytest.raises(ValueError):
        imitator.make("DualCubePlace-v1", "dual_ur5_wsg", num_envs=4, sim="cpu")


# ------------------------------------------------------------------------------------------------ contracts
def test_new_contracts_valid_and_hash_frozen():
    import warnings
    from veritask.adapters.protocol_v3 import check_contract_v3
    d = json.load(open(FREEZE))
    assert d["unchanged_reference"]["sha256"] == d["unchanged_reference"]["expected_sha256_from_e19"]
    v3 = d["unchanged_reference"]["contract"]
    assert hashlib.sha256(open(v3, "rb").read()).hexdigest() == "ad15c3bd9bb92aba52d24a9b12d93441ed1558ae0a6c1801ef6a4600a52def0a"
    for c in d["contracts"]:
        assert hashlib.sha256(open(c["contract"], "rb").read()).hexdigest() == c["sha256"], c["contract"]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            from veritask.spec.contract import TaskContract
            k = TaskContract.load(c["contract"])
        assert check_contract_v3(k) == [], c["contract"]


def test_retreat_term_in_verdict():
    """v3.1: the aftermath measurement decides; missing -> unknown"""
    c = v31.load_contract("dual_cube_place_v3.1")
    n_pol, n_set = 30, 40
    tr = {k: np.zeros(n_pol + n_set) for k in ("cube_x", "cube_y", "cube_z", "cube_speed", "cube_bottom_dz_m",
                                               "robot_qvel_max", "robot_cube_impulse_Ns")}
    tr["cube_target_margin_m"] = np.full(n_pol + n_set, 0.02)
    tr["cube_enclosure_all_m"] = np.ones(n_pol + n_set)
    tr["phase"] = np.r_[np.zeros(n_pol), np.ones(n_set)]
    tr["bench_success"] = np.ones(n_pol + n_set)
    assert v31.judge_traces(c, tr, 20.0, {v31.RETREAT_METRIC: 0.0})["success"] is True
    assert v31.judge_traces(c, tr, 20.0, {v31.RETREAT_METRIC: 0.03})["success"] is False
    assert v31.judge_traces(c, tr, 20.0, {})["success"] is None
    tr2 = dict(tr); tr2["cube_enclosure_all_m"] = np.full(n_pol + n_set, 0.001)        # enclosed -> not released
    assert v31.judge_traces(c, tr2, 20.0, {v31.RETREAT_METRIC: 0.0})["success"] is False


# ------------------------------------------------------------------------------------------------ geometry
def test_box_distance_anisotropic():
    T = np.eye(4); T[:3, 3] = [0.1, 0.2, 0.3]
    h = np.array([0.02, 0.09, 0.02])
    P = np.array([[0.1, 0.2, 0.3], [0.1 + 0.05, 0.2, 0.3], [0.1, 0.2 + 0.10, 0.3]])
    d = v31.box_distance(P, T, h)
    assert np.allclose(d, [-0.02, 0.03, 0.01], atol=1e-9)
    S = v31.box_surface_points(h, 5)
    assert np.allclose(np.abs(v31.box_distance(S + T[:3, 3], T, h)), 0, atol=1e-9)


def test_enclosure_all_between_open_fingers_fk():
    """FK of an open WSG-50: a cube between the pads is enclosed (clearance > 0); moved 10 cm away it is not"""
    from imitator.embodiments import registry as reg
    from imitator.embodiments.urdf_tools import load_urdf
    r = reg.ARMS["ur5_wsg"]
    m = load_urdf(r.urdf_path)
    Ts = m.fk(r.home_qpos())
    g = reg.pad_geometry(r, Ts)
    c = (g["c1"] + g["c2"]) / 2
    Tb = np.eye(4); Tb[:3, :3] = Ts[r.tcp_link][:3, :3]; Tb[:3, 3] = c
    half = 0.025
    a = v31.all_links_clearance(r, Ts, Tb, half)
    p = v31.pad_clearance(r, Ts, Tb, half)
    assert a is not None and p is not None and 0.0 < a <= p + 1e-9 and a < g["opening"]
    Tb2 = Tb.copy(); Tb2[:3, 3] = c + Ts[r.tcp_link][:3, 2] * 0.15
    assert v31.all_links_clearance(r, Ts, Tb2, half) is None
    assert v31.gripper_reach(r) > 0.05


def test_cloth_close_command_gap():
    from imitator.embodiments import registry as reg
    from imitator.soft import embodiment_colliders as EC
    for n in ("ur5_wsg", "arx_x5", "piper"):
        r = reg.ARMS[n]
        a = EC.cloth_close_command(r, 0.004)
        q = r.gripper_lower + (a + 1) / 2 * (r.gripper_upper - r.gripper_lower)
        assert abs(EC.opening_at(r, q) - 0.004) < 2e-4, n
        cols, pads = EC.embodiment_colliders(r, "L:")
        assert pads is not None and pads.left in [c.link for c in cols]


# ------------------------------------------------------------------------------------------------ handover signal
def _ho_traces(L, R, T, dz):
    return dict(armL_obj_impulse_Ns=np.asarray(L, float), armR_obj_impulse_Ns=np.asarray(R, float),
                table_obj_impulse_Ns=np.asarray(T, float), obj_bottom_dz_m=np.asarray(dz, float))


def test_handover_signal_cases():
    n = 40
    on, off = np.ones(n), np.zeros(n)
    # table 0-9, left carry 10-19, both 20-24, right carry 25-34, table 35-39
    L = np.r_[off[:10], on[:15], off[:15]]; R = np.r_[off[:20], on[:15], off[:5]]
    T = np.r_[on[:10], off[:25], on[:5]]; dz = np.r_[off[:10], np.full(25, 0.05), off[:5]]
    s = HO.handover_signal(_ho_traces(L, R, T, dz))
    assert s["handover_done"].max() == 1 and s["_info"]["t_done"] == 29
    # drop between the carries -> no handover
    T2 = T.copy(); T2[22] = 1; dz2 = dz.copy(); dz2[22] = 0
    assert HO.handover_signal(_ho_traces(L, R, T2, dz2))["handover_done"].max() == 0
    # right picks from the table (left never carries) -> no handover
    assert HO.handover_signal(_ho_traces(off, R, T, dz))["handover_done"].max() == 0
    # both always touching -> no solo carry -> no handover
    assert HO.handover_signal(_ho_traces(np.r_[off[:10], np.ones(30)], np.r_[off[:10], np.ones(30)],
                                         np.r_[on[:10], np.zeros(30)], np.r_[off[:10], np.full(30, 0.05)]))["handover_done"].max() == 0
    # right carry shorter than carry_steps -> no handover
    R3 = np.r_[off[:20], np.ones(8), np.zeros(12)]
    assert HO.handover_signal(_ho_traces(L, R3, T, dz))["handover_done"].max() == 0


def test_handover_contract_rejects_no_handover_traces():
    c = v31.load_contract("dual_handover_v1")
    n = 80
    tr = {k: np.zeros(n) for k in ("obj_x", "obj_y", "obj_z", "obj_speed", "obj_bottom_dz_m", "robot_qvel_max",
                                   "robot_obj_impulse_Ns", "armL_obj_impulse_Ns", "armR_obj_impulse_Ns", "table_obj_impulse_Ns")}
    tr["obj_target_margin_m"] = np.full(n, 0.02); tr["obj_enclosure_all_m"] = np.ones(n)
    tr["phase"] = np.r_[np.zeros(40), np.ones(40)]; tr["bench_success"] = np.ones(n)
    derive = HO.make_derive(c)
    j = v31.judge_traces(c, tr, 20.0, {v31.RETREAT_METRIC: 0.0}, derive=derive)
    assert j["success"] is False and j["binding_term"] == "trace.max:handover_done"
    tr2 = dict(tr)
    tr2["armL_obj_impulse_Ns"] = np.r_[np.zeros(5), np.ones(10), np.zeros(65)]
    tr2["armR_obj_impulse_Ns"] = np.r_[np.zeros(12), np.ones(15), np.zeros(53)]
    tr2["obj_bottom_dz_m"] = np.r_[np.zeros(5), np.full(25, 0.05), np.zeros(50)]
    tr2["table_obj_impulse_Ns"] = np.r_[np.ones(5), np.zeros(25), np.ones(50)]
    j2 = v31.judge_traces(c, tr2, 20.0, {v31.RETREAT_METRIC: 0.0}, derive=derive)
    assert j2["success"] is True, j2


# ------------------------------------------------------------------------------------------------ envs (CPU, cheap)
def test_make_handover_cpu_and_step():
    env = imitator.make("DualHandover-v1", "dual_ur5_wsg", sim="cpu", render="none")
    obs, info = env.reset(seed=0)
    u = env.unwrapped
    assert u.imsim["render"] == "none" and u.block_half[1] == 0.09
    assert float(u.block.pose.p[0, 1]) > 0.1 and float(u.target_pos[0, 1]) < -0.1     # left zone, right target
    acts = {k: ag.home_action() for k, ag in zip(u.single_action_space.spaces.keys(), u.agent.agents)}
    for _ in range(3):
        obs, r, te, tr, info = env.step(acts)
    assert "success" in info and not bool(info["success"][0])
    env.close()


def test_cli_list():
    out = subprocess.run([sys.executable, "-B", "-m", "imitator", "list", "--json"], capture_output=True, text=True,
                         cwd=ROOT, timeout=300, env=dict(os.environ))
    assert out.returncode == 0, out.stderr[-2000:]
    d = json.loads(out.stdout[out.stdout.index("{"):])
    assert "DualHandover-v1" in d["tasks"] and "dual_piper" in d["embodiments"]


# ------------------------------------------------------------------------------------------------ slow
@pytest.mark.skipif(not SLOW, reason="set IMSIM_SLOW=1 (scripted episodes)")
def test_slow_evaluate_cube_v31(tmp_path_factory):
    out = os.path.join(TMP, "test_platform_cube")
    r = imitator.evaluate("DualCubePlace-v1", "dual_ur5_wsg", "expert", seeds=[0], out_dir=out, verbose=False)
    v = {x["contract"]: x["verdict"] for x in r["rows"]}
    assert v["imsim_dual_cube_place_v3.1"] is True and v["imsim_dual_cube_place_v3"] is True
    r = imitator.evaluate("DualCubePlace-v1", "dual_ur5_wsg", "hold", seeds=[0], out_dir=out, verbose=False)
    assert all(x["verdict"] is False for x in r["rows"])


@pytest.mark.skipif(not SLOW, reason="set IMSIM_SLOW=1 (scripted episodes)")
def test_slow_handover_expert_and_no_handover():
    out = os.path.join(TMP, "test_platform_ho")
    r = imitator.evaluate("DualHandover-v1", "dual_ur5_wsg", "handover", seeds=[0], out_dir=out, verbose=False)
    assert r["rows"][0]["verdict"] is True and r["rows"][0]["terms"]["trace.max:handover_done"] == 1.0
    r = imitator.evaluate("DualHandover-v1", "dual_ur5_wsg", "no_handover", seeds=[0], out_dir=out, verbose=False)
    assert r["rows"][0]["verdict"] is False and r["rows"][0]["bench_final"] is True


def _subprocess_json(code: str, timeout: int = 1800) -> dict:
    """GPU PhysX can only be enabled before any CPU PhysX use in a process (SAPIEN): run GPU checks in a fresh process"""
    out = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, cwd=ROOT, timeout=timeout,
                         env=dict(os.environ))
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads([l for l in out.stdout.splitlines() if l.startswith("RESULT ")][-1][len("RESULT "):])


@pytest.mark.skipif(not (SLOW and _cuda()), reason="set IMSIM_SLOW=1 (GPU PhysX + Newton)")
def test_slow_dual_fold_expert_vs_drag():
    out = os.path.join(TMP, "test_platform_fold")
    v = _subprocess_json(f"""
import json, shutil, imitator
shutil.rmtree({out!r}, ignore_errors=True)
r = imitator.evaluate("DualFoldTowel-v1", "dual_ur5_wsg", "all", seeds=[0], num_envs=8, out_dir={out!r}, verbose=False)
d = {{}}
for x in r["rows"]:
    d.setdefault(x["behaviour"], []).append(x["verdict"])
print("RESULT " + json.dumps(d))
""")
    # expert success is a rate (13-14 of 16 worlds in E23): require >= 1 of 2 expert worlds, no failure accepted
    assert any(x is True for x in v["expert"]), v
    assert not any(x is True for b in ("no_grip", "drop_midway", "drag_on_ground") for x in v[b]), v


@pytest.mark.skipif(not (SLOW and _cuda()), reason="set IMSIM_SLOW=1 (GPU PhysX)")
def test_slow_gpu_batched_cube():
    out = os.path.join(TMP, "test_platform_gpu")
    v = _subprocess_json(f"""
import json, shutil, imitator
shutil.rmtree({out!r}, ignore_errors=True)
r = imitator.evaluate("DualCubePlace-v1", "dual_ur5_wsg", "expert", seeds=[0], sim="gpu", num_envs=4, out_dir={out!r},
                   verbose=False, save_traces=False)
rows = [x for x in r["rows"] if x["contract"] == "imsim_dual_cube_place_v3.1" and x["executed"]]
print("RESULT " + json.dumps(dict(n=len(rows), acc=sum(bool(x["verdict"]) for x in rows))))
""")
    assert v["n"] >= 3 and v["acc"] >= v["n"] - 1


@pytest.mark.skipif(not (SLOW and _cuda()), reason="set IMSIM_SLOW=1 (Newton camera)")
def test_slow_newton_rigid_camera():
    env = imitator.make("DualHandover-v1", "dual_ur5_wsg", sim="cpu", render="newton", obs="rgb")
    obs, _ = env.reset(seed=0)
    img = obs["sensor_data"]["newton_camera"]["rgb"]
    assert tuple(img.shape) == (1, 128, 128, 3) and float(img.float().std()) > 5.0
    env.close()
