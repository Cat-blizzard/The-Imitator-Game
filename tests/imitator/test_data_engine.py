"""E29a data engine: PerturbingEnv, storage (ManiSkill3 HDF5 + JSON), card and certification bookkeeping (fast; a fake
env, no simulator). The slow end-to-end test (IMSIM_SLOW=1) generates and replays real DualCubePlace episodes."""
import json
import os
import shutil
from collections import OrderedDict

import numpy as np
import pytest
import torch

from imitator.data import store
from imitator.data.engine import PerturbingEnv, certify_verdicts, contract_cause
from imitator.data.perturb import NoiseProcess, PerturbSpec, preset
from imitator.data.phase import CONTACT, FREE, set_phase
from _paths import DATA, REPO, TMP as _TMP, repo_path  # noqa: E402,F401
TMP = _TMP

ROOT = REPO


@pytest.fixture
def tmp_path_project(request):
    d = os.path.join(TMP, "pytest_e29a", request.node.name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ fake env
class _Robot:
    def __init__(self, n):
        self.q = torch.zeros(1, n)

    def get_qpos(self):
        return self.q


class _Ag:
    def __init__(self, n):
        self.robot = _Robot(n)


class _Agent:
    def __init__(self):
        self.agents = [_Ag(7), _Ag(7)]


class _U:
    num_envs = 1

    def __init__(self):
        self.agent = _Agent()
        self.t = 0

    def get_state_dict(self):
        return dict(actors=dict(cube=torch.full((1, 13), float(self.t))),
                    articulations=dict(r0=torch.cat([torch.zeros(1, 13), self.agent.agents[0].robot.q,
                                                     torch.zeros(1, 7)], 1)))


class FakeEnv:
    """dual-arm dict actions: arm 6 joint targets + 1 gripper per agent; the robot follows the targets exactly"""

    def __init__(self):
        self.unwrapped = _U()
        self.received = []

    def step(self, a):
        self.received.append({k: np.array(v, copy=True) for k, v in a.items()})
        u = self.unwrapped
        u.t += 1
        for ag, k in zip(u.agent.agents, ("a0", "a1")):
            ag.robot.q[0, :6] = torch.as_tensor(a[k][:6])
        obs = torch.full((1, 5), float(u.t))
        return obs, torch.zeros(1), torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), \
            dict(success=torch.tensor([u.t >= 3]))


def _layout():
    return store.Layout(["a0", "a1"], [7, 7], [(0, 6), (0, 6)], [list(range(6)), list(range(6))], {"x": (0, 5)}, 5,
                        "pd_joint_pos")


def _expert(env, n=6):
    for t in range(n):
        set_phase(env, FREE if t < 3 else CONTACT, arm=0)       # arm 1 never annotated: CONTACT
        env.step(OrderedDict(a0=np.r_[np.full(6, 0.1 * t), 1.0].astype(np.float32),
                             a1=np.r_[np.zeros(6), -1.0].astype(np.float32)))
    return dict(status="ok")


def test_layout_flatten_unflatten():
    lay = _layout()
    a = OrderedDict(a0=np.arange(7, dtype=np.float32), a1=np.arange(7, 14, dtype=np.float32))
    f = lay.flatten_action(a)
    assert f.shape == (14,) and np.allclose(f, np.arange(14))
    b = lay.unflatten_action(f)
    assert list(b) == ["a0", "a1"] and np.allclose(b["a1"], np.arange(7, 14))
    assert lay.arm_dims_flat() == list(range(6)) + list(range(7, 13)) and lay.gripper_dims_flat() == [6, 13]
    assert store.Layout.from_json(lay.to_json()).to_json() == lay.to_json()


def test_perturbing_env_noise_only_on_arm_joints_labels_clean():
    env = FakeEnv()
    lay = _layout()
    pe = PerturbingEnv(env, lay, [NoiseProcess(preset("dart"), lay.arm_dofs)])
    pe.begin(torch.zeros(1, 5), [np.random.default_rng(0)])
    _expert(pe)
    arr = pe.arrays(0)
    d = arr["actions"] - arr["actions_clean"]
    assert np.allclose(arr["actions_clean"][:, :6], np.repeat(0.1 * np.arange(6)[:, None], 6, 1), atol=1e-6)
    assert np.abs(d[:3, :6]).max() > 0                       # arm 0 FREE: noised
    assert np.all(d[3:, :6] == 0)                            # arm 0 CONTACT: dart is free-only
    assert np.all(d[:, 7:13] == 0)                           # arm 1 never annotated -> CONTACT -> no noise
    assert np.all(d[:, [6, 13]] == 0)                        # grippers never noised
    assert np.allclose(env.received[0]["a0"], arr["actions"][0, :7])     # the env got the executed action
    assert arr["phase"].tolist() == [[0, 1]] * 3 + [[1, 1]] * 3
    assert arr["obs"].shape == (7, 5) and arr["arm_qpos"].shape == (7, 12)
    assert arr["env_states"]["actors"]["cube"].shape == (7, 13) and arr["success"].tolist() == [False, False, True, True, True, True]
    ns = pe.noise_stats(0)
    assert ns["arm0"]["free_frac"] == 0.5 and ns["arm1"]["rms_rad"] == 0.0 and ns["gripper_max_abs"] == 0.0


def test_perturbing_env_passthrough_without_noise():
    env = FakeEnv()
    lay = _layout()
    pe = PerturbingEnv(env, lay, [NoiseProcess(preset("mp"), lay.arm_dofs)])
    pe.begin(torch.zeros(1, 5), [np.random.default_rng(0)])
    seen = []
    orig = env.step

    def spy(a):
        seen.append(a)
        return orig(a)
    env.step = spy
    act = OrderedDict(a0=np.ones(7, np.float32), a1=np.zeros(7, np.float32))
    pe.step(act)
    assert seen[0] is act                                     # the expert's action object, unchanged
    assert np.array_equal(pe.arrays(0)["actions"], pe.arrays(0)["actions_clean"])


def test_certification_causes():
    ok = dict(success=True, released=True, binding_term="m")
    bad = dict(success=False, released=True, binding_term="trace.window_max:cube_speed")
    unk = dict(success=None, info=dict(window_status="not_quiescent"))
    nrel = dict(success=False, released=False, binding_term="x")
    assert contract_cause("c", ok) is None
    assert contract_cause("c", bad) == "c:trace.window_max:cube_speed"
    assert contract_cause("c", unk) == "c:unknown:not_quiescent"
    assert contract_cause("c", nrel) == "c:not_released"
    assert certify_verdicts({"c1": ok, "c2": ok}, ["c1", "c2"], "ok")["certified"]
    r = certify_verdicts({"c1": ok, "c2": bad}, ["c1", "c2"], "ok")
    assert not r["certified"] and r["rejection_cause"] == "c2:trace.window_max:cube_speed"
    r = certify_verdicts({}, ["c1"], "pre: RRT failed")
    assert r["rejection_cause"] == "plan:pre" and not r["certified"]
    r = certify_verdicts({}, ["c1"], None, error="RuntimeError('x')")
    assert r["rejection_cause"].startswith("error:RuntimeError")
    assert certify_verdicts({}, [], "ok")["certified"]        # certify=False: only the expert status counts


def _write_fake_dataset(root, n=4):
    env = FakeEnv()
    lay = _layout()
    w = store.ShardWriter(root, 0, keep_rejected=True)
    for i in range(n):
        pe = PerturbingEnv(env, lay, [NoiseProcess(preset("dart"), lay.arm_dofs)])
        pe.begin(torch.zeros(1, 5), [np.random.default_rng(i)])
        _expert(pe)
        cert = i % 2 == 0
        im = dict(certified=cert, split="data" if cert else "rejected", rejection_cause=None if cert else "c:term",
                  causes=[] if cert else ["c:term"], verdicts={"c": dict(success=cert, margin=0.1 if cert else -0.2)},
                  bench=dict(final=True), noise=pe.noise_stats(0), phys={"cube": dict(dx=0.001 * i, mass_scale=1.1)},
                  seconds=1.0)
        w.write(f"s{i}", pe.arrays(0), dict(episode_seed=i, reset_kwargs=dict(seed=i), control_mode="pd_joint_pos",
                                            elapsed_steps=6, success=True, imsim=im))
    w.close()
    meta = dict(env_info=dict(env_id="Fake-v0", env_kwargs={}), task="Fake", embodiment="fake", perturb=preset("dart").to_dict(),
                perturb_text=preset("dart").describe(), contracts={"c": dict(name="c", sha256="0" * 64)},
                code_hash=dict(imsim="x"), layout=lay.to_json(), seeds=[0, n - 1, n], keep_rule="all accept",
                summary=dict(wall_s=1.0, workers=1, episodes_per_hour=3600.0))
    store.merge(root, meta, keep_rejected=True)
    js = json.load(open(os.path.join(root, "dataset.json")))
    js["summary"] = meta["summary"]
    json.dump(js, open(os.path.join(root, "dataset.json"), "w"))


def test_store_merge_maniskill_layout_and_card(tmp_path_project):
    import h5py
    root = tmp_path_project
    _write_fake_dataset(root, 4)
    assert not os.path.exists(os.path.join(root, "shards"))
    tj = json.load(open(os.path.join(root, "trajectory.json")))
    assert tj["env_info"]["env_id"] == "Fake-v0" and len(tj["episodes"]) == 2
    assert [e["episode_id"] for e in tj["episodes"]] == [0, 1] and [e["episode_seed"] for e in tj["episodes"]] == [0, 2]
    assert tj["imsim"]["label_key"] == "actions_clean" and tj["imsim"]["phase_codes"] == {"free": 0, "contact": 1}
    with h5py.File(os.path.join(root, "trajectory.h5"), "r") as f:
        assert sorted(f.keys()) == ["traj_0", "traj_1"]
        g = f["traj_0"]
        for k in ("actions", "actions_clean", "terminated", "truncated", "success", "obs", "env_states", "phase", "arm_qpos"):
            assert k in g, k
        assert g["actions"].shape == (6, 14) and g["obs"].shape == (7, 5) and g["env_states/actors/cube"].shape == (7, 13)
        assert g["actions"].dtype == np.float32 and g["terminated"].dtype == bool
    with h5py.File(os.path.join(root, "rejected.h5"), "r") as f:
        assert sorted(f.keys()) == ["rej_0", "rej_1"]
    rows = [json.loads(l) for l in open(os.path.join(root, "attempts.jsonl"))]
    assert len(rows) == 4 and sum(r["imsim"]["certified"] for r in rows) == 2
    ds = store.open_dataset(root)
    assert len(ds.episodes) == 2 and ds.layout().dims == [7, 7]
    a = ds.arrays(1)
    assert a["env_states"]["actors"]["cube"].shape == (7, 13)
    ds.close()
    eps = store.load_episodes(root)
    assert eps[0]["obs"].shape == (6, 5) and eps[0]["act"].shape == (6, 14)
    assert np.allclose(eps[0]["act"][:, :6], np.repeat(0.1 * np.arange(6)[:, None], 6, 1), atol=1e-6)  # clean labels
    from imitator.data.card import summarize, wilson, write_card
    c = write_card(root)
    assert os.path.exists(c["md"]) and os.path.exists(c["json"])
    s = summarize(root)
    assert s["counts"] == dict(attempted=4, certified=2, rejected=2, errors=0)
    assert s["rejection_causes"] == {"c:term": 2} and s["length_certified"]["p50"] == 6
    lo, hi = wilson(2, 4)
    assert lo < 0.5 < hi and tuple(s["rate_ci95"]) == (lo, hi)
    md = open(c["md"]).read()
    assert "certification rate: 50.0%" in md and "c:term" in md and "actions_clean" in md


def test_dataset_reads_maniskill_dict_actions(tmp_path_project):
    """an external ManiSkill RecordEpisode file with a dict action group and no imsim metadata"""
    import h5py
    from imitator.data.certify import _actions
    p = os.path.join(tmp_path_project, "ext.h5")
    with h5py.File(p, "w") as f:
        g = f.create_group("traj_0")
        a = g.create_group("actions")
        a.create_dataset("a0", data=np.ones((3, 7), np.float32))
        a.create_dataset("a1", data=np.zeros((3, 7), np.float32))
        g.create_dataset("terminated", data=np.zeros(3, bool))
    json.dump(dict(env_info=dict(env_id="X"), episodes=[dict(episode_id=0, episode_seed=5, reset_kwargs=dict(seed=5),
                                                                control_mode="pd_joint_pos", elapsed_steps=3)]),
              open(os.path.join(tmp_path_project, "ext.json"), "w"))
    ds = store.open_dataset(p)
    assert ds.episodes[0]["episode_seed"] == 5 and ds.meta == {}
    acts = _actions(ds.arrays(0), _layout())
    assert len(acts) == 3 and np.allclose(acts[0]["a0"], 1)
    ds.close()


@pytest.mark.skipif(os.environ.get("IMSIM_SLOW") != "1", reason="slow: real DualCubePlace episodes (IMSIM_SLOW=1)")
def test_generate_and_certify_dualcubeplace(tmp_path_project):
    import imitator
    r = imitator.generate("DualCubePlace-v1", "dual_ur5_wsg", seeds=[0, 1], perturb="dart_hi_phys", out=tmp_path_project,
                       workers=2, verbose=False)
    assert r["attempted"] == 2
    tj = json.load(open(os.path.join(tmp_path_project, "trajectory.json")))
    for e in tj["episodes"]:
        assert e["imsim"]["certified"] and e["imsim"]["phys"]["cube"]["mass_scale"] is not None
    rep = imitator.certify(tmp_path_project, "DualCubePlace-v1", workers=2, verbose=False)
    assert rep["n"] == r["certified"] and rep["diverged"] == 0
    assert rep["agreement_with_stored"] in (None, 1.0)
