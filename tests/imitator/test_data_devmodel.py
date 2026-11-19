"""E29a data engine: deviation model (E26 procedure) and phases (fast, no simulator)."""
import json
import os

import numpy as np
import pytest

from imitator.data import phase as P
from imitator.data.devmodel import DeviationModel, fit_from_rollouts, match_deviation, rollout_deviation
from _paths import DATA, REPO, TMP as _TMP, repo_path  # noqa: E402,F401
TMP = _TMP

ROOT = REPO
E26_DEV = os.path.join(DATA, "devmodel_poke_cube_v1.json")


@pytest.fixture
def tmp_path_project(request):
    """a fresh directory inside the project (never /tmp)"""
    import shutil
    d = os.path.join(TMP, "pytest_e29a", request.node.name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _ar1(n, d, sd, rho, rng):
    x, out = np.zeros(d), []
    for _ in range(n):
        x = rho * x + rng.normal(0, sd * np.sqrt(1 - rho ** 2), d)
        out.append(x.copy())
    return np.asarray(out)


def test_match_deviation_monotone_window():
    t = np.linspace(0, 1, 200)
    q_exp = np.stack([t, t ** 2], 1)
    q_pol = q_exp[::2] + 0.001                     # the policy runs twice as fast, small offset
    dev, k = match_deviation(q_pol, q_exp)
    assert np.all(np.diff(k) >= -5)                 # monotone up to the backward slack
    assert np.abs(dev).max() < 0.01
    assert k[-1] >= 190


def test_fit_recovers_covariance_rho_and_trims_excursions():
    rng = np.random.default_rng(0)
    rollouts = []
    for _ in range(6):
        T = 300
        dev = np.zeros((T, 4))
        ph0 = np.array([P.FREE] * 150 + [P.CONTACT] * 150)
        dev[:150, :2] = _ar1(150, 2, 0.01, 0.9, rng)
        dev[150:, :2] = _ar1(150, 2, 0.03, 0.9, rng)
        dev[:, 2:] = _ar1(T, 2, 0.02, 0.5, rng)
        dev[200:205, :2] += 1.0                     # failure excursion in CONTACT (trimmed from the covariance)
        rollouts.append(dict(dev=dev, phase=[list(ph0), [P.CONTACT] * T], dofs=[2, 2]))
    dm = fit_from_rollouts(rollouts)
    assert abs(dm.iso_sd(0, P.FREE) / 0.01 - 1) < 0.25
    assert abs(dm.iso_sd(0, P.CONTACT) / 0.03 - 1) < 0.25
    assert abs(dm.rho(0, P.FREE) - 0.9) < 0.1
    assert abs(dm.iso_sd(1, P.CONTACT) / 0.02 - 1) < 0.15
    assert abs(dm.rho(1, P.CONTACT) - 0.5) < 0.1
    st = dm.meta["stats"]["arm0_contact"]
    assert st["iso_sd_full_rad"] > 3 * st["iso_sd_rad"]          # the untrimmed covariance sees the excursion
    assert dm.meta["stats"]["arm0_free"]["iso_sd_full_rad"] < 1.5 * dm.meta["stats"]["arm0_free"]["iso_sd_rad"]
    assert dm.d[1][P.FREE]["n"] == 0                              # arm 1 never FREE


def test_rollout_deviation_phase_codes():
    q = np.zeros((10, 3))
    r = rollout_deviation(q + 0.01, q, np.array([[0, 1]] * 10, np.uint8), [2, 1])
    assert r["phase"][0][0] == P.FREE and r["phase"][1][0] == P.CONTACT
    assert r["dev"].shape == (10, 3)


def test_json_roundtrip_and_e26_format(tmp_path_project):
    dm = DeviationModel.isotropic([6, 6], dict(free=0.01, contact=0.02), dict(free=0.9, contact=0.95), meta=dict(task="t"))
    p = dm.save(os.path.join(tmp_path_project, "dev.json"))
    dm2 = DeviationModel.load(p)
    assert np.allclose(dm2.cov(1, P.CONTACT), dm.cov(1, P.CONTACT)) and dm2.rho(0, P.FREE) == 0.9
    assert dm2.meta["task"] == "t"
    raw = json.load(open(p))
    assert set(raw["model"]["0"]) == {"free", "contact"} and "scale" in raw
    if os.path.exists(E26_DEV):                     # E26's own file loads unchanged (read only)
        e26 = DeviationModel.load(E26_DEV)
        assert e26.cov(0, P.FREE).shape == (7, 7) and 0.0 <= e26.rho(0, P.CONTACT) <= 1.0
        assert e26.scale == json.load(open(E26_DEV)).get("scale", 1.0)


def test_phase_api_semantics_unchanged():
    class U:
        pass
    env = U()
    assert P.get_phase(env, 0) == P.CONTACT                       # never annotated: CONTACT
    P.set_phase(env, P.FREE, arm=1)
    assert P.get_phase(env, 1) == P.FREE and P.get_phase(env, 0) == P.CONTACT
    P.set_phase(env, P.FREE)                                      # all arms
    assert P.get_phase(env, 0) == P.FREE and P.get_phase(env, 1) == P.FREE
    P.set_phase(env, P.CONTACT, arm=0)
    assert P.get_phase(env, 0) == P.CONTACT and P.get_phase(env, 1) == P.FREE
    P.reset_phase(env)
    assert P.get_phase(env, 1) == P.CONTACT
    with pytest.raises(ValueError):
        P.set_phase(env, "hover")


def test_towel_expert_phases_and_corner_reading():
    from imitator.soft.expert import segment_phases, towel_points, towel_waypoints, phase_at, total_time
    res, H, c = 4, 0.15, np.array([-0.05, 0.0])
    g = np.linspace(-H, H, res + 1)
    X, Y = np.meshgrid(g, g)                      # index = y * (res + 1) + x
    q = np.stack([X.ravel() + c[0], Y.ravel() + c[1], np.zeros(X.size)], 1)
    A, C = towel_points(q, res)
    wp = towel_waypoints("expert", half=H, center=tuple(c))
    assert np.allclose(A, wp[0][:2]) and np.allclose(C, wp[-1][:2])
    a = np.radians(10)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    q2 = q.copy(); q2[:, :2] = (q[:, :2] - c) @ R.T + c + [0.01, -0.01]
    A2, C2 = towel_points(q2, res)
    assert np.allclose(A2, (A - c) @ R.T + c + [0.01, -0.01], atol=1e-9)
    wps = [(0.3, 0.0, 0.3, 0, 0.6)] + towel_waypoints("expert", A=A, C=C)
    ph = segment_phases(wps, True)
    assert ph[0] == P.FREE and ph[1] == P.CONTACT and ph[-1] == P.FREE
    i_open = [i for i, w in enumerate(wps) if i + 1 < len(wps) and w[3] == 1 and wps[i + 1][3] == 0][-1]
    assert ph[i_open] == P.CONTACT and ph[i_open + 1] == P.FREE   # release is CONTACT, the retreat after it FREE
    dt = 0.05
    seq = [phase_at(wps, ph, (k + 1) * dt) for k in range(int(np.ceil(total_time(wps) / dt)))]
    assert seq[0] == P.FREE and seq[-1] == P.FREE and P.CONTACT in seq


def test_pick_place_is_annotated():
    import inspect
    from imitator.planning import scripted
    src = inspect.getsource(scripted.pick_place)
    assert src.count("ex.phase(") >= 4 and "FREE" in src and "CONTACT" in src
