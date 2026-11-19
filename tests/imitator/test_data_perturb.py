"""E29a data engine: perturbation spec, noise process, presets (fast, no simulator)."""
import numpy as np
import pytest

from imitator.data.devmodel import DeviationModel
from imitator.data.perturb import NoiseProcess, PerturbSpec, PhysSpec, PRESETS, as_spec, preset
from imitator.data.phase import CONTACT, FREE


def _iso_dev(dofs=(6, 6), sd_free=0.02, sd_contact=0.04, rho_free=0.9, rho_contact=0.8, scale=1.0):
    return DeviationModel.isotropic(dofs, dict(free=sd_free, contact=sd_contact), dict(free=rho_free, contact=rho_contact),
                                    scale=scale)


def test_presets_match_e26_e27_definitions():
    dm = _iso_dev().to_json()
    assert not preset("mp").has_noise and preset("mp").phys is None
    d = preset("dart")
    assert (d.noise, d.cov, d.sd, d.gating, d.phys) == ("white", "iso", 0.03, "free_only", None)
    dd = preset("dart_dir", devmodel=dm)
    assert (dd.noise, dd.cov, dd.gating, dd.contact_scale, dd.scale) == ("ar1", "devmodel", "all", 0.25, None)
    assert dd.phys == PhysSpec(pose_xy_m=0.01, yaw_deg=10.0, friction=(0.7, 1.3), mass=(0.7, 1.3))
    assert preset("phys").phys is not None and not preset("phys").has_noise
    assert preset("dart_hi_phys").phys is not None and preset("dart_hi_phys").gating == "free_only"
    assert preset("dart_dir_nophys", devmodel=dm).phys is None
    assert preset("dart_dir_white", devmodel=dm).noise == "white"
    assert preset("dart_dir_freeonly", devmodel=dm).gating == "free_only"
    assert preset("mp_fresh").name == "mp_fresh" and preset("dart_hi").sd == 0.03 and preset("phys_only").phys
    for n in PRESETS:
        if "dart_dir" not in n:
            preset(n)
    with pytest.raises(ValueError):
        preset("dart_dir")                       # needs a deviation model
    with pytest.raises(KeyError):
        preset("nope")


def test_spec_validation_and_roundtrip():
    with pytest.raises(ValueError):
        PerturbSpec(noise="pink")
    with pytest.raises(ValueError):
        PerturbSpec(noise="ar1", cov="iso", sd=0.01)          # ar1 + iso needs rho
    with pytest.raises(ValueError):
        PhysSpec(cloth="melt")
    s = PerturbSpec(name="x", noise="ar1", cov="iso", sd=0.01, rho=0.9, phys=PhysSpec(cloth="skip"))
    s2 = PerturbSpec.from_dict(s.to_dict())
    assert s2 == s
    assert as_spec("dart").sd == 0.03 and as_spec(None).name == "mp" and as_spec(s.to_dict()) == s
    assert "white" in preset("dart").describe() and "FREE" in preset("dart").describe()
    assert "physical" in preset("phys").describe()
    assert s.replace(sd=0.5).sd == 0.5 and s.sd == 0.01


def test_white_free_only_noise_statistics():
    npr = NoiseProcess(preset("dart"), [6, 7])
    npr.reset(np.random.default_rng(0))
    X = []
    for _ in range(4000):
        offs = npr.offsets([FREE, CONTACT])
        assert offs[1] is None                             # CONTACT: no noise under gating free_only
        X.append(offs[0])
    X = np.asarray(X)
    assert X.shape == (4000, 6)
    assert abs(X.std() - 0.03) < 0.002
    c = np.corrcoef(X[:-1, 0], X[1:, 0])[0, 1]
    assert abs(c) < 0.06                                    # white


def test_ar1_stationary_covariance_rho_and_contact_scale():
    dm = _iso_dev(dofs=(5,), sd_free=0.02, sd_contact=0.02, rho_free=0.9, rho_contact=0.9, scale=0.5)
    spec = PerturbSpec(name="t", noise="ar1", cov="devmodel", devmodel=dm.to_json(), gating="all", contact_scale=0.25)
    npr = NoiseProcess(spec, [5], dm)
    assert npr.s == 0.5                                      # scale None -> the deviation model's calibrated s_t
    for ph, want in ((FREE, 0.02 * 0.5), (CONTACT, 0.02 * 0.5 * 0.5)):
        npr.reset(np.random.default_rng(1))
        X = np.asarray([npr.offsets([ph])[0] for _ in range(30000)])[500:]
        assert abs(X.std() / want - 1) < 0.08, (ph, X.std(), want)
        r = np.mean([np.corrcoef(X[:-1, j], X[1:, j])[0, 1] for j in range(5)])
        assert abs(r - 0.9) < 0.03
    # rho is clipped at rho_max
    spec2 = spec.replace(rho=0.999)
    assert NoiseProcess(spec2, [5], dm).rho(0, FREE) == 0.95
    # white with the same stationary covariance
    w = NoiseProcess(spec.replace(noise="white"), [5], dm)
    assert w.rho(0, FREE) == 0.0


def test_free_only_resets_ar_state_and_scale_override():
    dm = _iso_dev(dofs=(3,))
    spec = PerturbSpec(name="t", noise="ar1", cov="devmodel", devmodel=dm.to_json(), gating="free_only", scale=2.0)
    npr = NoiseProcess(spec, [3], dm)
    assert npr.s == 2.0
    npr.reset(np.random.default_rng(2))
    for _ in range(20):
        npr.offsets([FREE])
    assert np.abs(npr.x[0]).sum() > 0
    assert npr.offsets([CONTACT]) == [None]
    assert np.all(npr.x[0] == 0)


def test_devmodel_dimension_mismatch_and_no_noise():
    dm = _iso_dev(dofs=(7,))
    spec = preset("dart_dir", devmodel=dm.to_json())
    with pytest.raises(ValueError):
        NoiseProcess(spec, [6], dm)
    npr = NoiseProcess(preset("phys"), [6, 6])
    npr.reset(np.random.default_rng(0))
    assert npr.offsets([FREE, FREE]) == [None, None]


def test_iso_ar1_explicit_rho():
    spec = PerturbSpec(name="t", noise="ar1", cov="iso", sd=0.01, rho=0.5, gating="all", contact_scale=1.0)
    npr = NoiseProcess(spec, [4])
    npr.reset(np.random.default_rng(3))
    X = np.asarray([npr.offsets([CONTACT])[0] for _ in range(20000)])
    assert abs(X.std() - 0.01) < 0.0008
