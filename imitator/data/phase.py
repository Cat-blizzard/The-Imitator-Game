"""Phase annotation for scripted experts: the shared contract between task code and imsim's data engine.

A scripted expert marks, per arm, whether the motion it is about to execute is a FREE transit (nothing grasped, no
intended contact: approach from afar, retreat, move to a pre-grasp pose) or a CONTACT phase (final approach, grasp,
carry, insert, press, place, release, fold, any motion with an object in hand). The data engine reads the phase at every
env.step to shape perturbations (e.g. E26 directed DART: per-phase covariance, attenuated or zero noise in CONTACT).

Rules for expert authors:
  * call set_phase(env, FREE | CONTACT, arm=i) BEFORE the env.step calls of the stage it describes;
  * single-arm experts may omit `arm` (it then applies to every arm);
  * an arm that was never annotated is CONTACT (conservative: no free-space noise is assumed safe);
  * annotation never changes the simulation; experts must behave identically with or without a data engine attached.
"""
from __future__ import annotations

FREE = "free"
CONTACT = "contact"
_ATTR = "_imsim_phase"
_ALL = "all"


def _store(env) -> dict:
    u = getattr(env, "unwrapped", env)
    d = getattr(u, _ATTR, None)
    if d is None:
        d = {}
        setattr(u, _ATTR, d)
    return d


def set_phase(env, phase: str, arm: int | None = None) -> None:
    if phase not in (FREE, CONTACT):
        raise ValueError(f"phase must be {FREE!r} or {CONTACT!r}, got {phase!r}")
    d = _store(env)
    if arm is None:
        d.clear()
        d[_ALL] = phase
    else:
        d[int(arm)] = phase


def get_phase(env, arm: int | None = None) -> str:
    d = _store(env)
    if arm is not None and int(arm) in d:
        return d[int(arm)]
    return d.get(_ALL, CONTACT)


def reset_phase(env) -> None:
    _store(env).clear()
