"""imsim: ManiSkill-3-based simulation platform (dual-arm embodiments, Newton cloth bridge, contract-judged benchmarks).

    import imitator
    imitator.list_tasks(); imitator.list_embodiments()
    env = imitator.make("DualHandover-v1", embodiment="dual_ur5_wsg", sim="cpu")          # ManiSkill gym env
    res = imitator.evaluate("DualHandover-v1", "dual_ur5_wsg", "handover", seeds=range(4))  # contract verdicts + evidence
    ds = imitator.generate("DualCubePlace-v1", "dual_ur5_wsg", seeds=range(50), perturb="dart", workers=8)  # data engine
    rep = imitator.certify("trajectory.h5", "DualCubePlace-v1")                                            # replay + judge
CLI: python -m imitator {list, run, eval, card, generate, certify, devmodel}. See docs/imitator2/README.md.
"""
from __future__ import annotations

import os
import sys

__version__ = "0.2.0"
__all__ = ["make", "evaluate", "list_tasks", "list_embodiments", "get_task", "script", "generate", "certify",
           "fit_deviation_model", "calibrate_scale", "dataset_card", "PerturbSpec", "PhysSpec", "preset", "PRESETS"]


def make(*a, **k):
    from .api import make as _m
    return _m(*a, **k)


def evaluate(*a, **k):
    from .harness import evaluate as _e
    return _e(*a, **k)


def list_tasks():
    from .tasks import list_tasks as _l
    return _l()


def list_embodiments(detail: bool = False):
    from .api import list_embodiments as _l
    return _l(detail)


def get_task(name):
    from .tasks import get_task as _g
    return _g(name)


def script(fn):
    from .harness import script as _s
    return _s(fn)


def generate(*a, **k):
    from .api import generate as _g
    return _g(*a, **k)


def certify(*a, **k):
    from .api import certify as _c
    return _c(*a, **k)


def fit_deviation_model(*a, **k):
    from .api import fit_deviation_model as _f
    return _f(*a, **k)


def calibrate_scale(*a, **k):
    from .api import calibrate_scale as _c
    return _c(*a, **k)


def dataset_card(*a, **k):
    from .api import dataset_card as _d
    return _d(*a, **k)


def __getattr__(name):            # light-weight access to the perturbation spec classes (no simulator import)
    if name in ("PerturbSpec", "PhysSpec", "preset", "PRESETS"):
        from .data import perturb as _p
        return getattr(_p, name)
    raise AttributeError(name)
