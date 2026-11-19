"""Runtime compatibility shims (no file of ManiSkill / SAPIEN is modified).

* ManiSkill 3.0.1 splits render_backend "pci:0" into ("pci", "0") and then fails to find device "pci"; the CPU
  renderer (e.g. lavapipe) is only reachable as SAPIEN device "pci:0". The Imitator fork of ManiSkill (3.0.0b21)
  already accepts "pci:0" and has no parse_backend_device_id, so the patch is skipped there.
* ensure_sapien_home(): optional HOME redirection for SAPIEN's PhysX GPU download (IMITATOR_SAPIEN_HOME).
* check_env(): optional check that caches / tmp live under IMITATOR_PROJECT_ROOT (for shared machines).
"""
from __future__ import annotations

import os

_PATCHED = False


def patch_render_backend() -> None:
    global _PATCHED
    if _PATCHED:
        return
    from mani_skill.envs.utils.system import backend as be
    if not hasattr(be, "parse_backend_device_id"):      # Imitator fork: "pci:0" works as is
        _PATCHED = True
        return
    orig = be.parse_backend_device_id
    be.parse_backend_device_id = lambda b: (b, None) if isinstance(b, str) and b.startswith("pci:") else orig(b)
    _PATCHED = True


def sapien_home() -> str | None:
    """the HOME to use for SAPIEN (IMITATOR_SAPIEN_HOME), or None to leave HOME alone"""
    return os.environ.get("IMITATOR_SAPIEN_HOME") or None


def ensure_sapien_home() -> str:
    """SAPIEN downloads the PhysX GPU library to Path.home()/.sapien/physx/<version>/ the first time GPU sim is used
    (sapien/physx/__init__.py). If IMITATOR_SAPIEN_HOME is set, point HOME there before any GPU env is created."""
    h = sapien_home()
    if h:
        os.environ["HOME"] = h
    return os.environ.get("HOME", "")


def check_env() -> list[str]:
    """if IMITATOR_PROJECT_ROOT is set: cache / tmp variables that do not point inside it (empty list otherwise)"""
    root = os.environ.get("IMITATOR_PROJECT_ROOT")
    if not root:
        return []
    need = ["MS_ASSET_DIR", "TMPDIR", "XDG_CACHE_HOME", "MPLCONFIGDIR", "PYTHONPYCACHEPREFIX", "CUDA_CACHE_PATH",
            "MESA_SHADER_CACHE_DIR", "HF_HOME", "TORCH_HOME"]
    return [f"{k}={os.environ.get(k, '')!r}" for k in need if not os.environ.get(k, "").startswith(root)]


def grasp_utils():
    """ManiSkill 3.0.1's OBB grasp helpers (get_actor_obb, compute_grasp_info_by_obb), vendored in
    imitator/planning/ms_planner/utils.py (the Imitator fork keeps an older copy in examples/motionplanning/panda)"""
    from .planning.ms_planner import utils
    return utils
