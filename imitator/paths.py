"""Filesystem locations used by the imitator package (no absolute paths in the code).

IMITATOR_OUT          root for outputs of imitator.evaluate / imitator.generate (default: ./results)
IMITATOR_SAPIEN_HOME  optional HOME override used when GPU physics is first started: SAPIEN downloads its PhysX GPU
                      library into $HOME/.sapien (set it on machines where HOME must not be written)
MS_ASSET_DIR          ManiSkill's asset root (default ~/.maniskill/data), used for the Imitator objects and scenes
"""
from __future__ import annotations

import os
from pathlib import Path

PKG = Path(__file__).resolve().parent
REPO = PKG.parent
EMBODIMENT_ASSETS = PKG / "assets" / "embodiments"          # derived URDF/SRDF + collision hulls (in the repo)
EMBODIMENT_SOURCES = EMBODIMENT_ASSETS / "_src"             # vendored source URDFs and meshes (see NOTICE.md there)


def out_root() -> str:
    return os.environ.get("IMITATOR_OUT", "results")


def out_dir(*parts: str) -> str:
    return os.path.join(out_root(), *parts)
