"""Small LeRobot schema helpers for the project-local AgiBot converter."""

from __future__ import annotations

from typing import Iterable

from .config import ACTION_SPECS, CAMERA_SPECS, STATE_SPECS


def generate_features(cameras: Iterable[str]) -> dict:
    """Generate features accepted by The-Imitator-Game's LeRobotDataset.create.

    Only RGB videos plus the 16-D gripper control state/action are emitted.
    The converter keeps the semantic v3 field names used by current
    any4lerobot instead of collapsing everything into observation.state/action.
    """
    features: dict = {}

    for camera in cameras:
        if camera not in CAMERA_SPECS:
            raise KeyError(
                f"Unknown camera {camera!r}. Available cameras: {sorted(CAMERA_SPECS)}"
            )
        height, width, _ = CAMERA_SPECS[camera]
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": (3, height, width),
            "names": ["channels", "height", "width"],
        }

    for key, spec in STATE_SPECS.items():
        features[key] = {
            "dtype": "float32",
            "shape": tuple(spec["shape"]),
            "names": list(spec["names"]),
        }

    for key, spec in ACTION_SPECS.items():
        features[key] = {
            "dtype": "float32",
            "shape": tuple(spec["shape"]),
            "names": list(spec["names"]),
        }

    return features


def control_dims() -> tuple[int, int]:
    state_dim = sum(int(spec["shape"][0]) for spec in STATE_SPECS.values())
    action_dim = sum(int(spec["shape"][0]) for spec in ACTION_SPECS.values())
    return state_dim, action_dim
