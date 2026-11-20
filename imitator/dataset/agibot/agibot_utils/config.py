"""AgiBot gripper schema used by SelfPromptRobotDataset pretraining.

This is intentionally a *training-oriented subset* of the current any4lerobot
AgiBot gripper schema: we keep the RGB streams used by HyperICL and the two
proprio/action groups that form a 16-D state and 16-D action.

Output visual feature shapes follow this repository's LeRobot convention
(C, H, W), while decoded samples are still returned by LeRobot as channel-first
PyTorch tensors.
"""

from __future__ import annotations

from typing import Dict, Tuple


# Logical output camera -> (height, width, candidate raw AgiBot video stems).
# Older AgiBotWorld-Alpha conversions used "top_head" while current
# any4lerobot uses "head".  The converter accepts either on disk and always
# writes the standardized output key observation.images.head.
CAMERA_SPECS: Dict[str, Tuple[int, int, tuple[str, ...]]] = {
    "head": (480, 640, ("head", "top_head")),
    "hand_left": (480, 640, ("hand_left",)),
    "hand_right": (480, 640, ("hand_right",)),
    "head_center_fisheye": (768, 960, ("head_center_fisheye",)),
    "head_left_fisheye": (768, 960, ("head_left_fisheye",)),
    "head_right_fisheye": (768, 960, ("head_right_fisheye",)),
    "back_left_fisheye": (768, 960, ("back_left_fisheye",)),
    "back_right_fisheye": (768, 960, ("back_right_fisheye",)),
}

DEFAULT_CAMERAS = ("head", "hand_left", "hand_right")

LEFT_ARM_NAMES = [f"left_arm_{i}" for i in range(7)]
RIGHT_ARM_NAMES = [f"right_arm_{i}" for i in range(7)]
JOINT_NAMES = LEFT_ARM_NAMES + RIGHT_ARM_NAMES
EFFECTOR_NAMES = ["left_gripper", "right_gripper"]

# Semantic HDF5 paths used by current any4lerobot's AgiBot parser.
STATE_SPECS = {
    "observation.states.joint.position": {
        "h5_path": "state/joint/position",
        "shape": (14,),
        "names": JOINT_NAMES,
    },
    "observation.states.effector.position": {
        "h5_path": "state/effector/position",
        "shape": (2,),
        "names": EFFECTOR_NAMES,
    },
}

ACTION_SPECS = {
    "actions.joint.position": {
        "h5_path": "action/joint/position",
        "index_path": "action/joint/index",
        "fallback_state_key": "observation.states.joint.position",
        "shape": (14,),
        "names": JOINT_NAMES,
    },
    "actions.effector.position": {
        "h5_path": "action/effector/position",
        "index_path": "action/effector/index",
        "fallback_state_key": "observation.states.effector.position",
        "shape": (2,),
        "names": EFFECTOR_NAMES,
    },
}

# Current any4lerobot task-type table.  Gripper means "all remaining tasks".
# We only need these sets to avoid silently mixing incompatible action spaces.
DEXHAND_TASK_IDS = {
    "task_475", "task_536", "task_547", "task_548", "task_549", "task_554",
    "task_577", "task_578", "task_591", "task_595", "task_608", "task_620",
    "task_622", "task_660", "task_679", "task_705", "task_710", "task_727",
    "task_730", "task_731", "task_749", "task_753",
}

TACTILE_TASK_IDS = {
    "task_666", "task_675", "task_676", "task_677", "task_694", "task_737",
    "task_774",
}

NON_GRIPPER_TASK_IDS = DEXHAND_TASK_IDS | TACTILE_TASK_IDS


def is_gripper_task(task_name: str) -> bool:
    return str(task_name) not in NON_GRIPPER_TASK_IDS
