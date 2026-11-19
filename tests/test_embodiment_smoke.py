from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box

from mani_skill.examples.embodiments.smoke import action_for_pose


def group(width):
    return SimpleNamespace(single_action_space=Box(-1, 1, shape=(width,), dtype=np.float32))


def agent_with_groups(groups):
    controller = SimpleNamespace(controllers=OrderedDict(groups))
    controller.from_action_dict = lambda values: torch.hstack([values[name] for name in controller.controllers])
    return SimpleNamespace(device=torch.device("cpu"), controller=controller)


def test_action_packing_follows_controller_order_and_keeps_gripper_command():
    agent = agent_with_groups([("gripper", group(1)), ("arm", group(6)), ("gripper_passive", group(0))])
    target = np.arange(6, dtype=np.float32) / 10
    action = action_for_pose(agent, target, -1.0)
    np.testing.assert_array_equal(action, np.r_[-1.0, target])


def test_action_packing_rejects_incomplete_arm_target():
    agent = agent_with_groups([("arm", group(6)), ("gripper", group(1))])
    with pytest.raises(ValueError, match="action width"):
        action_for_pose(agent, np.zeros(5, dtype=np.float32), 1.0)


def test_action_packing_does_not_silently_move_unknown_joints():
    agent = agent_with_groups([("arm", group(6)), ("head", group(2))])
    with pytest.raises(ValueError, match="hold targets"):
        action_for_pose(agent, np.zeros(6, dtype=np.float32), 1.0)
