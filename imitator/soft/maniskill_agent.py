"""Panda variant for cloth tasks in ManiSkill 3 (route A, the PhysX side of the bridge).

The cloth lives in Newton, so PhysX does not know it is there: a Panda closing on a towel would close fully. This agent
stops the fingers where the cloth layer would stop them: the gripper close target (normalised action -1) is the finger
joint position CLOTH_Q_CLOSE per finger (pad gap 2*q) instead of the stock -0.01 m. With a particle radius r the pads
then compress each pinched particle by (r - q); the cloth's push-back on the fingers is not simulated (one-way coupling),
but it is measured on the Newton side (robot_cloth_impulse / robot_cloth_fmax) so its size is known.
"""
from __future__ import annotations

import copy

from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.panda.panda import Panda

CLOTH_Q_CLOSE = 0.002


@register_agent()
class PandaCloth(Panda):
    uid = "panda_cloth"
    cloth_q_close = CLOTH_Q_CLOSE

    # ManiSkill 3.0.1's Panda has these properties; the Imitator fork's Panda does not
    @property
    def tcp_pos(self):
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        return self.tcp.pose

    @property
    def _controller_configs(self):
        cfgs = copy.deepcopy(super()._controller_configs)
        for mode in cfgs.values():
            g = mode.get("gripper") if isinstance(mode, dict) else None
            if g is not None and hasattr(g, "lower"):
                g.lower = self.cloth_q_close
        return cfgs
