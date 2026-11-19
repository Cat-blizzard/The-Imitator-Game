"""PD joint-position mimic controller with explicit mimic maps (multiplier / offset per mimic joint).

Copied from ManiSkill 3.0.1 (mani_skill/agents/controllers/pd_joint_pos.py, Apache-2.0) because the Imitator fork of
ManiSkill (3.0.0b21) ships the older mimic controller, which supports only identical joint targets. The imitator
embodiments (ARX X5, PiPER, UR5 + WSG-50, xArm6 + Robotiq) need the per-joint maps. Only the class names differ from
upstream; the fork's own controller is left untouched so the 1.0 tasks behave exactly as before.
"""
from dataclasses import dataclass, field

import torch

from mani_skill.agents.controllers.pd_joint_pos import PDJointPosController, PDJointPosControllerConfig
from mani_skill.utils.structs.types import Array

try:
    from mani_skill.utils.logging_utils import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger("imitator")


class MimicJointPosController(PDJointPosController):
    config: "MimicJointPosControllerConfig"

    def _initialize_joints(self):
        super()._initialize_joints()
        # do some sanity checks and setup the mimic controller

        self.mimic_joint_indices = []
        self.mimic_control_joint_indices = []
        if len(self.config.mimic) == 0:
            if len(self.config.joint_names) == 2:
                logger.warning(
                    f"Mimic targets dictionary is missing for controller config for {self.articulation.name}. Assuming the first joint is the control joint and the second joint is the mimic joint"
                )
                self.config.mimic = {
                    self.config.joint_names[0]: {"joint": self.config.joint_names[1]}
                }
            else:
                raise ValueError(
                    "Mimic targets dictionary is missing. Please provide a mimic targets dictionary to setup mimic controllers with more than 2 joints"
                )

        self._multiplier = torch.ones(
            len(self.config.mimic), device=self.device
        )  # this is parallel to self.mimic_joint_indices
        self._offset = torch.zeros(len(self.config.mimic), device=self.device)

        for i, (mimic_joint_name, mimic_data) in enumerate(self.config.mimic.items()):
            control_joint_name = mimic_data["joint"]
            multiplier = mimic_data.get("multiplier", 1.0)
            offset = mimic_data.get("offset", 0.0)
            assert (
                self.articulation.joints_map[mimic_joint_name].active_index is not None
            ), f"Mimic joint {mimic_joint_name} is not active, cannot be used as a mimic joint in a mimic controller"
            assert (
                self.articulation.joints_map[control_joint_name].active_index
                is not None
            ), f"Control joint {control_joint_name} is not active, cannot be used as a control joint in a mimic controller"

            # NOTE (stao): we are assuming all the articulations are the exact same structure. At the moment I see very little reason to try and support training/evaluating on different embodiments
            # simultaneously in one process
            self.mimic_joint_indices.append(
                torch.argwhere(
                    self.active_joint_indices
                    == self.articulation.joints_map[mimic_joint_name].active_index[0]
                )
            )
            self.mimic_control_joint_indices.append(
                torch.argwhere(
                    self.active_joint_indices
                    == self.articulation.joints_map[control_joint_name].active_index[0]
                )
            )

            self._multiplier[i] = multiplier
            self._offset[i] = offset

        self.mimic_joint_indices = torch.tensor(
            self.mimic_joint_indices, dtype=torch.int32, device=self.device
        )
        self.mimic_control_joint_indices = torch.tensor(
            self.mimic_control_joint_indices, dtype=torch.int32, device=self.device
        )
        """list of control joint indices corresponding to mimic joint indices"""
        self.control_joint_indices = torch.unique(self.mimic_control_joint_indices).to(
            torch.int32
        )
        """list of all directly controlled joint indices"""

        self.effective_dof = len(self.control_joint_indices)

    def _get_joint_limits(self):
        joint_limits = super()._get_joint_limits()
        joint_limits = joint_limits[self.control_joint_indices.cpu().numpy()]
        if len(joint_limits.shape) == 1:
            joint_limits = joint_limits[None, :]
        return joint_limits

    def set_action(self, action: Array):
        action = self._preprocess_action(action)
        self._step = 0
        self._start_qpos = self.qpos
        if self.config.use_delta:
            if self.config.use_target:
                self._target_qpos[:, self.control_joint_indices] += action
            else:
                self._target_qpos[:, self.control_joint_indices] = (
                    self._start_qpos[:, self.control_joint_indices] + action
                )
        else:
            self._target_qpos[:, self.control_joint_indices] = action
        self._target_qpos[:, self.mimic_joint_indices] = (
            self._target_qpos[:, self.mimic_control_joint_indices]
            * self._multiplier[None, :]
            + self._offset[None, :]
        )
        if self.config.interpolate:
            self._step_size = (self._target_qpos - self._start_qpos) / self._sim_steps
        else:
            self.set_drive_targets(self._target_qpos)

    def __repr__(self):
        data = {k: v["joint"] for k, v in self.config.mimic.items()}
        main_str = "{\n"
        for k, v in data.items():
            main_str += f"    {k}: {v},\n"
        main_str += "}"
        data = main_str
        return f"PDJointPosMimicController(dof={self.single_action_space.shape[0]}, active_joints={len(self.joints)}, mimic_to_control_joint_map={main_str})"


@dataclass
class MimicJointPosControllerConfig(PDJointPosControllerConfig):
    """
    PD Joint Position mimic controller configuration. This kind of controller is used to emuluate a mimic joint. For some simulation backends mimic joints are not
    well simulated and/or are hard to tune and so a mimic controller can be used for better stability as well as alignment with a real robot.

    The `joint_names` field is expected to be a list of all the joints, whether they are the joints to be controlled or the joints that are controlled via mimicing.

    `mimic` is a dictionary that maps the mimic joint name to a dictionary of the format dict(joint: str, multiplier: float, offset: float). `joint` is the controlling joint name and is required. All given joints in
    `joint_names` must be in the `mimic` dictionary as either a key or a value. The ones that are keys are referred to as the mimic joints and the ones that are values
    are referred to as the control joints. Mimic joints are the ones directly controlled by the agent/user and control joints are implicitly controlled by following the mimic joints via the following equation:

    q_mimic = q_controlling * multiplier + offset

    By default, the multiplier is 1.0 and the offset is 0.0.
    """

    controller_cls = MimicJointPosController
    mimic: dict[str, dict[str, float]] = field(default_factory=dict)
    """the mimic targets. Maps the actual mimic joint name to a dictionary of the format dict(joint: str, multiplier: float, offset: float)"""
