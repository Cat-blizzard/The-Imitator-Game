"""ManiSkill agents for the embodiment library.

* New arms (ARX X5, PiPER, UR5 + WSG-50, xArm6 + Allegro left/right): one generic BaseAgent subclass per record
  (`make_agent_class`), registered under record.agent_uid. Controllers:
      pd_joint_pos, pd_joint_delta_pos, pd_joint_target_delta_pos   arm joints (absolute / delta)
      pd_ee_delta_pose, pd_ee_target_delta_pose, pd_ee_pose          end-effector (record.tcp_link) via ManiSkill's
                                                                     PDEEPoseController (pinocchio on CPU,
                                                                     pytorch_kinematics on GPU)
  plus "gripper": parallel grippers = one normalized PD mimic controller (-1 .. 1 -> gripper_lower .. gripper_upper);
  hands = absolute PD targets for every hand joint (same controller in every mode, so protocol-v3's hold A works).
* Reused ManiSkill agent: xArm6 + Robotiq 2F-85 (mani_skill.agents.robots.xarm6.XArm6Robotiq) subclassed only to point
  at the derived URDF (extra 'tcp' link), use it as ee link, and rename its 'gripper_active' controller to 'gripper'.
  ManiSkill's own uids (panda, xarm6_robotiq, ...) stay available unchanged.

Every agent exposes: arm_joint_names, gripper_joint_names (all non-arm joints: excluded from robot-speed checks),
tcp (Link), finger1_link / finger2_link (pads, grippers only), grasp_links (list of Links), record (ArmRecord),
is_grasping(actor), measured_opening(), home_qpos_vector().
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import sapien
import torch

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import (PDEEPoseControllerConfig, PDJointPosControllerConfig,
                                           PassiveControllerConfig, deepcopy_dict)
from mani_skill.agents.registration import REGISTERED_AGENTS, register_agent
from mani_skill.utils import sapien_utils

from . import registry as reg
from .controllers import MimicJointPosControllerConfig

GRASP_MIN_FORCE = 0.5


class _ImsimMixin:
    record: reg.ArmRecord

    def _imsim_after_init(self):
        r = self.record
        links = self.robot.links_map
        self.tcp = links[r.tcp_link]
        self.ee_link = self.tcp
        self.grasp_links = [links[n] for n in r.grasp_links if n in links]
        if r.pad_links:
            self.finger1_link, self.finger2_link = links[r.pad_links[0]], links[r.pad_links[1]]
        self.hand = reg.hand_of(r)
        if self.hand:
            self.fingertip_links = [links[n] for n in self.hand.fingertip_links]

    @property
    def tcp_pose(self):
        return self.tcp.pose

    @property
    def tcp_pos(self):
        return self.tcp.pose.p

    def measured_opening(self):
        return reg.measured_opening(self, self.record)

    def home_qpos_vector(self, open_gripper: bool = True) -> np.ndarray:
        d = self.record.home_qpos(open_gripper)
        names = [j.name for j in self.robot.active_joints]
        return np.array([d.get(n, 0.0) for n in names], np.float32)

    def home_action(self, mode: str = "pd_joint_pos") -> np.ndarray:
        r = self.record
        if mode != "pd_joint_pos":
            raise ValueError("home_action is defined for pd_joint_pos")
        g = r.gripper_cmd_open if r.kind == "arm_gripper" else self.hand.synergy(0.0).tolist()
        return np.array(list(r.home_arm_qpos) + list(g), np.float32)

    def gripper_action(self, closed: float) -> np.ndarray:
        """gripper part of a pd_joint_pos action: closed in [0 (open), 1 (closed)] (hands: synergy value)"""
        r = self.record
        if r.kind == "arm_hand":
            return self.hand.synergy(closed)
        o, c = np.asarray(r.gripper_cmd_open, np.float32), np.asarray(r.gripper_cmd_closed, np.float32)
        return (1 - closed) * o + closed * c

    def is_grasping(self, obj, min_force: float = GRASP_MIN_FORCE):
        """grippers: both pads push on the object with >= min_force; hands: thumb tip and >= 1 other finger link do"""
        sc = self.scene
        if self.record.kind == "arm_gripper":
            f1 = torch.linalg.norm(sc.get_pairwise_contact_forces(self.finger1_link, obj), dim=-1)
            f2 = torch.linalg.norm(sc.get_pairwise_contact_forces(self.finger2_link, obj), dim=-1)
            return (f1 >= min_force) & (f2 >= min_force)
        lm = self.robot.links_map
        thumb = [lm[n] for n in ("link_14.0", "link_15.0", "link_15.0_tip") if n in lm]
        others = [lm[f"link_{i}.0"] for i in (1, 2, 3, 5, 6, 7, 9, 10, 11) if f"link_{i}.0" in lm] + \
                 [lm[n] for n in self.hand.fingertip_links if n != self.hand.thumb_tip]
        ft = sum(torch.linalg.norm(sc.get_pairwise_contact_forces(l, obj), dim=-1) for l in thumb)
        fo = torch.stack([torch.linalg.norm(sc.get_pairwise_contact_forces(l, obj), dim=-1) for l in others], -1)
        return (ft >= min_force) & ((fo >= min_force).sum(-1) >= 1)

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()
        idx = [i for i, j in enumerate(self.robot.active_joints) if j.name in self.record.arm_joints]
        return torch.max(torch.abs(qvel[..., idx]), 1)[0] <= threshold


def _arm_controllers(r: reg.ArmRecord) -> dict:
    kw = dict(stiffness=r.stiffness, damping=r.damping, force_limit=r.force_limit)
    arm_pos = PDJointPosControllerConfig(r.arm_joints, None, None, normalize_action=False, **kw)
    arm_dpos = PDJointPosControllerConfig(r.arm_joints, -0.1, 0.1, use_delta=True, **kw)
    arm_tdpos = deepcopy(arm_dpos); arm_tdpos.use_target = True
    ee_dpose = PDEEPoseControllerConfig(r.arm_joints, pos_lower=-0.1, pos_upper=0.1, rot_lower=-0.1, rot_upper=0.1,
                                        ee_link=r.tcp_link, urdf_path=r.urdf_path, **kw)
    ee_tdpose = deepcopy(ee_dpose); ee_tdpose.use_target = True
    ee_pose = PDEEPoseControllerConfig(r.arm_joints, pos_lower=-2.0, pos_upper=2.0, rot_lower=-np.pi, rot_upper=np.pi,
                                       ee_link=r.tcp_link, urdf_path=r.urdf_path, use_delta=False, normalize_action=False, **kw)
    return dict(pd_joint_pos=arm_pos, pd_joint_delta_pos=arm_dpos, pd_joint_target_delta_pos=arm_tdpos,
                pd_ee_delta_pose=ee_dpose, pd_ee_target_delta_pose=ee_tdpose, pd_ee_pose=ee_pose)


def _gripper_controller(r: reg.ArmRecord):
    gkw = dict(stiffness=r.gripper_stiffness, damping=r.gripper_damping, force_limit=r.gripper_force_limit)
    if r.kind == "arm_hand":
        return PDJointPosControllerConfig(r.gripper_joints, None, None, normalize_action=False, **gkw)
    return MimicJointPosControllerConfig(r.gripper_joints, lower=r.gripper_lower, upper=r.gripper_upper,
                                           mimic=deepcopy(r.gripper_mimic), **gkw)


def make_agent_class(r: reg.ArmRecord):
    """Generic ManiSkill agent for an ArmRecord (registered under r.agent_uid)."""
    grip_mat_links = list(r.pad_links) if r.pad_links else [l for l in r.grasp_links if l != "base_link"]

    class ImsimArm(_ImsimMixin, BaseAgent):
        uid = r.agent_uid
        urdf_path = r.urdf_path
        urdf_config = dict(
            _materials=dict(gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)),
            link={l: dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1) for l in grip_mat_links})
        record = r
        arm_joint_names = list(r.arm_joints)
        gripper_joint_names = list(r.gripper_joints) + list(r.passive_joints)
        ee_link_name = r.tcp_link
        disable_self_collisions = r.disable_self_collisions
        keyframes = dict(home=Keyframe(qpos=None, pose=sapien.Pose()))     # qpos filled by name at runtime (home_qpos_vector)

        @property
        def _controller_configs(self):
            g = _gripper_controller(r)
            cc = {m: dict(arm=a, gripper=g) for m, a in _arm_controllers(r).items()}
            return deepcopy_dict(cc)

        def _after_init(self):
            self._imsim_after_init()

    ImsimArm.__name__ = ImsimArm.__qualname__ = "Imsim_" + r.name
    return ImsimArm


def _make_xarm6_robotiq(r: reg.ArmRecord):
    from mani_skill.agents.robots.xarm6.xarm6_robotiq import XArm6Robotiq

    class ImsimXArm6Robotiq(_ImsimMixin, XArm6Robotiq):
        uid = r.agent_uid
        urdf_path = r.urdf_path
        record = r
        ee_link_name = r.tcp_link
        gripper_joint_names = list(r.gripper_joints) + list(r.passive_joints)

        @property
        def physical_gripper_coupling(self):
            from .coupling import ROBOTIQ_COUPLING
            return deepcopy(ROBOTIQ_COUPLING)

        def _after_loading_articulation(self):
            from .coupling import couple_robotiq_outer_knuckles
            self.gripper_tendon_count = couple_robotiq_outer_knuckles(self.robot)
            super()._after_loading_articulation()

        @property
        def _controller_configs(self):
            cc = super()._controller_configs
            out = {}
            for mode, d in cc.items():
                out[mode] = {("gripper" if k == "gripper_active" else k): v for k, v in d.items()}
            return out

        def _after_init(self):
            super()._after_init()
            self._imsim_after_init()

    return ImsimXArm6Robotiq


AGENT_CLASSES: dict[str, type] = {}


def register_all() -> dict[str, type]:
    """Register every record's agent with ManiSkill (idempotent)."""
    for name, r in reg.ARMS.items():
        if r.agent_uid in REGISTERED_AGENTS:
            AGENT_CLASSES[name] = REGISTERED_AGENTS[r.agent_uid].agent_cls
            continue
        cls = _make_xarm6_robotiq(r) if name == "xarm6_robotiq" else make_agent_class(r)
        register_agent(override=True)(cls)
        AGENT_CLASSES[name] = cls
    return AGENT_CLASSES


register_all()
