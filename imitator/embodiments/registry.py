"""Embodiment registry: one record per single arm (and per dexterous hand), the per-robot facts the planner, the
simulator and the task verifier need.

Every arm record points at a DERIVED URDF in imitator/assets/embodiments/<name>/ (built by `build_assets()`, see
urdf_tools.py): relative mesh paths into the vendored source (imitator/assets/embodiments/_src) (RoboTwin 2.0 / ManiSkill assets), corrected joint limits,
camera links dropped, an explicit TCP link (z = approach, y = finger closing axis, ManiSkill convention), and two SRDFs
(minimal for the simulator, full for mplib).

Verifier facts
  grasp_links      links that touch objects in a grasp (finger pads / whole hand). The release rule of the contract uses
                   ALL robot links (any contact = not released); grasp_links is used for the enclosure test and for
                   diagnostics.
  pad_links        the two opposing pads of a parallel gripper (None for hands).
  measured_opening(agent) -> torch (num_envs,): opening measured from the SIMULATED link poses (not the command):
                   parallel grippers: distance between the inner faces of the two pads along the closing axis, from the
                   pads' collision vertices; hands: thumb-tip to nearest-fingertip distance (pinch aperture).
  joint_opening    the same quantity from finger joint positions where the gripper is a pair of prismatic joints
                   (cross-check; None for linkage grippers / hands).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .urdf_tools import ExtraLink, derive_urdf, generate_srdfs, load_urdf, make_planning_urdf, matrix_to_rpy

from ..paths import EMBODIMENT_ASSETS, EMBODIMENT_SOURCES

GEN = EMBODIMENT_ASSETS                           # derived URDF/SRDF per arm (checked in)
RT_EMB = EMBODIMENT_SOURCES / "robotwin"          # RoboTwin 2.0 embodiments (MIT): ARX-X5, piper, ur5-wsg (camera meshes omitted)
MS_ASSET = EMBODIMENT_SOURCES                     # ManiSkill-XArm6 v0.1.1 (Apache-2.0) -> MS_ASSET / "xarm6"
MS_PKG = EMBODIMENT_SOURCES                       # ManiSkill 3.0.1 Allegro hand assets -> MS_PKG / "allegro"

PI = float(np.pi)


@dataclass
class HandRecord:
    name: str
    source: str
    license: str
    side: str                       # left | right
    joints: list[str]               # actuated hand joints (all position-controlled here)
    fingertip_links: list[str]      # thumb first
    thumb_tip: str
    synergy_open: dict[str, float]
    synergy_close: dict[str, float]
    notes: str = ""

    @property
    def dof(self) -> int:
        return len(self.joints)

    def synergy(self, s: float) -> np.ndarray:
        """hand joint targets for synergy value s in [0 (open), 1 (closed)], in `joints` order"""
        s = float(np.clip(s, 0.0, 1.2))
        return np.array([(1 - s) * self.synergy_open[j] + s * self.synergy_close[j] for j in self.joints], np.float32)


@dataclass
class ArmRecord:
    name: str
    kind: str                       # arm_gripper | arm_hand
    source: str
    source_urdf: str
    license: str
    agent_uid: str                  # ManiSkill uid (registered by imitator.embodiments.agents)
    agent_base: str                 # "imsim generic" or the reused ManiSkill agent class
    arm_joints: list[str]
    gripper_joints: list[str]       # actuated gripper / hand joints (controller "gripper")
    passive_joints: list[str]       # joints without actuator (linkage grippers)
    flange_link: str
    tcp_link: str
    tcp: ExtraLink | None           # TCP added to the derived URDF (None: already in the source URDF)
    grasp_links: list[str]
    pad_links: tuple[str, str] | None
    gripper_max_opening_m: float
    gripper_cmd_open: list[float]   # gripper part of the action (pd_joint_pos) for fully open
    gripper_cmd_closed: list[float] # ... and for closed (squeezing)
    joint_limits: dict[str, tuple[float, float]]
    velocity_limits: dict[str, float]
    home_arm_qpos: list[float]      # arm joints only; gripper/hand open
    mount_yaw: float                # base yaw so that the arm's front faces +x (world) [rad]
    reach_m: float                  # nominal reach (vendor), for workspace sizing
    stiffness: float = 1e3
    damping: float = 1e2
    force_limit: float | list = 100.0      # arm drive force/torque limit(s) [N m]; per joint where the vendor states them
    gripper_stiffness: float = 1e3
    gripper_damping: float = 1e2
    gripper_force_limit: float = 100.0
    gripper_lower: float = 0.0      # PD target range of the gripper controller (normalized action -1 .. 1)
    gripper_upper: float = 0.0
    gripper_mimic: dict = field(default_factory=dict)
    hand: str | None = None         # HandRecord name for arm_hand
    drop_links: tuple = ()
    path_map: dict = field(default_factory=dict)
    curobo_yml: str | None = None
    collision_spheres_yml: str | None = None
    source_srdf: str | None = None
    control_modes: tuple = ("pd_joint_pos", "pd_joint_delta_pos", "pd_ee_delta_pose", "pd_ee_pose")
    disable_self_collisions: bool = False    # simulator only (ManiSkill's group-2 bit 29 on every link; dual.py keeps the
                                             # two arms colliding with each other); the planner still checks them
    finger_side_links: tuple | None = None   # ([links of finger 1], [links of finger 2]) for opening / enclosure;
                                             # default ([pad_links[0]], [pad_links[1]])
    notes: str = ""

    @property
    def sides(self) -> tuple[list[str], list[str]] | None:
        if self.finger_side_links:
            return list(self.finger_side_links[0]), list(self.finger_side_links[1])
        return ([self.pad_links[0]], [self.pad_links[1]]) if self.pad_links else None

    # ---------------------------------------------------------------- derived paths
    @property
    def urdf_path(self) -> str:
        return str(GEN / self.name / f"{self.name}.urdf")

    @property
    def srdf_path(self) -> str:
        return str(GEN / self.name / f"{self.name}.srdf")

    @property
    def mplib_urdf_path(self) -> str:
        return str(GEN / self.name / f"{self.name}_mplib.urdf")

    @property
    def mplib_srdf_path(self) -> str:
        return str(GEN / self.name / f"{self.name}_mplib.srdf")

    @property
    def dof_arm(self) -> int:
        return len(self.arm_joints)

    @property
    def dof_gripper(self) -> int:
        """actuated gripper / hand DOF (mimic-coupled finger pairs count once)"""
        return len(self.gripper_joints) - len(self.gripper_mimic)

    @property
    def action_dim(self) -> int:
        return self.dof_arm + self.dof_gripper

    def planner_config(self) -> dict:
        return dict(urdf=self.mplib_urdf_path, srdf=self.mplib_srdf_path, move_group=self.tcp_link,
                    arm_joints=self.arm_joints, curobo_yml=self.curobo_yml, collision_spheres=self.collision_spheres_yml)

    def home_qpos(self, open_gripper: bool = True) -> dict[str, float]:
        d = dict(zip(self.arm_joints, self.home_arm_qpos))
        if self.kind == "arm_hand":
            h = HANDS[self.hand]
            d.update(dict(zip(h.joints, h.synergy(0.0).tolist())))
        else:
            g = self.gripper_open_qpos() if open_gripper else {j: 0.0 for j in self.gripper_joints}
            d.update(g)
        for j in self.passive_joints:
            d.setdefault(j, 0.0)
        return d

    def gripper_open_qpos(self) -> dict[str, float]:
        lim = {j: self.joint_limits.get(j) for j in self.gripper_joints}
        out = {}
        for j in self.gripper_joints:
            lo, hi = lim[j] if lim[j] else (0.0, 0.0)
            out[j] = hi if abs(hi) >= abs(lo) else lo           # the limit farthest from 0 = open for these grippers
        return out

    def to_json(self) -> dict:
        d = asdict(self)
        d.update(urdf_path=self.urdf_path, srdf_path=self.srdf_path, mplib_srdf_path=self.mplib_srdf_path,
                 dof_arm=self.dof_arm, dof_gripper=self.dof_gripper, action_dim=self.action_dim)
        d["tcp"] = asdict(self.tcp) if self.tcp else None
        return d


def _R_cols(x, y, z) -> tuple:
    return matrix_to_rpy(np.stack([np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)], axis=1))


# ------------------------------------------------------------------------------------------------ hands
_ALLEGRO_FINGERS = [f"joint_{i}.0" for i in range(16)]
_ALLEGRO_OPEN = {**{f"joint_{i}.0": 0.0 for i in range(12)}, "joint_12.0": 0.9, "joint_13.0": 0.3, "joint_14.0": 0.2,
                 "joint_15.0": 0.2}
# power-grasp synergy (fixed): MCP/PIP/DIP of index, middle, ring flexed, thumb rotated into opposition and flexed
_ALLEGRO_CLOSE = {"joint_0.0": 0.0, "joint_1.0": 1.25, "joint_2.0": 1.05, "joint_3.0": 0.9,
                  "joint_4.0": 0.0, "joint_5.0": 1.25, "joint_6.0": 1.05, "joint_7.0": 0.9,
                  "joint_8.0": 0.0, "joint_9.0": 1.25, "joint_10.0": 1.05, "joint_11.0": 0.9,
                  "joint_12.0": 1.35, "joint_13.0": 0.55, "joint_14.0": 0.9, "joint_15.0": 0.9}

HANDS: dict[str, HandRecord] = {}
for _side in ("left", "right"):
    HANDS[f"allegro_{_side}"] = HandRecord(
        name=f"allegro_{_side}",
        source=f"Wonik Allegro Hand v4 ({_side}); URDF and meshes as shipped in ManiSkill 3.0.1 (mani_skill/assets/robots/allegro), "
               f"mounted on xArm6 by ManiSkill-XArm6 v0.1.1 (xarm6_allegro_{_side}.urdf)",
        license="BSD-3-Clause (allegro/LICENSE, Copyright (c) 2016 SimLab); mount URDF Apache-2.0 (ManiSkill-XArm6)",
        side=_side, joints=list(_ALLEGRO_FINGERS),
        fingertip_links=["link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip"], thumb_tip="link_15.0_tip",
        synergy_open=dict(_ALLEGRO_OPEN), synergy_close=dict(_ALLEGRO_CLOSE),
        notes="16 DOF, all position-controlled; the scripted power grasp uses the fixed 1-D synergy open->close.")


# ------------------------------------------------------------------------------------------------ arms
def _arx_limits():
    return {"joint1": (-3.14, 2.618), "joint2": (-0.05, 3.50), "joint3": (-0.1, 3.20), "joint4": (-1.6, 1.55),
            "joint5": (-1.57, 1.57), "joint6": (-2.0, 2.0), "joint7": (0.0, 0.044), "joint8": (0.0, 0.044)}


ARMS: dict[str, ArmRecord] = {}

ARMS["arx_x5"] = ArmRecord(
    name="arx_x5", kind="arm_gripper",
    source="ARX X5 (ARX Robotics) as packaged by RoboTwin 2.0 (assets/embodiments/ARX-X5/X5A.urdf); joint limits, velocity "
           "limits and gripper width from real-stanford/arx5-sdk include/app/config.h ('X5')",
    source_urdf=str(RT_EMB / "ARX-X5/X5A.urdf"),
    license="MIT (RoboTwin 2.0 embodiment assets: README 'license: MIT License'; RoboTwin repo LICENSE MIT); joint limits from "
            "arx5-sdk (MIT, Copyright (c) 2024 Yihuai Gao). Upstream ARX URDF license not stated.",
    agent_uid="imsim_arx_x5", agent_base="imsim generic",
    arm_joints=[f"joint{i}" for i in range(1, 7)], gripper_joints=["joint7", "joint8"], passive_joints=[],
    flange_link="link6", tcp_link="tcp",
    tcp=ExtraLink("tcp", "link6", (0.13, 0.0, 0.0), _R_cols([0, 0, -1], [0, 1, 0], [1, 0, 0])),
    grasp_links=["link7", "link8", "link6"], pad_links=("link7", "link8"),
    gripper_max_opening_m=0.088, gripper_cmd_open=[1.0], gripper_cmd_closed=[-1.0],
    joint_limits=_arx_limits(),
    velocity_limits={"joint1": 5.0, "joint2": 5.0, "joint3": 5.5, "joint4": 5.5, "joint5": 5.0, "joint6": 5.0, "joint7": 0.3,
                     "joint8": 0.3},
    home_arm_qpos=[0.0] * 6, mount_yaw=0.0, reach_m=0.62, force_limit=[30.0, 40.0, 30.0, 15.0, 10.0, 10.0],
    gripper_lower=-0.01, gripper_upper=0.044, gripper_mimic={"joint8": {"joint": "joint7"}},
    drop_links=("camera_base",),
    curobo_yml=str(RT_EMB / "ARX-X5/curobo.yml"), collision_spheres_yml=str(RT_EMB / "ARX-X5/collision_X5A.yml"),
    notes="RoboTwin's X5A.urdf writes +-10 rad joint limits (placeholders); replaced by the SDK's limits. Fingers are two "
          "independent prismatic joints in the URDF; the gripper controller couples them (mimic). TCP 0.13 m along link6 x "
          "(finger pads span 0.072-0.158 m; RoboTwin uses 0.12, the SDK's eef_link 0.145).")

ARMS["piper"] = ArmRecord(
    name="piper", kind="arm_gripper",
    source="AgileX PiPER 6-DOF arm with parallel gripper, as packaged by RoboTwin 2.0 (assets/embodiments/piper/piper.urdf; "
           "official joint limits from the bundled AgileX piper_description.urdf)",
    source_urdf=str(RT_EMB / "piper/piper.urdf"),
    license="MIT (RoboTwin 2.0 embodiment assets); upstream agilexrobotics/piper_ros is MIT per its repository (not re-checked here)",
    agent_uid="imsim_piper", agent_base="imsim generic",
    arm_joints=[f"joint{i}" for i in range(1, 7)], gripper_joints=["joint7", "joint8"], passive_joints=[],
    flange_link="link6", tcp_link="tcp",
    tcp=ExtraLink("tcp", "link6", (0.0, 0.0, 0.12), (0.0, 0.0, 0.0)),
    grasp_links=["link7", "link8", "link6"], pad_links=("link7", "link8"),
    gripper_max_opening_m=0.08, gripper_cmd_open=[1.0], gripper_cmd_closed=[-1.0],
    joint_limits={"joint1": (-2.618, 2.618), "joint2": (0.0, 3.14), "joint3": (-2.697, 0.0), "joint4": (-1.832, 1.832),
                  "joint5": (-1.22, 1.22), "joint6": (-3.14, 3.14), "joint7": (0.0, 0.04), "joint8": (0.0, 0.04)},
    velocity_limits={f"joint{i}": 3.0 for i in range(1, 7)} | {"joint7": 1.0, "joint8": 1.0},
    home_arm_qpos=[0.0] * 6, mount_yaw=0.0, reach_m=0.626,
    gripper_lower=-0.01, gripper_upper=0.04, gripper_mimic={"joint8": {"joint": "joint7"}},
    drop_links=("camera_mount",),
    curobo_yml=str(RT_EMB / "piper/curobo.yml"), collision_spheres_yml=str(RT_EMB / "piper/collision_piper.yml"),
    source_srdf=str(RT_EMB / "piper/piper.srdf"),
    notes="RoboTwin's piper.urdf widens joint3 to +-2.697; the official AgileX description (same folder) has [-2.697, 0], used "
          "here. TCP 0.12 m along link6 z (pads span 0.064-0.140 m; RoboTwin gripper_bias 0.12).")

ARMS["ur5_wsg"] = ArmRecord(
    name="ur5_wsg", kind="arm_gripper",
    source="Universal Robots UR5 with Weiss WSG-50 parallel gripper, as packaged by RoboTwin 2.0 "
           "(assets/embodiments/ur5-wsg/ur5_wsg_gripper.urdf, ur5.srdf)",
    source_urdf=str(RT_EMB / "ur5-wsg/ur5_wsg_gripper.urdf"),
    license="MIT (RoboTwin 2.0 embodiment assets); UR5 description originates from ros-industrial universal_robot (BSD-3, "
            "not re-checked here); WSG-50 meshes origin not stated",
    agent_uid="imsim_ur5_wsg", agent_base="imsim generic",
    arm_joints=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
    gripper_joints=["base_joint_gripper_left", "base_joint_gripper_right"], passive_joints=[],
    flange_link="ee_link", tcp_link="tcp",
    tcp=ExtraLink("tcp", "ee_link", (0.117, 0.0, 0.0), _R_cols([0, 0, -1], [0, 1, 0], [1, 0, 0])),
    grasp_links=["finger_left", "finger_right", "gripper_left", "gripper_right", "wsg_50_base_link"],
    pad_links=("finger_left", "finger_right"),
    gripper_max_opening_m=0.104, gripper_cmd_open=[-1.0], gripper_cmd_closed=[1.0],
    joint_limits={"shoulder_pan_joint": (-3.142, 3.142), "shoulder_lift_joint": (-3.142, 0.0), "elbow_joint": (-3.142, 3.142),
                  "wrist_1_joint": (-3.142, 3.142), "wrist_2_joint": (-3.142, 3.142), "wrist_3_joint": (-3.142, 3.142),
                  "base_joint_gripper_left": (-0.055, -0.0027), "base_joint_gripper_right": (0.0027, 0.055)},
    velocity_limits={"shoulder_pan_joint": 3.15, "shoulder_lift_joint": 3.15, "elbow_joint": 3.15, "wrist_1_joint": 3.2,
                     "wrist_2_joint": 3.2, "wrist_3_joint": 3.2},
    home_arm_qpos=[-0.5191, -2.3408, 2.328, -1.558, -1.5708, -0.5191], mount_yaw=0.0, reach_m=0.85,
    force_limit=[150.0, 150.0, 150.0, 28.0, 28.0, 28.0],
    gripper_lower=-0.055, gripper_upper=0.002,
    gripper_mimic={"base_joint_gripper_right": {"joint": "base_joint_gripper_left", "multiplier": -1.0}},
    drop_links=("camera_base",),
    curobo_yml=str(RT_EMB / "ur5-wsg/curobo.yml"), collision_spheres_yml=str(RT_EMB / "ur5-wsg/collision_wsg.yml"),
    source_srdf=str(RT_EMB / "ur5-wsg/ur5.srdf"),
    notes="Gripper: left finger joint in [-0.055, -0.0027], right = -left (mimic). Normalized gripper action -1 -> -0.055 "
          "(fully open), +1 -> +0.002 (closed, squeezing): the sign is the reverse of the other grippers, so planners use "
          "gripper_cmd_open / gripper_cmd_closed. wrist_1 limit widened from RoboTwin's +-1.75 to +-pi "
          "(UR5 hardware: +-2 pi). TCP 0.117 m along ee_link x (finger pads span 0.081-0.154 m).")

ARMS["xarm6_robotiq"] = ArmRecord(
    name="xarm6_robotiq", kind="arm_gripper",
    source="UFACTORY xArm6 with Robotiq 2F-85, ManiSkill-XArm6 v0.1.1 asset (xarm6_robotiq.urdf/.srdf) and ManiSkill 3.0.1 "
           "agent XArm6Robotiq (reused: closed-chain finger drives, passive joints, controllers)",
    source_urdf=str(MS_ASSET / "xarm6/xarm6_robotiq.urdf"),
    license="Apache-2.0 (haosulab/ManiSkill-XArm6 LICENSE); xArm description from xarm_ros (BSD-3, not re-checked); Robotiq "
            "meshes from ros-industrial robotiq (BSD-2, not re-checked)",
    agent_uid="imsim_xarm6_robotiq", agent_base="mani_skill.agents.robots.xarm6.XArm6Robotiq",
    arm_joints=[f"joint{i}" for i in range(1, 7)],
    gripper_joints=["left_outer_knuckle_joint", "right_outer_knuckle_joint"],
    passive_joints=["left_inner_knuckle_joint", "right_inner_knuckle_joint", "left_inner_finger_joint", "right_inner_finger_joint"],
    flange_link="link6", tcp_link="tcp",
    tcp=ExtraLink("tcp", "link6", (0.0, 0.0, 0.125), (0.0, 0.0, 0.0)),
    grasp_links=["left_inner_finger_pad", "right_inner_finger_pad", "left_inner_finger", "right_inner_finger",
                 "left_outer_finger", "right_outer_finger", "left_inner_knuckle", "right_inner_knuckle",
                 "left_outer_knuckle", "right_outer_knuckle", "robotiq_arg2f_base_link"],
    pad_links=("left_inner_finger_pad", "right_inner_finger_pad"),
    finger_side_links=(["left_inner_finger_pad", "left_inner_finger"], ["right_inner_finger_pad", "right_inner_finger"]),
    gripper_max_opening_m=0.085, gripper_cmd_open=[0.0], gripper_cmd_closed=[0.81],
    joint_limits={"joint1": (-6.28318530718, 6.28318530718), "joint2": (-2.059, 2.0944), "joint3": (-3.8, 0.19198),
                  "joint4": (-6.28318530718, 6.28318530718), "joint5": (-1.69297, 3.14159265359),
                  "joint6": (-6.28318530718, 6.28318530718)},
    velocity_limits={f"joint{i}": 3.14 for i in range(1, 7)},
    home_arm_qpos=[0.0, 0.22, -1.23, 0.0, 1.01, 0.0], mount_yaw=0.0, reach_m=0.70,
    gripper_mimic={"right_outer_knuckle_joint": {"joint": "left_outer_knuckle_joint"}},
    control_modes=("pd_joint_pos", "pd_joint_delta_pos", "pd_ee_delta_pose", "pd_ee_pose"),
    source_srdf=str(MS_ASSET / "xarm6/xarm6_robotiq.srdf"),
    notes="Gripper command is the outer-knuckle angle: 0 open, 0.81 closed (absolute, ManiSkill convention for this robot). "
          "ManiSkill's own 'eef' link sits at 0.15 m, beyond the pad tips (0.149 m); imsim adds 'tcp' at 0.125 m (pad centre "
          "between open and closed). ManiSkill sets collision-group-2 bit 31 on every Robotiq link of EVERY copy, which "
          "disables gripper-gripper contact between the two arms; dual.py re-assigns the bits per arm.")

for _side, _yaw in (("left", 0.0), ("right", 0.0)):
    ARMS[f"xarm6_allegro_{_side}"] = ArmRecord(
        name=f"xarm6_allegro_{_side}", kind="arm_hand",
        source=f"UFACTORY xArm6 + Wonik Allegro Hand ({_side}), ManiSkill-XArm6 v0.1.1 xarm6_allegro_{_side}.urdf/.srdf "
               f"(Allegro meshes from ManiSkill 3.0.1 mani_skill/assets/robots/allegro); no ManiSkill agent exists for it",
        source_urdf=str(MS_ASSET / f"xarm6/xarm6_allegro_{_side}.urdf"),
        license="Apache-2.0 (ManiSkill-XArm6); Allegro BSD-3-Clause (SimLab)",
        agent_uid=f"imsim_xarm6_allegro_{_side}", agent_base="imsim generic",
        arm_joints=[f"joint{i}" for i in range(1, 7)], gripper_joints=list(_ALLEGRO_FINGERS), passive_joints=[],
        flange_link="link6", tcp_link="palm_tcp",
        tcp=ExtraLink("palm_tcp", "link6", (-0.045, 0.0, 0.13), _R_cols([0, -1, 0], [0, 0, 1], [-1, 0, 0])),
        grasp_links=["base_link"] + [f"link_{i}.0" for i in range(16)] + [f"link_{i}.0_tip" for i in (3, 7, 11, 15)],
        pad_links=None, gripper_max_opening_m=0.20, gripper_cmd_open=[], gripper_cmd_closed=[],
        joint_limits={"joint1": (-3.14, 3.14), "joint2": (-2.059, 2.0944), "joint3": (-3.927, 0.19198), "joint4": (-3.14, 3.14),
                      "joint5": (-1.69297, 3.14159265359), "joint6": (-3.14, 3.14)},
        velocity_limits={f"joint{i}": 3.14 for i in range(1, 7)},
        home_arm_qpos=[0.0, 0.22, -1.23, 0.0, 1.01, 0.0], mount_yaw=_yaw, reach_m=0.70,
        force_limit=[50.0, 50.0, 32.0, 32.0, 32.0, 20.0],
        gripper_stiffness=3e2, gripper_damping=3e1, gripper_force_limit=10.0, disable_self_collisions=True,
        hand=f"allegro_{_side}", path_map={"../allegro/": str(MS_PKG / "allegro") + "/"},
        source_srdf=str(MS_ASSET / f"xarm6/xarm6_allegro_{_side}.srdf"),
        notes="'palm_tcp': 4.5 cm in front of the palm (palm normal = -x of link6, fingers along +z of link6); z = palm "
              "normal (approach), y = finger direction. gripper_cmd_* are hand synergies (HandRecord). Simulator "
              "self-collisions are disabled for this arm+hand: at the home pose PhysX tracked 111 hand-internal contact "
              "pairs within the contact offset (no penetration), which cut CPU throughput to ~14 steps/s.")


def get_arm(name: str) -> ArmRecord:
    return ARMS[name]


def hand_of(rec: ArmRecord) -> HandRecord | None:
    return HANDS.get(rec.hand) if rec.hand else None


# ------------------------------------------------------------------------------------------------ assets
def build_assets(names=None, force: bool = False, n_samples: int = 3000) -> dict:
    """Write derived URDFs + SRDFs for the given arms (default: all). Idempotent unless force."""
    out = {}
    for n in (names or ARMS):
        r = ARMS[n]
        if os.path.exists(r.urdf_path) and os.path.exists(r.srdf_path) and not force:
            out[n] = "cached"
            continue
        os.makedirs(os.path.dirname(r.urdf_path), exist_ok=True)
        lim = {k: v for k, v in r.joint_limits.items()}
        mimic = {k: (v["joint"], v.get("multiplier", 1.0), v.get("offset", 0.0)) for k, v in r.gripper_mimic.items()}
        derive_urdf(r.source_urdf, r.urdf_path, joint_limits=lim, drop_links=r.drop_links,
                    extra_links=[r.tcp] if r.tcp else [], robot_name=r.name, velocity_limits=r.velocity_limits,
                    mimic=mimic if r.kind == "arm_gripper" and r.name != "xarm6_robotiq" else None,
                    notes=f"(imsim record {r.name})", path_map=r.path_map)
        home = r.home_qpos()
        skip = []
        if r.name == "xarm6_robotiq":
            # the closed-chain linkage links overlap by construction; ManiSkill's agent filters them (and link5) with
            # collision-group-2 bit 31, so the simulator SRDF leaves them out (SAPIEN allows at most 31 SRDF pairs)
            skip = [list(r.grasp_links) + ["link5", "link6"]]
        make_planning_urdf(r.urdf_path, r.mplib_urdf_path)
        s = generate_srdfs(r.urdf_path, home=home, n_samples=n_samples, planning_urdf=r.mplib_urdf_path, sim_skip_groups=skip)
        out[n] = dict(n_sim_pairs=s["n_sim_pairs"], n_mplib_pairs=s["n_mplib_pairs"], home_collisions=s["home_collisions"])
    return out


# ------------------------------------------------------------------------------------------------ opening
_PAD_CACHE: dict = {}


def _hull_samples(V: np.ndarray, n: int = 1500) -> np.ndarray:
    """dense samples on the convex hull of V (PhysX collides with the hull) plus its vertices"""
    if len(V) < 4:
        return V
    import trimesh
    h = trimesh.Trimesh(V).convex_hull
    pts, _ = trimesh.sample.sample_surface_even(h, n, seed=0)
    return np.vstack([pts, h.vertices])


def _pad_vertices(rec: ArmRecord):
    """per finger side: [(link, points in the link frame, is_pad)], points = dense samples of the link's collision hull"""
    if rec.name not in _PAD_CACHE:
        m = load_urdf(rec.urdf_path)
        _PAD_CACHE[rec.name] = [[(l, _hull_samples(m.collision_points(l)), l in rec.pad_links) for l in side]
                                for side in rec.sides]
    return _PAD_CACHE[rec.name]


def pad_geometry(rec: ArmRecord, Ts: dict) -> dict:
    """Inner faces of the two fingers (world frame) from link transforms Ts {link: 4x4} (one env).
    u = pad-1 -> pad-2 centroid direction (closing axis); the lateral region (v, w intervals) is the overlap of the two
    PADS' extents; each side's inner face = its innermost hull point (over all of the side's links, e.g. Robotiq pad +
    inner finger) inside that lateral region. Returns u, v, w, s1 < s2, opening = s2 - s1, iv, iw."""
    sides = _pad_vertices(rec)
    W, P = [], []
    for side in sides:
        Ws = [(Ts[l][:3, :3] @ V.T).T + Ts[l][:3, 3] for l, V, _ in side if len(V)]
        W.append(np.vstack(Ws))
        P.append(np.vstack([w for w, (_, V, pad) in zip(Ws, [x for x in side if len(x[1])]) if pad]))
    u = P[1].mean(0) - P[0].mean(0)
    u = u / (np.linalg.norm(u) + 1e-12)
    a = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1.0, 0])
    v = np.cross(u, a); v /= np.linalg.norm(v); w = np.cross(u, v)
    iv = (max((P[0] @ v).min(), (P[1] @ v).min()), min((P[0] @ v).max(), (P[1] @ v).max()))
    iw = (max((P[0] @ w).min(), (P[1] @ w).min()), min((P[0] @ w).max(), (P[1] @ w).max()))

    def inside(X):
        xv, xw = X @ v, X @ w
        return X[(xv >= iv[0]) & (xv <= iv[1]) & (xw >= iw[0]) & (xw <= iw[1])]
    W1, W2 = inside(W[0]), inside(W[1])
    if not len(W1) or not len(W2):
        W1, W2 = P[0], P[1]
    s1 = float((W1 @ u).max()); s2 = float((W2 @ u).min())
    return dict(u=u, v=v, w=w, s1=s1, s2=s2, opening=s2 - s1, iv=iv, iw=iw, W1=W1, W2=W2, W1_all=W[0], W2_all=W[1],
                c1=P[0].mean(0), c2=P[1].mean(0))


def measured_opening(agent, rec: ArmRecord | None = None):
    """Measured gripper opening [m] per parallel env (torch, on the agent's device), from the SIMULATED link poses:
    parallel grippers = pad_geometry's opening (innermost collision-hull points of each finger side inside the pads'
    lateral overlap, along the closing axis); hands = thumb tip to nearest other fingertip (pinch aperture).
    Works on CPU and GPU sim (batched)."""
    import torch
    rec = rec or record_of_agent(agent)
    lm = agent.robot.links_map
    if rec.kind == "arm_hand":
        h = HANDS[rec.hand]
        th = lm[h.thumb_tip].pose.p
        d = torch.stack([torch.linalg.norm(lm[t].pose.p - th, dim=-1) for t in h.fingertip_links if t != h.thumb_tip], -1)
        return d.min(-1).values
    Ws, Ps = [], []
    for side in _pad_vertices(rec):
        parts, pads = [], []
        for l, V, pad in side:
            if not len(V):
                continue
            Vt = torch.as_tensor(V, dtype=torch.float32, device=agent.device)
            T = lm[l].pose.to_transformation_matrix()
            X = torch.einsum("bij,nj->bni", T[:, :3, :3], Vt) + T[:, None, :3, 3]
            parts.append(X)
            if pad:
                pads.append(X)
        Ws.append(torch.cat(parts, 1)); Ps.append(torch.cat(pads, 1))
    u = Ps[1].mean(1) - Ps[0].mean(1)
    u = u / (torch.linalg.norm(u, dim=-1, keepdim=True) + 1e-12)
    a = torch.zeros_like(u); a[:, 0] = 1.0
    alt = torch.zeros_like(u); alt[:, 1] = 1.0
    a = torch.where((u[:, :1].abs() < 0.9), a, alt)
    v = torch.linalg.cross(u, a); v = v / torch.linalg.norm(v, dim=-1, keepdim=True)
    w = torch.linalg.cross(u, v)
    pv = [torch.einsum("bni,bi->bn", P, v) for P in Ps]
    pw = [torch.einsum("bni,bi->bn", P, w) for P in Ps]
    v0 = torch.maximum(pv[0].min(1).values, pv[1].min(1).values); v1 = torch.minimum(pv[0].max(1).values, pv[1].max(1).values)
    w0 = torch.maximum(pw[0].min(1).values, pw[1].min(1).values); w1 = torch.minimum(pw[0].max(1).values, pw[1].max(1).values)
    out = []
    for k, W in enumerate(Ws):
        xu = torch.einsum("bni,bi->bn", W, u)
        xv = torch.einsum("bni,bi->bn", W, v); xw = torch.einsum("bni,bi->bn", W, w)
        m = (xv >= v0[:, None]) & (xv <= v1[:, None]) & (xw >= w0[:, None]) & (xw <= w1[:, None])
        if k == 0:
            out.append(torch.where(m, xu, torch.full_like(xu, -1e9)).max(1).values)
        else:
            out.append(torch.where(m, xu, torch.full_like(xu, 1e9)).min(1).values)
    return out[1] - out[0]


def joint_opening(agent, rec: ArmRecord | None = None):
    """opening from finger joint positions (prismatic grippers only; None otherwise)"""
    rec = rec or record_of_agent(agent)
    q = agent.robot.get_qpos()
    names = [j.name for j in agent.robot.active_joints]
    if rec.name in ("arx_x5", "piper"):
        i7, i8 = names.index("joint7"), names.index("joint8")
        return q[:, i7] + q[:, i8] + (0.0008 if rec.name == "arx_x5" else 0.0)
    if rec.name == "ur5_wsg":
        il, ir = names.index("base_joint_gripper_left"), names.index("base_joint_gripper_right")
        return (q[:, ir] - q[:, il]) - 0.006                                 # inner pad faces sit 3 mm inside the joint frames
    return None


def record_of_agent(agent) -> ArmRecord:
    uid = getattr(agent, "uid", "")
    for r in ARMS.values():
        if r.agent_uid == uid:
            return r
    raise KeyError(f"no embodiment record for agent uid {uid!r}")


def summary_table() -> list[dict]:
    rows = []
    for r in ARMS.values():
        rows.append(dict(name=r.name, kind=r.kind, dof_arm=r.dof_arm, dof_gripper=r.dof_gripper, tcp=r.tcp_link,
                         pads=r.pad_links, max_opening=r.gripper_max_opening_m, license=r.license.split(";")[0],
                         agent=r.agent_uid, reuse=r.agent_base))
    return rows


def dump_json(path: str) -> None:
    with open(path, "w") as f:
        json.dump(dict(arms={k: v.to_json() for k, v in ARMS.items()}, hands={k: asdict(v) for k, v in HANDS.items()}), f,
                  indent=1, default=str)
