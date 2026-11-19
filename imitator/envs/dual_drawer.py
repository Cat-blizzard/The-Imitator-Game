"""DualDrawer-v1 (E29b): stabilise and open. A free-standing (dynamic, NOT fixed) small cabinet with a prismatic drawer
stands on the table; the left arm holds the cabinet still by a knob on its side wall, the right arm pulls the drawer open by
the handle on the drawer front, then both let go and move away.

Scene (world frame of imitator.embodiments.dual: table top z = 0, robots at x = 0 facing +x)
  cabinet   root link (dynamic articulation root), outer 16 (x) x 16 (y) x 10 (z) cm, panels 8 mm, open towards -x
            (the robots), density 600 kg/m^3 (body ~0.5 kg); side knob 3 (x) x 8 (y, protruding) x 4 (z) cm on the
            left (+y) side wall, level with the cabinet centre, at mid-height (the holder grasps it top-down, closing
            axis = cabinet x, so the drawer's pull reaches the holder as a normal force on its pads)
  drawer    child link, prismatic along the cabinet's -x axis, travel [0, 0.12] m; tray 13 x 13.8 x 7.6 cm, front plate
            15.6 x 9.6 cm, handle knob 6 (x, protruding) x 3 (y) x 2.4 (z) cm on the plate's upper half, 3 cm right of
            its centre (drawer ~0.29 kg);
            joint drive: stiffness 0, damping 120 N s/m (a sticky drawer: pulling at 5 cm/s takes ~6 N, more than
            the ~2.2 N of table friction on the cabinet, so an unheld cabinet slides along); knob and handle friction 1.5
  reset     cabinet centre x in [0.36, 0.40], y in [-0.02, 0.02], yaw in [-0.1, 0.1] rad, drawer closed; arms at home
  success   benchmark-style flag (written the way a RoboTwin / ManiSkill drawer task does it, for the card's bench column
            only; the contract judges episodes): drawer opening >= 8 cm at the evaluated step. One step: no settle, no
            release check, no check that the cabinet stayed where it was.
Signals for the contract: imitator.verify.dual_tasks_b2.drawer.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import sapien
import torch

from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from ..embodiments import dual as D
from .dual_tabletop import DualTabletopEnv

CAB = dict(D=0.16, W=0.16, H=0.10, t=0.008)
SIDE_KNOB_HALF = (0.015, 0.04, 0.02)           # holder knob on the cabinet's left (+y) side wall (x, protrusion y, z)
SIDE_KNOB_X = 0.0
TRAY = dict(D=0.13, W=0.138, H=0.076, t=0.006)
HANDLE_HALF = (0.03, 0.015, 0.012)
HANDLE_Y = -0.03                               # handle offset on the drawer front (towards the right arm)
TRAVEL = 0.12
GRIP_FRICTION = 1.5            # friction of the knob and the handle (PhysX averages it with the pads'); panels keep 0.3
OPEN_BENCH = 0.08


def _rotz(a: float):
    return [float(np.cos(a / 2)), 0.0, 0.0, float(np.sin(a / 2))]


def geometry() -> dict:
    """part centres / half extents in the cabinet (root) and drawer (child) link frames"""
    Dc, Wc, Hc, t = CAB["D"], CAB["W"], CAB["H"], CAB["t"]
    Dd, Wd, Hd, tb = TRAY["D"], TRAY["W"], TRAY["H"], TRAY["t"]
    body = [((0, 0, -Hc / 2 + t / 2), (Dc / 2, Wc / 2, t / 2)),            # bottom
            ((0, 0, Hc / 2 - t / 2), (Dc / 2, Wc / 2, t / 2)),             # top
            ((Dc / 2 - t / 2, 0, 0), (t / 2, Wc / 2, Hc / 2 - t)),         # back (+x)
            ((0, Wc / 2 - t / 2, 0), (Dc / 2, t / 2, Hc / 2 - t)),         # left
            ((0, -Wc / 2 + t / 2, 0), (Dc / 2, t / 2, Hc / 2 - t))]        # right
    knob_c = (SIDE_KNOB_X, Wc / 2 + SIDE_KNOB_HALF[1], 0.0)
    xc = -Dc / 2 + Dd / 2                                                  # drawer origin in the cabinet frame (closed)
    zc = -Hc / 2 + t + 0.002 + Hd / 2
    plate_c = (-Dd / 2 - 0.006, 0.0, -zc)
    plate_h = (0.005, (Wc - 0.004) / 2, (Hc - 0.004) / 2)
    handle_c = (plate_c[0] - 0.005 - HANDLE_HALF[0], HANDLE_Y, plate_c[2] + 0.015)
    drawer = [((0, 0, -Hd / 2 + tb / 2), (Dd / 2, Wd / 2, tb / 2)),
              ((Dd / 2 - tb / 2, 0, 0), (tb / 2, Wd / 2, Hd / 2)),
              ((0, Wd / 2 - tb / 2, 0), (Dd / 2, tb / 2, Hd / 2)),
              ((0, -Wd / 2 + tb / 2, 0), (Dd / 2, tb / 2, Hd / 2)),
              (plate_c, plate_h)]
    # grasp points 1.5 cm from the free ends (6.5 cm from the side wall, 4.5 cm from the drawer plate): the fingers
    # stay clear of the wall / plate they stick out of
    knob_g = (knob_c[0], knob_c[1] + SIDE_KNOB_HALF[1] - 0.015, knob_c[2])
    handle_g = (handle_c[0] - HANDLE_HALF[0] + 0.015, handle_c[1], handle_c[2])
    return dict(body=body, knob_c=knob_c, knob_h=SIDE_KNOB_HALF, knob_g=knob_g, drawer_origin=(xc, 0.0, zc), drawer=drawer,
                handle_c=handle_c, handle_h=HANDLE_HALF, handle_g=handle_g)


@register_env("DualDrawer-v1", max_episode_steps=3000)
class DualDrawerEnv(DualTabletopEnv):
    def __init__(self, *args, cab_x=(0.36, 0.40), cab_y=(-0.02, 0.02), yaw_range: float = 0.1, density: float = 600.0,
                 drawer_damping: float = 120.0, **kwargs):
        self.cab_x, self.cab_y, self.yaw_range = tuple(cab_x), tuple(cab_y), float(yaw_range)
        self.density, self.drawer_damping = float(density), float(drawer_damping)
        self.geo = geometry()
        super().__init__(*args, **kwargs)

    # ---------------------------------------------------------------- scene
    def _load_scene(self, options: dict):
        b = self.scene.create_actor_builder()
        hx, hy = (D.TABLE_X[1] - D.TABLE_X[0]) / 2, (D.TABLE_Y[1] - D.TABLE_Y[0]) / 2
        cx, cy = (D.TABLE_X[1] + D.TABLE_X[0]) / 2, (D.TABLE_Y[1] + D.TABLE_Y[0]) / 2
        half = [hx, hy, D.TABLE_THICK / 2]
        b.add_box_collision(half_size=half)
        b.add_box_visual(half_size=half, material=sapien.render.RenderMaterial(base_color=[0.78, 0.70, 0.60, 1]))
        b.initial_pose = sapien.Pose([cx, cy, D.TABLE_TOP_Z - D.TABLE_THICK / 2])
        self.table = b.build_kinematic(name="table")
        g = self.geo
        ab = self.scene.create_articulation_builder()
        root = ab.create_link_builder()
        root.set_name("cabinet")
        wood = sapien.render.RenderMaterial(base_color=[0.55, 0.38, 0.22, 1])
        for c, h in g["body"]:
            root.add_box_collision(sapien.Pose(c), h, density=self.density)
            root.add_box_visual(sapien.Pose(c), h, material=wood)
        grip_mat = sapien.pysapien.physx.PhysxMaterial(GRIP_FRICTION, GRIP_FRICTION, 0.0)
        root.add_box_collision(sapien.Pose(g["knob_c"]), g["knob_h"], material=grip_mat, density=self.density)
        root.add_box_visual(sapien.Pose(g["knob_c"]), g["knob_h"],
                            material=sapien.render.RenderMaterial(base_color=[0.15, 0.35, 0.85, 1]))
        dl = ab.create_link_builder(root)
        dl.set_name("drawer")
        flip = _rotz(np.pi)
        dl.set_joint_name("drawer_joint")
        dl.set_joint_properties("prismatic", limits=[[0.0, TRAVEL]], pose_in_parent=sapien.Pose(g["drawer_origin"], flip),
                                pose_in_child=sapien.Pose([0, 0, 0], flip), friction=0.0, damping=0.0)
        light = sapien.render.RenderMaterial(base_color=[0.80, 0.66, 0.45, 1])
        for c, h in g["drawer"]:
            dl.add_box_collision(sapien.Pose(c), h, density=self.density)
            dl.add_box_visual(sapien.Pose(c), h, material=light)
        dl.add_box_collision(sapien.Pose(g["handle_c"]), g["handle_h"], material=grip_mat, density=self.density)
        dl.add_box_visual(sapien.Pose(g["handle_c"]), g["handle_h"],
                          material=sapien.render.RenderMaterial(base_color=[0.85, 0.15, 0.12, 1]))
        ab.initial_pose = sapien.Pose([0.44, 0.0, CAB["H"] / 2])
        self.cabinet = ab.build(name="cabinet_art", fix_root_link=False)
        self.cabinet_link = self.cabinet.links_map["cabinet"]
        self.drawer_link = self.cabinet.links_map["drawer"]
        self.drawer_joint = self.cabinet.joints_map["drawer_joint"]
        self.drawer_joint.set_drive_properties(stiffness=0.0, damping=self.drawer_damping)
        self.drawer_joint.set_drive_target(torch.zeros(1))
        self.drawer_joint.set_drive_velocity_target(torch.zeros(1))
        self.cube = self.cabinet_link                     # generic helpers read env.cube
        tb = self.scene.create_actor_builder()
        tb.add_box_visual(half_size=[0.0005, 0.0005, 0.0005])
        tb.initial_pose = sapien.Pose([0.3, 0.0, -0.01])
        self.target = tb.build_kinematic(name="target")   # unused marker (DualTabletopEnv keeps a target)
        ags = list(self.agent.agents)
        self.group2_remaps = D.make_group2_bits_disjoint(ags)
        for ag, links in zip(ags, self.base_table_links):
            D.filter_base_table(ag, self.table, links)

    @property
    def cube_half(self):
        return np.asarray(SIDE_KNOB_HALF, float)

    @cube_half.setter
    def cube_half(self, v):
        pass

    # ---------------------------------------------------------------- episode
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            for ag, pose in zip(self.agent.agents, self.dual.mount_poses()):
                q0 = torch.as_tensor(ag.home_qpos_vector(), dtype=torch.float32)
                ag.reset(q0.repeat(b, 1))
                ag.robot.set_pose(Pose.create_from_pq(torch.as_tensor(pose.p, dtype=torch.float32).repeat(b, 1),
                                                      torch.as_tensor(pose.q, dtype=torch.float32).repeat(b, 1)))
            u = lambda lo, hi: torch.rand(b) * (hi - lo) + lo  # noqa: E731
            p = torch.zeros((b, 3))
            p[:, 0] = u(*self.cab_x); p[:, 1] = u(*self.cab_y); p[:, 2] = CAB["H"] / 2 + 1e-4
            yaw = u(-self.yaw_range, self.yaw_range)
            q = torch.stack([torch.cos(yaw / 2), torch.zeros(b), torch.zeros(b), torch.sin(yaw / 2)], -1)
            self.cabinet.set_pose(Pose.create_from_pq(p, q))
            self.cabinet.set_qpos(torch.zeros((b, 1)))
            self.cabinet.set_qvel(torch.zeros((b, 1)))
            if not hasattr(self, "cab_p0") or self.cab_p0.shape[0] != self.num_envs:
                self.cab_p0 = torch.zeros((self.num_envs, 3), device=self.device)
                self.target_pos = torch.zeros((self.num_envs, 3), device=self.device)
                self.cube_side = torch.ones(self.num_envs, device=self.device)
            self.cab_p0[env_idx] = p.to(self.device)

    # ---------------------------------------------------------------- task signals
    def drawer_open(self) -> torch.Tensor:
        return self.drawer_joint.qpos.reshape(self.num_envs)

    def cabinet_disp(self) -> torch.Tensor:
        return torch.linalg.norm(self.cabinet_link.pose.p[:, :2] - self.cab_p0[:, :2], dim=-1)

    def evaluate(self) -> dict[str, Any]:
        o = self.drawer_open()
        return dict(success=o >= OPEN_BENCH, drawer_open=o, cabinet_disp=self.cabinet_disp())

    def _get_obs_extra(self, info: dict):
        obs = dict()
        for i, ag in enumerate(self.agent.agents):
            obs[f"tcp_pose_{i}"] = ag.tcp.pose.raw_pose
        if "state" in self.obs_mode:
            obs["cabinet_pose"] = self.cabinet_link.pose.raw_pose
            obs["drawer_qpos"] = self.drawer_joint.qpos.reshape(self.num_envs, 1)
        return obs
