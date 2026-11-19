"""DualClothCoverBox-v1 (E29b): fold the far half of a Newton (VBD) towel over a box that sits on the towel's near half, so
that the box is covered; both arms; any prismatic parallel-gripper dual configuration (route A: SAPIEN / PhysX robots ->
Newton cloth, one-way, per control step; as DualFoldTowel-v1).

  towel     as DualFoldTowel-v1 (30 x 30 cm, 25 x 25 vertices, 0.15 kg, centre towel_center, corners NR / FR / NL / FL)
  box       6 x 6 x 2 cm (box_half), centred at (towel centre x - 4.5 cm, towel centre y), resting on the towel: its
            bottom is box_lift = 8.5 mm above the table (the towel layer is 2 x 4 mm particle radius thick). The box is
            KINEMATIC in both simulators (SAPIEN actor 'box' for the robots, a kinematic box collider in the Newton model
            for the cloth; one-way coupling: the cloth cannot push it). It is not part of any arm's contact group, so the
            release signals (robot_cloth_impulse) count the two arms only; the box-cloth impulse is reported separately.
  task      the left arm grasps the far-left corner FL, the right arm FR (2.5 cm inside the corner); both lift the far
            half over the box and lay it
            down beyond the box's near face (release points fold_points(): 3 cm beyond it), release, retreat. Geometry: the
            far half (15 cm) reaches over the 2 cm box and 1-2 cm onto the table beyond its near face, so the released
            corners rest on the table (a 3 cm box made them lean on its near face and creep, E29b pilot)
  signals   cloth_signals() of DualFoldTowel-v1 plus
              box_cover_frac      fraction of 9 x 9 sample points on the box's top face (inset 3 mm) that have a towel
                                  particle above the top face (particle centre >= top + 2 mm) within 1.2 cm horizontally
              towel_frac_above_8cm  fraction of particles more than 8 cm above the table
              box_cloth_impulse   cloth contact impulse on the box collider (diagnostic)
            robot_cloth_impulse / released are recomputed over the two arms' colliders only.
  success   benchmark-style flag (one step, E20 / E23 cloth style): box_cover_frac >= 0.9, <= 2 % of the towel above 8 cm,
            released (arm contact impulse <= 5e-4 N s and finger enclosure > 2 mm), no NaN. It does not check that the
            corners were lifted and has no settle.
"""
from __future__ import annotations

import numpy as np
import sapien
import torch

from mani_skill.utils.registration import register_env

from ..soft.colliders import Box, Collider
from .dual_fold_towel import ENCLOSURE_MARGIN_M, RELEASE_IMPULSE_NS, DualFoldTowelEnv

COVER_N = 9
COVER_INSET = 0.003
COVER_R_XY = 0.012
COVER_DZ = 0.002


@register_env("DualClothCoverBox-v1", max_episode_steps=800)
class DualClothCoverBoxEnv(DualFoldTowelEnv):
    def __init__(self, *args, box_half=(0.03, 0.03, 0.01), box_offset_x: float = -0.045, box_lift: float = 0.0085,
                 **kwargs):
        self.box_half = np.asarray(box_half, float)
        self.box_offset_x = float(box_offset_x)
        self.box_lift = float(box_lift)
        self.box = None
        super().__init__(*args, **kwargs)

    # ---------------------------------------------------------------- scene
    @property
    def box_center(self) -> np.ndarray:
        cx, cy = self.towel_spec.center
        return np.array([cx + self.box_offset_x, cy, self.box_lift + self.box_half[2]])

    @property
    def box_top_z(self) -> float:
        return float(self.box_lift + 2 * self.box_half[2])

    def _extra_cloth_colliders(self) -> list:
        b = self.scene.create_actor_builder()
        h = self.box_half.tolist()
        b.add_box_collision(half_size=h)
        b.add_box_visual(half_size=h, material=sapien.render.RenderMaterial(base_color=[0.85, 0.45, 0.12, 1]))
        b.initial_pose = sapien.Pose(self.box_center.tolist())
        self.box = b.build_kinematic(name="box")
        return [(Collider("box", [Box(tuple(h))]), self.box)]

    # ---------------------------------------------------------------- task geometry (planner)
    def fold_points(self, i: int, ex: dict, mode: str = "expert"):
        """(grasp xy, release xy) of arm i (0 = left: FL, 1 = right: FR). Release beyond the box's near face (2 cm), at
        |y| = H - inset_release; short_fold releases on the fold line (towel centre x) instead."""
        t = self.towel_spec
        cx, cy = t.center
        H = t.half_size
        sy = 1.0 if i == 0 else -1.0
        gi = ex.get("cover_inset_grasp", ex["inset_grasp"])     # 2.5 cm, as the fold
        A = np.array([cx + H - gi, cy + sy * (H - gi)])
        x_rel = cx if mode == "short_fold" else self.box_center[0] - self.box_half[0] - ex.get("cover_release_gap", 0.03)
        C = np.array([x_rel, cy + sy * (H - ex["inset_release"])])
        return A, C

    def fold_waypoint_overrides(self) -> dict:
        """carry 12 cm high; release with the fingertips 1.2 cm above the grasp height and stay 0.3 s after opening (the
        fold expert's E29b values before its final change; cover pilots 20/20 on each config with these)"""
        return dict(up=0.12, low_rel=0.012, d_post=0.3)

    # ---------------------------------------------------------------- signals
    def _cover_samples(self) -> torch.Tensor:
        c, h = self.box_center, self.box_half
        g = lambda k: np.linspace(c[k] - h[k] + COVER_INSET, c[k] + h[k] - COVER_INSET, COVER_N)  # noqa: E731
        X, Y = np.meshgrid(g(0), g(1))
        return torch.as_tensor(np.stack([X.ravel(), Y.ravel()], 1), dtype=torch.float32, device=self.device)

    def cloth_signals(self) -> dict[str, torch.Tensor]:
        s = super().cloth_signals()
        arms = s["robot_cloth_impulse_0"] + s["robot_cloth_impulse_1"]
        s["box_cloth_impulse"] = (s["robot_cloth_impulse"] - arms).clamp(min=0.0)
        s["robot_cloth_impulse"] = arms
        s["released"] = ((arms <= RELEASE_IMPULSE_NS) & (s["finger_cloth_enclosure"] > ENCLOSURE_MARGIN_M)).float()
        q = self.cloth.particle_q().to(self.device).float()
        S = self._cover_samples()                                               # (S, 2)
        d = torch.cdist(S[None].expand(q.shape[0], -1, -1), q[..., :2])         # (N, S, V)
        above = (q[..., 2] >= self.box_top_z + COVER_DZ)[:, None, :]            # (N, 1, V)
        cov = ((d <= COVER_R_XY) & above).any(-1).float().mean(-1)
        s["box_cover_frac"] = torch.nan_to_num(cov, nan=0.0)
        s["towel_frac_above_8cm"] = torch.nan_to_num((q[..., 2] > 0.08).float().mean(-1), nan=1.0)
        return s

    def evaluate(self) -> dict:
        if self.cloth is None:
            return {}
        s = self.cloth_signals()
        covered = (s["box_cover_frac"] >= 0.9) & (s["towel_frac_above_8cm"] <= 0.02)
        success = covered & (s["released"] > 0.5) & (s["cloth_nonfinite"] == 0)
        return dict(success=success, covered=covered, fail=s["cloth_nonfinite"] > 0)

    def compute_dense_reward(self, obs, action, info):
        s = self.cloth_signals()
        return s["box_cover_frac"] + info["success"].float()

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 2.0
