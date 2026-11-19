"""v3.2 recorders for the E29b fork-B2 dual-arm tasks (DualPegInHole-v1, DualDrawer-v1): the generic v3.2 signals of
imitator.verify.v32.build_recorder plus the task geometry each contract reads (env 0, CPU PhysX).

DualPegInHole-v1 (objects: block = the holed block incl. its handle, peg)
  peg_radial_m          distance of the peg bottom centre from the hole axis (block frame)
  peg_height_m          height of the peg bottom centre above the hole floor (block frame)
  peg_hole_tilt_deg     angle between the peg's long axis and the hole axis
  pairs                 peg_block (support), arm0_block / arm1_block (who holds the block), table_block, table_peg
  aftermath             only the peg (the left arm keeps holding the block; arms near the PEG retreat)
DualDrawer-v1 (objects: handle = the drawer link's handle box, knob = the cabinet link's top-knob box)
  drawer_open_m         drawer joint position (0 closed .. 0.12 m)
  cabinet_disp_m        horizontal displacement of the cabinet origin from its reset position
  cabinet_bottom_dz_m   lowest corner of the cabinet body box above the table
  knob_tilt_deg         tilt of the cabinet (its z axis to vertical)
  pairs                 robot_handle / robot_knob = robot vs drawer link / cabinet link (release); table_knob (support)
"""
from __future__ import annotations

import numpy as np

from .v32 import Obj, build_recorder


def _pair(a, b):
    from veritask.adapters.maniskill_v3 import PairV3
    return PairV3(a, b)


# ------------------------------------------------------------------------------------------------ DualPegInHole-v1
def peg_in_hole(env):
    from ..envs.dual_peg_in_hole import HANDLE_C, HANDLE_HALF, HOLE_OUT, PEG_HALF
    y0, y1 = -HOLE_OUT[1], HANDLE_C[1] + HANDLE_HALF[1]
    objs = {"block": Obj("block", half=(HOLE_OUT[0], (y1 - y0) / 2, HOLE_OUT[2]), center=(0.0, (y1 + y0) / 2, 0.0),
                         tilt_axis=2),
            "peg": Obj("peg", half=PEG_HALF, tilt_axis=2)}

    def tilt(u):
        a = u.peg.pose.to_transformation_matrix()[0, :3, 2].cpu().numpy()
        n = u.block.pose.to_transformation_matrix()[0, :3, 2].cpu().numpy()
        return float(np.degrees(np.arccos(np.clip(abs(float(a @ n)), 0.0, 1.0))))
    probes = {"peg_radial_m": lambda u: float(u.peg_in_block()[0][0]),
              "peg_height_m": lambda u: float(u.peg_in_block()[1][0]),
              "peg_hole_tilt_deg": tilt}
    pairs = {"peg_block": _pair(lambda u: [u.peg], lambda u: [u.block])}
    return build_recorder(env, objs, release=["peg"], extra_probes=probes, extra_pairs=pairs, per_arm_pairs=["block"],
                          aftermath=["peg"])


# ------------------------------------------------------------------------------------------------ DualDrawer-v1
def drawer(env):
    from ..envs.dual_drawer import CAB
    g = env.unwrapped.geo
    objs = {"handle": Obj("drawer_link", half=g["handle_h"], center=g["handle_c"], tilt_axis=2),
            "knob": Obj("cabinet_link", half=g["knob_h"], center=g["knob_c"], tilt_axis=2)}
    body_h = np.array([CAB["D"] / 2, CAB["W"] / 2, CAB["H"] / 2])
    corners = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float) * body_h

    def bottom(u):
        T = u.cabinet_link.pose.to_transformation_matrix()[0].cpu().numpy().astype(float)
        return float((corners @ T[:3, :3].T + T[:3, 3])[:, 2].min())
    probes = {"drawer_open_m": lambda u: float(u.drawer_open()[0]),
              "cabinet_disp_m": lambda u: float(u.cabinet_disp()[0]),
              "cabinet_bottom_dz_m": bottom}
    return build_recorder(env, objs, release=["handle", "knob"], extra_probes=probes)
