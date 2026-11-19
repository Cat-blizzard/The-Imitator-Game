"""Contact-wrench sources for the tactile pads, accumulated over the physics substeps of one control step.

Two measurements, both taken after EVERY physics substep (hook on BaseEnv._after_simulation_step, chained with any
recorder hook already installed):

1. Per (pad, contacting body):
   CPU PhysX  contact points (scene.px.get_contacts()): position, normal impulse. A point counts for a pad when it lies
              on the pad's footprint (pad frame |x| <= W/2 + m, |y| <= L/2 + m, z within [h_min - 3 mm, 3 mm]) and pushes
              into the pad; accumulated: normal impulse Jn, first moments Jn*x, Jn*y (=> per-body force and CoP), impulse
              vector. Points elsewhere on the link are accumulated as "off-pad" impulse (diagnostic; real pads do not
              sense them either).
   GPU PhysX  one contact-pair impulse query over (pad link, candidate body, env) for all pads / candidates / envs:
              net impulse vector per pair (per-body force; no positions on GPU).
   PhysX reports only NORMAL impulses at contact points (SAPIEN 3.0.3; measured: tangential part exactly 0 while friction
   holds a load), so neither gives friction.

2. Per pad link: the incoming joint force (spatial force from the parent, in the child joint frame) of the link and of
   its child links, CPU (articulation.get_link_incoming_joint_forces) or GPU (cuda_articulation_link_incoming_joint_forces).
   Newton-Euler on the link over the control step gives the TOTAL contact wrench on the link, friction and moments
   included:
       F_c = m (dv / T - g) - J_in / T + J_children / T
       tau_c (about the link origin) = (I dw + c x m dv) / T - c x m g - Jt_in / T + Jt_children / T
   (link frame; dv, dw = change of COM linear / angular velocity over the step; gyroscopic term and the rotation of the link
   during the step neglected). Measured on a two-finger pinch (CPU): normal 13.9118 N vs 13.9117 N from the points, and the
   tangential parts add up to the held weight.

Sign / frame conventions: everything is the force ON the pad link; pad frame = (x across, y along the finger, n out of
the surface); "normal" = compressive = -F . n.
"""
from __future__ import annotations

import numpy as np
import torch


# ------------------------------------------------------------------------------------------------ small math (wxyz)
def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


def mat_to_quat_np(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(np.asarray(R, float)).as_quat()
    return np.array([w, x, y, z])


def pose_np(p) -> np.ndarray:
    """sapien.Pose -> 4x4"""
    return np.asarray(p.to_transformation_matrix(), float)


def _comp(x, e: int = 0):
    """SAPIEN physx component of a ManiSkill Actor / Link for env e"""
    return x._bodies[e]


def _entity(x, e: int = 0):
    return x._bodies[e].entity


# ------------------------------------------------------------------------------------------------ link dynamics
class LinkInfo:
    """constant per-pad-link data (from env 0; all envs share the robot model)"""

    def __init__(self, link, pad_R: np.ndarray, pad_o: np.ndarray):
        comp = _comp(link)
        self.link = link
        self.art = link.articulation
        self.index = int(comp.index)
        J = comp.joint
        T_LJ = pose_np(J.get_pose_in_child())                  # joint frame in the link frame
        self.R_LJ, self.p_LJ = T_LJ[:3, :3], T_LJ[:3, 3]
        self.mass = float(comp.mass)
        Tc = pose_np(comp.cmass_local_pose)
        self.com = Tc[:3, 3]
        self.I = Tc[:3, :3] @ np.diag(np.asarray(comp.inertia, float)) @ Tc[:3, :3].T
        # child links: transform of each child's joint frame into this link's frame (fixed joints exact; movable joints
        # use q = 0 here and are refreshed from live poses in end_step)
        self.children = []
        for ch in comp.children:
            cj = ch.joint
            T_L_ch0 = pose_np(cj.get_pose_in_parent()) @ np.linalg.inv(pose_np(cj.get_pose_in_child()))
            T_ch_J = pose_np(cj.get_pose_in_child())
            self.children.append(dict(index=int(ch.index), fixed=(cj.type == "fixed"), T_ch_J=T_ch_J, T_L_ch0=T_L_ch0,
                                      name=ch.name))
        self.pad_R, self.pad_o = np.asarray(pad_R, float), np.asarray(pad_o, float)


def joint_to_link(F6: torch.Tensor, R: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """spatial force (..., 6) [f, tau about the joint origin] in a joint frame -> link frame, tau about the link origin.
    R (..., 3, 3) joint->link rotation, p (..., 3) joint origin in the link frame."""
    f = torch.einsum("...ij,...j->...i", R, F6[..., :3])
    t = torch.einsum("...ij,...j->...i", R, F6[..., 3:]) + torch.linalg.cross(p.expand_as(f), f, dim=-1)
    return torch.cat([f, t], -1)
