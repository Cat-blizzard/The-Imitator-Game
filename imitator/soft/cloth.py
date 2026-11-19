"""Newton 1.7 (VBD) towel for N parallel worlds, driven by kinematic robot colliders.

Coupling model (both routes use this class):
  robot -> cloth   every control step the caller passes the poses of a few robot links (hand, fingers, ...); the cloth
                   solver treats them as kinematic bodies (VBD `integrate_with_external_rigid_solver=True`): poses are
                   interpolated over the substeps (linear position, slerp rotation), velocities by finite difference, so
                   contact friction sees the true link motion.
  cloth -> robot   NOT fed back. The cloth's contact wrench on every link is measured (VBD CONTACT_F observable) and
                   reported (`robot_cloth_impulse`, peak force), so the size of the neglected reaction is known per step.
Grasping:
  'friction'       the cloth is held only by contact friction between the finger pads and the particles (the pads must
                   close to a width below the cloth thickness 2*radius; the gripper close target is clamped by the caller).
  'attach'         documented attachment rule: when the caller reports a closed gripper (`grip` flag 0->1), particles
                   whose centres lie between the two pads (pad box grown by radius) become kinematic and follow the
                   midpoint frame of the pads rigidly; when the flag goes 1->0 they are released (mass restored).

Units SI, z up, every world in its own local frame (worlds are separated by Newton's world index, not by space).
Vertex layout of the towel (same as research/protocols/towel_fold_kinematic_v1.json): idx = iy*(res+1)+ix, vertex 0 =
corner A at (cx-H, cy-H), vertex (res+1)^2-1 = corner C at (cx+H, cy+H).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import warp as wp

import newton

from .colliders import Collider, PadPair

wp.config.quiet = True


@dataclass
class TowelSpec:
    half_size: float = 0.15
    res: int = 24                    # cells per side -> (res+1)^2 vertices
    mass: float = 0.15               # kg, whole towel
    particle_radius: float = 0.004
    tri_ke: float = 5e2
    tri_ka: float = 5e2
    tri_kd: float = 1e-1
    edge_ke: float = 1e-3
    edge_kd: float = 1e-3
    center: tuple[float, float] = (0.0, 0.0)
    z0: float = 0.004                # initial particle-centre height above the table
    table_z: float = 0.0

    @property
    def n_vertices(self) -> int:
        return (self.res + 1) ** 2

    @property
    def corner_a(self) -> int:
        return 0

    @property
    def corner_c(self) -> int:
        return self.n_vertices - 1


@dataclass
class ClothSolverSpec:
    substeps: int = 30               # VBD substeps per control step (control 20 Hz -> dt = 1/600 s)
    iterations: int = 5
    self_contact: bool = True
    self_contact_radius: float = 0.002
    self_contact_margin: float = 0.003
    soft_contact_ke: float = 1e3
    soft_contact_kd: float = 1e1
    body_mu: float = 1.5             # finger-cloth friction (the earlier towel probes used 1.5)
    ground_mu: float = 0.3
    soft_contact_gap: float = 0.01
    grasp_mode: str = "friction"     # 'friction' | 'attach'
    use_graph: bool = True
    contact_forces: bool = True      # export VBD contact wrenches (robot_cloth_impulse); small extra cost
    self_contact_detection: str = "auto"   # VBD self-contact detection schedule: auto (=PRE_POST_INIT) | PRE_INIT
    full_surface_contact: bool = True      # rigid-soft contact on triangles/edges too, not only particle spheres: stops
                                           # finger edges from slipping between particles and hooking the sheet


# ------------------------------------------------------------------------------------------------------- kernels
@wp.func
def _quat_wxyz(a: wp.array2d[float], i: int) -> wp.quat:
    return wp.normalize(wp.quat(a[i, 4], a[i, 5], a[i, 6], a[i, 3]))


@wp.kernel
def _set_body_poses(prev: wp.array2d[float], target: wp.array2d[float], a0: float, a1: float, inv_dt: float,
                    body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector]):
    i = wp.tid()
    p0 = wp.vec3(prev[i, 0], prev[i, 1], prev[i, 2])
    p1 = wp.vec3(target[i, 0], target[i, 1], target[i, 2])
    q0 = _quat_wxyz(prev, i)
    q1 = _quat_wxyz(target, i)
    if wp.dot(q0, q1) < 0.0:
        q1 = -q1
    pa = p0 + (p1 - p0) * a0
    pb = p0 + (p1 - p0) * a1
    qa = wp.quat_slerp(q0, q1, a0)
    qb = wp.quat_slerp(q0, q1, a1)
    dq = qb * wp.quat_inverse(qa)
    s = 1.0
    if dq[3] < 0.0:
        s = -1.0
    w = wp.vec3(dq[0], dq[1], dq[2]) * (2.0 * s * inv_dt)
    body_q[i] = wp.transform(pb, qb)
    body_qd[i] = wp.spatial_vector((pb - pa) * inv_dt, w)


@wp.kernel
def _accum_contact(count: wp.array[int], particle: wp.array[int], shape: wp.array[int], shape_body: wp.array[int],
                   f: wp.array[wp.spatial_vector], offset: int, ppw: int, dt: float,
                   impulse: wp.array[float], fmax: wp.array[float]):
    c = wp.tid()
    if c >= count[0]:
        return
    s = shape[c]
    if s < 0:
        return
    if shape_body[s] < 0:
        return
    p = particle[c]
    if p < 0:
        return
    w = p // ppw
    fl = wp.length(wp.spatial_top(f[offset + c]))
    wp.atomic_add(impulse, w, fl * dt)
    wp.atomic_max(fmax, w, fl)


@wp.kernel
def _accum_contact_grouped(count: wp.array[int], particle: wp.array[int], shape: wp.array[int], shape_body: wp.array[int],
                           f: wp.array[wp.spatial_vector], offset: int, ppw: int, bpw: int, dt: float,
                           body_group: wp.array[int], n_groups: int, impulse: wp.array[float]):
    """per (world, collider group) contact impulse, e.g. one group per arm (impulse index = world * n_groups + group)"""
    c = wp.tid()
    if c >= count[0]:
        return
    s = shape[c]
    if s < 0:
        return
    b = shape_body[s]
    if b < 0:
        return
    p = particle[c]
    if p < 0:
        return
    w = p // ppw
    g = body_group[b % bpw]
    if g < 0:
        return
    fl = wp.length(wp.spatial_top(f[offset + c]))
    wp.atomic_add(impulse, w * n_groups + g, fl * dt)


@wp.func
def _in_pad_frame(x: wp.vec3, body: wp.transform, pad_pos: wp.vec3, pad_rot: wp.quat) -> wp.vec3:
    xl = wp.transform_point(wp.transform_inverse(body), x)
    return wp.quat_rotate_inv(pad_rot, xl - pad_pos)


@wp.func
def _lateral_inside(y: wp.vec3, h: wp.vec3, axis: int, radius: float) -> bool:
    """True if |y_k| <= h_k + radius on the two axes other than the closing axis."""
    e0 = wp.abs(y[0]) - h[0] - radius
    e1 = wp.abs(y[1]) - h[1] - radius
    e2 = wp.abs(y[2]) - h[2] - radius
    if axis == 0:
        e0 = -1.0
    if axis == 1:
        e1 = -1.0
    if axis == 2:
        e2 = -1.0
    return wp.max(e0, wp.max(e1, e2)) <= 0.0


@wp.func
def _box_sd(y: wp.vec3, h: wp.vec3) -> float:
    """signed distance from a point (box frame) to an axis-aligned box of half extents h"""
    qx = wp.abs(y[0]) - h[0]
    qy = wp.abs(y[1]) - h[1]
    qz = wp.abs(y[2]) - h[2]
    return wp.min(wp.max(qx, wp.max(qy, qz)), 0.0) + wp.length(wp.vec3(wp.max(qx, 0.0), wp.max(qy, 0.0), wp.max(qz, 0.0)))


@wp.kernel
def _world_signals(q: wp.array[wp.vec3], qd: wp.array[wp.vec3], ppw: int, table_z: float, height: float,
                   body_q: wp.array[wp.transform], bpw: int, left_b: int, right_b: int,
                   lpad_pos: wp.vec3, lpad_rot: wp.quat, lpad_half: wp.vec3, laxis: int, lsign: float,
                   rpad_pos: wp.vec3, rpad_rot: wp.quat, rpad_half: wp.vec3, raxis: int, rsign: float,
                   radius: float,
                   n_above: wp.array[int], n_bad: wp.array[int], vmax: wp.array[float], zmin: wp.array[float],
                   clear_l: wp.array[float], clear_r: wp.array[float], n_enclosed: wp.array[int],
                   pad_pen: wp.array[float]):
    i = wp.tid()
    w = i // ppw
    x = q[i]
    v = qd[i]
    if not (wp.isfinite(x) and wp.isfinite(v)):
        wp.atomic_add(n_bad, w, 1)
        return
    if x[2] - table_z > height:
        wp.atomic_add(n_above, w, 1)
    wp.atomic_max(vmax, w, wp.length(v))
    wp.atomic_min(zmin, w, x[2] - table_z)
    if left_b < 0:
        return
    bl = body_q[w * bpw + left_b]
    br = body_q[w * bpw + right_b]
    yl = _in_pad_frame(x, bl, lpad_pos, lpad_rot)
    yr = _in_pad_frame(x, br, rpad_pos, rpad_rot)
    # signed distance from each pad's inner face, positive towards the other finger
    dl = lsign * yl[laxis] - lpad_half[laxis]
    dr = rsign * yr[raxis] - rpad_half[raxis]
    # lateral extent test in the left pad frame (the two axes other than the closing axis)
    inside = _lateral_inside(yl, lpad_half, laxis, radius)
    if inside and dl > -radius and dr > -radius:
        wp.atomic_add(n_enclosed, w, 1)
        wp.atomic_min(clear_l, w, dl - radius)
        wp.atomic_min(clear_r, w, dr - radius)
    # penetration of the particle sphere into either pad box (beyond the contact radius)
    sd = wp.min(_box_sd(yl, lpad_half), _box_sd(yr, rpad_half))
    if sd < radius:
        wp.atomic_max(pad_pen, w, radius - sd)


@wp.kernel
def _attach_update(q: wp.array[wp.vec3], ppw: int, grip: wp.array[int], grip_prev: wp.array[int],
                   body_q: wp.array[wp.transform], bpw: int, left_b: int, right_b: int,
                   lpad_pos: wp.vec3, lpad_rot: wp.quat, lpad_half: wp.vec3, laxis: int, lsign: float,
                   rpad_pos: wp.vec3, rpad_rot: wp.quat, rpad_half: wp.vec3, raxis: int, rsign: float, radius: float,
                   mass0: wp.array[float], mass: wp.array[float], inv_mass: wp.array[float],
                   attached: wp.array[int], offset: wp.array[wp.vec3]):
    """grip 0->1: attach particles between the pads (offset in the left-finger frame); 1->0: release."""
    i = wp.tid()
    w = i // ppw
    g = grip[w]
    gp = grip_prev[w]
    if g == 1 and gp == 0:
        x = q[i]
        bl = body_q[w * bpw + left_b]
        br = body_q[w * bpw + right_b]
        yl = _in_pad_frame(x, bl, lpad_pos, lpad_rot)
        yr = _in_pad_frame(x, br, rpad_pos, rpad_rot)
        dl = lsign * yl[laxis] - lpad_half[laxis]
        dr = rsign * yr[raxis] - rpad_half[raxis]
        inside = _lateral_inside(yl, lpad_half, laxis, radius)
        if inside and dl > -radius and dr > -radius:
            attached[i] = 1
            offset[i] = wp.transform_point(wp.transform_inverse(bl), x)
            mass[i] = 0.0
            inv_mass[i] = 0.0
    if g == 0 and gp == 1:
        if attached[i] == 1:
            attached[i] = 0
            mass[i] = mass0[i]
            inv_mass[i] = 1.0 / mass0[i]


@wp.kernel
def _attach_drive(q: wp.array[wp.vec3], qd: wp.array[wp.vec3], ppw: int, body_q: wp.array[wp.transform],
                  body_qd: wp.array[wp.spatial_vector], bpw: int, left_b: int, attached: wp.array[int],
                  offset: wp.array[wp.vec3]):
    i = wp.tid()
    if attached[i] == 0:
        return
    w = i // ppw
    b = body_q[w * bpw + left_b]
    x = wp.transform_point(b, offset[i])
    v = body_qd[w * bpw + left_b]
    q[i] = x
    qd[i] = wp.spatial_top(v) + wp.cross(wp.spatial_bottom(v), x - wp.transform_get_translation(b))


@wp.kernel
def _corner_dist(q: wp.array[wp.vec3], ppw: int, a: int, c: int, out: wp.array[float]):
    w = wp.tid()
    d = q[w * ppw + a] - q[w * ppw + c]
    out[w] = wp.sqrt(d[0] * d[0] + d[1] * d[1])


@wp.kernel
def _reset_particles(q: wp.array[wp.vec3], qd: wp.array[wp.vec3], q0: wp.array[wp.vec3], ppw: int,
                     mask: wp.array[int], mass0: wp.array[float], mass: wp.array[float], inv_mass: wp.array[float],
                     attached: wp.array[int]):
    i = wp.tid()
    w = i // ppw
    if mask[w] == 0:
        return
    q[i] = q0[i]
    qd[i] = wp.vec3(0.0, 0.0, 0.0)
    attached[i] = 0
    mass[i] = mass0[i]
    inv_mass[i] = 1.0 / mass0[i]


def _mat_to_quat_xyzw(R):
    w = math.sqrt(max(1e-12, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    return ((R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w), w)


def _wq(quat_wxyz) -> wp.quat:
    w, x, y, z = quat_wxyz
    return wp.quat(x, y, z, w)


# ------------------------------------------------------------------------------------------------------- class
class NewtonClothWorlds:
    """N towels (one per world) + kinematic colliders, stepped once per control step."""

    def __init__(self, num_worlds: int, colliders: list[Collider], pads: PadPair | list | None = None,
                 towel: TowelSpec | None = None, solver: ClothSolverSpec | None = None, control_dt: float = 0.05,
                 device: str = "cuda:0", extra_builder=None, collider_groups: list[int] | None = None):
        """pads: one PadPair (E20, single gripper) or a list (one per gripper; the first one also drives the attachment
        rule). collider_groups: group index per collider (e.g. 0 = left arm, 1 = right arm, -1 = none) for the per-group
        contact impulse signals robot_cloth_impulse_<g>."""
        self.n = int(num_worlds)
        self.towel = towel or TowelSpec()
        self.cfg = solver or ClothSolverSpec()
        self.colliders = colliders
        self.pad_list = [] if pads is None else ([pads] if isinstance(pads, PadPair) else list(pads))
        self.pads = self.pad_list[0] if self.pad_list else None
        self.collider_groups = list(collider_groups) if collider_groups is not None else None
        self.n_groups = (max(self.collider_groups) + 1) if self.collider_groups else 0
        self.control_dt = float(control_dt)
        self.device = wp.get_device(device)
        t, c = self.towel, self.cfg
        self.ppw = t.n_vertices
        self.bpw = len(colliders)
        names = [col.link for col in colliders]
        self.link_names = names

        tmpl = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        tmpl.add_cloth_grid(pos=wp.vec3(t.center[0] - t.half_size, t.center[1] - t.half_size, t.table_z + t.z0),
                            rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0), dim_x=t.res, dim_y=t.res,
                            cell_x=2 * t.half_size / t.res, cell_y=2 * t.half_size / t.res,
                            mass=t.mass / t.n_vertices, tri_ke=t.tri_ke, tri_ka=t.tri_ka, tri_kd=t.tri_kd,
                            edge_ke=t.edge_ke, edge_kd=t.edge_kd, particle_radius=t.particle_radius)
        scfg = newton.ModelBuilder.ShapeConfig(mu=c.body_mu, has_shape_collision=False, has_particle_collision=True,
                                               density=0.0)
        hidden = newton.ModelBuilder.ShapeConfig(mu=c.body_mu, has_shape_collision=False, has_particle_collision=True,
                                                 density=0.0, is_visible=False)
        vcfg = newton.ModelBuilder.ShapeConfig(has_shape_collision=False, has_particle_collision=False, density=0.0,
                                               is_visible=True)
        for col in colliders:
            b = tmpl.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0 + 0.3 * len(tmpl.body_q)), wp.quat_identity()),
                              mass=0.0, is_kinematic=True, label=col.link)
            for bx in col.boxes:   # collision boxes are hidden from the camera when the link has visual meshes
                tmpl.add_shape_box(b, xform=wp.transform(wp.vec3(*bx.pos), _wq(bx.quat_wxyz)), hx=bx.half[0],
                                   hy=bx.half[1], hz=bx.half[2], cfg=hidden if col.visual else scfg)
            for (V, F, T) in col.visual:
                R = np.asarray(T)[:3, :3]
                qw = _mat_to_quat_xyzw(R)
                mesh = newton.Mesh(np.asarray(V, np.float32), np.asarray(F, np.int32).reshape(-1), compute_inertia=False)
                tmpl.add_shape_mesh(b, xform=wp.transform(wp.vec3(*np.asarray(T)[:3, 3]), wp.quat(*qw)), mesh=mesh,
                                    cfg=vcfg, color=(0.92, 0.92, 0.92) if "finger" not in col.link else (0.25, 0.25, 0.25))
        if extra_builder is not None:          # e.g. visual-only shapes for the camera
            extra_builder(tmpl)
        tmpl.color()                                   # colour one towel; the N copies reuse it (graph colouring of
        groups = [np.asarray(g, int) for g in tmpl.particle_color_groups]   # all N towels at once is slow on the CPU)
        scene = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        scene.replicate(tmpl, world_count=self.n)
        self.ground_shape = scene.add_ground_plane(height=t.table_z, cfg=newton.ModelBuilder.ShapeConfig(mu=c.ground_mu))
        offs = np.arange(self.n)[:, None] * self.ppw
        scene.set_coloring([(g[None, :] + offs).reshape(-1) for g in groups])
        with wp.ScopedDevice(self.device):
            self.model = m = scene.finalize()
        m.soft_contact_ke, m.soft_contact_kd, m.soft_contact_mu = c.soft_contact_ke, c.soft_contact_kd, c.body_mu
        assert m.particle_count == self.n * self.ppw and m.body_count == self.n * self.bpw
        q0 = m.particle_q.numpy()[: self.ppw]
        H = t.half_size
        assert np.allclose(q0[0, :2], [t.center[0] - H, t.center[1] - H], atol=1e-6)
        assert np.allclose(q0[-1, :2], [t.center[0] + H, t.center[1] + H], atol=1e-6)
        self.tri = m.tri_indices.numpy()[: 2 * t.res * t.res].copy()

        kw = {}
        if c.self_contact and c.self_contact_detection.upper() != "AUTO":
            SB = newton.solvers.SolverBase
            kw["collision_frequency_type"] = {SB.CollisionSlot.SOFT_SELF_CONTACT:
                                              getattr(SB.CollisionFrequencyType, c.self_contact_detection.upper())}
        self.solver = newton.solvers.SolverVBD(
            m, iterations=c.iterations, integrate_with_external_rigid_solver=True,
            particle_enable_self_contact=c.self_contact, particle_self_contact_radius=c.self_contact_radius,
            particle_self_contact_margin=c.self_contact_margin, **kw)
        self.coll = newton.CollisionPipeline(m, soft_contact_gap=c.soft_contact_gap,
                                             enable_rigid_soft_full_surface_contact=c.full_surface_contact)
        self.contacts = self.coll.contacts()
        self.s0, self.s1, self.ctrl = m.state(), m.state(), m.control()
        self.obs = None
        if c.contact_forces:
            try:
                self.obs = self.solver.observables({newton.solvers.SolverBase.ObservableKind.CONTACT_F},
                                                   contacts=self.contacts)
            except Exception as e:  # pragma: no cover - API drift
                print(f"[imitator.soft] contact-force observable unavailable ({e}); robot_cloth_impulse = 0")
                self.obs = None

        dev = self.device
        nb = self.n * self.bpw
        self.prev = wp.zeros((nb, 7), dtype=float, device=dev)
        self.target = wp.zeros((nb, 7), dtype=float, device=dev)
        self.q_init = wp.clone(m.particle_q)
        self.mass0 = wp.clone(m.particle_mass)
        self.attached = wp.zeros(m.particle_count, dtype=int, device=dev)
        self.attach_off = wp.zeros(m.particle_count, dtype=wp.vec3, device=dev)
        self.grip = wp.zeros(self.n, dtype=int, device=dev)
        self.grip_prev = wp.zeros(self.n, dtype=int, device=dev)
        # per-world signals
        self.impulse = wp.zeros(self.n, dtype=float, device=dev)
        self.fmax = wp.zeros(self.n, dtype=float, device=dev)
        self.n_above = wp.zeros(self.n, dtype=int, device=dev)
        self.n_bad = wp.zeros(self.n, dtype=int, device=dev)
        self.vmax = wp.zeros(self.n, dtype=float, device=dev)
        self.zmin = wp.zeros(self.n, dtype=float, device=dev)
        self.clear_l = wp.zeros(self.n, dtype=float, device=dev)
        self.clear_r = wp.zeros(self.n, dtype=float, device=dev)
        self.n_enc = wp.zeros(self.n, dtype=int, device=dev)
        self.pad_pen = wp.zeros(self.n, dtype=float, device=dev)
        self.cdist = wp.zeros(self.n, dtype=float, device=dev)
        self._pad_args = self._make_pad_args()
        # extra pad pairs (dual-arm): own enclosure / penetration outputs; scratch arrays for the per-particle stats
        self._extra_pads = []
        for pp in self.pad_list[1:]:
            self._extra_pads.append(dict(args=self._make_pad_args(pp),
                                         clear_l=wp.zeros(self.n, dtype=float, device=dev),
                                         clear_r=wp.zeros(self.n, dtype=float, device=dev),
                                         n_enc=wp.zeros(self.n, dtype=int, device=dev),
                                         pad_pen=wp.zeros(self.n, dtype=float, device=dev)))
        self._scratch = [wp.zeros(self.n, dtype=t, device=dev) for t in (int, int, float, float)]
        if self.n_groups:
            self.body_group = wp.array(np.asarray(self.collider_groups, np.int32), dtype=int, device=dev)
            self.impulse_g = wp.zeros(self.n * self.n_groups, dtype=float, device=dev)
        self.graph = None
        self.sim_time = 0.0
        self._sub_dt = self.control_dt / c.substeps
        self.steps = 0

    # ------------------------------------------------------------------ helpers
    def _make_pad_args(self, pair: PadPair | None = None):
        p = pair if pair is not None else self.pads
        if p is None:
            z = wp.vec3(0.0, 0.0, 0.0)
            return [-1, -1, z, wp.quat_identity(), z, 1, 1.0, z, wp.quat_identity(), z, 1, 1.0]
        li, ri = self.link_names.index(p.left), self.link_names.index(p.right)
        return [li, ri, wp.vec3(*p.left_pad.pos), _wq(p.left_pad.quat_wxyz), wp.vec3(*p.left_pad.half),
                int(p.left_inner_axis[0]), float(p.left_inner_axis[1]),
                wp.vec3(*p.right_pad.pos), _wq(p.right_pad.quat_wxyz), wp.vec3(*p.right_pad.half),
                int(p.right_inner_axis[0]), float(p.right_inner_axis[1])]

    @staticmethod
    def _as_wp2d(x, dev):
        if isinstance(x, wp.array):
            return x
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return wp.from_torch(x.detach().to(dtype=torch.float32).contiguous().reshape(-1, 7), dtype=wp.float32)
        except ImportError:  # pragma: no cover
            pass
        return wp.array(np.asarray(x, np.float32).reshape(-1, 7), dtype=wp.float32, device=dev)

    # ------------------------------------------------------------------ public API
    def set_link_poses(self, poses, first: bool = False):
        """poses: (N, K, 7) [x y z qw qx qy qz] of the collider links in each world's frame (torch / numpy / warp).
        first=True (after reset) also sets the previous pose, so the next step does not sweep from a stale pose."""
        src = self._as_wp2d(poses, self.device)
        wp.copy(self.target, src)
        if first:
            wp.copy(self.prev, src)
            wp.launch(_set_body_poses, dim=self.n * self.bpw, inputs=[self.prev, self.prev, 1.0, 1.0, 0.0],
                      outputs=[self.s0.body_q, self.s0.body_qd], device=self.device)
            wp.copy(self.s1.body_q, self.s0.body_q)

    def set_grip(self, grip):
        """Attachment rule input (grasp_mode='attach'): per-world 0/1 'gripper closed' flags."""
        g = np.asarray(grip.detach().cpu().numpy() if hasattr(grip, "detach") else grip, dtype=np.int32).reshape(-1)
        self.grip.assign(g)

    def _substeps(self):
        c, dev = self.cfg, self.device
        S = c.substeps
        n_all = self.model.particle_count
        self.impulse.zero_()
        self.fmax.zero_()
        if self.n_groups:
            self.impulse_g.zero_()
        if c.grasp_mode == "attach" and self.pads is not None:
            wp.launch(_attach_update, dim=n_all, inputs=[self.s0.particle_q, self.ppw, self.grip, self.grip_prev,
                                                          self.s0.body_q, self.bpw, *self._pad_args,
                                                          self.towel.particle_radius, self.mass0],
                      outputs=[self.model.particle_mass, self.model.particle_inv_mass, self.attached, self.attach_off],
                      device=dev)
            wp.copy(self.grip_prev, self.grip)
        self.solver.rebuild_bvh(self.s0)
        for k in range(S):
            wp.launch(_set_body_poses, dim=self.n * self.bpw,
                      inputs=[self.prev, self.target, k / S, (k + 1) / S, 1.0 / self._sub_dt],
                      outputs=[self.s1.body_q, self.s1.body_qd], device=dev)
            if c.grasp_mode == "attach" and self.pads is not None:
                li = self._pad_args[0]
                wp.launch(_attach_drive, dim=n_all, inputs=[self.s0.particle_q, self.s0.particle_qd, self.ppw,
                                                             self.s0.body_q, self.s1.body_qd, self.bpw, li,
                                                             self.attached, self.attach_off], device=dev)
            self.s0.clear_forces()
            self.coll.collide(self.s0, self.contacts)
            self.solver.step(self.s0, self.s1, self.ctrl, self.contacts, self._sub_dt, observables=self.obs)
            if self.obs is not None:
                wp.launch(_accum_contact, dim=self.contacts.soft_contact_max,
                          inputs=[self.contacts.soft_contact_count, self.contacts.soft_contact_particle,
                                  self.contacts.soft_contact_shape, self.model.shape_body, self.obs.contact_f,
                                  self.contacts.rigid_contact_max, self.ppw, self._sub_dt],
                          outputs=[self.impulse, self.fmax], device=dev)
                if self.n_groups:
                    wp.launch(_accum_contact_grouped, dim=self.contacts.soft_contact_max,
                              inputs=[self.contacts.soft_contact_count, self.contacts.soft_contact_particle,
                                      self.contacts.soft_contact_shape, self.model.shape_body, self.obs.contact_f,
                                      self.contacts.rigid_contact_max, self.ppw, self.bpw, self._sub_dt,
                                      self.body_group, self.n_groups],
                              outputs=[self.impulse_g], device=dev)
            self.s0, self.s1 = self.s1, self.s0
        if S % 2 == 1:   # keep s0 the 'current' state object across graph replays (state objects swap each substep)
            wp.copy(self.s1.particle_q, self.s0.particle_q); wp.copy(self.s1.particle_qd, self.s0.particle_qd)
            wp.copy(self.s1.body_q, self.s0.body_q); wp.copy(self.s1.body_qd, self.s0.body_qd)
            self.s0, self.s1 = self.s1, self.s0
        wp.copy(self.prev, self.target)

    def step(self):
        """Advance one control step (cfg.substeps VBD substeps), then compute the per-world signals."""
        if self.cfg.use_graph and self.device.is_cuda:
            if self.graph is None:
                self._substeps()          # first call runs uncaptured (compiles kernels, sizes contact buffers) ...
                wp.synchronize_device(self.device)
                with wp.ScopedCapture(device=self.device, force_module_load=False) as cap:
                    self._substeps()      # ... and records the graph; recording does not execute, so the state
                self.graph = cap.graph    # advanced exactly one control step on this call
            else:
                wp.capture_launch(self.graph)
        else:
            self._substeps()
        self.steps += 1
        self.sim_time += self.control_dt
        self._signals()

    def _signals(self):
        dev = self.device
        for a in (self.n_above, self.n_bad, self.n_enc):
            a.zero_()
        self.vmax.zero_(); self.pad_pen.zero_()
        self.zmin.fill_(1e3); self.clear_l.fill_(1.0); self.clear_r.fill_(1.0)
        wp.launch(_world_signals, dim=self.model.particle_count,
                  inputs=[self.s0.particle_q, self.s0.particle_qd, self.ppw, self.towel.table_z, 0.05,
                          self.s0.body_q, self.bpw, *self._pad_args, self.towel.particle_radius],
                  outputs=[self.n_above, self.n_bad, self.vmax, self.zmin, self.clear_l, self.clear_r, self.n_enc,
                           self.pad_pen], device=dev)
        for ep in self._extra_pads:
            ep["n_enc"].zero_(); ep["pad_pen"].zero_(); ep["clear_l"].fill_(1.0); ep["clear_r"].fill_(1.0)
            for a in self._scratch:
                a.zero_()
            wp.launch(_world_signals, dim=self.model.particle_count,
                      inputs=[self.s0.particle_q, self.s0.particle_qd, self.ppw, self.towel.table_z, 0.05,
                              self.s0.body_q, self.bpw, *ep["args"], self.towel.particle_radius],
                      outputs=[self._scratch[0], self._scratch[1], self._scratch[2], self._scratch[3], ep["clear_l"],
                               ep["clear_r"], ep["n_enc"], ep["pad_pen"]], device=dev)
        wp.launch(_corner_dist, dim=self.n, inputs=[self.s0.particle_q, self.ppw, self.towel.corner_a,
                                                     self.towel.corner_c], outputs=[self.cdist], device=dev)

    def signals(self) -> dict:
        """Per-world signals of the last control step as torch tensors (cuda), names as recorded in traces."""
        import torch
        t = lambda a: wp.to_torch(a)
        enc = torch.where(t(self.n_enc) > 0, t(self.clear_l) + t(self.clear_r), torch.ones_like(t(self.clear_l)))
        q = wp.to_torch(self.s0.particle_q).reshape(self.n, self.ppw, 3)
        A, C = q[:, self.towel.corner_a], q[:, self.towel.corner_c]
        out = dict(
            towel_A_x=A[:, 0].clone(), towel_A_y=A[:, 1].clone(), towel_A_z=A[:, 2].clone(),
            towel_C_x=C[:, 0].clone(), towel_C_y=C[:, 1].clone(), towel_C_z=C[:, 2].clone(),
            towel_corner_dist=t(self.cdist).clone(),
            towel_frac_above_5cm=t(self.n_above).float() / self.ppw,
            robot_cloth_impulse=t(self.impulse).clone(),
            robot_cloth_fmax=t(self.fmax).clone(),
            finger_cloth_enclosure=enc,
            cloth_n_enclosed=t(self.n_enc).float(),
            cloth_speed_max=t(self.vmax).clone(),
            cloth_zmin=t(self.zmin).clone(),
            cloth_nonfinite=t(self.n_bad).float(),
            pad_penetration=t(self.pad_pen).clone(),
        )
        if self._extra_pads:                     # per gripper (pair k) + the smallest over all grippers
            encs = [enc] + [torch.where(t(ep["n_enc"]) > 0, t(ep["clear_l"]) + t(ep["clear_r"]),
                                        torch.ones_like(t(ep["clear_l"]))) for ep in self._extra_pads]
            for k, e in enumerate(encs):
                out[f"finger_cloth_enclosure_{k}"] = e.clone()
            out["finger_cloth_enclosure"] = torch.stack(encs, 0).amin(0)
            out["pad_penetration"] = torch.stack([out["pad_penetration"]] + [t(ep["pad_pen"]) for ep in self._extra_pads], 0).amax(0)
        if self.n_groups:
            ig = t(self.impulse_g).reshape(self.n, self.n_groups)
            for g in range(self.n_groups):
                out[f"robot_cloth_impulse_{g}"] = ig[:, g].clone()
        return out

    def particle_q(self):
        """(N, V, 3) torch view of the current particle positions."""
        return wp.to_torch(self.s0.particle_q).reshape(self.n, self.ppw, 3)

    def particle_qd(self):
        return wp.to_torch(self.s0.particle_qd).reshape(self.n, self.ppw, 3)

    def particle_q_init(self):
        return wp.to_torch(self.q_init).reshape(self.n, self.ppw, 3)

    def reset(self, world_mask=None):
        """Reset the towel of the selected worlds (all if None) to its initial flat state."""
        m = np.ones(self.n, np.int32) if world_mask is None else np.asarray(
            world_mask.detach().cpu().numpy() if hasattr(world_mask, "detach") else world_mask, np.int32).reshape(-1)
        mask = wp.array(m, dtype=int, device=self.device)
        for s in (self.s0, self.s1):
            wp.launch(_reset_particles, dim=self.model.particle_count,
                      inputs=[s.particle_q, s.particle_qd, self.q_init, self.ppw, mask, self.mass0],
                      outputs=[self.model.particle_mass, self.model.particle_inv_mass, self.attached], device=self.device)
        g = self.grip.numpy(); g[m.astype(bool)] = 0; self.grip.assign(g); self.grip_prev.assign(g)
        try:  # cold-start VBD's contact / history state for the reset worlds
            wm = np.zeros(self.n + 1, bool); wm[: self.n] = m.astype(bool)
            self.solver.reset(self.s0, world_mask=wp.array(wm, dtype=wp.bool, device=self.device))
        except Exception:
            pass

    def describe(self) -> dict:
        return dict(num_worlds=self.n, particles_per_world=self.ppw, colliders=self.link_names, towel=asdict(self.towel),
                    solver=asdict(self.cfg), control_dt=self.control_dt, newton=newton.__version__, warp=wp.__version__)
