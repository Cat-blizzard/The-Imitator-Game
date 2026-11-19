"""Wrench-consistent tactile pads for imsim (E1): engine-independent T0 / T1 / T2 signals on top of a rigid-body solver.

Levels (cumulative; obs["tactile"] holds every field up to the level asked for):
  T0  contact   (N, P) bool        pad touched (normal force above `contact_threshold` N)
  T1  force     (N, P, 3) float    contact force ON the pad in the pad frame: (shear x, shear y, normal); normal > 0 =
                                   compression [N]
      cop       (N, P, 2) float    centre of pressure in the pad frame (x across, y along the finger) [m]; 0 if no contact
      force_world (N, P, 3)        the same force in the world frame [N]
  T2  pressure  (N, P, nx, ny)     mean normal pressure over each taxel [Pa] (taxel (i, j) at x_i, y_j, see pads.py)
      shear     (N, P, nx, ny, 2)  mean tangential traction (x, y) over each taxel [Pa]
N = num_envs, P = number of pads (sensor.pad_names), (nx, ny) = layout ("12x32" FlexiTac default, or "16x16").

Method (T2). The rigid solver decides the dynamics; the pad only redistributes the solver's contact wrench:
  * Elastic (Winkler) foundation: p(x, y) = k * max(0, delta(x, y) - gap(x, y)), k = E / h of the pad layer.
  * gap(x, y): distance along the pad normal from the pad surface to each contacting body, ray-cast against the body's
    collision shapes (imitator.sensors.geometry: analytic primitives, convex hulls, triangle meshes; Warp kernels).
  * delta(x, y) = delta0 + a x + b y (rigid relative displacement: indentation + two small tilts; a = -theta_y,
    b = theta_x). The 3 unknowns are solved so that the pressure reproduces the pad's normal force and its two in-plane
    moments (first moments sum f x, sum f y = Fn * CoP) from the solver's contact impulses accumulated over the physics
    substeps of the control step. This is a convex problem (gradient of Phi = sum k A/2 max(0, delta - gap)^2): damped
    Newton with a backtracking line search and an exact 1-D polish of delta0, batched over all envs / pads / bodies.
  * Several bodies on one pad: one delta0 per body; tilt per body when per-body moments are known (CPU contact points),
    shared tilt otherwise (GPU: per-body forces from pair queries, total moments from the link wrench).
  * Tangential force: distributed in proportion to p; the twist moment about the pad normal is matched with a
    rigid-rotation slip field t_i = p_i (F_t / P + kappa z x (r_i - CoP)) (option twist=True).
  * Taxel value = integral over the taxel cell (supersample^2 sub-points per taxel), so a CoP anywhere on the pad is
    representable and thin contacts (edges) are not lost between taxel centres.

Wrench sources (sources.py): CPU PhysX = contact points (normal impulses, positions) for normal force and CoP, link
incoming joint forces for friction and twist (PhysX contact-point impulses carry no friction: measured). GPU PhysX =
contact-pair impulse queries (per-body normal force) + link incoming joint forces (full 6-D contact wrench). Both are
accumulated over every physics substep.

Measured limits (SAPIEN 3.0.3 / PhysX 5, research/results/e1_tactile/STATUS.md):
  * GPU: link incoming joint forces carry ONE contact pair per articulation (in a two-finger grasp one finger reports no
    contact force) and the contact-body query has no friction, so on GPU friction and in-plane moments exist only for the
    pad whose link wrench reproduces its pair force (diag link_ok); other pads get shear 0 and a geometric CoP.
  * CPU: links of one articulation touching each other (self-contact) report no incoming joint force -> shear 0 (flagged);
    sleeping islands report frozen, non-equilibrium impulses -> keep_awake (default) disables sleeping of observed bodies.
  * Moment matching reproduces the rigid solver's CoP. For flat / edge / vertex contacts the solver often concentrates
    the impulse at an extreme point (statically indeterminate patch, or the rigid limit of a slightly tilted face); a
    Winkler layer of thickness h cannot follow it beyond the tilt bound and the map stops short (moments='geometric'
    ignores the solver's moments instead).
  * Winkler = no lateral coupling: k must be calibrated (vs a linear-elastic FEM pad: sphere maps correlate 0.98 with
    k = E/h, peak -20 %; 1.00 with k = 1.69 E/h; edges 0.99; flat punch 0.975).

Prior art (the method is engineering on top of these; validation: research/results/e1_tactile/): Kasolowsky & Baeuml
2409.12735 (linear-elastic skin from engine normal force + geometry), OpenTac-Sim 2610.31661 (Winkler layer matched to
the normal force), 2610.10510, TactileStep 2609.28959, HydroelasticTouch 2501.08077. Added here: matching both in-plane
moments, friction (force + twist), several bodies per pad, one implementation for PhysX CPU / GPU.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .geometry import BIG

LEVELS = ("T0", "T1", "T2")
LAYOUTS = {"12x32": (12, 32), "16x16": (16, 16), "flexitac": (12, 32)}
DEFAULT_E_PA = 1.0e6            # pad layer Young's modulus [Pa] (soft elastomer)
DEFAULT_H_M = 0.002             # pad layer thickness [m]
DEFAULT_K = DEFAULT_E_PA / DEFAULT_H_M      # Winkler modulus [Pa / m]
FAR = 1.0e3                     # "no surface" gap used inside the solver [m]


def parse_layout(layout) -> tuple[int, int]:
    if isinstance(layout, (tuple, list)):
        return int(layout[0]), int(layout[1])
    key = str(layout).lower().replace(" ", "")
    if key in LAYOUTS:
        return LAYOUTS[key]
    a, b = key.split("x")
    return int(a), int(b)


def parse_level(level) -> int:
    if level is None or level is False:
        return -1
    s = str(level).upper()
    if s not in LEVELS:
        raise ValueError(f"tactile level must be one of {LEVELS} (got {level!r})")
    return LEVELS.index(s)


# ================================================================================================ pad grid
@dataclass
class PadGrid:
    """sub-points of one pad: (nx*ss) x (ny*ss) cell-centred points, row-major in (i, j), each carrying its cell's area
    (taxel value = integral over the taxel cell; every sub-point belongs to exactly one taxel). The outermost row and column
    are moved onto the pad boundary, so a centre of pressure anywhere on the pad, up to its edge, is representable
    (cell centres alone stop half a sub-pitch short of the edge)."""
    nx: int
    ny: int
    ss: int
    W: float
    L: float

    @property
    def mx(self) -> int:
        return self.nx * self.ss

    @property
    def my(self) -> int:
        return self.ny * self.ss

    @property
    def S(self) -> int:
        return self.mx * self.my

    @staticmethod
    def _axis(n, length):
        x = -length / 2 + (np.arange(n) + 0.5) * length / n
        if n > 1:
            x[0], x[-1] = -length / 2, length / 2
        return x

    def coords(self) -> np.ndarray:
        """(S, 2) sub-point coordinates in the pad frame"""
        X, Y = np.meshgrid(self._axis(self.mx, self.W), self._axis(self.my, self.L), indexing="ij")
        return np.stack([X.ravel(), Y.ravel()], -1)

    def weights(self) -> np.ndarray:
        """(S,) area carried by every sub-point (its cell; sums to W * L)"""
        return np.full(self.S, self.W * self.L / self.S)

    def binning(self) -> np.ndarray:
        """(S, nx * ny) 0/1 matrix: sub-point -> taxel"""
        I, J = np.meshgrid(np.arange(self.mx) // self.ss, np.arange(self.my) // self.ss, indexing="ij")
        M = np.zeros((self.S, self.nx * self.ny))
        M[np.arange(self.S), (I * self.ny + J).ravel()] = 1.0
        return M

    def taxel_centres(self) -> np.ndarray:
        xs = -self.W / 2 + (np.arange(self.nx) + 0.5) * self.W / self.nx
        ys = -self.L / 2 + (np.arange(self.ny) + 0.5) * self.L / self.ny
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        return np.stack([X, Y], -1)

    @property
    def sub_area(self) -> float:
        return self.W * self.L / self.S

    @property
    def taxel_area(self) -> float:
        return self.W * self.L / (self.nx * self.ny)


def bin_taxels(v: torch.Tensor, M: torch.Tensor, nx: int, ny: int) -> torch.Tensor:
    """node values (..., S) or (..., S, C) -> taxel sums (..., nx, ny) / (..., nx, ny, C) with the binning matrix M (S, T)"""
    if v.shape[-1] == M.shape[0]:
        return (v @ M).reshape(*v.shape[:-1], nx, ny)
    out = torch.einsum("...sc,st->...tc", v, M)
    return out.reshape(*v.shape[:-2], nx, ny, v.shape[-1])


# ================================================================================================ solver
def _d_exact(Gp, c, Fn, m_ok):
    """exact solve of sum_s c_s max(0, d - Gp_s) = Fn for d (per batch row): sort the gaps, the force is piecewise linear
    in d. Gp (..., S); c broadcastable to Gp; Fn (...); m_ok (..., S) usable points. Gaps are shifted by their minimum
    before the cumulative sums (float32 cancellation)."""
    Gm = torch.where(m_ok, Gp, torch.full_like(Gp, FAR))
    g0 = Gm.min(-1, keepdim=True).values
    Gs = torch.where(m_ok, Gm - g0, torch.full_like(Gm, FAR))
    cm = torch.where(m_ok, c.expand_as(Gp), torch.zeros_like(Gp))
    gs, idx = Gs.sort(-1)
    cs = cm.gather(-1, idx)
    C = cs.cumsum(-1)
    D = (cs * gs).cumsum(-1)
    g_next = torch.cat([gs[..., 1:], torch.full_like(gs[..., :1], 2 * FAR)], -1)
    F_next = C * g_next - D                                             # force when d reaches the next gap
    j = (F_next < Fn[..., None]).sum(-1, keepdim=True).clamp_max(gs.shape[-1] - 1)
    d = (Fn[..., None] + D.gather(-1, j)) / C.gather(-1, j).clamp_min(1e-30)
    return (d + g0)[..., 0]


def _d_newton(d, Gp, c, Fn, m_ok, n_iter: int = 3):
    """warm-started Newton refinement of the 1-D solve (convex piecewise-linear: converges from either side)"""
    for _ in range(n_iter):
        e = d[..., None] - Gp
        act = (e > 0) & m_ok
        F = (c * e * act).sum(-1)
        dF = (c * act).sum(-1)
        d = torch.where(dF > 0, d - (F - Fn) / dF.clamp_min(1e-30), d)
    return d


_STEPS: dict = {}


def _steps(dev):
    key = str(dev)
    if key not in _STEPS:
        _STEPS[key] = torch.tensor([1.0, 0.25, 0.0625, 0.0], device=dev, dtype=torch.float32)
    return _STEPS[key]


def winkler_solve(gap: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor, c: torch.Tensor, Fn: torch.Tensor,
                  Mx: torch.Tensor | None = None, My: torch.Tensor | None = None, tilt: bool = True, n_iter: int = 10,
                  polish: int = 2, amax: torch.Tensor | float = 0.2, bmax: torch.Tensor | float = 0.2
                  ) -> tuple[torch.Tensor, dict]:
    """Batched Winkler fit with a shared tilt (pure tensor ops, fixed iteration counts: CUDA-graph capturable).

    gap (B, K, S) [m] gap of each body's surface above each sub-point (>= BIG/2: no surface);
    xs, ys (B or 1, S) sub-point coordinates in the pad frame; c (B or 1, S) = k * sub-point area [N/m] (0 = dead point);
    Fn (B, K) normal force per body (<= 0: body inactive); Mx, My (B,) target first moments sum f*x, sum f*y over all
    bodies (ignored when tilt=False: delta is then a pure indentation per body).
    amax, bmax: bounds on the tilts |a|, |b| (scalar or (B,)); the pad layer cannot be compressed by more than its thickness
    h, so the sensor passes amax = 2 h / W, bmax = 2 h / L. A target CoP outside what the contact geometry can produce is
    then approached as closely as the bound allows (the residual is reported), instead of an unbounded solve.
    Returns f (B, K, S) sub-point forces [N] and info: d (B, K), a, b (B,), on (B, K), no_surface (B, K), M (B, 2)."""
    B, K, S = gap.shape
    dev, dt = gap.device, torch.float32
    gap = gap.to(dt)
    xs = xs.to(dt).reshape(-1, 1, S); ys = ys.to(dt).reshape(-1, 1, S); c = c.to(dt).reshape(-1, 1, S)
    Fn = Fn.to(dt).clamp_min(0.0)
    finite = (gap < BIG / 2) & (c > 0)
    G = torch.where(finite, gap, torch.full_like(gap, FAR))
    m_ok = finite & (Fn > 0)[..., None]
    has = m_ok.any(-1)
    on = (Fn > 0) & has
    m_ok = m_ok & on[..., None]
    Fon = torch.where(on, Fn, torch.zeros_like(Fn))
    a = torch.zeros(B, device=dev, dtype=dt); b = torch.zeros(B, device=dev, dtype=dt)
    d = _d_exact(G, c, Fon, m_ok)
    if tilt and Mx is not None:
        amax = torch.as_tensor(amax, device=dev, dtype=dt).expand(B)
        bmax = torch.as_tensor(bmax, device=dev, dtype=dt).expand(B)
        Ftot = Fon.sum(-1)
        # target CoP clamped into the bounding box of the points that can carry load
        big = torch.full_like(G, 1e9)
        x_lo = torch.where(m_ok, xs.expand_as(G), big).flatten(1).min(-1).values
        x_hi = torch.where(m_ok, xs.expand_as(G), -big).flatten(1).max(-1).values
        y_lo = torch.where(m_ok, ys.expand_as(G), big).flatten(1).min(-1).values
        y_hi = torch.where(m_ok, ys.expand_as(G), -big).flatten(1).max(-1).values
        pos = Ftot > 0
        cx = torch.where(pos, Mx.to(dt) / Ftot.clamp_min(1e-30), torch.zeros_like(Ftot))
        cy = torch.where(pos, My.to(dt) / Ftot.clamp_min(1e-30), torch.zeros_like(Ftot))
        cx = torch.where(pos, torch.minimum(torch.maximum(cx, x_lo), x_hi), cx)
        cy = torch.where(pos, torch.minimum(torch.maximum(cy, y_lo), y_hi), cy)
        tMx, tMy = Ftot * cx, Ftot * cy
        steps = _steps(dev)
        # Levenberg-Marquardt damping of the tilt block, scaled by the stiffness of ALL loadable points: when only one or
        # two sub-points are in contact (vertex / edge contact) the active tilt Hessian is singular and a plain Newton step
        # is useless (the line search would reject it and the tilt would never move)
        cu = c * m_ok
        lm = 1e-3 * 0.5 * ((cu * xs * xs).sum((-1, -2)) + (cu * ys * ys).sum((-1, -2))) + 1e-12

        def psi(dd, aa, bb):                    # dd (..., B, K), aa / bb (..., B)
            e = dd[..., None] - (G - aa[..., None, None] * xs - bb[..., None, None] * ys)
            e = torch.where(m_ok, e.clamp_min(0.0), torch.zeros_like(e))
            return 0.5 * (c * e * e).sum((-1, -2)) - (Fon * dd).sum(-1) - tMx * aa - tMy * bb

        # per-problem feature matrix [1, x, y, x^2, x y, y^2] (S, 6): all gradient / Hessian sums are two batched matmuls
        X1, Y1 = xs.expand(B, 1, S)[:, 0], ys.expand(B, 1, S)[:, 0]
        Feat = torch.stack([torch.ones_like(X1), X1, Y1, X1 * X1, X1 * Y1, Y1 * Y1], -1)          # (B, S, 6)
        for _ in range(n_iter):
            e = d[..., None] - (G - a[:, None, None] * xs - b[:, None, None] * ys)
            m = (e > 0) & m_ok
            cm = c * m
            f = cm * e
            gF = torch.bmm(f, Feat[..., :3])                                       # (B, K, 3): sum f, f x, f y
            hF = torch.bmm(cm, Feat)                                               # (B, K, 6)
            gd = torch.where(on, gF[..., 0] - Fon, torch.zeros_like(Fon))
            ga = gF[..., 1].sum(-1) - tMx
            gb = gF[..., 2].sum(-1) - tMy
            Hdd = hF[..., 0]
            Hda = hF[..., 1]; Hdb = hF[..., 2]
            Haa = hF[..., 3].sum(-1); Hab = hF[..., 4].sum(-1); Hbb = hF[..., 5].sum(-1)
            inv = torch.where(on & (Hdd > 0), 1.0 / Hdd.clamp_min(1e-30), torch.zeros_like(Hdd))
            Saa = Haa - (Hda * Hda * inv).sum(-1); Sab = Hab - (Hda * Hdb * inv).sum(-1); Sbb = Hbb - (Hdb * Hdb * inv).sum(-1)
            Saa = Saa + lm; Sbb = Sbb + lm
            ra = ga - (Hda * gd * inv).sum(-1); rb = gb - (Hdb * gd * inv).sum(-1)
            det = (Saa * Sbb - Sab * Sab).clamp_min(1e-30)
            da = -(Sbb * ra - Sab * rb) / det
            db = -(-Sab * ra + Saa * rb) / det
            # active set of the box constraint: a tilt component at its bound that would move outward is frozen and the
            # other one is solved alone (projected Newton)
            fa = (a.abs() >= amax * (1 - 1e-6)) & (da * a > 0)
            fb = (b.abs() >= bmax * (1 - 1e-6)) & (db * b > 0)
            da1 = torch.where(fb, -ra / Saa, da)
            db1 = torch.where(fa, -rb / Sbb, db)
            da = torch.where(fa, torch.zeros_like(da), da1)
            db = torch.where(fb, torch.zeros_like(db), db1)
            dd = -(gd + Hda * da[:, None] + Hdb * db[:, None]) * inv
            # step length capped so that the tilt bounds are not crossed (d, a, b stay a consistent Newton step), then
            # backtracking over t_cap * (1, 1/4, 1/16, 0): lowest objective wins
            ta = torch.where(da > 0, (amax - a) / da.clamp_min(1e-30),
                             torch.where(da < 0, (-amax - a) / da.clamp_max(-1e-30), torch.ones_like(a)))
            tb = torch.where(db > 0, (bmax - b) / db.clamp_min(1e-30),
                             torch.where(db < 0, (-bmax - b) / db.clamp_max(-1e-30), torch.ones_like(b)))
            tcap = torch.minimum(torch.ones_like(a), torch.minimum(ta, tb)).clamp_min(0.0)
            tt = steps[:, None] * tcap[None]                                        # (n_steps, B)
            Ac = a[None] + tt * da[None]
            Bc = b[None] + tt * db[None]
            Dc = d[None] + tt[..., None] * dd[None]
            k = psi(Dc, Ac, Bc).argmin(0)
            sel = torch.arange(B, device=dev)
            a, b, d = Ac[k, sel], Bc[k, sel], Dc[k, sel]
            d = _d_newton(d, G - a[:, None, None] * xs - b[:, None, None] * ys, c, Fon, m_ok, n_iter=polish)
        d = _d_exact(G - a[:, None, None] * xs - b[:, None, None] * ys, c, Fon, m_ok)
    e = d[..., None] - (G - a[:, None, None] * xs - b[:, None, None] * ys)
    f = torch.where(m_ok & (e > 0), c * e, torch.zeros_like(e))
    info = dict(d=d, a=a, b=b, on=on, no_surface=(Fn > 0) & ~has,
                M=torch.stack([(f * xs).sum((-1, -2)), (f * ys).sum((-1, -2))], -1))
    return f, info


class GraphedWinkler:
    """CUDA-graph replay of winkler_solve for fixed shapes (one graph per (B, K, S, P, tilt)); plain call on CPU.
    Inputs are copied into static buffers; outputs are views of static buffers, valid until the next call."""

    def __init__(self, n_iter: int = 10, polish: int = 2, use_graph: bool = True):
        self.n_iter, self.polish, self.use_graph = n_iter, polish, use_graph
        self._g = {}

    def __call__(self, gap, xs, ys, c, Fn, Mx, My, amax, bmax, tilt: bool = True):
        dev = gap.device
        if dev.type != "cuda" or not self.use_graph:
            return winkler_solve(gap, xs, ys, c, Fn, Mx, My, tilt=tilt, n_iter=self.n_iter, polish=self.polish,
                                 amax=amax, bmax=bmax)
        key = (tuple(gap.shape), tuple(xs.shape), bool(tilt), str(dev))
        ent = self._g.get(key)
        if ent is None:
            ins = [t.detach().clone().contiguous() for t in (gap, xs, ys, c, Fn, Mx, My,
                                                              torch.as_tensor(amax, device=dev, dtype=torch.float32).expand(gap.shape[0]),
                                                              torch.as_tensor(bmax, device=dev, dtype=torch.float32).expand(gap.shape[0]))]
            _steps(dev)
            fn = lambda: winkler_solve(*ins[:7], tilt=tilt, n_iter=self.n_iter, polish=self.polish, amax=ins[7], bmax=ins[8])
            s = torch.cuda.Stream(device=dev)
            s.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(s):
                for _ in range(2):
                    fn()
            torch.cuda.current_stream(dev).wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outs = fn()
            ent = self._g[key] = (graph, ins, outs)
        graph, ins, outs = ent
        for dst, src in zip(ins, (gap, xs, ys, c, Fn, Mx, My, amax, bmax)):
            dst.copy_(torch.as_tensor(src, device=dev, dtype=dst.dtype).expand_as(dst))
        graph.replay()
        return outs


def shear_field(f: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor, Ft: torch.Tensor, Mz: torch.Tensor | None = None):
    """tangential sub-point forces (..., S, 2) from normal sub-point forces f (..., S): t_i = f_i / F * Ft +
    kappa f_i z x (r_i - CoP), kappa set so that the twist about the CoP equals Mz (moment about the pad normal through
    the pad origin, of the tangential forces). Ft (..., 2)."""
    F = f.sum(-1, keepdim=True)
    w = torch.where(F > 0, f / F.clamp_min(1e-30), torch.zeros_like(f))
    t = w[..., None] * Ft[..., None, :]
    if Mz is not None:
        cx = (w * xs).sum(-1, keepdim=True); cy = (w * ys).sum(-1, keepdim=True)
        rx, ry = xs - cx, ys - cy
        J = (f * (rx * rx + ry * ry)).sum(-1, keepdim=True)
        Mc = Mz[..., None] - (cx * Ft[..., 1:2] - cy * Ft[..., 0:1])           # twist about the CoP
        kap = torch.where(J > 0, Mc / J.clamp_min(1e-30), torch.zeros_like(J))
        t = t + torch.stack([-kap * f * ry, kap * f * rx], -1)
    return t


# ================================================================================================ sensor
def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def _has_shapes(obj) -> bool:
    try:
        return len(obj._bodies) > 0 and len(obj._bodies[0].get_collision_shapes()) > 0
    except Exception:
        return False


class TactileSensor:
    """Tactile pads on a ManiSkill env (CPU or GPU PhysX). Typical use goes through imitator.make(..., tactile="T2") or
    TactileWrapper; standalone:

        s = TactileSensor(env, level="T2"); env.reset(); s.on_reset()
        s.begin_step(); env.step(a); tac = s.end_step()      # dict of torch tensors, see the module docstring

    pads             list[PadSpec] (default: pads.default_pads(env) for the env's agents)
    layout           "12x32" (FlexiTac count, default) | "16x16" | (nx, ny)
    supersample      sub-points per taxel side (taxel value = integral over the cell)
    stiffness        Winkler modulus k = E / h [Pa/m] (default 1 MPa / 2 mm); per pad via PadSpec.stiffness
    layer_thickness  h [m]: bounds the tilts (|a| <= 2 h / W, |b| <= 2 h / L)
    contact_threshold  T0 threshold on the pad normal force [N]
    max_bodies       bodies per pad kept for the pressure map (largest normal force first; <= 4)
    moments          "match" (fit both in-plane moments, default) | "geometric" (pure indentation per body: the CoP
                     follows the geometry, the solver's moments are ignored)
    twist            match the moment about the pad normal with a rotational slip field
    source           "auto" (CPU: contact points + link wrench; GPU: pair queries + link wrench) | "points" | "link"
    keep_awake       CPU PhysX: disable sleeping of observed bodies (sleeping islands report frozen, non-equilibrium impulses)
    """

    def __init__(self, env, level="T2", pads=None, layout="12x32", supersample: int = 2, stiffness: float = DEFAULT_K,
                 layer_thickness: float = DEFAULT_H_M, contact_threshold: float = 0.05, max_bodies: int = 2,
                 moments: str = "match", twist: bool = True, source: str = "auto", backoff: float = 0.01,
                 n_iter: int = 10, use_graph: bool = True, footprint_margin: float = 5e-5, diagnostics: bool = False,
                 keep_awake: bool = True):
        self.env = env
        self.level = parse_level(level)
        if self.level < 0:
            raise ValueError("TactileSensor needs a level T0 / T1 / T2")
        self.nx, self.ny = parse_layout(layout)
        self.ss = int(supersample)
        self.pad_specs = list(pads) if pads is not None else None
        self.k_default = float(stiffness)
        self.h_layer = float(layer_thickness)
        self.thr = float(contact_threshold)
        self.K = int(max_bodies)
        if not 1 <= self.K <= 4:
            raise ValueError("max_bodies must be 1..4")
        if moments not in ("match", "geometric"):
            raise ValueError("moments must be 'match' or 'geometric'")
        self.moments, self.twist, self.source_opt = moments, bool(twist), source
        self.backoff = float(backoff)
        self.margin = float(footprint_margin)
        self.diagnostics = bool(diagnostics)
        self.keep_awake = bool(keep_awake)
        self.solver = GraphedWinkler(n_iter=n_iter, use_graph=use_graph)
        self._key = None
        self._hooked = None
        self.last: dict = {}
        self.timing = dict(substep_s=0.0, end_s=0.0, n_steps=0)

    # ---------------------------------------------------------------- setup
    def _scene_key(self):
        u = self.env.unwrapped
        return (id(u.scene), tuple(id(s) for s in getattr(u.scene, "sub_scenes", [])[:2]))

    def setup(self, force: bool = False):
        from . import geometry as G
        from .pads import default_pads
        from .sources import LinkInfo, mat_to_quat_np
        u = self.env.unwrapped
        key = self._scene_key()
        if self._key == key and not force:
            return
        self.u = u
        self.N = int(u.num_envs)
        self.dev = torch.device(u.device)
        self.gpu = bool(u.scene.gpu_sim_enabled)
        src = self.source_opt
        if src == "auto":
            src = "link" if self.gpu else "points"
        if src == "points" and self.gpu:
            raise ValueError("source='points' needs CPU PhysX (contact points are not available on GPU)")
        self.src = src
        agents = [] if u.agent is None else (list(u.agent.agents) if hasattr(u.agent, "agents") else [u.agent])
        self.pads = self.pad_specs if self.pad_specs is not None else default_pads(self.env)
        self.P = len(self.pads)
        self.pad_names = [p.name for p in self.pads]
        self.links = [(u.scene.articulations[p.articulation] if p.articulation else agents[p.agent].robot).links_map[p.link]
                      for p in self.pads]
        self.linfo = [LinkInfo(l, p.rot(), p.pos()) for l, p in zip(self.links, self.pads)]
        dev, N, P = self.dev, self.N, self.P
        f32 = dict(device=dev, dtype=torch.float32)
        # ---- pad grids
        self.grids = [PadGrid(self.nx, self.ny, self.ss, float(p.size[0]), float(p.size[1])) for p in self.pads]
        self.S = self.grids[0].S
        XY = np.stack([g.coords() for g in self.grids])                                  # (P, S, 2)
        self.xs = torch.tensor(XY[..., 0], **f32)
        self.ys = torch.tensor(XY[..., 1], **f32)
        self.sub_area = torch.tensor([g.sub_area for g in self.grids], **f32)
        self.node_w = torch.tensor(np.stack([g.weights() for g in self.grids]), **f32)            # (P, S) [m^2]
        self.Mbin = torch.tensor(self.grids[0].binning(), **f32)                                 # same layout on every pad
        self.taxel_area = torch.tensor([g.taxel_area for g in self.grids], **f32)
        self.kpad = torch.tensor([p.stiffness or self.k_default for p in self.pads], **f32)
        self.amax = torch.tensor([2 * self.h_layer / g.W for g in self.grids], **f32)
        self.bmax = torch.tensor([2 * self.h_layer / g.L for g in self.grids], **f32)
        self.pad_R_L = torch.tensor(np.stack([p.rot() for p in self.pads]), **f32)        # (P, 3, 3)
        self.pad_o_L = torch.tensor(np.stack([p.pos() for p in self.pads]), **f32)        # (P, 3)
        self.pad_q_L = torch.tensor(np.stack([mat_to_quat_np(p.rot()) for p in self.pads]), **f32)
        self.W = torch.tensor([g.W for g in self.grids], **f32)
        self.L = torch.tensor([g.L for g in self.grids], **f32)
        # ---- candidate bodies (every actor / link with collision shapes, except the pad links themselves)
        pad_ents = {id(l): i for i, l in enumerate(self.links)}
        cands = []
        for a in u.scene.actors.values():
            if _has_shapes(a):
                cands.append(("actor", a.name, a))
        for art in u.scene.articulations.values():
            for l in art.links:
                if _has_shapes(l):
                    cands.append(("link", f"{art.name}/{l.name}", l))
        self.cands = cands
        C = self.C = len(cands)
        self.cand_names = [n for _, n, _ in cands]
        # exclusion: a pad never reads its own link (or its parent / children: PhysX filters adjacent links)
        excl = np.zeros((P, C), bool)
        for p, l in enumerate(self.links):
            comp = l._bodies[0]
            near = {id(comp)} | {id(ch) for ch in comp.children}
            if comp.parent is not None:
                near.add(id(comp.parent))
            for c, (_, _, obj) in enumerate(cands):
                if id(obj._bodies[0]) in near:
                    excl[p, c] = True
        self.excl = torch.tensor(excl, device=dev)
        # ---- geometry bank: row c * N + e
        bank = G.GeomBank(G.wp_device(dev))
        self.cand_env = []
        for c, (_, name, obj) in enumerate(cands):
            sidx = obj._scene_idxs.tolist() if hasattr(obj, "_scene_idxs") else list(range(len(obj._bodies)))
            if len(obj._bodies) == 1 and N > 1 and len(sidx) != 1:
                sidx = [0]
            m = {int(e): j for j, e in enumerate(sidx)}
            self.cand_env.append(m)
            for e in range(N):
                if e in m:
                    sh = G.shapes_of_component(obj._bodies[m[e]])
                    bank.add_body(sh)
                else:
                    bank.add_body([])
        self.bank = bank
        # ---- pad surface profile (ray-cast the pad link's own collision shapes, link frame, env 0)
        self._profile(G)
        # ---- pose gathering
        self._pose_setup()
        # ---- source specific
        self._acc_alloc()
        if self.src == "points":
            self._ent_pads = {}
            for p, l in enumerate(self.links):
                self._ent_pads.setdefault(l._bodies[0].entity, []).append(p)
            self._ent_cand = {}
            for c, (_, _, obj) in enumerate(cands):
                for j, b in enumerate(obj._bodies):
                    self._ent_cand[b.entity] = c
        self._query = None
        self._art_rows = None
        if not self.gpu and self.keep_awake:
            self._disable_sleep()
        self._install_hook()
        self._key = key

    def _disable_sleep(self):
        """CPU PhysX: a sleeping island reports FROZEN contact impulses that are not in equilibrium (measured: a 0.31 kg
        box resting on a pad fell asleep after ~0.5 s and kept reporting 2.858 N instead of 3.0 N with the CoP 3.4 mm off
        centre). The sensor therefore disables sleeping of every dynamic body it can observe and of the articulations
        carrying pads (sleep threshold 0, woken up). Opt out with keep_awake=False."""
        import sapien.physx as px
        seen = set()
        for kind, _, obj in self.cands:
            comp = obj._bodies[0]
            if kind == "actor" and isinstance(comp, px.PhysxRigidDynamicComponent) and not comp.kinematic:
                comp.set_sleep_threshold(0.0)
                comp.wake_up()
        for l in self.links:
            art = l.articulation._objs[0]
            if id(art) not in seen:
                seen.add(id(art))
                art.set_sleep_threshold(0.0)
                l._bodies[0].wake_up()                 # SAPIEN 3.0.3: wake_up lives on the link components

    def _profile(self, G):
        """per pad: surface height h (S,) of the link's collision surface below / above the pad plane, valid mask"""
        P, S = self.P, self.S
        h = np.zeros((P, S), np.float32)
        valid = np.zeros((P, S), bool)
        z0 = 0.05
        for p, (pad, l) in enumerate(zip(self.pads, self.links)):
            bank = G.GeomBank("cpu")
            bank.add_body(G.shapes_of_component(l._bodies[0]))
            R = pad.rot(); o = pad.pos()
            Rf = R @ np.diag([1.0, -1.0, -1.0])                     # frame (x, -y, -n): rays go along -n
            from .sources import mat_to_quat_np
            q = torch.tensor(mat_to_quat_np(Rf)[None], dtype=torch.float32)
            XY = self.grids[p].coords()
            sub = torch.tensor(np.c_[XY[:, 0], -XY[:, 1], np.full(S, -z0)][None], dtype=torch.float32)
            t = bank.cast(torch.tensor(o[None], dtype=torch.float32), q, sub, torch.zeros((1, 1), dtype=torch.int32),
                          torch.zeros((1, 3)), torch.tensor([[1.0, 0, 0, 0]]))[0, 0].numpy()
            hit = t < BIG / 2
            hh = z0 - t
            ok = hit & (hh >= -0.006) & (hh <= 0.006)
            h[p] = np.where(ok, hh, 0.0)
            valid[p] = ok
        f32 = dict(device=self.dev, dtype=torch.float32)
        self.h = torch.tensor(h, **f32)
        self.valid = torch.tensor(valid, device=self.dev)
        self.c = self.kpad[:, None] * self.node_w * self.valid.float()                    # (P, S) [N/m]
        self.zlo = torch.tensor([float(h[p][valid[p]].min()) if valid[p].any() else 0.0 for p in range(P)], **f32) - 0.003
        self._zlo_np = self.zlo.cpu().numpy()
        self.sub_local = torch.stack([self.xs, self.ys, self.h - self.backoff], -1)       # ray origins in the pad frame
        self.profile_valid_frac = valid.mean(1)

    def _pose_setup(self):
        """GPU: one gather of all candidate poses from the PhysX body buffer; static actors cached"""
        N, C = self.N, self.C
        self._dyn_rows, self._dyn_src = [], []
        self._static_pose = torch.zeros((C * N, 7), device=self.dev)
        self._static_pose[:, 3] = 1.0
        for c, (kind, _, obj) in enumerate(self.cands):
            static = kind == "actor" and getattr(obj, "px_body_type", "") == "static"
            m = self.cand_env[c]
            if static or not self.gpu:
                if static:
                    rp = obj.pose.raw_pose.to(self.dev).float()
                    for e, j in m.items():
                        self._static_pose[c * N + e] = rp[min(j, rp.shape[0] - 1)]
                continue
            idx = obj._body_data_index
            for e, j in m.items():
                self._dyn_rows.append(c * N + e)
                self._dyn_src.append(int(idx[j]))
        self._dyn_rows = torch.tensor(self._dyn_rows, device=self.dev, dtype=torch.long)
        self._dyn_src = torch.tensor(self._dyn_src, device=self.dev, dtype=torch.long)
        self._cpu_dyn = [(c, obj) for c, (kind, _, obj) in enumerate(self.cands)
                         if not (kind == "actor" and getattr(obj, "px_body_type", "") == "static")]

    def body_poses(self) -> torch.Tensor:
        """(C * N, 7) world poses (p, q wxyz) of every body-table row"""
        poses = self._static_pose.clone()
        if self.gpu:
            if len(self._dyn_rows):
                buf = self.u.scene.px.cuda_rigid_body_data.torch()
                poses[self._dyn_rows] = buf[self._dyn_src, :7].float()
        else:
            for c, obj in self._cpu_dyn:
                poses[c * self.N] = obj.pose.raw_pose[0].to(self.dev).float()
        return poses

    def pad_poses(self):
        """(N, P, 3), (N, P, 4) world pose of every pad frame, and (N, P, 3, 3) rotation"""
        lp = torch.stack([l.pose.raw_pose.to(self.dev).float() for l in self.links], 1)   # (N, P, 7)
        from .sources import quat_to_mat
        Rl = quat_to_mat(lp[..., 3:])
        pos = lp[..., :3] + torch.einsum("npij,pj->npi", Rl, self.pad_o_L)
        q = _quat_mul(lp[..., 3:], self.pad_q_L[None].expand_as(lp[..., 3:]))
        return pos, q, torch.einsum("npij,pjk->npik", Rl, self.pad_R_L), Rl

    # ---------------------------------------------------------------- accumulators
    def _acc_alloc(self):
        N, P, C, dev = self.N, self.P, self.C, self.dev
        self.acc_n = torch.zeros((N, P, C), device=dev)          # normal impulse per body [N s]
        self.acc_x = torch.zeros((N, P, C), device=dev)          # first moments [N s m]
        self.acc_y = torch.zeros((N, P, C), device=dev)
        self.acc_v = torch.zeros((N, P, C, 3), device=dev)       # impulse vector on the pad link (world) [N s]
        self.acc_off = torch.zeros((N, P), device=dev)           # off-pad normal impulse (CPU points)
        self.acc_in = torch.zeros((N, P, 6), device=dev)         # incoming joint impulse, link frame, about link origin
        self.acc_ch = torch.zeros((N, P, 6), device=dev)         # children's incoming joint impulses (same frame)
        if not self.gpu:
            # CPU points: in-pad per (pad, body); impulse vectors and off-pad force / moment in the PAD frame
            self._np_n = np.zeros((P, C)); self._np_x = np.zeros((P, C)); self._np_y = np.zeros((P, C))
            self._np_v = np.zeros((P, C, 3)); self._np_off = np.zeros(P); self._np_in = np.zeros((P, 6))
            self._np_ch = np.zeros((P, 6)); self._np_offF = np.zeros((P, 3)); self._np_offM = np.zeros((P, 3))
        self.n_sub = 0

    def _install_hook(self):
        u = self.env.unwrapped
        if self._hooked is u:
            return
        orig = u._after_simulation_step

        def hook():
            orig()
            if self._active:
                self._on_substep()
        u._after_simulation_step = hook
        self._hooked = u
        self._active = False

    def _need_link(self) -> bool:
        return self.level >= 1

    def _build_gpu_query(self):
        px = self.u.scene.px
        N, P, C = self.N, self.P, self.C
        pairs, dense = [], []
        for p, l in enumerate(self.links):
            for c, (_, _, obj) in enumerate(self.cands):
                if bool(self.excl[p, c]):
                    continue
                for e, j in self.cand_env[c].items():
                    pairs.append((l._bodies[e], obj._bodies[j]))
                    dense.append((e * P + p) * C + c)
        self._query = px.gpu_create_contact_pair_impulse_query(pairs) if pairs else None
        self._dense = torch.tensor(dense, device=self.dev, dtype=torch.long)
        # incoming joint force rows: (articulation gpu index per env, link index) per pad and per child
        self._art_rows = []
        for p, (l, li) in enumerate(zip(self.links, self.linfo)):
            di = li.art._data_index.to(self.dev).long()                   # (N,)
            self._art_rows.append((di, li.index, [ch["index"] for ch in li.children]))
        self._jt_R = torch.tensor(np.stack([li.R_LJ for li in self.linfo]), device=self.dev, dtype=torch.float32)
        self._jt_p = torch.tensor(np.stack([li.p_LJ for li in self.linfo]), device=self.dev, dtype=torch.float32)
        # children: joint frame -> pad link frame (fixed joints; q = 0 transform otherwise)
        self._ch_T = [[torch.tensor(ch["T_L_ch0"] @ ch["T_ch_J"], device=self.dev, dtype=torch.float32) for ch in li.children]
                      for li in self.linfo]

    # ---------------------------------------------------------------- per substep
    def _on_substep(self):
        import time
        t0 = time.perf_counter()
        u = self.u
        dt = float(u.scene.px.timestep) if hasattr(u.scene.px, "timestep") else 1.0 / float(u.sim_freq)
        self.n_sub += 1
        if self.gpu:
            px = u.scene.px
            if self._query is None and self._art_rows is None:
                self._build_gpu_query()
            if self._query is not None:
                px.gpu_query_contact_pair_impulses(self._query)
                imp = self._query.cuda_impulses.torch()
                self.acc_v.view(-1, 3).index_add_(0, self._dense, imp.float())
            if self._need_link():
                px.gpu_fetch_articulation_link_incoming_joint_forces()
                buf = px.cuda_articulation_link_incoming_joint_forces.torch()
                from .sources import joint_to_link
                for p, (di, li_idx, ch_idx) in enumerate(self._art_rows):
                    F = buf[di, li_idx].float()                                       # (N, 6)
                    self.acc_in[:, p] += joint_to_link(F, self._jt_R[p], self._jt_p[p]) * dt
                    for T, ci in zip(self._ch_T[p], ch_idx):
                        Fc = buf[di, ci].float()
                        self.acc_ch[:, p] += joint_to_link(Fc, T[:3, :3], T[:3, 3]) * dt
        else:
            self._cpu_substep(dt)
        self.timing["substep_s"] += time.perf_counter() - t0

    def _cpu_substep(self, dt):
        u = self.u
        contacts = u.scene.px.get_contacts()
        pad_T = {}
        for c in contacts:
            b0, b1 = c.bodies
            e0, e1 = b0.entity, b1.entity
            for side, ent, other in ((0, e0, e1), (1, e1, e0)):
                pads = self._ent_pads.get(ent)
                if not pads:
                    continue
                ci = self._ent_cand.get(other)
                if ci is None:
                    continue
                sign = 1.0 if side == 0 else -1.0
                for pt in c.points:
                    J = sign * np.asarray(pt.impulse, float)
                    if not J.any():
                        continue
                    pos = np.asarray(pt.position, float)
                    placed = False
                    for p in pads:
                        if self.excl_np(p, ci):
                            continue
                        T = pad_T.get(p)
                        if T is None:
                            T = pad_T[p] = self._pad_T_cpu(p)
                        r = T[:3, :3].T @ (pos - T[:3, 3])
                        JP = T[:3, :3].T @ J
                        jn = -float(JP[2])
                        g = self.grids[p]
                        if (jn > 0 and abs(r[0]) <= g.W / 2 + self.margin and abs(r[1]) <= g.L / 2 + self.margin
                                and self._zlo_np[p] <= r[2] <= 0.003):
                            self._np_n[p, ci] += jn
                            self._np_x[p, ci] += jn * r[0]
                            self._np_y[p, ci] += jn * r[1]
                            self._np_v[p, ci] += JP
                            placed = True
                            break
                    if not placed:
                        p = pads[0]
                        T = pad_T.get(p)
                        if T is None:
                            T = pad_T[p] = self._pad_T_cpu(p)
                        r = T[:3, :3].T @ (pos - T[:3, 3])
                        JP = T[:3, :3].T @ J
                        self._np_off[p] += float(np.linalg.norm(J))
                        self._np_offF[p] += JP
                        self._np_offM[p] += np.cross(r, JP)
        if self._need_link():
            seen = {}
            for p, li in enumerate(self.linfo):
                art = li.art._objs[0]
                F = seen.get(id(art))
                if F is None:
                    F = seen[id(art)] = np.asarray(art.get_link_incoming_joint_forces(), float)
                f = li.R_LJ @ F[li.index, :3]
                t = li.R_LJ @ F[li.index, 3:] + np.cross(li.p_LJ, f)
                self._np_in[p, :3] += f * dt
                self._np_in[p, 3:] += t * dt
                for ch in li.children:
                    T = ch["T_L_ch0"] @ ch["T_ch_J"]
                    fc = T[:3, :3] @ F[ch["index"], :3]
                    tc = T[:3, :3] @ F[ch["index"], 3:] + np.cross(T[:3, 3], fc)
                    self._np_ch[p, :3] += fc * dt
                    self._np_ch[p, 3:] += tc * dt

    def excl_np(self, p, c) -> bool:
        if not hasattr(self, "_excl_np"):
            self._excl_np = self.excl.cpu().numpy()
        return bool(self._excl_np[p, c])

    def _pad_T_cpu(self, p) -> np.ndarray:
        from .sources import pose_np
        Tl = pose_np(self.links[p]._bodies[0].entity_pose)
        Tp = np.eye(4)
        Tp[:3, :3] = self.pads[p].rot()
        Tp[:3, 3] = self.pads[p].pos()
        return Tl @ Tp

    # ---------------------------------------------------------------- per control step
    def on_reset(self):
        self.setup()
        self._acc_zero()
        self._snap_vel()

    def _acc_zero(self):
        for t in (self.acc_n, self.acc_x, self.acc_y, self.acc_v, self.acc_off, self.acc_in, self.acc_ch):
            t.zero_()
        if not self.gpu:
            for a in (self._np_n, self._np_x, self._np_y, self._np_v, self._np_off, self._np_in, self._np_ch, self._np_offF,
                      self._np_offM):
                a[...] = 0.0
        self.n_sub = 0

    def _snap_vel(self):
        if self._need_link():
            self._v0 = torch.stack([l.linear_velocity.to(self.dev).float() for l in self.links], 1)
            self._w0 = torch.stack([l.angular_velocity.to(self.dev).float() for l in self.links], 1)

    def begin_step(self):
        if self._key != self._scene_key():
            self.setup()
        self._acc_zero()
        self._snap_vel()
        self._active = True

    def end_step(self) -> dict:
        import time
        self._active = False
        t0 = time.perf_counter()
        out = self._compute()
        self.timing["end_s"] += time.perf_counter() - t0
        self.timing["n_steps"] += 1
        self.last = out
        return out

    def zeros(self) -> dict:
        N, P, dev = self.N, self.P, self.dev
        out = dict(contact=torch.zeros((N, P), dtype=torch.bool, device=dev))
        if self.level >= 1:
            out.update(force=torch.zeros((N, P, 3), device=dev), cop=torch.zeros((N, P, 2), device=dev),
                       force_world=torch.zeros((N, P, 3), device=dev))
        if self.level >= 2:
            out.update(pressure=torch.zeros((N, P, self.nx, self.ny), device=dev),
                       shear=torch.zeros((N, P, self.nx, self.ny, 2), device=dev))
        return out

    def link_wrench(self, T: float, pad_R_W: torch.Tensor, Rl: torch.Tensor) -> torch.Tensor:
        """(N, P, 6) contact wrench on the pad link from Newton-Euler, in the PAD frame: force, torque about the pad origin"""
        g = torch.tensor(list(self.u.sim_config.scene_config.gravity), device=self.dev, dtype=torch.float32)
        v1 = torch.stack([l.linear_velocity.to(self.dev).float() for l in self.links], 1)
        w1 = torch.stack([l.angular_velocity.to(self.dev).float() for l in self.links], 1)
        RlT = Rl.transpose(-1, -2)
        m = torch.tensor([li.mass for li in self.linfo], device=self.dev)[None, :, None]
        com = torch.tensor(np.stack([li.com for li in self.linfo]), device=self.dev, dtype=torch.float32)[None]
        I = torch.tensor(np.stack([li.I for li in self.linfo]), device=self.dev, dtype=torch.float32)[None]
        dv = torch.einsum("npij,npj->npi", RlT, v1 - self._v0)
        dw = torch.einsum("npij,npj->npi", RlT, w1 - self._w0)
        gL = torch.einsum("npij,j->npi", RlT, g)
        Jin = self.acc_in if self.gpu else torch.tensor(self._np_in, device=self.dev, dtype=torch.float32)[None]
        Jch = self.acc_ch if self.gpu else torch.tensor(self._np_ch, device=self.dev, dtype=torch.float32)[None]
        F = (m * dv - m * gL * T - Jin[..., :3] + Jch[..., :3]) / T
        tau = (torch.einsum("npij,npj->npi", I.expand(dv.shape[0], -1, -1, -1), dw)
               + torch.linalg.cross(com.expand_as(dv), m * dv, dim=-1)
               - torch.linalg.cross(com.expand_as(dv), m * gL * T, dim=-1) - Jin[..., 3:] + Jch[..., 3:]) / T
        RP = self.pad_R_L[None].transpose(-1, -2)                                            # link -> pad
        Fp = torch.einsum("npij,npj->npi", RP.expand(F.shape[0], -1, -1, -1), F)
        tp = torch.einsum("npij,npj->npi", RP.expand(F.shape[0], -1, -1, -1),
                          tau - torch.linalg.cross(self.pad_o_L[None].expand_as(F), F, dim=-1))
        return torch.cat([Fp, tp], -1)

    def _compute(self) -> dict:
        u, N, P, C, K, dev = self.u, self.N, self.P, self.C, self.K, self.dev
        f32 = dict(device=dev, dtype=torch.float32)
        T = float(u.control_timestep)
        pos, quat, R_WP, Rl = self.pad_poses()
        RT = R_WP.transpose(-1, -2)
        excl = self.excl[None]
        half = torch.stack([self.W, self.L], -1)[None] / 2                                   # (1, P, 2)
        # ---- per-body normal force (and first moments on CPU)
        if self.gpu:
            Fv = torch.einsum("npij,npcj->npci", RT, self.acc_v) / T                       # force on the pad, pad frame
            Fv = torch.where(excl[..., None], torch.zeros_like(Fv), Fv)
            Fb = (-Fv[..., 2]).clamp_min(0.0)
            Mb = None
        else:
            Fb = torch.tensor(self._np_n, **f32)[None] / T
            Mb = torch.stack([torch.tensor(self._np_x, **f32)[None], torch.tensor(self._np_y, **f32)[None]], -1) / T
            Fv = torch.tensor(self._np_v, **f32)[None] / T
        Fn = Fb.sum(-1)
        contact = Fn > self.thr
        out = dict(contact=contact)
        diag = dict(Fb=Fb, Fn=Fn)
        tilt_ok = None
        if self.level >= 1:
            W = self.link_wrench(T, R_WP, Rl)                                               # (N, P, 6), pad frame
            diag["W_link"] = W
            if not self.gpu:
                offF = torch.tensor(self._np_offF, **f32)[None] / T
                offM = torch.tensor(self._np_offM, **f32)[None] / T
                Wp = W - torch.cat([offF, offM], -1)                                       # remove off-pad contact points
                Msum = Mb.sum(2)
                cop = torch.where(Fn[..., None] > 0, Msum / Fn.clamp_min(1e-12)[..., None], torch.zeros_like(Msum))
                # trust the link wrench (friction, twist) only if its normal force reproduces the contact points:
                # CPU PhysX reports no incoming joint force for contacts between links of the same articulation
                Fn_l = -Wp[..., 2]
                link_ok = (Fn_l - Fn).abs() <= 0.1 * Fn + 0.05
                Ft = torch.where(link_ok[..., None], Wp[..., :2], torch.zeros_like(Wp[..., :2]))
                Mz = torch.where(link_ok, Wp[..., 5], torch.zeros_like(Wp[..., 5]))
                diag.update(off_F=offF, off_M=offM, link_ok=link_ok)
            else:
                # PhysX GPU reports the incoming joint force of ONE contacting body per link (measured, SAPIEN 3.0.3):
                # attribute the link wrench to the body whose pair force it reproduces
                Fmag = Fv.norm(dim=-1)
                proj = (W[:, :, None, :3] * Fv).sum(-1) / Fmag.clamp_min(1e-9)
                mism = torch.where((Fmag > 1e-6) & (Fb > 0.2 * self.thr), (proj - Fmag).abs() / Fmag.clamp_min(1e-9),
                                   torch.full_like(Fmag, 1e9))
                best, star = mism.min(-1)
                link_ok = best < 0.1
                Fn_l = (-W[..., 2]).clamp_min(1e-9)
                cop_l = torch.stack([W[..., 4] / Fn_l, -W[..., 3] / Fn_l], -1)
                cop_l = torch.maximum(torch.minimum(cop_l, half), -half)
                # other bodies: projection of the body origin on the pad (refined from the pressure map at T2)
                bp = self.body_poses().reshape(C, N, 7).permute(1, 0, 2)
                rel = torch.einsum("npji,npcj->npci", R_WP, bp[:, None, :, :3] - pos[:, :, None, :])[..., :2]
                rel = torch.maximum(torch.minimum(rel, half[:, :, None]), -half[:, :, None])
                is_star = (torch.arange(C, device=dev)[None, None] == star[..., None]) & link_ok[..., None]
                cop_b = torch.where(is_star[..., None], cop_l[:, :, None, :], rel)
                Mb_est = Fb[..., None] * cop_b
                Msum = Mb_est.sum(2)
                cop = torch.where(Fn[..., None] > 0, Msum / Fn.clamp_min(1e-12)[..., None], torch.zeros_like(Msum))
                others = torch.where(is_star[..., None], torch.zeros_like(Fv), Fv).sum(2)
                Ft = torch.where(link_ok[..., None], W[..., :2] + others[..., :2], Fv[..., :2].sum(2))
                Mz = torch.where(link_ok, W[..., 5], torch.zeros_like(W[..., 5]))
                tilt_ok = is_star
                Mb = Mb_est
                n_bodies = (Fb > 0.2 * self.thr).sum(-1)
                diag.update(link_ok=link_ok, star=star, link_mismatch=best, n_bodies=n_bodies, cop_link=cop_l)
            Ft = torch.where(contact[..., None], Ft, torch.zeros_like(Ft))
            Mz = torch.where(contact, Mz, torch.zeros_like(Mz))
            cop = torch.where(contact[..., None], cop, torch.zeros_like(cop))
            diag.update(src_cop=cop, src_Ft=Ft, src_Mz=Mz)
            force = torch.cat([Ft, torch.where(contact, Fn, torch.zeros_like(Fn))[..., None]], -1)
            out.update(force=force, cop=cop)
        if self.level >= 2:
            kk = min(K, C)
            Fsel, csel = torch.topk(Fb, kk, dim=-1)                                              # (N, P, K)
            Fsel = torch.where(contact[..., None], Fsel, torch.zeros_like(Fsel))
            rows = csel * N + torch.arange(N, device=dev)[:, None, None]
            rows = torch.where(Fsel > 0, rows, torch.full_like(rows, -1))
            bp = self.body_poses()
            t = self.bank.cast(pos.reshape(-1, 3), quat.reshape(-1, 4), self.sub_local, rows.reshape(N * P, kk),
                               bp[:, :3], bp[:, 3:])                                             # (N*P, K, S)
            gap = torch.where(t < BIG / 2, t - self.backoff, torch.full_like(t, BIG))
            po = torch.arange(P, device=dev).repeat(N).repeat_interleave(kk)                     # problem -> pad
            Msel = torch.gather(Mb.expand(N, -1, -1, -1), 2, csel[..., None].expand(-1, -1, -1, 2))
            Msel = torch.where(Fsel[..., None] > 0, Msel, torch.zeros_like(Msel))
            if self.moments == "match":
                ok = torch.ones_like(Fsel, dtype=torch.bool) if tilt_ok is None else torch.gather(tilt_ok, 2, csel)
            else:
                ok = torch.zeros_like(Fsel, dtype=torch.bool)
            okf = ok.reshape(-1).float()
            f, info = self.solver(gap.reshape(N * P * kk, 1, -1), self.xs[po], self.ys[po], self.c[po],
                                  Fsel.reshape(-1, 1), Msel[..., 0].reshape(-1), Msel[..., 1].reshape(-1),
                                  self.amax[po] * okf, self.bmax[po] * okf, tilt=True)
            f = f.reshape(N, P, kk, -1)
            fpad = f.sum(2)                                                                      # (N, P, S)
            # bodies that touch the pad LINK but have no surface over the pad (e.g. squeezed out past its edge) do not
            # load the sensing area: drop them from the pad force (T0 / T1 follow)
            nos = info["no_surface"].reshape(N, P, kk) & (Fsel > 0)
            drop = (Fsel * nos).sum(-1)
            Fn2 = Fn - drop
            contact2 = Fn2 > self.thr
            out["contact"] = contact2
            if self.level >= 1:
                out["force"] = torch.where(contact2[..., None], torch.cat([out["force"][..., :2], Fn2[..., None]], -1),
                                           torch.zeros_like(out["force"]))
                out["cop"] = torch.where(contact2[..., None], out["cop"], torch.zeros_like(out["cop"]))
            fpad = torch.where(contact2[..., None], fpad, torch.zeros_like(fpad))
            diag.update(dropped_no_surface=drop, Fn_pad=Fn2)
            if self.gpu and self.level >= 1:
                # bodies solved geometrically: take the CoP from the map
                Fm = fpad.sum(-1)
                cop_map = torch.stack([(fpad * self.xs[None]).sum(-1), (fpad * self.ys[None]).sum(-1)], -1) / Fm.clamp_min(1e-12)[..., None]
                geo = (~ok & (Fsel > 0)).any(-1) & (Fm > 0)
                out["cop"] = torch.where(geo[..., None], cop_map, out["cop"])
            tsub = shear_field(fpad, self.xs[None], self.ys[None], out["force"][..., :2],
                               Mz if self.twist else None)
            A = self.taxel_area[None, :, None, None]
            out["pressure"] = bin_taxels(fpad, self.Mbin, self.nx, self.ny) / A
            out["shear"] = bin_taxels(tsub, self.Mbin, self.nx, self.ny) / A[..., None]
            diag.update(fsub=fpad, tsub=tsub, gap=gap.reshape(N, P, kk, -1), sel=csel, Fsel=Fsel, Msel=Msel, tilt_ok=ok,
                        solver={k: (v.reshape(N, P, kk) if v.dim() == 2 and v.shape[-1] == 1 else v) for k, v in info.items()})
        if self.level >= 1:
            fw = torch.einsum("npij,npj->npi", R_WP, torch.cat([out["force"][..., :2], -out["force"][..., 2:3]], -1))
            out["force_world"] = fw
        if self.diagnostics:
            out["_diag"] = diag
        return out


class TactileWrapper:
    """gymnasium-style wrapper adding obs['tactile'] (dict of tensors, see the module docstring). Dict observations get
    the key added; flat state observations become {'state': obs, 'tactile': ...}. Everything else is passed through
    (env.unwrapped, render, close, attributes)."""

    def __init__(self, env, level="T2", **sensor_kw):
        import gymnasium as gym
        self.env = env
        self.sensor = TactileSensor(env, level=level, **sensor_kw)
        self.level = LEVELS[self.sensor.level]
        self._space = None
        self._gym = gym

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def _merge(self, obs, tac):
        if isinstance(obs, dict):
            obs = dict(obs)
            obs["tactile"] = tac
            return obs
        return {"state": obs, "tactile": tac}

    def tactile_space(self):
        gym = self._gym
        s = self.sensor
        N, P = s.N, s.P
        d = {"contact": gym.spaces.Box(0, 1, (N, P), np.bool_)}
        if s.level >= 1:
            d["force"] = gym.spaces.Box(-np.inf, np.inf, (N, P, 3), np.float32)
            d["cop"] = gym.spaces.Box(-np.inf, np.inf, (N, P, 2), np.float32)
            d["force_world"] = gym.spaces.Box(-np.inf, np.inf, (N, P, 3), np.float32)
        if s.level >= 2:
            d["pressure"] = gym.spaces.Box(0, np.inf, (N, P, s.nx, s.ny), np.float32)
            d["shear"] = gym.spaces.Box(-np.inf, np.inf, (N, P, s.nx, s.ny, 2), np.float32)
        return gym.spaces.Dict(d)

    @property
    def observation_space(self):
        gym = self._gym
        base = self.env.observation_space
        if self._space is None:
            self.sensor.setup()
            ts = self.tactile_space()
            if isinstance(base, gym.spaces.Dict):
                self._space = gym.spaces.Dict({**base.spaces, "tactile": ts})
            else:
                self._space = gym.spaces.Dict({"state": base, "tactile": ts})
        return self._space

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self.sensor.on_reset()
        return self._merge(obs, self.sensor.zeros()), info

    def step(self, action):
        self.sensor.begin_step()
        try:
            obs, rew, term, trunc, info = self.env.step(action)
        finally:
            self.sensor._active = False
        tac = self.sensor.end_step()
        return self._merge(obs, tac), rew, term, trunc, info

    def close(self):
        return self.env.close()


def attach_tactile(env, level="T2", **kw) -> TactileWrapper:
    return TactileWrapper(env, level=level, **kw)
