"""Protocol-v2 task bindings for the three rigid ManiSkill tasks (probes, contact pairs, release rule).

  press_juicer_v2     Imitator TwoRobotPressJuicer-v1   (venv ms2)
  place_cup_plate_v2  Imitator TwoRobotPlaceCupPlate-v1 (venv ms2)
  poke_cube_v2        ManiSkill 3.0.1 PokeCube-v1       (venv msup)

Each binding gives: probes() (per-control-step state probes), pairs() (substep-accumulated contact pairs for
adapters/maniskill_v2.TraceRecorderV2), release() (protocol_v2.ReleaseRule or None) and preconditions(u) (facts the
contract assumes, checked at reset and stored in the trace meta).

Geometry used by place_cup_plate_v2 (all derived from the collision meshes, nothing hand-measured):
  TrayFloor   top-surface height map of the tray on a 2 mm grid, from its convex collision parts (exact upper hull of
              each part). Floor reference height = median height of the central quarter of the tray. Inner floor area =
              cells connected to the tray centre whose height <= floor reference + FLOOR_BAND_M (the raised rim and its
              inner slope are higher).
  CupBase     base footprint = convex hull (in the cup's base plane) of the collision vertices within BASE_BAND_M of the
              cup's lowest local height, filled with a 2 mm grid of sample points; handles are above that band.
  signals     cup<i>_bottom_dz_m        |lowest point of the cup - tray floor height under it|  (floor reference height
                                        when the lowest point is not over the floor area)
              cup<i>_base_in_floor_frac fraction of the base footprint samples that lie over the inner floor area
"""
from __future__ import annotations

import os

import numpy as np

from ..outcomes.metrics import measure
from .maniskill_v2 import Pair, TraceRecorderV2
from .protocol_v2 import RELEASE_GAP_M, RELEASE_MAX_IMPULSE_NS, ReleaseRule

TASKS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tasks")   # veritask/tasks

GRID_M = 0.002        # tray height-map / base-footprint sampling resolution
FLOOR_BAND_M = 0.003  # a tray cell is "floor" if its top surface is within this of the floor reference height
BASE_BAND_M = 0.005   # collision vertices within this of the cup's lowest local height form the base ring


# ------------------------------------------------------------------------------------------------ small helpers
def _quat_mat(q):
    from scipy.spatial.transform import Rotation as R
    return R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def _pose_Rp(obj):
    raw = obj.pose.raw_pose[0].cpu().numpy()
    return _quat_mat(raw[3:]), raw[:3].astype(float)


def collision_parts_local(obj) -> list[np.ndarray]:
    """Vertices of each collision shape of an Actor / Link, in the body frame."""
    from mani_skill.utils.geometry.trimesh_utils import get_component_meshes
    return [np.asarray(m.vertices, float) for m in get_component_meshes(obj._bodies[0])]


def _upper_hull_height(V: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Top-surface height of conv(V) above each (x, y); NaN where (x, y) is outside its xy projection."""
    from scipy.spatial import ConvexHull
    eq = ConvexHull(V).equations                      # n . p + d <= 0 inside
    a, b, c, d = eq[:, 0], eq[:, 1], eq[:, 2], eq[:, 3]
    lin = a[:, None] * X.ravel()[None] + b[:, None] * Y.ravel()[None] + d[:, None]   # (facets, points)
    up, dn, side = c > 1e-9, c < -1e-9, np.abs(c) <= 1e-9
    top = np.min(-lin[up] / c[up][:, None], axis=0) if up.any() else np.full(lin.shape[1], np.inf)
    bot = np.max(-lin[dn] / c[dn][:, None], axis=0) if dn.any() else np.full(lin.shape[1], -np.inf)
    ok = top >= bot - 1e-9
    if side.any():
        ok &= (lin[side] <= 1e-9).all(axis=0)
    return np.where(ok, top, np.nan).reshape(X.shape)


class TrayFloor:
    """Height map of the tray's top surface in the world frame and its inner floor area (see module docstring)."""

    def __init__(self, parts_world: list[np.ndarray], grid: float = GRID_M, band: float = FLOOR_BAND_M):
        from scipy import ndimage
        allv = np.vstack(parts_world)
        lo, hi = allv.min(0), allv.max(0)
        self.grid, self.band = grid, band
        self.x0, self.y0 = lo[0] - grid, lo[1] - grid
        nx, ny = int(np.ceil((hi[0] - lo[0]) / grid)) + 3, int(np.ceil((hi[1] - lo[1]) / grid)) + 3
        xs, ys = self.x0 + grid * np.arange(nx), self.y0 + grid * np.arange(ny)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        H = np.full(X.shape, np.nan)
        for V in parts_world:
            h = _upper_hull_height(V, X, Y)
            H = np.where(np.isnan(H), h, np.where(np.isnan(h), H, np.maximum(H, h)))
        self.H = H
        cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
        central = (np.abs(X - cx) <= (hi[0] - lo[0]) / 4) & (np.abs(Y - cy) <= (hi[1] - lo[1]) / 4) & ~np.isnan(H)
        self.floor_ref = float(np.median(H[central]))
        cand = ~np.isnan(H) & (H <= self.floor_ref + band)
        lab, _ = ndimage.label(cand)
        ci, cj = self._cell(np.array([[cx, cy]]))
        self.mask = (lab == lab[ci[0], cj[0]]) & cand if lab[ci[0], cj[0]] else cand
        m = self.mask
        self.floor_bounds = [[float(X[m].min()), float(Y[m].min())], [float(X[m].max()), float(Y[m].max())]]
        self.floor_area_m2 = float(m.sum() * grid * grid)
        self.floor_height_range = [float(np.nanmin(H[m])), float(np.nanmax(H[m]))]
        self.rim_top = float(np.nanmax(H))

    def _cell(self, xy: np.ndarray):
        i = np.rint((xy[:, 0] - self.x0) / self.grid).astype(int)
        j = np.rint((xy[:, 1] - self.y0) / self.grid).astype(int)
        return i, j

    def on_floor(self, xy: np.ndarray) -> np.ndarray:
        i, j = self._cell(np.atleast_2d(xy))
        inside = (i >= 0) & (i < self.mask.shape[0]) & (j >= 0) & (j < self.mask.shape[1])
        out = np.zeros(len(i), bool)
        out[inside] = self.mask[i[inside], j[inside]]
        return out

    def floor_height(self, xy) -> float:
        """Floor height under one point (floor reference height if the point is not over the floor area)."""
        xy = np.atleast_2d(xy)
        if self.on_floor(xy)[0]:
            i, j = self._cell(xy)
            return float(self.H[i[0], j[0]])
        return self.floor_ref

    def summary(self) -> dict:
        return dict(floor_ref_z=self.floor_ref, floor_bounds_xy=self.floor_bounds, floor_area_m2=self.floor_area_m2,
                    floor_height_range=self.floor_height_range, rim_top_z=self.rim_top, grid_m=self.grid, band_m=self.band)


class CupBase:
    """Base footprint samples and all collision vertices of a cup, in the cup frame (up = local +y, as in v1)."""

    def __init__(self, parts_local: list[np.ndarray], grid: float = GRID_M, band: float = BASE_BAND_M):
        from scipy.spatial import ConvexHull, Delaunay
        V = np.vstack(parts_local)
        self.V = V
        self.y_min = float(V[:, 1].min())
        ring = V[V[:, 1] <= self.y_min + band][:, [0, 2]]
        hull = ConvexHull(ring)
        poly = ring[hull.vertices]
        lo, hi = poly.min(0), poly.max(0)
        gx, gz = np.meshgrid(np.arange(lo[0], hi[0] + grid / 2, grid), np.arange(lo[1], hi[1] + grid / 2, grid), indexing="ij")
        pts = np.c_[gx.ravel(), gz.ravel()]
        pts = pts[Delaunay(poly).find_simplex(pts) >= 0]
        pts = np.vstack([pts, poly])                                   # interior grid + the outline itself
        self.samples = np.c_[pts[:, 0], np.full(len(pts), self.y_min), pts[:, 1]]
        self.base_area_m2 = float(hull.volume)
        self.base_radius_m = float(np.sqrt(hull.volume / np.pi))
        self.base_centre = np.array([poly[:, 0].mean(), self.y_min, poly[:, 1].mean()])

    def summary(self) -> dict:
        return dict(n_samples=int(len(self.samples)), base_area_m2=self.base_area_m2, base_equiv_radius_m=self.base_radius_m,
                    base_centre_local=self.base_centre.round(5).tolist(), height_m=float(self.V[:, 1].max() - self.y_min))


# ------------------------------------------------------------------------------------------------ bindings
class _Binding:
    task_id = env_id = venv = ""

    @property
    def contract_file(self) -> str:
        return os.path.join(TASKS, f"{self.task_id}.json")

    def probes(self) -> dict:
        return {}

    def pairs(self) -> dict[str, Pair]:
        return {}

    def release(self) -> ReleaseRule | None:
        return None

    def preconditions(self, u) -> dict:
        return {}

    def recorder(self, env, v1_probes: dict | None = None) -> TraceRecorderV2:
        """Recorder with this binding's signals. v1_probes (optional) are recorded first, under their own names, so
        the v1 contract can be judged on the same rollout; v2 probes with the same name must be the same quantity."""
        probes = dict(v1_probes or {})
        probes.update(self.probes())
        return TraceRecorderV2(env, probes, pairs=self.pairs(), release=self.release())


def _all_robot_links(u) -> list:
    agents = list(u.agent.agents) if hasattr(u.agent, "agents") else [u.agent]
    return [l for ag in agents for l in ag.robot.links if l._bodies[0].get_collision_shapes()]


def _finger_links(u) -> list:
    agents = list(u.agent.agents) if hasattr(u.agent, "agents") else [u.agent]
    return [f for ag in agents for f in (ag.finger1_link, ag.finger2_link)]


class PressJuicerV2(_Binding):
    task_id, env_id, venv = "press_juicer_v2", "TwoRobotPressJuicer-v1", "ms2"

    def probes(self):
        return {"knob_qpos": lambda u: float(u.partnet_link.joint.qpos.flatten()[0])}

    def pairs(self):
        knob = lambda u: [u.partnet_link]  # noqa: E731
        return {"robot_knob": Pair(_all_robot_links, knob, gap=True),
                "fingers_knob": Pair(_finger_links, knob, gap=True)}

    def release(self):
        return ReleaseRule(contact_signals=["robot_knob_impulse_Ns"], gap_signals=["fingers_knob_gap_m"],
                           gap_m=RELEASE_GAP_M, max_impulse_Ns=RELEASE_MAX_IMPULSE_NS)

    def preconditions(self, u):
        """The contract assumes a knob WITHOUT a return spring (it stays where it is left)."""
        j = u.partnet_link.joint
        out = dict(knob_joint_stiffness=float(np.asarray(j.stiffness.cpu() if hasattr(j.stiffness, "cpu") else j.stiffness).flatten()[0]),
                   knob_joint_damping=float(np.asarray(j.damping.cpu() if hasattr(j.damping, "cpu") else j.damping).flatten()[0]),
                   knob_joint_friction=float(np.asarray(j.friction.cpu() if hasattr(j.friction, "cpu") else j.friction).flatten()[0]),
                   env_target_qpos=float(np.asarray(u.target_qpos.cpu() if hasattr(u.target_qpos, "cpu") else u.target_qpos).flatten()[0]))
        out["ok"] = out["knob_joint_stiffness"] == 0.0
        return out


class PokeCubeV2(_Binding):
    task_id, env_id, venv = "poke_cube_v2", "PokeCube-v1", "msup"

    def probes(self):
        def half(u):
            h = u.cube_half_size
            return float(h[0]) if hasattr(h, "__len__") else float(h)
        return {
            "cube_goal_xy": lambda u: float(np.linalg.norm((u.cube.pose.p[0, :2] - u.goal_region.pose.p[0, :2]).cpu().numpy())),
            "cube_speed": lambda u: float(np.linalg.norm(u.cube.linear_velocity[0].cpu().numpy())),
            "cube_lift_m": lambda u: float(abs(float(u.cube.pose.p[0, 2]) - half(u))),
        }

    def pairs(self):
        cube = lambda u: [u.cube]  # noqa: E731
        return {"robot_cube": Pair(_all_robot_links, cube, gap=True, cumulative=True),
                "peg_cube": Pair(lambda u: [u.peg], cube)}

    def preconditions(self, u):
        return dict(goal_radius=float(u.goal_radius), cube_half_size=float(u.cube_half_size),
                    cube_mass=float(u.cube._bodies[0].mass), ok=True)


class PlaceCupPlateV2(_Binding):
    task_id, env_id, venv = "place_cup_plate_v2", "TwoRobotPlaceCupPlate-v1", "ms2"

    def __init__(self):
        self._cache: dict = {}

    def geometry(self, u) -> dict:
        """Per-episode geometry (actors are rebuilt on every reset; the tray is static but may be re-posed)."""
        Rt, pt = _pose_Rp(u.tray)
        key = (id(u.tray._objs[0]), tuple(np.round(np.r_[pt, Rt.ravel()], 6)))
        if self._cache.get("key") != key:
            tray = TrayFloor([(Rt @ V.T).T + pt for V in collision_parts_local(u.tray)])
            cups = [CupBase(collision_parts_local(c)) for c in (u.cup_1, u.cup_2)]
            self._cache = dict(key=key, tray=tray, cups=cups)
        return self._cache

    def cup_state(self, u, i: int) -> tuple[float, float]:
        """(bottom_dz_m, base_in_floor_frac) of cup i (1 or 2) at the current state."""
        g = self.geometry(u); tray, cb = g["tray"], g["cups"][i - 1]
        R, p = _pose_Rp((u.cup_1, u.cup_2)[i - 1])
        W = (R @ cb.V.T).T + p
        k = int(np.argmin(W[:, 2]))
        dz = abs(float(W[k, 2]) - tray.floor_height(W[k, :2]))
        S = (R @ cb.samples.T).T + p
        return dz, float(tray.on_floor(S[:, :2]).mean())

    def probes(self):
        S: dict = {}

        def tilt(a):
            return measure("rigid.tilt_deg", up_axis_world=_pose_Rp(a)[0][:, 1])   # model up = local +y (as in v1)

        def state(u, i):
            S[i] = self.cup_state(u, i); return S[i][0]
        return {
            "cup1_tilt_deg": lambda u: tilt(u.cup_1),
            "cup2_tilt_deg": lambda u: tilt(u.cup_2),
            "cup1_bottom_dz_m": lambda u: state(u, 1),
            "cup1_base_in_floor_frac": lambda u: S[1][1],
            "cup2_bottom_dz_m": lambda u: state(u, 2),
            "cup2_base_in_floor_frac": lambda u: S[2][1],
            "cup_max_speed": lambda u: float(max(np.linalg.norm(u.cup_1.linear_velocity[0].cpu().numpy()),
                                                 np.linalg.norm(u.cup_2.linear_velocity[0].cpu().numpy()))),
        }

    def pairs(self):
        cups = lambda u: [u.cup_1, u.cup_2]  # noqa: E731
        return {"robot_cups": Pair(_all_robot_links, cups, gap=True),
                "fingers_cups": Pair(_finger_links, cups, gap=True)}

    def release(self):
        return ReleaseRule(contact_signals=["robot_cups_impulse_Ns"], gap_signals=["fingers_cups_gap_m"],
                           gap_m=RELEASE_GAP_M, max_impulse_Ns=RELEASE_MAX_IMPULSE_NS)

    def preconditions(self, u):
        g = self.geometry(u)
        return dict(tray=g["tray"].summary(), cups=[c.summary() for c in g["cups"]], ok=True)


BINDINGS = {b.task_id: b for b in (PressJuicerV2, PokeCubeV2, PlaceCupPlateV2)}
