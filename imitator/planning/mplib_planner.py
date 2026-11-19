"""mplib (0.1.1) planning for one arm of a dual configuration.

ArmPlanner(agent, base_pose)
  * planning model = the record's planning URDF (convex hulls) + full SRDF; move group = record.tcp_link
  * table as an FCL box (top 3 mm below the real top, so the base links resting on it are not in collision)
  * the OTHER arm (and anything else) as a point cloud: `set_obstacle_points(points)`; `arm_points(agent)` samples the
    surface of every collision hull of an agent at its current configuration
  * ik(pose_world) / plan(pose_world, method='screw' | 'rrt') / plan_qpos(q_goal)
Results are arm-joint position sequences at the env's control timestep (TOPP-RA), to be sent as pd_joint_pos targets.
"""
from __future__ import annotations

import numpy as np
import sapien

import mplib
from mplib.pymp.collision_detection import fcl

from ..embodiments import dual as D
from ..embodiments.urdf_tools import load_urdf


def pose7(p: sapien.Pose) -> np.ndarray:
    return np.concatenate([np.asarray(p.p, float), np.asarray(p.q, float)])


class ArmPlanner:
    def __init__(self, agent, base_pose: sapien.Pose, control_dt: float = 0.05, vel: float = 0.9, acc: float = 0.9,
                 table: bool = True, env_idx: int = 0):
        r = agent.record
        self.agent, self.record, self.dt = agent, r, control_dt
        self.env_idx = int(env_idx)          # which parallel env's state qpos() reads (GPU sim: one planner per env)
        self.link_names = [l.name for l in agent.robot.get_links()]
        self.joint_names = [j.name for j in agent.robot.active_joints]
        self.arm_idx = np.array([self.joint_names.index(j) for j in r.arm_joints])
        home = r.home_qpos()
        gj = [j for j in self.joint_names if j not in r.arm_joints]
        self.grip_idx = np.array([self.joint_names.index(j) for j in gj], dtype=int)
        self.grip_open = np.array([home.get(j, 0.0) for j in gj], dtype=np.float64)
        n = len(r.arm_joints)
        self.planner = mplib.Planner(urdf=r.mplib_urdf_path, srdf=r.mplib_srdf_path, user_link_names=self.link_names,
                                     user_joint_names=self.joint_names, move_group=r.tcp_link,
                                     joint_vel_limits=np.full(n, vel), joint_acc_limits=np.full(n, acc))
        self.base_pose = base_pose
        self.planner.set_base_pose(pose7(base_pose))
        self.points = None
        if table:
            self.add_table()

    # ---------------------------------------------------------------- world
    def add_table(self, top_z: float = D.TABLE_TOP_Z - 0.003):
        sx, sy, sz = D.TABLE_X[1] - D.TABLE_X[0], D.TABLE_Y[1] - D.TABLE_Y[0], 0.2
        c = np.array([(D.TABLE_X[1] + D.TABLE_X[0]) / 2, (D.TABLE_Y[1] + D.TABLE_Y[0]) / 2, top_z - sz / 2])
        self.planner.set_normal_object("table", fcl.CollisionObject(fcl.Box(sx, sy, sz), c, np.array([1.0, 0, 0, 0])))

    def set_obstacle_points(self, pts: np.ndarray | None, radius: float = 0.01):
        self.points = None if pts is None or len(pts) == 0 else np.asarray(pts, float)
        if self.points is not None:
            self.planner.update_point_cloud(self.points, radius=radius)

    @property
    def use_pc(self) -> bool:
        return self.points is not None

    def attach_box(self, size, pose_in_tcp: sapien.Pose):
        """attach a box to the move-group link (TCP) for collision checking (e.g. the grasped cube)"""
        self.planner.update_attached_box(np.asarray(size, float), pose7(pose_in_tcp),
                                         link_id=self.planner.link_name_2_idx[self.record.tcp_link])
        self.attached = True

    def detach(self):
        self.attached = False

    # ---------------------------------------------------------------- queries
    def qpos(self) -> np.ndarray:
        """current configuration for planning, with the gripper / hand joints set to their OPEN values: a closed hand's
        fingers touch each other (mplib would call every state a self-collision) and the grasped object is represented
        by the attached box instead"""
        q = self.agent.robot.get_qpos()[self.env_idx].cpu().numpy().astype(np.float64)
        q[self.grip_idx] = self.grip_open
        return q

    def ik(self, pose_world: sapien.Pose, q_start=None, n_init: int = 20):
        goal = self.planner.transform_goal_to_wrt_base(pose7(pose_world))
        self.planner.planning_world.set_use_point_cloud(self.use_pc)     # IK collision checks see the other arm too
        self.planner.planning_world.set_use_attach(False)
        st, res = self.planner.IK(goal, self.qpos() if q_start is None else np.asarray(q_start, float), n_init_qpos=n_init)
        return st == "Success", res

    def fk_tcp(self, q_full) -> np.ndarray:
        pin = self.planner.pinocchio_model
        pin.compute_forward_kinematics(np.asarray(q_full, float))
        p = pin.get_link_pose(self.planner.link_name_2_idx[self.record.tcp_link])
        T = np.eye(4); T[:3, 3] = p[:3]
        from transforms3d.quaternions import quat2mat
        T[:3, :3] = quat2mat(p[3:])
        B = self.base_pose.to_transformation_matrix()
        return B @ T

    def plan(self, pose_world: sapien.Pose, method: str = "screw", q_start=None, attach: bool = False,
             planning_time: float = 2.0):
        q = self.qpos() if q_start is None else np.asarray(q_start, float)
        tgt = pose7(pose_world)
        if method == "screw":
            res = self.planner.plan_screw(tgt, q, time_step=self.dt, use_point_cloud=self.use_pc, use_attach=attach)
            if res["status"] != "Success":
                res = self.planner.plan_screw(tgt, q, time_step=self.dt, use_point_cloud=self.use_pc, use_attach=attach)
        else:
            res = self.planner.plan_qpos_to_pose(tgt, q, time_step=self.dt, use_point_cloud=self.use_pc,
                                                 use_attach=attach, planning_time=planning_time, wrt_world=True)
        return res

    def cartesian(self, pose_world: sapien.Pose, q_start=None, speed: float = 0.1, min_steps: int = 5):
        """straight-line TCP motion (position lerp, orientation slerp) by incremental damped IK (pinocchio CLIK) from the
        previous solution; NO collision checking (short moves next to the object: approach, lift, place, retreat).
        Returns a plan dict like mplib's (status, position = arm joint path at the control rate)."""
        from transforms3d.quaternions import mat2quat, quat2mat, qmult, qinverse
        q = (self.qpos() if q_start is None else np.asarray(q_start, float)).copy()
        T0 = self.fk_tcp(q)
        p0, p1 = T0[:3, 3], np.asarray(pose_world.p, float)
        q0r, q1r = mat2quat(T0[:3, :3]), np.asarray(pose_world.q, float)
        if np.dot(q0r, q1r) < 0:
            q1r = -q1r
        dist = float(np.linalg.norm(p1 - p0))
        ang = 2 * np.arccos(np.clip(abs(np.dot(q0r, q1r)), -1, 1))
        n = max(min_steps, int(np.ceil(max(dist / (speed * self.dt), ang / (1.0 * self.dt)))))
        pin = self.planner.pinocchio_model
        idx = self.planner.link_name_2_idx[self.record.tcp_link]
        Binv = np.linalg.inv(self.base_pose.to_transformation_matrix())
        lims = np.concatenate(pin.get_joint_limits())
        out = []
        for k in range(1, n + 1):
            t = k / n
            p = (1 - t) * p0 + t * p1
            # slerp
            d = float(np.clip(np.dot(q0r, q1r), -1, 1)); th = np.arccos(d)
            qr = q1r if th < 1e-6 else (np.sin((1 - t) * th) * q0r + np.sin(t * th) * q1r) / np.sin(th)
            T = np.eye(4); T[:3, :3] = quat2mat(qr); T[:3, 3] = p
            Tb = Binv @ T
            goal = np.concatenate([Tb[:3, 3], mat2quat(Tb[:3, :3])])
            res = pin.compute_IK_CLIK(idx, goal, q, [])
            qn, err = np.asarray(res[0]), np.linalg.norm(res[2])
            if err > 2e-3:
                return dict(status=f"cartesian IK failed at {t:.2f} (err {err:.4f})")
            if (qn < lims[:, 0] - 1e-4).any() or (qn > lims[:, 1] + 1e-4).any():
                return dict(status=f"cartesian path leaves joint limits at {t:.2f}")
            if np.abs(qn - q)[self.arm_idx].max() > 0.3:
                return dict(status=f"cartesian path jumps at {t:.2f}")
            q = qn
            out.append(q[self.arm_idx].copy())
        return dict(status="Success", position=np.array(out))

    def track(self, points_world: np.ndarray, R_world: np.ndarray, q_start, max_jump: float = 0.3,
              tol: float = 2e-3):
        """incremental CLIK through a sequence of TCP positions (one per control step) with a fixed world orientation,
        from the full configuration q_start; NO collision checking. Returns (arm joint path (n, dof), None) or
        (None, reason)."""
        from transforms3d.quaternions import mat2quat
        pin = self.planner.pinocchio_model
        idx = self.planner.link_name_2_idx[self.record.tcp_link]
        Binv = np.linalg.inv(self.base_pose.to_transformation_matrix())
        lims = np.concatenate(pin.get_joint_limits())
        q = np.asarray(q_start, float).copy()
        out = []
        for k, p in enumerate(points_world):
            T = np.eye(4); T[:3, :3] = R_world; T[:3, 3] = p
            Tb = Binv @ T
            res = pin.compute_IK_CLIK(idx, np.concatenate([Tb[:3, 3], mat2quat(Tb[:3, :3])]), q, [])
            qn, err = np.asarray(res[0]), float(np.linalg.norm(res[2]))
            if err > tol:
                return None, f"IK error {err:.4f} at step {k}"
            if (qn < lims[:, 0] - 1e-4).any() or (qn > lims[:, 1] + 1e-4).any():
                return None, f"joint limits at step {k}"
            if np.abs(qn - q)[self.arm_idx].max() > max_jump and k > 0:
                return None, f"jump at step {k}"
            q = qn
            out.append(q[self.arm_idx].copy())
        return np.array(out), None

    def joint_interp(self, pose_world: sapien.Pose, q_start=None, max_step: float = 0.03):
        """IK (solution closest to the current configuration) + linear joint-space interpolation, NO collision check
        along the path (IK itself is collision-checked against the table / obstacle cloud)."""
        q = (self.qpos() if q_start is None else np.asarray(q_start, float)).copy()
        ok, sols = self.ik(pose_world, q_start=q, n_init=20)
        if not ok:
            return dict(status="joint_interp: IK failed")
        best = min(sols, key=lambda s: np.abs((np.asarray(s) - q)[self.arm_idx]).sum())
        g = np.asarray(best)[self.arm_idx]
        a = q[self.arm_idx]
        n = max(2, int(np.ceil(np.abs(g - a).max() / max_step)))
        return dict(status="Success", position=np.array([a + (g - a) * k / n for k in range(1, n + 1)]))

    def plan_qpos(self, q_goal_arm, q_start=None, attach: bool = False, planning_time: float = 2.0):
        q = self.qpos() if q_start is None else np.asarray(q_start, float)
        g = q.copy(); g[self.arm_idx] = q_goal_arm
        return self.planner.plan_qpos_to_qpos([g], q, time_step=self.dt, use_point_cloud=self.use_pc, use_attach=attach,
                                              planning_time=planning_time)


# ------------------------------------------------------------------------------------------------ point clouds
_HULL_PTS: dict = {}


def _link_surface_points(urdf: str, link: str, n: int = 160) -> np.ndarray:
    key = (urdf, link)
    if key not in _HULL_PTS:
        m = load_urdf(urdf)
        V = m.collision_points(link)
        if len(V) < 4:
            _HULL_PTS[key] = V
        else:
            import trimesh
            try:
                h = trimesh.Trimesh(V).convex_hull
                pts, _ = trimesh.sample.sample_surface_even(h, n, seed=0)
                _HULL_PTS[key] = np.vstack([pts, h.vertices])
            except Exception:
                _HULL_PTS[key] = V
    return _HULL_PTS[key]


def arm_points(agent, env_idx: int = 0) -> np.ndarray:
    """surface samples of every collision hull of `agent` at its current pose (world frame)"""
    r = agent.record
    pts = []
    for link in agent.robot.get_links():
        P = _link_surface_points(r.urdf_path, link.name)
        if not len(P):
            continue
        T = link.pose.to_transformation_matrix()[env_idx].cpu().numpy()
        pts.append((T[:3, :3] @ P.T).T + T[:3, 3])
    return np.vstack(pts) if pts else np.zeros((0, 3))
