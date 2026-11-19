"""v3.2 recorders for the ManiSkill Panda tasks (E29b): the generic v3.2 signals (imitator.verify.v32.build_recorder) plus
the task-specific geometry each contract reads. Every probe reads env 0 (CPU PhysX, one env per process).

  PickCube       cube_goal_dist_m        |cube centre - goal|
  StackCube      ab_xy_m, ab_dz_err_m     |xy(A) - xy(B)|, | z(A) - z(B) - 2h |;  pair cubeA_cubeB (support contact)
  PushCube/Pull  goal_dist_xy_m          |xy(cube) - xy(goal)|
  PlaceSphere    sphere_bin_xy_m, sphere_floor_dz_err_m   |xy(sphere) - xy(bin)|, |(z - r) - bin floor top|; pair sphere_bin
  StackPyramid   ab_xy_m, c_dz_err_m       |xy(A) - xy(B)|, |z(C) - 3h|;  pairs cubeC_cubeA, cubeC_cubeB
  LiftPegUpright peg_tilt_deg (local x = the peg's long axis to vertical), peg_bottom_dz_m
"""
from __future__ import annotations

import numpy as np

from .v32 import Obj, build_recorder, table_top_z


def _pair(a: str, b: str):
    from veritask.adapters.maniskill_v3 import PairV3
    return PairV3(lambda u: [getattr(u, a)], lambda u: [getattr(u, b)])


def _p(u, attr):
    return getattr(u, attr).pose.p[0].cpu().numpy().astype(float)


def pick_cube(env):
    from .v32 import EnclosureObj
    objs = {"cube": Obj("cube", half=0.02)}
    probes = {"cube_goal_dist_m": lambda u: float(np.linalg.norm(_p(u, "cube") - _p(u, "goal_site"))),
              "cube_held_enclosure_m": EnclosureObj(objs["cube"])}
    return build_recorder(env, objs, release=[], extra_probes=probes)


def stack_cube(env):
    objs = {"cubeA": Obj("cubeA", half=0.02), "cubeB": Obj("cubeB", half=0.02)}
    probes = {"ab_xy_m": lambda u: float(np.linalg.norm(_p(u, "cubeA")[:2] - _p(u, "cubeB")[:2])),
              "ab_dz_err_m": lambda u: float(abs(_p(u, "cubeA")[2] - _p(u, "cubeB")[2] - 0.04))}
    return build_recorder(env, objs, release=["cubeA", "cubeB"], extra_probes=probes,
                          extra_pairs={"cubeA_cubeB": _pair("cubeA", "cubeB")})


def _goal_xy(u):
    return float(np.linalg.norm(_p(u, "obj")[:2] - _p(u, "goal_region")[:2]))


def push_cube(env):
    objs = {"cube": Obj("obj", half=0.02)}
    return build_recorder(env, objs, release=["cube"], extra_probes={"goal_dist_xy_m": _goal_xy})


pull_cube = push_cube


def place_sphere(env):
    objs = {"sphere": Obj("obj", half=0.02)}

    def floor_err(u):
        r = float(u.radius)
        floor = _p(u, "bin")[2] + float(u.block_half_size[0])
        return float(abs(_p(u, "obj")[2] - r - floor))
    probes = {"sphere_bin_xy_m": lambda u: float(np.linalg.norm(_p(u, "obj")[:2] - _p(u, "bin")[:2])),
              "sphere_floor_dz_err_m": floor_err}
    return build_recorder(env, objs, release=["sphere"], extra_probes=probes,
                          extra_pairs={"sphere_bin": _pair("obj", "bin")})


def stack_pyramid(env):
    objs = {n: Obj(n, half=0.02) for n in ("cubeA", "cubeB", "cubeC")}
    probes = {"ab_xy_m": lambda u: float(np.linalg.norm(_p(u, "cubeA")[:2] - _p(u, "cubeB")[:2])),
              "c_dz_err_m": lambda u: float(abs(_p(u, "cubeC")[2] - table_top_z(u) - 0.06))}
    return build_recorder(env, objs, release=["cubeA", "cubeB", "cubeC"], extra_probes=probes,
                          extra_pairs={"cubeC_cubeA": _pair("cubeC", "cubeA"), "cubeC_cubeB": _pair("cubeC", "cubeB")})


def lift_peg_upright(env):
    objs = {"peg": Obj("peg", half=(0.12, 0.025, 0.025), tilt_axis=0)}
    return build_recorder(env, objs, release=["peg"])


def _plug_geom(u):
    import sapien
    from transforms3d.euler import euler2quat
    g = u.receptacle.pose.sp * sapien.Pose(q=euler2quat(0, 0, np.pi))
    c = u.charger.pose.sp
    Rg, Rc = g.to_transformation_matrix()[:3, :3], c.to_transformation_matrix()[:3, :3]
    ang = float(np.degrees(np.arccos(np.clip((np.trace(Rg.T @ Rc) - 1) / 2, -1, 1))))
    return float(np.linalg.norm(np.asarray(g.p) - np.asarray(c.p))), ang


def plug_charger(env):
    """charger_goal_dist_m / charger_goal_angle_deg: charger pose vs the plugged pose computed from the CURRENT receptacle
    pose (ManiSkill's goal_pose is stored at reset)"""
    objs = {"charger": Obj("charger", half=None)}
    probes = {"charger_goal_dist_m": lambda u: _plug_geom(u)[0], "charger_goal_angle_deg": lambda u: _plug_geom(u)[1]}
    return build_recorder(env, objs, release=["charger"], extra_probes=probes,
                          extra_pairs={"charger_receptacle": _pair("charger", "receptacle")})


# ------------------------------------------------------------------------------------------------ E27's held-out tasks
def _peg_geom(u):
    """E27 peg_geometry (research/scripts/e27_tasks.py), same quantities: head coordinate along the hole axis measured from
    the box centre (depth), lateral excess of the head over the hole radius, angle between peg and hole axes, TCP
    distance to the peg's axis segment"""
    def T(x):
        return x.to_transformation_matrix()[0].cpu().numpy().astype(float)
    L = float(u.peg_half_sizes[0, 0]); R = float(u.box_hole_radii[0])
    Th, Tp = T(u.box_hole_pose), T(u.peg.pose)
    head = Tp[:3, 3] + Tp[:3, 0] * L
    d = Th[:3, :3].T @ (head - Th[:3, 3])
    ang = float(np.degrees(np.arccos(np.clip(abs(float(Th[:3, 0] @ Tp[:3, 0])), -1, 1))))
    tcp = u.agent.tcp.pose.p[0].cpu().numpy().astype(float)
    loc = Tp[:3, :3].T @ (tcp - Tp[:3, 3])
    axis_d = float(np.linalg.norm(loc - np.array([np.clip(loc[0], -L, L), 0, 0])))
    return dict(depth=float(d[0]), excess=float(max(abs(d[1]), abs(d[2])) - R), angle=ang, tcp_axis=axis_d)


def peg_insertion_side(env):
    """peg box size changes at every reconfiguration (ManiSkill re-randomises it per episode with one env): the bounding
    box is read from u.peg_half_sizes at every step"""
    objs = {"peg": Obj("peg", half=lambda u: [float(x) for x in u.peg_half_sizes[0].cpu().numpy()], tilt_axis=0)}
    probes = {"peg_head_depth_m": lambda u: _peg_geom(u)["depth"],
              "peg_head_radial_excess_m": lambda u: _peg_geom(u)["excess"],
              "peg_axis_angle_deg": lambda u: _peg_geom(u)["angle"],
              "tcp_peg_axis_dist_m": lambda u: _peg_geom(u)["tcp_axis"]}
    return build_recorder(env, objs, release=["peg"], extra_probes=probes,
                          extra_pairs={"peg_box": _pair("peg", "box")})


def pull_cube_tool(env):
    """cube_base_dist_m = ManiSkill's success quantity (horizontal distance of the cube to the robot's root link);
    pair tool_cube = contact between the L-shaped tool and the cube (process: the cube was moved by the tool)"""
    objs = {"cube": Obj("cube", half=0.02), "tool": Obj("l_shape_tool", half=None)}

    def base_dist(u):
        b = u.agent.robot.get_links()[0].pose.p[0].cpu().numpy().astype(float)
        return float(np.linalg.norm(_p(u, "cube")[:2] - b[:2]))
    return build_recorder(env, objs, release=["tool", "cube"], extra_probes={"cube_base_dist_m": base_dist},
                          extra_pairs={"tool_cube": _pair("l_shape_tool", "cube")})
