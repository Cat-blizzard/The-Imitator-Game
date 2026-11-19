"""v3.2 recorders for the E29b B1 dual-arm tasks (DualLiftBox-v1, DualStack-v1): the generic v3.2 signals
(imitator.verify.v32.build_recorder) plus task geometry; per-arm contact pairs arm<i>_<obj>_impulse_Ns.

DualLiftBox-v1   box_* (box of half extents (0.02, 0.18, 0.02)), box_long_tilt_deg (long axis to horizontal),
                 arm0_box_impulse_Ns / arm1_box_impulse_Ns, table_box_impulse_Ns, robot_box_impulse_Ns; no release
DualStack-v1     cubeA_* / cubeB_*, a_target_margin_m (A centre inside the target square: min(half - |dx|, half - |dy|)),
                 ab_xy_m (|xy(B) - xy(A)|), ab_dz_err_m (| z(B) - z(A) - 2h |), cubeA_cubeB_impulse_Ns (support),
                 arm<i>_cube<A|B>_impulse_Ns; release over both cubes (contact + all-link enclosure), grip_cmd_gap_m
  derived (make_stack_derive): carry_A_by_arm0 / carry_B_by_arm1 = 1 from the end of the first run of >= carry_steps
  control steps in which the cube is off the table (no table contact, bottom > airborne_dz_m) and touched by that arm
  and not by the other arm; 0 before. 'Each arm carried its own cube'.
"""
from __future__ import annotations

import numpy as np

from .v32 import Obj, build_recorder

CARRY_ARGS = dict(carry_steps=5, airborne_dz_m=0.003)


def _pair(a: str, b: str):
    from veritask.adapters.maniskill_v3 import PairV3
    return PairV3(lambda u: [getattr(u, a)], lambda u: [getattr(u, b)])


def _p(u, attr):
    return getattr(u, attr).pose.p[0].cpu().numpy().astype(float)


def lift_box(env):
    u = env.unwrapped
    objs = {"box": Obj("box", half=tuple(float(x) for x in u.box_half))}

    def long_tilt(u):
        R = u.box.pose.to_transformation_matrix()[0, :3, :3].cpu().numpy().astype(float)
        return float(np.degrees(np.arcsin(min(1.0, abs(R[2, 1])))))
    return build_recorder(env, objs, release=[], per_arm_pairs=["box"], extra_probes={"box_long_tilt_deg": long_tilt})


def stack(env):
    u = env.unwrapped
    h = float(u.cube_half)
    objs = {"cubeA": Obj("cubeA", half=h), "cubeB": Obj("cubeB", half=h)}

    def margin(u):
        d = np.abs(_p(u, "cubeA")[:2] - u.target_pos[0, :2].cpu().numpy())
        return float(u.dual.target_half - d.max())
    probes = {"a_target_margin_m": margin,
              "ab_xy_m": lambda u: float(np.linalg.norm(_p(u, "cubeB")[:2] - _p(u, "cubeA")[:2])),
              "ab_dz_err_m": lambda u: float(abs(_p(u, "cubeB")[2] - _p(u, "cubeA")[2] - 2 * h))}
    return build_recorder(env, objs, release=["cubeA", "cubeB"], per_arm_pairs=["cubeA", "cubeB"], extra_probes=probes,
                          extra_pairs={"cubeA_cubeB": _pair("cubeA", "cubeB")})


def carry_signal(traces: dict, obj: str, arm: int, carry_steps: int = 5, airborne_dz_m: float = 0.003) -> np.ndarray:
    me = np.asarray(traces[f"arm{arm}_{obj}_impulse_Ns"], float)
    other = np.asarray(traces[f"arm{1 - arm}_{obj}_impulse_Ns"], float)
    table = np.asarray(traces[f"table_{obj}_impulse_Ns"], float)
    dz = np.asarray(traces[f"{obj}_bottom_dz_m"], float)
    m = (table <= 0.0) & (dz > airborne_dz_m) & (me > 0.0) & (other <= 0.0)
    out = np.zeros(len(m))
    run = 0
    for t, x in enumerate(m):
        run = run + 1 if x else 0
        if run >= int(carry_steps):
            out[t:] = 1.0
            break
    return out


def stack_args(contract) -> dict:
    for t in contract.outcome:
        if (t.args or {}).get("signal") in ("carry_A_by_arm0", "carry_B_by_arm1"):
            return {**CARRY_ARGS, **{k: t.args[k] for k in CARRY_ARGS if k in t.args}}
    return dict(CARRY_ARGS)


def make_stack_derive(contract):
    a = stack_args(contract)

    def derive(tr: dict) -> dict:
        tr["carry_A_by_arm0"] = carry_signal(tr, "cubeA", 0, **a)
        tr["carry_B_by_arm1"] = carry_signal(tr, "cubeB", 1, **a)
        return tr
    return derive
