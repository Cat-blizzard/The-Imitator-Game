"""Signals and the derived handover event for DualHandover-v1 (contract imsim/contracts/dual_handover_v1.json).

Per-step signals (TraceRecorderV3 probes + substep-accumulated contact pairs, CPU PhysX, env 0):
  obj_x / obj_y / obj_z, obj_speed, obj_target_margin_m, obj_bottom_dz_m      as the cube task (box of half extents)
  robot_obj_impulse_Ns        object <-> every collision link of both robots (release rule)
  armL_obj_impulse_Ns / armR_obj_impulse_Ns   object <-> every collision link of the left / right robot
  table_obj_impulse_Ns        object <-> table
  obj_enclosure_all_m         v3.1 all-link enclosure (imitator.verify.v31.EnclosureAll)
  robot_qvel_max              arm joint speed (veritask RobotSpeed)
E29b: make_recorder_v32 (contract imsim_dual_handover_v2) records the v3.2 signals (imitator.verify.v32) under the same
names where the quantity is the same (obj_x/_y/_z, obj_bottom_dz_m, robot_obj / table_obj / armL_obj / armR_obj impulses)
and new names where it changed (objs_speed / obj_speed from corner positions, obj_enclosure_m, grip_cmd_gap_m), so the
derive below serves both contracts.
Derived (pure function of the traces, derive_handover(); parameters from the contract term's args):
  carry_L(t) = object airborne (no table impulse in the step, bottom > airborne_dz_m) AND left robot touches it AND the
               right robot does not; carry_R symmetric
  handover_done(t) = 1 from the first step t at which a run of >= carry_steps consecutive carry_R steps ends, provided an
               earlier run of >= carry_steps consecutive carry_L steps exists and the object did not touch the table
               between the end of that carry_L run and the start of the carry_R run (mid-air transfer); 0 before.
  i.e. "the object was carried by the left arm alone, then by the right arm alone, without touching the table in
  between". An airborne object touched by only one robot for 0.25 s is supported by that robot (a free fall of 0.25 s
  would drop it 30 cm), so the signal needs no grasp model of the embodiment.
"""
from __future__ import annotations

import numpy as np

from . import v31

HANDOVER_TERM_SIGNAL = "handover_done"
DEFAULT_ARGS = dict(carry_steps=5, airborne_dz_m=0.003)


def _robot_links(i):
    def f(u):
        ag = u.agent.agents[i]
        return [l for l in ag.robot.links if l._bodies[0].get_collision_shapes()]
    return f


def probes() -> dict:
    from veritask.adapters.maniskill_v3_tasks import RobotSpeed

    def xyz(i):
        return lambda u: float(u.block.pose.p[0, i])

    def margin(u):
        d = (u.block.pose.p[0, :2] - u.target_pos[0, :2]).abs().cpu().numpy()
        return float(u.dual.target_half - d.max())

    return dict(obj_x=xyz(0), obj_y=xyz(1), obj_z=xyz(2),
                obj_speed=lambda u: float(np.linalg.norm(u.block.linear_velocity[0].cpu().numpy())),
                obj_target_margin_m=margin,
                obj_bottom_dz_m=lambda u: float(u.cube_bottom_z()[0]),
                robot_qvel_max=RobotSpeed(),
                obj_enclosure_all_m=v31.EnclosureAll("block", "block_half"),
                opening_0_m=lambda u: float(u.agent.agents[0].measured_opening()[0]),
                opening_1_m=lambda u: float(u.agent.agents[1].measured_opening()[0]))


def make_recorder(env):
    from veritask.adapters.maniskill_v2_tasks import _all_robot_links
    from veritask.adapters.maniskill_v3 import PairV3, TraceRecorderV3
    obj = lambda u: [u.block]  # noqa: E731
    pairs = {"robot_obj": PairV3(_all_robot_links, obj, cumulative=True),
             "armL_obj": PairV3(_robot_links(0), obj), "armR_obj": PairV3(_robot_links(1), obj),
             "table_obj": PairV3(lambda u: [u.table], obj)}
    return TraceRecorderV3(env, probes(), pairs=pairs, release=None)


def make_recorder_v32(env):
    """protocol v3.2 recorder (E29b; contract imsim_dual_handover_v2): imitator.verify.v32.build_recorder over the block
    (object name 'obj': obj_x/_y/_z, obj_speed from corner POSITIONS, obj_bottom_dz_m, objs_speed, robot_obj_impulse_Ns,
    table_obj_impulse_Ns, obj_enclosure_m (all-link), grip_cmd_gap_m, robot_qvel_max, opening_i_m) + obj_target_margin_m
    + the per-arm pairs armL_obj / armR_obj under their v1 names, so the handover derive (make_derive) is unchanged.
    The v1 contract (dual_handover_v1) needs the v1 recorder make_recorder (PhysX obj_speed, obj_enclosure_all_m)."""
    from veritask.adapters.maniskill_v3 import PairV3
    from . import v32
    u = env.unwrapped
    blk = lambda u: [u.block]  # noqa: E731

    def margin(u):
        d = (u.block.pose.p[0, :2] - u.target_pos[0, :2]).abs().cpu().numpy()
        return float(u.dual.target_half - d.max())
    objs = {"obj": v32.Obj("block", half=[float(x) for x in u.block_half])}
    return v32.build_recorder(env, objs, release=["obj"], extra_probes={"obj_target_margin_m": margin},
                              extra_pairs={"armL_obj": PairV3(_robot_links(0), blk), "armR_obj": PairV3(_robot_links(1), blk)})


def handover_args(contract) -> dict:
    for t in contract.outcome:
        if (t.args or {}).get("signal") == HANDOVER_TERM_SIGNAL:
            return {**DEFAULT_ARGS, **{k: t.args[k] for k in DEFAULT_ARGS if k in t.args}}
    return dict(DEFAULT_ARGS)


def _runs(mask: np.ndarray):
    """(start, end) inclusive of every run of True"""
    out, s = [], None
    for i, m in enumerate(mask):
        if m and s is None:
            s = i
        elif not m and s is not None:
            out.append((s, i - 1)); s = None
    if s is not None:
        out.append((s, len(mask) - 1))
    return out


def handover_signal(traces: dict, carry_steps: int = 5, airborne_dz_m: float = 0.003) -> dict:
    L = np.asarray(traces["armL_obj_impulse_Ns"], float)
    R = np.asarray(traces["armR_obj_impulse_Ns"], float)
    T = np.asarray(traces["table_obj_impulse_Ns"], float)
    dz = np.asarray(traces["obj_bottom_dz_m"], float)
    air = (T <= 0.0) & (dz > airborne_dz_m)
    cL, cR = air & (L > 0) & (R <= 0), air & (R > 0) & (L <= 0)
    K = int(carry_steps)
    runs_L = [r for r in _runs(cL) if r[1] - r[0] + 1 >= K]
    runs_R = [r for r in _runs(cR) if r[1] - r[0] + 1 >= K]
    done = np.zeros(len(L))
    t_done = None
    for rs, re in runs_R:
        prior = [r for r in runs_L if r[1] < rs]
        for ls, le in prior[::-1]:
            if not (T[le + 1:rs] > 0).any():          # no table contact between the left carry and the right carry
                t_done = rs + K - 1
                break
        if t_done is not None:
            break
    if t_done is not None:
        done[t_done:] = 1.0
    return dict(carry_L=cL.astype(float), carry_R=cR.astype(float), handover_done=done,
                _info=dict(runs_L=runs_L, runs_R=runs_R, t_done=t_done))


def make_derive(contract):
    a = handover_args(contract)

    def derive(tr: dict) -> dict:
        s = handover_signal(tr, **a)
        for k in ("carry_L", "carry_R", "handover_done"):
            tr[k] = s[k]
        return tr
    return derive
