"""Scripted expert and designed counterexamples for DualStack-v1 (E29b), any parallel-gripper dual configuration.

run(env, variant) with variant in
  expert            the LEFT arm (0) picks cube A and puts it on the target centre (released 6 mm above the table),
                    retreats and goes home; then the RIGHT arm (1) picks cube B and puts it on A (centre over A's actual
                    centre, re-aimed from B's actual position above the place point, released 6 mm above A's top), backs off
                    along its approach axis (8 / 5 / 3 cm, else straight up) and goes home (E29b script v2: v1 stayed
                    5-10 cm above B, which left the ARX X5 gripper within the aftermath's 3 cm and made its retreat
                    unplannable -> unknown verdicts)
  one_arm_does_both the left arm stacks both: A on the target, then B (from the right zone) on A; the right arm never moves
  hold_top          as the expert, but the right gripper stays closed on B (B resting on A)
  side_by_side      as the expert, but B is put on the table next to A (touching side, on the side of B's start)
  cage              as the expert, but after setting B on A the right gripper opens only ~1.2 mm beyond the cube width
  no_op             both arms stay at home
Grasp / place candidates: dual_b1_common.cube_candidates (top-down, approach tilt 0 / 0.3 / 0.55 rad, heights 10 / 7 / 5
cm, every waypoint IK-feasible with the other arm as a point cloud). Phases: FREE for transit with an empty gripper (to
the pre-grasp, retreat after release, home, an idle arm), CONTACT for grasp / lift / carry / place / release.
"""
from __future__ import annotations

import numpy as np

from ..data.phase import FREE
from .dual_b1_common import Arms, actor_T, pick_place_cube, result

BEHAVIOURS = ("expert", "one_arm_does_both", "hold_top", "side_by_side", "cage", "no_op")
DROP = 0.006           # release height of a cube's bottom above its support [m]


def run(env, variant: str = "expert") -> dict:
    if variant not in BEHAVIOURS:
        raise ValueError(variant)
    A = Arms(env)
    u = A.u
    h = float(u.cube_half)
    log = {}
    A.wait(1, FREE, FREE)
    if variant == "no_op":
        A.wait(20, FREE, FREE)
        return result(A, True, variant, log)
    tgt = u.target_pos[0].cpu().numpy().astype(float)
    ok = pick_place_cube(A, 0, u.cubeA, h, tgt[:2], h + DROP, "release", log, "A")
    if not ok:
        return result(A, False, variant, log)
    a = actor_T(u.cubeA)[:3, 3]
    if variant == "side_by_side":
        b0 = actor_T(u.cubeB)[:3, 3]
        d = b0[:2] - a[:2]; d /= np.linalg.norm(d) + 1e-9
        place_xy, place_cz = a[:2] + d * (2 * h + 0.01), h + DROP
    else:
        place_xy, place_cz = a[:2], a[2] + 2 * h + DROP
    arm = 0 if variant == "one_arm_does_both" else 1
    mode = {"hold_top": "hold", "cage": "cage"}.get(variant, "release")
    ok = pick_place_cube(A, arm, u.cubeB, h, place_xy, place_cz, mode, log, "B", retreat_home=True)
    if ok and mode == "release":
        A.wait(5, FREE, FREE)
    log["arm_B"] = arm
    return result(A, ok, variant, log)
