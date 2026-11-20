"""Finite physical coupling for the Robotiq's single opening motor.

Equal PD drive targets do not mechanically link the outer knuckles. The
existing four-bar constraints within each finger remain owned by ManiSkill.
"""
from __future__ import annotations

from typing import Any

ROBOTIQ_OUTER_JOINTS = ("left_outer_knuckle_joint", "right_outer_knuckle_joint")
ROBOTIQ_COUPLING = {"version": "robotiq-outer-tendon-v1", "joint_names": list(ROBOTIQ_OUTER_JOINTS),
                    "stiffness": 1e3, "damping": 10.0, "rest_length": 0.0}


def couple_robotiq_outer_knuckles(robot: Any) -> int:
    """Link the two angular coordinates in each native articulation, qL=qR.

    Input is the existing ManiSkill articulation (B worlds). Return B tendons.
    Joint coordinates remain radians; the tendon adds finite spring/damping.
    """
    left, right = (robot.active_joints_map[name] for name in ROBOTIQ_OUTER_JOINTS)
    count = 0
    for root, first, second, articulation in zip(
            left.get_parent_link()._objs, left.get_child_link()._objs,
            right.get_child_link()._objs, robot._objs, strict=True):
        articulation.create_fixed_tendon(
            [root, first, second], [0., -1., 1.], [0., -1., 1.],
            rest_length=ROBOTIQ_COUPLING["rest_length"],
            stiffness=ROBOTIQ_COUPLING["stiffness"], damping=ROBOTIQ_COUPLING["damping"])
        count += 1
    return count


if __name__ == "__main__":
    from types import SimpleNamespace as NS
    calls: list[dict[str, Any]] = []
    articulation = NS(create_fixed_tendon=lambda *args, **kwargs: calls.append(kwargs))
    joint = NS(get_parent_link=lambda: NS(_objs=[object()]),
               get_child_link=lambda: NS(_objs=[object()]))
    robot = NS(active_joints_map={name: joint for name in ROBOTIQ_OUTER_JOINTS}, _objs=[articulation])
    assert couple_robotiq_outer_knuckles(robot) == 1
    assert len(calls) == 1 and calls[0]["stiffness"] == 1e3
    print("coupling: one finite outer-knuckle tendon per articulation passed")
