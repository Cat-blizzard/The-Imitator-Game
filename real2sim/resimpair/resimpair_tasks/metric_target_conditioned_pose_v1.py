"""Target-side source sampling for the existing fixed dual-Panda worktable.

Use the retained XY bounds and full yaw range, but reflect source Y into the
same robot half as the assigned destination. This is a proposal distribution,
not a grasp/IK certificate. Native table separation and same-arm local-path
checks still follow. Sizes, poses in asset coordinates and target slots stay.
"""
import math
import numpy as np

VERSION = 'target_half_conditioned_source_pose_v1'


def propose(*, rng, container, container_descriptor, target_offset_xy_m, **context):
    angle = float(container['yaw'])
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    local = np.asarray(container_descriptor['cavity']['center_xy_m']) + np.asarray(target_offset_xy_m)
    destination = np.asarray(container['xy']) + rotation @ local
    x = float(rng.uniform(-.36, -.14))
    y = float(rng.uniform(-.61, .61))
    if abs(destination[1]) > 1e-9:
        y = float(np.copysign(abs(y), destination[1]))
    return {'xy': [x, y], 'yaw': float(rng.uniform(-math.pi, math.pi))}
