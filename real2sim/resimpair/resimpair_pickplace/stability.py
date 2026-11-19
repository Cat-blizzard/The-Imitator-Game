"""State-window diagnostics, distinct from unchanged final task acceptance."""
import numpy as np
from scipy.spatial.transform import Rotation


def opening_width(measured_width, clearance=.008):
    if not np.isfinite(measured_width) or not .005 <= measured_width <= .077:
        raise ValueError('invalid_measured_grasp_aperture')
    width = measured_width + clearance
    if width > .08:
        raise ValueError('insufficient_opening_reserve')
    return float(width)


def window_stability(samples, arm, index):
    if len(samples) < 6:
        return {'stable': False, 'reason': 'insufficient_window'}
    relative = []
    positions = []
    rotations = []
    for row in samples:
        state = np.asarray(row['object_states'][index])
        obj = np.eye(4)
        obj[:3, :3] = Rotation.from_quat(state[[4, 5, 6, 3]]).as_matrix()
        obj[:3, 3] = state[:3]
        relative.append(np.linalg.inv(row['tcp_world'][arm]) @ obj)
        positions.append(obj[:3, 3])
        rotations.append(obj[:3, :3])
    relative = np.asarray(relative)
    dp = float(np.max(np.linalg.norm(relative[:, :3, 3] - relative[-1, :3, 3], axis=1)))
    da = float(np.max(Rotation.from_matrix(relative[-1, :3, :3].T @ relative[:, :3, :3]).magnitude()))
    wp = float(np.max(np.linalg.norm(np.asarray(positions) - positions[-1], axis=1)))
    wa = float(np.max(Rotation.from_matrix(rotations[-1].T @ np.asarray(rotations)).magnitude()))
    held = all(bool(row['grasp'][arm][index]) for row in samples)
    return {'stable': bool(held and dp <= .0015 and da <= np.deg2rad(2) and wp <= .0015 and wa <= np.deg2rad(2)),
            'all_frames_grasped': held, 'relative_translation_range_m': dp,
            'relative_rotation_range_deg': float(np.rad2deg(da)),
            'world_translation_range_m': wp, 'world_rotation_range_deg': float(np.rad2deg(wa)),
            'frames': len(samples), 'translation_bound_m': .0015, 'rotation_bound_deg': 2.0,
            'scope': 'sampled settled pose, not external disturbance robustness or substep stability'}


def aperture_command(width, lower, upper):
    """Normalize the actual mimic-controller bounds, including negative squeeze."""
    if not all(np.isfinite(x) for x in [width,lower,upper]) or upper <= lower:
        raise ValueError('invalid_controller_aperture_bounds')
    joint=width/2
    if not lower <= joint <= upper:
        raise ValueError('aperture_outside_controller_bounds')
    return float(2*(joint-lower)/(upper-lower)-1)
