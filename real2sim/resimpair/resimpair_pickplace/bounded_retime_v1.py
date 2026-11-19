"""Shape-preserving joint interpolation with explicit velocity/acceleration bounds.

MPLib TOPP's cubic spline can overshoot position limits between feasible knots.
PCHIP stays in each joint's adjacent knot range. A global quintic time law then
starts/stops at rest and bounds speed/acceleration using conservative polynomial
coefficient bounds. Full native collision/path checks remain mandatory.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator

VERSION = 'shape_preserving_bounded_retime_v1'


def retime(waypoints, timestep, joint_limits, velocity_limits, acceleration_limits):
    values = np.asarray(waypoints, float)
    limits = np.asarray(joint_limits, float)
    if values.ndim != 2 or len(values) < 2 or values.shape[1] != 7 or not np.isfinite(values).all():
        raise ValueError('finite_seven_joint_path_required')
    if not .01 <= timestep <= .1 or limits.shape != (7, 2):
        raise ValueError('native_timing_and_joint_limits_required')
    if np.any(values < limits[:, 0]) or np.any(values > limits[:, 1]):
        raise ValueError('waypoint_position_limit_no_clipping')
    keep = np.r_[True, np.max(np.abs(np.diff(values, axis=0)), axis=1) > 1e-10]
    values = values[keep]
    if len(values) < 2:
        raise ValueError('nonzero_path_required')
    increments = np.linalg.norm(np.diff(values, axis=0), axis=1)
    parameter = np.r_[0., np.cumsum(increments)]
    parameter /= parameter[-1]
    spline = PchipInterpolator(parameter, values, axis=0)
    # On each interval p=a*x^3+b*x^2+c*x+d, 0<=x<=h.
    # Triangle-inequality bounds are conservative, not sampled estimates.
    h = np.diff(parameter)[:, None]
    a, b, c, _ = spline.c
    first_bound = np.max(3*np.abs(a)*h*h + 2*np.abs(b)*h + np.abs(c), axis=0)
    second_bound = np.max(6*np.abs(a)*h + 2*np.abs(b), axis=0)
    v = np.broadcast_to(np.asarray(velocity_limits, float).reshape(-1)[:7], (7,))
    acc = np.broadcast_to(np.asarray(acceleration_limits, float).reshape(-1)[:7], (7,))
    if not np.isfinite(v).all() or not np.isfinite(acc).all() or np.any(v <= 0) or np.any(acc <= 0):
        raise ValueError('positive_finite_dynamic_limits_required')
    # The unchanged guard caps command increments at 0.08 rad as well.
    v = np.minimum(v, .079 / timestep)
    # s(u)=10u^3-15u^4+6u^5; max s'=1.875, max |s''|=10/sqrt(3).
    maximum_s1, maximum_s2 = 1.875, 10 / np.sqrt(3)
    duration = max(float(np.max(first_bound*maximum_s1/v)),
        float(np.sqrt(np.max((second_bound*maximum_s1**2 + first_bound*maximum_s2)/acc))), timestep)
    duration *= 1.001
    if duration > 30:
        raise ValueError('bounded_retime_exceeds_30_seconds')
    count = int(np.ceil(duration/timestep))+1
    duration = (count-1)*timestep
    if duration > 30:
        raise ValueError('quantized_retime_exceeds_30_seconds')
    times = np.arange(count)*timestep
    u = times/duration
    s = 10*u**3-15*u**4+6*u**5
    sd = (30*u**2-60*u**3+30*u**4)/duration
    sdd = (60*u-180*u**2+120*u**3)/duration**2
    q = spline(s)
    dq = spline(s, 1)*sd[:, None]
    ddq = spline(s, 2)*sd[:, None]**2 + spline(s, 1)*sdd[:, None]
    if (np.any(q < limits[:, 0]-1e-12) or np.any(q > limits[:, 1]+1e-12)
            or np.any(np.abs(dq) > v+1e-9) or np.any(np.abs(ddq) > acc+1e-9)):
        raise RuntimeError('bounded_retime_certificate_failed')
    return times, q, dq, ddq, duration


class PlannerAdapter:
    """Only substitutes retiming; IK inputs and original planner are preserved."""
    def __init__(self, planner):
        self.planner = planner

    def __getattr__(self, name):
        return getattr(self.planner, name)

    def TOPP(self, path, timestep):
        return retime(path, timestep, np.asarray(self.planner.joint_limits)[:7],
                      self.planner.joint_vel_limits, self.planner.joint_acc_limits)


def cartesian_path(planner, root, current, goal, timestep=.05, waypoint_m=.002):
    from .continuation_ik_v2 import cartesian_path as continuation
    result = continuation(PlannerAdapter(planner), root, current, goal, timestep, waypoint_m)
    result['retiming_policy'] = VERSION
    return result
