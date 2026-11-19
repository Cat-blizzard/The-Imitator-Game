"""PCHIP geometric path plus the retained TOPPRA timing algorithm.

The global conservative time law in bounded_retime_v1 exceeded the runtime
limits on all three development probes. This candidate keeps TOPPRA's local
speed/acceleration scheduling, replacing only the overshooting geometric spline.
It returns the original path under a constant-acceleration parameterization;
there is no second cubic fit in joint/time space and no clipping of positions.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator
from toppra.interpolator import AbstractGeometricPath
from toppra.algorithm import TOPPRA
from toppra.constraint import JointVelocityConstraint, JointAccelerationConstraint

VERSION = 'shape_preserving_toppra_v2'


class ShapePath(AbstractGeometricPath):
    def __init__(self, knots, values):
        self.knots = knots
        self.values = values
        self.spline = PchipInterpolator(knots, values, axis=0, extrapolate=False)

    @property
    def dof(self):
        return self.values.shape[1]

    @property
    def path_interval(self):
        return np.array([0., 1.])

    def __call__(self, positions, order=0):
        positions = np.asarray(positions)
        if np.any(positions < -1e-10) or np.any(positions > 1+1e-10):
            raise ValueError('path_parameter_outside_domain')
        # Clamp only roundoff of the scalar path parameter, never joint angles.
        return self.spline(np.clip(positions, 0, 1), order)


def retime(waypoints, timestep, joint_limits, velocity_limits, acceleration_limits):
    values = np.asarray(waypoints, float)
    limits = np.asarray(joint_limits, float)
    if values.ndim != 2 or values.shape[1] != 7 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('finite_seven_joint_path_required')
    if limits.shape != (7, 2) or not .01 <= timestep <= .10:
        raise ValueError('native_limits_and_timestep_required')
    if np.any(values < limits[:, 0]) or np.any(values > limits[:, 1]):
        raise ValueError('waypoint_position_limit_no_clipping')
    keep = np.r_[True, np.max(abs(np.diff(values, axis=0)), axis=1) > 1e-10]
    values = values[keep]
    if len(values) < 2:
        raise ValueError('nonzero_path_required')
    knots = np.linspace(0, 1, len(values))
    path = ShapePath(knots, values)
    velocity = np.minimum(np.asarray(velocity_limits, float)[:7], .079/timestep)
    acceleration = np.asarray(acceleration_limits, float)[:7]
    if np.any(velocity <= 0) or np.any(acceleration <= 0) or not np.isfinite(np.r_[velocity, acceleration]).all():
        raise ValueError('positive_finite_dynamic_limits_required')
    # Include both sides of every PCHIP second-derivative discontinuity, plus
    # interior points. These are solver constraints, not relaxed acceptance.
    grid = np.unique(np.r_[np.linspace(0, 1, (len(values)-1)*8+1),
                           knots[1:-1]-1e-9, knots[1:-1]+1e-9])
    solver = TOPPRA([JointVelocityConstraint(velocity), JointAccelerationConstraint(acceleration)],
                   path, gridpoints=grid, parametrizer='ParametrizeConstAccel')
    trajectory = solver.compute_trajectory(0, 0)
    if trajectory is None or not np.isfinite(trajectory.duration) or trajectory.duration <= 0:
        raise RuntimeError('time_parameterization_failed')
    check_times = np.linspace(0, trajectory.duration, max(1001, len(grid)*4))
    ratio = max(1., float(np.max(np.abs(trajectory(check_times, 1))/velocity)),
                float(np.sqrt(np.max(np.abs(trajectory(check_times, 2))/acceleration))))
    duration = trajectory.duration * ratio * 1.001
    count = int(np.ceil(duration/timestep))+1
    if count > 300:
        raise ValueError('retime_exceeds_retained_300_command_limit')
    duration = (count-1)*timestep
    factor = duration/trajectory.duration
    times = np.arange(count)*timestep
    q = trajectory(times/factor)
    dq = trajectory(times/factor, 1)/factor
    ddq = trajectory(times/factor, 2)/factor**2
    if (not np.isfinite(q).all() or np.any(q < limits[:, 0]-1e-12)
            or np.any(q > limits[:, 1]+1e-12)
            or np.any(np.abs(dq) > velocity+1e-8) or np.any(np.abs(ddq) > acceleration+1e-8)):
        raise RuntimeError('sampled_position_and_dynamic_checks_failed')
    return times, q, dq, ddq, duration


class PlannerAdapter:
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
