"""Measured state, rigid transforms and control-action validation."""
from __future__ import annotations
import math
from typing import Any
import numpy as np
from .trajectory_execution import ArmBinding, as_numpy

def _positive(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result



def _budget(value: Any, name: str, *, zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if zero else 1):
        raise ValueError(f"{name} must be a {'nonnegative' if zero else 'positive'} integer")
    return value



def _measure(arms: tuple[ArmBinding, ArmBinding]) -> dict[str, Any]:
    result = {}
    for arm in arms:
        full = arm.full_qpos()
        velocity = as_numpy(arm.agent.robot.get_qvel())
        if velocity.shape != (1, 9) or not np.isfinite(velocity).all():
            raise ValueError("Expected one finite nine-joint Panda velocity state")
        fingers = [i for i in range(9) if i not in arm.indices]
        result[arm.key] = {"qpos_rad": full[arm.indices].tolist(),
                           "qvel_rad_s": velocity[0, arm.indices].tolist(),
                           "finger_qpos_m": full[fingers].tolist(),
                           "finger_qvel_m_s": velocity[0, fingers].tolist()}
    return result



def _actions_valid(env: Any, arms: tuple[ArmBinding, ArmBinding], action: dict[str, Any]) -> None:
    if set(action) != {arm.key for arm in arms}:
        raise ValueError("Every control tick must contain exactly both robot actions")
    converted = {}
    for arm in arms:
        value = np.asarray(action[arm.key], dtype=np.float64)
        if value.shape != (8,) or not np.isfinite(value).all() or abs(value[7]) > 1:
            raise ValueError("Each action must contain seven finite radians and one normalized gripper command")
        if np.any(value[:7] < arm.limits[:, 0] - 1e-6) or np.any(value[:7] > arm.limits[:, 1] + 1e-6):
            raise ValueError("Arm target exceeds joint limits")
        converted[arm.key] = value.astype(np.float32)
    if not env.unwrapped.agent.single_action_space.contains(converted):
        raise ValueError("Action exceeds the actual controller action space")



def _error(report: dict[str, Any], reason: str, exc: Exception) -> None:
    report.update(status="aborted", reason=reason, error_type=type(exc).__name__,
                  error=str(exc))



def _transform(value: Any) -> np.ndarray:
    matrix = np.array(as_numpy(value), dtype=np.float64, copy=True)
    if matrix.shape == (1, 4, 4):
        matrix = matrix[0]
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-6)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), rtol=0, atol=1e-5)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, rtol=0, atol=1e-5)):
        raise ValueError("Expected a single finite rigid 4x4 transform")
    return matrix



def _velocity(actor: Any, name: str) -> np.ndarray:
    value = np.array(as_numpy(getattr(actor, f"get_{name}_velocity")()), dtype=np.float64, copy=True)
    if value.shape == (1, 3):
        value = value[0]
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"Expected finite measured object {name} velocity")
    return value



def _rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
    """SO(3) geodesic; project float32 pose rounding before taking the angle."""
    u, _, vh = np.linalg.svd(first.T @ second)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vh)
    relative = u @ correction @ vh
    sine = np.linalg.norm([relative[2, 1] - relative[1, 2],
                           relative[0, 2] - relative[2, 0],
                           relative[1, 0] - relative[0, 1]]) / 2
    cosine = np.clip((np.trace(relative) - 1) / 2, -1.0, 1.0)
    return float(np.arctan2(sine, cosine))



def _observe(arms: Any, actor: Any, reference: np.ndarray, initial_z: float, *,
             measurements: dict[str, Any] | None = None) -> dict[str, Any]:
    """Measure one object, optionally sharing joints from this same observation.

    Single-arm callers retain their own joint read. A caller measuring both
    targets may supply the joints it just read; it must not reuse them after a
    physics step. The supplied mapping is returned without modification.
    """
    agent = arms[0].agent
    tcp = _transform(agent.tcp.pose.to_transformation_matrix())
    obj = _transform(actor.pose.to_transformation_matrix())
    relative = np.eye(4)
    relative[:3, :3] = tcp[:3, :3].T @ obj[:3, :3]
    relative[:3, 3] = tcp[:3, :3].T @ (obj[:3, 3] - tcp[:3, 3])
    angle_cos = (np.trace(reference[:3, :3].T @ relative[:3, :3]) - 1) / 2
    predicate = as_numpy(agent.is_grasping(actor))
    if predicate.size != 1 or not np.isfinite(predicate).all() or not np.isin(predicate, [False, True]).all():
        raise ValueError("Expected one physical contact-based grasp predicate")
    forces = None
    if all(hasattr(agent, name) for name in ("scene", "finger1_link", "finger2_link")):
        forces = {}
        for name, link in (("finger1", agent.finger1_link), ("finger2", agent.finger2_link)):
            force = np.array(as_numpy(agent.scene.get_pairwise_contact_forces(link, actor)), dtype=np.float64)
            if force.shape != (1, 3) or not np.isfinite(force).all():
                raise ValueError("Expected a single finite finger-object contact force")
            forces[name] = force[0].tolist()
    linear, angular = _velocity(actor, "linear"), _velocity(actor, "angular")
    return {"measurements": _measure(arms) if measurements is None else measurements,
            "object_T_world": obj.tolist(),
            "tcp_T_world": tcp.tolist(), "T_tcp_object": relative.tolist(),
            "position_slip_m": float(np.linalg.norm(relative[:3, 3] - reference[:3, 3])),
            "rotation_slip_rad": float(np.arccos(np.clip(angle_cos, -1.0, 1.0))),
            "object_height_m": float(obj[2, 3]), "object_height_gain_m": float(obj[2, 3] - initial_z),
            "object_linear_velocity_m_s": linear.tolist(), "object_speed_m_s": float(np.linalg.norm(linear)),
            "object_angular_velocity_rad_s": angular.tolist(),
            "object_angular_speed_rad_s": float(np.linalg.norm(angular)),
            "is_grasping": bool(predicate.reshape(-1)[0]), "contact_forces_N": forces}
