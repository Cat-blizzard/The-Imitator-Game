"""Native contracts for original TIG tasks using the existing embodiment library.

The contract describes public qpos and controller actions, not object state.
Joint names and controller order are read from the live environment. Model and
mount identities prevent matching dimensions from implying matching physics.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np

TASKS = ("TwoRobotPickPillToRegions-v1", "TwoRobotPickAppleBananaToBaskets-v1")
ROBOTS = ("panda", "piper", "arx_x5", "ur5_wsg", "xarm6_robotiq")


def resolve_robot(name: str) -> str:
    """Resolve public aliases without registering a second set of agents."""
    name = str(name)
    for prefix in ("dual_", "imsim_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    if name == "panda_wristcam":
        name = "panda"
    if name not in ROBOTS:
        raise ValueError(f"Unsupported original-task embodiment {name!r}; choose {ROBOTS}")
    return name


def robot_pair(name: str) -> tuple[str, str]:
    robot = resolve_robot(name)
    uid = "panda_wristcam" if robot == "panda" else f"imsim_{robot}"
    return uid, uid


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    if isinstance(value, (np.ndarray, np.generic)) or hasattr(value, "detach"):
        return _array(value).tolist()
    return value


_XML_COMMENT = re.compile(rb"<!--.*?-->", re.DOTALL)
_XML_INTERTAG_SPACE = re.compile(rb">\s+<")


def _file_hash(path: str | Path) -> str:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Cannot identify loaded robot model: {p}")
    data = p.read_bytes()
    if p.suffix.lower() in {".urdf", ".srdf", ".xml"}:
        # Generated URDFs may contain absolute source paths in comments. They
        # do not change the loaded model and must not invalidate a contract.
        data = _XML_COMMENT.sub(b"", data).replace(b"\r\n", b"\n")
        data = _XML_INTERTAG_SPACE.sub(b"><", data).strip()
    return sha256(data).hexdigest()


def _controller_identity(controller: Any) -> dict[str, Any]:
    cfg = controller.config
    fields = ("lower", "upper", "normalize_action", "use_delta", "use_target",
              "stiffness", "damping", "force_limit", "friction", "mimic")
    return {key: _json_value(getattr(cfg, key)) for key in fields if hasattr(cfg, key)}


def native_contract(env: Any, cameras: Sequence[str] = ("zed2i",)) -> dict[str, Any]:
    """Describe B=1 native public state/actions and the actual physical model.

    qpos has shape (1, sum(len(state_joint_names[i]))); actions are an ordered
    mapping of two vectors of widths action_order[i]['size']. Arm targets are
    absolute radians. Each supported gripper has one native controller action.
    """
    raw = env.unwrapped
    if int(raw.num_envs) != 1:
        raise ValueError("The original-task native contract requires one environment")
    actions, models, mounts, state_names = [], [], [], []
    for key, agent in raw.agent.agents_dict.items():
        arm = agent.controller.controllers["arm"]
        if agent.control_mode != "pd_joint_pos" or arm.config.normalize_action or arm.config.use_delta:
            raise ValueError("Native original-task actions require absolute pd_joint_pos radians")
        names = [joint.name for joint in agent.robot.active_joints]
        arm_names = [joint.name for joint in arm.joints]
        if len(set(names)) != len(names) or not set(arm_names).issubset(names):
            raise ValueError("Controller joints must uniquely name physical coordinates")
        rec = getattr(agent, "record", None)
        opened = float(rec.gripper_cmd_open[0]) if rec else 1.0
        closed = float(rec.gripper_cmd_closed[0]) if rec else -1.0
        size = int(np.prod(agent.single_action_space.shape))
        if size != len(arm_names) + 1:
            raise ValueError("The original-task bridge supports scalar parallel grippers")
        unit = "rad" if rec and rec.name == "xarm6_robotiq" else "m"
        row = dict(key=key, size=size, arm_joint_names=arm_names, state_joint_names=names,
                   arm_coordinate_unit="rad", gripper_coordinate_unit=unit,
                   gripper_joint_coordinate_unit=unit,
                   gripper_action_unit=("rad" if unit == "rad" else "normalized"),
                   gripper_open_action=opened, gripper_closed_action=closed)
        actions.append(row)
        state_names.append(names)
        urdf = str(agent.urdf_path)
        if "{" in urdf:
            from mani_skill import format_path
            urdf = str(format_path(urdf))
        model = dict(uid=agent.uid, urdf_sha256=_file_hash(urdf),
                     tcp_link=agent.ee_link_name,
                     arm_controller=_controller_identity(arm),
                     gripper_controllers={k: _controller_identity(c) for k, c in
                                          agent.controller.controllers.items() if k != "arm"})
        if hasattr(agent, "physical_gripper_coupling"):
            model["physical_gripper_coupling"] = _json_value(agent.physical_gripper_coupling)
        if rec:
            model.update(record=rec.name, joint_limits=_json_value(rec.joint_limits),
                         grasp_links=list(rec.grasp_links), pad_links=list(rec.pad_links or ()),
                         derived_mplib_urdf_sha256=_file_hash(rec.mplib_urdf_path),
                         derived_mplib_srdf_sha256=_file_hash(rec.mplib_srdf_path))
        models.append(model)
        mounts.append(_array(agent.robot.pose.raw_pose).reshape(-1).tolist())
    model_fingerprints = [sha256(json.dumps(m, sort_keys=True, allow_nan=False).encode()).hexdigest()
                          for m in models]
    result = dict(schema_version=1, env_id=raw.spec.id, robot_uids=list(raw.robot_uids),
                  cameras=list(cameras), control_mode="pd_joint_pos",
                  control_freq=int(raw.control_freq), sim_freq=int(raw.sim_freq),
                  state_joint_names=state_names, action_layout=actions, action_order=actions,
                  state_dim=sum(map(len, state_names)), action_dim=sum(r["size"] for r in actions),
                  models=models, model_fingerprints=model_fingerprints, mounts=mounts,
                  coordinate_convention="world metres; WXYZ quaternions; absolute native joint targets",
                  physics_profile="imitator2-native-v1")
    json.dumps(result, allow_nan=False)
    return result


def make_native_env(env_id: str, robot: str = "panda", **kwargs: Any) -> Any:
    """Create a supported original task with its official scene and horizon."""
    import gymnasium as gym
    from mani_skill.envs.tasks import tabletop  # noqa: F401
    robot = resolve_robot(robot)
    if env_id not in TASKS or env_id == TASKS[1] and robot not in ("panda", "piper"):
        raise ValueError(f"Unsupported original task/embodiment: {env_id}/{robot}")
    if int(kwargs.get("num_envs", 1)) != 1:
        raise ValueError("Native original-task execution currently requires num_envs=1")
    if robot != "panda":
        from . import agents  # noqa: F401
    kwargs.setdefault("control_mode", "pd_joint_pos")
    kwargs.setdefault("sim_backend", "physx_cpu")
    if kwargs["sim_backend"] not in ("cpu", "physx_cpu"):
        raise ValueError("The original-task native bridge currently supports CPU PhysX")
    kwargs.setdefault("num_envs", 1)
    horizon = kwargs.pop("max_episode_steps", 200)
    if int(horizon) != 200:
        raise ValueError("The original task's 200-step horizon is fixed")
    return gym.make(env_id, embodiment=(None if robot == "panda" and env_id == TASKS[1] else robot),
                    max_episode_steps=200, **kwargs)


if __name__ == "__main__":
    assert resolve_robot("dual_piper") == "piper"
    assert robot_pair("arx_x5") == ("imsim_arx_x5",) * 2
    assert robot_pair("panda") == ("panda_wristcam",) * 2
    print("Native aliases and scalar-gripper contract checks passed")
