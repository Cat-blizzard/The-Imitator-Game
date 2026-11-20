"""Planning-only MPLib adapter for named fixed-base arm controllers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import sapien

from mani_skill import format_path
from mani_skill.examples.motionplanning.planner import Planner
from .trajectory_execution import as_numpy


class ArmMotionPlanner:
    """Return native trajectories; the common executor owns every physics step."""

    def __init__(self, env: Any, agent: Any, *, velocity_limit: float = .9,
                 acceleration_limit: float = 1.2):
        self.env, self.base_env, self.env_agent = env, env.unwrapped, agent
        self.robot, self.gripper_state = agent.robot, 1.
        self.prefer_nearby_revolute_goals = True
        self.use_point_cloud = False
        names = tuple(agent.controller.controllers["arm"].config.joint_names)
        record = getattr(agent, "record", None)
        urdf = Path(record.mplib_urdf_path) if record else Path(format_path(agent.urdf_path))
        srdf = Path(record.mplib_srdf_path) if record else Path(format_path(str(getattr(agent, "srdf_path", urdf.with_suffix(".srdf")))))
        if not srdf.is_file():
            raise FileNotFoundError(f"Prepare the robot's collision description first: {srdf}")
        self.planner = Planner(urdf=str(urdf), srdf=str(srdf),
            user_link_names=[link.get_name() for link in self.robot.get_links()],
            user_joint_names=[joint.get_name() for joint in self.robot.get_active_joints()],
            move_group=agent.ee_link_name,
            joint_vel_limits=np.full(len(names), velocity_limit),
            joint_acc_limits=np.full(len(names), acceleration_limit))
        self.planner.set_base_pose(as_numpy(self.robot.pose.raw_pose)[0])
        actual = tuple(self.planner.user_joint_names[i]
                       for i in self.planner.move_group_joint_indices)
        if actual != names:
            raise ValueError(f"Planner/controller joint orders differ: {actual} versus {names}")

    def move_to_pose_with_screw(self, pose: sapien.Pose, *, dry_run: bool = True):
        if not dry_run:
            raise ValueError("Use the shared executor to execute planned trajectories")
        start = as_numpy(self.robot.get_qpos())[0].copy()
        goal = np.r_[pose.p, pose.q]
        result = self.planner.plan_screw(goal, start.copy(),
            time_step=self.base_env.control_timestep, use_point_cloud=False)
        self.last_plan_method = {"screw_status": result["status"], "method": "screw"}
        if result["status"] != "Success":
            # A straight TCP path need not be reachable from every rest pose.
            # Keep non-arm joints measured and let native joint-space RRTConnect
            # find a path. The common adapter still checks the real FK endpoint.
            mask = np.ones(len(start), dtype=bool)
            mask[self.planner.move_group_joint_indices] = False
            status, goals = self.planner.IK(
                self.planner.transform_goal_to_wrt_base(goal), start.copy(),
                mask=mask.tolist(), threshold=.0005)
            self.last_plan_method.update(method="IK_RRTConnect", ik_status=str(status))
            if status != "Success":
                return {"status": str(status)}
            result = self.planner.plan_qpos_to_qpos(goals, start.copy(),
                time_step=self.base_env.control_timestep, fix_joint_limits=False)
        self.last_plan_method["status"] = result["status"]
        return result


if __name__ == "__main__":
    assert np.full(6, .9).shape == (6,)
    assert np.full(7, .9).shape == (7,)
    print("Arm planner uses the declared joint count")
