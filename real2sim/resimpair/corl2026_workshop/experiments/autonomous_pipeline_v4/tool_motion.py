"""Continuous local Cartesian tool paths with measured grasp-frame feedback."""
import numpy as np
from scipy.spatial.transform import Rotation
from .core import require
from .simulation import array


def _plan(rec, planners, arm, pose, stage, refine=10):
    """No RRT detour during the final approach to a small functional feature.

Check time-parameterized commands against self/table collisions. Scene contact
success remains an independent native-state decision, not a planner flag.
"""
    from examples.baselines.hypericlzq.eef import panda_ik,panda_chain
    from examples.baselines.hypericl_task_expansion.table_obstacle import first_command_collision
    agent=rec.unwrapped.agent.agents[arm];planner=planners[arm]
    state=array(agent.robot.qpos).reshape(9)
    root=array(agent.robot.pose.to_transformation_matrix()).reshape(4,4)
    start=array(agent.tcp.pose.to_transformation_matrix()).reshape(4,4)
    goal=pose.to_transformation_matrix()
    delta=Rotation.from_matrix(start[:3,:3].T@goal[:3,:3]).as_rotvec()
    distance=float(np.linalg.norm(goal[:3,3]-start[:3,3]))
    if distance<.0002 and np.linalg.norm(delta)<.002:
        rec.hold(refine,tuple(p.gripper_state for p in planners));return
    count=max(3,int(np.ceil(max(distance/.008,np.linalg.norm(delta)/.05)))+1)
    q=state[:7].copy();path=[q.copy()]
    for alpha in np.linspace(0.,1.,count)[1:]:
        target=np.eye(4);target[:3,3]=start[:3,3]+alpha*(goal[:3,3]-start[:3,3])
        target[:3,:3]=start[:3,:3]@Rotation.from_rotvec(alpha*delta).as_matrix()
        candidate,ok,_=panda_ik(np.linalg.inv(root)@target,q)
        require(bool(ok) and np.max(np.abs(candidate-q))<.35,"tool_continuous_ik_failed:"+stage)
        q=candidate;path.append(q.copy())
    _,positions,velocity,acceleration,_=planner.planner.TOPP(np.array(path),rec.unwrapped.control_timestep)
    require(0<len(positions)<=500,"tool_path_time_bound:"+stage)
    limits=np.array([bounds for _,active,bounds in panda_chain() if active])
    require(np.all(positions>=limits[:,0]) and np.all(positions<=limits[:,1]),"time_parameterized_joint_limit_exceeded:"+stage)
    for q in positions:
        require(not planner.planner.check_for_self_collision(qpos=np.r_[q,state[7:]]),"tool_path_self_collision:"+stage)
    require(first_command_collision(planner,positions,state) is None,"tool_path_table_collision:"+stage)
    return {"status":"Success","position":positions,"velocity":velocity,"acceleration":acceleration}


def linear(rec,planners,arm,pose,stage,refine=10):
    with rec.performance.section("tool_cartesian_ik_collision_and_parameterization"):
        path=_plan(rec,planners,arm,pose,stage,refine)
    if path is not None:
        rec.stage=stage
        planners[arm].follow_path(path,other_gripper_state=planners[1-arm].gripper_state,refine_steps=refine)
