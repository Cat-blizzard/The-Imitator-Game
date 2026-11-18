"""Named controller order and paired execution across different arm sizes."""

from types import SimpleNamespace as NS

from gymnasium import spaces
import numpy as np
import pytest
import sapien
import sapien.physx as physx
from mplib.pymp.collision_detection import fcl

from mani_skill.examples.motionplanning.dual.sync.combined_collision import (
    NativeDualCollisionChecker, validate_schedule,
)
from mani_skill.examples.motionplanning.dual.sync.dual_arm_execution import _Phase, execute_dual_reach
from mani_skill.examples.motionplanning.dual.sync.dual_payload_collision import DualPayloadCollisionChecker
from mani_skill.examples.motionplanning.dual.sync.measurements import (
    _actions_valid, _measure, gripper_is_open, gripper_open_settings,
    finger_displacement_m,
)
from mani_skill.examples.motionplanning.dual.sync.planning_endpoint import checked_screw_plan
from mani_skill.examples.motionplanning.dual.sync.table_support_collision import (
    NumericalTableSupportCollisionChecker, TABLE_NAME, _scaled_vertices,
)
from mani_skill.examples.motionplanning.dual.sync.held_collision import _object_at, _pair_result
from mani_skill.examples.motionplanning.dual.sync.trajectory_execution import bind_arm, native_arm_action
from mani_skill.examples.motionplanning.dual.sync.trajectory_schedule import ArmTrajectory, build_hold_schedule


def trajectory(key, count, duration, displacement, grip=1):
    names = tuple(f"j{i}" for i in range(count))
    return ArmTrajectory(key, names, [0., duration],
                         np.array([[0.] * count, [displacement] * count]), grip)


def test_unequal_arm_sizes_keep_source_time_and_gripper_endpoint():
    left, right = trajectory("left", 6, .1, .02, -1), trajectory("right", 7, .2, .04)
    schedule = build_hold_schedule(left, right, control_timestep_s=.05)
    assert schedule.left_commands.shape == (4, 7)
    assert schedule.right_commands.shape == (4, 8)
    assert np.allclose(schedule.left_commands[1:, :6], .02)
    assert np.all(schedule.left_commands[:, -1] == -1)
    assert np.all(schedule.right_commands[:, -1] == 1)
    seen = []

    def check(left_qpos, right_qpos):
        assert left_qpos.shape == (6,) and right_qpos.shape == (7,)
        seen.append((left_qpos.copy(), right_qpos.copy()))
        return []

    report = validate_schedule(schedule, np.zeros(6), np.zeros(7), NS(check=check))
    assert report["status"] == "passed"
    assert np.allclose(seen[-1][0], .02) and np.allclose(seen[-1][1], .04)
    assert all(np.max(np.abs(left_qpos)) <= .02 for left_qpos, _ in seen)


def test_six_joint_trajectory_rejects_wrong_columns_or_joint_order():
    source = trajectory("left", 6, .1, .02)
    with pytest.raises(ValueError, match="shape"):
        ArmTrajectory("left", source.joint_names, [0., .1], np.zeros((2, 7)), 1)
    with pytest.raises(ValueError, match="unique"):
        ArmTrajectory("left", (), [0.], np.zeros((1, 0)), 1)
    result = {"status": "Success", "time": source.times_s,
              "position": source.positions_rad, "duration": .1}
    with pytest.raises(ValueError, match="order"):
        ArmTrajectory.from_planner_result(result, action_key="left",
            planner_joint_names=source.joint_names[::-1],
            controller_joint_names=source.joint_names, gripper_command=1)


class Robot:
    """Shuffled physical joint state prevents accidentally passing prefix slices."""

    def __init__(self, count):
        self.arm_names = tuple(f"j{i}" for i in range(count))
        self.names = ("finger_b", *self.arm_names[::-1], "finger_a")
        self.q = np.zeros((1, count + 2))
        self.q[0, [0, count + 1]] = .04
        self.pose = NS(raw_pose=np.array([[0., 0., 0., 1., 0., 0., 0.]]))

    def get_active_joints(self):
        return [NS(get_name=lambda name=name: name) for name in self.names]

    def get_qpos(self):
        return self.q.copy()

    def get_qvel(self):
        return np.zeros_like(self.q)

    def get_qlimits(self):
        limits = np.array([[-3., 3.]] * len(self.names))
        limits[[0, -1]] = [0., .04]
        return limits[None]


class MixedEnv:
    def __init__(self):
        self.unwrapped = self
        self.num_envs = 1
        self.control_timestep = .05
        self.elapsed_steps = np.array([0])
        self.actions = []
        self.agent = NS(agents=[], agents_dict={})
        action_spaces = {}
        for key, count in (("left", 6), ("right", 7)):
            robot = Robot(count)
            arm = NS(joints=[NS(get_name=lambda name=name: name) for name in robot.arm_names],
                     config=NS(normalize_action=False, use_delta=False, joint_names=robot.arm_names))
            agent = NS(robot=robot, control_mode="pd_joint_pos",
                       controller=NS(controllers={"arm": arm}),
                       gripper_joint_names=("finger_a", "finger_b"),
                       tcp=NS(pose=NS(to_transformation_matrix=lambda: np.eye(4)[None])))
            self.agent.agents.append(agent)
            self.agent.agents_dict[key] = agent
            action_spaces[key] = spaces.Box(np.array([-3.] * count + [-1.], dtype=np.float32),
                                            np.array([3.] * count + [1.], dtype=np.float32))
        self.left_agent, self.right_agent = self.agent.agents
        self.agent.single_action_space = spaces.Dict(action_spaces)

    def step(self, action):
        assert set(action) == {"left", "right"}
        self.actions.append({name: value.copy() for name, value in action.items()})
        for key, agent in self.agent.agents_dict.items():
            for name, value in zip(agent.robot.arm_names, action[key][:-1]):
                agent.robot.q[0, agent.robot.names.index(name)] = value
        self.elapsed_steps += 1
        return None, 0., np.array([False]), np.array([False]), {}


def test_named_binding_reads_all_joints_and_rejects_wrong_action_size():
    env = MixedEnv()
    arms = tuple(bind_arm(key, agent) for key, agent in env.agent.agents_dict.items())
    left = arms[0]
    assert left.indices.tolist() == [6, 5, 4, 3, 2, 1]
    assert left.finger_indices.tolist() == [7, 0]
    left.agent.robot.q[0, left.indices] = np.arange(6) / 10
    assert np.array_equal(left.arm_qpos(), np.arange(6) / 10)
    measured = _measure(arms)
    assert measured["left"]["finger_qpos_m"] == [.04, .04]
    action = {arm.key: np.r_[arm.arm_qpos(), 1.] for arm in arms}
    _actions_valid(env, arms, action)
    with pytest.raises(ValueError, match="named arm"):
        _actions_valid(env, arms, {**action, "left": np.zeros(8)})
    left.agent.controller.controllers["arm"].joints[0] = NS(get_name=lambda: "missing")
    with pytest.raises(ValueError, match="actual active"):
        bind_arm("left", left.agent)


def test_shared_executor_steps_both_sized_actions_once_per_tick():
    env = MixedEnv()
    left, right = trajectory("left", 6, .1, .02), trajectory("right", 7, .2, .04)
    schedule = build_hold_schedule(left, right, control_timestep_s=.05)
    checker = NS(refresh=lambda: None, check=lambda *_: [],
                 set_target_contacts_allowed=lambda _: None, targets=(None, None))
    report = execute_dual_reach(env, schedule, checker, stable_steps=1)
    assert report["status"] == "passed"
    assert len(env.actions) == report["environment_steps"] == 4
    assert all(row["left"].shape == (7,) and row["right"].shape == (8,) for row in env.actions)
    assert np.allclose(env.actions[1]["left"], env.actions[-1]["left"])
    assert np.allclose(env.left_agent.robot.q[0, [6, 5, 4, 3, 2, 1]], .02)
    assert env.elapsed_steps.tolist() == [4]


def test_native_gripper_mapping_keeps_passive_joints_and_native_recorded_actions():
    env = MixedEnv()
    for key, agent in env.agent.agents_dict.items():
        robot = agent.robot
        passive = tuple(f"passive_{i}" for i in range(4))
        robot.names = (*robot.names, *passive)
        robot.q = np.column_stack((robot.q, np.zeros((1, 4))))
        agent.gripper_joint_names = ("finger_a", "finger_b", *passive)
        agent.gripper_open_action, agent.gripper_closed_action = 0., .81
        env.agent.single_action_space.spaces[key] = spaces.Box(
            np.array([-3.] * len(robot.arm_names) + [0.], dtype=np.float32),
            np.array([3.] * len(robot.arm_names) + [.81], dtype=np.float32))
    phase = _Phase(env, NS(refresh=lambda: None, check=lambda *_: [],
                          set_target_contacts_allowed=lambda _: None), "native-controller")
    assert all(len(arm.finger_indices) == 6 for arm in phase.arms)
    for arm in phase.arms:
        command = np.r_[arm.arm_qpos(), 1.]
        native = native_arm_action(arm, command)
        assert native[-1] == 0. and command[-1] == 1.
        command[-1] = -1.
        assert np.isclose(native_arm_action(arm, command)[-1], .81)
    phase.tick(phase.hold(1), "opening")
    phase.tick(phase.hold(-1), "closing")
    assert all(value[-1] == 0. for value in env.actions[0].values())
    assert all(np.isclose(value[-1], .81) for value in env.actions[1].values())
    assert env.elapsed_steps.tolist() == [2]
    assert len(phase.report["trace"][0]["measurements"]["left"]["finger_qpos_m"]) == 6
    # H5 rows already use native 0/.81. Replay must not apply the mapping twice.
    for recorded_grip in (0., .81):
        recorded = {arm.key: np.r_[arm.arm_qpos(), recorded_grip].astype(np.float32)
                    for arm in phase.arms}
        before = {key: value.copy() for key, value in recorded.items()}
        phase.tick(recorded, 'replay', native_actions=True)
        for key in recorded:
            np.testing.assert_array_equal(env.actions[-1][key], before[key])
            np.testing.assert_array_equal(recorded[key], before[key])
    invalid = {arm.key: np.r_[arm.arm_qpos(), -.1].astype(np.float32) for arm in phase.arms}
    with pytest.raises(ValueError, match='actual controller action space'):
        phase.tick(invalid, 'invalid-native', native_actions=True)


def test_three_axis_finger_motion_uses_same_physical_distance_in_generation_and_replay():
    before = np.zeros((2, 3))
    after = np.array([[.00015, .00015, .00015], [0., 0., 0.]])
    assert np.max(np.abs(after-before)) < .0002
    assert finger_displacement_m(after, before) > .0002
    assert finger_displacement_m(np.array([.0001, .00019]), np.zeros(2)) == .00019


def test_rotary_finger_stability_is_measured_in_tcp_metres():
    env = MixedEnv()
    agent = env.left_agent
    agent.gripper_coordinate_unit = "rad"
    tcp = np.eye(4)
    tcp[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
    tcp[:3, 3] = [.4, -.2, .3]
    local = np.array([[0., -.02, .01], [0., .02, .01]])
    agent.tcp.pose.to_transformation_matrix = lambda: tcp[None].copy()
    agent.finger1_link, agent.finger2_link = [
        NS(pose=NS(p=(point @ tcp[:3, :3].T + tcp[:3, 3])[None])) for point in local]
    arms = tuple(bind_arm(key, robot) for key, robot in env.agent.agents_dict.items())
    observed = _measure(arms)
    assert np.allclose(observed["left"]["finger_positions_tcp_m"], local)
    assert observed["left"]["gripper_coordinate_unit"] == "rad"
    assert observed["left"]["gripper_velocity_unit"] == "rad/s"
    assert "finger_qpos_m" not in observed["left"]
    assert "finger_qvel_m_s" not in observed["left"]
    # Correctly measured prismatic joints retain their original metre contract.
    assert "finger_positions_tcp_m" not in observed["right"]
    # Rigid arm movement does not look like finger opening or rotary motion.
    translation = np.array([.1, .2, .3])
    tcp[:3, 3] += translation
    agent.finger1_link.pose.p += translation
    agent.finger2_link.pose.p += translation
    assert np.allclose(_measure(arms)["left"]["finger_positions_tcp_m"], local)


def test_rotary_open_uses_explicit_coordinate_tolerance_without_metre_labels():
    env = MixedEnv()
    agent = env.left_agent
    agent.gripper_coordinate_unit = "rad"
    agent.gripper_open_qpos = 0.
    agent.gripper_open_tolerance = .002
    arm = bind_arm("left", agent)
    assert gripper_open_settings(arm) == {"coordinate_unit": "rad", "tolerance": .002}
    assert gripper_is_open(arm, [.0015, -.0015])
    assert not gripper_is_open(arm, [.0025, -.0015])
    phase = _Phase(env, NS(refresh=lambda: None, check=lambda *_: [],
                          set_target_contacts_allowed=lambda _: None), "rotary-open")
    assert "finger_open_m" not in phase.report["thresholds"]
    assert phase.report["thresholds"]["gripper_open"]["left"]["coordinate_unit"] == "rad"


def test_open_detection_uses_declared_controller_target_with_panda_fallback():
    env = MixedEnv()
    checker = NS(refresh=lambda: None, check=lambda *_: [],
                 set_target_contacts_allowed=lambda _: None, targets=(None, None))
    phase = _Phase(env, checker, "open-target")
    initial = phase.report["initial_observation"]
    assert all(phase.opened(arm, initial) for arm in phase.arms)
    left = phase.arms[0]
    left.agent.gripper_open_qpos = .03  # Controller travel may stop short of URDF limits.
    left.agent.robot.q[0, left.finger_indices] = .03
    assert phase.opened(left, phase.observe())
    del left.agent.gripper_open_qpos
    assert not phase.opened(left, phase.observe())


def test_native_pair_query_sets_both_full_models_and_preserves_fingers():
    states = (np.r_[.03, np.zeros(6), .04], np.r_[np.zeros(7), .02, .01])
    indices = (np.arange(1, 7), np.arange(7))
    models = [NS(set_qpos=lambda q, full, index=index: actual.__setitem__(index, q.copy()))
              for index in range(2)]
    actual = [None, None]
    queried = []

    def collide_full(index):
        assert actual[0] is not None and actual[1] is not None
        queried.append(index)
        assert np.allclose(actual[0][indices[0]], .1)
        assert np.allclose(actual[1][indices[1]], -.1)
        assert np.array_equal(actual[0][[0, 7]], [.03, .04])
        assert np.array_equal(actual[1][-2:], [.02, .01])
        return []

    checker = NativeDualCollisionChecker(models, NS(collide_full=collide_full), states,
        indices, (np.array([[-1, 1]] * 6), np.array([[-1, 1]] * 7)), ())
    assert checker.check(np.full(6, .1), np.full(7, -.1)) == []
    assert queried == [0, 1]
    with pytest.raises(ValueError, match="shape"):
        checker.check(np.zeros(7), np.zeros(7))


def test_finger_allowlist_uses_actual_links_and_physical_payload_pair():
    lives = tuple(NS(finger_links={"left": frozenset(("piper_left_pad", "piper_right_pad")),
                                  "right": frozenset(("xarm_left_pad", "xarm_right_pad"))},
                     allowed_mount_contacts=frozenset()) for _ in range(2))
    checker = DualPayloadCollisionChecker()
    checker.lives, checker._target_names = lives, (frozenset(("cube_left",)), frozenset(("cube_right",)))
    checker.attached, checker.target_contacts_allowed = True, (False, False)
    record = {"object_name1": "left", "link_name1": "piper_left_pad", "object_name2": "cube_left"}
    assert checker._allowed(record)
    assert not checker._allowed({**record, "object_name2": "cube_right"})
    assert not checker._allowed({**record, "link_name1": "panda_leftfinger"})
    checker.attached = False
    assert not checker._allowed(record)
    checker.target_contacts_allowed = (True, False)
    assert checker._allowed(record)


def test_box_support_uses_actual_half_sizes_and_candidate_pose():
    material = physx.PhysxMaterial(.5, .5, 0.)
    shape = physx.PhysxCollisionShapeBox([.012, .014, .016], material)
    live = NS(_shape_refs={"cube": (sapien.Entity(), shape)})
    vertices = _scaled_vertices(live, "cube")
    assert vertices.shape == (8, 3)
    assert np.allclose(vertices.min(axis=0), [-.012, -.014, -.016])
    assert np.allclose(vertices.max(axis=0), [.012, .014, .016])
    checker = NumericalTableSupportCollisionChecker()
    checker.live = live
    checker._initial_object = np.eye(4)
    checker._initial_object[2, 3] = .016 - 1e-7
    checker._shape_local = {"cube": np.eye(4)}
    checker._target_names, checker._support_pairs = frozenset(("cube",)), {}
    checker.support_released = False
    checker.support_band_m = checker.support_penetration_m = .003
    checker.support_depth_slop_m = .0002
    table_pose = np.eye(4)
    table_pose[2, 3] = -.05
    objects = {TABLE_NAME: _object_at(fcl.Box([1., 1., .1]), table_pose),
               "cube": _object_at(fcl.Box([.024, .028, .032]), checker._initial_object)}
    checker._register_initial_support(objects)
    assert checker._legacy_support_mode is None and checker._initial_support_exists
    assert checker._initial_geometry_export["cube"]["scaled_shape_local_vertices"] == vertices.tolist()
    candidate = checker._initial_object.copy()
    candidate[2, 3] -= .004
    collision, contacts = _pair_result(_object_at(fcl.Box([.024, .028, .032]), candidate), objects[TABLE_NAME])
    assert collision and contacts
    assert not checker._support_allowed("cube", TABLE_NAME, candidate, contacts)
    assert not checker.support_released


def test_six_joint_endpoint_preserves_named_fingers_in_shuffled_native_order():
    names = ("finger_a", "j5", "j4", "j3", "j2", "j1", "j0", "finger_b")
    robot = Robot(6)
    robot.q[0, 0], robot.q[0, -1] = .031, .027
    indices = np.array([names.index(f"j{i}") for i in range(6)])
    source = robot.q[0, [robot.names.index(name) for name in names]]
    goal = np.array([.1, .2, .3, 1., 0., 0., 0.])
    solver = NS(robot=robot, base_env=NS(control_timestep=.05), use_point_cloud=False,
                last_native_result=None)
    fk_state = []

    def fk(full):
        assert np.array_equal(full[[0, -1]], [.027, .031])
        fk_state[:] = [np.r_[full[indices][:3], 1., 0., 0., 0.]]

    def topp(path, step, verbose=False):
        return np.array([0., .1]), path, np.zeros_like(path), np.zeros_like(path), .1

    planner = NS(user_joint_names=names, move_group_joint_indices=indices, move_group="tcp",
                 link_name_2_idx={"tcp": 0}, transform_goal_to_wrt_base=lambda value: value,
                 TOPP=topp, pinocchio_model=NS(compute_forward_kinematics=fk,
                                             get_link_pose=lambda _: fk_state[0]))
    solver.planner = planner

    def screw(pose, *, dry_run):
        assert dry_run
        endpoint = source[indices].copy()
        endpoint[:3] = pose[:3]
        values = planner.TOPP(np.vstack((source[indices], endpoint)), .05)
        return {"status": "Success", **dict(zip(("time", "position", "velocity", "acceleration", "duration"), values))}

    solver.move_to_pose_with_screw = screw
    result = checked_screw_plan(solver, goal)
    assert result["position"].shape == (2, 6)
    assert result["endpoint_audit"]["status"] == "passed"
    assert result["endpoint_audit"]["preserved_non_arm_joint_positions"] == {"finger_a": .027, "finger_b": .031}
    assert np.array_equal(robot.q[0, [0, -1]], [.031, .027])
