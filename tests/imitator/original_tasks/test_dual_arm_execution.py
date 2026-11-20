from __future__ import annotations
import math
from collections.abc import Sequence
from typing import Any, Callable
import numpy as np
from imitator.planning._tig_sync.dual_arm_execution import (
    JOINT_STEP, ARM_SPEED, _Phase, plan_dual_pose, execute_dual_reach, close_dual_and_confirm, execute_dual_held_move, release_dual_and_confirm, verify_dual_placement, _measure, _observe, _rotation_distance, DualArmSchedule
)
from imitator.planning._tig_sync.runtime_policy import (
    DEFAULT_POLICY
)


def self_test() -> dict[str, Any]:
    """Analytical CPU fixtures use actual schedule/controller contracts, no physics."""
    from types import SimpleNamespace as NS
    from gymnasium import spaces

    names = tuple(f"joint{index}" for index in range(7))

    class Pose:
        def __init__(self, matrix: Any = None): self.matrix = np.eye(4) if matrix is None else np.array(matrix, dtype=float, copy=True)
        def to_transformation_matrix(self) -> np.ndarray: return self.matrix[None].copy()
        @property
        def raw_pose(self) -> np.ndarray: return np.r_[self.matrix[:3, 3], 1., 0., 0., 0.][None]

    class Robot:
        def __init__(self, holding: bool):
            self.q = np.array([[0.] * 7 + [.024 if holding else .04] * 2]); self.v = np.zeros((1, 9)); self.pose = Pose()
            self.qpos_reads = self.qvel_reads = 0
        def get_qpos(self) -> np.ndarray:
            self.qpos_reads += 1
            return self.q.copy()
        def get_qvel(self) -> np.ndarray:
            self.qvel_reads += 1
            return self.v.copy()
        def get_qlimits(self) -> np.ndarray: return np.array([[[-3., 3.]] * 7 + [[0., .04]] * 2])
        def get_active_joints(self) -> list[Any]: return [NS(get_name=lambda name=name: name) for name in (*names, "f1", "f2")]

    class Actor:
        def __init__(self): self.pose = Pose(); self.linear = np.zeros((1, 3)); self.angular = np.zeros((1, 3))
        def get_linear_velocity(self) -> np.ndarray: return self.linear.copy()
        def get_angular_velocity(self) -> np.ndarray: return self.angular.copy()

    class Env:
        def __init__(self, holding: bool = True, mode: str = "normal"):
            self.unwrapped = self; self.num_envs = 1; self.control_timestep = .05; self.elapsed_steps = np.array([0])
            self.mode = mode; self.actions = []; self.targets = (Actor(), Actor()); self.grasp = [holding, holding]
            self.agent = NS(agents=[], agents_dict={})
            for index, key in enumerate(("left", "right")):
                robot = Robot(holding)
                controller = NS(joints=robot.get_active_joints()[:7], config=NS(normalize_action=False, use_delta=False, joint_names=names))
                agent = NS(robot=robot, control_mode="pd_joint_pos", controller=NS(controllers={"arm": controller}), tcp=NS(pose=Pose()))
                agent.is_grasping = lambda target, index=index: np.array([self.grasp[index]])
                self.agent.agents.append(agent); self.agent.agents_dict[key] = agent
            self.left_agent, self.right_agent = self.agent.agents
            box = spaces.Box(np.array([-3.] * 7 + [-1.], dtype=np.float32), np.array([3.] * 7 + [1.], dtype=np.float32))
            self.agent.single_action_space = spaces.Dict({"left": box, "right": box})
            self.update()

        def update(self) -> None:
            for index, agent in enumerate(self.agent.agents):
                matrix = np.eye(4); matrix[:3, 3] = agent.robot.q[0, :3] * [1., 1., 1.]
                matrix[0, 3] += -.3 if index == 0 else .3
                agent.tcp.pose.matrix = matrix
                if self.grasp[index]: self.targets[index].pose.matrix = matrix @ attachment(index)

        def step(self, action: dict[str, Any]) -> tuple[Any, ...]:
            assert list(action) == ["left", "right"] and set(action) == set(self.agent.agents_dict)
            self.actions.append({key: np.array(value, copy=True) for key, value in action.items()})
            self.elapsed_steps += 1; tick = len(self.actions)
            for index, (key, agent) in enumerate(self.agent.agents_dict.items()):
                agent.robot.q[0, :7] = action[key][:7]
                closed = action[key][7] == -1 or (self.mode == "retain_right" and index == 1)
                self.grasp[index] = closed and not (self.mode == "empty_right" and index == 1)
                agent.robot.q[0, 7:] = .024 if closed else .04
                if self.mode == "delayed_right" and index == 1 and tick <= 3: agent.robot.q[0, 7:] += .001 * (tick % 2)
                if self.mode == "joint_fd": agent.robot.q[0, 0] += .004 * tick
                if self.mode == "hover_drift": agent.robot.q[0, 0] += .0002 * tick
                agent.robot.v[:] = 1.6 if self.mode == "high_raw" else 0.
            self.update()
            if self.mode == "slip_right": self.targets[1].pose.matrix[0, 3] += .006
            if self.mode == "drop_right": self.targets[1].pose.matrix[2, 3] -= .012
            if self.mode == "rotation_right":
                angle = .2; self.targets[1].pose.matrix[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            if self.mode == "placement_drift": self.targets[1].pose.matrix[0, 3] += .0002
            for target in self.targets:
                target.angular[:] = [0., 0., 1.6] if self.mode == "high_raw" else 0.
            return None, 0., np.array([self.mode == "terminated"]), np.array([self.mode == "truncated"]), {}

        def evaluate(self) -> dict[str, Any]:
            inside = not (self.mode == "outside_right" or (self.mode == "bounce_right" and len(self.actions) == 11))
            return {"apple_in_basket": np.array([True]), "banana_in_basket": np.array([inside])}

    def attachment(index: int) -> np.ndarray:
        value = np.eye(4); value[2, 3] = .02 + .01 * index
        return value

    class Checker:
        def __init__(self, env: Env, *, held: bool = True, mode: str = "normal"):
            self.env, self.targets, self.attached, self.mode = env, env.targets, held, mode
            self.scope = "analytical_fixture"; self.allowed = (False, False); self.candidates = []
        @property
        def attachments(self) -> tuple[np.ndarray, np.ndarray]:
            if not self.attached: raise ValueError("not attached")
            return attachment(0), attachment(1)
        def set_target_contacts_allowed(self, value: Any) -> None: self.allowed = value
        def refresh(self) -> None: pass
        def check(self, left: np.ndarray, right: np.ndarray) -> list[dict[str, Any]]:
            self.candidates.append((left.copy(), right.copy()))
            return [{"fixture": "candidate"}] if self.mode == "candidate" else []
        def measured_check(self) -> list[dict[str, Any]]:
            return [{"fixture": "live"}] if self.mode == "measured" and self.env.actions else []

    class Solver:
        """Analytical native-style planner; the real endpoint adapter still runs."""
        def __init__(self, env: Env, index: int):
            self.base_env = env; self.env_agent = env.agent.agents[index]; self.gripper_state = 1; self.index = index; self.calls = 0
            self.robot = self.env_agent.robot
            self.use_point_cloud = False
            self.last_native_result = None
            self.topp_calls = 0
            self.base_xyz = np.array([-.3 if index == 0 else .3, 0., 0.])
            self.planner = NS(user_joint_names=(*names, "f1", "f2"), move_group_joint_indices=list(range(7)),
                move_group="tcp", link_name_2_idx={"tcp": 0}, planning_world=NS(use_attach=False),
                transform_goal_to_wrt_base=self.goal_in_base, TOPP=self.topp, plan_screw=self.screw,
                pinocchio_model=NS(compute_forward_kinematics=self.fk, get_link_pose=lambda _: self.fk_pose.copy()))

        def goal_in_base(self, value: Any) -> np.ndarray:
            goal = np.array(value, dtype=float, copy=True)
            assert goal.shape == (7,)
            goal[:3] -= self.base_xyz
            return goal

        def fk(self, full_qpos: Any) -> None:
            full = np.asarray(full_qpos)
            assert full.shape == (9,) and np.array_equal(full[7:], self.robot.q[0, 7:])
            self.fk_pose = np.r_[full[:3], 1., 0., 0., 0.]

        def topp(self, path: Any, step: float, verbose: bool = False) -> Any:
            points = np.array(path, dtype=float, copy=True)
            assert points.shape == (2, 7) and step == self.base_env.control_timestep
            self.topp_calls += 1
            duration = .1 if self.index == 0 else .2
            velocity = np.repeat(((points[1] - points[0]) / duration)[None], 2, axis=0)
            return np.array([0., duration]), points, velocity, np.zeros_like(points), duration

        def screw(self, value: Any, full_qpos: Any, **kwargs: Any) -> dict[str, Any]:
            full = np.array(full_qpos, dtype=float, copy=True)
            assert full.shape == (9,) and np.array_equal(full[7:], self.robot.q[0, 7:])
            goal = self.goal_in_base(value)
            assert np.array_equal(goal[3:], [1., 0., 0., 0.])
            start = full[:7].copy(); end = start.copy(); end[:3] = goal[:3]
            raw = self.planner.TOPP(np.vstack((start, end)), self.base_env.control_timestep)
            return {"status": "Success", **dict(zip(("time", "position", "velocity", "acceleration", "duration"), raw))}

        def move_to_pose_with_screw(self, value: Any, *, dry_run: bool) -> dict[str, Any]:
            assert dry_run and not self.base_env.actions
            self.calls += 1
            result = self.screw(value, self.robot.get_qpos()[0], time_step=self.base_env.control_timestep)
            self.last_native_result = result
            return result

    def target_pose(env: Env, index: int, delta: Any) -> np.ndarray:
        # Preserve the old fixture's joint-translation increments as XYZ/WXYZ.
        xyz = env.agent.agents[index].robot.q[0, :3].copy()
        delta = np.asarray(delta)
        xyz[:len(delta)] += delta
        xyz[0] += -.3 if index == 0 else .3
        return np.r_[xyz, 1., 0., 0., 0.]

    def schedule(env: Env, *, held: bool = True, lift: bool = False) -> DualArmSchedule:
        deltas = ([0., 0., .12], [0., 0., .12]) if lift else ([.04], [.08])
        poses = tuple(target_pose(env, index, delta) for index, delta in enumerate(deltas))
        return plan_dual_pose(Solver(env, 0), Solver(env, 1), poses, gripper_commands=(-1, -1) if held else (1, 1))

    def reject(call: Callable[[], Any]) -> None:
        try: call()
        except (ValueError, TypeError): return
        raise AssertionError("Invalid input was accepted")

    checks = []
    env = Env(); phase = _Phase(env, Checker(env), "observation", targets=env.targets, held=True)
    robots = [agent.robot for agent in env.agent.agents]
    for robot in robots:
        robot.qpos_reads = robot.qvel_reads = 0
    observed = phase.observe()
    assert all((robot.qpos_reads, robot.qvel_reads) == (1, 1) for robot in robots)
    # Reconstruct the previous three-read path at the same physical state,
    # including each object's old single-arm default. Its output is identical.
    previous_objects = []
    for index, (arm, target) in enumerate(zip(phase.arms, phase.targets)):
        value = _observe((arm, phase.arms[1-index]), target, phase.references[index], phase.initial_z[index])
        value.pop("measurements")
        value["rotation_slip_rad"] = _rotation_distance(phase.references[index][:3, :3], np.asarray(value["T_tcp_object"])[:3, :3])
        value.update(target_present=True, action_key=arm.key, object_index=index)
        previous_objects.append(value)
    previous = {"measurements": _measure(phase.arms), "objects": previous_objects}
    assert observed == previous and all((robot.qpos_reads, robot.qvel_reads) == (4, 4) for robot in robots)
    action = phase.hold(-1)
    for value in action.values():
        value[0] += .01
    env.step(action)
    fresh = phase.observe()
    assert env.elapsed_steps.tolist() == [1] and all((robot.qpos_reads, robot.qvel_reads) == (5, 5) for robot in robots)
    assert all(observed["measurements"][arm.key]["qpos_rad"][0] == 0.
               and fresh["measurements"][arm.key]["qpos_rad"][0] == .01 for arm in phase.arms)
    checks.append("same-observation joint reads reduced from three to one with identical values and fresh reads after stepping")
    env = Env(False); solvers = (Solver(env, 0), Solver(env, 1)); planned = plan_dual_pose(*solvers, (target_pose(env, 0, [.04]), target_pose(env, 1, [.08])))
    assert not env.actions and all(solver.calls == 1 for solver in solvers)
    assert all(solver.topp_calls == 1 and solver.last_endpoint_audit["status"] == "passed"
               and solver.last_endpoint_audit["corrections"] == 0 for solver in solvers)
    assert np.allclose(planned.left_commands[2:, 0], .04) and planned.num_steps == 4
    reject(lambda: plan_dual_pose(solvers[1], solvers[0], (None, None)))
    held = plan_dual_pose(*solvers, (None, target_pose(env, 1, [.08]))); assert solvers[0].calls == 1 and np.all(held.left_commands[:, :7] == 0)
    assert solvers[0].last_native_result is None and solvers[1].last_native_result is not None
    checks.append("dry-run both planners, canonical identity, unequal endpoint hold and None hold")
    reached = execute_dual_reach(env, planned, Checker(env, held=False), stable_steps=2)
    assert reached["status"] == "passed" and len(env.actions) == reached["environment_steps"]
    assert all(list(row["measurements"]) == ["left", "right"] and len(row["objects"]) == 2 for row in reached["trace"])
    checks.append("one paired step per control tick and canonical object/measurement arrays")
    env = Env(False, "delayed_right"); checker = Checker(env, held=False)
    closed = close_dual_and_confirm(env, checker, env.targets)
    assert closed["status"] == "passed" and len(env.actions) >= 8 and min(closed["stable_steps_observed"]) >= 5
    assert checker.allowed == (False, False)
    checks.append("both contact grasps and per-arm finger stability phase barrier")
    env = Env(False, "empty_right"); failed = close_dual_and_confirm(env, Checker(env, held=False), env.targets, max_steps=8)
    assert failed["status"] == "grasp_timeout" and len(env.actions) == 8
    checks.append("one empty grasp cannot pass and bounded timeout stops both")
    env = Env(); lifted = execute_dual_held_move(env, schedule(env, lift=True), Checker(env), env.targets, min_height_gain_m=(.10, .10), hold_steps=20)
    assert lifted["status"] == "passed" and lifted["hold_steps_completed"] == 20 and lifted["settling_steps"] >= 3
    assert all(obj["object_height_gain_m"] >= .10 for obj in lifted["trace"][-1]["objects"])
    assert len(env.actions) == lifted["environment_steps"] and lifted["trace"][-1]["step"] == len(env.actions)
    checks.append("both actual height gains, explicit settling, 20 shared hover ticks")
    env = Env(mode="high_raw"); result = execute_dual_held_move(env, schedule(env, lift=True), Checker(env), env.targets, min_height_gain_m=(.10, .10), hold_steps=20)
    assert result["status"] == "passed" and result["trace"][-1]["max_arm_velocity_rad_s"] == 1.6
    assert result["trace"][-1]["max_control_frame_joint_speed_rad_s"] == 0
    assert all(obj["object_angular_speed_rad_s"] == 1.6 for obj in result["trace"][-1]["objects"])
    checks.append("held pose FD preserves high raw joint/object velocity telemetry")
    env = Env(mode="joint_fd"); result = execute_dual_held_move(env, schedule(env), Checker(env), env.targets, settle_max_steps=3)
    assert result["status"] == "held_move_timeout" and result["trace"][-1]["max_arm_velocity_rad_s"] == 0
    assert result["trace"][-1]["max_control_frame_joint_speed_rad_s"] > ARM_SPEED
    env = Env(); asymmetric = plan_dual_pose(Solver(env, 0), Solver(env, 1), (target_pose(env, 0, [0., 0., .12]), target_pose(env, 1, [0., 0., .08])), gripper_commands=(-1, -1))
    result = execute_dual_held_move(env, asymmetric, Checker(env), env.targets, min_height_gain_m=(.10, .10), settle_max_steps=3)
    assert result["status"] == "held_move_timeout" and result["trace"][-1]["objects"][0]["object_height_gain_m"] >= .10
    assert result["trace"][-1]["objects"][1]["object_height_gain_m"] < .10
    checks.append("measured joint FD and the second object's height independently block readiness")
    for mode, reason in (("slip_right", "position_slip"), ("drop_right", "object_dropped"), ("rotation_right", "rotation_slip"), ("empty_right", "grasp_lost")):
        env = Env(mode=mode)
        candidate = (plan_dual_pose(Solver(env, 0), Solver(env, 1), (None, None), gripper_commands=(-1, -1))
                     if mode == "empty_right" else schedule(env))
        result = execute_dual_held_move(env, candidate, Checker(env), env.targets)
        assert result["status"] == "aborted" and result.get("reason") == reason and result.get("object_index") == 1, (mode, result["status"], result.get("reason"))
        assert len(env.actions) <= 2
    checks.append("either object slip, rotation, drop or two-tick grasp loss aborts both")
    env = Env(mode="hover_drift"); result = execute_dual_held_move(env, schedule(env, lift=True), Checker(env), env.targets, hold_steps=20)
    assert result["status"] == "aborted" and result["reason"] == "hold_not_stable"
    checks.append("cumulative hover joint/object drift cannot hide under low FD speed")
    env = Env(); result = execute_dual_held_move(env, schedule(env), Checker(env, mode="candidate"), env.targets)
    assert result["status"] == "rejected" and not env.actions
    env = Env(); result = execute_dual_held_move(env, schedule(env), Checker(env, mode="measured"), env.targets)
    assert result["reason"] == "measured_collision" and len(env.actions) == 1
    checks.append("candidate collision rejects before motion and live collision stops after one tick")
    for mode in ("terminated", "truncated"):
        env = Env(mode=mode); result = execute_dual_held_move(env, schedule(env), Checker(env), env.targets)
        assert result["status"] == mode and len(env.actions) == 1
    checks.append("received termination and truncation stop without masking")
    env = Env(); checker = Checker(env, held=False); released = release_dual_and_confirm(env, checker, env.targets)
    assert released["status"] == "passed" and len(env.actions) == 3 and all(released["trace"][-1]["fingers_open"])
    env = Env(mode="retain_right"); checker = Checker(env, held=False); retained = release_dual_and_confirm(env, checker, env.targets, max_steps=7)
    assert retained["status"] == "release_timeout" and len(env.actions) == 7
    assert not retained["trace"][-1]["objects"][0]["is_grasping"] and retained["trace"][-1]["objects"][1]["is_grasping"]
    checks.append("left releases while right retain-grip times out without continuation")
    predicates = (lambda target: {"inside": True, "height_bounds_m": [0., .3]},) * 2
    env = Env(False); placed = verify_dual_placement(env, Checker(env, held=False), env.targets, region_predicates=predicates)
    assert placed["status"] == "passed" and len(env.actions) == 20
    env = Env(False); result = verify_dual_placement(env, Checker(env, held=False), env.targets,
        region_predicates=(predicates[0], lambda actor: {"inside": False}), max_steps=21)
    assert result["status"] == "placement_timeout" and len(env.actions) == 21
    for mode in ("outside_right", "placement_drift"):
        env = Env(False, mode); result = verify_dual_placement(env, Checker(env, held=False), env.targets, region_predicates=predicates, max_steps=25)
        assert result["status"] == "placement_timeout" and len(env.actions) == 25
    env = Env(False, "bounce_right"); placed = verify_dual_placement(env, Checker(env, held=False), env.targets, region_predicates=predicates)
    assert placed["status"] == "passed" and len(env.actions) == 31
    checks.append("both official/physical regions, 20-tick common window, bounce and slow drift reset")
    env = Env(False); reject(lambda: release_dual_and_confirm(env, Checker(env), env.targets))
    reject(lambda: verify_dual_placement(env, Checker(env, held=False), (None, env.targets[1]), region_predicates=predicates))
    reject(lambda: close_dual_and_confirm(env, Checker(env, held=False), env.targets[::-1]))
    reject(lambda: execute_dual_held_move(env, schedule(env), Checker(env), env.targets, min_height_gain_m=(None, -1.)))
    assert not env.actions
    checks.append("attached release and invalid targets/thresholds reject before stepping")
    from dataclasses import replace
    selected = replace(DEFAULT_POLICY, policy_id="cpu-fixture-selected-values",
        grasp_stable_steps=7, release_stable_steps=4, reach_stable_steps=4,
        final_stable_steps=22, max_settle_steps=6, position_slip_m=.004, rotation_slip_rad=.1)
    env = Env(False); closed = close_dual_and_confirm(env, Checker(env, held=False), env.targets, policy=selected)
    assert closed["status"] == "passed" and closed["stable_steps_required"] == 7 and len(env.actions) == 8
    env = Env(); released = release_dual_and_confirm(env, Checker(env, held=False), env.targets, policy=selected)
    assert released["status"] == "passed" and released["stable_steps_required"] == 4 and len(env.actions) == 4
    env = Env(False)
    candidate = plan_dual_pose(Solver(env, 0), Solver(env, 1),
        (target_pose(env, 0, [.04]), target_pose(env, 1, [.08])), policy=selected)
    reached = execute_dual_reach(env, candidate, Checker(env, held=False), policy=selected)
    assert reached["status"] == "passed" and reached["stable_steps_required"] == 4 and reached["settle_max_steps"] == 6
    env = Env(False); placed = verify_dual_placement(env, Checker(env, held=False), env.targets,
        region_predicates=predicates, policy=selected)
    assert placed["status"] == "passed" and placed["stable_steps_required"] == 22 and len(env.actions) == 22
    # Existing explicit caller overrides still resolve and are reported.
    env = Env(False); overridden = close_dual_and_confirm(env, Checker(env, held=False), env.targets,
        stable_steps=2, policy=selected)
    assert overridden["status"] == "passed" and overridden["stable_steps_required"] == 2
    checks.append("resolved policy supplies grasp/release/reach/final windows and settle defaults; explicit overrides remain visible")
    for field, value, reason in (("position_slip_m", .0045, "position_slip"),
                                 ("rotation_slip_rad", .12, "rotation_slip")):
        env = Env(); phase = _Phase(env, Checker(env), "policy-monitor-fixture", held=True, policy=selected)
        row = phase.observe(); row["objects"][1][field] = value
        assert not phase.monitor(row) and phase.report["reason"] == reason
        assert phase.report["thresholds"][field] == getattr(selected, field) and not env.actions
        env = Env(); legacy_phase = _Phase(env, Checker(env), "legacy-monitor-fixture", held=True)
        row = legacy_phase.observe(); row["objects"][1][field] = value
        assert legacy_phase.monitor(row) and not env.actions
    env = Env(); env.targets[1].pose.matrix[0, 3] += .0045
    result = execute_dual_held_move(env, schedule(env), Checker(env), env.targets, policy=selected)
    assert result["status"] == "rejected" and result["reason"] == "initial_grasp_or_attachment_invalid" and not env.actions
    assert DEFAULT_POLICY.position_slip_m == .005 and DEFAULT_POLICY.rotation_slip_rad == .15
    checks.append("selected slip thresholds govern initial and sampled held checks; legacy aliases stay unchanged")
    env = Env(False)
    reject(lambda: close_dual_and_confirm(env, Checker(env, held=False), env.targets, policy=None))
    assert not env.actions
    checks.append("invalid policy rejects before any paired step")
    class MeasuredChecker(Checker):
        """Business-only snapshot spy; real FCL is covered by its own fixture."""
        def __init__(self, env: Env, *, prediction: bool = True, reject_collision: bool = False):
            super().__init__(env)
            self.measured_attachment_prediction = prediction
            self.reject_collision = reject_collision
            self.refresh_count = 0; self.prediction_calls = []; self.current_step = None
        def refresh(self) -> None:
            self.refresh_count += 1; self.current_step = int(self.env.elapsed_steps[0])
        def validate_measured_command(self, left: Any, right: Any, *, grasping: Any,
                                      max_joint_step_rad: float) -> dict[str, Any]:
            self.prediction_calls.append({"step": int(self.env.elapsed_steps[0]),
                "refresh_count": self.refresh_count, "grasping": list(grasping)})
            assert max_joint_step_rad == JOINT_STEP
            if int(self.env.elapsed_steps[0]) != self.current_step:
                raise RuntimeError("fixture: stale physical control-step snapshot")
            reason = ("attachment_grasp_unconfirmed" if not all(grasping) else
                      "updated_candidate_collision" if self.reject_collision else None)
            return {"status": "rejected" if reason else "passed", "reason": reason,
                    "geometry_epoch": self.refresh_count, "control_step": self.current_step}

    default_env = Env(); default_checker = Checker(default_env)
    default_result = execute_dual_held_move(default_env, schedule(default_env), default_checker, default_env.targets)
    explicit_env = Env(); explicit_checker = MeasuredChecker(explicit_env, prediction=False)
    explicit_result = execute_dual_held_move(explicit_env, schedule(explicit_env), explicit_checker, explicit_env.targets)
    updated_env = Env(); updated_checker = MeasuredChecker(updated_env)
    updated_result = execute_dual_held_move(updated_env, schedule(updated_env), updated_checker, updated_env.targets)
    for result in (default_result, explicit_result, updated_result): assert result["status"] == "passed"
    assert len(default_env.actions) == len(explicit_env.actions) == len(updated_env.actions)
    for default, explicit, updated in zip(default_env.actions, explicit_env.actions, updated_env.actions):
        assert all(np.array_equal(default[key], explicit[key]) and np.array_equal(default[key], updated[key]) for key in default)
    assert not explicit_checker.prediction_calls
    assert len(updated_checker.prediction_calls) == len(updated_env.actions)
    assert updated_checker.refresh_count == len(updated_env.actions) + 1
    assert all(row["attachment_geometry"]["status"] == "passed" and row["control_command_submitted"]
               for row in updated_result["trace"])
    checks.append("default and explicit legacy retain identical paired actions; opt-in reuses exactly initial+post-tick refreshes")
    for cause in ("unconfirmed", "collision", "stale"):
        env = Env(); checker = MeasuredChecker(env, reject_collision=cause == "collision")
        phase = _Phase(env, checker, "measured-command-negative-fixture", held=True)
        assert phase.preflight()
        if cause == "unconfirmed": env.grasp[1] = False
        if cause == "stale": env.elapsed_steps += 1  # Telemetry mutation, no fake or real step.
        before = env.elapsed_steps.copy()
        assert phase.tick(phase.hold(-1), "moving") is None
        assert not env.actions and phase.report["environment_steps"] == 0
        assert np.array_equal(env.elapsed_steps, before) and checker.refresh_count == 1
        row = phase.report["trace"][-1]
        assert row["control_command_submitted"] is False
        if cause == "stale": assert row["elapsed_step_delta_on_error"] == [0]
        else: assert phase.report["reason"] == ("attachment_grasp_unconfirmed" if cause == "unconfirmed" else "updated_candidate_collision")
    checks.append("actual grasp-false, rejected geometry and stale snapshot submit zero additional fake environment steps")
    env = Env(mode="slip_right"); checker = MeasuredChecker(env)
    result = execute_dual_held_move(env, schedule(env), checker, env.targets)
    assert result["status"] == "aborted" and result["reason"] == "position_slip" and len(env.actions) == 1
    assert result["trace"][-1]["objects"][1]["position_slip_m"] > DEFAULT_POLICY.position_slip_m
    checks.append("measured prediction never reanchors or bypasses original fixed-reference slip quality monitor")
    return {"status": "passed", "gpu_used": False, "native_physics_tested": False,
            "checks": checks, "check_count": len(checks),
            "measured_prediction_fake_ticks": len(updated_env.actions),
            "measured_prediction_refreshes": updated_checker.refresh_count,
            "negative_cases_additional_fake_ticks": 0,
            "physics_steps": 0}


def test_runtime_contract():
    assert self_test()["status"] == "passed"
