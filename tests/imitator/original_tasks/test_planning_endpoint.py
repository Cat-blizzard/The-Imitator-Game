from __future__ import annotations
import math
from collections.abc import Sequence
from typing import Any
import numpy as np
from imitator.planning._tig_sync.planning_endpoint import (
    POSITION_TOLERANCE_M, ROTATION_TOLERANCE_RAD, _array, _pose, pose_error, checked_screw_plan
)


def self_test() -> dict[str, Any]:
    """Exercise correction, frame/order handling and failure without native libs."""
    from types import SimpleNamespace

    class Fixture:
        def __init__(self, bias: float) -> None:
            self.indices = np.array([2, 0, 1, 4, 3, 5, 6])
            self.state = np.array([0., 0., 0., 0., 0., 0., 0., .031, .027])
            self.bias, self.calls, self.topp_calls = bias, 0, []
            self.env_agent = SimpleNamespace(control_mode="pd_joint_pos")
            self.base_env = SimpleNamespace(control_timestep=.05)
            self.robot = SimpleNamespace(get_qpos=lambda: self.state[None].copy())
            self.gripper_state, self.use_point_cloud = -1, True
            self.last_native_result = None
            self.planner = SimpleNamespace(user_joint_names=[f"j{i}" for i in range(9)],
                move_group_joint_indices=self.indices.tolist(), move_group="tcp",
                link_name_2_idx={"tcp": 0}, planning_world=SimpleNamespace(use_attach=True),
                transform_goal_to_wrt_base=lambda pose: np.r_[pose[:3] - [1., 2., 3.], pose[3:]],
                TOPP=self.topp, plan_screw=self.screw, plan_qpos_to_qpos=self.joint_path)
            self.planner.pinocchio_model = SimpleNamespace(
                compute_forward_kinematics=self.fk, get_link_pose=lambda _: self.actual)

        def fk(self, full: np.ndarray) -> None:
            assert np.array_equal(full[7:], self.state[7:])
            self.actual = np.r_[full[self.indices][:3], 1., 0., 0., 0.]

        def topp(self, path: np.ndarray, step: float, verbose: bool = False) -> Any:
            self.topp_calls.append(path.copy())
            self.last_native_parameterization = path.copy()
            return np.linspace(0., .1, len(path)), path.copy(), path * 0, path * 0, .1

        def screw(self, goal: np.ndarray, full: np.ndarray, **kwargs: Any) -> Any:
            self.calls += 1
            assert np.array_equal(full[7:], self.state[7:])
            if self.calls > 1:
                assert kwargs["use_point_cloud"] and kwargs["use_attach"]
            end = full[self.indices].copy()
            end[:3] = goal[:3] - [1., 2., 3.]
            if self.calls == 1:
                end[0] -= self.bias
            raw = self.planner.TOPP(np.vstack((full[self.indices], end)), .05)
            result = dict(zip(("time", "position", "velocity", "acceleration", "duration"), raw))
            return {"status": "Success", **result}

        def move_to_pose_with_screw(self, pose: Any, *, dry_run: bool) -> Any:
            assert dry_run
            return self.screw(_pose(pose), self.state.copy())

        def joint_path(self, goals: Any, full: Any, **kwargs: Any) -> Any:
            assert np.array_equal(full, self.state)
            return self.planner.TOPP(np.vstack((full[self.indices], goals[0][self.indices])), .05)

    goal = np.array([1.1, 2.2, 3.3, 1., 0., 0., 0.])
    solver = Fixture(.014)
    state = solver.state.copy()
    original_topp = solver.planner.TOPP
    result = checked_screw_plan(solver, goal)
    assert solver.planner.TOPP == original_topp and solver.calls == 2
    assert solver.last_native_result is result
    assert np.array_equal(solver.last_native_parameterization, solver.topp_calls[-1])
    assert len(solver.topp_calls) == 2 and len(solver.topp_calls[-1]) == 2
    assert not np.any(np.all(np.isclose(solver.topp_calls[-1],
        solver.topp_calls[0][-1], atol=1e-12, rtol=0), axis=1))
    assert np.array_equal(solver.state, state)
    assert result["endpoint_audit"]["final_error"]["position_m"] < 1e-12
    assert result["endpoint_audit"]["attempts"][0]["position_m"] > .013
    assert result["endpoint_audit"]["corrections"] == 1
    clean = Fixture(0.)
    checked_screw_plan(clean, goal)
    assert clean.calls == 1 and len(clean.topp_calls) == 1
    bad = Fixture(.014)
    try:
        checked_screw_plan(bad, goal, max_corrections=0)
    except ValueError:
        assert bad.last_endpoint_audit["status"] == "failed"
        assert bad.planner.TOPP == bad.topp
    else:
        raise AssertionError("Uncorrected endpoint must be rejected")

    class SmallAngleFixture(Fixture):
        """Model native IK's mixed-distance acceptance, not the CLIK epsilon."""
        def __init__(self, *, position_residual: float = 8e-6,
                     rotation_residual: float = 16e-6,
                     force_candidate: bool = False, change_finger: bool = False,
                     native_failure: bool = False) -> None:
            super().__init__(.008)
            self.position_residual = position_residual
            self.rotation_residual = rotation_residual
            self.force_candidate, self.change_finger = force_candidate, change_finger
            self.native_failure = native_failure
            self.ik_calls: list[dict[str, Any]] = []
            self.joint_calls: list[np.ndarray] = []
            self.planner.IK = self.ik

        def fk(self, full: np.ndarray) -> None:
            assert np.array_equal(full[7:], self.state[7:])
            arm = full[self.indices]
            self.actual = np.r_[arm[:3], np.cos(arm[3] / 2), 0., 0., np.sin(arm[3] / 2)]

        def screw(self, goal: np.ndarray, full: np.ndarray, **kwargs: Any) -> Any:
            self.calls += 1
            end = full[self.indices].copy()
            end[:3] = goal[:3] - [1., 2., 3.]
            end[0] -= self.bias
            end[3] = .002
            raw = self.planner.TOPP(np.vstack((full[self.indices], end)), .05)
            return {"status": "Success", **dict(zip(
                ("time", "position", "velocity", "acceleration", "duration"), raw))}

        def ik(self, target: np.ndarray, full: np.ndarray, *, mask: list[bool],
               n_init_qpos: int, threshold: float) -> tuple[str, list[np.ndarray]]:
            assert n_init_qpos == 1
            assert mask == [False] * 7 + [True, True]
            assert np.array_equal(full[7:], self.state[7:])
            candidate = self.state.copy()
            arm = candidate[self.indices].copy()
            arm[:3] = target[:3]
            arm[0] += self.position_residual
            arm[3] = self.rotation_residual
            candidate[self.indices] = arm
            distance = self.position_residual + 2 * np.sin(self.rotation_residual / 4)
            self.ik_calls.append({"threshold": threshold, "mixed_distance": float(distance),
                                  "mask": mask.copy()})
            if self.native_failure:
                return "IK Failed! Cannot find valid solution.", []
            if distance >= threshold and not self.force_candidate:
                return "IK Failed! Distance is greater than threshold.", []
            if self.change_finger:
                candidate[7] += .001
            return "Success", [candidate]

        def joint_path(self, goals: Any, full: Any, **kwargs: Any) -> Any:
            assert kwargs["use_point_cloud"] and kwargs["use_attach"]
            assert kwargs["fix_joint_limits"] is False
            assert np.array_equal(full[7:], self.state[7:])
            assert np.array_equal(goals[0][7:], self.state[7:])
            self.joint_calls.append(_array(full))
            return self.planner.TOPP(np.vstack((full[self.indices], goals[0][self.indices])), .05)

    # A converged small-angle candidate can have mixed residual 16e-6 even
    # though both final FK errors are far inside 0.5mm / 0.001rad. The old
    # 1e-5 outer acceptance threshold rejected it before joint planning.
    small = SmallAngleFixture()
    old_status, old_goals = small.ik(np.r_[goal[:3] - [1., 2., 3.], goal[3:]],
        small.state.copy(), mask=[False] * 7 + [True, True], n_init_qpos=1, threshold=1e-5)
    assert old_status != "Success" and not old_goals
    small.ik_calls.clear()
    small_state, small_topp = small.state.copy(), small.planner.TOPP
    small_result = checked_screw_plan(small, goal)
    small_audit = small_result["endpoint_audit"]
    threshold = min(POSITION_TOLERANCE_M, 2 * np.sin(ROTATION_TOLERANCE_RAD / 4))
    assert small_audit["ik_distance_threshold"] == threshold
    assert len(small.ik_calls) == 1 and len(small.joint_calls) == 2
    assert small_audit["attempts"][1]["kind"] == "native_ik_joint_path_correction"
    assert small_audit["ik_attempts"][0]["candidates"][0]["within_final_fk_contract"]
    assert small_audit["final_error"]["position_m"] < POSITION_TOLERANCE_M
    assert small_audit["final_error"]["rotation_rad"] < ROTATION_TOLERANCE_RAD
    assert np.array_equal(small.state, small_state) and small.planner.TOPP == small_topp
    assert small_audit["environment_steps"] == 0 and small_audit["ik_locked_joint_indices"] == [7, 8]
    boundary = SmallAngleFixture(position_residual=threshold * (1 - 1e-6), rotation_residual=0.)
    checked_screw_plan(boundary, goal)

    rejected = [
        (SmallAngleFixture(position_residual=threshold * (1 + 1e-6), rotation_residual=0.),
         "Native endpoint IK failed"),
        (SmallAngleFixture(position_residual=.0006, force_candidate=True),
         "Native endpoint IK candidate exceeds FK contract"),
        (SmallAngleFixture(rotation_residual=.0011, force_candidate=True),
         "Native endpoint IK candidate exceeds FK contract"),
        (SmallAngleFixture(change_finger=True), "Native endpoint IK changed non-arm joints"),
        (SmallAngleFixture(native_failure=True), "Native endpoint IK failed"),
    ]
    for fixture, message in rejected:
        initial, topp = fixture.state.copy(), fixture.planner.TOPP
        try:
            checked_screw_plan(fixture, goal)
        except ValueError as exc:
            assert message in str(exc), str(exc)
            assert fixture.last_endpoint_audit["status"] == "failed"
            assert fixture.last_endpoint_audit["ik_attempts"][0]["native_status"]
            assert not fixture.joint_calls and fixture.planner.TOPP == topp
            assert np.array_equal(fixture.state, initial)
        else:
            raise AssertionError(f"Expected rejection: {message}")

    class StalledScrewFixture(SmallAngleFixture):
        """Retreat failure: screw leaves 0.523mm with rotation already valid."""
        def screw(self, goal: np.ndarray, full: np.ndarray, **kwargs: Any) -> Any:
            self.calls += 1
            end = full[self.indices].copy()
            end[:3] = goal[:3] - [1., 2., 3.]
            end[0] -= .0026 if self.calls == 1 else .000523
            end[3] = .000716
            raw = self.planner.TOPP(np.vstack((full[self.indices], end)), .05)
            return {"status": "Success", **dict(zip(
                ("time", "position", "velocity", "acceleration", "duration"), raw))}

    stalled = StalledScrewFixture()
    stalled_state = stalled.state.copy()
    stalled_result = checked_screw_plan(stalled, goal)
    stalled_audit = stalled_result["endpoint_audit"]
    assert stalled.calls == 2 and len(stalled.ik_calls) == 1
    assert stalled_audit["corrections"] == 2
    assert [entry["kind"] for entry in stalled_audit["attempts"]] == [
        "original_native_screw", "native_screw_correction", "native_ik_joint_path_correction"]
    assert stalled_audit["attempts"][1]["position_m"] > POSITION_TOLERANCE_M
    assert stalled_audit["attempts"][1]["rotation_rad"] < ROTATION_TOLERANCE_RAD
    assert stalled_audit["ik_attempts"][0]["trigger"] == "previous_native_correction_outside_fk_contract"
    assert stalled_audit["final_error"]["position_m"] < POSITION_TOLERANCE_M
    assert stalled_audit["final_error"]["rotation_rad"] < ROTATION_TOLERANCE_RAD
    assert np.array_equal(stalled.state, stalled_state) and stalled.planner.TOPP == stalled.topp
    assert len(stalled.topp_calls) == 2 and stalled_audit["environment_steps"] == 0
    # The fallback consumes an existing correction slot, never adds retries or
    # bypasses native IK's refusal. A caller allowing only one correction fails.
    for fixture, limit, expected_ik_calls in [
            (StalledScrewFixture(), 1, 0), (StalledScrewFixture(native_failure=True), 3, 1)]:
        try:
            checked_screw_plan(fixture, goal, max_corrections=limit)
        except ValueError:
            assert fixture.last_endpoint_audit["status"] == "failed"
            assert fixture.calls == 2 and len(fixture.ik_calls) == expected_ik_calls
            assert not fixture.joint_calls and fixture.planner.TOPP == fixture.topp
            assert np.array_equal(fixture.state, stalled_state)
        else:
            raise AssertionError("Bounded fallback must preserve insufficient-budget/native-refusal failure")
    assert pose_error(goal, np.r_[goal[:3], -goal[3:]])["rotation_rad"] == 0.
    return {"status": "passed", "environment_steps": 0, "gpu_used": False,
        "checks": ["actual FK rejects 14mm native Success", "same-backend correction",
            "fresh measured-start native joint path discards the old bad endpoint",
            "full corrected native TOPP and audit cache", "nonsequential arm joint order",
            "nonzero base translation", "finger positions preserved", "original start preserved",
            "already accurate native result unchanged", "bounded failure and wrapper restoration",
            "small-angle native IK branch accepts independently valid 16e-6 mixed residual",
            "IK mixed-distance threshold derives from unchanged FK bounds",
            "IK threshold boundary acceptance and rejection", "IK explicitly locks non-arm joints",
            "inaccurate IK candidates rejected before joint planning",
            "native IK refusal and changed-finger candidate remain rejected",
            "stalled screw switches to native IK while rotation already satisfies FK bounds",
            "IK fallback consumes existing correction budget and preserves native refusal",
            "quaternion sign invariance"]}


def test_runtime_contract():
    assert self_test()["status"] == "passed"
