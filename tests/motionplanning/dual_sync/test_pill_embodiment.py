"""CPU contracts for the original pill task with a selectable robot pair."""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import h5py
import numpy as np
import pytest

from mani_skill.agents.robots.parallel_gripper import Piper
from mani_skill.examples.motionplanning.dual import two_robot_run as runner
from mani_skill.examples.motionplanning.dual.sync import phase_kinematics as timing
from mani_skill.examples.motionplanning.dual.sync import physical_observation as contacts
from mani_skill.examples.motionplanning.dual.sync import replay
from mani_skill.examples.motionplanning.dual.sync.runtime_policy import DEFAULT_POLICY
from mani_skill.examples.motionplanning.dual.sync.solve import phase_sequence
from mani_skill.examples.motionplanning.planner import Planner

TASK = "TwoRobotPickPillToRegions-v1"


def _semantic_ast(node):
    """Canonical fields across Python versions that add empty AST attributes."""
    if isinstance(node, ast.AST):
        return (type(node).__name__, tuple((key, _semantic_ast(value))
            for key, value in ast.iter_fields(node) if value is not None and value != []))
    if isinstance(node, list):
        return tuple(_semantic_ast(value) for value in node)
    return node


def _ast_receipt(node):
    return hashlib.sha256(repr(_semantic_ast(node)).encode()).hexdigest()


def test_piper_sync_selection_reaches_environment_with_the_selected_pair(monkeypatch):
    class ReachedEnvironment(Exception):
        pass

    calls = []

    def make(env_id, **kwargs):
        calls.append((env_id, kwargs))
        raise ReachedEnvironment

    monkeypatch.setattr(runner.gym, "make", make)
    args = runner.parse_args(["-e", TASK, "--robot", "piper", "--execution-mode", "sync"])
    with pytest.raises(ReachedEnvironment):
        runner._main(args)
    assert calls[0][0] == TASK
    assert calls[0][1]["robot_uids"] == ("piper", "piper")
    assert calls[0][1]["control_mode"] == "pd_joint_pos"


@pytest.mark.parametrize("argv,reason", [
    (["-e", TASK, "--robot", "piper"], "requires --execution-mode sync"),
    (["-e", TASK, "--robot", "arx_x5", "--execution-mode", "sync"], "no profile"),
    (["-e", "TwoRobotPickAppleBananaToBaskets-v1", "--robot", "piper",
      "--execution-mode", "sync"], "no profile"),
    (["-e", TASK, "--robot", "missing-robot", "--execution-mode", "sync"], "Unsupported"),
])
def test_unsupported_robot_or_mode_is_rejected_before_gym_make(monkeypatch, argv, reason):
    monkeypatch.setattr(runner.gym, "make", lambda *a, **k: pytest.fail("created an environment"))
    with pytest.raises(ValueError, match=reason):
        runner._main(runner.parse_args(argv))


def _recording(tmp_path, robot, *, legacy=False):
    width = 7 if robot == "piper" else 8
    order = tuple(f"{robot}-{i}" for i in range(2))
    stages = [{"name": name, "status": "passed", "environment_steps": 1}
              for name in phase_sequence(DEFAULT_POLICY)]
    count = 10 + len(stages)
    motion = {"action_order": list(order)}
    if not legacy:
        motion.update(robot_uids=[robot, robot], action_sizes=[width, width])
    metadata = {"env_info": {"env_id": TASK}, "motion_planning": motion,
                "episodes": [{"episode_id": 0, "episode_seed": 401,
                    "dual_sync": {"schema": "dual-panda-sync-v1" if legacy else "dual-arm-sync-v2",
                                  "policy": DEFAULT_POLICY.to_dict(), "stages": stages}}]}
    actions = np.arange(count*width*2, dtype=np.float32).reshape(count, width*2)/1000
    states = np.arange((count+1)*13, dtype=np.float32).reshape(count+1, 13)
    path = tmp_path / "trajectory.h5"
    path.with_suffix(".json").write_text(json.dumps(metadata))
    with h5py.File(path, "w") as h5:
        h5["traj_0/actions"] = actions
        h5["traj_0/env_states/actors/pill"] = states
    return path, metadata, actions, states


@pytest.mark.parametrize("robot,legacy,width", [("piper", False, 7), ("panda_wristcam", True, 8)])
def test_replay_loads_native_width_and_physical_arm_order(tmp_path, robot, legacy, width):
    path, metadata, expected_actions, expected_states = _recording(tmp_path, robot, legacy=legacy)
    loaded, _, policy, phases, actions, initial = replay.load_episode(path)
    order, sizes = replay.action_layout(loaded)
    assert sizes == (width, width)
    assert policy.max_episode_steps == 200 and policy.last_completion_step == 199
    assert phases[0] == ("preparation", 10)
    np.testing.assert_array_equal(actions, expected_actions)
    np.testing.assert_array_equal(initial["actors"]["pill"], expected_states[0])
    paired = replay.paired_action(actions[-1], order, sizes)
    assert tuple(paired) == (f"{robot}-0", f"{robot}-1")
    np.testing.assert_array_equal(paired[f"{robot}-0"], expected_actions[-1, :width])
    np.testing.assert_array_equal(paired[f"{robot}-1"], expected_actions[-1, width:])
    paired[f"{robot}-0"][0] = -999
    np.testing.assert_array_equal(actions, expected_actions)
    assert loaded == metadata


@pytest.mark.parametrize("mutation", [
    lambda data: data["motion_planning"].update(action_order=["piper-1", "piper-0"]),
    lambda data: data["motion_planning"].update(action_sizes=[8, 8]),
    lambda data: data["motion_planning"].update(robot_uids=["piper", "panda_wristcam"]),
])
def test_piper_archive_rejects_wrong_order_width_or_mixed_pair(tmp_path, mutation):
    path, metadata, _, _ = _recording(tmp_path, "piper")
    mutation(metadata)
    path.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        replay.load_episode(path)


def test_piper_archive_cannot_silently_accept_a_panda_action_row(tmp_path):
    path, _, _, _ = _recording(tmp_path, "piper")
    with h5py.File(path, "a") as h5:
        count = len(h5["traj_0/actions"])
        del h5["traj_0/actions"]
        h5["traj_0/actions"] = np.zeros((count, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="native paired controller width"):
        replay.load_episode(path)


def test_contact_permissions_follow_configured_piper_fingers_and_arm_identity():
    agents, entities = [], {}
    for arm in range(2):
        links = []
        for name in ("link6", "link7", "link8", "panda_leftfinger"):
            entity = object()
            entities[arm, name] = entity
            links.append(NS(name=name, _objs=[NS(entity=entity)]))
        agents.append(NS(finger_link_names=Piper.finger_link_names,
                         robot=NS(get_links=lambda links=links: links)))
    payloads = [NS(_objs=[object()]), NS(_objs=[object()])]
    identities, _ = contacts.contact_identities(NS(scene=NS(actors={})), agents,
                                               ("pill_a", "pill_b"), payloads)
    srdf = {0: set(), 1: set()}
    for arm in range(2):
        payload = identities[payloads[arm]._objs[0]]
        for robot_arm in range(2):
            for name in ("link6", "link7", "link8", "panda_leftfinger"):
                robot = identities[entities[robot_arm, name]]
                allowed = robot_arm == arm and name in ("link7", "link8")
                assert robot["is_grasp_finger"] is (name in ("link7", "link8"))
                category = contacts.contact_class(robot, arm, True, "carry_lower",
                                                  ("table", "table"), ("table", "table"))
                assert category == ("own_finger_allowed" if allowed else "forbidden")
                for phase in ("descend", "close_and_confirm", "lift_and_hold", "carry_lower", "release"):
                    assert contacts.contact_allowed([robot, payload], phase, srdf) is allowed
                    assert contacts.contact_allowed([payload, robot], phase, srdf) is allowed
                assert not contacts.contact_allowed([robot, payload], "pregrasp", srdf)


def _six_joint_path():
    parameter = np.linspace(0., 1., 23)
    return np.column_stack((.3*parameter, .1*parameter**2, .08*np.sin(1.2*parameter),
                            -.2*parameter, .05*parameter**3, .04*np.cos(parameter)))


def test_six_joint_phase_limits_update_both_native_planners():
    solvers = [NS(planner=NS(move_group_joint_indices=list(range(6)))) for _ in range(2)]
    timing.compact_set_phase_limits(solvers, {"solver_velocity_limit_rad_s": .891,
                                           "solver_acceleration_limit_rad_s2": 1.188})
    for solver in solvers:
        np.testing.assert_array_equal(solver.planner.joint_vel_limits, np.full(6, .891))
        np.testing.assert_array_equal(solver.planner.joint_acc_limits, np.full(6, 1.188))


def test_real_six_joint_topp_reconstruction_and_continuous_retiming():
    import toppra as ta
    import toppra.algorithm as algorithm
    import toppra.constraint as constraint

    points = _six_joint_path()
    planner = NS(joint_vel_limits=np.full(6, .7), joint_acc_limits=np.full(6, 1.1))
    native = Planner.TOPP(planner, points, .05)
    trace = timing.compact_topp_reconstruction(points, planner.joint_vel_limits,
                                              planner.joint_acc_limits, .05, native)
    assert trace["path_coefficients"].shape == (4, len(points)-1, 6)
    for field, expected in zip(("time", "position", "velocity", "acceleration"), native):
        np.testing.assert_array_equal(trace["sample_"+field], expected)
    raw = dict(zip(("time", "position", "velocity", "acceleration", "duration"), native), status="Success")
    unchanged = {key: value.copy() for key, value in raw.items() if isinstance(value, np.ndarray)}
    output, receipt = timing.compact_retime_pair((raw, None), (trace, None),
                                                velocity_limit=.19, acceleration_limit=.23)
    scale = receipt["shared_phase_scale"]
    assert scale > 1 and output[1] is None
    np.testing.assert_array_equal(output[0]["position"], native[1])
    np.testing.assert_allclose(output[0]["velocity"]*scale, native[2], rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(output[0]["acceleration"]*scale**2, native[3], rtol=1e-14, atol=1e-14)
    for key, expected in unchanged.items():
        np.testing.assert_array_equal(raw[key], expected)
    # Independently solve/evaluate a dense native curve rather than relying on
    # the implementation's extrema calculations to validate their own output.
    spline = ta.SplineInterpolator(np.linspace(0., 1., len(points)), points)
    curve = algorithm.TOPPRA([constraint.JointVelocityConstraint(planner.joint_vel_limits),
        constraint.JointAccelerationConstraint(planner.joint_acc_limits)], spline,
        parametrizer="ParametrizeConstAccel").compute_trajectory()
    times = np.linspace(0., curve.duration, 20001)
    assert np.max(np.abs(curve(times, 1)))/scale <= .19+1e-6
    assert np.max(np.abs(curve(times, 2)))/scale**2 <= .23+1e-6
    assert len(receipt["arms"][0]["raw_source_curve"]["per_joint_max_velocity_rad_s"]) == 6
    wrong_dimension = deepcopy(trace)
    wrong_dimension["path_coefficients"] = np.pad(trace["path_coefficients"], ((0, 0), (0, 0), (0, 1)))
    with pytest.raises(ValueError, match="Malformed"):
        timing.compact_retime_pair((raw, None), (wrong_dimension, None),
                                  velocity_limit=.19, acceleration_limit=.23)
    with pytest.raises(ValueError, match="matching joint limits"):
        timing.compact_topp_reconstruction(points, np.full(7, .7), np.full(7, 1.1), .05, native)


def test_original_pill_success_horizon_assets_sampling_and_camera_contract_are_unchanged():
    """AST receipts were independently taken from the pre-adaptation task.

    These digests cover semantic syntax without comments/formatting and do not
    require historical commits to exist in the checkout running this test.
    Only the new Piper mount initialization is removed for the comparison.
    """
    path = Path(__file__).parents[3] / "mani_skill/envs/tasks/tabletop/dual_tasks/_031_pick_pillbottle_to_regions.py"
    task = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef))
    registration = next(dec for dec in task.decorator_list if isinstance(dec, ast.Call)
                        and isinstance(dec.func, ast.Name) and dec.func.id == "register_env")
    assert ast.literal_eval(registration.args[0]) == TASK
    assert {kw.arg: ast.literal_eval(kw.value) for kw in registration.keywords} == {"max_episode_steps": 200}
    expected = {
        "_is_in_region": "4c8ad4e854b284d8dcb40d4499cc4603279810a6598961bd5ff742bf994021ca",
        "evaluate": "a0880ef5a3bafb9d11e9f92cf37264343a75160f243880aa55559f738e91e9e3",
        "_after_reconfigure": "820618300eb45ff42d56fc78641d58fe6c2c6b2823c60241f88446b0418b40d8",
        "_default_sensor_configs": "bac33e66ae88640b725dfcb6c439cba86df050bb0fcf75dcdbf751a441a3b876",
        "_default_human_render_camera_configs": "3dc71db7d2567819e360e8a6983cf0c9c350f6613313a37b11d397a215158739",
        "_initialize_episode": "4bdd18b7541e2228ea13aefa2b9c6ae7b051378384687367602323e90cef505c",
        "_load_scene": "c0850537ec977043b76e2f83ed4f2c93522ca82b1c76f97b429b393a559ae7cb",
    }
    for name, receipt in expected.items():
        node = deepcopy(next(node for node in task.body if isinstance(node, ast.FunctionDef) and node.name == name))
        for parent in ast.walk(node):
            if hasattr(parent, "body") and isinstance(parent.body, list):
                parent.body[:] = [statement for statement in parent.body
                    if not (isinstance(statement, ast.If)
                            and ast.dump(statement.test) == ast.dump(ast.parse('self.robot_uids == ("piper", "piper")', mode="eval").body))]
        assert _ast_receipt(node) == receipt, name
    init = next(node for node in task.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    calls = [node for node in ast.walk(init) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "apply_l2_robotwin_model"]
    model_ast = ast.Module(body=[ast.Expr(value=node) for node in calls], type_ignores=[])
    assert _ast_receipt(model_ast) == (
        "c4ce2e8807ab3df275d65d297762621ef361aa4a0186e7967bee48f0bdaf3620")
    constants = {node.targets[0].id: ast.literal_eval(node.value)
                 for node in task.body if isinstance(node, ast.Assign)
                 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in {
                     "PILL_MODEL_NAME", "pill_a_xy", "pill_b_xy", "region_a_xy", "region_b_xy",
                     "region_half_size", "region_height_threshold", "pill_scale"}}
    assert constants == {"PILL_MODEL_NAME": "080_pillbottle", "pill_a_xy": (-.25, -.15),
        "pill_b_xy": (-.25, .15), "region_a_xy": (.10, -.20), "region_b_xy": (.10, .20),
        "region_half_size": (.08, .08, .005), "region_height_threshold": .20, "pill_scale": 1.}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
