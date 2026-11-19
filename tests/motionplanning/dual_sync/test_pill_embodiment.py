"""CPU contracts for the original pill task with a selectable robot pair."""
from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace as NS

import h5py
import numpy as np
import pytest
import torch

from mani_skill.agents.robots.parallel_gripper import Piper
from mani_skill.envs.tasks.tabletop.dual_arm_profiles import get_task_profile
from mani_skill.envs.tasks.tabletop.dual_tasks import _031_pick_pillbottle_to_regions as pill_env
from mani_skill.examples.motionplanning.dual import two_robot_run as runner
from mani_skill.examples.motionplanning.dual.sync import phase_kinematics as timing
from mani_skill.examples.motionplanning.dual.sync import physical_observation as contacts
from mani_skill.examples.motionplanning.dual.sync import replay
from mani_skill.examples.motionplanning.dual.sync.runtime_policy import DEFAULT_POLICY
from mani_skill.examples.motionplanning.dual.sync.solve import phase_sequence
from mani_skill.examples.motionplanning.planner import Planner

TASK = "TwoRobotPickPillToRegions-v1"


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


@pytest.mark.parametrize("robot", ["ur5_wsg", "arx_x5", "xarm6_robotiq"])
def test_new_sync_selection_reaches_environment_with_native_pair(monkeypatch, robot):
    class ReachedEnvironment(Exception):
        pass

    calls = []

    def make(env_id, **kwargs):
        calls.append((env_id, kwargs))
        raise ReachedEnvironment

    monkeypatch.setattr(runner.gym, "make", make)
    with pytest.raises(ReachedEnvironment):
        runner._main(runner.parse_args(["-e", TASK, "--robot", robot,
                                        "--execution-mode", "sync"]))
    assert len(calls) == 1 and calls[0][0] == TASK
    assert calls[0][1]["robot_uids"] == (robot, robot)
    assert calls[0][1]["control_mode"] == "pd_joint_pos"


def test_configured_ur5_grasp_does_not_inherit_the_rest_tcp_closing_axis():
    from mani_skill.examples.motionplanning.dual.sync.pill_task import PillTask
    from mani_skill.utils.structs.pose import Pose

    # A fixed task quaternion is independent of a projected rest-pose axis.
    # No TCP property or dynamic grasp construction should be read here.
    raw = NS(embodiment_profile=get_task_profile(TASK, "ur5_wsg"),
             agent=NS(agents=[object(), object()]))
    task = object.__new__(PillTask)
    task.raw = raw
    task.targets = tuple(NS(pose=Pose.create_from_pq(p=[[x, y, .0014]],
                            q=[[1., 0., 0., 0.]])) for x, y in ((-.25, -.15), (-.25, .15)))
    pairs = task.grasp_poses()
    for (_, grasp), target in zip(pairs, task.targets):
        np.testing.assert_allclose(grasp.p, target.pose.p[0].numpy()+[0., 0., .12], atol=1e-7)
        np.testing.assert_allclose(np.abs(grasp.q), [0., 0., 1., 0.], atol=1e-7)


@pytest.mark.parametrize("argv,reason", [
    (["-e", TASK, "--robot", "piper"], "requires --execution-mode sync"),
    (["-e", "TwoRobotScanPillBottle-v1", "--robot", "piper",
      "--execution-mode", "sync"], "no profile"),
    (["-e", TASK, "--robot", "missing-robot", "--execution-mode", "sync"], "Unsupported"),
])
def test_unsupported_robot_or_mode_is_rejected_before_gym_make(monkeypatch, argv, reason):
    monkeypatch.setattr(runner.gym, "make", lambda *a, **k: pytest.fail("created an environment"))
    with pytest.raises(ValueError, match=reason):
        runner._main(runner.parse_args(argv))


def _recording(tmp_path, robot, *, legacy=False):
    width = 8 if robot == "panda_wristcam" else 7
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


@pytest.mark.parametrize("robot,legacy,width", [("piper", False, 7), ("ur5_wsg", False, 7),
                                               ("arx_x5", False, 7), ("xarm6_robotiq", False, 7),
                                               ("panda_wristcam", True, 8)])
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


@pytest.mark.parametrize("root_name,pedestal", [("panda_link0", False),
                                               ("link0", True), ("base_link", False)])
def test_only_each_configured_root_can_contact_its_own_mount_support(root_name, pedestal):
    agents, entities = [], {}
    for arm in range(2):
        links = []
        for name in (root_name, "wrist"):
            entity = object()
            entities[arm, name] = entity
            links.append(NS(name=name, _objs=[NS(entity=entity)]))
        robot = NS(root=NS(name=root_name), get_links=lambda links=links: links)
        agent = NS(robot=robot)
        if root_name != "panda_link0":
            agent.robot_root_link_name = root_name
        agents.append(agent)
    supports = {name: NS(_objs=[object()]) for name in
                (("table", "mount_0", "mount_1") if pedestal else ("table", "obstacle"))}
    identities, _ = contacts.contact_identities(NS(scene=NS(actors=supports)), agents,
                                               ("a", "b"), [NS(_objs=[]), NS(_objs=[])])
    for arm in range(2):
        expected = f"mount_{arm}" if pedestal else "table"
        root = identities[entities[arm, root_name]]
        wrist = identities[entities[arm, "wrist"]]
        assert root["is_mount_root"] and not wrist["is_mount_root"]
        for name, actor in supports.items():
            support = identities[actor._objs[0]]
            for pair in ([root, support], [support, root]):
                assert contacts.contact_allowed(pair, "preparation", {0: set(), 1: set()}) is (name == expected)
            assert not contacts.contact_allowed([wrist, support], "preparation", {0: set(), 1: set()})
    assert not contacts.contact_allowed([identities[entities[0, root_name]],
                                         identities[entities[1, root_name]]],
                                        "preparation", {0: set(), 1: set()})


def test_legacy_panda_root_contact_dict_keeps_its_table_permission():
    assert contacts.contact_allowed([{"kind": "robot", "link_name": "panda_link0"},
                                     {"kind": "actor", "actor": "table"}], "preparation", {})


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


def test_original_pill_success_depends_on_both_regions_with_static_as_diagnostic():
    task = pill_env.TwoRobotPickPillToRegionsEnv
    # Preserve the original origin-only boundary and upper height predicate.
    positions = torch.tensor([[0., 0., .1], [.080001, 0., .1], [0., 0., .1],
                              [.08, .08, .2], [0., 0., .200001]], dtype=torch.float64)
    other = torch.tensor([[0., 0., .1], [0., 0., .1], [0., -.080001, .1],
                          [.08, -.08, .2], [0., 0., .1]], dtype=torch.float64)
    count = len(positions)
    moving_grasped = NS(is_grasping=lambda _: torch.ones(count, dtype=torch.bool),
                        is_static=lambda _: torch.zeros(count, dtype=torch.bool))
    raw = NS(pill_a=NS(pose=NS(p=positions)), pill_b=NS(pose=NS(p=other)),
        region_a=NS(pose=NS(p=torch.zeros_like(positions))),
        region_b=NS(pose=NS(p=torch.zeros_like(other))),
        region_half_size=task.region_half_size, region_height_threshold=task.region_height_threshold,
        left_agent=moving_grasped, right_agent=moving_grasped)
    raw._is_in_region = task._is_in_region.__get__(raw)
    result = task.evaluate(raw)
    assert result['pill_a_in_region'].tolist() == [True, False, True, True, False]
    assert result['pill_b_in_region'].tolist() == [True, True, False, True, True]
    assert result['success'].tolist() == [True, False, False, True, False]
    assert not result['is_robot_static'].any()
    assert result['is_pill_a_grasped'].all() and result['is_pill_b_grasped'].all()
    assert runner.gym.spec(TASK).max_episode_steps == 200


def _original_pill_scene(monkeypatch, robot='panda_wristcam'):
    task = pill_env.TwoRobotPickPillToRegionsEnv
    fields = ('PILL_MODEL_NAME', 'pill_a_xy', 'pill_b_xy', 'region_a_xy', 'region_b_xy',
              'region_half_size', 'region_height_threshold', 'pill_scale')
    raw = NS(**{name: getattr(task, name) for name in fields},
        robot_uids=(robot, robot), robot_init_qpos_noise=.02, scene=object(),
        embodiment_profile=get_task_profile(TASK, robot),
        pill_a_modelname='080_pillbottle', pill_a_model_id=1,
        pill_b_modelname='080_pillbottle', pill_b_model_id=5)
    creations = []

    def actor(pose):
        result = NS(pose=pose)
        result.set_pose = lambda value: setattr(result, 'pose', value)
        return result

    monkeypatch.setattr(pill_env, 'TableSceneBuilder',
        lambda **_: NS(build=lambda: None, initialize=lambda _: None))
    monkeypatch.setattr(pill_env, 'is_l3_enabled', lambda: False)
    monkeypatch.setattr(pill_env.actors, 'build_box',
        lambda scene, **kwargs: actor(kwargs['initial_pose']))
    monkeypatch.setattr(pill_env, 'get_model_id', lambda name, model_id: model_id)

    def create(**kwargs):
        creations.append(kwargs)
        return NS(actor=actor(kwargs['pose']))

    monkeypatch.setattr(pill_env, 'create_actor', create)
    task._load_scene(raw, {})
    return raw, creations


def test_original_pill_assets_region_sizes_and_panda_mounts_are_preserved(monkeypatch):
    raw, models = _original_pill_scene(monkeypatch)
    assert raw.PILL_MODEL_NAME == '080_pillbottle'
    assert raw.pill_a_xy == (-.25, -.15) and raw.pill_b_xy == (-.25, .15)
    assert raw.region_a_xy == (.10, -.20) and raw.region_b_xy == (.10, .20)
    assert raw.region_half_size == (.08, .08, .005) and raw.region_height_threshold == .20
    assert raw.pill_scale == 1.
    assert len(models) == 2
    for index, model in enumerate(models):
        assert model['modelname'] == '080_pillbottle'
        assert model['model_id'] == (1, 5)[index]
        assert model['mass'] == .5 and model['scale'] == (1., 1., 1.)
        assert model['convex'] is True and model['_idx_if_repeat'] == index+1
        np.testing.assert_allclose(model['pose'].p,
            [raw.pill_a_xy, raw.pill_b_xy][index]+(0.,), atol=1e-7)
    profile = get_task_profile(TASK, 'panda_wristcam')
    assert tuple(mount.position for mount in profile.mount_poses) == ((-.60, -.35, 0.), (-.60, .35, 0.))
    assert all(mount.quaternion == (1., 0., 0., 0.) for mount in profile.mount_poses)


@pytest.mark.parametrize('robot', ['panda_wristcam', 'piper', 'ur5_wsg', 'arx_x5', 'xarm6_robotiq'])
def test_original_pill_sampling_uses_the_same_random_offsets_and_object_heights(monkeypatch, robot):
    raw, _ = _original_pill_scene(monkeypatch, robot)
    raw.device, raw.num_envs = 'cpu', 3
    raw.reward_tracker = NS(reset=lambda _: None)
    raw._pill_a_z, raw._pill_b_z = torch.tensor([.047]), torch.tensor([.052])
    measured_mounts = []
    raw.agent = NS(agents=[NS(keyframes={'rest': NS(qpos=np.zeros(8))}, reset=lambda _: None,
        robot=NS(set_pose=lambda pose: measured_mounts.append(pose))) for _ in range(2)])
    offsets = []
    monkeypatch.setattr(pill_env, 'apply_l1_offset_xy',
        lambda value, offset: offsets.append(offset) or value)
    # Independent generator preserves the original four RNG draws and keeps
    # the test from depending on other tests' global randomization settings.
    generator = torch.Generator().manual_seed(723)
    region_a, region_b, pill_a, pill_b = [torch.rand((3, 2), generator=generator) for _ in range(4)]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(723)
        pill_env.TwoRobotPickPillToRegionsEnv._initialize_episode(raw, torch.arange(3), {})
    for actual, xy, draw in ((raw.region_a, raw.region_a_xy, region_a),
                              (raw.region_b, raw.region_b_xy, region_b)):
        expected = torch.tensor([*xy, .005]).repeat(3, 1)
        expected[:, :2] += draw*.02
        torch.testing.assert_close(actual.pose.p, expected)
    for actual, xy, draw, z in ((raw.pill_a, raw.pill_a_xy, pill_a, .047),
                                 (raw.pill_b, raw.pill_b_xy, pill_b, .052)):
        expected = torch.tensor([*xy, z]).repeat(3, 1)
        expected[:, :2] += (draw-.5)*.02
        torch.testing.assert_close(actual.pose.p, expected)
        np.testing.assert_allclose(actual.pose.q[0], [np.sqrt(.5), np.sqrt(.5), 0., 0.], atol=1e-7)
    assert offsets == [(.05, -.1), (.05, .1)]
    for actual, mount in zip(measured_mounts, raw.embodiment_profile.mount_poses):
        np.testing.assert_allclose(actual.p, mount.position, atol=1e-7)
        np.testing.assert_allclose(actual.q, mount.quaternion, atol=1e-7)
    assert len(measured_mounts) == 2


def test_original_external_cameras_keep_resolution_fov_and_view_direction():
    task = pill_env.TwoRobotPickPillToRegionsEnv
    low = task._default_sensor_configs.fget(NS(hi_res=False))
    assert len(low) == 1 and low[0].uid == 'zed2i'
    assert (low[0].width, low[0].height) == (224, 224)
    assert low[0].fov == np.pi*1.1/2 and (low[0].near, low[0].far) == (.01, 100)
    np.testing.assert_allclose(low[0].pose.p[0], [-.4, 0., .6], atol=1e-7)
    np.testing.assert_allclose(low[0].pose.to_transformation_matrix()[0, :3, 0],
                               np.array([1., 0., -1.])/np.sqrt(2), atol=1e-6)
    high = task._default_sensor_configs.fget(NS(hi_res=True))
    assert [(camera.uid, camera.width, camera.height) for camera in high] == [
        ('cam1', 640, 480), ('cam2', 640, 480), ('cam3', 640, 480), ('zed2i', 1280, 720)]
    np.testing.assert_allclose(high[-1].pose.raw_pose, low[0].pose.raw_pose)
    render = task._default_human_render_camera_configs.fget(NS())
    assert (render.uid, render.width, render.height, render.fov, render.near, render.far) == (
        'render_camera', 512, 512, 1, .01, 100)
    np.testing.assert_allclose(render.pose.p[0], [.8, 0., .75], atol=1e-7)



@pytest.mark.parametrize("robot", ["panda_wristcam", "piper", "arx_x5", "ur5_wsg", "xarm6_robotiq"])
def test_original_sync_preparation_sends_native_open_actions(monkeypatch, robot):
    import importlib
    module = importlib.import_module("mani_skill.examples.motionplanning.dual.sync.solve")
    from mani_skill.agents.robots.embodiment_profiles import get_robot_profile
    profile = get_robot_profile(robot)
    agents = [NS(control_mode="pd_joint_pos", gripper_open_action=profile.gripper_open_action,
                 gripper_closed_action=profile.gripper_closed_action) for _ in range(2)]
    arms = tuple(NS(key=f"{robot}-{i}", agent=agent, names=profile.arm_joint_names,
                    arm_qpos=lambda: np.zeros(len(profile.arm_joint_names)))
                 for i, agent in enumerate(agents))
    raw = NS(spec=NS(id=TASK), num_envs=1, agent=NS(agents=agents,
             agents_dict=dict(zip((a.key for a in arms), agents))), robot_uids=(robot, robot),
             embodiment_profile=get_task_profile(TASK, robot),
             backend=NS(sim_backend="cpu"), control_timestep=.05, sim_timestep=.01,
             elapsed_steps=np.array([0]), evaluate=lambda: {"success": np.array([False])})
    env = NS(unwrapped=raw, reset=lambda **_: None)
    actions = []
    class Capture:
        def __init__(self, *args): self.last_step = None
        def install(self): pass
        def restore(self): pass
        def step(self, action):
            actions.append(deepcopy(action)); raw.elapsed_steps += 1
            self.last_step = (None, None, False, False, {"success": False})
        def summary(self): return {}
    monkeypatch.setattr(module, "ObservedEnvironment", Capture)
    monkeypatch.setattr(module, "bindings", lambda *_: arms)
    def stop(*args, **kwargs): raise RuntimeError("stop after preparation")
    monkeypatch.setattr(module, "pill_planners", stop)
    module.solve(env, seed=401, task_factory=lambda _: object())
    assert len(actions) == 10 and raw.elapsed_steps.item() == 10
    for action in actions:
        assert tuple(action) == tuple(a.key for a in arms)
        for arm in arms:
            np.testing.assert_array_equal(action[arm.key][:-1], np.zeros(len(profile.arm_joint_names)))
            assert action[arm.key][-1] == profile.gripper_open_action
    assert raw.dual_sync_result["execution"]["reason"] == "RuntimeError: stop after preparation"


def test_pill_effort_setting_only_changes_matching_instance_gripper_controllers():
    from mani_skill.agents.robots.xarm6.xarm6_robotiq import XArm6Robotiq
    from mani_skill.agents.robots.embodiment_profiles import get_robot_profile
    profile = get_task_profile(TASK, "xarm6_robotiq")
    assert profile.gripper_drive_force_limit == .5
    assert XArm6Robotiq.gripper_force_limit == .1
    calls, agents = [], []
    for index in range(2):
        contract = get_robot_profile("xarm6_robotiq")
        parts = {name: NS(config=NS(joint_names=joints, force_limit=force),
            set_drive_property=lambda name=name, index=index: calls.append((index, name)))
            for name, joints, force in (("arm", contract.arm_joint_names, 100.),
                ("gripper_active", contract.controlled_gripper_joints, .1),
                ("gripper_passive", contract.passive_gripper_joints, 0.))}
        agents.append(NS(embodiment_profile=contract, gripper_force_limit=.1,
                         controllers={"pd_joint_pos": NS(controllers=parts)}))
    raw = NS(embodiment_profile=profile, agent=NS(agents=agents))
    pill_env.TwoRobotPickPillToRegionsEnv._configure_embodiment_gripper(raw)
    assert calls == [(0, "gripper_active"), (1, "gripper_active")]
    for agent in agents:
        parts = agent.controllers["pd_joint_pos"].controllers
        assert agent.gripper_force_limit == .5
        assert parts["gripper_active"].config.force_limit == .5
        assert parts["arm"].config.force_limit == 100.
        assert parts["gripper_passive"].config.force_limit == 0.
    for robot in ("panda", "piper", "arx_x5", "ur5_wsg"):
        profile = get_task_profile(TASK, robot)
        assert profile.gripper_drive_force_limit is None
        pill_env.TwoRobotPickPillToRegionsEnv._configure_embodiment_gripper(
            NS(embodiment_profile=profile, agent=NS(agents=[])))


def test_side_grasp_uses_world_offset_and_tool_local_approach_without_changing_state():
    from mani_skill.examples.motionplanning.dual.sync.pill_task import PillTask
    from mani_skill.utils.structs.pose import Pose
    profile = get_task_profile(TASK, "xarm6_robotiq")
    raw = NS(embodiment_profile=profile, agent=NS(agents=[object(), object()]))
    task = object.__new__(PillTask)
    task.raw = raw
    task.targets = tuple(NS(pose=Pose.create_from_pq(p=[[x, y, .0014]], q=[[1., 0., 0., 0.]]))
                         for x, y in ((-.25, -.15), (-.25, .15)))
    original = [target.pose.raw_pose.clone() for target in task.targets]
    for i, ((pregrasp, grasp), target) in enumerate(zip(task.grasp_poses(), task.targets)):
        sign = 1. if i == 0 else -1.
        np.testing.assert_allclose(grasp.p, target.pose.p[0].numpy()+[0., .007 if i == 0 else -.009, .08], atol=1e-7)
        np.testing.assert_allclose(grasp.p-pregrasp.p, [0., sign*.09, -.10], atol=1e-7)
        np.testing.assert_allclose(grasp.to_transformation_matrix()[:3, 2], [0., sign, 0.], atol=1e-7)
        torch.testing.assert_close(target.pose.raw_pose, original[i])
    for robot in ("panda", "piper", "arx_x5", "ur5_wsg"):
        assert get_task_profile(TASK, robot).grasp_position_offsets_m is None
        assert get_task_profile(TASK, robot).pregrasp_position_offsets_m is None
        assert get_task_profile(TASK, robot).release_clearance_m == .015
    assert profile.release_clearance_m == .005

def test_held_motion_limits_are_task_specific_and_leave_approach_and_other_robots_unchanged():
    from mani_skill.examples.motionplanning.dual.sync.solve import LIMITS, task_phase_limits
    xarm = get_task_profile(TASK, "xarm6_robotiq")
    for phase in ("lift_and_hold", "carry_lower"):
        assert task_phase_limits(xarm, phase)["nominal_acceleration_limit_rad_s2"] == 2.
        for robot in ("panda", "piper", "arx_x5", "ur5_wsg"):
            profile = get_task_profile(TASK, robot)
            assert profile.held_joint_acceleration_limit_rad_s2 is None
            assert task_phase_limits(profile, phase)["nominal_acceleration_limit_rad_s2"] == LIMITS["joint_acc_limit_rad_s2"]
    assert task_phase_limits(xarm, "retreat")["nominal_acceleration_limit_rad_s2"] == LIMITS["joint_acc_limit_rad_s2"]
    for phase in ("pregrasp", "descend"):
        assert task_phase_limits(xarm, phase)["nominal_acceleration_limit_rad_s2"] == LIMITS["approach_joint_acc_limit_rad_s2"]
    from dataclasses import replace
    for acceleration in (0., -1., float("nan")):
        with pytest.raises(ValueError, match="finite and positive"):
            task_phase_limits(replace(xarm, held_joint_acceleration_limit_rad_s2=acceleration), "carry_lower")


def test_task_retreat_keeps_selected_vertical_distance_and_world_xy_withdrawal():
    from mani_skill.examples.motionplanning.dual.sync.pill_task import PillTask
    from mani_skill.utils.structs.pose import Pose
    agents = [NS(tcp=NS(pose=Pose.create_from_pq(p=[[.1, sign*.2, .09]], q=[[.5, -.5, .5, .5]])))
              for sign in (-1, 1)]
    for robot in ("panda", "piper", "arx_x5", "ur5_wsg", "xarm6_robotiq"):
        task = object.__new__(PillTask)
        task.raw = NS(embodiment_profile=get_task_profile(TASK, robot), agent=NS(agents=agents))
        original = [agent.tcp.pose.raw_pose.clone() for agent in agents]
        for i, pose in enumerate(task.retreat_poses(.085)):
            xy = (0., (-.06 if i == 0 else .06)) if robot == "xarm6_robotiq" else (0., 0.)
            np.testing.assert_allclose(pose.p, agents[i].tcp.pose.p[0].numpy()+[*xy, .085], atol=1e-7)
            np.testing.assert_allclose(pose.q, agents[i].tcp.pose.q[0].numpy(), atol=1e-7)
            torch.testing.assert_close(agents[i].tcp.pose.raw_pose, original[i])
        if robot != "xarm6_robotiq":
            assert task.raw.embodiment_profile.retreat_horizontal_offsets_m is None
        for distance in (0., -1., float("nan")):
            with pytest.raises(ValueError, match="finite and positive"):
                task.retreat_poses(distance)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
