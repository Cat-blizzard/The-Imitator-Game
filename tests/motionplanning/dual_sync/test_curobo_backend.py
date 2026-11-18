"""CPU contracts for the optional worker transport and shared trajectory clock."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pytest

from mani_skill.examples.motionplanning.dual.sync import curobo_backend as client


NAMES = client.ARM_NAMES
START = np.arange(7, dtype=float) / 20


def native_path(start=START, duration=.125, distance=.01):
    return {"success": True, "status": "SUCCESS", "joint_names": list(NAMES),
            "times_s": [0., duration / 2, duration],
            "positions": [start.tolist(), (start + distance / 2).tolist(),
                          (start + distance).tolist()],
            "velocity": [[.1] * 7] * 3, "acceleration": [[.2] * 7] * 3}


def convert(result, start=START):
    return client.checked_trajectory(result, start, NAMES, -1., "left",
                                     velocity_limit=.9, acceleration_limit=1.2)


def test_native_named_columns_are_reordered_and_native_seconds_preserved():
    result = native_path()
    permutation = [6, 0, 5, 1, 4, 2, 3]
    result["joint_names"] = [NAMES[index] for index in permutation]
    for key in ("positions", "velocity", "acceleration"):
        result[key] = np.asarray(result[key])[:, permutation].tolist()
    before = deepcopy(result)
    trajectory = convert(result)
    np.testing.assert_array_equal(trajectory.times_s, [0., .0625, .125])
    np.testing.assert_allclose(trajectory.positions_rad[0], START)
    np.testing.assert_allclose(trajectory.positions_rad[-1], START + .01)
    assert trajectory.joint_names == NAMES
    assert trajectory.gripper_command == -1.
    assert trajectory.action_key == "left"
    assert result == before
    assert not trajectory.positions_rad.flags.writeable


@pytest.mark.parametrize("status", ["IK_FAIL", "TRAJOPT_FAIL", "INVALID_START_STATE"])
def test_native_failure_never_becomes_an_executable_trajectory(status):
    result = native_path()
    result.update(success=False, status=status)
    with pytest.raises(ValueError, match=status):
        convert(result)


@pytest.mark.parametrize("names", [list(reversed(NAMES[:-1])), [NAMES[0]] * 7,
                                  [*NAMES[:-1], "panda_finger_joint1"]])
def test_incomplete_duplicate_or_wrong_joint_names_are_rejected(names):
    result = native_path()
    result["joint_names"] = names
    with pytest.raises(ValueError, match="joint names"):
        convert(result)


@pytest.mark.parametrize("times", [[0.], [.01, .05, .1], [0., 0., .1],
                                  [0., .1, .05], [0., float("nan"), .1],
                                  [0., .05, float("inf")]])
def test_invalid_native_clock_is_rejected(times):
    result = native_path()
    result["times_s"] = times
    with pytest.raises(ValueError):
        convert(result)


def test_measured_start_is_required_without_rebasing_the_native_path():
    result = native_path()
    result["positions"][0][2] += .001
    with pytest.raises(ValueError, match="measured arm configuration"):
        convert(result)


@pytest.mark.parametrize("key,value", [
    ("positions", [[0.] * 6] * 3),
    ("positions", [[float("nan")] * 7] * 3),
    ("velocity", None), ("velocity", [[0.] * 7] * 2),
    ("velocity", [[float("inf")] * 7] * 3),
    ("velocity", [[-.901] * 7] * 3),
    ("acceleration", [[1.201] * 7] * 3),
])
def test_malformed_or_excessive_native_samples_are_rejected(key, value):
    result = native_path()
    result[key] = value
    with pytest.raises(ValueError):
        convert(result)


def test_payload_cover_is_frozen_once_at_grasp_and_pose_keeps_refreshing(monkeypatch):
    from mani_skill.examples.motionplanning.dual.sync import curobo_scene as geometry
    backend = client.CuroboBackend.__new__(client.CuroboBackend)
    identity = np.array([0., 0., 0., 1., 0., 0., 0.])
    target = SimpleNamespace(_objs=[object()], pose=identity.copy())
    agent = SimpleNamespace(robot=SimpleNamespace(pose=identity.copy()),
        tcp=SimpleNamespace(pose=identity.copy()), is_grasping=lambda _: np.array([True]))
    backend.raw = SimpleNamespace(agent=SimpleNamespace(agents=[agent]))
    backend.task = SimpleNamespace(targets=[target])
    backend.cache_dir = None
    backend.models = [{"capacity": 64, "target_obstacle_names": ["scene::target::shape_0"],
        "sphere_count": 1, "spheres_actor_local": [[0., 0., 0., .01]]}]
    calls = []
    monkeypatch.setattr(geometry, "export_curobo_snapshot", lambda *args, **kwargs:
                        {"robot_cfg": {}, "q_start": [0.] * 7})
    monkeypatch.setattr(geometry, "reserve_payload_config", lambda cfg, capacity:
                        {"capacity": capacity})
    monkeypatch.setattr(geometry, "_entity_shapes", lambda _: [object()])

    def refine(shape, pitch, relation):
        calls.append(list(relation))
        return [{"center": [0., 0., 0.], "radius": .01}], {"complete_solid_coverage": True}

    monkeypatch.setattr(geometry, "attached_shape_spheres", refine)
    checker = SimpleNamespace(attached=True)
    first = backend._snapshot(0, phase="carry_lower", closed=True, checker=checker)
    frozen = deepcopy(backend.models[0])
    target.pose[0] += .1
    second = backend._snapshot(0, phase="carry_lower", closed=True, checker=checker)
    assert len(calls) == 1 and backend.models[0] == frozen
    assert first["payload_model"]["cover_frozen_at_grasp"]
    assert first["payload_model"]["cover"]["version"] == "full_solid_tcp_negative_z_2mm_v1"
    assert second["payload_state"]["attachment"]["object_pose_base_wxyz"][0] == pytest.approx(.1)
    assert first["payload_state"]["attachment"]["object_pose_base_wxyz"][0] == 0.


# This is a real owned subprocess with pipes/select, but imports no worker or GPU
# package. Its responses exercise the public NDJSON boundary, not a mock request.
FAKE_WORKER = r'''
import json, os, sys, time
for line in sys.stdin:
    request = json.loads(line)
    op = request['op']
    if op == 'partial-line':
        sys.stdout.write('{"id":')
        sys.stdout.flush()
        time.sleep(5)
        continue
    elif op == 'error':
        print('native diagnostic: fixture failure', file=sys.stderr, flush=True)
        response = {'id': request['id'], 'ok': False, 'error': 'fixture worker error'}
    elif op == 'wrong-id':
        response = {'id': request['id'] + 1, 'ok': True, 'result': {}}
    elif op == 'invalid-result':
        response = {'id': request['id'], 'ok': True, 'result': []}
    else:
        response = {'id': request['id'], 'ok': True, 'result': {
            'received': request, 'success': op != 'native-failure',
            'status': 'IK_FAIL' if op == 'native-failure' else 'SUCCESS',
            'environment': {key: os.environ.get(key) for key in (
                'PYTHONPATH', 'PYTHONNOUSERSITE', 'WARP_CACHE_PATH', 'TORCH_EXTENSIONS_DIR')}}}
    print(json.dumps(response), flush=True)
    if op == 'shutdown':
        break
'''


@pytest.fixture
def worker_factory(monkeypatch):
    real_popen = subprocess.Popen
    clients, invocations = [], []

    def fake_popen(command, **kwargs):
        invocations.append(command)
        return real_popen([sys.executable, "-u", "-c", FAKE_WORKER], **kwargs)

    monkeypatch.setattr(client.subprocess, "Popen", fake_popen)

    def create(**kwargs):
        rpc = client.CuroboProcess(python=sys.executable, **kwargs)
        clients.append(rpc)
        return rpc

    yield create, invocations
    for rpc in clients:
        rpc.close()
        assert rpc.process.poll() is not None


def test_owned_transport_preserves_ndjson_ids_and_isolates_environment(
        worker_factory, monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONPATH", "parent-only-code")
    create, invocations = worker_factory
    rpc = create(device="cuda:3", cache_dir=str(tmp_path / "cache"))
    assert invocations[0][-1] == "--serve"
    assert rpc.device == "cuda:3"
    first = rpc.request("configure", session_id="left:approach", nested={"x": [1, 2]})
    second = rpc.request("plan_pose", session_id="left:approach", snapshot_id="snapshot-2")
    assert first["received"] == {"id": 1, "op": "configure", "session_id": "left:approach",
                                 "nested": {"x": [1, 2]}}
    assert second["received"]["id"] == 2
    assert second["received"]["snapshot_id"] == "snapshot-2"
    environment = first["environment"]
    assert environment["PYTHONPATH"] is None
    assert environment["PYTHONNOUSERSITE"] == "1"
    for key, directory in (("WARP_CACHE_PATH", "warp"),
                           ("TORCH_EXTENSIONS_DIR", "torch_extensions")):
        assert Path(environment[key]) == tmp_path / "cache" / directory
        assert Path(environment[key]).is_dir()
    assert [receipt["op"] for receipt in rpc.receipts] == ["configure", "plan_pose"]
    assert all(receipt["ok"] and receipt["wall_seconds"] >= 0 for receipt in rpc.receipts)
    rpc.close()
    assert rpc.process.stdin.closed and rpc.process.stdout.closed and rpc.log.closed
    assert rpc.receipts[-1]["op"] == "shutdown"
    rpc.close()


def test_native_planning_failure_is_a_valid_transport_response(worker_factory):
    create, _ = worker_factory
    rpc = create()
    result = rpc.request("native-failure")
    assert result["success"] is False and result["status"] == "IK_FAIL"
    assert rpc.failed is False
    assert rpc.receipts[-1]["ok"] is True
    assert rpc.request("echo")["success"] is True


def test_transport_deadline_also_covers_a_partial_response_line(worker_factory):
    create, _ = worker_factory
    rpc = create()
    rpc.request("echo")
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="exceeded"):
        rpc.request("partial-line", timeout_s=.05)
    assert time.monotonic() - started < 2.
    assert rpc.failed
    rpc.close()


@pytest.mark.parametrize("operation,error", [
    ("error", "fixture worker error"),
    ("wrong-id", "does not match"),
    ("invalid-result", "result must be an object"),
])
def test_worker_error_and_protocol_error_poison_only_the_owned_worker(
        worker_factory, operation, error):
    create, _ = worker_factory
    rpc = create()
    with pytest.raises(RuntimeError, match=error) as failure:
        rpc.request(operation)
    if operation == "error":
        assert "native diagnostic: fixture failure" in str(failure.value)
    assert rpc.failed and rpc.receipts[-1]["ok"] is False
    with pytest.raises(RuntimeError, match="unavailable"):
        rpc.request("echo")
    rpc.close()
    assert rpc.process.poll() is not None
    assert rpc.process.stdin.closed and rpc.process.stdout.closed and rpc.log.closed


@pytest.mark.parametrize("phase,closed,profile", [
    ("pregrasp", False, "approach"), ("descend", False, "approach"),
    ("carry_lower", True, "global"),
])
def test_backend_maps_measured_snapshots_base_poses_and_shared_hold_clock(phase, closed, profile):
    backend = client.CuroboBackend.__new__(client.CuroboBackend)
    backend.raw = SimpleNamespace(control_timestep=.05)
    backend.serial, backend.planning = 0, []
    backend.profile = profile
    backend.profile = profile
    starts = [START, START + .1]
    poses = [np.array([1., .2, .3, 1., 0., 0., 0.]),
             np.array([-1., .4, .5, 1., 0., 0., 0.])]
    bases = [np.array([.5, 0., 0., 1., 0., 0., 0.]),
             np.array([-.5, 0., 0., 1., 0., 0., 0.])]
    goals_base = [goal.copy() for goal in poses]
    for goal, base in zip(goals_base, bases):
        goal[:3] -= base[:3]
    snapshots, calls, fk_inputs = [], [], []
    user_names = ("panda_finger_joint2", NAMES[1], NAMES[0], *NAMES[2:], "panda_finger_joint1")
    active_names = (*NAMES, "panda_finger_joint1", "panda_finger_joint2")
    backend.arms, backend.solvers = [], []
    for index in (0, 1):
        model = SimpleNamespace(
            compute_forward_kinematics=lambda q, i=index: fk_inputs.append((i, q.copy())),
            get_link_pose=lambda link_index, i=index: goals_base[i])
        planner = SimpleNamespace(user_joint_names=user_names,
            move_group_joint_indices=[user_names.index(name) for name in NAMES],
            pinocchio_model=model, link_name_2_idx={"tcp": 7}, move_group="tcp")
        backend.solvers.append(SimpleNamespace(planner=planner))
        robot = SimpleNamespace(pose=bases[index], get_active_joints=lambda: [
            SimpleNamespace(get_name=lambda name=name: name) for name in active_names])
        backend.arms.append(SimpleNamespace(key=("left", "right")[index], names=NAMES,
            arm_qpos=lambda i=index: starts[i].copy(),
            full_qpos=lambda i=index: np.r_[starts[i], .012, .034],
            agent=SimpleNamespace(robot=robot)))

    def snapshot(index, **kwargs):
        snapshots.append((index, kwargs))
        return {"robot_cfg": {"arm": index}, "world": {"snapshot": index},
                "payload_state": {"closed": closed}, "q_start": starts[index].tolist()}

    def request(operation, **kwargs):
        calls.append((operation, kwargs))
        index = 0 if kwargs["session_id"].startswith("left:") else 1
        if operation == "update":
            return {"snapshot_id": f"measured-{index}"}
        return native_path(starts[index], duration=(.1, .15)[index])

    backend._snapshot = snapshot
    backend.rpc = SimpleNamespace(request=request)
    checker = object()
    schedule = backend.plan(phase, tuple(poses), closed=closed, checker=checker,
                            limits={"nominal_velocity_limit_rad_s": .9,
                                    "nominal_acceleration_limit_rad_s2": 1.2})
    assert snapshots == [(i, {"phase": phase, "closed": closed, "checker": checker}) for i in (0, 1)]
    assert [op for op, _ in calls] == ["update", "plan_pose", "update", "plan_pose"]
    for index, (_, request_body) in enumerate(calls[1::2]):
        assert request_body["session_id"] == f"{('left', 'right')[index]}:{profile}"
        assert request_body["snapshot_id"] == f"measured-{index}"
        np.testing.assert_allclose(request_body["goal_pose"], goals_base[index])
        np.testing.assert_array_equal(request_body["start"], starts[index])
    for index, q in fk_inputs:
        expected = dict(zip(active_names, np.r_[starts[index] + .01, .012, .034]))
        np.testing.assert_allclose(q, [expected[name] for name in user_names])
    assert schedule.num_steps == 3
    np.testing.assert_allclose(schedule.left_commands[-1, :7], START + .01)
    assert schedule.trajectories[0].duration_s == .1
    assert schedule.trajectories[1].duration_s == .15
    for trajectory in schedule.trajectories:
        assert trajectory.gripper_command == (-1. if closed else 1.)
    assert backend.planning[0]["name"] == phase
    assert all(arm["endpoint"]["status"] == "passed" for arm in backend.planning[0]["arms"])


@pytest.mark.parametrize("native_failure", [True, False])
def test_failed_native_or_rejected_trajectory_remains_in_phase_report(native_failure):
    backend = client.CuroboBackend.__new__(client.CuroboBackend)
    backend.raw, backend.serial, backend.planning = SimpleNamespace(control_timestep=.05), 0, []
    backend.profile = "global"
    identity = np.array([0., 0., 0., 1., 0., 0., 0.])
    backend.arms = [SimpleNamespace(key="left", names=NAMES, arm_qpos=lambda: START.copy(),
        agent=SimpleNamespace(robot=SimpleNamespace(pose=identity)))]
    backend._snapshot = lambda index, **kwargs: {"robot_cfg": {}, "world": {},
        "payload_state": {}, "q_start": START.tolist()}
    native = native_path()
    native.update(native_result_is_none=native_failure, graph_calls=0 if native_failure else 1,
                  graph_success=not native_failure, graph_time_s=0. if native_failure else .01,
                  solve_wall_s=.098, attempts=2)
    if native_failure:
        native.update(success=False, status="NO_SOLUTION", positions=None, times_s=None)
    else:
        native["velocity"][0][0] = 2.
    backend.rpc = SimpleNamespace(request=lambda op, **kwargs:
        {"snapshot_id": 2} if op == "update" else native)
    with pytest.raises(ValueError, match="did not find|exceeds"):
        backend.plan("lift_and_hold", (identity, identity), closed=True, checker=object(),
            limits={"nominal_velocity_limit_rad_s": .9, "nominal_acceleration_limit_rad_s2": 1.2})
    assert len(backend.planning) == 1
    report = backend.planning[0]
    assert report["name"] == "lift_and_hold" and report["status"] == "failed"
    assert report["wall_seconds"] >= 0 and report["error"].startswith("ValueError:")
    assert len(report["arms"]) == 1
    arm = report["arms"][0]
    assert arm["session_id"] == "left:global" and arm["update"]["snapshot_id"] == 2
    for key in ("status", "native_result_is_none", "graph_calls", "graph_success", "graph_time_s", "solve_wall_s"):
        assert arm[key] == native[key]
    assert "endpoint" not in arm
    json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
