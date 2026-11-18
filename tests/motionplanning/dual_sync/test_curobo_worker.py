"""CPU protocol/model tests; the optional CUDA stack is deliberately absent."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


WORKER_PATH = Path(__file__).resolve().parents[3] / "mani_skill/examples/motionplanning/dual/sync/curobo_worker.py"
spec = importlib.util.spec_from_file_location("standalone_curobo_worker", WORKER_PATH)
worker_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker_module)


def test_direct_process_needs_no_optional_imports() -> None:
    result = subprocess.run([sys.executable, str(WORKER_PATH), "--serve"],
        input=json.dumps({"id": 7, "op": "shutdown"})+"\n", text=True, capture_output=True, check=True)
    assert json.loads(result.stdout) == {"id": 7, "ok": True, "result": {"shutdown": True}}
    script = ("import runpy,sys; runpy.run_path(sys.argv[1],run_name='import_test'); "
              "assert not any(n in sys.modules for n in ['torch','curobo','sapien','mani_skill'])")
    subprocess.run([sys.executable, "-c", script, str(WORKER_PATH)], check=True)


def test_explicit_cuda_device_is_current_before_native_allocation() -> None:
    selected = []
    torch = SimpleNamespace(device=lambda name: SimpleNamespace(type=name.split(":")[0], index=3),
        cuda=SimpleNamespace(is_available=lambda: True, set_device=selected.append))
    device = worker_module.select_cuda_device(torch, "cuda:3")
    assert device.index == 3 and selected == [device]
    with pytest.raises(ValueError, match="CUDA device"):
        worker_module.select_cuda_device(torch, "cpu")
    assert selected == [device]


@pytest.mark.parametrize("native,graph_calls,graph_success,graph_time", [
    (None, 0, False, 0.), (None, 2, False, .02),
    (SimpleNamespace(success=[False], feasible=[False], position_error=[.0001], rotation_error=[.0002]),
     1, True, .015),
])
def test_native_failure_diagnostics_preserve_ik_graph_and_trajopt_boundaries(
        native, graph_calls, graph_success, graph_time) -> None:
    calls = []
    session = worker_module.Session()
    session.ready, session.snapshot_id, session.tool_frame = True, 3, "panda_hand_tcp"
    session.torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
    session.device = SimpleNamespace(to_device=lambda value: value)
    session.JointState = SimpleNamespace(from_position=lambda *args, **kwargs: (args, kwargs))
    session.Pose = SimpleNamespace(from_list=lambda *args, **kwargs: (args, kwargs))
    session.GoalToolPose = SimpleNamespace(from_poses=lambda *args, **kwargs: (args, kwargs))
    def solve(*args, **kwargs):
        calls.append(kwargs)
        return native
    session.planner = SimpleNamespace(plan_pose=solve, last_planning_attempts=2,
        last_graph_calls=graph_calls, last_graph_success=graph_success, last_graph_time=graph_time)
    result = session.plan_pose({"snapshot_id": 3, "start": [0.]*7,
        "goal_pose": [0., 0., 0., 1., 0., 0., 0.], "attempts": 2})
    assert result["success"] is False and result["positions"] is None
    assert result["native_result_is_none"] is (native is None)
    assert result["graph_calls"] == graph_calls and result["graph_success"] is graph_success
    assert result["graph_time_s"] == graph_time and result["attempts"] == 2
    assert calls == [{"max_attempts": 2, "enable_graph_attempt": 0}]
    if native is not None:
        assert result["feasible"] == [False] and result["position_error"] == [.0001]


def test_real_urdf_axes_and_version_specific_limit_compensation(tmp_path: Path) -> None:
    urdf = tmp_path / "panda.urdf"
    arms = "".join(f'<joint name="{n}"><limit velocity="{2.+i/10}"/></joint>'
                   for i, n in enumerate(worker_module.ARM_NAMES))
    fingers = "".join(f'<joint name="{n}" type="prismatic"><parent link="panda_hand"/>'
        f'<child link="{child}"/><origin xyz="1 2 3" rpy="0 0 {math.pi/2}"/>'
        f'<axis xyz="{direction} 0 0"/></joint>'
        for n, child, direction in zip(worker_module.FINGER_NAMES, worker_module.FINGER_LINKS, (1, -1)))
    urdf.write_text(f"<robot>{arms}{fingers}</robot>")
    limits, joints = worker_module.load_urdf(str(urdf))
    first = worker_module.finger_transform(joints[worker_module.FINGER_NAMES[0]], .03)
    second = worker_module.finger_transform(joints[worker_module.FINGER_NAMES[1]], .017)
    assert first[0][3] == pytest.approx(1.) and first[1][3] == pytest.approx(2.03)
    assert second[1][3] == pytest.approx(1.983)
    names = worker_module.FINGER_NAMES+worker_module.ARM_NAMES[::-1]
    robot = {"kinematics": {"lock_joints": dict(zip(worker_module.FINGER_NAMES, (.03, .017))),
                            "cspace": {"joint_names": names, "default_joint_position": [0.]*9}}}
    original = deepcopy(robot)
    profiled = worker_module.profile_config(robot, .9, 1.2, limits)
    assert robot == original
    for name, scale in zip(names, profiled["kinematics"]["cspace"]["velocity_scale"]):
        assert (limits[name]*scale*scale == pytest.approx(.9)) if name in limits else scale == 1.
    urdf.write_text(urdf.read_text().replace('type="prismatic"', 'type="revolute"', 1))
    with pytest.raises(ValueError, match="independent-finger"):
        worker_module.load_urdf(str(urdf))


def payload_fixture() -> tuple[dict, dict, dict, list]:
    names = ["scene::own::shape_0", "scene::own::shape_1"]
    model = {"capacity": 2, "target_obstacle_names": names,
             "spheres_actor_local": [[0., 0., 0., .01]]}
    state = worker_module.empty_payload_state()
    world = {"cuboid": {n: {} for n in [*names, "scene::peer::shape_0", worker_module.SOURCE_TABLE]}}
    links = ["panda_link0", "panda_link7", *worker_module.FINGER_LINKS, worker_module.PAYLOAD_LINK]
    return model, state, world, links


def test_payload_permissions_keep_peer_and_table_and_robot_geometry() -> None:
    model, state, world, links = payload_fixture()
    layers, masks, disabled = worker_module.collision_policy(model, state, world, links)
    assert disabled == set() and masks["panda_link7"] == 7
    assert masks["panda_link0"] == 3  # only fixed base/table mounting exception
    state["contact_allowed"] = True
    _, masks, disabled = worker_module.collision_policy(model, state, world, links)
    assert masks["panda_leftfinger"] == 5 and masks[worker_module.PAYLOAD_LINK] == 7
    assert disabled == set()
    state.update(source_support_allowed=True, source_support_obstacle_names=[worker_module.SOURCE_TABLE],
        attachment={"obstacle_names": model["target_obstacle_names"], "positions": [0.]*7,
                    "object_pose_base_wxyz": [0., 0., 0., 1., 0., 0., 0.],
                    "spheres": model["spheres_actor_local"]})
    layers, masks, disabled = worker_module.collision_policy(model, state, world, links)
    assert disabled == set(model["target_obstacle_names"])
    assert "scene::peer::shape_0" not in disabled and worker_module.SOURCE_TABLE not in disabled
    assert layers["scene::peer::shape_0"] == 1 and layers[worker_module.SOURCE_TABLE] == 4
    assert masks[worker_module.PAYLOAD_LINK] == 3 and masks["panda_link7"] == 7
    assert masks["panda_leftfinger"] == 5  # fingers still collide with the table
    state["source_support_allowed"] = False
    state["source_support_obstacle_names"] = []
    _, masks, _ = worker_module.collision_policy(model, state, world, links)
    assert masks[worker_module.PAYLOAD_LINK] == 7


def test_payload_rejects_undeclared_geometry_and_source_permissions() -> None:
    model, state, world, links = payload_fixture()
    state.update(source_support_allowed=True, source_support_obstacle_names=[worker_module.SOURCE_TABLE])
    with pytest.raises(ValueError, match="held object"):
        worker_module.collision_policy(model, state, world, links)
    state["source_support_obstacle_names"] = ["scene::peer::shape_0"]
    with pytest.raises(ValueError, match="exact original table"):
        worker_module.collision_policy(model, state, world, links)
    state = worker_module.empty_payload_state()
    state["contact_allowed"] = True
    with pytest.raises(ValueError, match="reserved payload"):
        worker_module.collision_policy(None, state, world, links)


def test_payload_model_owns_exact_target_and_preserves_self_collision() -> None:
    model, _, world, links = payload_fixture()
    model.update(target_actor_name="own", link_name=worker_module.PAYLOAD_LINK,
                 finger_links=worker_module.FINGER_LINKS)
    kin = {"collision_link_names": links, "extra_collision_spheres": {worker_module.PAYLOAD_LINK: 2},
           "extra_links": {worker_module.PAYLOAD_LINK: {"parent_link_name": "panda_hand_tcp",
                "joint_type": "FIXED", "fixed_transform": [0., 0., 0., 1., 0., 0., 0.]}},
           "self_collision_ignore": {worker_module.PAYLOAD_LINK: worker_module.FINGER_LINKS}}
    worker_module.validate_payload_model(model, kin, world)
    changed = deepcopy(model)
    changed["target_obstacle_names"].append("scene::peer::shape_0")
    with pytest.raises(ValueError, match="own target actor"):
        worker_module.validate_payload_model(changed, kin, world)
    kin["self_collision_ignore"][worker_module.PAYLOAD_LINK] = [*worker_module.FINGER_LINKS, "panda_link7"]
    with pytest.raises(ValueError, match="exactly its own fingers"):
        worker_module.validate_payload_model(model, kin, world)


def test_named_trajectory_columns_derivatives_and_finite_contract() -> None:
    names = worker_module.FINGER_NAMES+worker_module.ARM_NAMES[::-1]
    raw = [[.02, .03, 7., 6., 5., 4., 3., 2., 1.], [.02, .03, 8., 7., 6., 5., 4., 3., 2.]]
    trajectory = SimpleNamespace(joint_names=names, position=[raw], velocity=[raw], acceleration=[raw], dt=[.05])
    result = worker_module.trajectory_fields(trajectory)
    assert result["positions"] == [[1., 2., 3., 4., 5., 6., 7.], [2., 3., 4., 5., 6., 7., 8.]]
    assert result["velocity"] == result["acceleration"] == result["positions"]
    assert result["times_s"] == [0., .05]
    trajectory.position[0][0][2] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        worker_module.trajectory_fields(trajectory)


def test_limits_are_checked_from_actual_native_readback() -> None:
    limits = SimpleNamespace(joint_names=worker_module.ARM_NAMES, velocity=[[-.9]*7, [.9]*7],
                             acceleration=[[-1.2]*7, [1.2]*7])
    assert worker_module.model_limit_receipt(limits, .9, 1.2)["timeopt_default_limit_scale"] == .95
    limits.velocity[0][0] = -.9*.9/2.175
    with pytest.raises(RuntimeError, match="Native velocity"):
        worker_module.model_limit_receipt(limits, .9, 1.2)


def test_resident_topology_changes_require_reconfigure() -> None:
    robot = {"kinematics": {"lock_joints": {"f": .01}, "collision_spheres": {"link": [1]},
                            "cspace": {"default_joint_position": [0.], "max_acceleration": 1.2}}}
    other = deepcopy(robot)
    other["kinematics"]["lock_joints"]["f"] = .02
    other["kinematics"]["cspace"]["default_joint_position"] = [1.]
    assert worker_module.robot_topology(robot) == worker_module.robot_topology(other)
    other["kinematics"]["collision_spheres"]["link"] = [2]
    assert worker_module.robot_topology(robot) != worker_module.robot_topology(other)
    world = {"cuboid": {"box": {"pose": [0., 0., 0., 1., 0., 0., 0.], "dims": [1., 2., 3.]}}}
    shifted = deepcopy(world)
    shifted["cuboid"]["box"]["pose"][0] = 5.
    assert worker_module.world_topology(world) == worker_module.world_topology(shifted)
    shifted["cuboid"]["box"]["dims"][0] = 2.
    assert worker_module.world_topology(world) != worker_module.world_topology(shifted)


class FakeSession:
    def __init__(self) -> None:
        self.ready = False
        self.snapshot_id = 0
        self.closed = False
    def configure(self, request: dict) -> dict:
        return self.update(request)
    def update(self, request: dict) -> dict:
        self.ready = False
        if request.get("fail"):
            raise RuntimeError("Synthetic interrupted world update")
        self.snapshot_id += 1
        self.ready = True
        return {"snapshot_id": self.snapshot_id}
    def plan_pose(self, request: dict) -> dict:
        if not self.ready or request.get("snapshot_id") != self.snapshot_id:
            raise RuntimeError("Stale snapshot")
        return {"success": False, "status": "NO_SOLUTION"}
    def shutdown(self) -> None:
        self.closed = True
        self.ready = False


def test_four_sessions_reuse_stale_update_and_failure_recovery() -> None:
    worker = worker_module.Worker(FakeSession)
    for name in ("left:approach", "left:global", "right:approach", "right:global"):
        assert worker.dispatch({"op": "configure", "session_id": name})["snapshot_id"] == 1
    resident = worker.sessions["left:global"]
    worker.dispatch({"op": "update", "session_id": "left:global"})
    assert worker.sessions["left:global"] is resident
    with pytest.raises(RuntimeError, match="Stale"):
        worker.dispatch({"op": "plan_pose", "session_id": "left:global", "snapshot_id": 1})
    assert worker.dispatch({"op": "plan_pose", "session_id": "left:global", "snapshot_id": 2})["success"] is False
    with pytest.raises(RuntimeError, match="Synthetic"):
        worker.dispatch({"op": "update", "session_id": "left:global", "fail": True})
    assert not resident.ready
    assert all(not s.ready for s in worker.sessions.values())
    with pytest.raises(RuntimeError):
        worker.dispatch({"op": "plan_pose", "session_id": "left:global", "snapshot_id": 2})
    worker.dispatch({"op": "update", "session_id": "left:global"})
    assert worker.dispatch({"op": "plan_pose", "session_id": "left:global", "snapshot_id": 3})["success"] is False
    with pytest.raises(RuntimeError, match="already"):
        worker.dispatch({"op": "configure", "session_id": "left:global"})
    with pytest.raises(RuntimeError, match="two arms"):
        worker.dispatch({"op": "configure", "session_id": "extra"})
    worker.shutdown()
    assert resident.closed and worker.sessions == {}


def test_jsonl_preserves_ids_errors_and_native_failure() -> None:
    requests = [[], {"id": 2, "op": "unknown"}, {"id": 3, "op": "configure", "session_id": "left"},
                {"id": 4, "op": "plan_pose", "session_id": "left", "snapshot_id": 1},
                {"id": 5, "op": "shutdown"}]
    source = io.StringIO("\n".join(json.dumps(r) for r in requests))
    sink = io.StringIO()
    worker_module.serve(source, sink, worker_module.Worker(FakeSession))
    rows = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert [r["id"] for r in rows] == [None, 2, 3, 4, 5]
    assert [r["ok"] for r in rows] == [False, False, True, True, True]
    assert rows[3]["result"]["status"] == "NO_SOLUTION"


def test_retired_approach_sessions_release_resources_before_transport() -> None:
    worker = worker_module.Worker(FakeSession)
    for arm in ("left", "right"):
        worker.dispatch({"op": "configure", "session_id": f"{arm}:approach"})
    previous = list(worker.sessions.values())
    for arm in ("left", "right"):
        worker.dispatch({"op": "release", "session_id": f"{arm}:approach"})
    assert worker.sessions == {} and all(session.closed for session in previous)
    for arm in ("left", "right"):
        worker.dispatch({"op": "configure", "session_id": f"{arm}:global"})
    assert len(worker.sessions) == 2
    with pytest.raises(RuntimeError, match="configure must succeed"):
        worker.dispatch({"op": "plan_pose", "session_id": "left:approach", "snapshot_id": 1})
    worker.shutdown()
