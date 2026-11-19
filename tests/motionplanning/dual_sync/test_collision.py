"""Collision sampling and actual cooked-geometry support boundaries."""

from types import SimpleNamespace

import numpy as np
import pytest
import sapien
import sapien.physx as physx
from mplib.pymp.collision_detection import fcl
from transforms3d.euler import euler2mat

from mani_skill.examples.motionplanning.dual.sync.combined_collision import (
    validate_schedule
)
from mani_skill.examples.motionplanning.dual.sync.dual_payload_collision import (
    DualPayloadCollisionChecker
)
from mani_skill.examples.motionplanning.dual.sync.held_collision import (
    _object_at, _pair_result
)
from mani_skill.examples.motionplanning.dual.sync.physical_observation import (
    contact_allowed, physical_contact_points, physical_support_transition
)
from mani_skill.examples.motionplanning.dual.sync.pill_support_collision import (
    PillTableSupportCollisionChecker
)
from mani_skill.examples.motionplanning.dual.sync.table_support_collision import (
    TABLE_NAME
)
from mani_skill.examples.motionplanning.dual.sync.trajectory_schedule import (
    ArmTrajectory, build_hold_schedule
)


def trajectory(key, start, end, duration):
    positions = np.zeros((2, 7))
    positions[:, 0] = [start, end]
    return ArmTrajectory(key, tuple(f"joint{i}" for i in range(7)), [0, duration], positions, -1)


def test_combined_sampling_detects_collision_between_clear_endpoints():
    left, right = trajectory("left", 0, 1, 1), trajectory("right", 1, 0, 1)
    schedule = build_hold_schedule(left, right, control_timestep_s=1)
    samples = []

    def check(left7, right7):
        samples.append(np.r_[left7, right7])
        return [{"collision_type": "mutual"}] if abs(left7[0] - right7[0]) < 0.01 else []

    checker = SimpleNamespace(check=check)
    assert not check(left.positions_rad[0], right.positions_rad[0])
    assert not check(left.positions_rad[-1], right.positions_rad[-1])
    samples.clear()
    report = validate_schedule(schedule, left.positions_rad[0], right.positions_rad[0], checker,
                               max_joint_step_rad=0.125)
    assert report["status"] == "collision"
    assert report["first_collision"]["time_s"] == 0.5
    assert report["sample_count"] == 5
    assert all(np.isclose(q[0] + q[7], 1) for q in samples)


def test_sampling_budget_rejected_before_any_collision_queries():
    left, right = trajectory("left", 0, 1, 1), trajectory("right", 2, 0, 2)
    schedule = build_hold_schedule(left, right, control_timestep_s=1)
    checker = SimpleNamespace(check=lambda *_: pytest.fail("Budget must be checked before sampling"))
    with pytest.raises(ValueError, match="sampling budget|samples"):
        validate_schedule(schedule, left.positions_rad[0], right.positions_rad[0], checker,
                          max_joint_step_rad=0.01, max_samples=3)


def pill_fixture(height=-1.14e-6, *, x=0.0, two_pieces=False):
    """Cook actual PhysX convexes and query them using the deployed FCL backend."""
    table_pose = np.eye(4)
    table_pose[2, 3] = -0.05
    table = _object_at(fcl.Box(np.array([2.0, 1.2, 0.1])), table_pose)
    material = physx.PhysxMaterial(0.5, 0.5, 0.0)
    cube = np.array([[a, b, c] for a in (-0.01, 0.01) for b in (-0.01, 0.01)
                     for c in (0.0, 0.02)], dtype=np.float32)
    checker = PillTableSupportCollisionChecker()
    checker._initial_object = np.eye(4)
    checker._initial_object[:3, 3] = [x, 0, height]
    checker._target_names = frozenset(("piece0", "piece1") if two_pieces else ("piece0",))
    checker._shape_local = {}
    refs, objects = {}, {TABLE_NAME: table}
    entity = sapien.Entity()
    entity.pose = sapien.Pose([x, 0, height])
    for index, name in enumerate(sorted(checker._target_names)):
        shape = physx.PhysxCollisionShapeConvexMesh(cube, [1.0, 1.0, 1.0], material)
        local = np.eye(4)
        if two_pieces:
            local[0, 3] = 0.04 * (2 * index - 1)
            local[2, 3] = 0.004 * index
        shape.set_local_pose(sapien.Pose(local[:3, 3]))
        refs[name] = (entity, shape)
        checker._shape_local[name] = local
        geometry = fcl.Convex(np.asarray(shape.vertices, dtype=float),
                              np.asarray(shape.get_triangles(), dtype=np.int32), True)
        objects[name] = _object_at(geometry, checker._initial_object @ local)
    checker.live = SimpleNamespace(_shape_refs=refs)
    checker._support_pairs = {}
    checker.support_released = False
    checker.support_band_m = checker.support_penetration_m = 0.003
    checker.support_depth_slop_m = 0.0002
    checker._register_initial_support(objects)
    checker.coverage = {}
    return checker, objects


def objects_at(checker, objects, pose):
    return {**objects, **{name: _object_at(objects[name].get_collision_geometry(),
                                         pose @ checker._shape_local[name])
                         for name in checker._target_names}}


def support_allowed(checker, objects, pose):
    collided, contacts = _pair_result(objects_at(checker, objects, pose)["piece0"], objects[TABLE_NAME])
    assert collided and contacts
    return checker._support_allowed("piece0", TABLE_NAME, pose, contacts)


@pytest.mark.parametrize("kwargs", [{"height": -0.00301}, {"height": -0.115}, {"x": 0.999}])
def test_initial_support_rejects_deep_underside_or_table_edge(kwargs):
    with pytest.raises(RuntimeError):
        pill_fixture(**kwargs)


def test_initial_micrometre_gap_does_not_fabricate_contact_or_close_support():
    checker, objects = pill_fixture(2.01e-6)
    assert checker._initial_support_exists and not checker._support_pairs
    assert checker._initial_near_top_support_receipts[0]["registration_does_not_assert_contact"]
    checker._update_support_release(objects, checker._initial_object)
    assert checker._support_state_receipt["geometry_all_pieces_clear"]
    assert not checker.support_released
    contact_pose = checker._initial_object.copy()
    contact_pose[:3, :3] = euler2mat(0.0, 0.0004, 0.0)
    contact_pose[2, 3] += 0.5e-6
    assert support_allowed(checker, objects, contact_pose)


@pytest.mark.parametrize("displacement,angle", [
    ((-1.5e-6, -2.2e-6, -8.3e-6), .00076),
    ((.005, 0., 0.), .02),
])
def test_supported_geometry_survives_origin_motion_window(displacement, angle):
    checker, objects = pill_fixture()
    pose = checker._initial_object.copy()
    pose[:3, 3] += displacement
    pose[:3, :3] = euler2mat(0., 0., angle)
    assert support_allowed(checker, objects, pose)
    receipt = checker._last_support_validation
    assert not receipt["original_motion_window_within_bounds"]
    assert receipt["original_motion_window_used_for_acceptance"] is False
    assert all(p["entire_footprint_inside"] and p["vertices_above_top"]
               and p["vertex_penetration_m"] <= receipt["depth_limit_m"]
               for p in receipt["whole_object_pieces"])
    assert not checker.support_released


@pytest.mark.parametrize("height,downward_m,expected_limit_m", [
    (-1.14e-6, .00021, .00020114),
    (-.0029, .00015, .003),
])
def test_geometry_rejects_depth_growth_and_absolute_penetration_cap(
        height, downward_m, expected_limit_m):
    checker, objects = pill_fixture(height)
    pose = checker._initial_object.copy()
    pose[2, 3] -= downward_m
    assert not support_allowed(checker, objects, pose)
    receipt = checker._last_support_validation
    assert receipt["depth_limit_m"] == pytest.approx(expected_limit_m, abs=1e-7)
    assert receipt["reason"] == "Supplied whole-object geometry exceeds bounded top support"
    assert any(p["vertex_penetration_m"] > receipt["depth_limit_m"]
               for p in receipt["whole_object_pieces"])


@pytest.mark.parametrize("two_pieces,x", [(False, .995), (True, .96)])
def test_finite_table_rejects_any_convex_piece_crossing_edge(two_pieces, x):
    checker, objects = pill_fixture(two_pieces=two_pieces)
    pose = checker._initial_object.copy()
    pose[0, 3] = x
    assert not support_allowed(checker, objects, pose)
    receipt = checker._last_support_validation
    assert receipt["reason"] == "Supplied whole-object geometry exceeds bounded top support"
    pieces = {p["target"]: p for p in receipt["whole_object_pieces"]}
    assert any(not p["entire_footprint_inside"] for p in pieces.values())
    if two_pieces:
        # The contacting piece stays valid; the noncontacting piece crosses the edge.
        assert pieces["piece0"]["entire_footprint_inside"]
        assert not pieces["piece1"]["entire_footprint_inside"]


def test_support_departure_requires_all_pieces_then_permanently_rejects_recontact():
    checker, objects = pill_fixture(two_pieces=True)
    tilted = checker._initial_object.copy()
    tilted[2, 3] += 0.0031
    tilted[:3, :3] = euler2mat(0.0, -0.1, 0.0)
    checker._update_support_release(objects_at(checker, objects, tilted), tilted)
    assert checker._support_state_receipt["actor_exceeds_initial_support_band"]
    assert not checker._support_state_receipt["geometry_all_pieces_clear"]
    assert not checker.support_released
    clear = checker._initial_object.copy()
    clear[2, 3] += 0.004
    checker._update_support_release(objects_at(checker, objects, clear), clear)
    assert checker.support_released
    checker._update_support_release(objects, checker._initial_object)
    assert checker.support_released
    assert not support_allowed(checker, objects, checker._initial_object)


def test_candidate_uses_supplied_geometry_without_changing_measured_release():
    checker, objects = pill_fixture()
    assert support_allowed(checker, objects, checker._initial_object)
    deep = checker._initial_object.copy()
    deep[2, 3] -= 0.004
    state_before = dict(checker._support_state_receipt)
    assert not support_allowed(checker, objects, deep)
    assert checker._support_state_receipt == state_before
    assert not checker.support_released
    assert checker._world_vertices("piece0", deep)[:, 2].min() < -0.004


def test_initial_support_uses_actual_geometry_instead_of_vertical_pick_window():
    checker, objects = pill_fixture()
    pose = checker._initial_object.copy()
    pose[0, 3] += .002
    pose[2, 3] -= 20e-6
    pose[:3, :3] = euler2mat(0., 0., .05)
    assert support_allowed(checker, objects, pose)
    receipt = checker._last_support_validation
    assert receipt['motion_window_is_collision_gate'] is False
    assert receipt['legacy_motion_window_passed'] is False
    assert not checker.support_released


@pytest.mark.parametrize('axis,value', [(2, -.0003), (0, .995)])
def test_initial_support_still_rejects_depth_growth_and_finite_table_edge(axis, value):
    checker, objects = pill_fixture()
    pose = checker._initial_object.copy()
    pose[axis, 3] = value
    assert not support_allowed(checker, objects, pose)


def test_measured_attachment_rejects_snapshot_after_physical_tick():
    checker = DualPayloadCollisionChecker()
    checker._fresh = checker.attached = checker.measured_attachment_prediction = True
    checker._prediction_relations = (np.eye(4), np.eye(4))
    checker._prediction_control_step = 1
    checker.env = SimpleNamespace(elapsed_steps=np.array([2]))
    with pytest.raises(RuntimeError, match="Physical control step changed"):
        checker._assert_prediction_current()


def test_native_contact_permissions_and_physical_support_are_separate_facts():
    proximity = {"separation_m": 1e-5, "impulse_Ns": [0, 0, 0]}
    assert not physical_contact_points([proximity])
    assert physical_contact_points([{**proximity, "separation_m": 0}])
    assert physical_contact_points([{**proximity, "impulse_Ns": [0, 1e-10, 0]}])
    sides = [{"kind": "robot", "robot_index": 0, "link_name": "panda_leftfinger"},
             {"kind": "payload", "target_index": 0}]
    assert contact_allowed(sides, "descend", {})
    assert not contact_allowed(sides, "pregrasp", {})
    assert not contact_allowed([{**sides[0], "robot_index": 1}, sides[1]], "descend", {})
    assert not contact_allowed([{**sides[0], "link_name": "panda_hand"}, sides[1]], "descend", {})
    assert not physical_support_transition(False, True, 0.0, 0.0029, False)
    assert not physical_support_transition(False, True, 0.0, 0.0031, True)
    assert physical_support_transition(False, True, 0.0, 0.0031, False)
    assert physical_support_transition(True, False, 0.0, 0.0, True)
