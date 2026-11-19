"""Failure-boundary regressions for the portable mesh/grasp interface."""
from pathlib import Path
import json
import tempfile
import numpy as np
import pytest
import trimesh

from resimpair_grasp.common import binding, read, resolve, write
from resimpair_grasp.grasps import GRIPPER, load, panda_frame, backend
from resimpair_grasp.assets import select, validate


def test_tool_frame_is_proper_and_closing_belongs_to_panda_y():
    source = np.eye(4); source[:3, :3] = np.diag([1, -1, -1]); source[:3, 3] = [.02, -.01, .03]
    target = panda_frame(source)
    np.testing.assert_allclose(target[:3, 1], source[:3, 0])
    np.testing.assert_allclose(target[:3, 2], [0, 0, -1])
    np.testing.assert_allclose(target[:3, 3], source[:3, 3])
    assert np.linalg.det(target[:3, :3]) == pytest.approx(1.)
    source[:3, 0] = [0, 0, 1]
    assert panda_frame(source) is None


def test_antipodal_backend_uses_surface_not_box():
    sampler, Config, Obj = backend()
    # A round solid provides actual curved opposing contacts. This is a CPU
    # geometry test, never reported as physical success.
    mesh = trimesh.creation.cylinder(radius=.025, height=.08, sections=24)
    obj = Obj(mesh, mesh.extents, np.eye(4), mesh.extents, "cylinder")
    cfg = Config(antipodal_samples=120, antipodal_max_grasps=12,
        approach_facets=((0, 0, -1),), cone_deg=3, directed_spread=1, contact_geometry_policy="mesh")
    gs = sampler(obj, cfg, np.random.default_rng(12))
    assert gs and all(g.kind == "antipodal" and .02 < g.width <= .08 for g in gs)
    assert all(g.meta["contact_geometry"] == "mesh" for g in gs)
    cfg.gripper_max_width = .01
    assert not sampler(obj, cfg, np.random.default_rng(12))


def test_mesh_hash_change_and_gripper_change_reject_cache(tmp_path):
    p = tmp_path / "descriptor.json"; write(p, {"scale": 1.})
    b = binding(p)
    cache = tmp_path / "cache.json"
    write(cache, {"signature": {"descriptor": b, "gripper": GRIPPER}, "candidates": []})
    assert load(binding(cache), b)["candidates"] == []
    mismatch = dict(b, sha256="0" * 64)
    with pytest.raises(ValueError): load(binding(cache), mismatch)
    p.write_text('{"scale": 2}')
    with pytest.raises(ValueError): load(binding(cache), b)


def test_nonrigid_pose_and_reflection_are_rejected(tmp_path):
    p = tmp_path / "d.json"; write(p, {})
    for i, scale in enumerate([2., -1.]):
        T = np.eye(4); T[0, 0] = scale
        cache = tmp_path / f"bad_{i}.json"
        write(cache, {"signature": {"descriptor": binding(p), "gripper": GRIPPER},
            "candidates": [{"tcp_in_asset": T.tolist()}]})
        with pytest.raises(ValueError): load(binding(cache), binding(p))


def test_static_asset_cannot_be_promoted_by_role_name(tmp_path):
    d = tmp_path / "d.json"
    write(d, {"schema": "resimpair_mesh_asset_v1", "id": "cup", "category": "cup", "dimensions_m": [.1, .1, .1],
              "half_size": [.05] * 3, "roles": {"static_distractor": True, "rigid_manipulation_candidate": False}})
    c = tmp_path / "catalog.json"; write(c, {"assets": [binding(d)]})
    assert len(select(c, role="static_distractor")) == 1
    assert not select(c, role="rigid_manipulation_candidate")
    assert not select(c, max_dimension=.05)
    with pytest.raises(ValueError):
        validate({"asset_descriptor": binding(d), "half_size": [.05] * 3, "kind": "movable"})


def test_evidence_is_immutable(tmp_path):
    p = tmp_path / "attempt.json"; write(p, {"passed": False})
    with pytest.raises(ValueError): write(p, {"passed": True})
    assert read(p) == {"passed": False}


def test_payload_checker_uses_world_frame_and_keeps_table_constraint():
    from types import SimpleNamespace
    from mplib.pymp.collision_detection import fcl
    from resimpair_grasp.motion import MeshMotion
    from data_generation_methods.human_to_robot_pipeline.strict_motion import matrix_to_pose7
    guard = MeshMotion.__new__(MeshMotion)
    guard.planner = SimpleNamespace(pinocchio_model=SimpleNamespace(compute_forward_kinematics=lambda q: None))
    guard.physical_robot_objects = []
    guard.selected_collision = fcl.CollisionObject(fcl.Box(.02, .02, .02))
    guard.payload_collision = fcl.CollisionObject(fcl.Box(.02, .02, .02))
    obstacle = fcl.CollisionObject(fcl.Box(.04, .04, .04))
    obstacle.set_transformation(np.array([-.15, -.30, .10, 1, 0, 0, 0]))
    guard.table_collision = fcl.CollisionObject(fcl.Box(2., 2., .1))
    guard.table_collision.set_transformation(np.array([0, 0, -.05, 1, 0, 0, 0]))
    guard.environment_collision_objects = [("obstacle", obstacle)]
    guard.use_attach = True; guard.payload_to_tcp = np.eye(4)
    guard.check_stage = "above_bowl"; guard.check_initial_tcp = np.eye(4)
    guard.support_floor_z = 0.; guard.check_dense_index = 0; guard.allowed_support_contacts = []
    tcp = np.eye(4); tcp[:3, 3] = [-.15, -.30, .10]
    pairs = guard.independent_table_collisions(np.zeros(9), tcp)
    assert {"link1": "held_object", "link2": "obstacle"} in pairs
    tcp[:3, 3] = [.44, .05, .10]
    assert not guard.independent_table_collisions(np.zeros(9), tcp)
    tcp[:3, 3] = [-.2, -.3, 0.]
    assert {"link1": "held_object", "link2": "complete_table"} in guard.independent_table_collisions(np.zeros(9), tcp)
