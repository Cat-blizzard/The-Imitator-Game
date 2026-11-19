"""Fixed support evidence is captured once and exported as independent snapshots."""
from __future__ import annotations

import copy
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import sapien
import sapien.physx as physx
from mplib.pymp.collision_detection import fcl

from mani_skill.examples.motionplanning.dual.sync import table_support_collision as support
from mani_skill.examples.motionplanning.dual.sync.held_collision import (
    AttachedObjectCollisionChecker, _object_at,
)
from mani_skill.examples.motionplanning.dual.sync.pill_support_collision import (
    PillTableSupportCollisionChecker,
)


def support_fixture(checker_type: type) -> tuple[object, dict]:
    """Register two actual cooked convex pieces against a finite native table."""
    checker = checker_type()
    checker._initial_object = np.eye(4)
    checker._initial_object[2, 3] = -1.14e-6
    checker._target_names = frozenset(("piece0", "piece1"))
    checker._shape_local = {}
    material = physx.PhysxMaterial(0.5, 0.5, 0.0)
    vertices = np.array([[x, y, z] for x in (-0.01, 0.01)
                         for y in (-0.01, 0.01) for z in (0., 0.02)], dtype=np.float32)
    entity = sapien.Entity()
    entity.pose = sapien.Pose(checker._initial_object[:3, 3])
    table_pose = np.eye(4)
    table_pose[2, 3] = -0.05
    objects = {support.TABLE_NAME: _object_at(fcl.Box(np.array([2., 1.2, 0.1])), table_pose)}
    refs = {}
    for index, name in enumerate(sorted(checker._target_names)):
        shape = physx.PhysxCollisionShapeConvexMesh(vertices, [1., 1., 1.], material)
        local = np.eye(4)
        local[0, 3] = 0.04 * index
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


@pytest.mark.parametrize("checker_type", [support.NumericalTableSupportCollisionChecker,
                                          PillTableSupportCollisionChecker])
def test_fingerprints_are_computed_only_at_registration(monkeypatch, checker_type):
    original = hashlib.sha256
    hashed_bytes = []

    def counted(data=b"", **kwargs):
        hashed_bytes.append(len(data))
        return original(data, **kwargs)

    monkeypatch.setattr(support.hashlib, "sha256", counted)
    checker, objects = support_fixture(checker_type)
    assert len(hashed_bytes) == 2
    expected = {name: original(vertices.tobytes()).hexdigest()
                for name, vertices in checker._local_vertices.items()}
    monkeypatch.setattr(AttachedObjectCollisionChecker, "refresh", lambda _: {})
    for _ in range(100):
        assert checker.refresh()["support_geometry_fingerprints"] == expected
    assert len(hashed_bytes) == 2
    checker._support_pairs = {}
    checker._register_initial_support(objects)
    assert len(hashed_bytes) == 4
    assert checker.refresh()["support_geometry_fingerprints"] == expected


@pytest.mark.parametrize("checker_type", [support.NumericalTableSupportCollisionChecker,
                                          PillTableSupportCollisionChecker])
def test_exported_static_evidence_is_independent_of_registration_and_other_snapshots(monkeypatch, checker_type):
    checker, _ = support_fixture(checker_type)
    monkeypatch.setattr(AttachedObjectCollisionChecker, "refresh", lambda _: {})
    snapshot = checker.refresh()
    expected = copy.deepcopy(snapshot)
    snapshot["initial_support_geometry"]["piece0"]["scaled_shape_local_vertices"][0][0] = 99.
    snapshot["initial_support_geometry"]["piece0"]["shape_local_T"][0][3] = 99.
    snapshot["initial_object_support_receipts"][0]["raw_contacts"][0]["position"][0] = 99.
    snapshot["support_geometry_fingerprints"]["piece0"] = "changed"
    assert checker.refresh() == expected
    exported = checker.support_initial_geometry()
    exported["piece0"]["scaled_shape_local_vertices"].clear()
    assert checker.support_initial_geometry() == expected["initial_support_geometry"]


def test_legacy_registration_has_empty_static_geometry(monkeypatch):
    checker = support.NumericalTableSupportCollisionChecker()
    checker._target_names = frozenset()
    checker._support_pairs = {}
    checker._register_initial_support({})
    monkeypatch.setattr(AttachedObjectCollisionChecker, "refresh", lambda _: {})
    snapshot = checker.refresh()
    assert snapshot["support_geometry_fingerprints"] == {}
    assert snapshot["initial_support_geometry"] == {}
    assert snapshot["support_semantics"] == "legacy_missing_table"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
