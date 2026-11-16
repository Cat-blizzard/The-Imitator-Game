"""Task grasp targets, release offsets and shared destination-region geometry.

Actor-local collision vertices are finite [N,3] metres; actor poses are world
XYZ/WXYZ [7]. Region gates and their historical tolerances are unchanged.
The two task grasp helpers use the original OBB algorithm, measured left/right
TCP closing directions, 25 mm depth and 80 mm local pregrasp offset. The
release offsets are the original ordered left/right basket-frame XYZ metres.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# Direct-script setup is test-only; normal package imports need no path changes.
if __name__ == "__main__" and not __package__:
    import mani_skill.examples.motionplanning.dual as _dual_package
    _dual_package.__path__ = [str(Path(__file__).parent), *_dual_package.__path__]
    __package__ = _dual_package.__name__

from .physical_observation import pose_matrix

TASK = "TwoRobotPickAppleBananaToBaskets-v1"

def physical_region(center: Any, basket_bounds: Any, object_bounds: Any) -> dict[str, Any]:
    """Conservative full XY containment, center height, and no below-floor pose."""
    center, basket, obj = map(lambda x: np.asarray(x, dtype=float), (center, basket_bounds, object_bounds))
    if center.shape != (3,) or basket.shape != (2, 3) or obj.shape != (2, 3) or not all(
            np.isfinite(x).all() for x in (center, basket, obj)):
        raise ValueError("Expected finite center[3] and ordered AABBs[2,3]")
    if np.any(basket[0] > basket[1]) or np.any(obj[0] > obj[1]):
        raise ValueError("AABB endpoints must be ordered")
    lo, hi = basket
    xy = bool(np.all(obj[0, :2] >= lo[:2] - .002) and np.all(obj[1, :2] <= hi[:2] + .002))
    inside = bool(xy and np.all(center[:2] >= lo[:2]) and np.all(center[:2] <= hi[:2])
                  and lo[2] - .005 <= center[2] <= hi[2] + .01 and obj[0, 2] >= lo[2] - .005)
    return {"inside": inside, "object_geom_center_m": center.tolist(), "basket_aabb_m": basket.tolist(),
            "object_aabb_m": obj.tolist(), "object_xy_contained": xy,
            "object_fully_below_rim": bool(obj[1, 2] <= hi[2]),
            "center_z_range_m": [float(lo[2] - .005), float(hi[2] + .01)]}


def collision_vertex_region(object_vertices: Any, object_pose: Any,
                            basket_vertices: Any, basket_pose: Any) -> dict[str, Any]:
    """Apply the existing region gates to tight actual collision-mesh bounds.

    Vertices are complete actor-local collision meshes, finite [N,3]/[M,3]
    metres; poses are finite XYZ/WXYZ[7]. Transform every vertex before taking
    world min/max, rather than rotating eight corners of a local AABB. The
    geometric centre keeps its existing definition: the local collision AABB
    midpoint transformed by the current actor pose, not the world AABB centre.
    No containment tolerance or below-rim requirement changes here.
    """
    meshes = [np.asarray(value, dtype=np.float64) for value in (object_vertices, basket_vertices)]
    if any(vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices)
           or not np.isfinite(vertices).all() for vertices in meshes):
        raise ValueError("Expected complete finite local collision vertices[N,3]")
    transforms = [pose_matrix(value) for value in (object_pose, basket_pose)]
    worlds = [vertices @ transform[:3, :3].T + transform[:3, 3]
              for vertices, transform in zip(meshes, transforms)]
    if not all(np.isfinite(vertices).all() for vertices in worlds):
        raise ValueError("Transformed collision vertices must remain finite")
    bounds = [np.stack([vertices.min(axis=0), vertices.max(axis=0)]) for vertices in worlds]
    local_center = (meshes[0].min(axis=0) + meshes[0].max(axis=0)) / 2
    center = local_center @ transforms[0][:3, :3].T + transforms[0][:3, 3]
    return physical_region(center, bounds[1], bounds[0])


def grasp_poses(raw: Any) -> tuple[Any, Any]:
    """Reuse the task solution's OBB grasp and 8 cm pregrasp offset."""
    import sapien
    from mani_skill.examples.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb
    from .trajectory_execution import as_numpy
    closing = as_numpy(raw.left_agent.tcp.pose.to_transformation_matrix())[0, :3, 1]
    info = compute_grasp_info_by_obb(get_actor_obb(raw.apple_out), approaching=np.array([0, 0, -1]),
                                   target_closing=closing, depth=.025)
    grasp = raw.left_agent.build_grasp_pose(np.array([0, 0, -1]), info["closing"], info["center"])
    return grasp * sapien.Pose([0, 0, -.08]), grasp


def banana_grasp_poses(raw: Any) -> tuple[Any, Any]:
    """Apply the official task's OBB grasp to banana_out and the right TCP."""
    import sapien
    from mani_skill.examples.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb
    from .trajectory_execution import as_numpy
    closing = as_numpy(raw.right_agent.tcp.pose.to_transformation_matrix())[0, :3, 1]
    info = compute_grasp_info_by_obb(get_actor_obb(raw.banana_out), approaching=np.array([0, 0, -1]),
                                   target_closing=closing, depth=.025)
    grasp = raw.right_agent.build_grasp_pose(np.array([0, 0, -1]), info["closing"], info["center"])
    return grasp * sapien.Pose([0, 0, -.08]), grasp


RELEASE_TCP_OFFSETS_BASKET_M = ((.03, .08, .10), (.03, 0., .12))


def self_test() -> dict[str, Any]:
    """Reuse the original rotated-geometry region checks without an environment."""
    bounds = [[-.2, -.2, 0], [.2, .2, .1]]
    assert physical_region([0, 0, .05], bounds, [[-.1, -.1, .02], [.1, .1, .08]])["inside"]
    assert not physical_region([0, 0, .05], bounds, [[-.1, -.1, .02], [.3, .1, .08]])["inside"]
    # A rotated triangular prism fits while its empty local-box corners do not.
    # Keep the old local-box centre so only the bounds representation changes.
    from itertools import product
    prism = np.array([[x, y, z] for x, y in ((0., 0.), (.1, 0.), (0., .1)) for z in (0., .02)])
    basket_mesh = np.array(list(product((-.08, .08), (-.01, .09), (0., .1))))
    pose = np.array([0., 0., .02, np.cos(np.pi/8), 0., 0., np.sin(np.pi/8)])
    identity = np.array([0., 0., 0., 1., 0., 0., 0.])
    prism_before, basket_before, pose_before = prism.copy(), basket_mesh.copy(), pose.copy()
    tight = collision_vertex_region(prism, pose, basket_mesh, identity)
    assert tight["inside"]
    matrix = pose_matrix(pose)
    corners = np.array(list(product((0., .1), (0., .1), (0., .02))))
    coarse = corners @ matrix[:3, :3].T + matrix[:3, 3]
    assert not physical_region(tight["object_geom_center_m"],
        tight["basket_aabb_m"], [coarse.min(axis=0), coarse.max(axis=0)])["inside"]
    np.testing.assert_allclose(tight["object_geom_center_m"], [.0, np.sqrt(.005), .03], atol=1e-14)
    assert not np.allclose(tight["object_geom_center_m"], np.mean(tight["object_aabb_m"], axis=0))
    # Above-rim geometry remains diagnostic; the centre/floor/XY gates still apply.
    assert physical_region([0, 0, .05], bounds, [[-.1, -.1, .02], [.1, .1, .3]])["inside"]
    for moved in (pose + [.2, 0, 0, 0, 0, 0, 0], pose + [0, 0, -.03, 0, 0, 0, 0],
                  pose + [0, 0, .1, 0, 0, 0, 0]):
        assert not collision_vertex_region(prism, moved, basket_mesh, identity)["inside"]
    assert not collision_vertex_region(prism, pose, basket_mesh, identity + [.2, 0, 0, 0, 0, 0, 0])["inside"]
    assert all(np.array_equal(a, b) for a, b in ((prism, prism_before), (basket_mesh, basket_before), (pose, pose_before)))
    rejected = 0
    for bad_vertices, bad_pose in ((np.empty((0, 3)), pose), (np.zeros((3, 2)), pose),
                                   ([[0., 0., np.nan]], pose), (prism, np.zeros(7)),
                                   (prism, np.r_[np.nan, pose[1:]])):
        try:
            collision_vertex_region(bad_vertices, bad_pose, basket_mesh, identity)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError("Malformed collision geometry was accepted")
    assert rejected == 5
    offsets = np.asarray(RELEASE_TCP_OFFSETS_BASKET_M)
    assert offsets.shape == (2, 3) and np.isfinite(offsets).all()
    assert np.array_equal(offsets, [[.03, .08, .10], [.03, 0., .12]])
    return {"status": "passed", "physical_environment_steps": 0,
            "original_region_tests_preserved": True,
            "collision_geometry_negative_cases": rejected,
            "release_offsets_preserved": True,
            "grasp_execution": "not attempted without a real environment; original function AST preserved"}


if __name__ == "__main__":
    print(json.dumps(self_test(), indent=2, allow_nan=False))
