"""Conservative sphere covers for finite convex solids, independent of simulation.

Vertices are [M,3] XYZ in metres in any one rigid frame. The coarse cover
returns [N,3] cell centers and one radius; the tight cover returns independent
[N,4] XYZ/radius arrays plus coverage metadata. The tight LP kernel bounds
every nonempty solid/cell intersection, including interior points, and caches
only immutable values. No scene, payload, robot, or GPU state is imported.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import itertools
import json
import math
from typing import Any

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

TIGHT_BOUND_PADDING_M = 1e-8


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def coarse_convex_spheres(vertices: Any, pitch_m: float) -> tuple[np.ndarray, float]:
    """Cover the convex solid with grid-cell circumspheres, never just vertices.

    Every cell intersecting the convex solid passes the expanded-halfspace
    predicate. Retained centers include extra cells but cannot omit an occupied
    cell: any point in a cell is at most sqrt(3)*pitch/2 from its center.
    """
    pitch = float(pitch_m)
    if not math.isfinite(pitch) or pitch <= 0:
        raise ValueError("Sphere grid pitch must be positive and finite")
    vertices = _array(vertices)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4 or not np.isfinite(vertices).all():
        raise ValueError("Nondegenerate convex vertices[N>=4,3] required")
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    if np.any(upper <= lower):
        raise ValueError("Sphere coverage requires a three-dimensional solid")
    counts = np.maximum(1, np.ceil((upper - lower) / pitch).astype(np.int64))
    if np.prod(counts.astype(object)) > 2_000_000:
        raise ValueError("Sphere coverage exceeds the explicit 2 million cell budget")
    axes = [lower[i] + (np.arange(int(counts[i])) + .5) * pitch for i in range(3)]
    centers = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    radius = math.sqrt(3) * pitch / 2
    planes = ConvexHull(vertices).equations
    # A circumsphere/halfspace test is a conservative superset of intersecting
    # cells, including at the floating-point boundary.
    limit = radius * np.linalg.norm(planes[:, :3], axis=1) + 1e-12
    keep = np.ones(len(centers), dtype=bool)
    for start in range(0, len(centers), 8192):
        values = centers[start:start + 8192] @ planes[:, :3].T + planes[:, 3]
        keep[start:start + 8192] = np.all(values <= limit, axis=1)
    centers = centers[keep]
    if not len(centers):
        raise ValueError("No collision sphere generated for a nonempty shape")
    assert centers.ndim == 2 and centers.shape[1] == 3 and np.isfinite(centers).all()
    return centers, radius


def _spheres(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if (result.ndim != 2 or result.shape[1] != 4 or not len(result)
            or not np.isfinite(result).all() or np.any(result[:, 3] <= 0)):
        raise ValueError("Payload spheres must be finite [N>=1,4] with positive radii")
    return result


@lru_cache(maxsize=64)
def _tight_convex_cover_cached(vertices_bytes: bytes, count: int, pitch_m: float,
                                positive_z_padding_m: float | None,
                                max_depth: int) -> tuple[tuple[tuple[float, ...], ...], str]:
    """Enclose every nonempty convex-solid/cell intersection by a sphere.

    The coarse grid partitions the full solid's bounding box. The existing
    expanded-halfspace prefilter cannot remove an intersecting cell. Six LPs
    give each intersection's coordinate extrema, whose enclosing AABB and
    circumsphere contain every intersection point. Only proven LP-infeasible
    cells are omitted. Subdivision partitions the ORIGINAL cell in X/Y; using
    its newly fitted sphere or tight AABB for subdivision would lose this
    partition proof. The fixed positive-Z criterion depends on geometry alone.
    LP bounds and radius receive 10 nm outward numerical padding; infeasible
    optimization or an unmet refinement bound fails closed.
    """
    vertices = np.frombuffer(vertices_bytes, dtype=np.float64).reshape(count, 3)
    hull = ConvexHull(vertices)
    equations = hull.equations
    matrix, bound = equations[:, :3], -equations[:, 3]
    centers, _ = coarse_convex_spheres(vertices, pitch_m)
    max_z = float(vertices[:, 2].max())
    spheres: list[tuple[float, ...]] = []
    stats = {"initial_cells": len(centers), "empty_cells": 0, "refined_cells": 0,
             "maximum_depth": 0, "linear_programs": 0}
    options = {"primal_feasibility_tolerance": 1e-9,
               "dual_feasibility_tolerance": 1e-9}

    def cell(lower: np.ndarray, upper: np.ndarray, depth: int) -> None:
        stats["maximum_depth"] = max(stats["maximum_depth"], depth)
        low, high = np.empty(3), np.empty(3)
        bounds = list(zip(lower.tolist(), upper.tolist(), strict=True))
        for axis in range(3):
            objective = np.eye(3)[axis]
            extrema = []
            for sign in (1., -1.):
                result = linprog(sign * objective, A_ub=matrix, b_ub=bound,
                                 bounds=bounds, method="highs", options=options)
                stats["linear_programs"] += 1
                if result.status == 2:
                    stats["empty_cells"] += 1
                    return
                if not result.success or not math.isfinite(float(result.fun)):
                    raise ValueError(f"Convex cell LP failed: status={result.status}")
                extrema.append(float(result.fun))
            low[axis], high[axis] = extrema[0], -extrema[1]
        if np.any(low > high + 2e-9):
            raise ValueError("Inconsistent convex-cell coordinate extrema")
        low -= TIGHT_BOUND_PADDING_M
        high += TIGHT_BOUND_PADDING_M
        center = (low + high) / 2.
        radius = float(np.linalg.norm((high - low) / 2.)) + TIGHT_BOUND_PADDING_M
        assert center.shape == (3,) and np.isfinite(center).all() and radius > 0
        if (positive_z_padding_m is not None
                and center[2] + radius > max_z + positive_z_padding_m + 1e-12):
            if depth >= max_depth:
                raise ValueError("Positive-Z conservative cover needs more than the fixed refinement depth")
            stats["refined_cells"] += 1
            midpoint = (lower + upper) / 2.
            for x_half in (0, 1):
                for y_half in (0, 1):
                    child_low, child_high = lower.copy(), upper.copy()
                    for axis, half in ((0, x_half), (1, y_half)):
                        if half == 0:
                            child_high[axis] = midpoint[axis]
                        else:
                            child_low[axis] = midpoint[axis]
                    cell(child_low, child_high, depth + 1)
        else:
            spheres.append((*center.tolist(), radius))

    for center in centers:
        cell(center - pitch_m / 2., center + pitch_m / 2., 0)
    array = _spheres(spheres)
    source_gap = float(np.max(np.min(
        np.linalg.norm(vertices[:, None] - array[None, :, :3], axis=2) - array[None, :, 3], axis=1)))
    if source_gap > 1e-8:
        raise ValueError("Tight sphere model failed the necessary source-vertex coverage check")
    proof = {"coverage_policy": "full_convex_solid_cell_intersection_six_LP_tight_AABB_circumspheres_v1",
             "sphere_count": len(spheres), "sphere_pitch_m": pitch_m,
             "linear_program_bound_padding_m": TIGHT_BOUND_PADDING_M,
             "radius_outward_padding_m": TIGHT_BOUND_PADDING_M,
             "positive_z_padding_m": positive_z_padding_m,
             "maximum_allowed_refinement_depth": max_depth, **stats,
             "maximum_source_vertex_uncovered_gap_m": source_gap,
             "source_solid_max_positive_z_m": max_z,
             "maximum_sphere_positive_z_extent_m": float(np.max(array[:, 2] + array[:, 3])),
             "complete_solid_coverage": True,
             "proof": "Grid covers the solid bounding box; expanded-halfspace prefilter preserves every intersecting cell; six LP extrema bound the entire convex-solid/cell intersection; circumsphere encloses its AABB; recursive children partition the original parent cell in XY; only LP-infeasible cells are removed.",
             "numerical_scope": "HiGHS primal/dual feasibility tolerance 1nm; LP extrema and sphere radius padded outward by10nm; numerical failure or unmet fixed envelope rejects export."}
    return tuple(spheres), json.dumps(proof, sort_keys=True, allow_nan=False)


def tight_convex_spheres(vertices: Any, pitch_m: float = .025, *,
                         positive_z_padding_m: float | None = None,
                         max_depth: int = 2) -> tuple[np.ndarray, dict[str, Any]]:
    """Return conservative [N,4] spheres for finite full-dimensional vertices[M,3]."""
    array = np.ascontiguousarray(vertices, dtype=np.float64)
    pitch = float(pitch_m)
    if (array.ndim != 2 or array.shape[1] != 3 or len(array) < 4
            or not np.isfinite(array).all() or np.any(np.ptp(array, axis=0) <= 0)
            or not math.isfinite(pitch) or pitch <= 0):
        raise ValueError("A finite nondegenerate solid and positive grid pitch are required")
    if (type(max_depth) is not int or not 0 <= max_depth <= 8
            or (positive_z_padding_m is not None
                and (not math.isfinite(positive_z_padding_m) or positive_z_padding_m <= 0))):
        raise ValueError("Invalid fixed refinement envelope or maximum depth")
    ConvexHull(array)  # Reject coplanar/degenerate inputs before caching.
    values, proof = _tight_convex_cover_cached(array.tobytes(), len(array), pitch,
        None if positive_z_padding_m is None else float(positive_z_padding_m), max_depth)
    return _spheres(values).copy(), json.loads(proof)


def self_test() -> dict[str, Any]:
    """CPU-only solid coverage, oblique cells, cache ownership and input checks."""
    rejected = 0

    def reject(function: Any) -> None:
        nonlocal rejected
        try:
            function()
        except (ValueError, RuntimeError, TypeError):
            rejected += 1
        else:
            raise AssertionError("Malformed convex-solid input was accepted")

    def maximum_gap(points: np.ndarray, spheres: np.ndarray) -> float:
        assert points.ndim == 2 and points.shape[1] == 3
        assert spheres.ndim == 2 and spheres.shape[1] == 4
        return float(np.max(np.min(
            np.linalg.norm(points[:, None] - spheres[None, :, :3], axis=2)
            - spheres[None, :, 3], axis=1)))

    cube = np.asarray(list(itertools.product([-.03, .03], repeat=3)))
    cube_points = np.asarray(list(itertools.product(np.linspace(-.03, .03, 13), repeat=3)))
    coarse_centers, coarse_radius = coarse_convex_spheres(cube, .025)
    coarse = np.column_stack((coarse_centers, np.full(len(coarse_centers), coarse_radius)))
    tight, cube_proof = tight_convex_spheres(cube, .025)
    coarse_gap = maximum_gap(cube_points, coarse)
    cube_gap = maximum_gap(cube_points, tight)
    assert coarse_gap <= 1e-12 and cube_gap <= 1e-12
    assert cube_proof["complete_solid_coverage"]

    # Rotate and shear a full solid: interior samples exercise more than hull vertices.
    basis, _ = np.linalg.qr(np.array([[1., .4, -.2], [.3, 1., .2], [.2, -.3, 1.]]))
    shear = np.array([[1., .35, 0.], [0., .7, .25], [0., 0., 1.2]])
    transform = basis @ shear
    oblique = cube @ transform.T + [.01, -.02, .03]
    oblique_points = cube_points @ transform.T + [.01, -.02, .03]
    oblique_cover, oblique_proof = tight_convex_spheres(oblique, .025)
    oblique_gap = maximum_gap(oblique_points, oblique_cover)
    assert oblique_gap <= 1e-12 and oblique_proof["complete_solid_coverage"]

    tetra = np.array([[0., 0., 0.], [.08, 0., 0.], [0., .08, 0.], [0., 0., .08]]) @ basis.T
    tetra_points = np.vstack((tetra, np.random.default_rng(406).dirichlet(np.ones(4), 2000) @ tetra))
    tetra_cover, tetra_proof = tight_convex_spheres(tetra, .025)
    tetra_gap = maximum_gap(tetra_points, tetra_cover)
    assert tetra_gap <= 1e-12 and tetra_proof["empty_cells"] > 0

    # The cached tuple/JSON are immutable; every caller owns its output array and dict.
    before = _tight_convex_cover_cached.cache_info()
    again, proof_again = tight_convex_spheres(oblique, .025)
    assert _tight_convex_cover_cached.cache_info().hits == before.hits + 1
    assert np.array_equal(again, oblique_cover) and proof_again == oblique_proof
    again[:] = np.nan
    proof_again["complete_solid_coverage"] = False
    unmodified, unmodified_proof = tight_convex_spheres(oblique, .025)
    assert np.array_equal(unmodified, oblique_cover) and unmodified_proof == oblique_proof

    envelope_cube = cube + [0., 0., .03]
    envelope, envelope_proof = tight_convex_spheres(
        envelope_cube, .025, positive_z_padding_m=.002)
    assert np.max(envelope[:, 2] + envelope[:, 3]) <= .062 + 1e-12
    assert envelope_proof["refined_cells"] > 0

    reject(lambda: coarse_convex_spheres(cube, 0.))
    reject(lambda: coarse_convex_spheres(cube, float("nan")))
    reject(lambda: coarse_convex_spheres(np.zeros((4, 3)), .025))
    reject(lambda: tight_convex_spheres(np.zeros((4, 2))))
    reject(lambda: tight_convex_spheres(np.full((4, 3), np.nan)))
    reject(lambda: tight_convex_spheres(np.array([[0., 0., 0.], [1., 1., 0.],
                                                [0., 1., 1.], [1., 2., 1.]])))
    reject(lambda: tight_convex_spheres(cube, -1.))
    reject(lambda: tight_convex_spheres(cube, positive_z_padding_m=-.001))
    reject(lambda: tight_convex_spheres(cube, max_depth=True))
    reject(lambda: tight_convex_spheres(envelope_cube, .025,
                                      positive_z_padding_m=.002, max_depth=0))
    assert rejected == 10
    return {"status": "passed", "gpu_initialized": False,
            "input_vertex_shape": list(cube.shape),
            "sphere_shapes": {"cube": list(tight.shape),
                              "oblique": list(oblique_cover.shape),
                              "tetrahedron": list(tetra_cover.shape)},
            "solid_sample_counts": {"cube": len(cube_points),
                                    "oblique": len(oblique_points),
                                    "tetrahedron": len(tetra_points)},
            "maximum_sample_uncovered_gap_m": {"coarse_cube": coarse_gap,
                "tight_cube": cube_gap, "oblique": oblique_gap, "tetrahedron": tetra_gap},
            "lp_empty_cells": tetra_proof["empty_cells"],
            "cache_output_ownership_verified": True,
            "positive_z_envelope_verified": True, "rejected_cases": rejected}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(self_test(), allow_nan=False))
