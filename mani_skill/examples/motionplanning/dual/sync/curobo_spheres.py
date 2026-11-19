"""Conservative sphere covers for finite convex solids, independent of simulation.

Vertices are [M,3] XYZ in metres in any one rigid frame. The coarse cover
returns [N,3] cell centers and one radius; the tight cover returns independent
[N,4] XYZ/radius arrays plus coverage metadata. The tight LP kernel bounds
every nonempty solid/cell intersection, including interior points, and caches
only immutable values. No scene, payload, robot, or GPU state is imported.
"""
from __future__ import annotations

from functools import lru_cache
import json
import math
from typing import Any

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

from ._arrays import as_float_array as _array

TIGHT_BOUND_PADDING_M = 1e-8


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
    """Check full-volume coverage and cache ownership without simulation."""
    import itertools
    cube = np.asarray(list(itertools.product([-.01, .01], repeat=3)))
    values, proof = tight_convex_spheres(cube, .025)
    points = np.asarray(list(itertools.product(np.linspace(-.01, .01, 5), repeat=3)))
    gap = np.min(np.linalg.norm(points[:, None] - values[None, :, :3], axis=2)
                 - values[None, :, 3], axis=1)
    assert np.max(gap) <= 1e-8 and proof["complete_solid_coverage"]
    again, _ = tight_convex_spheres(cube, .025)
    again[:] = np.nan
    unmodified, _ = tight_convex_spheres(cube, .025)
    assert np.array_equal(values, unmodified)
    return {"status": "passed", "vertices_shape": list(cube.shape),
            "spheres_shape": list(values.shape), "volume_sample_count": len(points),
            "environment_steps": 0}


if __name__ == "__main__":
    print(json.dumps(self_test(), allow_nan=False))
