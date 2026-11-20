"""Original-surface geometry and one-way release-contact separation state.

The triangle/AABB score is a projection overlap measure, not penetration
depth. Positive values prove separation; a nonpositive value denotes an
intersection or containment. Each call represents one closed, consistently
oriented target mesh in one actual finger Box's local coordinates. The scene
adapter validates mesh topology once and combines its actual shapes.

This module has no simulator, FCL, or planning-backend dependency.
"""
from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any

import numpy as np

# Same numeric band as table_support_collision.NUMERICAL_TOP_TOLERANCE_M.
# Keep this pure module independent of that module's PhysX/FCL imports.
NUMERICAL_TOP_TOLERANCE_M = 1e-6


def triangle_box_gap_m(half_size_m: Any, triangles_box_frame: Any) -> float:
    """Return an original-triangle/AABB separation score in metres.

    Args:
        half_size_m: Three positive Box half sizes, shape ``(3,)``.
        triangles_box_frame: One closed target surface, shape ``(N,3,3)``;
            all vertices are expressed in the Box's local frame.

    All calculations use float64. Thirteen SAT axes per triangle cover Box
    face axes, its normal, and all edge/Box-axis cross products. If no surface
    triangle intersects, winding and vertex containment distinguish disjoint
    solids from containment. No ConvexHull replaces the original surface.
    """
    half = np.asarray(half_size_m, dtype=np.float64)
    triangles = np.asarray(triangles_box_frame, dtype=np.float64)
    if (half.shape != (3,) or np.any(half <= 0) or not np.isfinite(half).all()
            or triangles.ndim != 3 or triangles.shape[1:] != (3, 3)
            or not len(triangles) or not np.isfinite(triangles).all()):
        raise ValueError('Expected positive Box half sizes[3] and finite target triangles[N,3,3]')
    edges = np.roll(triangles, -1, axis=1)-triangles
    box_axes = np.broadcast_to(np.eye(3), (len(triangles), 3, 3))
    normals = np.cross(edges[:, 0], edges[:, 1])[:, None, :]
    cross_axes = np.cross(edges[:, :, None, :], np.eye(3)[None, None, :, :]).reshape(-1, 9, 3)
    axes = np.concatenate((box_axes, normals, cross_axes), axis=1)
    lengths = np.linalg.norm(axes, axis=2)
    valid = lengths > 0
    axes = np.divide(axes, lengths[:, :, None], out=np.zeros_like(axes), where=valid[:, :, None])
    projections = np.einsum('nvc,nac->nav', triangles, axes)
    radii = np.abs(axes) @ half
    gaps = np.maximum(projections.min(axis=2)-radii, -radii-projections.max(axis=2))
    gaps[~valid] = -np.inf
    score = float(gaps.max(axis=1).min())
    if score <= 0:
        return score
    # The boundary is disjoint. A vertex inside the Box proves target-in-Box
    # containment, while the winding at Box origin catches Box-in-target.
    if np.any(np.all(np.abs(triangles) <= half, axis=2)):
        return -score
    lengths_vertex = np.linalg.norm(triangles, axis=2)
    first, second, third = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    denominator = lengths_vertex.prod(axis=1)
    denominator += np.einsum('ij,ij->i', first, second)*lengths_vertex[:, 2]
    denominator += np.einsum('ij,ij->i', second, third)*lengths_vertex[:, 0]
    denominator += np.einsum('ij,ij->i', third, first)*lengths_vertex[:, 1]
    numerator = np.einsum('ij,ij->i', first, np.cross(second, third))
    winding = float((2*np.arctan2(numerator, denominator)).sum()/(4*np.pi))
    return -score if abs(winding) > .5 else score


def _finite_gaps(values: Mapping[Hashable, float]) -> dict[Hashable, float]:
    result = {key: float(value) for key, value in values.items()}
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError('Contact separation scores must be finite')
    return result


class SeparationState:
    """Admit only initial contacts improving toward permanent separation.

    ``check`` updates the best score and latches only when all supplied pairs
    pass. Candidate scans use ``clone`` so they cannot consume measured state.
    The existing one-micrometre band permits numeric score variation without
    allowing accumulated deepening below the best value previously observed.
    """

    def __init__(self, initial_gaps: Mapping[Hashable, float], *,
                 tolerance_m: float = NUMERICAL_TOP_TOLERANCE_M) -> None:
        self.initial_gaps = _finite_gaps(initial_gaps)
        self.best_gaps = self.initial_gaps.copy()
        self.tolerance_m = float(tolerance_m)
        if not np.isfinite(self.tolerance_m) or self.tolerance_m < 0:
            raise ValueError('Separation numeric tolerance must be finite and nonnegative')
        self._active = {key for key, value in self.initial_gaps.items() if value <= 0}

    @property
    def active_pairs(self) -> frozenset[Hashable]:
        return frozenset(self._active)

    @property
    def all_clear(self) -> bool:
        return not self._active

    def clone(self) -> SeparationState:
        result = SeparationState(self.initial_gaps, tolerance_m=self.tolerance_m)
        result.best_gaps = self.best_gaps.copy()
        result._active = self._active.copy()
        return result

    def check(self, gaps: Mapping[Hashable, float]) -> list[dict[str, Any]]:
        current = _finite_gaps(gaps)
        if current.keys() != self.initial_gaps.keys():
            raise ValueError('A separation observation must retain the same physical pair identities')
        rejected = []
        for pair, value in current.items():
            reason = None
            if pair not in self._active and value <= 0:
                reason = ('release_contact_reentry' if self.initial_gaps[pair] <= 0
                          else 'release_contact_new_collision')
            elif pair in self._active and value < self.best_gaps[pair]-self.tolerance_m:
                reason = 'release_contact_deepened'
            if reason is not None:
                rejected.append({'reason': reason, 'pair': pair, 'gap_m': value,
                    'best_gap_m': self.best_gaps[pair], 'numeric_tolerance_m': self.tolerance_m})
        if not rejected:
            for pair, value in current.items():
                self.best_gaps[pair] = max(value, self.best_gaps[pair])
                if value > 0:
                    self._active.discard(pair)
        return rejected

    def finish(self) -> list[dict[str, Any]]:
        return [{'reason': 'release_contact_not_separated', 'pair': pair,
                 'best_gap_m': self.best_gaps[pair]} for pair in self.initial_gaps if pair in self._active]


def _self_test() -> None:
    vertices = np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
    faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
        [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    triangles = vertices[faces]
    assert abs(triangle_box_gap_m(np.ones(3), triangles+[3., 0, 0])-1) < 1e-14
    assert triangle_box_gap_m(np.ones(3), triangles+[1.5, 0, 0]) < 0
    assert triangle_box_gap_m(np.ones(3)*.25, triangles) < 0  # Box within target.
    assert triangle_box_gap_m(np.ones(3)*2, triangles) < 0  # Target within Box.
    state = SeparationState({'left': -2e-4, 'right': 1e-3})
    candidate = state.clone()
    assert not candidate.check({'left': 1e-5, 'right': 1e-3}) and candidate.all_clear
    assert not state.all_clear and state.finish()
    assert candidate.check({'left': 0., 'right': 1e-3})[0]['reason'] == 'release_contact_reentry'
    assert state.check({'left': -3e-4, 'right': 1e-3})[0]['reason'] == 'release_contact_deepened'
    assert state.check({'left': -2e-4, 'right': -1e-5})[0]['reason'] == 'release_contact_new_collision'
    assert not state.check({'left': -1e-4, 'right': 1e-3})
    assert not state.check({'left': 1e-5, 'right': 1e-3}) and not state.finish()
    print('release_contact CPU self-test passed: original triangles, SAT, containment, one-way pair state.')


if __name__ == '__main__':
    _self_test()
