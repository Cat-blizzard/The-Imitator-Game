"""Independent experimental surface statistic, never an asset admission writer.

Preserves the original 3 mm threshold. Native face halfspaces with exact triangle-winding fallback replace random
ray containment; the retained 0.1mm outgoing probe defines external samples.
Every retained point receives its native component area divided by400.
Tiny gaps and shared boundary ownership are deliberately not redefined. This is a finite-sample area estimate,
not exact Boolean-union area, and does not prove local grasp contact fidelity.
"""
from contextlib import contextmanager
import hashlib
import time

import numpy as np
import trimesh
from resimpair_grasp.common import read, resolve, binding

VERSION = 'independent_native_surface_candidate_v5'
RULE = {'threshold_m': .003, 'forward_samples': 1800, 'forward_seed': 71,
        'samples_per_native_hull': 400, 'hull_seed_base': 71,
        'legacy_outward_probe_m': .0001, 'native_boundary_decision': 'winding ambiguity retained explicitly',
        'strict_interior_rule': 'actual native closed-surface interior; certified face halfspaces only outside their roundoff band',
        'boundary_points': 'explicit winding ambiguity; retained 0.1mm outward probe only',
        'reverse_quantile': 'weighted empirical CDF inverse at 0.95',
        'forward_quantile': 'retained numpy linear quantile at 0.95',
        'per_external_point_weight': 'native_hull_triangle_area_m2 / 400',
        'native_face_halfspace_convexity_bound': '32*float32_epsilon*max_vertex_norm',
        'nonconvex_native_mesh_fallback': 'deterministic_signed_solid_angle_winding_on_unchanged_native_triangles',
        'exterior_probe_epsilon_m': .0001, 'retained_exterior_definition_unchanged': True,
        'ambiguous_winding_tolerance': 1e-7,
        'no_QHull_geometry_substitution': True,
        'surface_measure': 'retained exposed native-component surfaces; not exact Boolean union area',
        'overlapping_exterior_patches_may_be_counted_more_than_once': True,
        'physical_grasp_quality': 'unknown', 'writes_asset_qualification': False}


def _hash(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


class NativeSolid:
    """Native triangle geometry with a certified halfspace fast path.

    Some PhysX-exported triangulations have locally nonconvex small triangles.
    They must not be replaced by a QHull envelope. Winding uses the exact same
    triangles; only finite, outward-oriented watertight inputs are supported.
    """
    def __init__(self, part):
        vertices = np.asarray(part.vertices, float)
        triangles = np.asarray(part.triangles, float)
        if (not np.isfinite(vertices).all() or not np.isfinite(triangles).all()
                or not part.is_watertight or float(part.volume) <= 0):
            raise ValueError('native_triangle_solid_not_finite_outward_watertight')
        normals = np.asarray(part.face_normals, float)
        good = np.asarray(part.area_faces) > 0
        normals = normals[good]; triangles = triangles[good]
        offset = -np.einsum('ij,ij->i', normals, triangles[:, 0])
        self.planes = np.c_[normals, offset]
        self.triangles = triangles
        self.bounds = np.asarray(part.bounds, float)
        self.roundoff = float(32*np.finfo(np.float32).eps*max(float(np.linalg.norm(vertices, axis=1).max()), 1e-3))
        self.maximum_convexity_violation = float((vertices@normals.T+offset).max())
        self.halfspace_certified = bool(self.maximum_convexity_violation <= self.roundoff)
        self.metadata = {'geometry': 'unchanged_native_vertices_and_triangles',
            'native_triangle_area_m2': float(part.area), 'native_mesh_volume_m3': float(part.volume),
            'maximum_vertex_outside_native_face_plane_m': self.maximum_convexity_violation,
            'floating_point_convexity_bound_m': self.roundoff,
            'halfspace_fast_path_certified': self.halfspace_certified,
            'near_boundary_and_nonconvex_fallback': 'deterministic_solid_angle_winding',
            'QHull_geometry_used': False}

    def contains(self, points):
        points = np.asarray(points, float)
        if not np.isfinite(points).all(): raise ValueError('nonfinite_native_contains_points')
        inside = np.zeros(len(points), bool); ambiguous = np.zeros(len(points), bool)
        indices = np.flatnonzero(np.all(points >= self.bounds[0], axis=1)&np.all(points <= self.bounds[1], axis=1))
        if self.halfspace_certified and len(indices):
            signed = points[indices]@self.planes[:, :3].T+self.planes[:, 3]
            definitely_in = np.all(signed < -self.roundoff, axis=1)
            definitely_out = np.any(signed > self.roundoff, axis=1)
            inside[indices[definitely_in]] = True
            indices = indices[~(definitely_in|definitely_out)]
        for start in range(0, len(indices), 64):
            ids = indices[start:start+64]
            vectors = self.triangles[None, :, :, :]-points[ids, None, None, :]
            a, b, c = vectors[:, :, 0], vectors[:, :, 1], vectors[:, :, 2]
            na, nb, nc = np.linalg.norm(a, axis=2), np.linalg.norm(b, axis=2), np.linalg.norm(c, axis=2)
            numerator = np.einsum('ijk,ijk->ij', a, np.cross(b, c))
            denominator = na*nb*nc+np.einsum('ijk,ijk->ij',a,b)*nc+np.einsum('ijk,ijk->ij',b,c)*na+np.einsum('ijk,ijk->ij',c,a)*nb
            winding = np.abs(np.sum(2*np.arctan2(numerator, denominator), axis=1)/(4*np.pi))
            inside[ids] = winding > .5
            ambiguous[ids] = np.minimum(np.abs(winding), np.abs(winding-1)) > RULE['ambiguous_winding_tolerance']
        return inside, ambiguous


def native_halfspaces(part):
    # Historical function name retained for the analytic fixture interface;
    # this v4 object never substitutes a different convex hull geometry.
    return NativeSolid(part)


def contains_halfspace(points, solid):
    return solid.contains(points)[0]


def external_surface(points, normals, owner, all_equations):
    """Retain the original 0.1 mm outward-probe definition exactly.

    This deliberately does not redefine tiny gaps or shared patch ownership.
    It estimates exposed native-component area, not exact Boolean union area.
    """
    outward = np.asarray(points, float) + np.asarray(normals, float)*RULE['legacy_outward_probe_m']
    keep = np.ones(len(outward), bool); ambiguity = np.zeros(len(outward), bool)
    for j, solid in enumerate(all_equations):
        if j == owner: continue
        inside, uncertain = solid.contains(outward)
        keep &= ~inside; ambiguity |= uncertain
    return keep, np.ones(len(outward), int), ambiguity


def inverse_weighted_cdf(values, weights, quantile=.95):
    values = np.asarray(values, float); weights = np.asarray(weights, float)
    if (values.ndim != 1 or values.shape != weights.shape or not len(values)
            or not np.isfinite(values).all() or not np.isfinite(weights).all()
            or np.any(weights <= 0) or not 0 < quantile <= 1):
        raise ValueError('invalid_weighted_distribution')
    order = np.argsort(values, kind='stable')
    cdf = np.cumsum(weights[order]); index = int(np.searchsorted(cdf, quantile*cdf[-1], side='left'))
    return float(values[order[min(index, len(order)-1)]])


@contextmanager
def _legacy_rng(seed):
    before = np.random.get_state(); np.random.seed(seed)
    try: yield
    finally: np.random.set_state(before)


def evaluate(descriptor_path, *, legacy_ray_seed):
    """Re-evaluate real stored geometry; no cooking, simulation, or mutation."""
    start = time.monotonic(); d = read(descriptor_path)
    parts = [trimesh.load(resolve(ref), force='mesh', process=True) for ref in d['collisions']]
    visual = trimesh.load(resolve(d['visual']), force='scene', process=False).to_geometry()
    equations = [native_halfspaces(part) for part in parts]
    forward_points, _ = trimesh.sample.sample_surface(visual, RULE['forward_samples'], seed=RULE['forward_seed'])
    forward = np.min([trimesh.proximity.closest_point(p, forward_points)[1] for p in parts], axis=0)
    forward95 = float(np.quantile(forward, .95))
    old_points = []; probe_points = []; new_points = []; new_weights = []; hull_rows = []; masks = []
    with _legacy_rng(legacy_ray_seed):
        for i, part in enumerate(parts):
            pts, faces = trimesh.sample.sample_surface(part, RULE['samples_per_native_hull'], seed=RULE['hull_seed_base']+i)
            normals = part.face_normals[faces]
            outward = pts + normals * RULE['legacy_outward_probe_m']
            old_keep = np.ones(len(pts), bool); probe_keep = np.ones(len(pts), bool)
            for j, other in enumerate(parts):
                if i != j:
                    old_keep &= ~other.contains(outward)
                    probe_keep &= ~contains_halfspace(outward, equations[j])
            new_keep, multiplicity, tangent = external_surface(pts, normals, i, equations)
            old_points.extend(pts[old_keep]); new_points.extend(pts[new_keep])
            probe_points.extend(pts[probe_keep])
            weights = float(part.area)/RULE['samples_per_native_hull']/multiplicity
            new_weights.extend(weights[new_keep])
            masks.append(new_keep)
            hull_rows.append({'hull': i, 'native_triangle_area_m2': float(part.area), 'native_geometry_interface': equations[i].metadata,
                'legacy_exterior_samples': int(old_keep.sum()), 'deterministic_exterior_samples': int(new_keep.sum()),
                'deterministic_finite_probe_samples': int(probe_keep.sum()),
                'ray_vs_halfspace_finite_probe_differences': int(np.count_nonzero(old_keep != probe_keep)),
                'finite_vs_infinitesimal_probe_differences': int(np.count_nonzero(probe_keep != new_keep)),
                'shared_boundary_samples': int(np.count_nonzero(new_keep & (multiplicity > 1))),
                'tangent_boundary_samples': int(np.count_nonzero(new_keep & tangent)),
                'estimated_external_area_m2': float(weights[new_keep].sum())})
    if not old_points or not new_points: raise ValueError('no_external_surface_samples')
    old_points = np.asarray(old_points); new_points = np.asarray(new_points); new_weights = np.asarray(new_weights)
    old_reverse = trimesh.proximity.closest_point(visual, old_points)[1]
    probe_reverse = trimesh.proximity.closest_point(visual, np.asarray(probe_points))[1]
    reverse = trimesh.proximity.closest_point(visual, new_points)[1]
    legacy95 = float(np.quantile(old_reverse, .95)); raw95 = float(np.quantile(reverse, .95))
    equal_ecdf = inverse_weighted_cdf(reverse, np.ones(len(reverse)))
    weighted95 = inverse_weighted_cdf(reverse, new_weights)
    return {'schema': VERSION, 'descriptor': binding(descriptor_path), 'source_id': d['source_id'],
        'source_name': d['name'], 'source_partition': d['partition'], 'dimensions_m': d['dimensions_m'],
        'hull_count': len(parts), 'stored_surface': d['surface_comparison'],
        'stored_qualification_unchanged': d['qualification'], 'legacy_ray_seed_diagnostic': legacy_ray_seed,
        'forward_p95_m': forward95, 'retained_ray_equal_hull_linear_p95_m': legacy95,
        'deterministic_finite_probe_equal_hull_linear_p95_m': float(np.quantile(probe_reverse, .95)),
        'deterministic_contains_equal_hull_linear_p95_m': raw95,
        'deterministic_contains_equal_hull_ECDF_p95_m': equal_ecdf,
        'candidate_area_weighted_ECDF_p95_m': weighted95,
        'candidate_surface_decision_only': max(forward95, weighted95) <= RULE['threshold_m'],
        'retained_surface_recomputed_decision': max(forward95, legacy95) <= RULE['threshold_m'],
        'diagnostic_hypothetical_area_weighted_admission': False,
        'estimated_external_area_m2': float(new_weights.sum()),
        'external_area_over_3mm_fraction': float(new_weights[reverse > .003].sum()/new_weights.sum()),
        'reverse_max_error_m': float(reverse.max()), 'hulls': hull_rows,
        'deterministic_evidence_hashes': {'forward_points': _hash(forward_points), 'forward_distances': _hash(forward),
            'native_halfspaces': _hash(np.concatenate([x.planes for x in equations])),
            'native_triangles': _hash(np.concatenate([x.triangles for x in equations])), 'exterior_masks': _hash(np.asarray(masks)),
            'external_points': _hash(new_points), 'external_weights': _hash(new_weights),
            'external_distances': _hash(reverse)},
        'candidate_additional_cost_seconds_including_legacy_comparison': time.monotonic()-start,
        'native_geometry_ambiguity_samples': sum(h['tangent_boundary_samples'] for h in hull_rows),
        'surface_decision_certified_without_geometry_ambiguity': bool(max(forward95, weighted95) <= RULE['threshold_m'] and not any(h['tangent_boundary_samples'] for h in hull_rows)),
        'physical_quality': 'unknown_not_tested', 'asset_qualification_updated': False,
        'new_GPU_jobs': 0, 'new_physical_executions': 0, 'API_calls': 0}
