"""Independent experimental surface statistic, never an asset admission writer.

Preserves the original 3 mm threshold. Native convex halfspaces replace random
ray containment; infinitesimal outgoing directions identify external boundary.
Shared boundary samples divide their native area weight by overlap multiplicity.
The old finite probe is retained as an ablation. This is a finite-sample area estimate,
not exact Boolean-union area, and does not prove local grasp contact fidelity.
"""
from contextlib import contextmanager
import hashlib
import time

import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from resimpair_grasp.common import read, resolve, binding

VERSION = 'independent_native_surface_candidate_v3'
RULE = {'threshold_m': .003, 'forward_samples': 1800, 'forward_seed': 71,
        'samples_per_native_hull': 400, 'hull_seed_base': 71,
        'legacy_outward_probe_m': .0001, 'halfspace_boundary_epsilon_m': 1e-10,
        'outgoing_direction_epsilon': 1e-10,
        'strict_interior_rule': 'all unit-normal halfspace signed distances < -epsilon',
        'boundary_points': 'exclude when outgoing direction enters another hull; aligned shared external face counted once',
        'reverse_quantile': 'weighted empirical CDF inverse at 0.95',
        'forward_quantile': 'retained numpy linear quantile at 0.95',
        'per_external_point_weight': 'native_hull_triangle_area_m2 / 400 / shared_exterior_boundary_multiplicity',
        'native_mesh_vs_vertex_hull_relative_tolerance': 1e-5,
        'native_mesh_vs_vertex_hull_absolute_tolerance': 1e-12,
        'physical_grasp_quality': 'unknown', 'writes_asset_qualification': False}


def _hash(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def native_halfspaces(part):
    vertices = np.asarray(part.vertices, float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError('native_convex_vertices_invalid')
    hull = ConvexHull(vertices)
    if float(part.volume) <= 0:
        raise ValueError('native_convex_faces_not_consistently_outward')
    for actual, expected in ((abs(float(part.volume)), hull.volume), (float(part.area), hull.area)):
        bound = max(RULE['native_mesh_vs_vertex_hull_absolute_tolerance'],
                    RULE['native_mesh_vs_vertex_hull_relative_tolerance'] * expected)
        if abs(actual - expected) > bound:
            raise ValueError('stored_native_collision_mesh_not_equivalent_to_its_convex_hull')
    equations = np.asarray(hull.equations, float)
    equations /= np.linalg.norm(equations[:, :3], axis=1)[:, None]
    return equations


def contains_halfspace(points, equations):
    points = np.asarray(points, float); equations = np.asarray(equations, float)
    if not np.isfinite(points).all() or not np.isfinite(equations).all():
        raise ValueError('nonfinite_halfspace_input')
    return np.all(points @ equations[:, :3].T + equations[:, 3]
                  < -RULE['halfspace_boundary_epsilon_m'], axis=1)


def external_surface(points, normals, owner, all_equations):
    """Convex-union boundary with an infinitesimal outgoing direction.

    Interior faces are excluded without a finite ray step. Coplanar duplicate
    boundary patches divide their sample area by their multiplicity. Tangential
    edge cases (zero area in continuous sampling) are retained and reported.
    """
    points = np.asarray(points, float); normals = np.asarray(normals, float)
    keep = np.ones(len(points), bool); multiplicity = np.ones(len(points), int)
    tangent = np.zeros(len(points), bool)
    epsilon = RULE['halfspace_boundary_epsilon_m']; direction_epsilon = RULE['outgoing_direction_epsilon']
    for j, equations in enumerate(all_equations):
        if j == owner: continue
        signed = points @ equations[:, :3].T + equations[:, 3]
        maximum = signed.max(axis=1)
        inside = maximum < -epsilon
        boundary = (maximum <= epsilon) & ~inside
        active = np.abs(signed) <= epsilon
        derivative = normals @ equations[:, :3].T
        leaves = np.any(active & (derivative > direction_epsilon), axis=1)
        enters = np.any(active & (derivative < -direction_epsilon), axis=1) & ~leaves
        keep &= ~(inside | (boundary & enters))
        shared = boundary & np.any(active & (derivative > 1-direction_epsilon), axis=1)
        multiplicity += shared.astype(int)
        tangent |= boundary & ~leaves & ~enters
    return keep, multiplicity, tangent


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
            hull_rows.append({'hull': i, 'native_triangle_area_m2': float(part.area),
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
            'native_halfspaces': _hash(np.concatenate(equations)), 'exterior_masks': _hash(np.asarray(masks)),
            'external_points': _hash(new_points), 'external_weights': _hash(new_weights),
            'external_distances': _hash(reverse)},
        'candidate_additional_cost_seconds_including_legacy_comparison': time.monotonic()-start,
        'physical_quality': 'unknown_not_tested', 'asset_qualification_updated': False,
        'new_GPU_jobs': 0, 'new_physical_executions': 0, 'API_calls': 0}
