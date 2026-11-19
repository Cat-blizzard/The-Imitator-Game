"""Conservative native-footprint packing on a verified convex floor disk.

This is a geometry proposal, separate from source semantics and physics.
The floor must be one upward, near-horizontal, manifold disk whose actual
triangle projection covers its convex boundary exactly, without overlaps.
No exterior bounding box or recorded floor circle substitutes for that proof.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np
from scipy.spatial import ConvexHull
import trimesh

from resimpair_grasp.common import binding, read, resolve
from resimpair_pickplace.requested_tasks import rotation

VERSION = 'verified_floor_polygon_capacity_v1'
FLOOR_MARGIN_M = .012
PAYLOAD_GAP_M = .025
NUMERICAL_INSET_M = 1e-6
GEOMETRY_EPS_M = 1e-10
AREA_EPS_M2 = 1e-12
MAX_FLOOR_FACES = 4096
MAX_OVERLAP_PAIRS = 250000


def cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def polygon_area(points):
    p = np.asarray(points, float)
    return float(cross2(p, np.roll(p, -1, axis=0)).sum() / 2) if len(p) >= 3 else 0.


def clipped_polygon(subject, clip):
    """Intersection of convex counterclockwise polygons, used for overlaps."""
    points = np.asarray(subject, float)
    for a, b in zip(clip, np.roll(clip, -1, axis=0)):
        if not len(points):
            break
        values = cross2(b - a, points - a)
        output = []
        for i, end in enumerate(points):
            start = points[i - 1]; before, after = values[i - 1], values[i]
            inside_before, inside_after = before >= 0., after >= 0.
            if inside_before != inside_after:
                output.append(start + (end - start) * before / (before - after))
            if inside_after:
                output.append(end)
        points = np.asarray(output, float).reshape((-1, 2))
    return points


def extract_floor(mesh, cavity):
    vertices, faces = np.asarray(mesh.vertices, float), np.asarray(mesh.faces, int)
    center = np.asarray(cavity['center_xy_m'], float)
    z, tolerance = float(cavity['floor_z_m']), float(cavity.get('floor_tolerance_m', .003))
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
            or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or not np.isfinite(vertices).all() or not np.isfinite([z, tolerance, *center]).all()
            or center.shape != (2,) or not 0 < tolerance <= .003):
        raise ValueError('finite_native_floor_geometry_and_retained_height_tolerance_required')
    triangles = vertices[faces]
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    length = np.linalg.norm(normal, axis=1)
    upward = np.divide(normal[:, 2], length, out=np.zeros(len(length)), where=length > 0)
    near = np.abs(triangles[:, :, 2] - z).max(axis=1) <= tolerance
    if np.any(near & (upward > 0) & (upward < .85)):
        raise ValueError('near_floor_contains_steep_upward_triangles')
    selected = near & (upward >= .85)
    if not 1 <= int(selected.sum()) <= MAX_FLOOR_FACES:
        raise ValueError('bounded_upward_floor_triangles_required')
    # Merge only byte-equal coordinates. Never close cracks or repair topology.
    points, inverse = np.unique(triangles[selected].reshape((-1, 3)), axis=0, return_inverse=True)
    floor_faces = inverse.reshape((-1, 3)); count = len(floor_faces)
    if np.ptp(points[:, 2]) > tolerance:
        raise ValueError('floor_height_span_exceeds_declared_tolerance')
    if len(np.unique(np.sort(floor_faces, axis=1), axis=0)) != count:
        raise ValueError('duplicate_floor_triangles')
    oriented = floor_faces[:, [[0, 1], [1, 2], [2, 0]]].reshape((-1, 2))
    edges, edge_ids, counts = np.unique(np.sort(oriented, axis=1), axis=0,
                                       return_inverse=True, return_counts=True)
    if np.any(counts > 2) or len(points) - len(edges) + count != 1:
        raise ValueError('floor_must_be_a_manifold_disk_without_holes')
    uses = defaultdict(list)
    for i, eid in enumerate(edge_ids):
        uses[int(eid)].append(i)
    adjacency = [set() for _ in range(count)]
    for ids in uses.values():
        if len(ids) == 2:
            a, b = ids
            if not np.array_equal(oriented[a], oriented[b][::-1]):
                raise ValueError('floor_triangle_orientation_inconsistent')
            adjacency[a // 3].add(b // 3); adjacency[b // 3].add(a // 3)
    seen, pending = set(), [0]
    while pending:
        i = pending.pop()
        if i not in seen:
            seen.add(i); pending.extend(adjacency[i] - seen)
    if len(seen) != count:
        raise ValueError('floor_triangles_must_be_edge_connected')
    boundary_edges = edges[counts == 1]; neighbors = defaultdict(list)
    for a, b in boundary_edges:
        neighbors[int(a)].append(int(b)); neighbors[int(b)].append(int(a))
    if not neighbors or any(len(v) != 2 for v in neighbors.values()):
        raise ValueError('one_closed_floor_boundary_required')
    start = min(neighbors); boundary_ids = []; previous = None; current = start
    while current not in boundary_ids:
        boundary_ids.append(current)
        following = next(v for v in neighbors[current] if v != previous)
        previous, current = current, following
    if current != start or len(boundary_ids) != len(neighbors):
        raise ValueError('multiple_or_self_touching_floor_boundaries')
    boundary = points[boundary_ids, :2]
    if polygon_area(boundary) < 0:
        boundary = boundary[::-1]
    hull = ConvexHull(points[:, :2]); equations = hull.equations
    boundary_error = np.max(boundary @ equations[:, :2].T + equations[:, 2], axis=1)
    if np.max(np.abs(boundary_error)) > GEOMETRY_EPS_M:
        raise ValueError('nonconvex_floor_is_not_supported_by_this_version')
    flat = points[floor_faces, :2]
    areas = cross2(flat[:, 1] - flat[:, 0], flat[:, 2] - flat[:, 0]) / 2
    if np.any(areas <= 0) or abs(float(areas.sum()) - hull.volume) > AREA_EPS_M2:
        raise ValueError('projected_floor_triangles_do_not_cover_the_boundary')
    if abs(polygon_area(boundary) - hull.volume) > AREA_EPS_M2:
        raise ValueError('floor_boundary_does_not_trace_the_convex_polygon_once')
    low, high = flat.min(axis=1), flat.max(axis=1); tested = 0
    for i in range(count):
        possible = np.flatnonzero(np.all(np.minimum(high[:i], high[i]) - np.maximum(low[:i], low[i]) > 0., axis=1))
        for j in possible:
            tested += 1
            if tested > MAX_OVERLAP_PAIRS:
                raise ValueError('bounded_floor_overlap_check_exhausted')
            if polygon_area(clipped_polygon(flat[i], flat[j])) > AREA_EPS_M2:
                raise ValueError('floor_triangle_projections_overlap')
    if np.max(center @ equations[:, :2].T + equations[:, 2]) >= 0:
        raise ValueError('recorded_cavity_center_must_be_inside_verified_floor')
    return {'schema': VERSION, 'boundary_xy_m': boundary.tolist(), 'halfspaces': equations.tolist(),
        'selected_original_face_indices': np.flatnonzero(selected).tolist(), 'triangle_count': count,
        'connected_components': 1, 'euler_number': 1, 'boundary_edges': len(boundary_edges),
        'boundary_degree_all_two': True, 'convex_boundary_verified': True,
        'projected_triangle_area_m2': float(areas.sum()), 'floor_polygon_area_m2': float(hull.volume),
        'overlap_pairs_checked': tested, 'overlap_area_limit_m2': AREA_EPS_M2,
        'minimum_normal_z': float(upward[selected].min()), 'floor_heights_m': [float(points[:, 2].min()), float(points[:, 2].max())],
        'existing_floor_z_m': z, 'existing_floor_tolerance_m': tolerance,
        'recorded_circle_center_xy_m': center.tolist(), 'recorded_floor_circle_radius_m': cavity['flat_floor_radius_m'],
        'topology_repaired': False, 'physical_success': 'unmeasured'}


def load_floor(receiver_ref):
    from resimpair_pickplace.support_surface import check_support
    descriptor = read(resolve(receiver_ref))
    if not descriptor.get('roles', {}).get('stationary_container'):
        raise ValueError('bound_stationary_container_role_required')
    mesh = trimesh.load(resolve(descriptor['collision']), force='mesh', process=False)
    result = extract_floor(mesh, descriptor['cavity'])
    support = check_support(mesh, descriptor['cavity'])
    if not support['passed']:
        raise ValueError('retained_native_oriented_support_preflight_failed')
    return {**result, 'receiver': receiver_ref, 'source_collision': descriptor['collision'],
            'source_id': descriptor['source_id'], 'partition': descriptor['partition'],
            'retained_support_preflight': support, 'implementation': binding(__file__)}


def load_footprint(entry, native_geometry):
    """Enclose saved native actor vertices and the unchanged authored geometry.

    The native evidence is an explicit binding supplied by the caller. We never
    build an actor, cook geometry, infer a path from an asset name or substitute
    a newer descriptor. Its original equality/pose/density/mass limits apply.
    """
    descriptor = read(resolve(entry['asset']))
    meshes = [trimesh.load(resolve(ref), force='mesh', process=False) for ref in descriptor['collisions']]
    actual_objects = read(resolve(native_geometry))['objects']
    matches = [o for o in actual_objects if o['descriptor'] == entry['asset']]
    if len(matches) != 1 or len(matches[0]['shapes']) != len(meshes):
        raise ValueError('saved_native_geometry_must_bind_exact_original_descriptor_and_parts')
    actual = matches[0]; actual_vertices = []; errors = []
    for mesh, shape in zip(meshes, actual['shapes']):
        declared, cooked = np.asarray(mesh.vertices), np.asarray(shape['vertices'], float)
        if (cooked.ndim != 2 or cooked.shape[1] != 3 or not len(cooked)
                or not np.isfinite(cooked).all() or not len(declared)):
            raise ValueError('finite_nonempty_saved_native_vertices_required')
        distance = np.linalg.norm(declared[:, None] - cooked[None, :], axis=-1)
        error = float(max(distance.min(0).max(), distance.min(1).max())); errors.append(error)
        scale, pose = np.asarray(shape['scale'], float), np.asarray(shape['pose'], float)
        if (scale.shape != (3,) or pose.shape != (4, 4) or error >= 1e-5
                or not np.allclose(scale, [1, 1, 1]) or not np.allclose(pose, np.eye(4), atol=1e-6)
                or not abs(shape['density'] - descriptor['density_kg_m3']) < 1e-3):
            raise ValueError('saved_native_geometry_fails_original_vertex_pose_scale_or_density_rule')
        actual_vertices.append((cooked * scale) @ pose[:3, :3].T + pose[:3, 3])
    if actual['mass'] is None or not abs(actual['mass'] - descriptor['mass_kg']) < 1e-4:
        raise ValueError('saved_native_geometry_fails_original_mass_rule')
    declared = np.vstack([m.vertices for m in meshes]); actual = np.vstack(actual_vertices)
    native = np.vstack((declared, actual))
    visual = np.asarray(trimesh.load(resolve(descriptor['visual']), force='scene', process=False).to_geometry().vertices)
    if not len(native) or not len(visual) or not all(np.isfinite(v).all() for v in (native, visual)):
        raise ValueError('finite_original_native_and_visual_vertices_required')
    envelope = np.vstack((native, visual))
    receipt = {'asset': entry['asset'], 'cache': entry['cache'], 'variant_id': entry['variant_id'],
        'source_id': descriptor['source_id'], 'partition': descriptor['partition'], 'roles': descriptor['roles'],
        'native_collision_meshes': descriptor['collisions'], 'visual_mesh': descriptor['visual'],
        'saved_native_geometry': native_geometry, 'saved_native_vertex_count': len(actual),
        'declared_collision_vertex_count': len(declared), 'maximum_native_vertex_error_m': max(errors),
        'original_native_vertex_rule_m': 1e-5,
        'native_vertex_count': len(native), 'visual_vertex_count': len(visual),
        'native_bounds_xy_m': [native[:, :2].min(0).tolist(), native[:, :2].max(0).tolist()],
        'envelope_bounds_xy_m': [envelope[:, :2].min(0).tolist(), envelope[:, :2].max(0).tolist()],
        'dimensions_m': descriptor['dimensions_m'], 'half_size_m': descriptor['half_size'],
        'runtime_rescaling': False, 'physical_success': 'unmeasured'}
    return {'native': native, 'visual': visual, 'envelope': envelope, 'receipt': receipt}


def floor_margin(floor, points):
    equations = np.asarray(floor['halfspaces'], float)
    values = np.asarray(points, float)
    if values.ndim != 2 or values.shape[1] != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError('finite_nonempty_projected_footprint_required')
    return float(-np.max(values @ equations[:, :2].T + equations[:, 2]))


def rectangle_corners(bounds):
    low, high = np.asarray(bounds, float)
    return np.array([[low[0], low[1]], [high[0], low[1]], [high[0], high[1]], [low[0], high[1]]])


def inner_rectangle(floor, margin=FLOOR_MARGIN_M):
    if not np.isfinite(margin) or margin < FLOOR_MARGIN_M:
        raise ValueError('at_least_twelve_mm_floor_margin_required')
    boundary = np.asarray(floor['boundary_xy_m']); equations = np.asarray(floor['halfspaces'])
    centers = [(boundary.min(0) + boundary.max(0)) / 2, boundary.mean(0)]
    values = cross2(boundary, np.roll(boundary, -1, axis=0))
    centers.append(((boundary + np.roll(boundary, -1, axis=0)) * values[:, None]).sum(0) / (3 * values.sum()))
    a, b = np.abs(equations[:, 0]), np.abs(equations[:, 1]); xs, ys = a > 1e-12, b > 1e-12
    rectangles = []
    for center in centers:
        c = -(equations[:, 2] + equations[:, :2] @ center) - margin - NUMERICAL_INSET_M
        if np.any(c <= 0):
            continue
        limit = float(np.min(c[ys] / b[ys])); candidates = [limit]
        candidates.extend((c[xs & ys] / (2 * b[xs & ys])).tolist())
        for i in np.flatnonzero(xs):
            for j in np.flatnonzero(xs):
                denominator = b[i] / a[i] - b[j] / a[j]
                if abs(denominator) > 1e-12:
                    candidates.append(float((c[i] / a[i] - c[j] / a[j]) / denominator))
        for y in candidates:
            if 0 < y <= limit:
                x = float(np.min((c[xs] - b[xs] * y) / a[xs]))
                if x > 0:
                    bounds = np.array([center - [x, y], center + [x, y]])
                    if floor_margin(floor, rectangle_corners(bounds)) >= margin:
                        rectangles.append((x * y, bounds))
    if not rectangles:
        raise ValueError('no_verified_inner_rectangle_for_declared_margin')
    return max(rectangles, key=lambda item: item[0])[1]


def aabb_distance(first, second):
    a, b = np.asarray(first, float), np.asarray(second, float)
    separation = np.maximum(np.maximum(a[0] - b[1], b[0] - a[1]), 0.)
    return float(np.linalg.norm(separation))


def shelf_pack(footprints, floor, *, maximum_bins=4, yaws=None, margin=FLOOR_MARGIN_M, gap=PAYLOAD_GAP_M):
    if (not 1 <= len(footprints) <= 8 or type(maximum_bins) is not int or not 1 <= maximum_bins <= 4
            or not np.isfinite([margin, gap]).all() or margin < FLOOR_MARGIN_M or gap < PAYLOAD_GAP_M):
        raise ValueError('bounded_original_payload_count_and_strict_clearances_required')
    yaws = np.zeros(len(footprints)) if yaws is None else np.asarray(yaws, float)
    if yaws.shape != (len(footprints),) or not np.isfinite(yaws).all():
        raise ValueError('one_finite_yaw_per_original_payload_required')
    bounds = []
    for item, yaw in zip(footprints, yaws):
        xy = item['envelope'][:, :2] @ rotation(yaw).T
        bounds.append(np.array([xy.min(0), xy.max(0)]))
    box = inner_rectangle(floor, margin); width, height = box[1] - box[0]
    order = sorted(range(len(bounds)), key=lambda i: (-(bounds[i][1, 1] - bounds[i][0, 1]),
        -(bounds[i][1, 0] - bounds[i][0, 0]), footprints[i]['receipt']['source_id']))
    bins = []; placements = []; planning_gap = gap + NUMERICAL_INSET_M
    for index in order:
        low, high = bounds[index]; w, h = high - low
        selected_bin = selected_shelf = None
        if w > width or h > height:
            raise ValueError('original_payload_envelope_exceeds_verified_floor_rectangle')
        for receiving in bins:
            for shelf in receiving['shelves']:
                needed = w + (planning_gap if shelf['count'] else 0.)
                if h <= shelf['height'] and shelf['occupied_width'] + needed <= width:
                    selected_bin, selected_shelf = receiving, shelf; break
            if selected_shelf is not None:
                break
            start_y = sum(s['height'] for s in receiving['shelves']) + planning_gap * len(receiving['shelves'])
            if start_y + h <= height:
                shelf = {'height': float(h), 'start_y': float(start_y), 'occupied_width': 0., 'count': 0}
                receiving['shelves'].append(shelf); selected_bin, selected_shelf = receiving, shelf; break
        if selected_shelf is None:
            if len(bins) >= maximum_bins:
                raise ValueError('bounded_receiver_instance_capacity_exhausted')
            selected_shelf = {'height': float(h), 'start_y': 0., 'occupied_width': 0., 'count': 0}
            selected_bin = {'id': len(bins), 'shelves': [selected_shelf]}; bins.append(selected_bin)
        start_x = selected_shelf['occupied_width'] + (planning_gap if selected_shelf['count'] else 0.)
        origin = box[0] + [start_x, selected_shelf['start_y']] - low
        placed = bounds[index] + origin
        clearance = floor_margin(floor, rectangle_corners(placed))
        if clearance < margin:
            raise ValueError('packed_envelope_crosses_verified_floor_margin')
        selected_shelf['occupied_width'] = float(start_x + w); selected_shelf['count'] += 1
        placements.append({'original_index': index, 'source_id': footprints[index]['receipt']['source_id'],
            'receiver_instance': selected_bin['id'], 'origin_xy_in_receiver_m': origin.tolist(),
            'yaw_in_receiver_rad': float(yaws[index]), 'envelope_bounds_in_receiver_m': placed.tolist(),
            'floor_boundary_margin_m': clearance})
    pairs = []
    for i, a in enumerate(placements):
        for b in placements[:i]:
            if a['receiver_instance'] == b['receiver_instance']:
                distance = aabb_distance(a['envelope_bounds_in_receiver_m'], b['envelope_bounds_in_receiver_m'])
                if distance < gap:
                    raise ValueError('packed_payload_gap_below_original_twenty_five_mm')
                pairs.append({'original_indices': [a['original_index'], b['original_index']], 'gap_m': distance})
    return {'schema': VERSION, 'placements': sorted(placements, key=lambda p: p['original_index']),
        'bins': bins, 'receiver_instances': len(bins), 'inner_rectangle_bounds_m': box.tolist(),
        'boundary_margin_required_m': margin, 'inter_payload_gap_required_m': gap,
        'numerical_planning_inset_m': NUMERICAL_INSET_M, 'pair_clearances': pairs,
        'all_original_payloads_preserved': True, 'source_selection_uses_geometry': False,
        'footprint_scope': 'AABB enclosing every native and visual vertex at the declared target yaw',
        'physical_success': 'unmeasured'}
