"""Recook a metric variant and bind actual PhysX mass, COM and support geometry.

No time integration and no grasp success is inferred. This is a CPU asset
compiler; the retained geometric pose qualifier remains a separate stage.
"""
from __future__ import annotations
import argparse
import copy
import time
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from resimpair_grasp.common import binding, digest, read, resolve, write
from resimpair_grasp.assets import _stable_cook
from resimpair_tabletop.native_mass import calibrated_density
from resimpair_assets.metric_variants_v1 import compound_properties

VERSION = 'metric_native_compiler_v3'


def native_properties(collisions, density):
    import sapien
    system = sapien.physx.PhysxCpuSystem()
    scene = sapien.Scene([system])
    entity = sapien.Entity()
    body = sapien.physx.PhysxRigidDynamicComponent()
    material = sapien.physx.PhysxMaterial(.5, .5, 0.)
    for ref in collisions:
        shape = sapien.physx.PhysxCollisionShapeConvexMesh(
            filename=str(resolve(ref)), scale=[1., 1., 1.], material=material)
        shape.density = density
        body.attach(shape)
    entity.add_component(body); scene.add_entity(entity)
    center_pose = body.cmass_local_pose
    rotation = center_pose.to_transformation_matrix()[:3, :3]
    result = {'mass_kg': float(body.mass), 'center_of_mass_m': center_pose.p.tolist(),
        'principal_inertia_kg_m2': body.inertia.tolist(),
        'inertia_kg_m2': (rotation @ np.diag(body.inertia) @ rotation.T).tolist(),
        'cmass_pose_pq': list(center_pose.p) + list(center_pose.q),
        'dynamic_steps': 0, 'density_kg_m3': density}
    result['cmass_pose_pq'] = [float(x) for x in result['cmass_pose_pq']]
    if (not np.isfinite(result['mass_kg']) or result['mass_kg'] <= 0
            or not np.isfinite(result['center_of_mass_m']).all()
            or not np.isfinite(result['inertia_kg_m2']).all()
            or min(result['principal_inertia_kg_m2']) <= 0):
        raise ValueError('invalid_native_mass_COM_or_inertia')
    return result


def support_evidence(parts, center, tolerance=.0005):
    points = np.concatenate([p.vertices for p in parts]); z = float(points[:, 2].min())
    contact = np.unique(points[points[:, 2] <= z + tolerance, :2], axis=0)
    value = {'schema': 'native_static_support_geometry_v1',
        'method': 'COM_projection_inside_convex_hull_of_native_bottom_vertices',
        'contact_height_tolerance_m': tolerance, 'table_z_m': z,
        'native_COM_m': list(map(float, center)), 'support_vertices_xy_m': contact.tolist(),
        'dynamic_settling': 'not_run', 'physical_stability': 'unknown'}
    if len(contact) < 3 or np.linalg.matrix_rank(contact - contact[0], tol=1e-9) < 2:
        return {**value, 'status': 'unstable_or_point_support', 'minimum_margin_m': None}
    hull = ConvexHull(contact)
    distances = -(hull.equations[:, :2] @ np.asarray(center)[:2] + hull.equations[:, 2])
    margin = float(distances.min())
    return {**value, 'status': 'pass' if margin >= 0 else 'COM_outside_support',
        'minimum_margin_m': margin, 'support_polygon_xy_m': contact[hull.vertices].tolist()}


def compile_variant(descriptor_path, output):
    """New folder only; native cook plus unchanged 3 mm/10 um surface rules."""
    start = time.monotonic(); output = Path(output)
    if output.exists(): raise FileExistsError(output)
    descriptor_path = Path(descriptor_path)
    d = read(descriptor_path); output.mkdir(parents=True)
    write(output / 'input.json', {'parent': binding(descriptor_path), 'implementation': binding(__file__)})
    if d.get('semantic_role') != 'manipulation':
        raise ValueError('native_grasp_compiler_requires_manipulation_role')
    collisions, cooking = [], []
    for i, ref in enumerate(d['collisions']):
        part = trimesh.load(resolve(ref), force='mesh', process=False)
        if not np.isfinite(part.vertices).all(): raise ValueError('nonfinite_collision_vertices')
        ref, passes = _stable_cook(part, output, i)
        collisions.append(ref); cooking.append(passes)
    parts = [trimesh.load(resolve(ref), force='mesh', process=True) for ref in collisions]
    density, mass_checks = calibrated_density(collisions, d['density_kg_m3'], d['mass_kg'])
    native = native_properties(collisions, density)
    props = compound_properties(parts)
    native['collision_shape_volume_sum_m3'] = props['collision_shape_volume_sum_m3']
    native['density_overlapping_convex_volumes_not_boolean_union'] = True
    native['measured_real_mass'] = False
    support = support_evidence(parts, native['center_of_mass_m'])
    # A changed scale is never allowed to reuse stale distance statistics.
    visual = trimesh.load(resolve(d['visual']), force='scene', process=False).to_geometry()
    points, _ = trimesh.sample.sample_surface(visual, 1800, seed=71)
    forward = np.min([trimesh.proximity.closest_point(p, points)[1] for p in parts], axis=0)
    exterior = []
    for i, part in enumerate(parts):
        pts, faces = trimesh.sample.sample_surface(part, 400, seed=71+i)
        outward = pts + part.face_normals[faces] * .0001
        keep = np.ones(len(pts), bool)
        for j, other in enumerate(parts):
            if i != j: keep &= ~other.contains(outward)
        exterior.extend(pts[keep])
    if not exterior: raise ValueError('empty_exterior_surface')
    reverse = trimesh.proximity.closest_point(visual, np.asarray(exterior))[1]
    surface = {'visual_to_collision_p95_m': float(np.quantile(forward, .95)),
        'union_external_p95_m': float(np.quantile(reverse, .95)),
        'sample_count_forward': 1800, 'union_external_samples': len(exterior),
        'unchanged_absolute_tolerance_m': .003}
    native_errors = [max(p['vertex_error_m'] for p in passes[-1:]) for passes in cooking]
    geometry_ok = max(surface['visual_to_collision_p95_m'], surface['union_external_p95_m']) <= .003
    replay_ok = max(native_errors) < 1e-5
    allowed = geometry_ok and replay_ok and support['status'] == 'pass'
    write(output / 'native_properties.json', native)
    write(output / 'native_support.json', support)
    value = copy.deepcopy(d)
    value.update(schema='resimpair_mesh_asset_v1', collisions=collisions, cooking=cooking,
        density_kg_m3=density, native_mass_preflight=mass_checks,
        native_properties=binding(output / 'native_properties.json'),
        native_support=binding(output / 'native_support.json'),
        native_COM_m=native['center_of_mass_m'],
        surface_comparison=surface, fresh_process_native_vertex_errors_m=native_errors,
        metric_parent=binding(descriptor_path),
        variant_id=d.get('geometry_variant_key', d['id']),
        collision_bounds_m=np.array([np.min([p.bounds[0] for p in parts], axis=0),
                                     np.max([p.bounds[1] for p in parts], axis=0)]).tolist(),
        roles={'static_distractor': replay_ok, 'rigid_manipulation_candidate': allowed,
               'container_support': False, 'articulated': False},
        qualification={'geometry': 'pass' if geometry_ok else 'fail',
            'native_collision_replay': 'pass' if replay_ok else 'fail',
            'native_support_geometry': support['status'], 'grasp_proposals': 'pending',
            'physical_execution': 'unknown', 'training_admitted': False})
    path = output / 'descriptor.json'; write(path, value)
    result = {'schema': VERSION, 'asset': binding(path), 'source_id': d['source_id'],
        'variant_id': value['variant_id'], 'relative_scale': d['size_provenance']['selected_relative_scale'],
        'native_support': value['native_support'], 'native_properties': value['native_properties'],
        'geometry_qualified': allowed, 'qualification_stage': 'native_geometry_support_only',
        'physical_status': 'unmeasured', 'elapsed_seconds': time.monotonic()-start,
        'surface': surface, 'support_status': support['status'], 'dynamic_steps': 0, 'API_calls': 0}
    write(output / 'result.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--descriptor', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args(); print(compile_variant(a.descriptor, a.output))
