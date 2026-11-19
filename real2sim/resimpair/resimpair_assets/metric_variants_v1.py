"""Offline metric size variants with explicit mass assumptions and fresh cache keys.

This is a preprocessor, not an online rescaling of cached grasps. It preserves
the parent assets and writes a new visual/collision descriptor for every finite
declared variant. Native cooking, stable support and grasps still need separate
qualification. No solver, simulator step, GPU or model API is invoked here.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import trimesh

from resimpair_grasp.common import binding, digest, read, resolve, write

VERSION = 'source_aware_metric_variants_v1'
ROLES = {'manipulation', 'container', 'background'}


def positive(value, name):
    if isinstance(value, bool):
        raise ValueError(name + '_must_be_positive_finite')
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise ValueError(name + '_must_be_positive_finite')
    return number


def vector(value, name, size=3):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError(name + '_must_be_positive_finite_vector')
    return result


def evidence_refs(values, name, *, required=True):
    if not isinstance(values, list) or (required and not values):
        raise ValueError(name + '_bound_evidence_required')
    for reference in values:
        resolve(reference)


def load_geometry(reference):
    scene = trimesh.load(resolve(reference), force='scene', process=False)
    geometry = scene.to_geometry()
    if not isinstance(geometry, trimesh.Trimesh) or not len(geometry.faces):
        raise ValueError('triangle_mesh_required')
    if not np.isfinite(geometry.vertices).all() or np.any(geometry.extents <= 0):
        raise ValueError('finite_three_dimensional_geometry_required')
    return scene, geometry


def source_scale(transform):
    T = np.asarray(transform, float)
    if T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1]):
        raise ValueError('finite_source_transform_required')
    scales = np.linalg.svd(T[:3, :3], compute_uv=False)
    if scales.min() <= 0 or not np.allclose(scales, scales.mean(), rtol=1e-6, atol=1e-12):
        raise ValueError('source_transform_is_not_uniform_similarity')
    return float(scales.mean())


def size_plan(descriptor, mesh, policy):
    """Recover a base scale, then validate a finite explicit variation range."""
    if policy.get('schema') != VERSION or policy.get('source_id') != descriptor['source_id']:
        raise ValueError('policy_schema_or_source_identity_mismatch')
    role = policy.get('semantic_role')
    if role not in ROLES or not str(policy.get('semantic_category', '')).strip():
        raise ValueError('explicit_semantic_role_and_category_required')
    if policy.get('source_object_role') == 'furniture' and role != 'background':
        raise ValueError('furniture_cannot_be_silently_reclassified_for_grasping')
    if not str(policy.get('role_rationale', '')).strip():
        raise ValueError('explicit_role_rationale_required')
    size = policy['size']; mode = size.get('mode')
    local_obb = np.sort(mesh.bounding_box_oriented.primitive.extents)
    expected = None
    tolerance = float(size.get('dimension_tolerance_relative', .05))
    if not np.isfinite(tolerance) or not 0 <= tolerance <= .10:
        raise ValueError('dimension_tolerance_must_be_between0and10percent')
    if mode == 'source_metric':
        evidence_refs(size.get('evidence', []), 'metric_size')
        if 'dimensions_m' in size:
            expected = np.sort(vector(size['dimensions_m'], 'source_dimensions'))
        if 'source_unit_to_m' in size:
            multiplier = positive(size['source_unit_to_m'], 'unit_to_m') / source_scale(descriptor['source_to_asset'])
        elif expected is not None:
            multiplier = float(np.median(expected / local_obb))
        else:
            raise ValueError('source_metric_requires_units_or_dimensions')
        confidence = 'source_metric_evidence'
    elif mode == 'inferred_category':
        if not str(size.get('rationale', '')).strip():
            raise ValueError('inferred_size_rationale_required')
        evidence_refs(size.get('evidence', []), 'inferred_size', required=False)
        multiplier = positive(size['base_longest_m'], 'base_longest_m') / local_obb[-1]
        confidence = 'engineering_inference_not_measured'
    else:
        raise ValueError('source_metric_or_inferred_category_size_required')
    base = local_obb * multiplier
    agreement = None
    if expected is not None:
        errors = np.abs(base - expected) / expected
        agreement = {'comparison': 'sorted_oriented_box_extents', 'source_dimensions_m': expected.tolist(),
            'mesh_base_obb_m': base.tolist(), 'relative_errors': errors.tolist(),
            'tolerance': tolerance, 'passed': bool(np.all(errors <= tolerance))}
        if not agreement['passed']:
            raise ValueError('source_dimension_geometry_disagreement_no_anisotropic_fix')
    ratios = vector(policy.get('scale_factors', [1.0]), 'scale_factors', len(policy.get('scale_factors', [1.0])))
    limits = vector(policy.get('variant_ratio_bounds', [.8, 1.2]), 'variant_ratio_bounds', 2)
    if not 1 <= len(ratios) <= 9 or len(set(ratios)) != len(ratios) or not limits[0] <= 1 <= limits[1]:
        raise ValueError('one_to_nine_unique_bounded_variants_required')
    if np.any(ratios < limits[0]) or np.any(ratios > limits[1]):
        raise ValueError('variant_outside_declared_ratio_bounds')
    plausible = size.get('plausible_longest_m')
    if mode == 'inferred_category' and plausible is None:
        raise ValueError('unknown_units_require_explicit_plausible_size_interval')
    if plausible is not None:
        limits = vector(plausible, 'plausible_longest_m', 2)
        if limits[0] > limits[1] or np.any(base[-1] * ratios < limits[0]) or np.any(base[-1] * ratios > limits[1]):
            raise ValueError('variant_outside_category_size_interval')
    return {'base_multiplier_from_parent': multiplier, 'base_obb_m': base.tolist(),
        'parent_obb_m': local_obb.tolist(), 'mode': mode, 'confidence': confidence,
        'dimension_agreement': agreement, 'scale_factors': ratios.tolist(),
        'all_grasp_positions_must_be_bound_to_variant': True}


def compound_properties(parts):
    """Finite uniform-density aggregate, with convex overlap explicitly retained.

    The sum is the collision-shape mass approximation, not a measurement of a
    hollow object's material volume. Scaling preserves its COM and inertia law.
    """
    rows = []
    for mesh in parts:
        if not mesh.is_watertight or not mesh.is_winding_consistent:
            raise ValueError('closed_consistently_wound_collision_parts_required')
        volume = positive(mesh.volume, 'collision_volume')
        center, inertia = np.asarray(mesh.center_mass), np.asarray(mesh.moment_inertia)
        if not np.isfinite(center).all() or not np.isfinite(inertia).all():
            raise ValueError('nonfinite_collision_mass_properties')
        if np.any(center < mesh.bounds[0] - 1e-8) or np.any(center > mesh.bounds[1] + 1e-8):
            raise ValueError('collision_center_mass_outside_bounds')
        rows.append((volume, center, inertia))
    if not rows:
        raise ValueError('collision_parts_required')
    volume = sum(row[0] for row in rows)
    center = sum(row[0] * row[1] for row in rows) / volume
    inertia = np.zeros((3, 3))
    for part_volume, part_center, part_inertia in rows:
        r = part_center - center
        inertia += part_inertia + part_volume * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    if not np.isfinite(inertia).all() or np.min(np.linalg.eigvalsh(inertia)) <= 0:
        raise ValueError('positive_finite_compound_inertia_required')
    return {'collision_shape_volume_sum_m3': volume, 'center_of_mass_m': center.tolist(),
        'unit_density_inertia_kg_m2': inertia.tolist(), 'model': 'sum_of_uniform_density_collision_parts',
        'overlap_correction': 'not_applied_matches_additive_shape_model',
        'material_volume_or_true_mass_measured': False}


def mass_plan(policy, base_volume):
    mass = policy['mass']; mode = mass.get('mode')
    if not str(mass.get('rationale', '')).strip():
        raise ValueError('mass_rationale_required')
    if mode == 'measured_base_mass':
        evidence_refs(mass.get('evidence', []), 'measured_mass')
        base_mass = positive(mass['base_mass_kg'], 'base_mass_kg')
        confidence = 'source_record_mass_for_base_geometry'
    elif mode == 'assumed_base_mass':
        evidence_refs(mass.get('evidence', []), 'assumed_mass', required=False)
        base_mass = positive(mass['base_mass_kg'], 'base_mass_kg')
        confidence = 'engineering_assumption_not_measured'
    elif mode == 'effective_density':
        evidence_refs(mass.get('evidence', []), 'effective_density', required=False)
        base_mass = positive(mass['density_kg_m3'], 'density_kg_m3') * base_volume
        confidence = 'engineering_effective_density_not_measured_material_density'
    else:
        raise ValueError('explicit_mass_or_effective_density_required')
    return {'mode': mode, 'base_mass_kg': base_mass, 'effective_density_kg_m3': base_mass / base_volume,
        'confidence': confidence, 'variation_rule': 'same_effective_density_mass_times_scale_cubed'}


def materialize(descriptor_path, policy_path, output):
    descriptor_path, policy_path, output = Path(descriptor_path), Path(policy_path), Path(output)
    if output.exists():
        raise ValueError('preserve_existing_variant_directory')
    descriptor_ref, policy_ref = binding(descriptor_path), binding(policy_path)
    d, policy = read(descriptor_path), read(policy_path)
    if d.get('schema') != 'resimpair_mesh_asset_v1':
        raise ValueError('existing_native_mesh_descriptor_required')
    scene, visual = load_geometry(d['visual'])
    plan = size_plan(d, visual, policy)
    parts = [load_geometry(ref)[1] for ref in d['collisions']]
    parent_properties = compound_properties(parts)
    base_multiplier = plan['base_multiplier_from_parent']
    base_volume = parent_properties['collision_shape_volume_sum_m3'] * base_multiplier ** 3
    mass = mass_plan(policy, base_volume)
    # All inexpensive validation precedes the durable reservation; once created
    # an interrupted output remains explicit and is never silently rerun.
    output.mkdir(parents=True)
    write(output / 'reservation.json', {'schema': VERSION, 'parent_descriptor': descriptor_ref,
        'policy': policy_ref, 'implementation': binding(__file__), 'started_unix': time.time(),
        'API_calls': 0, 'GPU_jobs': 0, 'physical_executions': 0})
    write(output / 'resolved_plan.json', {'size': plan, 'mass': mass,
        'parent_collision_mass_properties': parent_properties, 'policy': policy_ref})
    entries = []
    try:
        for relative in plan['scale_factors']:
            factor = base_multiplier * relative
            variant_id = digest({'parent': descriptor_ref, 'policy': policy_ref, 'scale': relative})[:20]
            directory = output / variant_id; directory.mkdir()
            converted = scene.copy(); converted.apply_scale(factor)
            visual_path = directory / 'visual.glb'; converted.export(visual_path)
            exported, new_visual = load_geometry(binding(visual_path))
            expected_bounds = visual.bounds * factor
            if not np.allclose(new_visual.bounds, expected_bounds, rtol=1e-6, atol=1e-7):
                raise ValueError('visual_export_changed_scale_or_origin')
            new_parts, collisions = [], []
            for number, part in enumerate(parts):
                p = part.copy(); p.apply_scale(factor)
                path = directory / f'collision_{number:03d}.obj'; p.export(path, digits=12)
                collisions.append(binding(path)); new_parts.append(load_geometry(collisions[-1])[1])
            props = compound_properties(new_parts)
            intended_mass = mass['base_mass_kg'] * relative ** 3
            density = intended_mass / props['collision_shape_volume_sum_m3']
            physical = {**props, 'mass_kg': intended_mass, 'effective_density_kg_m3': density,
                'inertia_kg_m2': (np.array(props['unit_density_inertia_kg_m2']) * density).tolist(),
                'native_mass_calibration': 'pending', 'true_item_mass': 'source_record' if relative == 1 and mass['mode'] == 'measured_base_mass' else 'derived_or_assumed'}
            write(directory / 'physical_properties.json', physical)
            bounds = new_visual.bounds
            collision_bounds = np.array([np.min([p.bounds[0] for p in new_parts], axis=0),
                                         np.max([p.bounds[1] for p in new_parts], axis=0)])
            T = np.diag([factor, factor, factor, 1.]) @ np.asarray(d['source_to_asset'])
            role = policy['semantic_role']
            descriptor = {k: copy.deepcopy(d[k]) for k in ('schema', 'source_id', 'name', 'category', 'partition', 'provenance') if k in d}
            descriptor.update(id=d['id'] + '_metric_' + variant_id, visual=binding(visual_path), collisions=collisions,
                source_to_asset=T.tolist(), dimensions_m=(bounds[1] - bounds[0]).tolist(),
                half_size=((bounds[1] - bounds[0]) / 2).tolist(), bounds_m=bounds.tolist(),
                collision_bounds_m=collision_bounds.tolist(), mass_kg=intended_mass, density_kg_m3=density,
                mass_provenance=mass, size_provenance={**plan, 'selected_relative_scale': relative,
                    'scale_from_parent': factor, 'policy': policy_ref},
                semantic_role=role, semantic_category=policy['semantic_category'],
                semantic_status='explicit_policy_with_bound_evidence',
                roles={'static_distractor': role == 'background', 'rigid_manipulation_candidate': role == 'manipulation',
                       'container_support': False, 'container_candidate': role == 'container', 'articulated': False},
                collision_policy='uniformly_scaled_parent_parts_pending_native_recook_and_independent_check',
                physical_properties=binding(directory / 'physical_properties.json'),
                qualification={'geometry': 'pending_independent_check', 'native_collision_replay': 'unknown',
                    'native_mass': 'unknown', 'stable_initial_pose': 'unknown', 'physical_execution': 'unknown',
                    'grasp_candidates': 'not_generated', 'training_admitted': False},
                grasp_cache_invalidated=True, requires_new_grasp_qualification=role == 'manipulation',
                no_parent_pose_transferred=True, stable_support_requires_requalification=True,
                lineage={'parent_descriptor': descriptor_ref, 'policy': policy_ref,
                    'split_group_source_id': d['source_id'], 'all_variants_share_source_partition': True},
                generator=binding(__file__))
            path = directory / 'descriptor.json'; write(path, descriptor)
            entries.append({'variant_id': variant_id, 'source_id': d['source_id'], 'relative_scale': relative,
                'parent_scale_multiplier': factor, 'role': role, 'descriptor': binding(path),
                'dimensions_m': descriptor['dimensions_m'], 'mass_kg': intended_mass,
                'new_grasp_cache': None, 'physical_status': 'not_tested'})
    except Exception as error:
        write(output / 'terminal.json', {'status': 'failed_preserve_partial_outputs', 'reason': str(error),
            'completed_variants': entries, 'API_calls': 0, 'GPU_jobs': 0, 'physical_executions': 0})
        raise
    result = {'schema': VERSION, 'status': 'preprocessed_pending_native_and_task_qualification',
        'parent_descriptor': descriptor_ref, 'policy': policy_ref, 'variants': entries,
        'old_assets_and_caches_modified': False, 'API_calls': 0, 'GPU_jobs': 0, 'physical_executions': 0}
    write(output / 'catalog.json', {'schema': 'resimpair_mesh_catalog_v1',
        'assets': [row['descriptor'] for row in entries], 'failures': [], 'qualification': result['status']})
    write(output / 'terminal.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--descriptor', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = materialize(args.descriptor, args.policy, args.output)
    print(json.dumps({'status': result['status'], 'variants': len(result['variants']),
                      'output': str(args.output)}, ensure_ascii=False))
