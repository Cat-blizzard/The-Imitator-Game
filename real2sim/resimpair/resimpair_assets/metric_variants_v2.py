"""Role-aware numerical bounds over the retained metric-variant preprocessor.

Small metric variants retain density: mass scales as s^3, inertia as s^5.
This layer rejects inconsistent or implausible numbers without clipping them.
It is not a material estimator, native physics qualification or grasp solver.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write
from . import metric_variants_v1 as retained

VERSION = 'bounded_metric_variants_v2'
DEFAULT_BOUNDS = ROOT / 'configs/metric_numeric_bounds_v2.json'
RANGE_KEYS = ('longest_dimension_m', 'shortest_dimension_m', 'mass_kg', 'effective_density_kg_m3')


def load_bounds(path=DEFAULT_BOUNDS):
    path = Path(path)
    config = read(path)
    if (config.get('schema') != 'metric_role_numeric_bounds_v2'
            or config.get('automatic_clipping') is not False
            or config.get('assumed_mass_requires_rationale') is not True
            or config.get('physical_success_implied') is not False):
        raise ValueError('explicit_numeric_only_no_clip_profile_required')
    if {key: value.get('body_type') for key, value in config['roles'].items()} != {
            'manipulation': 'dynamic', 'container': 'static_support', 'background': 'static_visual'}:
        raise ValueError('role_body_types_must_separate_dynamic_and_static_assets')
    for name, profile in list(config['roles'].items()) + list(config['category_profiles'].items()):
        for key in RANGE_KEYS:
            if key not in profile:
                if name in config['roles']: raise ValueError('role_numeric_range_missing:' + key)
                continue
            interval = retained.vector(profile[key], name + ':' + key, 2)
            if interval[0] > interval[1]: raise ValueError('numeric_interval_reversed')
    ratios = retained.vector(config['variant_ratio_bounds'], 'variant_ratio_bounds', 2)
    if not .5 <= ratios[0] <= 1 <= ratios[1] <= 1.5:
        raise ValueError('bounded_small_variation_profile_required')
    if type(config['maximum_variants']) is not int or not 1 <= config['maximum_variants'] <= 9:
        raise ValueError('bounded_variant_count_required')
    retained.positive(config['maximum_inertia_condition_number'], 'inertia_condition_limit')
    retained.positive(config['relative_consistency_tolerance'], 'consistency_tolerance')
    return config


def role_profile(policy, config):
    role = policy['semantic_role']
    if role not in config['roles']: raise ValueError('unsupported_explicit_role')
    category = str(policy['semantic_category']).strip().lower().replace(' ', '_')
    selected = copy.deepcopy(config['roles'][role]); matched = []
    if policy.get('source_object_role') == 'furniture' and role != 'background':
        raise ValueError('furniture_requires_static_background_role')
    for name, profile in config['category_profiles'].items():
        if category not in profile['categories']: continue
        if role not in profile['allowed_roles']:
            raise ValueError('category_role_conflict:' + category)
        matched.append(name)
        for key in RANGE_KEYS:
            if key in profile:
                selected[key] = [max(selected[key][0], profile[key][0]), min(selected[key][1], profile[key][1])]
                if selected[key][0] > selected[key][1]: raise ValueError('category_role_bounds_do_not_overlap')
    selected.update(semantic_role=role, semantic_category=category, matched_category_profiles=matched,
                    numeric_bounds_are_engineering_assumptions=True)
    return selected


def _within(value, interval, name):
    if not np.isfinite(value) or not interval[0] <= value <= interval[1]:
        raise ValueError(f'{name}_outside_declared_bounds:{value}:{interval}')


def _consistent(value, expected, name, tolerance):
    if not np.allclose(value, expected, rtol=tolerance, atol=1e-12):
        raise ValueError(name + '_inconsistent_with_bound_geometry_or_scale')


def inspect_values(policy, size, mass, props, dimensions, collision_bounds, relative, config):
    """Pure numerical qualification; actual source/geometry hashes checked outside."""
    profile = role_profile(policy, config)
    tol = config['relative_consistency_tolerance']
    ratios = np.asarray(size['scale_factors'], float)
    limits = config['variant_ratio_bounds']
    if (not 1 <= len(ratios) <= config['maximum_variants'] or not np.isfinite(ratios).all()
            or len(set(ratios)) != len(ratios) or np.any(ratios < limits[0]) or np.any(ratios > limits[1])):
        raise ValueError('variant_ratios_outside_numeric_profile')
    declared = retained.vector(policy.get('variant_ratio_bounds', limits), 'declared_ratio_bounds', 2)
    if declared[0] < limits[0] or declared[1] > limits[1]:
        raise ValueError('declared_variant_range_exceeds_numeric_profile')
    if relative not in size['scale_factors']: raise ValueError('unplanned_metric_variant')
    dims = retained.vector(dimensions, 'dimensions_m')
    bounds = np.asarray(collision_bounds, float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
        raise ValueError('finite_ordered_collision_bounds_required')
    _within(float(max(dims)), profile['longest_dimension_m'], 'longest_dimension_m')
    _within(float(min(dims)), profile['shortest_dimension_m'], 'shortest_dimension_m')
    value = retained.positive(props['mass_kg'], 'mass_kg')
    density = retained.positive(props['effective_density_kg_m3'], 'density_kg_m3')
    volume = retained.positive(props['collision_shape_volume_sum_m3'], 'collision_volume')
    _within(value, profile['mass_kg'], 'mass_kg')
    _within(density, profile['effective_density_kg_m3'], 'effective_density_kg_m3')
    _consistent(value, mass['base_mass_kg'] * relative ** 3, 'cubic_mass', tol)
    _consistent(value, density * volume, 'mass_density_volume', tol)
    _consistent(density, mass['effective_density_kg_m3'], 'constant_effective_density', tol)
    center = np.asarray(props['center_of_mass_m'], float)
    inertia = np.asarray(props['inertia_kg_m2'], float)
    unit = np.asarray(props['unit_density_inertia_kg_m2'], float)
    if center.shape != (3,) or not np.isfinite(center).all() or np.any(center < bounds[0]-1e-8) or np.any(center > bounds[1]+1e-8):
        raise ValueError('finite_COM_inside_collision_bounds_required')
    if (inertia.shape != (3, 3) or not np.isfinite(inertia).all()
            or not np.allclose(inertia, inertia.T, rtol=tol, atol=1e-12)):
        raise ValueError('finite_symmetric_inertia_required')
    _consistent(inertia, unit * density, 'inertia_density', tol)
    eig = np.linalg.eigvalsh(inertia)
    if eig[0] <= 0: raise ValueError('positive_inertia_required')
    if eig[-1] > eig[0] + eig[1] + max(1e-12, tol * eig[-1]):
        raise ValueError('inertia_triangle_inequality_failed')
    if eig[-1] / eig[0] > config['maximum_inertia_condition_number']:
        raise ValueError('ill_conditioned_inertia')
    maximum_radius2 = float(np.sum(np.maximum(abs(bounds[0] - center), abs(bounds[1] - center)) ** 2))
    if eig[-1] > value * maximum_radius2 * (1 + tol) + 1e-12:
        raise ValueError('inertia_exceeds_mass_and_geometry_bound')
    mode = policy['mass']['mode']
    if mode != 'measured_base_mass' and not config['assumed_mass_allowed']:
        raise ValueError('engineering_mass_disabled_by_profile')
    if not str(policy['mass'].get('rationale', '')).strip(): raise ValueError('mass_assumption_rationale_required')
    dynamic = profile['body_type'] == 'dynamic'
    return {'passed': True, 'profile': profile, 'relative_scale': relative,
        'dimensions_m': dims.tolist(), 'mass_kg': value, 'effective_density_kg_m3': density,
        'principal_inertia_kg_m2': eig.tolist(), 'inertia_condition_number': float(eig[-1] / eig[0]),
        'mass_scaling': 'base_mass_times_relative_scale_cubed',
        'inertia_scaling': 'base_inertia_times_relative_scale_to_fifth',
        'mass_confidence': (mass['confidence'] if relative == 1 else 'constant_density_derivative_of_' + mass['confidence']),
        'actual_variant_mass_measured': mode == 'measured_base_mass' and relative == 1,
        'simulation_usage': {'body_type': profile['body_type'], 'grasp_target_allowed': dynamic,
            'counts_toward_robot_payload': dynamic, 'robot_payload_mass_kg': value if dynamic else 0.0},
        'native_mass_and_inertia_qualification': 'pending', 'physical_success': 'unknown',
        'no_values_clipped': True}


def _legacy_policy(policy):
    if policy.get('schema') not in {retained.VERSION, VERSION}:
        raise ValueError('metric_policy_v1_or_v2_required')
    value = copy.deepcopy(policy); value['schema'] = retained.VERSION
    return value


def preflight(descriptor_path, policy_path, bounds_path=DEFAULT_BOUNDS):
    d, p = read(descriptor_path), read(policy_path)
    policy = _legacy_policy(p); config = load_bounds(bounds_path)
    _, visual = retained.load_geometry(d['visual'])
    size = retained.size_plan(d, visual, policy)
    parts = [retained.load_geometry(ref)[1] for ref in d['collisions']]
    parent = retained.compound_properties(parts)
    base_factor = size['base_multiplier_from_parent']
    mass = retained.mass_plan(policy, parent['collision_shape_volume_sum_m3'] * base_factor ** 3)
    collision_bounds = np.array([np.min([m.bounds[0] for m in parts], axis=0), np.max([m.bounds[1] for m in parts], axis=0)])
    checks = []
    for ratio in size['scale_factors']:
        scale = base_factor * ratio
        props = {'collision_shape_volume_sum_m3': parent['collision_shape_volume_sum_m3'] * scale ** 3,
            'center_of_mass_m': (np.array(parent['center_of_mass_m']) * scale).tolist(),
            'unit_density_inertia_kg_m2': (np.array(parent['unit_density_inertia_kg_m2']) * scale ** 5).tolist(),
            'mass_kg': mass['base_mass_kg'] * ratio ** 3,
            'effective_density_kg_m3': mass['effective_density_kg_m3'],
            'inertia_kg_m2': (np.array(parent['unit_density_inertia_kg_m2']) * scale ** 5 * mass['effective_density_kg_m3']).tolist()}
        checks.append(inspect_values(policy, size, mass, props, visual.extents * scale,
                                      collision_bounds * scale, ratio, config))
    return {'schema': VERSION, 'parent_descriptor': binding(descriptor_path), 'policy': binding(policy_path),
        'bounds': binding(bounds_path), 'implementation': binding(__file__), 'checks': checks,
        'size_plan': size, 'mass_plan': mass, 'status': 'numeric_preflight_passed_not_physics'}


def audit_variant(descriptor_path, bounds_path=DEFAULT_BOUNDS):
    """Read-only recheck of existing v1/v2 outputs, including real mesh arrays."""
    d = read(descriptor_path); config = load_bounds(bounds_path)
    policy = read(resolve(d['size_provenance']['policy']))
    props = read(resolve(d['physical_properties']))
    _, visual = retained.load_geometry(d['visual'])
    parts = [retained.load_geometry(ref)[1] for ref in d['collisions']]
    reconstructed = retained.compound_properties(parts)
    tol = config['relative_consistency_tolerance']
    for key in ('collision_shape_volume_sum_m3', 'center_of_mass_m', 'unit_density_inertia_kg_m2'):
        _consistent(props[key], reconstructed[key], key, tol)
    collision_bounds = np.array([np.min([m.bounds[0] for m in parts], axis=0), np.max([m.bounds[1] for m in parts], axis=0)])
    _consistent(d['dimensions_m'], visual.extents, 'descriptor_visual_dimensions', tol)
    _consistent(d['collision_bounds_m'], collision_bounds, 'descriptor_collision_bounds', tol)
    _consistent(d['mass_kg'], props['mass_kg'], 'descriptor_mass', tol)
    _consistent(d['density_kg_m3'], props['effective_density_kg_m3'], 'descriptor_density', tol)
    if (d['semantic_role'] != policy['semantic_role'] or d['semantic_category'] != policy['semantic_category']
            or d['source_id'] != policy['source_id']):
        raise ValueError('bound_policy_role_or_identity_mismatch')
    # Re-derive scaling and mass from the bound parent/policy, not the output's
    # own claimed provenance. A self-consistent but substituted mass is invalid.
    parent = read(resolve(d['lineage']['parent_descriptor']))
    _, parent_visual = retained.load_geometry(parent['visual'])
    expected_size = retained.size_plan(parent, parent_visual, _legacy_policy(policy))
    relative = d['size_provenance']['selected_relative_scale']
    factor = expected_size['base_multiplier_from_parent'] * relative
    for key in ('base_multiplier_from_parent', 'base_obb_m', 'scale_factors'):
        _consistent(d['size_provenance'][key], expected_size[key], 'size_provenance:' + key, tol)
    _consistent(d['source_to_asset'], np.diag([factor, factor, factor, 1.]) @ np.asarray(parent['source_to_asset']),
                'source_to_asset', tol)
    _consistent(visual.bounds, parent_visual.bounds * factor, 'variant_visual_bounds', tol)
    expected_mass = retained.mass_plan(policy, reconstructed['collision_shape_volume_sum_m3'] / relative ** 3)
    for key in ('base_mass_kg', 'effective_density_kg_m3'):
        _consistent(d['mass_provenance'][key], expected_mass[key], 'mass_provenance:' + key, tol)
    if d['mass_provenance']['mode'] != expected_mass['mode'] or d['mass_provenance']['confidence'] != expected_mass['confidence']:
        raise ValueError('mass_provenance_mode_or_confidence_changed')
    result = inspect_values(policy, d['size_provenance'], d['mass_provenance'], props,
        d['dimensions_m'], collision_bounds, d['size_provenance']['selected_relative_scale'], config)
    return {'schema': VERSION, 'descriptor': binding(descriptor_path), 'bounds': binding(bounds_path),
        'physical_properties': d['physical_properties'], 'implementation': binding(__file__), **result,
        'all_geometry_bindings_verified': True, 'API_calls': 0, 'GPU_jobs': 0, 'physical_executions': 0}


def load_for_manipulation(cache_reference, descriptor_reference, bounds_path=DEFAULT_BOUNDS):
    """Runtime-facing gate: exact descriptor/scale/gripper binding is mandatory."""
    from resimpair_grasp.grasps import load
    descriptor_path = resolve(descriptor_reference)
    d = read(descriptor_path)
    if d.get('semantic_role') != 'manipulation' or not d.get('roles', {}).get('rigid_manipulation_candidate'):
        raise ValueError('static_asset_cannot_be_dynamic_grasp_payload')
    # The established loader rejects even geometrically similar but unbound
    # descriptors, so no new scale can inherit a parent's qualification.
    cache = load(cache_reference, descriptor_reference)
    report = audit_variant(descriptor_path, bounds_path)
    if not report['simulation_usage']['grasp_target_allowed']:
        raise ValueError('numeric_role_disallows_grasp_payload')
    return cache, report


def materialize(descriptor_path, policy_path, output, bounds_path=DEFAULT_BOUNDS):
    """Actual v2 entry: preflight -> retained geometry -> checked descriptors."""
    output = Path(output)
    if output.exists(): raise ValueError('preserve_existing_v2_variants')
    proof = preflight(descriptor_path, policy_path, bounds_path)
    # All planned variants must pass before writing any geometry.
    write(output / 'numeric_preflight.json', proof)
    policy = _legacy_policy(read(policy_path))
    staged_policy = output / 'geometry_policy.json'; write(staged_policy, policy)
    result = retained.materialize(descriptor_path, staged_policy, output / 'geometry')
    entries = []
    for row in result['variants']:
        source = resolve(row['descriptor']); d = read(source)
        audit = audit_variant(source, bounds_path)
        audit_path = output / row['variant_id'] / 'numeric_qualification.json'; write(audit_path, audit)
        final = copy.deepcopy(d)
        final.update(generator=binding(__file__), metric_variant_version=VERSION,
            numeric_qualification=binding(audit_path), numeric_bounds=binding(bounds_path),
            simulation_usage=audit['simulation_usage'], original_metric_policy=binding(policy_path),
            geometry_variant_key=digest({'visual': d['visual'], 'collisions': d['collisions'],
                'source_to_asset': d['source_to_asset'], 'dimensions_m': d['dimensions_m']}))
        # This metadata gate is not a native-mass, stable-pose or dynamic pass.
        final['qualification']['numeric_consistency'] = 'passed'
        destination = output / row['variant_id'] / 'descriptor.json'; write(destination, final)
        entries.append({**row, 'descriptor': binding(destination), 'numeric_qualification': binding(audit_path),
                        'simulation_usage': audit['simulation_usage']})
    terminal = {**result, 'schema': VERSION, 'variants': entries, 'numeric_bounds': binding(bounds_path),
                'status': 'numerically_bounded_pending_native_and_task_qualification'}
    write(output / 'catalog.json', {'schema': 'resimpair_mesh_catalog_v1', 'assets': [r['descriptor'] for r in entries],
        'qualification': terminal['status'], 'failures': []})
    write(output / 'terminal.json', terminal)
    return terminal


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['audit', 'materialize'])
    p.add_argument('--descriptor', type=Path, required=True); p.add_argument('--policy', type=Path)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--bounds', type=Path, default=DEFAULT_BOUNDS)
    args = p.parse_args()
    if args.command == 'audit': write(args.output, audit_variant(args.descriptor, args.bounds))
    else: print(materialize(args.descriptor, args.policy, args.output, args.bounds)['status'])
