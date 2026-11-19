"""Bound container consumer subset and measured affordances, without relabeling.

Input: an immutable original sampler catalog and its independent full census.
Output: the exact original descriptors that passed retained native support,
all measured rows (including rejections), and conservative relation metadata.
No mesh repair, resizing, new acceptance threshold, physical run or API call.

The index separates geometry bins from semantic place relations. A measured
opening or a title alone never authorizes ``into_open_container``. Consumers
may use geometry_fallback for geometry planning, but R2H relation assertions
remain unknown until a separately bound task/visual check exists.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from copy import deepcopy
import math
from pathlib import Path

from resimpair_grasp.common import binding, read, resolve, write

VERSION = 'measured_container_affordance_consumer_v1'
CANDIDATES = {
    'shallow_tray_like_cavity': 'on_tray',
    'elevated_shallow_recess_or_support_surface': 'on_support_surface',
    'deeper_open_cavity_geometry_candidate': 'into_open_container',
    'intermediate_cavity_geometry_unknown_semantics': 'unknown',
}


def place_relation(measured_row):
    support = measured_row['retained_oriented_floor_support']
    candidate = CANDIDATES.get(measured_row['descriptive_geometry_bin'], 'unknown')
    passed = support.get('passed') is True
    return {'value': 'unknown',
        'geometry_candidate': candidate if passed else 'unknown',
        'geometry_fallback': 'on_support_surface' if passed else None,
        'evidence_level': 'independent_sampled_native_support_and_dimensions' if passed else 'native_support_rejected',
        'semantic_status': 'unknown', 'assertable_by_R2H': False,
        'into_open_container_allowed': False,
        'missing_evidence': (['task_specific_payload_fit_and_nominal_rim_relationship',
                             'task_specific_opening_or_swept_payload_clearance',
                             'bound_visual_or_other_independent_semantic_confirmation'] if passed
                            else ['consistent_upward_native_support']),
        'scope': 'Geometry candidate/fallback is not semantic ground truth or completed placement evidence.'}


def measured_affordance(row, descriptor):
    numbers = [row[k] for k in ['native_body_height_m', 'native_min_z_m', 'native_max_z_m',
        'floor_z_m', 'rim_z_m', 'cavity_depth_m', 'flat_floor_radius_m']]
    if not all(math.isfinite(x) for x in numbers) or row['native_body_height_m'] <= 0:
        raise ValueError('finite_measured_container_geometry_required')
    if abs(row['cavity_depth_m'] - (row['rim_z_m'] - row['floor_z_m'])) > 1e-9:
        raise ValueError('census_depth_inconsistent')
    if any(abs(float(row[k]) - float(descriptor['cavity'][dkey])) > 1e-9
           for k, dkey in [('floor_z_m', 'floor_z_m'), ('rim_z_m', 'rim_z_m'),
                          ('flat_floor_radius_m', 'flat_floor_radius_m')]):
        raise ValueError('census_descriptor_geometry_mismatch')
    continuous = {k: row[k] for k in ['native_body_height_m', 'native_min_z_m', 'native_max_z_m',
        'floor_z_m', 'rim_z_m', 'cavity_depth_m', 'floor_relative_body_height',
        'rim_relative_body_height', 'depth_relative_body_height', 'flat_floor_radius_m',
        'depth_to_floor_diameter_ratio', 'declared_and_actual_collision_bounds_max_difference_m']}
    continuous.update(
        floor_elevation_above_native_base_m=row['floor_z_m'] - row['native_min_z_m'],
        rim_to_native_body_top_margin_m=row['native_max_z_m'] - row['rim_z_m'],
        minimum_wall_radius_m=descriptor['cavity'].get('minimum_wall_radius_m'),
        flat_floor_to_minimum_wall_radius_margin_m=(descriptor['cavity']['minimum_wall_radius_m'] - row['flat_floor_radius_m']
            if 'minimum_wall_radius_m' in descriptor['cavity'] else None),
        payload_top_below_rim_margin_m=None,
        payload_margin_status='unknown_without_bound_payload_and_nominal_goal')
    return {'descriptor': row['descriptor'], 'source_id': row['source_id'], 'variant_id': row['variant_id'],
        'name_from_source_not_semantic_truth': descriptor['name'], 'partition': row['partition'],
        'dimensions_m': descriptor['dimensions_m'], 'metric_size_provenance': descriptor.get('metric_size'),
        'license': descriptor.get('license'), 'original_source': descriptor.get('source'),
        'native_visual': descriptor['visual'], 'native_collision': descriptor['collision'],
        'measurements': continuous, 'descriptive_geometry_bin': row['descriptive_geometry_bin'],
        'geometry_bin_is_semantic_label': False, 'retained_oriented_floor_support': deepcopy(row['retained_oriented_floor_support']),
        'consumer_admitted': row['retained_oriented_floor_support'].get('passed') is True,
        'place_relation': place_relation(row), 'physical_status': 'unmeasured_by_this_index'}


def describe_goal(affordance_row, *, payload_height_m, nominal_payload_bottom_z_m=None,
                  payload=None, task=None):
    """Task/R2H sidecar; rim relationship alone never asserts containment.

    Coordinates are local to the unchanged stationary container descriptor.
    If no bottom is given, the descriptor's sampled floor is a NOMINAL assumed
    bottom. This is not a settled pose, grasp success or dynamic contact label.
    """
    height = float(payload_height_m)
    if not math.isfinite(height) or height <= 0:
        raise ValueError('finite_positive_payload_height_required')
    if payload is not None:
        resolve(payload)
    if task is not None:
        resolve(task)
    m = affordance_row['measurements']
    bottom = float(m['floor_z_m'] if nominal_payload_bottom_z_m is None else nominal_payload_bottom_z_m)
    if not math.isfinite(bottom):
        raise ValueError('finite_nominal_payload_bottom_required')
    margin = m['rim_z_m'] - (bottom + height)
    relation = deepcopy(affordance_row['place_relation'])
    relation['missing_evidence'] = [s for s in relation['missing_evidence']
        if s != 'task_specific_payload_fit_and_nominal_rim_relationship']
    if affordance_row['consumer_admitted']:
        relation['missing_evidence'].append('task_specific_payload_xy_fit')
    return {'container': affordance_row['descriptor'], 'source_id': affordance_row['source_id'],
        'partition': affordance_row['partition'], 'descriptive_geometry_bin': affordance_row['descriptive_geometry_bin'],
        'payload': payload, 'task': task, 'payload_native_height_m': height,
        'nominal_payload_bottom_z_m': bottom, 'nominal_payload_top_z_m': bottom + height,
        'payload_top_below_rim_margin_m': margin,
        'whole_nominal_payload_below_rim': margin >= 0,
        'nominal_fraction_below_rim': max(0., min(1., (m['rim_z_m'] - bottom) / height)),
        'bottom_evidence': 'descriptor_floor_assumption' if nominal_payload_bottom_z_m is None else 'caller_bound_nominal_goal',
        'place_relation': relation,
        'R2H_constraint': 'Do not claim fully inside or deep-container placement from name, color, or depth alone; use bound task/visual evidence.',
        'scope': 'Measured nominal geometry only; protrusion may be valid and is not itself a failure.'}


def lookup(index, descriptor):
    resolve(descriptor)
    value = read(index)
    found = [r for r in value['rows'] if r['descriptor']['sha256'] == descriptor['sha256']
             and resolve(r['descriptor']) == resolve(descriptor)]
    if len(found) != 1:
        raise ValueError('exact_container_descriptor_not_in_affordance_index')
    return found[0]


def build(catalog, census, output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('preserve_existing_consumer_evidence')
    parent = read(catalog); review = read(census); protocol = read(resolve(review['protocol']))
    if binding(resolve(protocol['catalog'])) != binding(catalog):
        raise ValueError('independent_census_catalog_binding_mismatch')
    resolve(protocol['reviewer'])
    refs = parent['assets']
    by_key = {r['descriptor']['sha256']: r for r in review['rows']}
    if len(by_key) != len(review['rows']) or len({r['sha256'] for r in refs}) != len(refs):
        raise ValueError('duplicate_descriptor_in_catalog_or_census')
    if set(by_key) != {r['sha256'] for r in refs}:
        raise ValueError('full_original_catalog_census_required')
    rows = []; qualified = []; rejected = []; partitions = defaultdict(set)
    proof = []
    for ref in refs:
        d = read(resolve(ref)); row = by_key[ref['sha256']]
        if binding(resolve(row['descriptor'])) != binding(resolve(ref)):
            raise ValueError('descriptor_path_binding_mismatch')
        for key in ['source_id', 'partition']:
            if row[key] != d[key]:
                raise ValueError('original_source_or_partition_changed')
        if row['variant_id'] != d.get('variant_id', d['id']):
            raise ValueError('original_variant_id_changed')
        for key in ['visual', 'collision']:
            resolve(d[key])
        support = row['retained_oriented_floor_support']
        if support['version'] != 'native_oriented_support_preflight_v1' or support['sample_count'] != 97:
            raise ValueError('retained_native97_support_rule_required')
        if type(support['passed']) is not bool:
            raise ValueError('explicit_independent_support_decision_required')
        measured = measured_affordance(row, d); rows.append(measured)
        partitions[d['source_id']].add(d['partition'])
        proof.append({'descriptor': ref, 'source_id': d['source_id'], 'variant_id': row['variant_id'],
                      'partition': d['partition'], 'original_bytes_retained': True,
                      'native_visual_and_collision_hashes_resolved': True,
                      'strict_subset_admitted': measured['consumer_admitted']})
        if support['passed']:
            qualified.append(ref)
        else:
            rejected.append({'descriptor': ref, 'source_id': d['source_id'], 'variant_id': row['variant_id'],
                'partition': d['partition'], 'reason': support['reason'], 'retained_check': deepcopy(support),
                'descriptive_geometry_bin': row['descriptive_geometry_bin']})
    if any(len(v) != 1 for v in partitions.values()):
        raise ValueError('source_partition_conflict_not_admitted')
    output.mkdir(parents=True)
    write(output / 'protocol.json', {'schema': VERSION, 'implementation': binding(__file__),
        'original_catalog': binding(catalog), 'independent_census': binding(census),
        'independent_protocol': review['protocol'], 'strict_subset_rule': 'retained_oriented_floor_support.passed is true',
        'geometry_bins': protocol['descriptive_bins_not_acceptance_rules'],
        'geometry_bins_change_admission': False, 'existing_tasks_changed': False,
        'physical_executions': 0, 'GPU_jobs': 0, 'API_calls': 0})
    write(output / 'measured_affordance_index.json', {'schema': VERSION, 'original_catalog': binding(catalog),
        'independent_census': binding(census), 'rows': rows, 'all_original_variants_retained': True,
        'relation_policy': 'Geometry evidence alone does not authorize an into_open_container semantic assertion.'})
    write(output / 'rejections.json', {'schema': 'retained_container_native_support_rejections_v1',
        'independent_census': binding(census), 'rows': rejected, 'no_threshold_changes': True})
    write(output / 'sampler_catalog.json', {'schema': 'resimpair_mesh_catalog_v1', 'assets': qualified,
        'original_catalog': binding(catalog), 'independent_census': binding(census),
        'affordance_index': binding(output / 'measured_affordance_index.json'),
        'rejected': binding(output / 'rejections.json'), 'strict_subset_no_descriptor_edits': True,
        'semantic_relation_admission': 'unknown; geometry-qualified receiving surfaces, not all deep containers'})
    write(output / 'subset_proof.json', {'original_catalog': binding(catalog),
        'strict_subset': binding(output / 'sampler_catalog.json'), 'rows': proof,
        'all_original_source_ids_variants_partitions_and_descriptors_preserved': True})
    admitted = [r for r in rows if r['consumer_admitted']]
    summary = {'original_variants': len(rows), 'original_sources': len(partitions),
        'strict_subset_variants': len(qualified), 'strict_subset_sources': len({r['source_id'] for r in admitted}),
        'rejected_variants': len(rejected), 'rejected_sources_nonadditive': len({r['source_id'] for r in rejected}),
        'original_geometry_bins': dict(Counter(r['descriptive_geometry_bin'] for r in rows)),
        'admitted_geometry_bins': dict(Counter(r['descriptive_geometry_bin'] for r in admitted)),
        'admitted_partition_variants': dict(Counter(r['partition'] for r in admitted)),
        'confirmed_semantic_into_open_container_variants': 0,
        'sampler_catalog': binding(output / 'sampler_catalog.json'),
        'affordance_index': binding(output / 'measured_affordance_index.json'),
        'physical_executions': 0, 'GPU_jobs': 0, 'API_calls': 0}
    write(output / 'summary.json', summary)
    columns = ['source_id', 'variant_id', 'partition', 'name_from_source_not_semantic_truth',
               'consumer_admitted', 'descriptive_geometry_bin', 'native_body_height_m', 'floor_z_m',
               'floor_relative_body_height', 'cavity_depth_m', 'flat_floor_radius_m',
               'rim_to_native_body_top_margin_m', 'flat_floor_to_minimum_wall_radius_margin_m',
               'geometry_candidate', 'place_relation', 'support_rejection_reason']
    with (output / 'measured_affordances.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader()
        for row in rows:
            values = {k: row.get(k, row['measurements'].get(k)) for k in columns}
            values.update(geometry_candidate=row['place_relation']['geometry_candidate'],
                          place_relation=row['place_relation']['value'],
                          support_rejection_reason=row['retained_oriented_floor_support']['reason'])
            writer.writerow(values)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['catalog', 'census', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(build(args.catalog, args.census, args.output))
