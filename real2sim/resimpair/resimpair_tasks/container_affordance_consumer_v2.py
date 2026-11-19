"""Metadata-only continuation of the frozen native-support consumer v1.

The strict descriptor subset is identical. Preserve nested source/licensing
metadata and distinguish caller-supplied numbers from values actually derived
from immutable scene/task/native-collision bindings. Neither establishes visual
semantic containment or completed physical placement.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from pathlib import Path

import numpy as np
import trimesh

from resimpair_grasp.common import binding, digest, read, resolve, write
from resimpair_pickplace.requested_tasks import floor_fit, rotation
from corl2026_workshop.experiments.autonomous_pipeline_v4.simulation import targets
from . import container_affordance_consumer_v1 as retained

VERSION = 'measured_container_affordance_consumer_v2'
lookup = retained.lookup


def describe_goal(*args, **kwargs):
    result = retained.describe_goal(*args, **kwargs)
    if kwargs.get('nominal_payload_bottom_z_m') is not None:
        result['bottom_evidence'] = 'caller_supplied_nominal_goal_unverified'
    return result


def describe_bound_goal(index, scene_binding, task_binding, operation_id):
    """Derive nominal goal geometry from bound inputs; never claim contact."""
    scene = read(resolve(scene_binding)); task = read(resolve(task_binding))
    if not task['id'].startswith(scene['id'] + '_'):
        raise ValueError('task_does_not_belong_to_bound_scene_identity')
    matches = [o for o in task['operations'] if o['id'] == operation_id]
    if len(matches) != 1:
        raise ValueError('unique_bound_task_operation_required')
    operation = matches[0]
    obj = next(o for o in scene['objects'] if o['id'] == operation['object'])
    support = next(s for s in scene['supports'] if s['id'] == operation['target']['support'])
    affordance = lookup(index, support['container_descriptor'])
    descriptor = read(resolve(obj['asset_descriptor']))
    container = read(resolve(support['container_descriptor']))
    if descriptor['partition'] != scene['partition'] or container['partition'] != scene['partition']:
        raise ValueError('source_partition_not_preserved')
    if abs(support['top_z'] - support['position'][2] - container['cavity']['floor_z_m']) > 1e-9:
        raise ValueError('nominal_support_elevation_does_not_match_bound_descriptor')
    native = np.concatenate([trimesh.load(resolve(ref), force='mesh', process=False).vertices
                             for ref in descriptor['collisions']])
    goal = np.asarray(targets(scene, task)[obj['id']], float)
    height = float(native[:, 2].max() - native[:, 2].min())
    bottom = float(goal[2] + native[:, 2].min() - support['position'][2])
    report = describe_goal(affordance, payload_height_m=height, nominal_payload_bottom_z_m=bottom,
                           payload=obj['asset_descriptor'], task=task_binding)
    local_offset = rotation(support['yaw']).T @ np.asarray(operation['target']['offset_xy'])
    local_offset -= np.asarray(container['cavity']['center_xy_m'])
    visual = trimesh.load(resolve(descriptor['visual']), force='scene', process=False).to_geometry()
    fit = floor_fit(visual.vertices, container['cavity'], operation['target']['yaw'], support['yaw'], local_offset)
    report.update(bottom_evidence='derived_from_bound_scene_task_and_native_collision',
        operation_id=operation_id, object_id=obj['id'], container_id=support['id'],
        nominal_world_goal_m=goal.tolist(), task_specific_payload_xy_fit=fit,
        input_bindings={'scene': scene_binding, 'task': task_binding, 'payload': obj['asset_descriptor'],
            'payload_native_collisions': descriptor['collisions'], 'payload_visual': descriptor['visual'],
            'container': support['container_descriptor'], 'affordance_index': binding(index)},
        derivation_implementation=binding(__file__))
    if fit['passed']:
        report['place_relation']['missing_evidence'] = [s for s in report['place_relation']['missing_evidence']
            if s != 'task_specific_payload_xy_fit']
    # No positive containment label follows from these nominal calculations.
    return report


def build(previous, output):
    previous = Path(previous); output = Path(output).resolve()
    if output.exists():
        raise ValueError('preserve_existing_metadata_outputs')
    catalog = read(previous / 'sampler_catalog.json')
    index = read(resolve(catalog['affordance_index']))
    rows = deepcopy(index['rows']); provenance_recovered = 0
    for row in rows:
        d = read(resolve(row['descriptor'])); provenance = deepcopy(d.get('provenance', {}))
        row['source_provenance'] = provenance
        row['license'] = d.get('license') or provenance.get('license')
        row['license_metadata_path'] = 'license' if d.get('license') else 'provenance.license' if provenance.get('license') else None
        if d.get('source') is not None:
            row['original_source'] = d['source']; row['original_source_kind'] = 'top_level_source_binding_or_metadata'
        else:
            row['original_source'] = provenance or None
            row['original_source_kind'] = 'nested_provenance_metadata' if provenance else 'unknown'
            provenance_recovered += bool(provenance)
    output.mkdir(parents=True)
    write(output / 'protocol.json', {'schema': VERSION, 'implementation': binding(__file__),
        'retained_implementation': binding(retained.__file__), 'previous_catalog': binding(previous / 'sampler_catalog.json'),
        'metadata_changes_only': ['read nested source/license provenance', 'verify nominal goal derivation or explicitly label caller supplied'],
        'strict_subset_qualification_unchanged': True, 'physical_executions': 0, 'GPU_jobs': 0, 'API_calls': 0})
    index.update(schema=VERSION, rows=rows, parent_index=catalog['affordance_index'])
    write(output / 'measured_affordance_index.json', index)
    new_catalog = deepcopy(catalog)
    new_catalog.update(affordance_index=binding(output / 'measured_affordance_index.json'),
                       metadata_parent=binding(previous / 'sampler_catalog.json'))
    write(output / 'sampler_catalog.json', new_catalog)
    assert new_catalog['assets'] == catalog['assets']
    write(output / 'subset_proof.json', {'previous_proof': binding(previous / 'subset_proof.json'),
        'previous_catalog': binding(previous / 'sampler_catalog.json'), 'catalog': binding(output / 'sampler_catalog.json'),
        'descriptor_lists_identical_including_order': True, 'descriptor_list_sha256': digest(catalog['assets']),
        'all_original_source_ids_variants_partitions_and_descriptor_bytes_preserved': True,
        'rejections_unchanged': catalog['rejected']})
    summary = read(previous / 'summary.json')
    summary.update(sampler_catalog=binding(output / 'sampler_catalog.json'),
        affordance_index=binding(output / 'measured_affordance_index.json'),
        nested_provenance_recovered_variants=provenance_recovered,
        license_metadata_present_variants=sum(bool(r['license']) for r in rows),
        descriptor_subset_identical_to_v1=True)
    write(output / 'summary.json', summary)
    columns = ['source_id', 'variant_id', 'partition', 'name_from_source_not_semantic_truth',
        'consumer_admitted', 'descriptive_geometry_bin', 'native_body_height_m', 'floor_z_m',
        'floor_relative_body_height', 'cavity_depth_m', 'flat_floor_radius_m',
        'rim_to_native_body_top_margin_m', 'flat_floor_to_minimum_wall_radius_margin_m',
        'geometry_candidate', 'semantic_place_relation', 'license', 'license_metadata_path']
    with (output / 'measured_affordances.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader()
        for row in rows:
            values = {k: row.get(k, row['measurements'].get(k)) for k in columns}
            values.update(geometry_candidate=row['place_relation']['geometry_candidate'],
                          semantic_place_relation=row['place_relation']['value'])
            writer.writerow(values)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); print(build(args.previous, args.output))
