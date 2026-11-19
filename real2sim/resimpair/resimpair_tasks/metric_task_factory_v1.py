"""Compile source-preserving, exact-metric pick/place requests without an agent.

One source may have several offline size variants. Select exactly one variant
per source per scene; never change source identity or rescale at execution.
The output is a frozen request/cohort, not a claim of physical success.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import time

import numpy as np

from resimpair_grasp.common import binding, digest, read, resolve, write
from resimpair_pickplace import requested_tasks as retained

VERSION = 'source_preserving_metric_task_factory_v1'


def select_variants(entries, requested_sources, seed, explicit=None):
    """Source-level sampling stays independent of the number of cached scales."""
    from resimpair_grasp.grasps import load
    explicit = explicit or {}
    if len(requested_sources) != len(set(requested_sources)):
        raise ValueError('one_instance_per_source_in_this_factory')
    if not set(explicit) <= set(requested_sources):
        raise ValueError('variant_choice_for_unrequested_source')
    grouped = defaultdict(list)
    partitions = defaultdict(set)
    for entry in entries:
        if entry['source_id'] not in requested_sources:
            continue
        d = read(resolve(entry['asset']))
        if d['source_id'] != entry['source_id']:
            raise ValueError('source_identity_mismatch')
        partitions[d['source_id']].add(d['partition'])
        if not d.get('roles', {}).get('rigid_manipulation_candidate'):
            raise ValueError('source_not_a_manipulation_role')
        if entry.get('variant_id') != d.get('variant_id'):
            raise ValueError('variant_descriptor_identity_mismatch')
        support = read(resolve(entry['native_support']))
        if support.get('status') != 'pass' or support.get('dynamic_settling') != 'pass':
            raise ValueError('settled_native_support_required')
        cache = load(entry['cache'], entry['asset'])
        if not cache['candidates'] or cache['signature'].get('geometry_scope') != 'native':
            raise ValueError('nonempty_native_qualified_grasp_required')
        grouped[d['source_id']].append(entry)
    selected = []
    for source in requested_sources:
        if len(partitions[source]) != 1:
            raise ValueError('missing_source_or_source_partition_conflict:' + source)
        choices = grouped[source]
        if source in explicit:
            choices = [e for e in choices if e['variant_id'] == explicit[source]]
        if not choices:
            raise ValueError('requested_source_variant_not_qualified:' + source)
        selected.append(min(choices, key=lambda e: digest([seed, source, e['variant_id']])))
    return selected


def request_schema(value):
    if value.get('schema') != VERSION or not 1 <= len(value.get('scenes', [])) <= 1000:
        raise ValueError('bounded_metric_scene_request_required')
    ids = set()
    for scene in value['scenes']:
        retained.fields(scene, ['id', 'objects', 'tasks'],
                        ['containers', 'distractors', 'variant_choices'])
        sid = retained.identity(scene['id'])
        if sid in ids:
            raise ValueError('duplicate_scene_id')
        ids.add(sid)
        if not 1 <= len(scene['objects']) <= 8 or not 1 <= len(scene['tasks']) <= 8:
            raise ValueError('one_to_eight_objects_and_tasks_required')
        if len({t['id'] for t in scene['tasks']}) != len(scene['tasks']):
            raise ValueError('duplicate_task_id')
        for task in scene['tasks']:
            retained.fields(task, ['id', 'placements'])
            retained.identity(task['id'])
    return value


def static_source_catalog(source, output):
    """One real descriptor per source, preserving role, size and source split."""
    groups = defaultdict(list)
    for ref in read(source)['assets']:
        d = read(resolve(ref))
        if d.get('roles', {}).get('static_distractor') and max(d['dimensions_m']) <= .18:
            groups[d['source_id']].append((ref, d))
    refs, sources = [], {}
    for sid, choices in sorted(groups.items()):
        if len({d['partition'] for _, d in choices}) != 1:
            raise ValueError('static_source_partition_conflict')
        ref, d = min(choices, key=lambda x: x[0]['sha256'])
        refs.append(ref)
        sources[sid] = {'descriptors': [ref], 'partitions': [d['partition']],
                        'allowed_as_distractor': True}
    catalog = Path(output) / 'static_catalog.json'
    role_index = Path(output) / 'source_role_index.json'
    write(catalog, {'schema': 'resimpair_mesh_catalog_v1', 'assets': refs})
    write(role_index, {'schema': 'source_role_catalog_v1', 'catalog': binding(catalog),
                       'sources': sources, 'parent_catalog': binding(source)})
    return binding(catalog), binding(role_index)


def compile_requests(bundle, requests, output, seed):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('preserve_existing_factory_outputs')
    original = request_schema(read(requests))
    b = read(bundle)
    index = read(resolve(b['native_cache_index']))
    containers = read(resolve(b['containers']))['assets']
    output.mkdir(parents=True)
    static = static_source_catalog(resolve(b['static_catalog']), output)
    write(output / 'protocol.json', {'schema': VERSION, 'implementation': binding(__file__),
        'bundle': binding(bundle), 'requests': binding(requests), 'seed': seed,
        'compiler': binding(retained.__file__), 'geometry_only': True,
        'variant_sampling': 'one bound variant per original source per scene',
        'physical_attempts': 0, 'API_calls': 0, 'new_grasp_generation': 0})
    began = time.monotonic(); rows = []; cases = []
    for number, original_scene in enumerate(original['scenes']):
        directory = output / 'scenes' / original_scene['id']
        spec = deepcopy(original_scene)
        choices = spec.pop('variant_choices', {})
        write(directory / 'source_request.json', spec)
        record = {'scene_id': spec['id'], 'requested_object_count': len(spec['objects']),
                  'requested_task_count': len(spec['tasks']), 'compiled': False}
        try:
            entries = select_variants(index['entries'], [o['source_id'] for o in spec['objects']],
                                      seed + number, choices)
            spec['distractors'] = {**spec.get('distractors', {}),
                'catalog': static[0], 'source_role_index': static[1],
                'exclude_source_ids': sorted({o['source_id'] for o in spec['objects']})}
            write(directory / 'source_request.json', spec)
            write(directory / 'selected_variants.json', {'entries': entries,
                'original_source_identity_preserved': True, 'runtime_rescaling': False})
            scene, tasks, checks = retained.compile_scene(spec, entries, containers, seed + number)
            scene['metric_variants'] = {o['id']: {'source_id': e['source_id'],
                'variant_id': e['variant_id'], 'asset': e['asset'], 'cache': e['cache'],
                'native_support': e['native_support'], 'native_properties': e['native_properties']}
                for o, e in zip(spec['objects'], entries)}
            scene['randomization']['metric_bundle'] = binding(bundle)
            scene['randomization']['source_split_preserved'] = True
            write(directory / 'scene.json', scene)
            write(directory / 'geometry_checks.json', checks)
            for task in tasks:
                task['required_operation_precedence'] = []
                task['placement_contract'] = {'schema': 'supported_or_low_drop_v1',
                    'maximum_release_height_over_goal_m': .15, 'xy_tolerance_m': .015,
                    'continuous_hold_until_release': True}
                for op in task['operations']:
                    container = op['target']['support'] != 'table'
                    op['release'] = {'mode': 'drop_from_above' if container else 'supported_place',
                        'allow_drop': container, 'fragile': False,
                        'material_status': 'rigid_nonbreakable_simulation_assumption_not_real_item_claim',
                        'bottom_above_rim_m': .03, 'hand_above_rim_m': .01, 'maximum_fall_m': .15}
                p = directory / 'tasks' / (task['requested_task_id'] + '.json')
                write(p, task)
                cases.append({'id': task['id'], 'execution_id': task['id'],
                    'partition': scene['partition'], 'source_ids': scene['split_group'],
                    'split_group': scene['split_group'], 'source_coverage': 'synthetic_only',
                    'source_request': binding(directory / 'source_request.json'),
                    'requested_task_id': task['requested_task_id'], 'scene': binding(directory / 'scene.json'),
                    'task': binding(p), 'eligible': True, 'method': 'offline_metric_native_cudex',
                    'metric_variants': binding(directory / 'selected_variants.json'),
                    'physical_status': 'not_executed'})
            record.update(compiled=True, scene=binding(directory / 'scene.json'),
                          selected_variants=binding(directory / 'selected_variants.json'))
        except (ValueError, RuntimeError, KeyError) as error:
            record.update(reason=getattr(error, 'code', type(error).__name__),
                          details=getattr(error, 'details', {'message': str(error)}))
            write(directory / 'rejection.json', record)
        rows.append(record)
    write(output / 'scenes.json', {'scenes': rows})
    write(output / 'cohort.json', {'cases': cases, 'original_rejections_retained': True})
    summary = {'schema': VERSION, 'requested_scenes': len(rows),
        'compiled_scenes': sum(r['compiled'] for r in rows), 'compiled_tasks': len(cases),
        'physical_attempts': 0, 'API_calls': 0, 'seconds': time.monotonic() - began}
    write(output / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'requests', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261030)
    args = parser.parse_args()
    print(compile_requests(args.bundle, args.requests, args.output, args.seed))
