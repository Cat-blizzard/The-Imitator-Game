"""Metric factory with immutable, exact container variants selected upstream.

The retained geometry compiler and its thresholds are unchanged. A source may
have several offline variants, but each scene binds one descriptor per source
for objects and containers. These artifacts are requests, not physical results.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import time

from resimpair_grasp.common import binding, read, resolve, write
from resimpair_pickplace import requested_tasks as retained
from . import metric_task_factory_v2 as previous

VERSION = 'source_preserving_metric_task_factory_v3'
select_variants = previous.select_variants
static_source_catalog = previous.static_source_catalog


def request_schema(value):
    if value.get('schema') != VERSION:
        raise ValueError('metric_factory_v3_schema_required')
    legacy = deepcopy(value)
    legacy['schema'] = previous.VERSION
    for scene in legacy.get('scenes', []):
        scene.pop('container_variant_choices', None)
    previous.request_schema(legacy)
    for scene in value['scenes']:
        if scene.get('containers') and not isinstance(scene.get('container_variant_choices'), dict):
            raise ValueError('explicit_container_variant_choices_required')
    return value


def select_container_variants(container_entries, requests, explicit, partition):
    """Bind original source -> exact descriptor; never pick a smaller variant.

    Multiple instances of a container source share that bound variant. Different
    variants of a source in one scene are intentionally not part of this schema.
    """
    sources = {c.get('source_id') for c in requests}
    if None in sources or set(explicit) != sources:
        raise ValueError('exact_container_choice_source_coverage_required')
    members = {(str(resolve(ref)), ref['sha256']) for ref in container_entries}
    selected = []
    for source in sorted(sources):
        ref = explicit[source]
        if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}:
            raise ValueError('container_descriptor_binding_required')
        path = resolve(ref)
        if (str(path), ref['sha256']) not in members:
            raise ValueError('container_variant_not_in_immutable_bundle')
        descriptor = read(path)
        if descriptor['source_id'] != source:
            raise ValueError('container_source_identity_mismatch')
        if descriptor['partition'] != partition:
            raise ValueError('container_source_partition_mismatch')
        if not descriptor.get('roles', {}).get('stationary_container'):
            raise ValueError('stationary_container_role_required')
        selected.append(ref)
    return selected


def compile_exact_scene(spec, entries, container_entries, seed):
    """Preflight and final factory share this exact-container compile path."""
    spec = deepcopy(spec)
    spec.pop('variant_choices', None)
    choices = spec.pop('container_variant_choices', {})
    partitions = {read(resolve(e['asset']))['partition'] for e in entries}
    if len(partitions) != 1:
        raise ValueError('one_partition_per_scene_required')
    selected = select_container_variants(container_entries, spec.get('containers', []),
                                        choices, next(iter(partitions)))
    scene, tasks, checks = retained.compile_scene(spec, entries, selected, seed)
    by_id = {c['id']: c for c in spec.get('containers', [])}
    actual = []
    for support in scene['supports']:
        source = by_id[support['id']]['source_id']
        expected = choices[source]
        if binding(resolve(support['container_descriptor'])) != binding(resolve(expected)):
            raise ValueError('compiled_container_variant_changed')
        actual.append({'container_id': support['id'], 'source_id': source,
                       'requested': expected, 'compiled': support['container_descriptor'],
                       'exact_match': True})
    checks['exact_container_bindings'] = actual
    return scene, tasks, checks


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
        'retained_factory': binding(previous.__file__), 'bundle': binding(bundle),
        'requests': binding(requests), 'seed': seed, 'compiler': binding(retained.__file__),
        'geometry_only': True, 'container_selection': 'exact bound descriptor per original source',
        'physical_attempts': 0, 'API_calls': 0, 'new_grasp_generation': 0})
    began = time.monotonic(); rows = []; cases = []
    for number, original_scene in enumerate(original['scenes']):
        directory = output / 'scenes' / original_scene['id']
        spec = deepcopy(original_scene)
        choices = spec.get('variant_choices', {})
        write(directory / 'original_request.json', original_scene)
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
                'containers': spec.get('container_variant_choices', {}),
                'original_source_identity_preserved': True, 'runtime_rescaling': False})
            scene, tasks, checks = compile_exact_scene(spec, entries, containers, seed + number)
            scene['metric_variants'] = {o['id']: {'source_id': e['source_id'],
                'variant_id': e['variant_id'], 'asset': e['asset'], 'cache': e['cache'],
                'native_support': e['native_support'], 'native_properties': e['native_properties']}
                for o, e in zip(spec['objects'], entries)}
            scene['randomization']['metric_bundle'] = binding(bundle)
            scene['randomization']['source_split_preserved'] = True
            scene['randomization']['exact_container_bindings'] = spec.get('container_variant_choices', {})
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
