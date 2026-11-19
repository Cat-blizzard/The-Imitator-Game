"""Publish an optional stable consumer of final poses and strict containers.

Source data, watched snapshots, cache bytes and split labels stay untouched.
Both qualification reports are standard immutable bindings and qualify only
consumer filtering/provenance, not semantic truth or physical task success.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path

from resimpair_grasp.common import binding, digest, read, resolve, write
from .semantic_scale_consumer_v1 import filter_bundle


def publish(bundle, sidecar, containers, container_review, output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('preserve_existing_final_consumer')
    original = read(bundle); revision = read(sidecar)
    inventory = original['source_semantics_checked_against_current_policy']['source_state_index']
    if binding(resolve(inventory)) != binding(resolve(revision['original_inventory'])):
        raise ValueError('recompute_semantic_sidecar_for_changed_source_inventory')
    revision_by_source = {r['source_id']: r for r in revision['rows']}
    entries = read(resolve(original['native_cache_index']))['entries']
    strict = read(containers)
    original_container_refs = {r['sha256']: r for r in read(resolve(original['containers']))['assets']}
    for ref in strict['assets']:
        if ref['sha256'] not in original_container_refs or binding(resolve(ref)) != binding(resolve(original_container_refs[ref['sha256']])):
            raise ValueError('strict_container_is_not_original_bound_descriptor')
    review = read(container_review)
    output.mkdir(parents=True)
    write(output / 'protocol.json', {'schema': 'stable_source_and_container_consumer_v1',
        'implementation': binding(__file__), 'original_bundle': binding(bundle),
        'source_sidecar': binding(sidecar), 'strict_container_catalog': binding(containers),
        'independent_container_review': binding(container_review),
        'old_snapshots_jobs_caches_sources_and_splits_unchanged': True,
        'physical_executions': 0, 'GPU_jobs': 0, 'API_calls': 0})
    final_audit = []
    for entry in entries:
        row = revision_by_source[entry['source_id']]
        d = read(resolve(entry['asset']))
        if d['source_id'] != row['source_id'] or d['partition'] != row['partition']:
            raise ValueError('final_cache_source_or_partition_conflict')
        final_audit.append({'source_id': row['source_id'], 'partition': row['partition'],
            'variant_id': entry['variant_id'], 'descriptor': entry['asset'], 'cache': entry['cache'],
            'retained_pose_count': entry['retained'], 'dimensions_m': d['dimensions_m'], 'mass_kg': d['mass_kg'],
            'size_provenance': d.get('size_provenance'), 'revision': row['revision'],
            'old_semantic_admission': entry['semantic_scale_admission'],
            'raw_source_sha256': row['raw_source_sha256']})
    write(output / 'final_admitted_audit.json', {'original_bundle': binding(bundle),
        'rows': final_audit, 'all_final_admitted_entries_checked': True,
        'source_ids_and_partitions_preserved': True})
    filtered_summary = filter_bundle(bundle, sidecar, output / 'semantic_filter')
    result = read(output / 'semantic_filter/BUNDLE.json')
    retained = read(resolve(result['native_cache_index']))['entries']
    original_entry_hashes = {digest(e) for e in entries}
    assert all(digest(e) in original_entry_hashes for e in retained)
    write(output / 'semantic_consumer_report.json', {'schema': 'source_role_scale_consumer_filter_v1',
        'consumer_qualified': True, 'qualification_scope': 'metadata hold policy and immutable source/cache subset integrity only',
        'semantic_truth_verified': False, 'physical_task_success_inferred': False,
        'original_bundle': binding(bundle), 'source_sidecar': binding(sidecar),
        'final_admitted_audit': binding(output / 'final_admitted_audit.json'),
        'held_original_entries': binding(output / 'semantic_filter/held_original_entries.json'),
        'native_cache_index': result['native_cache_index'], 'static_catalog': result['static_catalog'],
        'source_ids_split_labels_and_cache_entry_data_unchanged': True, **filtered_summary})
    write(output / 'container_consumer_report.json', {'schema': 'native_container_consumer_filter_v1',
        'consumer_qualified': True, 'qualification_scope': 'exact original descriptor subset approved by retained native support rule',
        'semantic_truth_verified': False, 'physical_task_success_inferred': False,
        'original_bundle': binding(bundle), 'catalog': binding(containers),
        'independent_review': binding(container_review), 'affordance_index': strict['affordance_index'],
        'native_support_variants': len(strict['assets']),
        'source_count': len({read(resolve(ref))['source_id'] for ref in strict['assets']}),
        'rejections': strict['rejected'], 'thresholds_changed': False})
    result['containers'] = binding(containers)
    result['consumer_semantic_revision'] = binding(output / 'semantic_consumer_report.json')
    result['container_affordance_consumer'] = binding(output / 'container_consumer_report.json')
    result['summary'] = {**result['summary'], 'container_variants': len(strict['assets']),
        'container_sources': len({read(resolve(ref))['source_id'] for ref in strict['assets']})}
    write(output / 'BUNDLE.json', result)
    summary = {'bundle': binding(output / 'BUNDLE.json'), 'strict_container_catalog': binding(containers),
        'original_sources': filtered_summary['original_sources'], 'original_variants': len(entries),
        'retained_sources': filtered_summary['retained_sources'], 'retained_variants': len(retained),
        'retained_poses': sum(e['retained'] for e in retained), 'held_sources': filtered_summary['held_sources'],
        'held_variants': filtered_summary['held_variants'],
        'held_small_static_descriptors': filtered_summary['held_small_static_descriptors'],
        'container_variants': result['summary']['container_variants'], 'container_sources': result['summary']['container_sources'],
        'final_held_families': dict(Counter(r['revision']['family'] for r in final_audit if r['revision']['automatic_manipulation_hold'])),
        'physical_executions': 0, 'GPU_jobs': 0, 'API_calls': 0}
    write(output / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['bundle', 'sidecar', 'containers', 'container-review', 'output']:
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args(); print(publish(a.bundle, a.sidecar, a.containers, a.container_review, a.output))
