"""Derive a candidate task bundle after exact final-collider reload checks.

Keep the original producer publication, including every failure. Membership
changes only for the active grasp pool; background and static catalogs retain
their original evidence and still require scene-level native verification.
"""
from collections import Counter
from pathlib import Path

from resimpair_grasp.common import binding, digest, read, resolve, write
from resimpair_grasp import evaluator
from resimpair_tasks.metric_population_sampler_v4 import admitted_pool
from . import native_reload_preflight_v1 as reload_check


def publish(publication, reload_report, output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('preserve_previous_membership_and_failures')
    result = read(publication)
    if result.get('publication_integrity_passed') is not True:
        raise ValueError('verified_native_producer_publication_required')
    parent = resolve(result['bundle']); b = read(parent)
    entries = read(resolve(b['native_cache_index']))['entries']
    proof = read(reload_report); p = read(resolve(proof['protocol']))
    refs = [e['asset'] for e in entries]
    if (p['implementation'] != binding(reload_check.__file__)
            or p['retained_evaluator'] != binding(evaluator.__file__)
            or p['asset_descriptors'] != refs or proof['checked'] != len(entries)
            or len(proof['rows']) != len(entries)):
        raise ValueError('exact_current_reload_protocol_and_membership_required')
    kept = []; excluded = []
    for entry, row in zip(entries, proof['rows']):
        if entry['asset'] != row['asset'] or entry['source_id'] != row['source_id']:
            raise ValueError('reload_source_or_geometry_binding_mismatch')
        if row['passed']:
            native = resolve(row['actual_native_geometry'])
            current = evaluator.geometry(native, {'objects': [{'id': 'asset', 'asset_descriptor': entry['asset'], 'kind': 'movable'}]})
            if current != row['original_geometry_evaluator']:
                raise ValueError('retained_geometry_verdict_changed')
            kept.append(entry)
        else:
            excluded.append(row)
    if not kept:
        raise ValueError('no_final_reload_qualified_active_sources')
    allowed = {e['source_id'] for e in kept}
    states = [s for s in read(resolve(b['source_state_index']))['entries'] if s['source_id'] in allowed]
    summary = {**b['summary'], 'admitted_grasp_sources': len(allowed), 'admitted_grasp_variants': len(kept),
        'poses': sum(e['retained'] for e in kept),
        'new_raw_sources': sum(e['ingestion_group'] == 'new_raw' for e in kept),
        'size_corrected_sources': sum(e['ingestion_group'] == 'size_corrected' for e in kept),
        'partition_source_counts': dict(Counter(s['partition'] for s in states)),
        'final_reload_excluded_sources': len(excluded), 'grasp_or_task_success_inferred': False}
    write(output / 'native_cache_index.json', {'schema': 'resimpair_grasp_cache_index_v1', 'entries': kept})
    write(output / 'source_state_index.json', {'schema': 'raw_candidate_source_states_v1', 'entries': states})
    write(output / 'EXCLUDED.json', {'rows': excluded, 'original_publication': binding(publication),
        'reload_report': binding(reload_report), 'reason': 'original geometry rule; requires new geometry and cache, never runtime scaling'})
    b = {**b, 'summary': summary, 'native_cache_index': binding(output / 'native_cache_index.json'),
        'source_state_index': binding(output / 'source_state_index.json'),
        'source_semantics_checked_against_current_policy': {**b['source_semantics_checked_against_current_policy'],
            'source_state_index': binding(output / 'source_state_index.json')},
        'final_native_reload': binding(reload_report), 'parent_producer_publication': binding(publication),
        'implementation': binding(__file__),
        'qualification_scope': 'active pool final native reload only; complete scene/trajectory/visual quality remain unmeasured'}
    write(output / 'BUNDLE.json', b)
    _, actual, sources, bins = admitted_pool(output / 'BUNDLE.json')
    if digest(actual) != digest(kept) or set(sources) != allowed:
        raise ValueError('task_consumer_did_not_load_exact_filtered_membership')
    final = {**summary, 'bundle': binding(output / 'BUNDLE.json'), 'parent_publication': binding(publication),
        'reload_report': binding(reload_report), 'actual_task_consumer_reload_passed': True,
        'publication_integrity_passed': True, 'new_pose_optimization': 0, 'API_calls': 0,
        'default_promoted': False, 'static_background_geometry_scope': 'unchanged; full-scene checks still required'}
    write(output / 'RESULT.json', final)
    return final
