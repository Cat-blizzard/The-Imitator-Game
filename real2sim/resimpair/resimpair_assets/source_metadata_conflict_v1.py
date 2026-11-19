"""Hold an engineered small object when its own product metadata conflicts.

This is a separate candidate-consumer revision. It changes no mesh, mass,
grasp, historical label or source split. Conflicting product/component identity
remains unknown instead of guessing a replacement size from the title.
"""
import argparse
from collections import Counter
import math
from pathlib import Path

from resimpair_grasp.common import binding, digest, read, resolve, write
from resimpair_tasks.metric_population_sampler_v4 import admitted_pool

VERSION = 'large_product_engineered_small_metadata_hold_v1'


def inspect(entry, state, job):
    source_id = entry['source_id']; raw = job['raw_row']
    if raw['id'] != source_id or state['source_id'] != source_id:
        raise ValueError('bound_original_source_metadata_required')
    descriptor = read(resolve(entry['asset']))
    result = {'source_id': source_id, 'asset': entry['asset'], 'hold': False,
        'reason': 'no_targeted_large_product_engineering_conflict',
        'source_size_confidence': state.get('size_confidence'), 'source_semantic_truth': 'unknown',
        'geometry_or_grasp_invalidated': False, 'replacement_size': None, 'evidence': []}
    if raw.get('source') != 'abo' or not str(state.get('size_confidence', '')).startswith('engineering_'):
        return result
    longest = float(entry['semantic_scale_admission']['actual_longest_obb_m'])
    if not math.isfinite(longest) or longest <= 0:
        raise ValueError('finite_measured_candidate_geometry_required')
    identifier = source_id.split(':', 1)[1]
    for record in job.get('metadata_records', []):
        if identifier not in {record.get('model_id'), record.get('item_id')}:
            raise ValueError('metadata_not_for_exact_original_product')
        dimensions = record.get('dimensions_m') or {}
        if set(dimensions) != {'height', 'length', 'width'}:
            continue
        values = [dimensions[k] for k in ('height', 'length', 'width')]
        if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in values):
            continue
        product_longest = max(values)
        # 0.6m is the existing raw manipulation role bound, not a new grasp
        # width. A >2x conflict is held for entity/units evidence, not rescaled.
        if product_longest > .6 and product_longest / longest > 2:
            resolve(record['metadata_file'])
            result['evidence'].append({'source_metadata': record['metadata_file'],
                'line_number': record['line_number'], 'item_id': record.get('item_id'),
                'model_id': record.get('model_id'), 'product_dimensions_m': dimensions,
                'generated_longest_OBB_m': longest, 'dimension_ratio': product_longest / longest,
                'product_type': record.get('product_type'), 'original_title': raw.get('name'),
                'retrieval_category': raw.get('category'), 'source_categories': raw.get('source_categories')})
    if result['evidence']:
        result.update(hold=True, reason='large_product_metadata_vs_engineered_small_geometry_entity_or_units_unresolved',
            missing_evidence=['whole product versus separate component identity',
                              'source geometry unit/shape agreement with product dimensions'],
            action='hold_automatic_manipulation; retain original geometry/cache for diagnosis')
    return result


def publish(bundle, raw_protocol, output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('preserve_existing_metadata_conflict_review')
    before, entries, _, bins = admitted_pool(bundle)
    protocol = read(raw_protocol)
    population = {r['source_id']: r for r in read(resolve(protocol['population']))['sources']}
    states = read(resolve(before['source_state_index']))['entries']
    by_source = {s['source_id']: s for s in states}
    kept, rows = [], []
    for entry in entries:
        job_ref = population[entry['source_id']]['job']; job = read(resolve(job_ref))
        row = inspect(entry, by_source[entry['source_id']], job)
        row['original_job'] = job_ref; rows.append(row)
        if not row['hold']:
            kept.append(entry)
    allowed = {e['source_id'] for e in kept}; selected_states = [s for s in states if s['source_id'] in allowed]
    review = {'schema': VERSION, 'implementation': binding(__file__), 'parent_bundle': binding(bundle),
        'raw_protocol': binding(raw_protocol), 'checked_sources': len(entries),
        'held_sources': sum(r['hold'] for r in rows), 'rows': rows,
        'existing_large_object_role_bound_m': .6, 'conflict_ratio_trigger': 2.,
        'geometry_mass_cache_modified': False, 'semantic_certification': False, 'API_calls': 0}
    write(output / 'METADATA_REVIEW.json', review)
    write(output / 'native_cache_index.json', {'schema': 'resimpair_grasp_cache_index_v1', 'entries': kept})
    write(output / 'source_state_index.json', {'schema': 'raw_candidate_source_states_v1', 'entries': selected_states})
    summary = {**before['summary'], 'admitted_grasp_sources': len(allowed), 'admitted_grasp_variants': len(kept),
        'poses': sum(e['retained'] for e in kept),
        'new_raw_sources': sum(e['ingestion_group'] == 'new_raw' for e in kept),
        'size_corrected_sources': sum(e['ingestion_group'] == 'size_corrected' for e in kept),
        'partition_source_counts': dict(Counter(s['partition'] for s in selected_states)),
        'metadata_conflict_held_sources': review['held_sources'], 'physical_success_inferred': False}
    derived = {**before, 'summary': summary, 'parent_bundle': binding(bundle),
        'native_cache_index': binding(output / 'native_cache_index.json'),
        'source_state_index': binding(output / 'source_state_index.json'),
        'source_semantics_checked_against_current_policy': {
            **before['source_semantics_checked_against_current_policy'],
            'source_state_index': binding(output / 'source_state_index.json'),
            'additional_source_metadata_review': binding(output / 'METADATA_REVIEW.json')},
        'implementation': binding(__file__), 'publication_status': 'opt_in_candidate_with_metadata_hold',
        'remaining_semantic_truth': 'unknown; this rule only detects targeted product/size contradictions'}
    write(output / 'BUNDLE.json', derived)
    _, actual, sources, actual_bins = admitted_pool(output / 'BUNDLE.json')
    if digest(actual) != digest(kept) or set(sources) != allowed or actual_bins != bins:
        raise ValueError('exact_unmodified_subset_and_receiver_membership_required')
    result = {**summary, 'bundle': binding(output / 'BUNDLE.json'), 'review': binding(output / 'METADATA_REVIEW.json'),
        'parent': binding(bundle), 'actual_task_consumer_reload_passed': True,
        'original_geometry_and_caches_preserved': True, 'default_promoted': False,
        'new_GPU_jobs': 0, 'new_robot_executions': 0, 'API_calls': 0}
    write(output / 'RESULT.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'raw-protocol', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = publish(args.bundle, args.raw_protocol, args.output)
    print({k: result[k] for k in ('admitted_grasp_sources', 'poses', 'metadata_conflict_held_sources', 'bundle')})
