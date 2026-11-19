"""Append one independently produced repaired native cache to a candidate bundle.

The added row has its own real CPU/GPU producer chain. Parent entries and
container catalogs stay byte-bound to the parent. All checks below are reads
of saved evidence and geometry; this publisher starts no physics or GPU work.
"""
import argparse
from collections import Counter
from pathlib import Path
import time

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write
from resimpair_grasp import evaluator, grasps
from resimpair_tasks import cudex_bulk_queue_v2 as grasp_validator
from resimpair_tasks.metric_population_sampler_v4 import admitted_pool
from . import metric_admission_v1 as admission
from . import semantic_scale_consumer_v1 as semantics
from . import native_reload_preflight_v1 as reload_implementation
from .raw_candidate_bundle_v2 import check_native_properties
from .raw_manipulation_v2 import _validated_job

VERSION = 'confirmed_cook_method_comparison_candidate_bundle_v1'


def reload_rows(report_ref):
    report = read(resolve(report_ref))
    protocol = read(resolve(report['protocol']))
    if (protocol['implementation'] != binding(reload_implementation.__file__)
            or protocol['retained_evaluator'] != binding(evaluator.__file__)
            or report['checked'] != len(report['rows'])
            or protocol['asset_descriptors'] != [row['asset'] for row in report['rows']]):
        raise ValueError('original_exact_native_reload_evidence_required')
    if len({digest(row['asset']) for row in report['rows']}) != len(report['rows']):
        raise ValueError('unique_reloaded_descriptors_required')
    return {digest(row['asset']): row for row in report['rows']}


def original_geometry_verdict(entry, report_ref, rows):
    row = rows.get(digest(entry['asset']))
    if (row is None or not row['passed'] or row['asset'] != entry['asset']
            or row['source_id'] != entry['source_id']):
        raise ValueError('exact_current_asset_missing_passing_native_evidence')
    current = evaluator.geometry(resolve(row['actual_native_geometry']), {'objects': [{
        'id': 'asset', 'asset_descriptor': entry['asset'], 'kind': 'movable'}]})
    if current != row['original_geometry_evaluator']:
        raise ValueError('original_native_evaluator_verdict_changed')
    return {'source_id': entry['source_id'], 'asset': entry['asset'], 'source_reload_report': report_ref,
            'actual_native_geometry': row['actual_native_geometry'], 'original_geometry_evaluator': current,
            'passed': True, 'new_native_actor_builds': 0}


def verified_new_entry(grasp_root, original_source_job):
    grasp_root = Path(grasp_root).resolve()
    protocol_ref = binding(grasp_root / 'PROTOCOL.json')
    protocol = read(resolve(protocol_ref))
    ledger = read(grasp_root / 'LEDGER.json')
    job_path = grasp_root / 'gpu/job.json'
    job = read(job_path)
    if (protocol['schema'] != 'confirmed_Mashroom_native_single_grasp_comparison_v1'
            or protocol['maximum_new_GPU_jobs'] != 1 or protocol['seed_count'] != 256
            or protocol['automatic_retries'] != 0 or protocol['old_raw_queue_producer_impersonated'] is not False
            or job['protocol'] != protocol_ref or ledger['protocol'] != protocol_ref
            or ledger['job'] != binding(job_path) or ledger['status'] != 'completed'
            or ledger['new_optimizer_attempts'] != 1 or ledger['automatic_retries'] != 0):
        raise ValueError('one_real_separate_new_geometry_grasp_attempt_required')
    for ref in protocol['code'] + protocol['configuration'] + protocol['real_native_producer_refs'] + protocol['historical_outputs_preserved']:
        resolve(ref)
    reservation = read(grasp_root / 'gpu/reservation.json')
    terminal = read(resolve(ledger['terminal']))
    if (reservation['job'] != binding(job_path) or reservation['protocol'] != protocol_ref
            or reservation['attempt'] != 1 or reservation['gpu']['gpu'] != 2
            or reservation['free_memory_MiB'] < 12 * 1024 or terminal['status'] != 'completed'
            or terminal['result'] != binding(grasp_root / 'gpu/result.json')):
        raise ValueError('actual_GPU_reservation_and_result_binding_required')
    for stage in ('worker', 'export'):
        terminal_stage = read(grasp_root / 'gpu' / f'{stage}_terminal.json')
        if terminal_stage['returncode'] != 0 or terminal_stage['elapsed_seconds'] > 120:
            raise ValueError('original_bounded_worker_and_export_completion_required')
    result = grasp_validator.validate_result(job_path)
    review_ref = ledger['CPU_native_result']
    review = read(resolve(review_ref))
    native = read(resolve(review['entry']))
    state = read(resolve(review['state']))
    descriptor = read(resolve(native['asset']))
    if (review_ref not in protocol['real_native_producer_refs'] or not review['passed']
            or review['status'] != 'native_ready_for_new_grasp_cache'
            or job['entry'] != native or job['asset'] != native['asset'] or result['asset'] != native['asset']
            or job['source_state'] != review['state'] or native['source_state'] != review['state']):
        raise ValueError('actual_new_native_entry_state_and_GPU_input_required')
    producer = native['native_producer']
    native_job_path, native_job = _validated_job(resolve(producer['job']))
    native_protocol = read(resolve(producer['protocol']))
    worker = read(resolve(producer['worker_result']))
    supervisor = read(resolve(producer['supervisor']))
    native_result = read(resolve(worker['native_result']))
    acceptance = read(resolve(review['acceptance']))
    if (native_job['kind'] != 'metric_descriptor' or native_job['maximum_asset_compile_attempts'] != 1
            or native_protocol['job'] != producer['job'] or supervisor['job'] != producer['job']
            or supervisor['worker_result'] != producer['worker_result'] or supervisor['returncode'] != 0
            or supervisor['status'] != 'worker_completed' or worker['status'] != 'native_geometry_ready'
            or worker['worker_pid'] != supervisor['worker_pid'] or worker['parent_pid'] != supervisor['supervisor_pid']
            or worker['asset'] != native['asset'] or native_result['asset'] != native['asset']
            or not native_result['geometry_qualified'] or not acceptance['passed']
            or not all(acceptance['checks'].values()) or acceptance['asset'] != native['asset']
            or native['final_native_acceptance'] != review['acceptance']):
        raise ValueError('actual_completed_native_job_and_ACCEPTANCE_required')
    check_native_properties(native, descriptor)
    raw_job_ref = binding(original_source_job)
    raw_job = read(resolve(raw_job_ref))
    review_protocol = read(resolve(review['protocol']))
    raw = raw_job['raw_row']
    if (raw_job_ref not in review_protocol['original_inputs_and_outcomes']
            or raw['id'] != descriptor['source_id'] or raw['partition'] != descriptor['partition']
            or raw['partition'] != state['partition'] or raw['source_sha'] != descriptor['provenance']['original_sha256']
            or job['source_aliases'] != raw_job.get('source_aliases', [])):
        raise ValueError('bound_original_source_metadata_identity_required_not_legacy_producer')
    gate = admission.inspect(native, state)
    revision = semantics.classify(raw, state)
    if gate != native['semantic_scale_admission'] or not gate['passed'] or revision['automatic_manipulation_hold']:
        raise ValueError('original_semantic_and_scale_rules_reject_new_entry')
    cache = grasps.load(result['cache'], native['asset'])
    if len(cache['candidates']) != result['retained'] or result['checked'] != 256:
        raise ValueError('exact_new_cache_and_original_full_pose_qualification_required')
    new = {**native, 'cache': result['cache'], 'retained': result['retained'],
        'method': 'cudex_native_com_v4', 'optimizer_result': binding(grasp_root / 'gpu/result.json'),
        'partition': descriptor['partition'], 'source_alias_split_consistent': True,
        'grasp_producer': {'job': binding(job_path), 'protocol': protocol_ref,
            'reservation': binding(grasp_root / 'gpu/reservation.json'),
            'terminal': ledger['terminal'], 'ledger': binding(grasp_root / 'LEDGER.json')},
        'semantic_scale_admission': gate, 'upstream_solver_success_count': result['upstream_success_count']}
    proof = {'source_id': new['source_id'], 'asset': new['asset'], 'cache': new['cache'],
        'native_entry': review['entry'], 'source_state': review['state'], 'native_producer': producer,
        'native_review': review_ref, 'native_acceptance': review['acceptance'],
        'grasp_producer': new['grasp_producer'], 'actual_grasp_result': new['optimizer_result'],
        'raw_metadata_job': raw_job_ref, 'raw_metadata_job_is_not_new_CPU_producer': True,
        'semantic_admission': gate, 'consumer_revision': revision,
        'native_pose_count': len(cache['candidates']), 'checked_pose_count': result['checked'],
        'upstream_solver_success_count': result['upstream_success_count'],
        'legacy_raw_queue_producer_impersonated': False, 'physical_success': 'unmeasured'}
    return new, state, proof


def publish(baseline, grasp_root, original_source_job, output):
    baseline, output = Path(baseline).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('fresh_method_comparison_bundle_required')
    began = time.monotonic()
    parent_ref = binding(baseline)
    parent = read(resolve(parent_ref))
    parent_release = read(baseline.parent / 'RESULT.json')
    if parent_release.get('publication_integrity_passed') is not True or parent_release['bundle'] != parent_ref:
        raise ValueError('verified_parent_candidate_bundle_required')
    old_entries = read(resolve(parent['native_cache_index']))['entries']
    old_states = read(resolve(parent['source_state_index']))['entries']
    old_excluded_ref = binding(baseline.parent / 'EXCLUDED.json')
    old_excluded = read(resolve(old_excluded_ref))
    write(output / 'PROTOCOL.json', {'schema': VERSION, 'implementation': binding(__file__),
        'parent_bundle': parent_ref, 'parent_result': binding(baseline.parent / 'RESULT.json'),
        'grasp_method_protocol': binding(Path(grasp_root) / 'PROTOCOL.json'),
        'original_source_metadata': binding(original_source_job), 'maximum_added_sources': 1,
        'container_catalog': parent['containers'], 'parent_exclusions': old_excluded_ref,
        'grasps_loader': binding(grasps.__file__), 'semantic_gate': binding(admission.__file__),
        'consumer_revision': binding(semantics.__file__), 'native_evaluator': binding(evaluator.__file__),
        'new_native_actor_builds': 0, 'simulation_steps': 0, 'GPU_jobs': 0, 'API_calls': 0,
        'robot_executions': 0, 'default_promoted': False})
    new, new_state, producer_proof = verified_new_entry(grasp_root, original_source_job)
    old_ids = {entry['source_id'] for entry in old_entries}
    if len(old_ids) != len(old_entries) or len(old_ids) != parent['summary']['admitted_grasp_sources']:
        raise ValueError('one_current_parent_variant_per_source_required')
    failed_old = [row for row in old_excluded['rows'] if row['source_id'] == new['source_id']]
    if new['source_id'] in old_ids or len(failed_old) != 1 or failed_old[0]['asset'] == new['asset']:
        raise ValueError('one_previously_excluded_source_with_new_exact_geometry_required')
    if any(entry['variant_id'] == new['variant_id'] for entry in old_entries):
        raise ValueError('new_active_variant_conflicts_with_parent')
    entries, states = [*old_entries, new], [*old_states, new_state]
    if len({s['source_id'] for s in states}) != len(states) or {s['source_id'] for s in states} != {e['source_id'] for e in entries}:
        raise ValueError('one_exact_state_per_active_source_required')
    state_by_id = {s['source_id']: s for s in states}
    old_reload = parent['final_native_reload']
    old_rows = reload_rows(old_reload)
    new_reload = new['fresh_final_reload']
    new_rows = reload_rows(new_reload)
    rows, geometry = [], []
    for entry in entries:
        state = state_by_id[entry['source_id']]
        descriptor = read(resolve(entry['asset']))
        cache = grasps.load(entry['cache'], entry['asset'])
        gate = admission.inspect(entry, state)
        if (not gate['passed'] or gate != entry['semantic_scale_admission']
                or entry['source_id'] != descriptor['source_id'] or state['partition'] != descriptor['partition']
                or len(cache['candidates']) != entry['retained']):
            raise ValueError('current_cache_source_partition_or_semantic_gate_changed')
        geometry.append(original_geometry_verdict(entry, new_reload if entry is new else old_reload,
                                                 new_rows if entry is new else old_rows))
        rows.append({'source_id': entry['source_id'], 'asset': entry['asset'], 'cache': entry['cache'],
            'partition': descriptor['partition'], 'admission': gate, 'accepted': True,
            'retained_poses': len(cache['candidates']), 'ingestion_group': entry['ingestion_group'],
            'new_method_entry': entry is new})
    write(output / 'PRODUCER_REVIEW.json', producer_proof)
    write(output / 'admission_audit.json', {'schema': VERSION, 'rows': rows,
        'parent_audit': parent['admission_audit'], 'new_source_consumer_revision': producer_proof['consumer_revision'],
        'semantic_truth_inferred': False, 'original_gate_implementation': binding(admission.__file__)})
    write(output / 'final_native_geometry_review.json', {'schema': 'composed_saved_native_geometry_review_v1',
        'rows': geometry, 'checked': len(geometry), 'passed': len(geometry),
        'source_reload_reports': [old_reload, new_reload], 'evaluator': binding(evaluator.__file__),
        'new_native_actor_builds': 0, 'simulation_steps': 0})
    write(output / 'native_cache_index.json', {'schema': 'resimpair_grasp_cache_index_v1', 'entries': entries})
    write(output / 'source_state_index.json', {'schema': 'raw_candidate_source_states_v1', 'entries': states})
    remaining = [row for row in old_excluded['rows'] if row['source_id'] != new['source_id']]
    write(output / 'EXCLUDED.json', {'rows': remaining, 'historical_exclusions': old_excluded_ref,
        'restored_with_new_geometry_and_cache': [{'historical_failure': failed_old[0],
            'new_asset': new['asset'], 'new_cache': new['cache'], 'producer_review': binding(output / 'PRODUCER_REVIEW.json')}],
        'historical_results_overwritten': False})
    summary = {**parent['summary'], 'admitted_grasp_sources': len(entries), 'admitted_grasp_variants': len(entries),
        'poses': sum(e['retained'] for e in entries),
        'new_raw_sources': sum(e['ingestion_group'] == 'new_raw' for e in entries),
        'size_corrected_sources': sum(e['ingestion_group'] == 'size_corrected' for e in entries),
        'partition_source_counts': dict(Counter(s['partition'] for s in states)),
        'final_reload_excluded_sources': len(remaining), 'historical_final_reload_failed_sources': len(old_excluded['rows']),
        'new_method_restored_sources': 1, 'grasp_or_task_success_inferred': False}
    bundle = {**parent, 'summary': summary, 'parent_bundle': parent_ref,
        'native_cache_index': binding(output / 'native_cache_index.json'),
        'source_state_index': binding(output / 'source_state_index.json'),
        'admission_audit': binding(output / 'admission_audit.json'),
        'source_semantics_checked_against_current_policy': {**parent['source_semantics_checked_against_current_policy'],
            'source_state_index': binding(output / 'source_state_index.json')},
        'final_native_reload': binding(output / 'final_native_geometry_review.json'),
        'new_method_producer_review': binding(output / 'PRODUCER_REVIEW.json'),
        'implementation': binding(__file__), 'created_unix': time.time(),
        'publication_status': 'opt_in_candidate; default unchanged',
        'qualification_scope': 'exact new native/cache producer plus unchanged parent; task execution remains unmeasured'}
    for field in ('raw_protocol', 'input_snapshot', 'parent_producer_publication'):
        if field in bundle:
            bundle['baseline_' + field] = bundle.pop(field)
    write(output / 'BUNDLE.json', bundle)
    _, actual, sources, bins = admitted_pool(output / 'BUNDLE.json')
    old_container_refs = read(resolve(parent['containers']))['assets']
    if (digest(actual) != digest(entries) or set(sources) != old_ids | {new['source_id']}
            or [ref for ref, _ in bins] != old_container_refs
            or len(bins) != parent['summary']['container_variants']
            or len({d['source_id'] for _, d in bins}) != parent['summary']['container_sources']
            or entries[:-1] != old_entries or states[:-1] != old_states):
        raise ValueError('actual_admitted_pool_changed_parent_membership_or_containers')
    resolve(parent_ref)
    result = {**summary, 'schema': VERSION, 'bundle': binding(output / 'BUNDLE.json'),
        'parent_bundle': parent_ref, 'producer_review': binding(output / 'PRODUCER_REVIEW.json'),
        'native_geometry_review': binding(output / 'final_native_geometry_review.json'),
        'new_source_id': new['source_id'], 'new_asset': new['asset'], 'new_cache': new['cache'],
        'new_poses': new['retained'], 'upstream_solver_success_count': new['upstream_solver_success_count'],
        'publication_integrity_passed': True, 'actual_task_consumer_reload_passed': True,
        'parent_entries_and_states_exactly_preserved': True, 'container_catalog_binding_preserved': True,
        'historical_exclusions_preserved': old_excluded_ref, 'new_native_actor_builds': 0,
        'new_CPU_native_attempts': 0, 'new_GPU_jobs': 0, 'simulation_steps': 0,
        'API_calls': 0, 'robot_executions': 0, 'default_promoted': False,
        'wall_seconds': time.monotonic() - began}
    write(output / 'RESULT.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'grasp-root', 'original-source-job', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = publish(args.baseline, args.grasp_root, args.original_source_job, args.output)
    print({k: result[k] for k in ('bundle', 'admitted_grasp_sources', 'poses', 'container_sources',
        'container_variants', 'publication_integrity_passed', 'actual_task_consumer_reload_passed', 'wall_seconds')})
