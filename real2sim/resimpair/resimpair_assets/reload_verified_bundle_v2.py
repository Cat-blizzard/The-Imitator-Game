"""Publish exact-reload candidates only after rechecking their real producers.

The final-collider test cannot prove that a grasp cache belongs to that collider.
This version also loads the original cache signature, replays the CPU/GPU
provenance checks and retains the current semantic/scale gate. No new simulation,
grasp optimization or relaxed acceptance rule is involved.
"""
import argparse
from pathlib import Path

from resimpair_grasp.common import binding, digest, read, resolve, write
from resimpair_grasp.grasps import load as load_cache
from resimpair_tasks.cudex_bulk_queue_v2 import validate_result
from . import raw_candidate_bundle_v2 as producer
from . import raw_candidate_bundle_v3 as consumer
from . import reload_verified_bundle_v1 as retained
from .metric_admission_v1 import inspect
from .semantic_scale_consumer_v1 import classify


def verify_parent(publication):
    result = read(publication)
    if result.get('publication_integrity_passed') is not True:
        raise ValueError('verified_native_producer_publication_required')
    bundle_path = resolve(result['bundle']); bundle = read(bundle_path)
    entries = read(resolve(bundle['native_cache_index']))['entries']
    states = {s['source_id']: s for s in read(resolve(bundle['source_state_index']))['entries']}
    if len({e['source_id'] for e in entries}) != len(entries):
        raise ValueError('one_exact_variant_per_source_required')
    # Do this before trusting any stored pass flag or review document. A valid
    # cache from a different real source is still an invalid cache here.
    for entry in entries:
        cache = load_cache(entry['cache'], entry['asset'])
        if len(cache['candidates']) != entry['retained']:
            raise ValueError('actual_cache_count_changed')
        state = read(resolve(entry['source_state']))
        gate = inspect(entry, state)
        # Re-evaluate the original rule. OBB recomputation may differ in final
        # floating-point digits; the stored report itself is bound below to the
        # captured producer entry, rather than using float equality as a rule.
        if states.get(entry['source_id']) != state or not gate['passed']:
            raise ValueError('actual_source_state_or_current_scale_admission_changed')

    fields = read(resolve(result['consumer_field_checks']))
    if result.get('producer_chain_review'):
        review = read(resolve(result['producer_chain_review']))
        if review['implementation'] != binding(producer.__file__):
            raise ValueError('retained_raw_producer_checker_required')
        root = resolve(review['raw_protocol']).parent; protocol = read(root / 'protocol.json')
        population = {s['source_id']: s for s in read(resolve(protocol['population']))['sources']}
        captured = read(resolve(review['captured_index']))
        recorded = {r['source_id']: r for r in review['grasp_rows']}
        rows = []
        for entry in entries:
            checked = producer.validate_grasp_chain(root, entry, protocol, population)
            if recorded.get(entry['source_id']) != checked:
                raise ValueError('raw_producer_review_does_not_cover_actual_entry')
            rows.append(checked)
        if fields.get('implementation') != binding(consumer.__file__) or fields.get('exact_captured_input') != review['captured_index']:
            raise ValueError('exact_raw_consumer_field_review_required')
    elif result.get('XML_GPU_producer_review'):
        from . import raw_robocasa_grasp_v1 as xml
        review = read(resolve(result['XML_GPU_producer_review']))
        if review['implementation'] != binding(xml.__file__):
            raise ValueError('retained_XML_producer_checker_required')
        root = resolve(review['protocol']).parent; protocol = xml.validate(root)
        captured = read(resolve(review['rolling_input']))
        specs = {s['source_id']: s for s in protocol['job_specs']}
        recorded = {r['source_id']: r for r in review['rows']}; rows = []
        for entry in entries:
            spec = specs[entry['source_id']]; job_path = Path(spec['output']) / 'job.json'
            job = read(job_path); terminal_path = job_path.parent / 'terminal.json'
            terminal = read(terminal_path); reservation = read(job_path.parent / 'reservation.json')
            native = xml.native_producer(job['entry'], protocol['XML_population_protocol'])
            actual = validate_result(job_path); row = recorded[entry['source_id']]
            if (native != spec['XML_native_producer'] or row['XML_native_producer'] != native
                    or row['GPU_job'] != binding(job_path) or row['GPU_terminal'] != binding(terminal_path)
                    or terminal['status'] != 'completed' or row['status'] != 'completed'
                    or terminal['result'] != entry['optimizer_result'] or row['export'] != terminal['result']
                    or actual['cache'] != entry['cache'] or actual['retained'] != entry['retained']
                    or reservation['job'] != binding(job_path) or reservation['protocol'] != binding(root / 'protocol.json')):
                raise ValueError('XML_GPU_export_not_bound_to_actual_native_producer')
            rows.append(row)
    else:
        raise ValueError('supported_actual_producer_chain_required')

    consumer.rolling_contract(captured)
    if captured != read(resolve(bundle['input_snapshot'])) or bundle['raw_protocol'] != binding(root / 'protocol.json'):
        raise ValueError('bundle_input_not_actual_captured_producer_snapshot')
    by_source = {e['source_id']: e for e in captured['entries']}
    for entry in entries:
        original = by_source[entry['source_id']]
        if any(entry.get(k) != v for k, v in original.items()):
            raise ValueError('consumer_changed_captured_cache_or_source_field')
        raw = read(root / 'cpu' / digest(entry['source_id'])[:24] / 'job.json')['raw_row']
        if classify(raw, states[entry['source_id']])['automatic_manipulation_hold']:
            raise ValueError('current_source_semantics_require_hold')
    current_fields = consumer.check_published(root, bundle_path)
    if current_fields != fields['rows']:
        raise ValueError('final_native_fields_differ_from_producer_review')
    receiving = read(resolve(bundle['receiving_geometry']))
    catalogs = receiving['input_catalogs']
    if catalogs[0] != read(resolve(bundle['parent_bundle']))['containers']:
        raise ValueError('receiver_baseline_changed')
    receiver_check = producer.validate_receiver_chains(resolve(bundle['parent_bundle']), resolve(catalogs[1]))
    if receiver_check != review['receiver_review']:
        raise ValueError('receiver_chain_changed')
    expected_receivers = [r['descriptor'] for r in receiving['rows'] if r['automatic_receiving_use']]
    if read(resolve(bundle['containers']))['assets'] != expected_receivers:
        raise ValueError('receiver_membership_changed')
    return {'schema': 'exact_native_cache_producer_recheck_v2', 'implementation': binding(__file__),
            'publication': binding(publication), 'bundle': result['bundle'], 'checked_sources': len(entries),
            'producer_rows': rows, 'cache_signatures_and_counts_passed': True,
            'current_source_semantics_passed': True, 'consumer_fields_match_registered_producers': True,
            'receiver_provenance_rechecked': True, 'simulation_steps': 0, 'GPU_jobs': 0, 'API_calls': 0}


def publish(publication, reload_report, output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('preserve_previous_membership_and_failures')
    checked = verify_parent(publication)
    # Reuse the exact old evaluator and filtering implementation after the
    # additional cache/producer check, retaining the old positive and negative.
    write(output / 'PRODUCER_RECHECK.json', checked)
    result = retained.publish(publication, reload_report, output / 'filtered')
    final = {**result, 'producer_recheck': binding(output / 'PRODUCER_RECHECK.json'),
             'implementation': binding(__file__), 'retained_reload_filter': binding(retained.__file__)}
    write(output / 'RESULT.json', final)
    return final


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('publication', 'reload-report', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(publish(args.publication, args.reload_report, args.output))
