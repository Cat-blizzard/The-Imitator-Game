"""Complete source census retaining historical split discrepancies as evidence.

Uses the frozen v1 metadata/scale rules. Old native descriptors can carry a
historical split different from raw metadata; record both rather than terminate
an otherwise complete census or silently overwrite either value.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import json
from pathlib import Path
import sqlite3
import time
from resimpair_grasp.common import binding, digest, read, resolve, write
from . import semantic_scale_consumer_v1 as retained
VERSION = 'semantic_scale_consumer_revision_v2'
classify = retained.classify
filter_bundle = retained.filter_bundle


def audit(protocol_path, output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('preserve_existing_semantic_audit')
    protocol = read(protocol_path); began = time.monotonic()
    states = read(resolve(protocol['source_inventory']))
    by_state = {s['source_id']: s for s in states['entries']}
    native = read(resolve(protocol['native_inventory']))['entries']
    native_by_id = {e['source_id']: e for e in native}
    bundle = read(resolve(protocol['bundle']))
    cache = read(resolve(bundle['native_cache_index']))['entries']
    admitted = defaultdict(list)
    for entry in cache:
        admitted[entry['source_id']].append(entry)
    db = sqlite3.connect('file:' + str(resolve(protocol['raw_metadata'])) + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    raw_rows = db.execute('SELECT id,source,name,category,partition,source_sha,record,classification FROM assets').fetchall()
    db.close()
    if {r['id'] for r in raw_rows} != set(by_state):
        raise ValueError('complete_raw_metadata_and_state_source_sets_required')
    rows = []; native_details = []; cache_details = []; partitions = defaultdict(set)
    for raw in raw_rows:
        state = by_state[raw['id']]
        record = json.loads(raw['record']) if raw['record'] else {}
        raw_classification = json.loads(raw['classification']) if raw['classification'] else None
        if raw['partition'] != state['partition'] or raw['source_sha'] != state['raw_sha256']:
            raise ValueError('raw_and_inventory_identity_split_or_bytes_mismatch')
        source = {'name': raw['name'], 'source_categories': record.get('categories') or state.get('source_categories') or [],
                  'category': raw['category']}
        decision = classify(source, state)
        sid = raw['id']; native_entry = native_by_id.get(sid)
        row = {'source_id': sid, 'partition': raw['partition'], 'name': raw['name'],
            'raw_metadata_record_sha256': digest(record), 'raw_source_sha256': raw['source_sha'],
            'original_retrieval_classification': raw_classification,
            'native_canonical': native_entry is not None, 'currently_admitted_variants': len(admitted.get(sid, [])),
            'old_role': state.get('role'), 'old_profile': state.get('profile', state.get('semantic_profile')),
            'old_semantic_admission': state.get('semantic_admission'), 'size_confidence': state.get('size_confidence'),
            'base_longest_m': state.get('base_longest_m'), 'old_longest_m': state.get('old_longest_m'),
            'base_multiplier_from_parent': state.get('base_multiplier_from_parent'),
            'current_policy': state.get('policy'), 'revision': decision}
        if native_entry is not None:
            descriptor = read(resolve(native_entry['asset']))
            policy = read(resolve(state['policy'])) if state.get('policy') else None
            legacy_identity_consistent = descriptor['source_id'] == sid
            legacy_split_consistent = descriptor['partition'] == row['partition']
            native_details.append({'source_id': sid, 'partition': row['partition'],
                'native_parent_descriptor': native_entry['asset'], 'parent_dimensions_m': descriptor['dimensions_m'],
                'historical_descriptor_source_id': descriptor['source_id'],
                'historical_descriptor_partition': descriptor['partition'],
                'historical_identity_consistent': legacy_identity_consistent,
                'historical_partition_consistent': legacy_split_consistent,
                'policy': state.get('policy'), 'size_policy': policy.get('size') if policy else None,
                'mass_policy': policy.get('mass') if policy else None,
                'base_multiplier_from_parent': state.get('base_multiplier_from_parent'),
                'old_normalized_scale_inherited_without_source_metric':
                    bool(state.get('size_confidence') == 'engineering_category_or_morphology_assumption'
                         and state.get('base_multiplier_from_parent') is not None
                         and abs(state['base_multiplier_from_parent'] - 1.) < 1e-4),
                'semantic_revision': decision})
        for entry in admitted.get(sid, []):
            d = read(resolve(entry['asset']))
            assert d['source_id'] == sid and d['partition'] == row['partition']
            cache_details.append({'source_id': sid, 'partition': row['partition'], 'variant_id': entry['variant_id'],
                'descriptor': entry['asset'], 'cache': entry['cache'], 'retained_pose_count': entry['retained'],
                'dimensions_m': d['dimensions_m'], 'mass_kg': d['mass_kg'],
                'size_provenance': d.get('size_provenance'), 'mass_provenance': d.get('mass_provenance'),
                'original_semantic_scale_admission': entry['semantic_scale_admission'], 'semantic_revision': decision})
        rows.append(row); partitions[sid].add(row['partition'])
    if len(native_details) != len(native) or len(cache_details) != len(cache):
        raise ValueError('all_native_and_admitted_sources_must_be_audited')
    output.mkdir(parents=True)
    write(output / 'implementation_freeze.json', {'protocol': binding(protocol_path), 'implementation': binding(__file__),
        'revision_classifier': binding(retained.__file__),
        'original_semantic_classifier': binding('resimpair_assets/metric_semantics_v1.py'),
        'original_inventory': states['implementation'], 'new_acceptance_or_physics_rules': False})
    write(output / 'source_revision_sidecar.json', {'schema': VERSION, 'protocol': binding(protocol_path),
        'original_inventory': protocol['source_inventory'], 'raw_metadata': protocol['raw_metadata'], 'rows': rows,
        'source_ids_and_partitions_preserved': True, 'semantic_truth_inferred_from_names': False})
    write(output / 'native_all_sources.json', {'schema': VERSION, 'native_inventory': protocol['native_inventory'], 'rows': native_details})
    write(output / 'admitted_all_variants.json', {'schema': VERSION, 'bundle': protocol['bundle'], 'rows': cache_details})
    scopes = {'raw_all': rows, 'native_canonical': [r for r in rows if r['native_canonical']],
              'admitted_sources': [r for r in rows if r['currently_admitted_variants']]}
    counts = {}
    for scope, items in scopes.items():
        targeted = [r for r in items if r['revision']['family'] != 'not_targeted_by_revision']
        counts[scope] = {'all_sources': len(items), 'targeted_retrieval_candidates': len(targeted),
            'actions_all_sources': dict(Counter(r['revision']['consumer_action'] for r in items)),
            'family_counts': dict(Counter(r['revision']['family'] for r in targeted)),
            'automatic_manipulation_holds': sum(r['revision']['automatic_manipulation_hold'] for r in items),
            'toy_or_miniature_ambiguity': sum(r['revision']['evidence']['toy_or_miniature_context'] for r in targeted),
            'accessory_or_packaging_not_blocked': sum(r['revision']['consumer_action'] == 'retain_existing_policy_accessory_review' for r in items),
            'per_family': {family: dict(Counter(r['revision']['consumer_action'] for r in targeted if r['revision']['family'] == family))
                           for family in sorted({r['revision']['family'] for r in targeted})}}
    summary = {'schema': VERSION, 'protocol': binding(protocol_path), 'counts': counts,
        'native_legacy_engineering_size_inherited': sum(r['old_normalized_scale_inherited_without_source_metric'] for r in native_details),
        'historical_native_descriptor_partition_disagreements': sum(not r['historical_partition_consistent'] for r in native_details),
        'historical_native_descriptor_identity_disagreements': sum(not r['historical_identity_consistent'] for r in native_details),
        'admitted_variants_audited': len(cache_details),
        'admitted_variants_held': sum(r['semantic_revision']['automatic_manipulation_hold'] for r in cache_details),
        'automatic_truth_or_size_claims': 0, 'seconds': time.monotonic() - began,
        'original_metadata_meshes_grasps_and_jobs_unchanged': True,
        'limitations': ['Uniform metadata screen over every source is not exhaustive semantic validation.',
            'Accessory and miniature contexts can still contain size errors; retained rows are not newly certified.',
            'Holds are reviewable potential false exclusions, not permanent declarations that an asset is unusable.',
            'No replacement dimensions or mass values are invented; updated geometry would need new source evidence and separate native/grasp qualification.'],
        'physical_executions': 0, 'GPU_jobs': 0, 'new_cudex_jobs': 0, 'API_calls': 0}
    write(output / 'summary.json', summary)
    columns = ['source_id', 'partition', 'name', 'native_canonical', 'currently_admitted_variants',
        'old_role', 'old_profile', 'size_confidence', 'base_longest_m', 'old_longest_m',
        'base_multiplier_from_parent', 'family', 'consumer_action', 'reason', 'semantic_status']
    with (output / 'targeted_review_queue.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader()
        for row in rows:
            if row['revision']['family'] != 'not_targeted_by_revision':
                writer.writerow({k: row.get(k, row['revision'].get(k)) for k in columns})
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    a = sub.add_parser('audit'); a.add_argument('--protocol', type=Path, required=True); a.add_argument('--output', type=Path, required=True)
    f = sub.add_parser('filter'); f.add_argument('--bundle', type=Path, required=True); f.add_argument('--sidecar', type=Path, required=True); f.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(audit(args.protocol, args.output) if args.command == 'audit' else filter_bundle(args.bundle, args.sidecar, args.output))
