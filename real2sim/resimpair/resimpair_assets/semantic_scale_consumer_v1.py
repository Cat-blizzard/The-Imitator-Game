"""Versioned metadata/scale review before sampling; old assets stay immutable.

Coarse retrieval categories cannot establish an engineering size for a clearly
different whole-device family. This consumer holds unresolved cases; it does
not guess replacement dimensions, certify semantics from names, or regenerate
meshes/grasps. Every original source and split remains addressable in the audit.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import json
from pathlib import Path
import re
import sqlite3
import time

from resimpair_grasp.common import binding, digest, read, resolve, write

VERSION = 'semantic_scale_consumer_revision_v1'
FAMILIES = [
    ('whole_computer', r'\b(laptops?|macbooks?|chromebooks?|notebook computers?|desktop computers?|computer towers?|all in one computers?|imac)\b'),
    ('display_furniture', r'\b((?:tv|television) stands?|entertainment centers?|meuble tv)\b'),
    ('large_display', r'\b(televisions?|tv|computer monitors?|desktop monitors?|gaming monitors?|crt monitors?|flat ?screen displays?)\b'),
    ('large_appliance', r'\b(fridge|refrigerator|freezer|dishwasher|washing machine|tumble dryer|sewing machine|microwave|air conditioner|radiator|vacuum cleaner|laser printer|inkjet printer|desktop printer|3d printer|printers?)\b'),
    ('furniture', r'\b(chair|stool|sofa|couch|ottoman|nightstand|bookcase|wardrobe|dresser|bookshelf|desk|table|cabinet|shelf|shelving|bed|bedframe|bench|sideboard|credenza|bathtub|toilet|sink)\b'),
    ('whole_vehicle', r'\b(car|truck|bus|buses|van|train|motorcycle|motorbike|bicycle|tractor|airplane|aeroplane|helicopter|yacht|sailboat|boat|spaceship|locomotive)\b'),
    ('toolset_or_assembly', r'\b(tool ?sets?|tool ?kits?|set of tools|tools set|tools collection|toolbox filled with tools)\b'),
]
MINIATURE = r'\b(toys?|miniature|figurine|dollhouse|dolls?|diecast|die cast|scale model|rc|radio controlled|mini|tiny|little)\b'
ACCESSORY = r'\b(remote(?: control)?|controller|charger|charging cable|adapter|replacement|spare part|keychain|key ring|ornament|cushion|cover|sleeve|mousepad|mouse pad|wall mount|screen mount|mounting bracket|wheel|tire|tyre|engine part|cup holder|trash can)\b|\b(?:laptop|notebook|computer|television|tv) (?:box|bag|case|sleeve|stand|charger|keyboard|cable)\b'
TOOL_STORAGE = r'\b(tool ?box|tool chest|tool case)\b'


def normalized(value):
    return re.sub(r'\s+', ' ', re.sub(r'[_/\-]+', ' ', str(value or '').lower())).strip()


def classify(raw, state):
    """Uniform source-level policy. Names are evidence for a hold, not truth."""
    name = normalized(raw.get('name'))
    matches = [(family, match.group()) for family, pattern in FAMILIES
               if (match := re.search(pattern, name))]
    categories = raw.get('source_categories') or []
    if isinstance(categories, str):
        categories = [categories]
    old_role = state.get('role', 'unknown')
    trustworthy = state.get('size_confidence') == 'source_metadata_geometry_agreement' and state.get('metric_metadata_status') == 'matched_uniform_metric_metadata'
    evidence = {'normalized_title': name, 'source_categories': categories,
        'matched_families': [x[0] for x in matches], 'matched_terms': [x[1] for x in matches],
        'toy_or_miniature_context': bool(re.search(MINIATURE, name)),
        'accessory_context': bool(re.search(ACCESSORY, name)),
        'source_metric_geometry_agreement': trustworthy, 'semantic_truth_verified': False}
    result = {'family': matches[0][0] if matches else 'not_targeted_by_revision',
        'consumer_action': 'retain_existing_policy', 'automatic_manipulation_hold': False,
        'semantic_status': 'unknown_not_newly_verified', 'evidence': evidence,
        'reason': 'no_targeted_whole_object_family_evidence',
        'potential_false_exclusion': False, 'recommended_size': None}
    if not matches:
        if re.search(TOOL_STORAGE, name):
            result.update(family='tool_storage_context', reason='tool_storage_not_automatically_equated_with_multi_object_toolset')
        return result
    # A TV stand is the whole furniture item, not the remote/control exception.
    is_display_furniture = any(f == 'display_furniture' for f, _ in matches)
    if evidence['accessory_context'] and not is_display_furniture:
        result.update(reason='accessory_or_packaging_context_not_whole_device',
                      consumer_action='retain_existing_policy_accessory_review', potential_false_exclusion=True)
        return result
    if old_role != 'manipulation':
        result.update(consumer_action='retain_existing_non_pick_role',
            reason='current_policy_already_non_manipulation',
            semantic_status='unknown_miniature_or_toy' if evidence['toy_or_miniature_context'] else 'unknown_not_newly_verified')
        return result
    if trustworthy and not any(f == 'toolset_or_assembly' for f, _ in matches):
        result.update(consumer_action='retain_metric_grounded_candidate',
            reason='source_metric_dimensions_match_geometry; no size guessed from category',
            semantic_status='unknown_miniature_or_toy' if evidence['toy_or_miniature_context'] else 'source_metric_candidate_not_semantic_truth')
        return result
    result.update(consumer_action='hold_for_role_scale_review', automatic_manipulation_hold=True,
        reason='whole_object_family_without_trustworthy_source_size',
        semantic_status='unknown_miniature_or_toy' if evidence['toy_or_miniature_context'] else 'whole_object_retrieval_candidate_not_verified',
        potential_false_exclusion=True,
        recommended_size={'mode': 'pending_source_metric_or_explicit_review', 'dimensions_m': None,
            'representative_length_m': None, 'inherit_legacy_normalized_parent': False,
            'rescale_or_regenerate_now': False},
        proposed_role='pending_not_automatic_pick',
        missing_evidence=['bound source metric dimensions or explicit dimension evidence',
                          'whole object versus toy/accessory/entity confirmation'])
    if evidence['toy_or_miniature_context']:
        result['reason'] = 'toy_or_miniature_scale_ambiguous; not_assumed_full_size'
    if any(f == 'toolset_or_assembly' for f, _ in matches):
        result['reason'] = 'toolset_entity_and_scale_unconfirmed; not_assumed_one_rigid_tool'
    return result


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
            assert descriptor['source_id'] == sid and descriptor['partition'] == row['partition']
            native_details.append({'source_id': sid, 'partition': row['partition'],
                'native_parent_descriptor': native_entry['asset'], 'parent_dimensions_m': descriptor['dimensions_m'],
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


def filter_bundle(bundle_path, sidecar_path, output):
    """Optional immutable consumer bundle; watcher and original cache unchanged."""
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('preserve_existing_filtered_bundle')
    bundle = read(bundle_path); sidecar = read(sidecar_path)
    revisions = {r['source_id']: r for r in sidecar['rows']}
    index = read(resolve(bundle['native_cache_index'])); retained = []; rejected = []
    for entry in index['entries']:
        source = revisions.get(entry['source_id'])
        if source is None:
            raise ValueError('new_bundle_source_missing_from_full_source_revision')
        d = read(resolve(entry['asset']))
        if d['partition'] != source['partition']:
            raise ValueError('original_source_partition_changed')
        (rejected if source['revision']['automatic_manipulation_hold'] else retained).append(entry)
    static = read(resolve(bundle['static_catalog'])); static_keep = []; static_rejected = []
    for ref in static['assets']:
        d = read(resolve(ref)); source = revisions.get(d['source_id'])
        if source and source['revision']['automatic_manipulation_hold']:
            static_rejected.append(ref)
        else:
            static_keep.append(ref)
    output.mkdir(parents=True)
    write(output / 'native_cache_index.json', {**index, 'entries': retained})
    write(output / 'static_catalog.json', {**static, 'assets': static_keep})
    write(output / 'held_original_entries.json', {'original_bundle': binding(bundle_path),
        'sidecar': binding(sidecar_path), 'native_cache_entries': rejected,
        'small_static_distractor_descriptors': static_rejected, 'original_bytes_retained': True})
    result = deepcopy(bundle)
    result.update(native_cache_index=binding(output / 'native_cache_index.json'),
                  static_catalog=binding(output / 'static_catalog.json'),
        consumer_semantic_revision={'schema': VERSION, 'original_bundle': binding(bundle_path),
            'sidecar': binding(sidecar_path), 'implementation': binding(__file__),
            'held_original_entries': binding(output / 'held_original_entries.json'),
            'scope': 'automatic manipulation and small static clutter only; containers/background unchanged',
            'source_ids_and_partitions_preserved': True, 'semantic_truth_verified': False})
    result['summary'] = {**result['summary'], 'admitted_grasp_variants': len(retained),
        'admitted_grasp_sources': len({e['source_id'] for e in retained}),
        'poses': sum(e['retained'] for e in retained), 'static_descriptors': len(static_keep),
        'consumer_semantic_held_grasp_variants': len(rejected), 'consumer_semantic_held_static_descriptors': len(static_rejected)}
    source_partitions = {e['source_id']: revisions[e['source_id']]['partition'] for e in retained}
    result['summary']['partition_source_counts'] = dict(Counter(source_partitions.values()))
    write(output / 'BUNDLE.json', result)
    summary = {'original_bundle': binding(bundle_path), 'consumer_bundle': binding(output / 'BUNDLE.json'),
        'original_variants': len(index['entries']), 'retained_variants': len(retained), 'held_variants': len(rejected),
        'original_sources': len({e['source_id'] for e in index['entries']}),
        'retained_sources': len(source_partitions), 'held_sources': len({e['source_id'] for e in rejected}),
        'held_small_static_descriptors': len(static_rejected), 'kept_small_static_descriptors': len(static_keep),
        'original_gate_is_preserved_not_rewritten': True, 'physics_or_grasp_retests': 0,
        'held_source_ids': sorted({e['source_id'] for e in rejected})}
    write(output / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    a = sub.add_parser('audit'); a.add_argument('--protocol', type=Path, required=True); a.add_argument('--output', type=Path, required=True)
    f = sub.add_parser('filter'); f.add_argument('--bundle', type=Path, required=True); f.add_argument('--sidecar', type=Path, required=True); f.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(audit(args.protocol, args.output) if args.command == 'audit' else filter_bundle(args.bundle, args.sidecar, args.output))
