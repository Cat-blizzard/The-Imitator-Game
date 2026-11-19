"""Independent coverage accounting for bound cached pick/place requests.

This is a manifest audit, not physical or human-video acceptance. A declared
family name, prose, random seed, camera, or nuisance appearance never creates a
new goal program. Exact/quantized metric instances, asset-bound assignments and
asset-independent relation structures are deliberately reported separately.

Rows contain id, scene_id, partition, mechanism, goal_family, source_ids,
requested_task_id, and source_request={path, sha256}. source_request binds one
scene in cached_pickplace_requests_v1 format (objects, containers, tasks).
The placements list declares execution order, as in that request interface.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


VERSION = 'task_coverage_audit_v1'
POSITION_QUANTUM_M = .001
ANGLE_QUANTUM_RAD = .001
ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {'id', 'scene_id', 'partition', 'mechanism', 'goal_family',
            'source_ids', 'requested_task_id'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('missing_or_invalid_' + field)
    return value


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('nonfinite_or_invalid_metric_value')
    return float(value)


def _xy(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError('invalid_metric_xy')
    return [_number(x) for x in value]


def _yaw(value):
    angle = (_number(value) + math.pi) % (2 * math.pi) - math.pi
    return 0. if abs(angle) < 1e-12 else angle


def _quantize(value, key=''):
    if isinstance(value, dict):
        return {k: _quantize(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_quantize(v, key) for v in value]
    if isinstance(value, float):
        quantum = ANGLE_QUANTUM_RAD if 'yaw' in key else POSITION_QUANTUM_M
        return int(round(value / quantum))
    return value


def _bound_request(row, base_root, require_bindings):
    bound = row.get('source_request')
    if bound is not None:
        if not isinstance(bound, dict) or not {'path', 'sha256'} <= set(bound):
            raise ValueError('invalid_source_request_binding')
        path = Path(_text(bound['path'], 'binding_path'))
        if not path.is_absolute():
            path = base_root / path
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != bound['sha256']:
            raise ValueError('source_request_binding_changed')
        return json.loads(raw), True
    if require_bindings:
        raise ValueError('source_request_binding_required')
    if not isinstance(row.get('request'), dict):
        raise ValueError('inline_request_missing')
    return deepcopy(row['request']), False


def _scene(request):
    """Canonical identities depend on source/pose, never caller-chosen IDs."""
    objects = request.get('objects')
    if not isinstance(objects, list) or not objects:
        raise ValueError('scene_objects_missing')
    normalized = []
    for obj in objects:
        if not isinstance(obj, dict):
            raise ValueError('invalid_scene_object')
        normalized.append({'id': _text(obj.get('id'), 'object_id'),
            'source_id': _text(obj.get('source_id'), 'source_id'),
            'xy': _xy(obj.get('xy')), 'yaw': _yaw(obj.get('yaw', 0.))})
    if len({o['id'] for o in normalized}) != len(normalized):
        raise ValueError('duplicate_object_id')
    normalized.sort(key=lambda o: (o['source_id'], o['xy'], o['yaw']))
    objects_by_id = {}; occurrences = Counter(); initial_objects = []
    for obj in normalized:
        token = [obj['source_id'], occurrences[obj['source_id']]]
        occurrences[obj['source_id']] += 1
        objects_by_id[obj['id']] = token
        initial_objects.append({'object': token, 'xy': obj['xy'], 'yaw': obj['yaw']})
    containers = []
    for c in request.get('containers', []):
        if not isinstance(c, dict):
            raise ValueError('invalid_scene_container')
        source = c.get('source_id')
        if source is not None:
            _text(source, 'container_source_id')
        containers.append({'id': _text(c.get('id'), 'container_id'),
            'source_id': source, 'xy': _xy(c.get('xy')), 'yaw': _yaw(c.get('yaw', 0.))})
    if len({c['id'] for c in containers}) != len(containers):
        raise ValueError('duplicate_container_id')
    if {c['id'] for c in containers} & set(objects_by_id):
        raise ValueError('object_container_identity_overlap')
    containers.sort(key=lambda c: (c['source_id'] or '', c['xy'], c['yaw']))
    containers_by_id = {}; occurrences = Counter(); initial_containers = []
    for c in containers:
        token = [c['source_id'], occurrences[c['source_id']]]
        occurrences[c['source_id']] += 1
        containers_by_id[c['id']] = token
        initial_containers.append({'container': token, 'xy': c['xy'], 'yaw': c['yaw']})
    return objects_by_id, containers_by_id, {'objects': initial_objects, 'containers': initial_containers}


def _goal_program(request, task_id, objects, containers):
    tasks = request.get('tasks')
    if not isinstance(tasks, list) or not tasks:
        raise ValueError('scene_tasks_missing')
    if any(not isinstance(t, dict) for t in tasks):
        raise ValueError('invalid_scene_task')
    ids = [_text(t.get('id'), 'task_id') for t in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate_task_id_in_request')
    matches = [t for t in tasks if t['id'] == task_id]
    if len(matches) != 1:
        raise ValueError('requested_task_not_found')
    task = matches[0]
    placements = task.get('placements')
    if not isinstance(placements, list) or not placements:
        raise ValueError('placements_missing')
    # Region rank distinguishes alternate targets in a shared scene without
    # using container/region names. Coordinates stay in the metric fingerprint.
    region_set = set()
    for other_task in tasks:
        if not isinstance(other_task.get('placements'), list):
            raise ValueError('placements_missing')
        for op in other_task['placements']:
            if not isinstance(op, dict) or not isinstance(op.get('target'), dict):
                raise ValueError('invalid_placement_or_target')
            if op['target'].get('kind') == 'table':
                region_set.add(tuple(_xy(op['target'].get('xy'))))
    regions = sorted(region_set)
    exact, assignments = [], []; selected = set(); destination_groups = {}; relation_ops = []
    for op in placements:
        oid = op.get('object')
        if oid not in objects:
            raise ValueError('unknown_placement_object')
        if oid in selected:
            raise ValueError('repeated_object_requires_operation_windows')
        selected.add(oid)
        goal = op.get('target')
        if not isinstance(goal, dict):
            raise ValueError('target_missing')
        if goal.get('kind') == 'table':
            xy = _xy(goal.get('xy'))
            target = {'kind': 'table', 'xy': xy}
            assignment = {'kind': 'table', 'region_rank': regions.index(tuple(xy))}
            group = ('table', *xy)
        elif goal.get('kind') == 'container':
            cid = goal.get('container')
            if cid not in containers:
                raise ValueError('unknown_target_container')
            target = {'kind': 'container', 'container': containers[cid],
                      'offset_xy_m': _xy(goal.get('offset_xy_m', [0., 0.]))}
            assignment = {'kind': 'container', 'container': containers[cid]}
            group = ('container', *containers[cid])
        else:
            raise ValueError('unsupported_goal_kind')
        if 'yaw' in goal:
            target['yaw'] = _yaw(goal['yaw'])
        assignment['orientation_constrained'] = 'yaw' in target
        exact.append({'object': objects[oid], 'target': target})
        assignments.append({'object': objects[oid], 'target': assignment})
        if group not in destination_groups:
            destination_groups[group] = len(destination_groups)
        relation_ops.append({'destination_kind': target['kind'],
                             'destination_group': destination_groups[group],
                             'orientation_constrained': 'yaw' in target})
    spatial = []
    for i, op in enumerate(exact):
        if op['target']['kind'] != 'table':
            continue
        for j, other in enumerate(exact[i+1:], i+1):
            if other['target']['kind'] != 'table':
                continue
            delta = [b-a for a, b in zip(op['target']['xy'], other['target']['xy'])]
            spatial.append({'from_operation': i, 'to_operation': j,
                            'xy_order': [int(v > POSITION_QUANTUM_M) - int(v < -POSITION_QUANTUM_M) for v in delta]})
    preserved = sorted([objects[oid] for oid in objects if oid not in selected])
    asset_program = {'ordered_assignments': assignments, 'preserved_objects': preserved}
    relation_structure = {'mechanism': 'pick_place', 'ordered_destinations': relation_ops,
                          'preserved_count': len(preserved), 'table_goal_relations': spatial}
    return exact, asset_program, relation_structure, [objects[oid][0] for oid in selected]


def audit(rows, expected_count=500, known_source_ids=None, *, base_root=None, require_bindings=True):
    """Return integrity and distinctness facts; never infer dynamic success."""
    if not isinstance(rows, list):
        raise ValueError('manifest_rows_must_be_a_list')
    if type(expected_count) is not int or expected_count < 1:
        raise ValueError('expected_count_must_be_positive_integer')
    if known_source_ids is not None and (not isinstance(known_source_ids, (list, tuple, set))
                                        or any(not isinstance(s, str) or not s for s in known_source_ids)):
        raise ValueError('known_source_ids_must_be_a_collection_of_strings')
    base_root = Path(base_root or ROOT)
    history = None if known_source_ids is None else set(known_source_ids)
    errors = []; row_reports = []; ids = []; sources = set(); operated = set(); container_sources = set()
    partitions = defaultdict(set); scenes = {}; scene_members = defaultdict(list)
    families = Counter(); mechanisms = Counter(); semantics = Counter(); novelty = Counter()
    fingerprints = {k: defaultdict(list) for k in ['exact_metric', 'quantized_metric', 'asset_goal',
                    'relation_structure', 'exact_layout', 'quantized_layout',
                    'exact_pose_layout', 'quantized_pose_layout', 'declared_nuisance']}
    if len(rows) != expected_count:
        errors.append({'code': 'row_count_mismatch', 'expected': expected_count, 'actual': len(rows)})
    for index, row in enumerate(rows):
        rid = row.get('id', 'row_' + str(index)) if isinstance(row, dict) else 'row_' + str(index)
        if not isinstance(rid, str):
            rid = 'row_' + str(index)
        try:
            if not isinstance(row, dict) or not REQUIRED <= set(row):
                raise ValueError('required_manifest_fields_missing')
            for key in REQUIRED - {'source_ids'}:
                _text(row[key], key)
            ids.append(row['id'])
            if row['mechanism'] != 'pick_place':
                raise ValueError('unsupported_mechanism_requires_separate_program_auditor')
            request, bound = _bound_request(row, base_root, require_bindings)
            if request.get('id') != row['scene_id']:
                raise ValueError('scene_identity_differs_from_bound_request')
            objects, containers, initial = _scene(request)
            actual_sources = {token[0] for token in objects.values()}
            declared = row['source_ids']
            if not isinstance(declared, list) or any(not isinstance(s, str) for s in declared):
                raise ValueError('invalid_source_ids')
            if len(set(declared)) != len(declared) or set(declared) != actual_sources:
                raise ValueError('source_ids_differ_from_bound_request')
            exact, assignments, structure, manipulated = _goal_program(request, row['requested_task_id'], objects, containers)
            metric = {'initial': initial, 'ordered_placements': exact}
            poses = {group: sorted([{'xy': item['xy'], 'yaw': item['yaw']} for item in initial[group]],
                                   key=lambda item: (item['xy'], item['yaw']))
                     for group in ['objects', 'containers']}
            names = {'exact_metric': digest(metric), 'quantized_metric': digest(_quantize(metric)),
                     'asset_goal': digest(assignments), 'relation_structure': digest(structure),
                     'exact_layout': digest(initial), 'quantized_layout': digest(_quantize(initial)),
                     'exact_pose_layout': digest(poses), 'quantized_pose_layout': digest(_quantize(poses)),
                     'declared_nuisance': digest(request.get('distractors', {}))}
            scene_key = digest({'initial': initial, 'distractors': request.get('distractors', {}),
                                'seed': request.get('seed', row.get('seed'))})
            if row['scene_id'] in scenes and scenes[row['scene_id']] != scene_key:
                raise ValueError('same_scene_id_has_different_initial_request')
            scenes[row['scene_id']] = scene_key
            seen = [] if history is None else sorted(actual_sources & history)
            if row.get('source_novelty') == 'global_unseen_asset' and (history is None or seen):
                raise ValueError('unsupported_global_unseen_source_claim')
            state = 'unknown_history' if history is None else ('contains_previously_seen' if seen else 'unseen_against_supplied_history')
            novelty[state] += 1
            for key, value in names.items():
                fingerprints[key][value].append(row['id'])
            scene_members[row['scene_id']].append(row['id'])
            for sid in actual_sources:
                partitions[sid].add(row['partition'])
            sources.update(actual_sources); operated.update(manipulated)
            container_sources.update(token[0] for token in containers.values() if token[0] is not None)
            families[row['goal_family']] += 1; mechanisms[row['mechanism']] += 1
            semantic_state = row.get('semantic_status', 'unverified')
            semantics[str(semantic_state)] += 1
            row_reports.append({'id': row['id'], 'scene_id': row['scene_id'], 'source_request_bound': bound,
                'source_novelty': state, 'previously_seen_sources': seen, 'semantic_status': semantic_state,
                'manipulated_source_ids': sorted(set(manipulated)), 'fingerprints': names,
                'derived_relation_structure': structure,
                'physical_status': 'not_evaluated', 'human_video_status': 'not_evaluated'})
        except (ValueError, KeyError, TypeError, OSError) as exc:
            errors.append({'row': rid, 'index': index, 'code': str(exc)})
    for rid, count in Counter(ids).items():
        if count > 1:
            errors.append({'row': rid, 'code': 'duplicate_row_id', 'count': count})
    leaks = {s: sorted(p) for s, p in partitions.items() if len(p) > 1}
    for source, splits in sorted(leaks.items()):
        errors.append({'code': 'source_partition_leakage', 'source_id': source, 'partitions': splits})
    duplicates = [members for members in fingerprints['quantized_metric'].values() if len(members) > 1]
    for members in duplicates:
        errors.append({'code': 'duplicate_quantized_task_request', 'rows': members})
    unique = {key: len(value) for key, value in fingerprints.items()}
    same_goal_layouts = defaultdict(set)
    for row in row_reports:
        same_goal_layouts[row['fingerprints']['asset_goal']].add(row['fingerprints']['quantized_metric'])
    try:
        input_digest = digest(rows)
    except ValueError:
        input_digest = None
        errors.append({'code': 'nonfinite_manifest_is_not_canonical_json'})
    return {'version': VERSION, 'rule': {'path': str(Path(__file__).resolve()),
            'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        'input_rows_digest': input_digest, 'expected_count': expected_count, 'passed': not errors,
        'scope': 'request identity, coverage and split integrity only; no physics, human quality or semantic-label certification',
        'position_quantum_m': POSITION_QUANTUM_M, 'angle_quantum_rad': ANGLE_QUANTUM_RAD,
        'counts': {'rows': len(rows), 'valid_rows': len(row_reports), 'source_assets': len(sources),
                   'manipulated_source_assets': len(operated), 'explicit_container_source_assets': len(container_sources),
                   'scene_ids': len(scenes), 'declared_goal_families': len(families), 'contact_mechanisms': len(mechanisms),
                   **{'unique_' + k: v for k, v in unique.items()},
                   'layout_or_metric_variants_of_same_asset_goal': sum(max(0, len(s)-1) for s in same_goal_layouts.values())},
        'declared_goal_family_distribution': dict(sorted(families.items())),
        'mechanism_distribution': dict(sorted(mechanisms.items())),
        'declared_semantic_status_distribution': dict(sorted(semantics.items())),
        'source_novelty_distribution': dict(sorted(novelty.items())),
        'history_scope': 'supplied source IDs only; missing history stays unknown',
        'known_source_ids_digest': None if history is None else digest(sorted(history)),
        'source_partition_scope': 'all manipulable objects in requests, including unselected objects; shared scene props and cameras excluded',
        'source_partition_leaks': leaks, 'duplicate_quantized_groups': duplicates,
        'scene_task_counts': {k: len(v) for k, v in sorted(scene_members.items())},
        'errors': errors, 'rows': row_reports,
        'interpretation': ['A goal_family name is a declaration, not evidence of a new semantic type.',
            'Asset-goal fingerprints remove metric layout, appearance, prose and caller IDs; they retain selected/preserved sources and target assignments.',
            'Relation structures remove asset identities and metric coordinates; this count is not a certification of task-language semantics.',
            'Layout fingerprints retain sources; pose-layout fingerprints count only requested XY/yaw configurations, not collision-geometry equivalence.',
            'The placements order is recorded as declared by the request interface; necessity of that order is not inferred.',
            'Declared nuisance configurations do not certify distinct rendered appearances.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-count', type=int, default=500)
    parser.add_argument('--known-sources', type=Path)
    parser.add_argument('--base-root', type=Path, default=ROOT)
    parser.add_argument('--allow-inline', action='store_true')
    args = parser.parse_args()
    value = json.loads(args.manifest.read_text())
    rows = value['rows'] if isinstance(value, dict) else value
    history = None
    if args.known_sources:
        value = json.loads(args.known_sources.read_text())
        history = value['source_ids'] if isinstance(value, dict) else value
    report = audit(rows, args.expected_count, history, base_root=args.base_root,
                   require_bindings=not args.allow_inline)
    report['manifest'] = {'path': str(args.manifest.resolve()),
                          'sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'passed': report['passed'], 'counts': report['counts'],
                      'error_count': len(report['errors']), 'output': str(args.output)}, ensure_ascii=False, indent=2))
