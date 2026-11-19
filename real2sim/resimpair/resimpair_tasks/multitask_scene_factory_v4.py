"""Admit 3..8 tasks from existing finite, independently inspectable IK evidence.

No layouts, sources, grasps, endpoints or physical executions are newly sampled.
For each source scene choose the already evaluated layout with most qualified
goals; ties use the original proposal order. Keep all original requested goals,
including rejected goals. This changes the scene-level admission policy, never
the endpoint or physical acceptance rules. Frozen v3 output remains untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import time

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write
from . import multitask_scene_factory_v3 as retained

VERSION = 'existing_endpoint_per_task_admission_factory_v4'


def summarize_proof(request, proof):
    expected = {t['id']: t for t in request['tasks']}
    actual = defaultdict(list)
    for row in proof['operations']:
        actual[row['task']].append(row)
    if set(actual) != set(expected):
        raise ValueError('endpoint_proof_does_not_cover_exact_requested_tasks')
    result = {}
    for short, task in expected.items():
        rows = actual[short]
        if len(rows) != len(task['placements']) or len({r['operation'] for r in rows}) != len(rows):
            raise ValueError('endpoint_operation_coverage_mismatch')
        if sorted(r['object'] for r in rows) != sorted(p['object'] for p in task['placements']):
            raise ValueError('endpoint_object_identity_mismatch')
        if any(type(r['passed']) is not bool for r in rows):
            raise ValueError('explicit_boolean_endpoint_decision_required')
        result[short] = {'endpoint_passed': all(r['passed'] for r in rows),
                         'failed_operations': [r['operation'] for r in rows if not r['passed']],
                         'operation_reports': [r['report'] for r in rows]}
    return result


def select_existing_layout(candidates, minimum_eligible=3):
    if type(minimum_eligible) is not int or not 3 <= minimum_eligible <= 8:
        raise ValueError('scene_requires_at_least3_and_at_most8_tasks')
    if not candidates:
        return None
    if len(candidates) > 32 or len({r['proposal'] for r in candidates}) != len(candidates):
        raise ValueError('original_unique_at_most32_proposals_required')
    ordered = sorted(candidates, key=lambda r: (-r['qualified_task_count'], r['proposal']))
    selected = ordered[0]
    return {**selected, 'scene_admitted': selected['qualified_task_count'] >= minimum_eligible}


def verify_selected_endpoint_bindings(scene, tasks, proof):
    """Reconstructed geometry must match each bound existing endpoint target."""
    actual = {(row['task'], row['operation']): row for row in proof['operations']}
    checked = 0
    for task in tasks:
        for operation in task['operations']:
            row = actual[(task['requested_task_id'], operation['id'])]
            key = digest({'object': operation['object'], 'target': operation['target']})
            saved = read(resolve(row['report']))
            if (row['unique_target'] != key or row['object'] != operation['object']
                    or saved['object'] != operation['object'] or saved['target'] != operation['target']
                    or saved['initial_scene_seed'] != scene['seed'] or saved['report']['passed'] != row['passed']):
                raise ValueError('bound_endpoint_geometry_identity_or_decision_changed')
            checked += 1
    if checked != len(proof['operations']):
        raise ValueError('endpoint_operations_not_bijective_with_compiled_tasks')
    return checked


def build(source, output, *, minimum_eligible_tasks=3):
    source, output = Path(source), Path(output)
    if output.exists(): raise ValueError('preserve_existing_admission_factory')
    parent = read(source / 'protocol.json')
    if parent['schema'] != retained.VERSION or parent['maximum_layout_proposals'] > 32:
        raise ValueError('frozen_v3_endpoint_factory_required')
    for key in ['implementation', 'compiler', 'endpoint_selector', 'endpoint_hand_sweep']:
        resolve(parent[key])
    if parent['implementation'] != binding(retained.__file__):
        raise ValueError('retained_v3_compiler_mismatch')
    select_existing_layout([], minimum_eligible_tasks)
    started = time.monotonic(); output.mkdir(parents=True)
    index, containers = resolve(parent['index']), resolve(parent['containers'])
    entries = {e['source_id']: e for e in read(index)['entries']}
    bin_entries = {read(resolve(b))['source_id']: b for b in read(containers)['assets']}
    protocol = {**parent, 'schema': VERSION, 'implementation': binding(__file__),
        'parent_factory_protocol': binding(source / 'protocol.json'),
        'parent_factory_scenes': binding(source / 'scenes.json'),
        'retained_geometry_compiler': binding(retained.__file__),
        'admission_policy': 'most_endpoint_qualified_tasks_then_original_proposal_index',
        'minimum_eligible_tasks_per_scene': minimum_eligible_tasks,
        'all_original_requested_goals_retained': True, 'new_layout_proposals': 0, 'new_endpoint_IK_calls': 0,
        'uses_physical_results_for_selection': False, 'physical_executions': 0, 'GPU_jobs': 0, 'API_calls': 0}
    write(output / 'protocol.json', protocol)
    write(output / 'inventory.json', read(source / 'inventory.json'))
    rows = []; eligible_count = 0; operation_requests = 0; eligible_operations = 0; task_rejections = Counter()
    for old in read(source / 'scenes.json')['scenes']:
        old_dir = Path(old['directory']); candidates = []
        attempts = read(resolve(old['attempts']))['attempts']
        for attempt in attempts:
            trial_dir = old_dir / 'proposals' / f"{attempt['trial']:02d}"
            proof_path = trial_dir / 'endpoint_IK/qualification.json'
            if not proof_path.exists(): continue
            original_ref = old['accepted']['endpoint_IK'] if attempt['accepted'] else attempt.get('details', {}).get('qualification')
            if original_ref is None or resolve(original_ref).resolve() != proof_path.resolve():
                raise ValueError('endpoint_report_missing_original_attempt_binding')
            proof = read(resolve(original_ref)); request = read(trial_dir / 'request.json')
            decisions = summarize_proof(request, proof)
            candidates.append({'proposal': attempt['trial'], 'seed': attempt['seed'],
                'qualified_task_count': sum(d['endpoint_passed'] for d in decisions.values()),
                'qualified_task_ids': [t for t, d in decisions.items() if d['endpoint_passed']],
                'request': binding(trial_dir / 'request.json'), 'endpoint_IK': binding(proof_path),
                'workspace': binding(trial_dir / 'workspace.json'), 'sampling': binding(trial_dir / 'sampling.json')})
        selected = select_existing_layout(candidates, minimum_eligible_tasks)
        directory = output / 'scenes' / old['scene_id']
        selection = {'schema': VERSION, 'parent_scene': old, 'existing_evaluated_proposals': candidates,
            'selected': selected, 'selection_rule': 'maximize_qualified_goal_count_tie_lowest_original_proposal',
            'no_new_layout_or_endpoint_search': True, 'physical_results_read': False}
        write(directory / 'proposal_selection.json', selection)
        if selected is None:
            # There is no evaluated geometry to reconstruct. Keep the source
            # request and all rejection identities, never fabricate a scene.
            request = read(resolve(old['request'])); write(directory / 'request.json', request)
            admission = {t['id']: {'eligible': False, 'reason': 'no_existing_endpoint_qualified_layout',
                                   'failed_operations': [], 'endpoint_qualification': None,
                                   'endpoint_task_id': t['id'], 'operation_reports': []} for t in request['tasks']}
            scene = tasks = checks = None; admitted = False
        else:
            request = read(resolve(selected['request'])); proof = read(resolve(selected['endpoint_IK']))
            if not read(resolve(selected['workspace']))['passed']:
                raise ValueError('existing_endpoint_proposal_lacks_workspace_gate')
            selected_entries = [entries[o['source_id']] for o in request['objects']]
            selected_bins = [bin_entries[c['source_id']] for c in request['containers']]
            scene, tasks, checks = retained.compile_scene_sequential(request, selected_entries, selected_bins, selected['seed'])
            verify_selected_endpoint_bindings(scene, tasks, proof)
            decisions = summarize_proof(request, proof); admitted = selected['scene_admitted']
            admission = {}
            for short, decision in decisions.items():
                eligible = admitted and decision['endpoint_passed']
                reason = None if eligible else 'scene_has_fewer_than3_endpoint_qualified_goals' if not admitted else 'endpoint_IK_or_container_hand_sweep_failed'
                admission[short] = {**decision, 'eligible': eligible, 'reason': reason,
                    'endpoint_qualification': selected['endpoint_IK'], 'endpoint_task_id': short}
            write(directory / 'request.json', request); write(directory / 'compiled_scene.json', scene)
            write(directory / 'compiled_tasks.json', {'tasks': tasks}); write(directory / 'geometry_checks.json', checks)
            sampling = read(resolve(selected['sampling']))
            metadata = {}
            for task in tasks:
                short = task['requested_task_id']; vacancy = checks['sequential_vacancy'] if short == 'clear_then_place' else None
                metadata[short] = {'structure': short, 'operation_count': len(task['operations']),
                    'required_object_precedence': vacancy['required_object_precedence'] if vacancy else [],
                    'initial_target_occupied_by': vacancy['initial_target_occupied_by'] if vacancy else None,
                    'goal_signature': retained.goal_signature(next(t for t in request['tasks'] if t['id'] == short)),
                    'placement_policy': 'size_only_center_line_triangle_square', 'ordered_only_if_task_requires': True}
            write(directory / 'task_metadata.json', {'tasks': metadata})
            # For the six all-eight v3 scenes the earliest maximum is exactly
            # the accepted v3 scene. Recompilation must be byte-equivalent.
            if old['accepted_endpoint_IK']:
                if old['accepted']['proposal'] != selected['proposal'] or read(resolve(old['compiled_scene'])) != scene:
                    raise ValueError('all8_admitted_scene_changed_during_policy_only_transition')
        write(directory / 'task_admission.json', {'schema': VERSION, 'tasks': admission,
            'minimum_eligible_tasks_per_scene': minimum_eligible_tasks, 'scene_admitted': admitted,
            'raw_endpoint_eligible_count': selected['qualified_task_count'] if selected else 0,
            'selection': binding(directory / 'proposal_selection.json'), 'requested_task_count': len(request['tasks']),
            'eligible_task_count': sum(d['eligible'] for d in admission.values()), 'original_requests_removed': 0})
        for task in request['tasks']:
            count = len(task['placements']); operation_requests += count
            if admission[task['id']]['eligible']: eligible_count += 1; eligible_operations += count
            else: task_rejections[admission[task['id']]['reason']] += 1
        row = {**old, 'directory': str(directory), 'request': binding(directory / 'request.json'),
            'accepted_geometry': scene is not None, 'accepted_endpoint_IK': admitted,
            'all_requested_tasks_endpoint_IK_passed': bool(selected and selected['qualified_task_count'] == old['requested_tasks']),
            'compiled_scene': binding(directory / 'compiled_scene.json') if scene else None,
            'compiled_tasks': binding(directory / 'compiled_tasks.json') if tasks else None,
            'task_metadata': binding(directory / 'task_metadata.json') if tasks else None,
            'task_admission': binding(directory / 'task_admission.json'),
            'admitted_task_count': sum(d['eligible'] for d in admission.values()),
            'eligible_task_ids': [t for t, d in admission.items() if d['eligible']],
            'preflight_rejected_task_ids': [t for t, d in admission.items() if not d['eligible']],
            'accepted': selected, 'selection': binding(directory / 'proposal_selection.json'),
            'failure': None if admitted else 'insufficient_taskwise_endpoint_qualification',
            'physical_status': 'not_run'}
        rows.append(row)
    write(output / 'scenes.json', {'scenes': rows})
    status = {'schema': VERSION, 'requested_scenes': len(rows), 'requested_tasks': sum(r['requested_tasks'] for r in rows),
        'admitted_scenes': sum(r['admitted_task_count'] >= minimum_eligible_tasks for r in rows),
        'eligible_tasks': eligible_count, 'ineligible_tasks': sum(r['requested_tasks'] for r in rows) - eligible_count,
        'requested_operation_instances': operation_requests, 'eligible_operation_instances': eligible_operations,
        'task_rejection_reasons': dict(task_rejections), 'seconds': time.monotonic() - started,
        'new_layout_proposals': 0, 'new_endpoint_IK_calls': 0, 'physical_executions': 0, 'GPU_jobs': 0, 'API_calls': 0,
        'selection_scope': 'existing nominal endpoint evidence only; physical success remains unmeasured'}
    write(output / 'status.json', status)
    return status


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); print(build(args.source, args.output))
