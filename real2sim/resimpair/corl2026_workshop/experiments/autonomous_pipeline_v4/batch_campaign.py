"""Bounded scene/task collection with immutable inputs and resumable episodes.

Preparation makes no provider calls. Simulation collection and the human queue
are independent; a completed worker is not a physical or human quality verdict.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import time

from .core import activate, bind, checked, read, require, replace_state, write


def register(root, scene, task, phase, group):
    from .library import task_semantic_key
    from .batch_protocols import protocol
    protocol(scene, task)  # Reject invalid task declarations before reserving GPU.
    sid = scene['scene_instance_id']
    sp = root / 'scenes' / (sid + '.json')
    tp = root / 'tasks' / (task['id'] + '.json')
    write(sp, scene); write(tp, task)
    case = {'id': task['id'], 'scene': bind(sp), 'task': bind(tp),
            'semantic_key': task_semantic_key(task), 'partition': scene['partition'],
            'split_group': sid, 'scene_instance_id': sid, 'source_session_ids': [],
            'source_coverage': 'missing', 'request_human_reference': False,
            'reference_policy': 'separate_qualified_native_queue; no fabricated_real_source',
            'planner_plugin': 'batch_parameterized_mechanisms_v1', 'phase': phase,
            'group': group}
    write(root / 'registered_cases' / (case['id'] + '.json'), case)
    return case


def prepare(root, *, preflight_only=False):
    """Prepare a new campaign from explicitly bound, observed budget lineage.

    ``preflight_only`` reads local evidence without creating any files. An
    unresolved ancestor permits an immutable usage observation only, never a
    callable descendant budget. Historical usage reports are not call counters.
    """
    from contextlib import ExitStack
    from .core import digest, identifier
    from .batch_sampling import KINDS, sample_scene, enumerate_tasks, vary_task
    from .contact_extensions import KINDS as CONTACT_KINDS, scene as contact_scene, task as contact_task
    from .budget import Budget, counts, KINDS as CALL_KINDS, REGISTRY
    from .billing import validate_retired_billed_review
    root = Path(root).resolve()
    require(not (root / 'experiment.json').exists(), 'batch_already_prepared_do_not_reset')
    registration = read(root / 'registration.json')
    require(isinstance(registration.get('campaign_id'), str) and registration['campaign_id'],
            'explicit_batch_campaign_id_required')
    campaign_id = identifier(registration['campaign_id'])
    require(not (REGISTRY / (campaign_id + '.json')).exists(), 'batch_campaign_id_already_registered')
    ancestor_binding = registration.get('ancestor_budget')
    require(isinstance(ancestor_binding, dict), 'explicit_ancestor_budget_binding_required')
    ancestor_path = checked(ancestor_binding)
    require(ancestor_path.parent != root, 'batch_ancestor_must_be_a_previous_campaign')
    previous = read(ancestor_path)
    require(previous.get('schema') == 'resimpair_cumulative_budget_v4', 'ancestor_v4_budget_required')
    identifier(previous.get('campaign_id'))
    require(previous['campaign_id'] != campaign_id, 'batch_ancestor_campaign_id_must_differ')
    for key in ['ancestor_started_calls', 'new_call_limits', 'lineage_cumulative_limits']:
        values = previous.get(key)
        require(isinstance(values, dict) and set(values) == CALL_KINDS
                and all(type(v) is int and v >= 0 for v in values.values()),
                'ancestor_nonnegative_integer_call_counts_required:' + key)
    require(all(previous['lineage_cumulative_limits'][k]
                == previous['ancestor_started_calls'][k] + previous['new_call_limits'][k]
                for k in CALL_KINDS), 'ancestor_cumulative_limits_inconsistent')
    require(previous.get('max_concurrent') == 1
            and type(previous.get('timeout_seconds')) is int
            and 1 <= previous['timeout_seconds'] <= 600, 'ancestor_bounded_serial_budget_required')
    require(isinstance(previous.get('ancestor_evidence'), list), 'ancestor_evidence_list_required')
    for bound in previous['ancestor_evidence']:
        checked(bound)
    require(isinstance(previous.get('ledger_path'), str), 'ancestor_ledger_path_required')
    ledger = Path(previous['ledger_path'])
    require(ledger.is_absolute() and ledger == ledger.resolve() and ledger.is_dir(),
            'ancestor_existing_canonical_ledger_required')
    require(ledger != root / 'campaign_ledger', 'batch_cannot_reuse_ancestor_ledger')
    require((ledger / 'budget.json').is_file() and read(ledger / 'budget.json') == previous,
            'ancestor_ledger_budget_mismatch')
    # Do not instantiate Budget for the ancestor: that constructor writes a
    # ledger/registry. Share its existing execution lock read-only when present.
    with ExitStack() as guards:
        lock_path = ledger / '.execution.lock'
        if lock_path.exists():
            lock = guards.enter_context(lock_path.open('r'))
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        markers = sorted(ledger.glob('stages/*/call_started.json'))
        used = counts(ledger)
        unresolved = {k: 0 for k in sorted(CALL_KINDS)}
        failures = {k: 0 for k in sorted(CALL_KINDS)}
        original_unknown = {k: 0 for k in sorted(CALL_KINDS)}
        rows = []
        for marker in markers:
            started = read(marker)
            kind, stage = started['kind'], marker.parent
            require(started.get('budget') == ancestor_binding, 'ancestor_started_call_budget_mismatch')
            row = {'kind': kind, 'call_started': bind(marker), 'counted': True}
            for name in ['result', 'failure', 'video_terminal', 'terminal_resolution']:
                if (stage / (name + '.json')).exists():
                    row[name] = bind(stage / (name + '.json'))
            failure = read(checked(row['failure'])) if 'failure' in row else {}
            if failure:
                failures[kind] += 1
                original_unknown[kind] += int(failure.get('provider_outcome') == 'unknown')
                row['original_provider_failure_outcome'] = failure.get('provider_outcome', 'unclassified')
            if 'result' in row:
                state = 'result_recorded'
                if kind == 'video':
                    result = read(checked(row['result']))
                    terminal = read(checked(row['video_terminal'])) if 'video_terminal' in row else {}
                    state = ('video_terminal' if terminal.get('status') in
                             {'succeeded', 'failed', 'expired', 'cancelled'}
                             and terminal.get('task_id') == result.get('task_id')
                             and bool(result.get('task_id')) else 'video_not_terminal')
            elif failure.get('provider_outcome') == 'definite_failure':
                state = 'definite_failure'
            else:
                state = 'submission_outcome_unknown'
                try:
                    retired = validate_retired_billed_review(stage)
                except (ValueError, KeyError, TypeError, FileNotFoundError) as exc:
                    retired = False
                    row['retirement_validation_error_type'] = type(exc).__name__
                if retired:
                    state = 'billed_terminal_review_output_unavailable'
            row['submission_state'] = state
            row['unresolved'] = state in {'submission_outcome_unknown', 'video_not_terminal'}
            unresolved[kind] += int(row['unresolved'])
            rows.append(row)
        require(used == {k: sum(row['kind'] == k for row in rows) for k in CALL_KINDS}
                and markers == sorted(ledger.glob('stages/*/call_started.json')),
                'ancestor_ledger_changed_during_snapshot')
        for row in rows:
            for key in ['call_started', 'result', 'failure', 'video_terminal', 'terminal_resolution']:
                if key in row:
                    checked(row[key])
        checked(ancestor_binding)
        lineage = {k: previous['ancestor_started_calls'][k] + used[k] for k in sorted(CALL_KINDS)}
        usage = {'schema': 'resimpair_ancestor_usage_snapshot_v1',
                 'budget': ancestor_binding, 'ledger_budget': bind(ledger / 'budget.json'),
                 'registration': bind(root / 'registration.json'),
                 'ancestor_started_calls': previous['ancestor_started_calls'],
                 'new_started': used, 'lineage_started': lineage,
                 'failure_recorded_calls': failures, 'original_unknown_failure_calls': original_unknown,
                 'unresolved_started_calls': unresolved, 'started_calls': rows,
                 'counting_source': 'actual ledger call_started markers; all failed/unknown calls retain their count',
                 'snapshot_only_not_ancestor_closure': True,
                 'requires_ancestor_recovery': any(unresolved.values()),
                 'old_terminal_campaigns_not_reopened': True}
    if preflight_only:
        return {'status': 'requires_ancestor_recovery' if usage['requires_ancestor_recovery'] else 'ancestor_ready',
                'ancestor_usage': usage, 'new_calls_started': 0, 'files_written': 0}
    require(shutil.disk_usage(root.parent).free > 100 * 2**30, 'insufficient_batch_disk_space')
    # Each unresolved observation is immutable. Once original submissions are
    # recovered, a later prepare can use a new observation without overwriting
    # the earlier unknown record or resetting any started-call count.
    if usage['requires_ancestor_recovery']:
        observation = root / 'ancestor_usage_observations' / (digest(usage) + '.json')
        write(observation, usage)
        raise ValueError('requires_ancestor_recovery:' + str(observation))
    ancestor = ancestor_path.parent
    config = read(ancestor / 'config.json')
    require(not (REGISTRY / (campaign_id + '.json')).exists(), 'batch_campaign_id_already_registered')
    activate()
    write(root / 'ancestor_usage.json', usage)
    limits = {'image': 4, 'video': 8, 'review': 16}
    budget_spec = {'schema': 'resimpair_cumulative_budget_v4',
        'campaign_id': campaign_id, 'ledger_path': str(root / 'campaign_ledger'),
        'new_call_limits': limits, 'ancestor_started_calls': lineage,
        'ancestor_evidence': [bind(root / 'ancestor_usage.json')],
        'lineage_cumulative_limits': {k: lineage[k]+limits[k] for k in lineage},
        'max_concurrent': 1, 'timeout_seconds': 600, 'automatic_retry': False,
        'currency_budget': None, 'observed_quota_stop_limit': 30000000,
        'nominal_usd_admission_limit': 60,
        'reserve_quota_before_new_call': {'image': 500000, 'video': 8000000, 'review': 500000},
        'authorization': 'Standing user authorization for task-data generation; finite new campaign, billed/unknown submissions retained; no semantic retries.'}
    # Reserve the explicit campaign ID under the same lock as Budget. The
    # early checks above give a cheap failure; this one also prevents two new
    # roots from racing those checks and writing budgets for the same ID.
    REGISTRY.mkdir(parents=True, exist_ok=True)
    with (REGISTRY / '.registration.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (REGISTRY / (campaign_id + '.json')).exists(), 'batch_campaign_id_already_registered')
        write(REGISTRY / (campaign_id + '.json'), budget_spec)
        write(root / 'budget.json', budget_spec)
    write(root / 'config.json', config)
    write(root / 'experiment.json', {'schema': 'resimpair_parameterized_batch_v1',
        'sources': [], 'cases': [], 'simulation_limit': 400, 'simulation_timeout_seconds': 240,
        'gpu': 2, 'gpu_uuid': 'GPU-807bd44d-36ea-31d0-427c-50e264428c67',
        'protected_gpus': [0, 1, 3], 'no_training': True, 'automatic_retry': False,
        'source_frame_budget': 24, 'review_frame_budget': 49,
        'phase_gates': 'Eight first cases before expansion; independently audit every admitted episode; frozen new release for revisions.',
        'physical_caps': {'base': 120, 'qualified_base_trajectory_variants': 240, 'development_and_new_contacts': 40},
        'scope': 'Native mechanism/task coverage; unpaired synthetic probes until verified real source supplied.',
        'human_queue_separate': True, 'registration': bind(root / 'registration.json'),
        'source_independent_holdout_claimed': False, 'new_api_limits': limits})
    groups, smoke = [], []
    for ki, kind in enumerate(KINDS):
        for layout in range(5):
            spec = sample_scene(kind, 2026102500 + ki*100 + layout, layout)
            members = []
            for ti, task in enumerate(enumerate_tasks(spec)):
                task = vary_task(task, 0)
                row = register(root, spec, task, 'base', kind)
                members.append(row)
                if layout == 0 and ti == 0: smoke.append(row)
            groups.append({'kind': kind, 'layout_index': layout, 'cases': members,
                           'split_group': spec['scene_instance_id'], 'partition': spec['partition']})
    contacts = []
    for ki, kind in enumerate(CONTACT_KINDS):
        for layout in range(2):
            spec = contact_scene(kind, 2026102501 + ki*100 + layout)
            spec.update(scene_instance_id=f'{kind}_L{layout}', split_group=f'{kind}_L{layout}',
                        partition='development' if layout == 0 else 'heldout_layout')
            members = []
            for ti in range(3):
                task = contact_task(spec, ti)
                task['id'] = spec['scene_instance_id'] + '__t' + str(ti)
                row = register(root, spec, task, 'contact', kind)
                members.append(row)
                if layout == 0 and ti == 0: smoke.append(row)
            contacts.append({'kind': kind, 'layout_index': layout, 'cases': members,
                             'split_group': spec['scene_instance_id'], 'partition': spec['partition']})
    write(root / 'cohort.json', {'groups': groups, 'contact_groups': contacts,
        'smoke_case_ids': [r['id'] for r in smoke], 'base_cases': 120,
        'contact_cases': 12, 'same_initial_scene_reset_per_task': True,
        'failed_inputs_remain_in_denominator': True, 'human_generation_is_separate': True,
        'source_coverage': 'missing', 'formal_pair_count': 0})
    Budget(root / 'budget.json')
    return {'root': str(root), 'frozen_base_cases': 120, 'contact_cases': 12,
            'smoke_cases': [r['id'] for r in smoke], 'new_call_limits': limits}


def select(root, phase):
    from .intake import cases
    available = {c['id']: c for c in cases(root)}
    cohort = read(root / 'cohort.json')
    if phase == 'smoke': return [available[cid] for cid in cohort['smoke_case_ids']]
    if phase.startswith('layout_'):
        layout = int(phase.split('_')[1])
        return [c for g in cohort['groups'] if g['layout_index'] == layout for c in g['cases']]
    if phase == 'contacts': return [c for g in cohort['contact_groups'] for c in g['cases']]
    if phase == 'base': return [c for g in cohort['groups'] for c in g['cases']]
    if phase == 'variants': return [c for c in available.values() if c.get('phase') == 'variants']
    raise ValueError('unknown_batch_phase')


def register_variants(root, audit, max_base=6):
    from .batch_sampling import vary_task
    root = Path(root).resolve(); report = read(audit)
    require(report['schema'] == 'batch_native_independent_acceptance_20261025_v1'
            and Path(report['root']).resolve() == root, 'bound_independent_batch_audit_required')
    require(report['batch_ready_for_sampling_within_verified_scope'], 'batch_audit_not_ready')
    require(type(max_base) is int and 1 <= max_base <= 120, 'invalid_variant_base_cap')
    eligible = [r for r in report['cases'] if r['independent_qualified']
                and r['case_record']['phase'] == 'base']
    # First take one per mechanism, then the remaining bound cases. Outcomes
    # select qualified inputs; failed attempts stay in the full audit denominator.
    ordered = []; seen = set()
    for row in eligible:
        key = row['case_record']['group']
        if key not in seen: ordered.append(row); seen.add(key)
    ordered += [r for r in eligible if r not in ordered]
    created = []
    for row in ordered[:max_base]:
        case = row['case_record']; scene = read(checked(case['scene']))
        task = read(checked(case['task']))
        for i in [1, 2]:
            variant = vary_task(task, i)
            created.append(register(root, scene, variant, 'variants', case['group']))
    write(root / 'variant_selection.json', {'independent_audit': bind(audit), 'cases': created,
        'changed_goal': False, 'new_dynamic_trajectory_claim_requires_measured_fingerprint': True,
        'same_split_group_as_base': True, 'failed_base_cases_not_replaced': True})
    return {'registered_variants': len(created), 'native_executions_started': 0}


def run(root, phase, case_ids=(), gate_audit=None):
    activate()
    from .pipeline import release_id
    from .simulation import launch
    from .novel_campaign import qualify
    from .intake import cases
    root = Path(root).resolve()
    release = release_id()
    available = {c['id']: c for c in cases(root)}
    selected = select(root, phase) if not case_ids else [available[cid] for cid in case_ids if cid in available]
    require(selected and (not case_ids or len(selected) == len(set(case_ids))), 'batch_case_selection_invalid')
    gate = None
    if phase != 'smoke':
        require(gate_audit is not None, 'non_smoke_batch_requires_independent_gate')
        gate = read(gate_audit)
        require(gate['schema'] == 'batch_native_independent_acceptance_20261025_v1'
                and Path(gate['root']).resolve() == root
                and gate['batch_ready_for_sampling_within_verified_scope'], 'invalid_batch_expansion_gate')
        holdback_path = root / 'admission_holdbacks.json'
        heldback = set()
        if holdback_path.exists():
            updates = read(holdback_path); checked(updates['evidence'])
            heldback = {(r['case'], r['attempt']) for r in updates['case_attempts']}
        admitted = {r['case_record']['group'] for r in gate['cases'] if r['independent_qualified']
                    and (r['case'], r['attempt']) not in heldback}
        require(all(c['group'] in admitted for c in selected), 'unqualified_mechanism_must_use_new_explicit_smoke_case')
    run_id = phase if not case_ids else phase + '_' + __import__('hashlib').sha256(','.join(sorted(case_ids)).encode()).hexdigest()[:8]
    write(root / 'run_plans' / (run_id + '.json'), {'release': release,
          'release_manifest': bind(root / 'releases' / release / 'release.json'),
          'cases': selected, 'attempt': 'attempt_01', 'max_attempts_per_case': 1,
          'evaluator_manifest': bind(root / 'evaluator_bindings.json'),
          'expansion_gate': bind(gate_audit) if gate_audit else None})
    with (root / '.batch_collection.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = []
        for case in selected:
            result_path = root / 'batch_results' / release / (case['id'] + '.json')
            prior = sorted((root / 'batch_results').glob('*/' + case['id'] + '.json'))
            require(len(prior) <= 1, 'ambiguous_prior_batch_case_results')
            if prior:
                preserved = read(prior[0])
                require(read(checked(preserved['qualification']))['case'] == case, 'prior_batch_inputs_changed')
                records.append(preserved); continue
            replace_state(root / 'batch_progress.json', {'status': 'running', 'pid': os.getpid(),
                'phase': phase, 'release': release, 'case': case['id'],
                'completed_this_selection': len(records), 'selected_cases': len(selected), 'updated_unix': time.time()})
            started = time.monotonic()
            try:
                launch(root, case['id'], 'attempt_01')
                result = qualify(root, case, 'attempt_01', release)
            except Exception as exc:
                replace_state(root / 'batch_progress.json', {'status': 'stopped_requires_recovery',
                    'pid': None, 'phase': phase, 'release': release, 'case': case['id'],
                    'completed_this_selection': len(records), 'selected_cases': len(selected),
                    'exception_type': type(exc).__name__, 'updated_unix': time.time(),
                    'resume_only_original_reservation_if_present': True, 'automatic_retry': False})
                raise
            row = {'case': case['id'], 'release': release, 'native_qualified': result['native_qualified'],
                   'qualification': bind(root / 'qualification' / release / case['id'] / 'result.json'),
                   'elapsed_seconds': time.monotonic()-started, 'human_generated': False,
                   'independent_audit_pending': True, 'formal_pair': False, 'errors': result['errors']}
            write(result_path, row); records.append(row)
            print(json.dumps({'case': row['case'], 'native_qualified': row['native_qualified'],
                              'seconds': round(row['elapsed_seconds'], 2), 'errors': row['errors']}, ensure_ascii=False), flush=True)
        result = {'status': 'selected_batch_terminal', 'phase': phase, 'release': release,
                  'selected_cases': len(records), 'native_qualified': sum(r['native_qualified'] for r in records),
                  'independent_audit_required_before_admission': True, 'records': records}
        write(root / 'phase_results' / (run_id + '.json'), result)
        replace_state(root / 'batch_progress.json', {'status': 'terminal', 'pid': os.getpid(),
            'phase': phase, 'selected_cases': len(records), 'native_qualified': result['native_qualified'],
            'updated_unix': time.time()})
        return {k: v for k, v in result.items() if k != 'records'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'variants'])
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--phase', default='smoke')
    parser.add_argument('--case-id', action='append', default=[])
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--max-base', type=int, default=6)
    args = parser.parse_args()
    result = (prepare(args.root) if args.action == 'prepare' else
              register_variants(args.root, args.audit, args.max_base) if args.action == 'variants' else
              run(args.root, args.phase, args.case_id, args.audit))
    print(json.dumps(result, ensure_ascii=False))
