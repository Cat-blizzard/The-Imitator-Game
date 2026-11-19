"""Activate an explicit finite subset of a frozen, zero-budget preparation.

The old preparation remains immutable. This module only registers new budget
and provenance, then delegates CPU execution to whole_path_campaign_v1. It
does not select layouts, recompute IK, generate data, or implement a controller.
"""
import argparse
import copy
import fcntl
from pathlib import Path
import time

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write
from . import random_pickplace_v2 as prepared_adapter
from . import whole_path_campaign_v1 as execution

VERSION = 'prepared_batch_execution_v1'
MAX_REQUESTS = 96
REGISTRY = ROOT / 'runs/prepared_batch_execution_registry_v1'


def bindings_in(value):
    """Find explicit file receipts, including scene/cache/appearance resources."""
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
            yield value
        else:
            for child in value.values():
                yield from bindings_in(child)
    elif isinstance(value, list):
        for child in value:
            yield from bindings_in(child)


def verify_preparation(prepared):
    from resimpair_pickplace.runner import verify
    prepared = Path(prepared).resolve(); folder = prepared / 'candidate'
    spec = verify(folder)
    protocol, cohort = read(folder / 'protocol.json'), read(folder / 'cohort.json')
    frozen = read(folder / 'frozen_inputs.json')
    for ref in frozen.values(): resolve(ref)
    budget = read(prepared / 'budget/limits.json')
    if (spec.get('schema') != prepared_adapter.VERSION or protocol.get('schema') != prepared_adapter.VERSION
            or spec.get('launch_enabled') is not False or protocol.get('launch_enabled') is not False
            or cohort.get('launch_enabled') is not False or budget.get('launch_enabled') is not False
            or budget.get('physical_attempt_limit') != 0 or budget.get('new_physical_authorization') is not False):
        raise ValueError('frozen_zero_budget_preparation_required')
    if any(budget.get(key) != 0 for key in ('api_limit', 'gpu_jobs')):
        raise ValueError('CPU_only_zero_API_preparation_required')
    if (protocol['driver'] != binding(execution.__file__)
            or read(folder / 'rules.json') != read(resolve(protocol['original_rules']))):
        raise ValueError('original_worker_and_independent_rules_required')
    dependencies = protocol['factory_dependencies']
    factory = resolve(dependencies['factory_protocol']).parent
    if prepared_adapter.factory_chain(factory) != dependencies:
        raise ValueError('factory_catalog_role_compiler_binding_mismatch')
    if any(frozen.get(key) != value for key, value in dependencies.items()):
        raise ValueError('factory_dependencies_not_frozen_in_preparation')
    for key, name in [('experiment', 'experiment.json'), ('cohort', 'cohort.json'), ('protocol', 'protocol.json')]:
        if frozen.get(key) != binding(folder / name):
            raise ValueError('prepared_root_binding_mismatch')
    if (frozen.get('budget') != binding(prepared / 'budget/limits.json')
            or protocol['shared_budget'] != frozen['budget']):
        raise ValueError('prepared_budget_binding_mismatch')
    if list((prepared / 'budget/reservations').glob('*.json')):
        raise ValueError('zero_budget_preparation_has_execution_reservations')
    return spec, protocol, cohort, frozen, budget


def select_cases(prepared, cohort, case_ids, independent):
    if (not case_ids or len(case_ids) > MAX_REQUESTS or len(case_ids) != len(set(case_ids))):
        raise ValueError('explicit_unique_case_ids_required_at_most_96')
    cohort_ref = binding(Path(prepared) / 'candidate/cohort.json')
    if independent['source_prepared_cohort'] != cohort_ref:
        raise ValueError('independent_protocol_prepared_cohort_mismatch')
    frozen = read(resolve(independent['cases']))
    if frozen['source_cohort'] != cohort_ref or frozen.get('selected_before_execution') is not True:
        raise ValueError('independent_selection_must_be_frozen_before_execution')
    rows = frozen['cases']
    if [row['case_id'] for row in rows] != list(case_ids):
        raise ValueError('explicit_cases_differ_from_independently_frozen_selection')
    result = []
    for row in rows:
        source = cohort['cases'][row['source_prepared_case_index']]
        if source != row['source_prepared_case'] or source['id'] != row['case_id']:
            raise ValueError('independent_selected_case_payload_mismatch')
        for key in ('source_request', 'requested_task_id', 'source_ids', 'split_group'):
            if source[key] != row[key]: raise ValueError('source_request_and_split_must_not_change')
        if source['partition'] != row['dataset_partition'] or source.get('launch_enabled') is not False:
            raise ValueError('source_partition_or_zero_authority_mismatch')
        for ref in bindings_in(source): resolve(ref)
        case = copy.deepcopy(source)
        case['launch_enabled'] = True
        case['experiment_role'] = row['experiment_role']
        result.append(case)
    return result


def base_lineage(budget):
    refs = budget['existing_scope_budget_references']
    allocations = {name: read(resolve(ref))['physical_attempt_limit'] for name, ref in refs.items()}
    if (allocations != budget['existing_scope_allocation']
            or sum(allocations.values()) != budget['scope_cumulative_attempt_limit']):
        raise ValueError('existing_scope_budget_lineage_mismatch')
    for ref in refs.values():
        value = read(resolve(ref))
        if value.get('api_limit') != 0 or value.get('gpu_jobs') != 0:
            raise ValueError('CPU_only_zero_API_budget_lineage_required')
    return refs, allocations


def authorize(root, prepared, independent, old_budget, cases, physical_limit, registered):
    """Check explicit increment and prior receipts; never reuse unspent limits."""
    if (type(physical_limit) is not int or not 1 <= physical_limit <= MAX_REQUESTS
            or physical_limit != len(cases)
            or independent['new_physical_attempt_limit'] != physical_limit
            or independent['new_request_limit'] != len(cases)):
        raise ValueError('finite_one_attempt_per_selected_request_required')
    expected = Path(independent['expected_output_root'])
    expected = expected if expected.is_absolute() else ROOT / expected
    if Path(root).resolve() != expected.resolve():
        raise ValueError('independently_frozen_output_root_required')
    for key, required in [('api_limit', 0), ('gpu_jobs', 0), ('parallel_workers', 1), ('render_backend', 'none')]:
        if independent.get(key) != required:
            raise ValueError('only_serial_CPU_no_render_execution_supported')
    refs, allocations = base_lineage(old_budget)
    scope_key = digest({'references': refs, 'allocation': allocations})
    same_scope = [r for r in registered if r['scope_key'] == scope_key]
    cohort_ref = binding(Path(prepared) / 'candidate/cohort.json')
    selected = {c['id'] for c in cases}
    for receipt in registered:
        if receipt['source_prepared_cohort'] == cohort_ref and selected.intersection(receipt['case_ids']):
            raise ValueError('selected_prepared_request_already_activated_recover_original_campaign')
    prior_limit = sum(allocations.values()) + sum(r['physical_attempt_limit'] for r in same_scope)
    if (independent['prior_round_limit_unchanged'] != prior_limit
            or independent['total_current_round_physical_limit_after_explicit_increment'] != prior_limit + physical_limit):
        raise ValueError('incremental_scope_limit_must_include_all_prior_registered_campaigns')
    return {'schema': VERSION, 'physical_attempt_limit': physical_limit, 'launch_enabled': True,
        'api_limit': 0, 'gpu_jobs': 0, 'parallel_workers': 1,
        'per_attempt_timeout_seconds': independent['per_attempt_timeout_seconds'],
        'existing_scope_budget_references': refs, 'existing_scope_allocation': allocations,
        'existing_scope_reservations_observed': {
            name: len(list((resolve(ref).parent / 'reservations').glob('*.json'))) for name, ref in refs.items()},
        'prior_activation_receipts': [r['_binding'] for r in same_scope],
        'prior_round_limit_unchanged': prior_limit, 'scope_cumulative_attempt_limit': prior_limit + physical_limit,
        'scope_key': scope_key, 'new_physical_authorization': True,
        'unspent_old_allowance_not_reassigned': True,
        'retries_per_request': 0, 'api_calls': 0}


def prepare(prepared, root, independent, case_ids, physical_limit, *, registry=REGISTRY):
    prepared, root, independent, registry = (Path(p).resolve() for p in (prepared, root, independent, registry))
    if root.exists(): raise ValueError('existing_campaign_must_be_preserved')
    parent, old_protocol, cohort, frozen, old_budget = verify_preparation(prepared)
    evaluation = read(independent)
    for ref in bindings_in(evaluation): resolve(ref)
    cases = select_cases(prepared, cohort, case_ids, evaluation)
    # Bind actual selected resources as well as the already frozen parent factory.
    resources = {}
    for case in cases:
        for ref in bindings_in(case):
            path = resolve(ref); resources[digest(ref)] = ref
            if path.suffix == '.json':
                for dependency in bindings_in(read(path)):
                    resolve(dependency); resources[digest(dependency)] = dependency
    registry.mkdir(parents=True, exist_ok=True)
    with (registry / '.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipts = []
        for path in sorted((registry / 'registrations').glob('*.json')):
            receipt = read(path); resolve(receipt['budget'])
            receipts.append({**receipt, '_binding': binding(path)})
        budget = authorize(root, prepared, evaluation, old_budget, cases, physical_limit, receipts)
        budget['independent_protocol'] = binding(independent)
        budget['source_zero_budget'] = binding(prepared / 'budget/limits.json')
        write(root / 'budget/limits.json', budget)
        out = root / 'candidate'
        spec = copy.deepcopy(parent)
        spec.update(schema=VERSION, budget=str(root / 'budget'), launch_enabled=True,
            source_preparation=binding(prepared / 'candidate/experiment.json'),
            independent_protocol=binding(independent))
        spec['code']['resimpair_tasks/prepared_batch_execution_v1.py'] = binding(__file__)
        write(out / 'experiment.json', spec)
        write(out / 'rules.json', read(prepared / 'candidate/rules.json'))
        write(root / 'selection.json', {'schema': VERSION, 'source_prepared_cohort': binding(prepared / 'candidate/cohort.json'),
            'case_ids': case_ids, 'independent_selection': evaluation['cases'],
            'input_files_copied_or_modified': False, 'new_IK_calls': 0, 'outcome_based_replacement': False})
        write(out / 'cohort.json', {'cases': cases, 'selection': binding(root / 'selection.json'),
            'source_prepared_cohort': binding(prepared / 'candidate/cohort.json'),
            'outcome_based_replacement': False, 'launch_enabled': True})
        receipt = {'schema': VERSION, 'scope_key': budget['scope_key'], 'created_unix': time.time(),
            'campaign_root': str(root), 'source_prepared_cohort': binding(prepared / 'candidate/cohort.json'),
            'case_ids': case_ids, 'physical_attempt_limit': physical_limit,
            'scope_cumulative_attempt_limit': budget['scope_cumulative_attempt_limit'],
            'budget': binding(root / 'budget/limits.json'), 'independent_protocol': binding(independent)}
        receipt_path = registry / 'registrations' / (digest({'root': str(root), 'source': receipt['source_prepared_cohort']}) + '.json')
        write(receipt_path, receipt)
        write(out / 'protocol.json', {'schema': VERSION, 'driver': old_protocol['driver'],
            'batch_driver': binding(__file__), 'arm': 'candidate', 'launch_enabled': True,
            'factory_schema': old_protocol['factory_schema'], 'factory_dependencies': old_protocol['factory_dependencies'],
            'shared_budget': binding(root / 'budget/limits.json'), 'original_rules': old_protocol['original_rules'],
            'source_preparation': binding(prepared / 'candidate/protocol.json'),
            'activation_receipt': binding(receipt_path), 'independent_protocol': binding(independent)})
        write(out / 'frozen_inputs.json', {
            'experiment': binding(out / 'experiment.json'), 'cohort': binding(out / 'cohort.json'),
            'protocol': binding(out / 'protocol.json'), 'budget': binding(root / 'budget/limits.json'),
            'rules': binding(out / 'rules.json'),
            'selection': binding(root / 'selection.json'), 'activation_receipt': binding(receipt_path),
            'independent_protocol': binding(independent), 'independent_cases': evaluation['cases'],
            'source_frozen_inputs': binding(prepared / 'candidate/frozen_inputs.json'),
            **{'prepared_' + k: v for k, v in frozen.items()},
            **{'selected_resource_' + k: v for k, v in resources.items()},
            **{'evaluation_' + str(i): v for i, v in enumerate(bindings_in(evaluation))}})
        result = {'schema': VERSION, 'registered_requests': len(cases), 'physical_attempt_limit': physical_limit,
            'scope_cumulative_attempt_limit': budget['scope_cumulative_attempt_limit'],
            'new_physical_attempts': 0, 'api_calls': 0, 'gpu_jobs': 0, 'new_endpoint_IK_calls': 0,
            'original_inputs_unchanged': True, 'source_zero_budget_unchanged': True,
            'candidate_default_promoted': False, 'controller_reused': parent['candidate_motion_override']}
        write(root / 'prepared.json', result)
    verify(root)
    return result


def verify(root):
    from resimpair_pickplace.runner import verify as verify_worker
    root = Path(root).resolve(); out = root / 'candidate'; spec = verify_worker(out)
    for ref in read(out / 'frozen_inputs.json').values(): resolve(ref)
    protocol, cohort = read(out / 'protocol.json'), read(out / 'cohort.json')
    budget = read(resolve(protocol['shared_budget']))
    if (spec['schema'] != VERSION or protocol['schema'] != VERSION or spec.get('launch_enabled') is not True
            or protocol.get('launch_enabled') is not True or budget.get('launch_enabled') is not True):
        raise ValueError('frozen_activation_required')
    if (not 1 <= budget['physical_attempt_limit'] <= MAX_REQUESTS
            or budget['physical_attempt_limit'] != len(cohort['cases'])
            or budget['api_limit'] != 0 or budget['gpu_jobs'] != 0 or budget['parallel_workers'] != 1):
        raise ValueError('finite_serial_CPU_budget_required')
    source = resolve(protocol['source_preparation']).parent.parent
    _, _, source_cohort, _, _ = verify_preparation(source)
    expected = select_cases(source, source_cohort, read(root / 'selection.json')['case_ids'],
                            read(resolve(protocol['independent_protocol'])))
    if cohort['cases'] != expected:
        raise ValueError('activation_cannot_change_source_task_scene_or_split')
    return spec


def run(root, limit):
    root = Path(root).resolve(); verify(root)
    budget = read(root / 'budget/limits.json')
    if type(limit) is not int or not 1 <= limit <= budget['physical_attempt_limit']:
        raise ValueError('bounded_launch_count_required')
    return execution.run(root, 'candidate', limit)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    for name in ('prepared', 'root', 'independent'):
        prep.add_argument('--' + name, type=Path, required=True)
    prep.add_argument('--physical-limit', type=int, required=True)
    prep.add_argument('--case', action='append', required=True)
    launch = sub.add_parser('run')
    launch.add_argument('--root', type=Path, required=True); launch.add_argument('--limit', type=int, required=True)
    args = parser.parse_args()
    print(prepare(args.prepared, args.root, args.independent, args.case, args.physical_limit)
          if args.command == 'prepare' else run(args.root, args.limit))
