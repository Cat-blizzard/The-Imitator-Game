"""Independently verify the existing rolling declaration schema for rendering.

This does not convert a post-hoc declaration into a global pre-action claim.
Every original step commitment, native start state and requested task is checked.
"""
from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path
import numpy as np
from resimpair_grasp.common import ROOT, binding, read, resolve

VERSION = 'rolling_native_declaration_render_acceptance_v5'
REVIEWER = ROOT / 'validation/multitask_20261030/independent/review_multitask_v3.py'
REVIEWER_SHA256 = '9014809ba987ee4c5af98fe39e92744cb882b9cfc107719e38315d0db4815cb6'


def require(value, reason):
    if not value: raise ValueError(reason)


@lru_cache(maxsize=1)
def retained():
    require(binding(REVIEWER)['sha256'] == REVIEWER_SHA256, 'retained_multitask_reviewer_changed')
    spec = importlib.util.spec_from_file_location('retained_rolling_render_acceptance_v5', REVIEWER)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def review(directory, source, declaration):
    directory = Path(directory); audit = retained()
    require(declaration.get('scope') == 'aggregation_of_stepwise_pre_action_commitments',
            'rolling_native_declaration_scope_invalid')
    require(declaration.get('execution_selection_policy') == 'rolling_state_v3',
            'rolling_native_selection_policy_invalid')
    require(declaration['case'] == source['case'], 'rolling_declaration_case_changed')
    require(resolve(declaration['event_protocol']) == resolve(source['event_protocol']),
            'rolling_native_event_protocol_changed')
    refs = declaration.get('operation_commitments', [])
    operations = source['task_intent']['operations']
    require(len(refs) == len(operations) and len({resolve(ref) for ref in refs}) == len(refs),
            'rolling_declaration_missing_or_duplicate_commitment')
    committed = []
    for ref in refs:
        path = resolve(ref)
        require(path.parent.resolve() == (directory / 'operation_commitments').resolve(),
                'rolling_commitment_from_foreign_execution')
        committed.append(read(path)['operation']['id'])
    require(committed == [o['id'] for o in operations], 'rolling_commitment_operation_order_changed')
    initial = audit.initial_receipt(directory, source['case'])
    require(initial is not None, 'rolling_initial_native_receipt_missing')
    deferred = read(directory / 'deferred_render_manifest.json')
    with np.load(resolve(deferred['physics_states']), allow_pickle=False) as z:
        states = z['physics_states']
        actual_initial = hashlib.sha256(np.ascontiguousarray(states[0]).tobytes()).hexdigest()
        require(initial['physics'] == actual_initial, 'rolling_native_initial_state_not_saved_first_state')
        commitment = audit.commitments(directory, source, states)
    require(commitment['passed'], 'rolling_pre_action_commitment_failed')
    original_task = read(resolve(source['case']['task']))
    original_audit = read(directory / 'audit.json')
    paths = read(resolve(source['planned_paths']))['paths']
    operation = audit.original_operation_contract(original_task, source['task_intent'],
                                                  original_audit['event_frames'], paths)
    require(operation['passed'], 'rolling_original_operation_contract_failed')
    require(all(x.get('native_withdraw_observed_before_dependent', True)
                for x in operation['precedence_evidence']), 'rolling_native_precedence_not_completed')
    nested = audit.nested_planning_errors(directory)
    require(not nested['integrity_failure'], 'rolling_nested_integrity_failure')
    return {'schema': VERSION, 'passed': True, 'scope': declaration['scope'],
        'reviewer': binding(__file__), 'retained_reviewer': binding(REVIEWER),
        'initial_receipt': initial, 'per_operation_commitments': commitment,
        'original_requested_operation_contract': operation,
        'nested_planning_integrity_checked': True,
        'global_before_execution_declaration_claimed': False,
        'physical_and_visibility_thresholds_changed': False}
