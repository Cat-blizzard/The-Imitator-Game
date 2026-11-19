"""Independent, versioned event semantics for supported-or-low-drop placement.

Both controller conditions must use the SAME preregistered placement_contract.
The release.mode controller flag is deliberately ignored. The existing event
DSL/interpreter and all final goals/contact/non-target predicates are retained.
Only the legal release region and hold-until-release event semantics change.
"""
import copy

import numpy as np

from corl2026_workshop.experiments.autonomous_pipeline_v4 import planning_protocols as old
from corl2026_workshop.experiments.autonomous_pipeline_v4.event_protocol import validate

VERSION = 'supported_or_low_drop_events_v1'
CONTRACT_SCHEMA = 'supported_or_low_drop_v1'
_retained_builder = old.pick_place


def validate_contract(task):
    contract = task.get('placement_contract')
    if contract is None:
        return None
    expected = {'schema', 'maximum_release_height_over_goal_m', 'xy_tolerance_m',
                'continuous_hold_until_release'}
    if (set(contract) != expected or contract['schema'] != CONTRACT_SCHEMA
            or contract['xy_tolerance_m'] != .015
            or contract['continuous_hold_until_release'] is not True):
        raise ValueError('explicit_frozen_supported_or_low_drop_contract_required')
    height = contract['maximum_release_height_over_goal_m']
    if not isinstance(height, (float, int)) or isinstance(height, bool) or not np.isfinite(height) or not 0 < height <= .15:
        raise ValueError('release_height_must_be_finite_positive_at_most_15cm')
    if any(op['target']['support'] == 'table' for op in task['operations']):
        raise ValueError('low_drop_protocol_v1_requires_container_operations')
    return copy.deepcopy(contract)


def pick_place(scene, task, object_ids, support_ids, targets):
    original = _retained_builder(scene, task, object_ids, support_ids, targets)
    contract = validate_contract(task)
    if contract is None:
        return original
    plan = copy.deepcopy(original)
    c, e, s = old.const, old.expr, old.signal
    changed_event_ids, changed_constraint_ids = set(), set()
    for operation in task['operations']:
        oid, arm, prefix = operation['object'], operation['arm'], operation['id']
        position = e('position', s(oid + '.pose'))
        held = s(f'arm{arm}.{oid}.grasp')
        free = e('not', e('any', s(f'arm0.{oid}.grasp'), s(f'arm1.{oid}.grasp')))
        target = np.asarray(targets[oid], dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError('finite_requested_final_target_required')
        # The same 15 mm lateral/final tolerance applies to both controllers.
        # The upper z bound is exact; no tolerance is added to the 15 cm cap.
        region = e('all', e('lt', e('norm', e('sub', e('xy', position), c(target[:2].tolist()))), c(.015)),
                   e('ge', old.component(position, 2), c(float(target[2] - .015))),
                   e('le', old.component(position, 2), c(float(target[2] + contract['maximum_release_height_over_goal_m']))))
        by_id = {event['id']: event for event in plan['events']}
        arrive, release = by_id[prefix + '.arrive'], by_id[prefix + '.release']
        arrive['when'] = e('all', held, region)
        release['when'] = e('all', free, region)
        release['strictly_after'] = [prefix + '.arrive']
        changed_event_ids.update([arrive['id'], release['id']])
        hold = next(item for item in plan['constraints'] if item['id'] == prefix + '.held')
        hold['until'] = prefix + '.release'
        hold['include_end'] = False
        hold['parameters'] = {'event_semantics_version': VERSION, 'placement_contract': contract}
        changed_constraint_ids.add(hold['id'])
        old.invariant(plan, prefix + '.release_region', 'at_position', [oid, operation['target']['support']],
            region, start=prefix + '.arrive', until=prefix + '.release', include_end=True,
            effector=f'arm{arm}', event_semantics_version=VERSION, placement_contract=contract,
            target_position_world=target.tolist(), xy_tolerance_m=.015,
            minimum_z_world_m=float(target[2] - .015),
            maximum_z_world_m=float(target[2] + contract['maximum_release_height_over_goal_m']),
            scope='held arrival and release region; final supported position remains original 3D goal')
        # A one-frame lost-grasp sample followed by a regrasp must not be
        # mistaken for the intended release. These tasks already forbid
        # repeated manipulation of the same object.
        old.invariant(plan, prefix + '.released_stays_free', 'no_grasp', [oid], free,
            start=prefix + '.release', until='terminal', include_end=True,
            event_semantics_version=VERSION, scope='no regrasp after the declared release')
    # Fail if this builder accidentally changes final or unrelated acceptance.
    for key in ('goals', 'initial_conditions', 'initial_hands', 'terminal_hands', 'terminal_window_frames',
                'channels', 'effectors', 'entities', 'precedence', 'equivalence_groups'):
        if plan[key] != original[key]:
            raise ValueError('retained_acceptance_changed:' + key)
    old_events = {x['id']: x for x in original['events']}
    for event in plan['events']:
        if event['id'] not in changed_event_ids and event != old_events[event['id']]:
            raise ValueError('unrelated_event_changed')
    kept = {x['id']: x for x in plan['constraints']}
    for item in original['constraints']:
        if item['id'] not in changed_constraint_ids and kept[item['id']] != item:
            raise ValueError('non_target_or_other_invariant_changed')
    validate(plan)
    return plan
