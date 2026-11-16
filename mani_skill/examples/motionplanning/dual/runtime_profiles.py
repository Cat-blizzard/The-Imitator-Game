"""Versioned task execution profile selection, independent of the simulator.

This module selects existing RuntimePolicy values and returns phase/budget data.
It never creates an environment, plans, executes, changes collision permissions,
declares physical success, or alters the RuntimePolicy serialization schema.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass, replace
import json
from typing import Any

if __package__:
    from .runtime_policy import LEGACY_STRICT, RuntimePolicy, canonical_sha256, resolved_policy_record
else:
    from mani_skill.examples.motionplanning.dual.runtime_policy import (
        LEGACY_STRICT, RuntimePolicy, canonical_sha256, resolved_policy_record)

TASK_IDS = ('TwoRobotPickAppleBananaToBaskets-v1', 'TwoRobotPickPillToRegions-v1',
            'TwoRobotPickFruitsToPlate-v1')
PROFILE_IDS = ('legacy-strict-v1', 'task-default-v1')
TRANSPORT_RECIPES = ('standard', 'pill-direct-release-v1', 'direct-release-v2')
ALLOWED_OVERRIDES = ('grasp_stable_steps', 'position_slip_m', 'rotation_slip_rad',
    'release_stable_steps', 'reach_stable_steps', 'retreat_command_m',
    'minimum_retreat_m', 'final_stable_steps', 'min_lift_m',
    'minimum_settle_budget', 'max_settle_steps', 'require_retreat', 'require_final_stability')
PROFILE_SCHEMA = 'tig-task-runtime-profile-v1'

# Task rows deliberately retain the currently evidenced grasp/slip and optional
# phase defaults. Turning a phase off requires an explicit override for the
# selected task; the registry does not invent task-specific relaxed thresholds.
_TASK_OPTIONAL_DEFAULTS = {
    TASK_IDS[0]: (True, True),
    TASK_IDS[1]: (True, True),
    TASK_IDS[2]: (True, True),
}
_MOTION_PHASES = ('pregrasp', 'descend', 'lift_and_hold',
                  'carry_orient_and_transport', 'carry_lower', 'retreat')


@dataclass(frozen=True, slots=True)
class ResolvedProfile:
    """Selection metadata stays outside the existing 17-field policy record."""

    task_id: str
    profile_id: str
    policy: RuntimePolicy
    overrides: tuple[tuple[str, Any], ...] = ()

    def policy_record(self) -> dict[str, Any]:
        return resolved_policy_record(self.policy)

    def metadata(self) -> dict[str, Any]:
        return {'schema': PROFILE_SCHEMA, 'task_id': self.task_id,
                'profile_id': self.profile_id, 'overrides': dict(self.overrides),
                'policy_sha256': canonical_sha256(self.policy),
                'parameter_scope': 'selected task execution and quality parameters',
                'attachment_geometry_validity': 'independent measured runtime evidence required',
                'physical_validation': 'not established by profile resolution'}


def _task(task_id: str) -> str:
    if not isinstance(task_id, str) or task_id not in TASK_IDS:
        raise ValueError('Expected one explicit supported canonical task_id')
    return task_id


def resolve_profile(task_id: str, profile_id: str = 'legacy-strict-v1', *,
                    overrides: Mapping[str, Any] | None = None) -> ResolvedProfile:
    """Resolve a task/profile pair without changing official limits or defaults.

    legacy-strict-v1 accepts no overrides. task-default-v1 exposes the existing
    execution/quality values for one explicitly selected task. RuntimePolicy
    supplies all value/type and cross-field validation; this registry adds no
    global numerical ceiling. Profile/schema identity and official 200/199
    limits cannot be overridden. Resolving quality parameters does not validate
    current attached geometry or authorize using a stale payload model.
    """
    _task(task_id)
    if not isinstance(profile_id, str) or profile_id not in PROFILE_IDS:
        raise ValueError('Unknown runtime profile_id')
    if overrides is None:
        values: dict[str, Any] = {}
    elif isinstance(overrides, Mapping):
        values = dict(overrides)
    else:
        raise ValueError('overrides must be an explicit mapping')
    if any(not isinstance(key, str) or key not in ALLOWED_OVERRIDES for key in values):
        raise ValueError('Unknown or locked runtime profile override')
    if profile_id == 'legacy-strict-v1':
        if values:
            raise ValueError('legacy-strict-v1 is immutable; select a new profile for overrides')
        return ResolvedProfile(task_id, profile_id, LEGACY_STRICT)
    retreat, final = _TASK_OPTIONAL_DEFAULTS[task_id]
    policy = replace(LEGACY_STRICT, policy_id=profile_id, minimum_settle_budget=0,
                     require_retreat=retreat, require_final_stability=final)
    policy = replace(policy, **values)
    if policy.max_episode_steps != 200 or policy.last_completion_step != 199:
        raise ValueError('Task profiles must retain the official200 completion boundary')
    # Record normalized RuntimePolicy values (e.g. metre/radian scalars as
    # floats), while retaining exactly which keys were explicitly overridden.
    normalized = tuple((key, getattr(policy, key)) for key in sorted(values))
    return ResolvedProfile(task_id, profile_id, policy, normalized)


def _selection(value: ResolvedProfile) -> ResolvedProfile:
    """Do not let a hand-built selection bypass registry or override validation."""
    if not isinstance(value, ResolvedProfile):
        raise ValueError('A resolved runtime profile is required')
    expected = resolve_profile(value.task_id, value.profile_id, overrides=dict(value.overrides))
    if value.policy != expected.policy or value.overrides != expected.overrides:
        raise ValueError('Resolved profile differs from its declared task/profile/overrides')
    return value


def selected_phases(selection: ResolvedProfile, *, transport_recipe: str = 'standard') -> tuple[str, ...]:
    """Return the selected logical sequence, not a generated trajectory or outcome."""
    selection = _selection(selection)
    if transport_recipe not in TRANSPORT_RECIPES:
        raise ValueError('Unknown transport recipe')
    if transport_recipe == 'pill-direct-release-v1' and selection.task_id != TASK_IDS[1]:
        raise ValueError('pill-direct-release-v1 is confined to the pill task')
    phases = ['preparation', 'pregrasp', 'descend', 'close_and_confirm', 'lift_and_hold']
    if transport_recipe == 'standard':
        phases.append('carry_orient_and_transport')
    phases += ['carry_lower', 'release']
    if selection.policy.require_retreat:
        phases.append('retreat')
    if selection.policy.require_final_stability:
        phases.append('verify_placement')
    return tuple(phases)


def stage_requirements(selection: ResolvedProfile, *,
                       transport_recipe: str = 'standard') -> dict[str, Any]:
    """Expose quality windows and lower bounds separately from planned durations.

    A movement's reach window is the existing endpoint-stability requirement,
    not an additional mandatory post-schedule wait. Its actual schedule may
    already provide these observations. Future minima are necessary bounds;
    they are not estimates or guarantees for unseen future paths.
    """
    phases = selected_phases(selection, transport_recipe=transport_recipe)
    policy = selection.policy
    minima = {}
    for phase in phases:
        if phase in _MOTION_PHASES:
            minima[phase] = policy.reach_stable_steps
        elif phase == 'close_and_confirm':
            minima[phase] = policy.grasp_stable_steps
        elif phase == 'release':
            minima[phase] = policy.release_stable_steps
        elif phase == 'verify_placement':
            minima[phase] = policy.final_stable_steps
        else:
            # Preparation length comes from task execution configuration, not
            # a new value or threshold invented by the profile registry.
            minima[phase] = None
    return {'task_id': selection.task_id, 'profile_id': selection.profile_id,
        'policy_sha256': canonical_sha256(policy), 'transport_recipe': transport_recipe,
        'selected_phases': list(phases), 'phase_minimum_control_ticks': minima,
        'minimum_additional_settle_ticks': policy.minimum_settle_budget,
        'maximum_additional_settle_ticks': policy.max_settle_steps,
        'official_horizon_steps': policy.max_episode_steps,
        'last_permitted_completion_step': policy.last_completion_step,
        'scope': 'configuration and necessary tick lower bounds only; no physical success prediction'}


def phase_budget(selection: ResolvedProfile, before: int, scheduled: int, phase: str, *,
                 transport_recipe: str = 'standard') -> dict[str, Any]:
    """Return a necessary budget check over selected future phases, without executing.

    fits_minimum_model is a scheduling predicate only. It does not establish
    reach, grasp, release, placement, collision safety or eventual completion.
    A false predicate is data; the integration owner decides how to abort and
    retain the receipt. A zero extra-settle allowance is legal for the new
    profile and never fabricates an observed stable endpoint.
    """
    if type(before) is not int or before < 0 or type(scheduled) is not int or scheduled < 1:
        raise ValueError('Nonnegative elapsed and positive integer scheduled ticks required')
    requirements = stage_requirements(selection, transport_recipe=transport_recipe)
    phases = requirements['selected_phases']
    if phase not in _MOTION_PHASES or phase not in phases:
        raise ValueError('Expected a selected motion phase')
    future = phases[phases.index(phase)+1:]
    minima = requirements['phase_minimum_control_ticks']
    reserve = sum(minima[name] for name in future)
    available = selection.policy.last_completion_step-before-scheduled-reserve
    # _move starts its stable counter at zero. Scheduled ticks may supply the
    # reach window, but fewer ticks cannot possibly do so even under ideal
    # observations. Zero *extra* settling never removes that current window.
    reach_extra = max(0, selection.policy.reach_stable_steps-scheduled)
    minimum = max(selection.policy.minimum_settle_budget, reach_extra)
    fits = available >= minimum and selection.policy.max_settle_steps >= minimum
    return {'elapsed_before': before, 'scheduled_steps': scheduled, 'phase': phase,
        'profile_id': selection.profile_id, 'policy_sha256': canonical_sha256(selection.policy),
        'transport_recipe': transport_recipe, 'selected_future_phases': future,
        'reserved_future_minimum_steps': reserve,
        'last_permitted_completion_step': selection.policy.last_completion_step,
        'official_horizon_steps': selection.policy.max_episode_steps,
        'available_current_settle_steps': available,
        'profile_minimum_settle_steps': selection.policy.minimum_settle_budget,
        'current_reach_minimum_extra_ticks': reach_extra,
        'minimum_current_settle_steps': minimum,
        'settle_max_steps': min(selection.policy.max_settle_steps, max(0, available)),
        'fits_minimum_model': fits,
        'scope': 'necessary selected-phase budget bound, not observed physical acceptance'}


def self_test() -> dict[str, Any]:
    """CPU-only config, hash, sequence and boundary checks; no simulator imports."""
    expected_hash = 'b196d14685d546ee46a47139f600453559019855352ab5d85a7a7662527aa5d3'
    legacy_bytes = json.dumps(resolved_policy_record(), sort_keys=True, separators=(',', ':'))
    rejected = []
    def reject(label: str, call: Any) -> None:
        try:
            call()
        except (ValueError, TypeError):
            rejected.append(label)
        else:
            raise AssertionError('Expected rejection: '+label)
    for task in TASK_IDS:
        legacy = resolve_profile(task)
        assert legacy.policy is LEGACY_STRICT and len(legacy.policy.to_dict()) == 17
        assert canonical_sha256(legacy.policy) == expected_hash
        assert json.dumps(legacy.policy_record(), sort_keys=True, separators=(',', ':')) == legacy_bytes
        modern = resolve_profile(task, 'task-default-v1')
        assert modern.policy.minimum_settle_budget == 0
        assert modern.policy.max_episode_steps == 200 and modern.policy.last_completion_step == 199
        assert modern.policy.grasp_stable_steps == LEGACY_STRICT.grasp_stable_steps
        assert modern.policy.position_slip_m == LEGACY_STRICT.position_slip_m
        assert modern.policy.rotation_slip_rad == LEGACY_STRICT.rotation_slip_rad
        assert modern.policy.require_retreat and modern.policy.require_final_stability
        metadata = modern.metadata(); record = modern.policy_record()
        assert set(record) == {'resolved', 'sha256'} and 'task_id' not in record['resolved']
        metadata['overrides']['require_retreat'] = False
        assert modern.policy.require_retreat
        for retreat in (False, True):
            for final in (False, True):
                selected = resolve_profile(task, 'task-default-v1', overrides={
                    'require_retreat': retreat, 'require_final_stability': final})
                phases = selected_phases(selected, transport_recipe='direct-release-v2')
                assert ('retreat' in phases) == retreat and ('verify_placement' in phases) == final
                assert 'release' in phases and 'lift_and_hold' in phases
                assert 'carry_orient_and_transport' not in phases
                budget = phase_budget(selected, 0, 1, 'carry_lower', transport_recipe='direct-release-v2')
                assert budget['reserved_future_minimum_steps'] == (LEGACY_STRICT.release_stable_steps
                    + (LEGACY_STRICT.reach_stable_steps if retreat else 0)
                    + (LEGACY_STRICT.final_stable_steps if final else 0))
    legacy = resolve_profile(TASK_IDS[1]);modern = resolve_profile(TASK_IDS[1], 'task-default-v1')
    expected_future = {'pregrasp':43, 'descend':40, 'lift_and_hold':32,
        'carry_orient_and_transport':29, 'carry_lower':26, 'retreat':20}
    assert {phase:phase_budget(legacy,0,1,phase)['reserved_future_minimum_steps']
            for phase in expected_future} == expected_future
    # Last completion199 minus final20 and schedule3 leaves176 before retreat.
    # Three scheduled observations can meet reach3 with zero extra settling;
    # one scheduled observation cannot, even before looking at physical data.
    zero = phase_budget(modern,176,3,'retreat')
    assert zero['fits_minimum_model'] and zero['settle_max_steps']==0
    assert not phase_budget(modern,178,1,'retreat')['fits_minimum_model']
    assert not phase_budget(legacy,178,1,'retreat')['fits_minimum_model']
    assert not phase_budget(modern,177,3,'retreat')['fits_minimum_model']
    assert phase_budget(legacy,175,1,'retreat')['fits_minimum_model']
    assert not phase_budget(legacy,176,1,'retreat')['fits_minimum_model']
    no_tail = resolve_profile(TASK_IDS[1], 'task-default-v1', overrides={
        'require_retreat':False, 'require_final_stability':False})
    assert phase_budget(no_tail,193,3,'carry_lower')['fits_minimum_model']
    assert not phase_budget(no_tail,194,3,'carry_lower')['fits_minimum_model']
    assert not phase_budget(no_tail,195,1,'carry_lower')['fits_minimum_model']
    short_settle = resolve_profile(TASK_IDS[1], 'task-default-v1', overrides={'max_settle_steps':1})
    assert not phase_budget(short_settle,0,1,'retreat')['fits_minimum_model']
    # Explicit task-local settings are valid configuration, not an observed
    # physical result or a proposed new default threshold.
    custom_values = {'grasp_stable_steps':7, 'position_slip_m':.006,
        'rotation_slip_rad':.2, 'release_stable_steps':4, 'reach_stable_steps':4,
        'retreat_command_m':.091, 'minimum_retreat_m':.09,
        'final_stable_steps':24, 'min_lift_m':.11,
        'minimum_settle_budget':16, 'max_settle_steps':16,
        'require_retreat':False, 'require_final_stability':False}
    custom = resolve_profile(TASK_IDS[1], 'task-default-v1', overrides=custom_values)
    assert dict(custom.overrides) == custom_values and len(custom.overrides) == 13
    assert custom.policy.grasp_stable_steps == 7 and custom.policy.position_slip_m == .006
    assert custom.metadata()['attachment_geometry_validity'] == 'independent measured runtime evidence required'
    assert resolve_profile(TASK_IDS[0], 'task-default-v1').policy.grasp_stable_steps == 5
    assert phase_budget(custom,0,1,'carry_lower')['reserved_future_minimum_steps'] == 4
    assert stage_requirements(custom)['phase_minimum_control_ticks']['close_and_confirm'] == 7
    assert canonical_sha256(custom.policy) != canonical_sha256(modern.policy)
    # No extra global ceiling: RuntimePolicy allows finite nonnegative slip
    # values and positive windows; this tests resolution only, never safety.
    permitted = resolve_profile(TASK_IDS[0], 'task-default-v1', overrides={
        'position_slip_m':1., 'rotation_slip_rad':10., 'grasp_stable_steps':201})
    assert permitted.policy.grasp_stable_steps == 201
    assert not phase_budget(permitted,0,1,'pregrasp')['fits_minimum_model']
    zero_slip = resolve_profile(TASK_IDS[0], 'task-default-v1', overrides={'position_slip_m':0})
    assert type(dict(zero_slip.overrides)['position_slip_m']) is float
    reject('unknown_task',lambda:resolve_profile('unknown'))
    reject('unknown_profile',lambda:resolve_profile(TASK_IDS[0],'unknown'))
    reject('legacy_override',lambda:resolve_profile(TASK_IDS[0],overrides={'require_retreat':False}))
    for key,value in [('max_episode_steps',400),('last_completion_step',299),('policy_id','other'),
        ('schema_version',1),('position_slip_m',float('nan')),('rotation_slip_rad',float('inf')),
        ('position_slip_m',-.001),('rotation_slip_rad','0.1'),('min_lift_m',True),
        ('minimum_retreat_m',-.1),('retreat_command_m',None),('grasp_stable_steps',True),
        ('grasp_stable_steps',0),('release_stable_steps',1.),('reach_stable_steps',0),
        ('final_stable_steps',False),('max_settle_steps',0),
        ('require_retreat',0),('require_final_stability','false'),('minimum_settle_budget',True),
        ('minimum_settle_budget',-.1),('minimum_settle_budget',16),('unknown',0)]:
        reject('invalid_override_'+key+'_'+repr(value),lambda key=key,value=value:
            resolve_profile(TASK_IDS[0],'task-default-v1',overrides={key:value}))
    reject('nonmapping_overrides',lambda:resolve_profile(TASK_IDS[0],'task-default-v1',overrides=[]))
    reject('unknown_recipe',lambda:selected_phases(modern,transport_recipe='unknown'))
    reject('pill_recipe_other_task',lambda:selected_phases(resolve_profile(TASK_IDS[0]),transport_recipe='pill-direct-release-v1'))
    reject('unselected_retreat_budget',lambda:phase_budget(no_tail,0,1,'retreat'))
    reject('invalid_phase',lambda:phase_budget(modern,0,1,'release'))
    reject('invalid_elapsed',lambda:phase_budget(modern,True,1,'retreat'))
    reject('zero_schedule',lambda:phase_budget(modern,0,0,'retreat'))
    reject('handmade_policy_bypass',lambda:selected_phases(ResolvedProfile(
        TASK_IDS[1],'task-default-v1',replace(modern.policy,max_episode_steps=400,last_completion_step=399))))
    assert json.dumps(resolved_policy_record(), sort_keys=True, separators=(',', ':')) == legacy_bytes
    return {'status':'passed','legacy_policy_fields':17,'legacy_policy_sha256':expected_hash,
        'legacy_json_unchanged':True,'known_tasks':len(TASK_IDS),'profile_ids':list(PROFILE_IDS),
        'task_local_override_fields':list(ALLOWED_OVERRIDES),
        'quality_parameters_independent_of_attachment_validity':True,
        'legacy_future_minima':expected_future,'optional_phase_combinations':12,
        'new_zero_extra_settle_boundary':zero,'rejected_cases':rejected,
        'environment_creations':0,'physics_steps':0,'physical_acceptance':'NOT_RUN'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if not args.self_test:
        parser.error('This configuration module only exposes --self-test as a CLI')
    print(json.dumps(self_test(),allow_nan=False))
