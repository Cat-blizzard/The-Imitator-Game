"""Artifact selection only; observes existing evidence without changing physics.

Diagnostic selects the legacy files and rendering. Normal keeps the unchanged
T control ticks/actions, RecordEpisode T+1 states and initial-state/geometry
identity, while archiving incident contacts, transitions and measured summaries.
Full100Hz lists stay in memory for the original safety/audit consumers. No
quality thresholds, grasp/contact rules or RuntimePolicy fields are introduced.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = 'tig-recording-policy-v1'
SUMMARY_SCHEMA = 'tig-compact-physical-observations-v1'


@dataclass(frozen=True)
class RecordingPolicy:
    mode: str = 'diagnostic'

    def __post_init__(self) -> None:
        if self.mode not in ('normal', 'diagnostic'):
            raise ValueError('Recording mode must be normal or diagnostic')

    @property
    def diagnostic(self) -> bool:
        return self.mode == 'diagnostic'

    def record(self) -> dict[str, Any]:
        resolved = {'mode': self.mode, 'render_video': self.diagnostic,
            'full_physics_substeps': self.diagnostic,
            'full_robot_contacts': self.diagnostic,
            'full_stage_trace': self.diagnostic,
            'physical_observations': 'unchanged100Hz',
            'in_memory_substeps': 'full; unchanged audit consumers',
            'ticks_and_actions': 'all; unchanged', 'state_frames': 'RecordEpisode T+1; unchanged'}
        encoded = json.dumps(resolved, sort_keys=True, separators=(',', ':')).encode()
        return {'schema': SCHEMA, 'resolved': resolved, 'sha256': hashlib.sha256(encoded).hexdigest()}


DIAGNOSTIC_RECORDING = RecordingPolicy('diagnostic')
NORMAL_RECORDING = RecordingPolicy('normal')


def resolve_recording_policy(policy: RecordingPolicy | None = None) -> RecordingPolicy:
    if policy is None:
        return DIAGNOSTIC_RECORDING
    if not isinstance(policy, RecordingPolicy):
        raise ValueError('A validated RecordingPolicy is required')
    return policy


def recording_policy_from_report(report: dict[str, Any]) -> RecordingPolicy:
    record = report.get('recording_policy')
    if record is None:
        return DIAGNOSTIC_RECORDING
    if not isinstance(record, dict) or not isinstance(record.get('resolved'), dict):
        raise ValueError('Invalid recording policy receipt')
    policy = RecordingPolicy(record['resolved'].get('mode'))
    if record != policy.record():
        raise ValueError('Recording policy configuration/hash mismatch')
    return policy


def archive_report(report: dict[str, Any], policy: RecordingPolicy | None = None) -> dict[str, Any]:
    """Return a serialization view; retain original report/trace objects in memory.

    The terminal trace sample remains for existing minimum-lift/release result
    reducers. Every other stage receipt and existing held audit is unchanged.
    Diagnostic returns the same original object with no new field or copying.
    """
    selected = resolve_recording_policy(policy)
    if selected.diagnostic:
        return report
    def terminal_trace(source: dict[str, Any]) -> dict[str, Any]:
        row = dict(source)
        if isinstance(source.get('trace'), list):
            trace = source['trace']
            row['trace'] = trace[-1:]
            row['trace_retention'] = {'selection': 'terminal_sample', 'observed_rows': len(trace),
                                     'archived_rows': min(len(trace), 1)}
        return row

    result = dict(report)
    if 'stages' in report:
        result['stages'] = [terminal_trace(stage) for stage in report['stages']]
    if isinstance(report.get('task_outcome_observation'), dict):
        result['task_outcome_observation'] = terminal_trace(report['task_outcome_observation'])
    result['recording_policy'] = selected.record()
    return result


def write_robot_contacts(output: Path, contacts: list[dict[str, Any]], *,
                         policy: RecordingPolicy | None = None) -> None:
    """Runner/replay seam replacing unconditional full-contact JSON output.

    Normal's actual incident contacts are included by write_compact_physics.
    This function does not delete pre-existing files or change monitoring lists.
    """
    if resolve_recording_policy(policy).diagnostic:
        (output/'all-robot-contacts.json').write_text(json.dumps(contacts, indent=2, allow_nan=False)+'\n')


def _location(row: dict[str, Any]) -> dict[str, Any]:
    return {name: row[name] for name in ('substep', 'control_step', 'phase', 'target_phases') if name in row}


def compact_physics(report: dict[str, Any], ticks: list[dict[str, Any]],
                    substeps: list[dict[str, Any]], robot_contacts: list[dict[str, Any]] | None,
                    *, active_error: BaseException | None = None,
                    held_audit: dict[str, Any] | None = None) -> dict[str, Any]:
    """Reduce actual observations; missing coverage/evidence stays explicit.

    Incident contact records are retained verbatim, including raw points and
    channel-specific permissions. Counts intentionally keep the channels
    separate. Grasp transitions are observed facts, not new abort decisions.
    """
    events, counts, phases = [], {}, {}
    previous_grasps: dict[int, bool] = {}
    previous_support: dict[int, bool] = {}
    previous_phase = None
    grasp_counts: dict[str, dict[str, int]] = {}
    for row in substeps:
        where = _location(row)
        phase = row.get('phase')
        phases[str(phase)] = phases.get(str(phase), 0) + 1
        if type(row.get('control_step')) is int:
            step = str(row['control_step'])
            counts[step] = counts.get(step, 0) + 1
        phase_changed = phase != previous_phase
        if phase_changed:
            events.append({'kind': 'phase_observed', **where})
        for channel, chosen in (('object_contact_rows', lambda c: c.get('class') == 'forbidden'),
                                ('interarm_contacts', lambda c: True)):
            incident = [copy.deepcopy(c) for c in row.get(channel, [])
                        if c.get('physical_contact') and chosen(c)]
            if incident:
                events.append({'kind': 'forbidden_contact', 'channel': channel, **where, 'contacts': incident})
        held = phase == 'lift_and_hold' or isinstance(phase, str) and phase.startswith('carry_')
        for index, obj in enumerate(row.get('objects', [])):
            grasp = obj.get('is_grasping')
            if type(grasp) is not bool:
                continue
            key = str(index)
            value = grasp_counts.setdefault(key, {'observed': 0, 'grasping': 0, 'not_grasping': 0})
            value['observed'] += 1
            value['grasping' if grasp else 'not_grasping'] += 1
            if index not in previous_grasps or grasp != previous_grasps[index]:
                events.append({'kind': 'grasp_observed' if index not in previous_grasps else 'grasp_transition',
                    **where, 'target_index': index, 'actor': obj.get('actor'),
                    'previous': previous_grasps.get(index), 'is_grasping': grasp})
            if held and not grasp and (phase_changed or previous_grasps.get(index) is not False):
                events.append({'kind': 'held_grasp_missing_observed', **where, 'target_index': index,
                               'actor': obj.get('actor'), 'is_grasping': False})
            previous_grasps[index] = grasp
        for index, released in enumerate(row.get('physical_support_released_snapshot', [])):
            if type(released) is not bool:
                continue
            if index not in previous_support or previous_support[index] != released:
                events.append({'kind': 'physical_support_observed' if index not in previous_support else 'physical_support_transition',
                    **where, 'target_index': index, 'previous': previous_support.get(index), 'released': released})
            previous_support[index] = released
        previous_phase = phase
    for row in robot_contacts or []:
        incident = [copy.deepcopy(c) for c in row.get('contacts', [])
                    if c.get('physical_contact') and not c.get('allowed')]
        if incident:
            events.append({'kind': 'forbidden_contact', 'channel': 'robot_world', **_location(row), 'contacts': incident})
    for tick in ticks:
        if tick.get('native_truncated'):
            events.append({'kind': 'native_time_limit_observed', 'control_step': tick.get('environment_step'),
                           'phase': tick.get('phase')})
    for stage in report.get('stages', []):
        event = {key: copy.deepcopy(stage[key]) for key in
                 ('name', 'phase', 'status', 'reason', 'error', 'environment_steps',
                  'stable_steps_observed', 'stable_steps_required', 'object_index') if key in stage}
        events.append({'kind': 'stage_result', **event})
    for fault in report.get('fault_events', []):
        events.append({'kind': 'injected_fault', 'evidence': copy.deepcopy(fault)})
    if active_error is not None:
        events.append({'kind': 'capture_exception', 'error_type': type(active_error).__name__,
                       'error': str(active_error), 'phase': report.get('stage')})
    if isinstance(report.get('error'), str):
        events.append({'kind': 'run_exception', 'error_type': report.get('error_type'),
                       'error': report['error'], 'phase': report.get('stage')})
    held = [row for row in substeps if row.get('phase') == 'lift_and_hold' or
            isinstance(row.get('phase'), str) and row['phase'].startswith('carry_')]
    if held_audit is None:
        held_audit = report.get('held_physics_audit')
    if held_audit is None:
        held_audit = {'status': 'not_observed' if not held else 'not_available',
            'physical_substeps': len(held), 'objects': [],
            'reason': 'no held samples' if not held else 'no measured attachment reference/audit supplied'}
    dt, sim_dt = report.get('control_timestep_s'), report.get('physics_timestep_s')
    expected_per_control = int(round(dt / sim_dt)) if dt and sim_dt else None
    tick_steps = {row.get('environment_step') for row in ticks}
    summary = {'schema': SUMMARY_SCHEMA,
        'physical_monitoring': copy.deepcopy(report.get('physical_monitoring')),
        'actual_substeps': len(substeps),
        'actual_robot_world_substeps': None if robot_contacts is None else len(robot_contacts),
        'phase_substep_counts': phases, 'control_substep_counts': counts,
        'expected_substeps_per_control': expected_per_control,
        'partial_control_steps': [int(step) for step, count in counts.items()
                                 if expected_per_control is not None and count != expected_per_control],
        'observed_control_steps_without_completed_tick': [int(step) for step in counts if int(step) not in tick_steps],
        'grasp_observations_per_target': grasp_counts, 'last_physical_support_released': {str(k): v for k, v in previous_support.items()},
        'held_physics_audit': copy.deepcopy(held_audit),
        'event_count': len(events), 'scope': 'observed100Hz data; unchanged physical/quality gates; not continuous collision proof'}
    return {'events': events, 'summary': summary}


def write_compact_physics(output: Path, report: dict[str, Any], ticks: list[dict[str, Any]],
                          substeps: list[dict[str, Any]], robot_contacts: list[dict[str, Any]] | None,
                          *, active_error: BaseException | None = None,
                          held_audit: dict[str, Any] | None = None) -> dict[str, Any]:
    bundle = compact_physics(report, ticks, substeps, robot_contacts,
                             active_error=active_error, held_audit=held_audit)
    with (output/'physics-events.jsonl').open('w') as handle:
        for event in bundle['events']:
            handle.write(json.dumps(event, allow_nan=False)+'\n')
    (output/'physics-summary.json').write_text(json.dumps(bundle['summary'], indent=2, allow_nan=False)+'\n')
    return bundle['summary']


def self_test() -> dict[str, Any]:
    original = {'status': 'passed', 'stages': [{'name': 'lift_and_hold',
        'trace': [{'objects': [{'object_height_gain_m': .1}]}, {'objects': [{'object_height_gain_m': .12}]}]}]}
    frozen = json.dumps(original)
    assert archive_report(original) is original
    projected = archive_report(original, NORMAL_RECORDING)
    assert json.dumps(original) == frozen and len(original['stages'][0]['trace']) == 2
    assert projected['stages'][0]['trace'] == original['stages'][0]['trace'][-1:]
    assert recording_policy_from_report(projected) == NORMAL_RECORDING
    for steps in (0, 3):
        observation = {'trace': [{'step': i} for i in range(steps)],
                       'final_observation': {'observed_elapsed_steps': [steps]}}
        live = {'task_outcome_observation': observation}
        frozen = copy.deepcopy(live)
        archived = archive_report(live, NORMAL_RECORDING)
        assert live == frozen and archive_report(live) is live
        assert archived['task_outcome_observation'] is not observation
        assert archived['task_outcome_observation']['trace'] == observation['trace'][-1:]
        assert archived['task_outcome_observation']['final_observation'] is observation['final_observation']
        assert len(archived['task_outcome_observation']['trace']) == min(steps, 1)
    for value in ('x', False, None, 'NORMAL'):
        try: RecordingPolicy(value)
        except ValueError: pass
        else: raise AssertionError('Invalid recording mode accepted')
    try: resolve_recording_policy('normal')
    except ValueError: pass
    else: raise AssertionError('Unvalidated recording policy accepted')
    projected['recording_policy']['sha256'] = '0'*64
    try: recording_policy_from_report(projected)
    except ValueError: pass
    else: raise AssertionError('Changed recording policy hash accepted')
    return {'status': 'passed', 'physical_environment_steps': 0,
        'diagnostic_same_object': True, 'normal_does_not_mutate_report': True,
        'terminal_sample_retained': True, 'recording_policy_hash_checked': True,
        'task_outcome_trace_projected_without_mutation': True,
        'zero_step_trace_and_final_observation_preserved': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.parse_args()
    print(json.dumps(self_test(), indent=2, allow_nan=False))
