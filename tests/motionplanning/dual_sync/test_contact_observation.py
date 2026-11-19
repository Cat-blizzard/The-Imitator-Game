"""Native contact snapshots retain support transitions and rejection order."""
from collections import Counter
from types import SimpleNamespace as NS
from typing import Any

import numpy as np
import pytest

from mani_skill.examples.motionplanning.dual.sync import observation
from mani_skill.examples.motionplanning.dual.sync.physical_observation import (
    classify_task_object_contacts, decode_native_contacts, task_object_contact_rows,
)


class Point:
    """Count native field readbacks without requiring a physics scene."""
    def __init__(self, separation: float = 0., impulse: Any = (0., 0., 0.)):
        self.reads = Counter()
        self.values = dict(position=np.array([.1, .2, .3]), normal=np.array([0., 0., 1.]),
            separation=separation, impulse=np.asarray(impulse))

    def __getattr__(self, name: str) -> Any:
        if name not in self.values:
            raise AttributeError(name)
        self.reads[name] += 1
        return self.values[name]


def fixture() -> tuple[dict[str, Any], Any]:
    class Entity:
        def __init__(self, name: str):
            self.name = name

    entities = {name: Entity(name) for name in ('left', 'right', 'source', 'destination',
        'finger', 'palm', 'contents')}
    identities = {
        entities['left']: dict(kind='payload', actor='left_target', target_index=0),
        entities['right']: dict(kind='payload', actor='right_target', target_index=1),
        entities['source']: dict(kind='actor', actor='table'),
        entities['destination']: dict(kind='actor', actor='basket'),
        entities['contents']: dict(kind='actor', actor='fruit'),
        entities['finger']: dict(kind='robot', actor='panda_leftfinger', robot_index=0,
            link_name='panda_leftfinger'),
        entities['palm']: dict(kind='robot', actor='panda_hand', robot_index=0,
            link_name='panda_hand'),
    }
    proof_calls = []

    def proof(index: int, actor: str, phase: str, points: list[dict]) -> dict:
        proof_calls.append((index, actor, phase))
        return {'allowed': phase == 'release' and bool(points)}

    task = NS(initial_supports=(NS(name='table'), NS(name='table')),
        destination_supports=(NS(name='basket'), NS(name='basket')),
        destination_contents_state_names=('fruit',), destination_contents_contact_proof=proof,
        proof_calls=proof_calls,
        targets=(NS(pose=NS(p=np.array([[0., 0., .004]]))),
                 NS(pose=NS(p=np.array([[0., 0., .004]])))))
    return dict(entities=entities, identities=identities), task


def contact(data: dict, first: str, second: str, point: Point) -> Any:
    return NS(bodies=[NS(entity=data['entities'][first]), NS(entity=data['entities'][second])],
        points=[point])


def test_snapshot_reclassifies_release_without_native_readback_or_state_leak():
    data, task = fixture()
    point = Point()
    native = [contact(data, 'left', 'source', point)]
    decoded = decode_native_contacts(native, data['identities'])
    before = classify_task_object_contacts(decoded, (False, False), 'lift_and_hold', task)
    after = classify_task_object_contacts(decoded, (True, False), 'lift_and_hold', task)
    assert before[0]['class'] == 'initial_support'
    assert after[0]['class'] == 'forbidden'
    assert before[0]['support_released_snapshot'] == [False, False]
    assert after[0]['support_released_snapshot'] == [True, False]
    assert point.reads == dict(separation=1, impulse=1, position=1, normal=1)
    assert before[0]['points'] == [{'position_m': [.1, .2, .3], 'normal': [0., 0., 1.],
        'impulse_Ns': [0., 0., 0.], 'separation_m': 0.}]
    assert task_object_contact_rows(native, data['identities'], (True, False),
        'lift_and_hold', task) == after


def test_target_phases_contents_proof_and_payload_pair_order_are_preserved():
    data, task = fixture()
    phases = ('carry_to_destination', 'release')
    native = [contact(data, 'left', 'destination', Point()),
        contact(data, 'destination', 'right', Point()),
        contact(data, 'right', 'contents', Point()),
        contact(data, 'left', 'right', Point())]
    decoded = decode_native_contacts(native, data['identities'])
    rows = classify_task_object_contacts(decoded, (True, True), phases, task)
    assert [row['class'] for row in rows] == [
        'forbidden', 'destination_support', 'destination_contents', 'forbidden']
    assert rows[-1]['target_indices'] == [0, 1]
    assert all(row['target_phases'] == list(phases) for row in rows)
    assert task.proof_calls == [(1, 'fruit', 'release')]
    rows = classify_task_object_contacts(decoded, (True, True), phases, task,
        include_destination_contents_proof=False)
    assert rows[2]['class'] == 'forbidden'
    assert 'destination_contents_proof' not in rows[2]
    assert task.proof_calls == [(1, 'fruit', 'release')]
    with pytest.raises(ValueError, match='Two canonical target phases'):
        classify_task_object_contacts(decoded, (True, True), ('release',), task)


@pytest.mark.parametrize(('separation', 'impulse', 'physical'), [
    (1e-5, (0., 0., 0.), False), (0., (0., 0., 0.), True),
    (1e-5, (0., 1e-10, 0.), True),
])
def test_proximity_boundary_does_not_change(separation, impulse, physical):
    data, task = fixture()
    rows = task_object_contact_rows([contact(data, 'source', 'left', Point(separation, impulse))],
        data['identities'], (False, False), 'lift_and_hold', task)
    assert rows[0]['physical_contact'] is physical
    assert rows[0]['positive_separation_zero_impulse_proximity'] is (not physical)


def monitor(data: dict, task: Any, contacts: list, monkeypatch: Any) -> Any:
    observed = observation.ObservedEnvironment.__new__(observation.ObservedEnvironment)
    observed.original_hook = lambda: None
    observed.substeps = 0
    observed.unwrapped = NS(scene=NS(px=NS(get_contacts=lambda: contacts)))
    observed.checker_provider = lambda: None
    observed.identities, observed.task = data['identities'], task
    observed.physical_released = [False, False]
    observed.reference_frozen = [False, False]
    observed.reference_z = [0., 0.]
    observed.phase = 'lift_and_hold'
    observed.references = None
    observed._retreat_contacts = None
    observed._retreat_initial_contacts = None
    observed._retreat_separated_contacts = set()
    observed._retreat_remaining_contacts = set()
    observed._last_own_finger_contacts = set()
    observed._last_own_finger_contact_phase = None
    observed.pending_errors = []
    observed.srdf = {0: set(), 1: set()}
    monkeypatch.setattr(observation, 'measured_source_departure', lambda *_args, **_kwargs: (True, True))
    return observed


def test_native_monitor_latches_departure_then_rejects_recontact_in_existing_order(monkeypatch):
    data, task = fixture()
    source = Point()
    contacts = [contact(data, 'left', 'source', source)]
    observed = monitor(data, task, contacts, monkeypatch)
    observed.record_substep()
    assert observed.physical_released == [False, True]
    assert not observed.pending_errors
    assert source.reads == dict(separation=1, impulse=1, position=1, normal=1)
    # A native proximity manifold is not physical support: latch release now.
    contacts[:] = [contact(data, 'left', 'source', Point(1e-5))]
    observed.record_substep()
    assert observed.physical_released == [True, True]
    assert not observed.pending_errors
    contacts[:] = [contact(data, 'left', 'source', Point()),
        contact(data, 'palm', 'left', Point())]
    observed.record_substep()
    assert observed.physical_released == [True, True]
    assert [error['reason'] for error in observed.pending_errors] == [
        'forbidden_payload_contact', 'forbidden_payload_contact', 'forbidden_robot_contact']
    assert observed.pending_errors[0]['bodies'][1]['actor'] == 'table'


def test_native_monitor_decodes_payload_robot_contact_once_and_preserves_finger_permission(monkeypatch):
    data, task = fixture()
    point = Point()
    contacts = [contact(data, 'finger', 'left', point)]
    observed = monitor(data, task, contacts, monkeypatch)
    observed.record_substep()
    assert not observed.pending_errors
    assert point.reads == dict(separation=1, impulse=1, position=1, normal=1)
    # Forbidden robot-only contacts retain position/normal for diagnosis.
    robot_point = Point()
    contacts[:] = [contact(data, 'palm', 'finger', robot_point)]
    observed.record_substep()
    assert [error['reason'] for error in observed.pending_errors] == ['forbidden_robot_contact']
    assert robot_point.reads == dict(separation=1, impulse=1, position=1, normal=1)


def test_multi_monitor_keeps_diagnostic_fields_and_reads_payload_geometry_once(monkeypatch):
    data, task = fixture()
    point = Point()
    contacts = [contact(data, 'palm', 'left', point)]
    observed = monitor(data, task, contacts, monkeypatch)
    observed.record_substep()
    assert [error['reason'] for error in observed.pending_errors] == [
        'forbidden_payload_contact', 'forbidden_robot_contact']
    for error in observed.pending_errors:
        assert error['physical_substep_index'] == 1
        assert error['points'] == [{'position_m': [.1, .2, .3], 'normal': [0., 0., 1.],
            'separation_m': 0., 'impulse_Ns': [0., 0., 0.]}]
    assert point.reads == dict(separation=1, impulse=1, position=1, normal=1)


def test_multi_monitor_contents_proof_runs_only_after_source_update(monkeypatch):
    data, task = fixture()
    contacts = [contact(data, 'right', 'contents', Point())]
    observed = monitor(data, task, contacts, monkeypatch)
    observed.phase = 'release'
    observed.record_substep()
    assert task.proof_calls == [(1, 'fruit', 'release')]
    assert not observed.pending_errors


def test_multi_monitor_preserves_native_grasp_link_permission(monkeypatch):
    data, task = fixture()
    data['identities'][data['entities']['finger']].update(
        link_name='robotiq_left_inner_finger', actor='robotiq_left_inner_finger', is_grasp_finger=True)
    contacts = [contact(data, 'finger', 'left', Point())]
    observed = monitor(data, task, contacts, monkeypatch)
    observed.record_substep()
    assert not observed.pending_errors


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
