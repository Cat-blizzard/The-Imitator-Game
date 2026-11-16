"""One physical payload and explicit fingertip permissions during handoff.

All geometry queries reuse the existing full dual-robot/scene/held checker.
One actor may have several convex pieces, each present once in the active
world. Giver and receiver relations are captured at separate real grasps;
switching prediction owner never changes physics. No simulator is constructed
or stepped, and no state, mask, weld, or existing source file is changed.
"""
from __future__ import annotations

import argparse
import copy
import json
from types import SimpleNamespace as NS
from typing import Any, Callable

import numpy as np

from .grasp_collision import FINGER_LINKS, LiveGraspCollisionChecker
from .held_collision import _pair_result
from .table_support_collision import (NumericalTableSupportCollisionChecker, NUMERICAL_TOP_TOLERANCE_M,
                                      TABLE_NAME, _rigid_transform)
from .trajectory_execution import as_numpy
from .validate_held_path_conflict import physical_contact_points

ROBOTS = ('left', 'right')
PHASES = ('giver_hold', 'receiver_contact', 'receiver_confirmed', 'giver_release',
          'giver_retreat', 'receiver_hold', 'safe_giver_hold')


def _require(value: Any, message: str) -> None:
    if not bool(value): raise ValueError(message)


def _index(value: int) -> int:
    _require(type(value) is int and value in (0, 1), 'Physical arm index must be0or1')
    return value


def _flag(value: Any) -> bool:
    array = as_numpy(value)
    _require(array.size == 1 and np.isfinite(array).all() and np.isin(array, [0, 1]).all(), 'Expected one native grasp boolean')
    return bool(array.reshape(-1)[0])


def _step(raw: Any) -> int:
    value = as_numpy(raw.elapsed_steps)
    _require(value.size == 1 and np.isfinite(value).all(), 'Expected single physical control clock')
    step = int(value.reshape(-1)[0]); _require(step >= 0 and step == value.reshape(-1)[0], 'Invalid control clock')
    return step


def _state(raw: Any) -> dict[str, np.ndarray]:
    def flatten(value: Any, prefix: str = '') -> dict[str, np.ndarray]:
        if isinstance(value, dict):
            result = {}
            for key, child in value.items(): result.update(flatten(child, prefix+'/'+str(key)))
            return result
        result = as_numpy(value).copy()
        _require(np.isfinite(result).all(), 'Nonfinite physics state')
        return {prefix: result}
    return flatten(raw.get_state_dict())


def _source_contact_evidence(contacts: Any, target: Any, table: Any) -> dict[str, Any]:
    """Read both entity orders using the recorder's exact physical-point predicate."""
    _require(target is not None and table is not None and target != table, 'Distinct source/target entities required')
    rows = []
    for contact in contacts:
        bodies = tuple(contact.bodies)
        _require(len(bodies) == 2 and all(getattr(b, 'entity', None) is not None for b in bodies),
                 'Invalid native contact bodies')
        entities = tuple(b.entity for b in bodies)
        if not ((entities[0] == target and entities[1] == table)
                or (entities[1] == target and entities[0] == table)): continue
        points = []
        for point in contact.points:
            position, normal, impulse = (np.asarray(v, dtype=float) for v in (point.position, point.normal, point.impulse))
            separation = float(point.separation)
            _require(all(v.shape == (3,) and np.isfinite(v).all() for v in (position, normal, impulse))
                     and np.isfinite(separation), 'Invalid native source-contact point')
            points.append({'position_m': position.tolist(), 'normal': normal.tolist(),
                           'impulse_Ns': impulse.tolist(), 'separation_m': separation})
        rows.append({'target_is_first_body': entities[0] == target, 'points': points,
                     'physical_contact': physical_contact_points(points)})
    return {'physical_contact': any(row['physical_contact'] for row in rows),
            'matching_manifolds': len(rows), 'point_count': sum(len(row['points']) for row in rows), 'contacts': rows}


class _RoleAttachedChecker(NumericalTableSupportCollisionChecker):
    """Role contacts and a stricter, measured-only permanent source departure.

    The departure proof extends the bounded initial-support phase until an
    actual >3mm, all-piece geometric and native-contact exit. This is a declared
    transfer protocol change; the inherited support motion/depth gates remain.
    """

    def _update_support_release(self, objects: dict[str, Any], actual_object: np.ndarray) -> None:
        """Commit once from one measured refresh; candidate queries never call this hook."""
        _require(self._legacy_support_mode is None, 'Transfer departure requires whole-object convex support geometry')
        actual = _rigid_transform(actual_object)
        before = bool(self.support_released)
        if self._table_pose is None or not self._initial_support_exists:
            self._support_state_receipt = {'mode': 'measured_refresh',
                'reason': 'no_table' if self._table_pose is None else 'no_initial_support',
                'released_before': before, 'released_after': before, 'control_step': self._control_step(),
                'source_departure_policy': 'transfer_height_geometry_native_contact_v1'}
            return
        clock = _step(self.env)
        self._table_pose, self._table_side = self._table(objects)
        pieces = self._pieces(actual)
        _require(bool(pieces), 'Departure proof requires all target convex pieces')
        touching = [name for name in sorted(self._target_names) if _pair_result(objects[name], objects[TABLE_NAME])[0]]
        all_clear = bool(all(p['bottom_minus_table_top_m'] > NUMERICAL_TOP_TOLERANCE_M for p in pieces) and not touching)
        reference_z = float(self._initial_object[2, 3]); measured_z = float(actual[2, 3])
        displacement_z = measured_z-reference_z
        _require(np.isfinite([reference_z, measured_z, displacement_z, self.support_band_m]).all(), 'Invalid departure heights')
        table_entity = self.live._shape_refs[TABLE_NAME][0]
        source = _source_contact_evidence(self.env.scene.px.get_contacts(), self._entity, table_entity)
        _require(_step(self.env) == clock, 'Native contact query advanced the control clock')
        height_clear = displacement_z > self.support_band_m
        qualified = bool(height_clear and all_clear and not source['physical_contact'])
        self.support_released = bool(before or qualified)
        reason = 'already_released' if before else ('qualified_geometry_and_native_source_exit' if qualified else 'object_support_active')
        self._support_state_receipt = {'mode': 'measured_refresh', 'reason': reason,
            'released_before': before, 'released_after': self.support_released, 'control_step': clock,
            'actor_vertical_displacement_m': displacement_z, 'reference_origin_z_m': reference_z,
            'measured_origin_z_m': measured_z, 'release_height_bound_m': self.support_band_m,
            'release_height_comparison': 'strict_gt',
            'height_bound_strictly_exceeded': bool(height_clear),
            'separation_numeric_band_m': NUMERICAL_TOP_TOLERANCE_M, 'all_pieces_separated': all_clear,
            'fcl_touching_pieces': touching, 'pieces': pieces,
            'original_geometric_release_candidate': bool(height_clear or all_clear),
            'native_source_contact': source['physical_contact'], 'native_source_contact_evidence': source,
            'source_departure_policy': 'transfer_height_geometry_native_contact_v1',
            'policy_effect': 'stricter departure evidence extends the original bounded initial-support phase'}
        previous = self._support_state_history[-1] if self._support_state_history else None
        identity = (reason, self.support_released, touching, source['physical_contact'], bool(height_clear), all_clear)
        prior_identity = None if previous is None else (previous['reason'], previous['released_after'], previous['fcl_touching_pieces'],
            previous.get('native_source_contact'), previous.get('height_bound_strictly_exceeded'), previous['all_pieces_separated'])
        if identity != prior_identity: self._support_state_history.append(copy.deepcopy(self._support_state_receipt))

    def _allowed_robot_contact(self, record: dict[str, Any]) -> bool:
        permitted = getattr(self, '_transfer_finger_robots', frozenset((self.active_robot,)))
        for robot_key, link_key, other_key in (('object_name1', 'link_name1', 'object_name2'),
                                               ('object_name2', 'link_name2', 'object_name1')):
            robot, link, other = record.get(robot_key), record.get(link_key), record.get(other_key)
            if (robot, link, other) in self.live.allowed_mount_contacts: return True
            if robot in permitted and link in FINGER_LINKS and other in self._target_names: return True
        return False


class TransferCollisionChecker:
    """Keep exactly one target actor, one active owner, and immutable captures.

    Confirmation evidence contains consecutive actual control steps, native
    grasp_flags[2], measured finger_change_m[2], and fingers_open[2] for release.
    Evidence is produced by transfer_execution; raw current predicates are
    checked again before a capture/owner change. An exception is never no-solution.
    """

    def __init__(self, raw: Any, target_actor: Any, giver_index: int, giver_checker: Any,
                 receiver_factory: Callable[[], Any]) -> None:
        self.env, self.target_actor = raw, target_actor
        self.giver_index, self.receiver_index = _index(giver_index), 1-_index(giver_index)
        self.owner_index = self.giver_index
        self._owned = {self.giver_index: giver_checker}
        self._receiver_factory = receiver_factory
        self._frozen = {self.giver_index: np.asarray(giver_checker.attachment_T_tcp_object).copy()}
        self.capture_counts = {self.giver_index: 1, self.receiver_index: 0}
        self.phase = 'giver_hold'; self.events: list[dict[str, Any]] = []
        self.coverage: dict[str, Any] = {}
        self.last_allowed_contacts: list[dict[str, Any]] = []
        self.scope = 'one_unique_actual_payload_explicit_handoff_fingertips_existing_full_scene_checker'
        self._check_unique(giver_checker)
        self.set_phase('giver_hold')

    @classmethod
    def from_solvers(cls, env: Any, left_solver: Any, right_solver: Any, *, target_actor: Any,
                     giver_index: int = 1) -> TransferCollisionChecker:
        raw = getattr(env, 'unwrapped', env); giver = _index(giver_index)
        _require(len(raw.agent.agents) == 2 and target_actor is not None, 'Two actual robots and one target actor required')
        _require(_flag(raw.agent.agents[giver].is_grasping(target_actor)), 'Actual giver grasp required before capture')
        before, clock = _state(raw), _step(raw)
        def factory(index: int) -> Any:
            live = LiveGraspCollisionChecker.from_solvers(raw, left_solver, right_solver,
                target_actor=target_actor, active_robot=ROBOTS[index])
            return _RoleAttachedChecker.from_live(live, target_actor, active_robot=ROBOTS[index])
        result = cls(raw, target_actor, giver, factory(giver), lambda: factory(1-giver))
        after = _state(raw)
        _require(clock == _step(raw) and set(before) == set(after)
                 and all(np.array_equal(before[k], after[k]) for k in before), 'Collision capture changed physical state')
        return result

    def _check_unique(self, checker: Any) -> None:
        names = tuple(checker._normal_names); targets = frozenset(checker._target_names)
        _require(len(names) == len(set(names)) and bool(targets) and targets <= set(names), 'Payload shapes duplicated or missing')
        if self._owned:
            original = self._owned[self.giver_index]
            _require(targets == frozenset(original._target_names), 'Receiver targets a different physical actor')
            _require(checker.live.target_actor_name == original.live.target_actor_name, 'Unique actor identity changed')

    @property
    def active(self) -> Any:
        return self._owned[self.owner_index]

    @property
    def models(self) -> Any:
        return self.active.models

    @property
    def world(self) -> Any:
        return self.active.world

    @property
    def attachment_T_tcp_object(self) -> np.ndarray:
        return self._frozen[self.owner_index].copy()

    @property
    def target_actor_name(self) -> str:
        return str(self._owned[self.giver_index].live.target_actor_name)

    @property
    def allowed_finger_robots(self) -> tuple[str, ...]:
        """Public contact policy for the native caller's100Hz raw classifier."""
        allowed = self.active._transfer_finger_robots
        return tuple(robot for robot in ROBOTS if robot in allowed)

    def allows_target_contact(self, robot_index: int, link_name: str) -> bool:
        """Only this unique target may use the returned finger permission."""
        return ROBOTS[_index(robot_index)] in self.allowed_finger_robots and link_name in FINGER_LINKS

    def reference(self, robot_index: int) -> np.ndarray:
        _require(_index(robot_index) in self._frozen, 'Requested owner has no actual confirmed capture')
        return self._frozen[robot_index].copy()

    def set_phase(self, phase: str) -> None:
        _require(phase in PHASES, 'Unknown handoff contact phase')
        if phase in ('receiver_hold', 'giver_retreat'):
            _require(self.owner_index == self.receiver_index, 'Receiver must own prediction after confirmed giver release')
        else:
            _require(self.owner_index == self.giver_index, 'Giver-only phase cannot reopen after ownership transfer')
        self.phase = phase
        both = phase in ('receiver_contact', 'receiver_confirmed', 'giver_release', 'giver_retreat')
        allowed = frozenset(ROBOTS if both else (ROBOTS[self.owner_index],))
        for checker in self._owned.values(): checker._transfer_finger_robots = allowed

    def _evidence(self, values: list[dict[str, Any]], count: int, *, release: bool) -> None:
        _require(len(values) == count, 'Incomplete handoff confirmation interval')
        steps = [row['environment_step'] for row in values]
        _require(all(type(s) is int for s in steps) and steps == list(range(steps[0], steps[0]+count))
                 and steps[-1] == _step(self.env), 'Confirmation is stale or not consecutive')
        for row in values:
            grasp = row['grasp_flags']
            _require(len(grasp) == 2 and all(type(v) is bool for v in grasp) and grasp[self.receiver_index], 'Receiver has not maintained original native grasp')
            if release:
                _require(not grasp[self.giver_index] and row['fingers_open'][self.giver_index], 'Giver release not physically confirmed')
            else:
                change = np.asarray(row['finger_change_m'], dtype=float)
                _require(grasp[self.giver_index] and change.shape == (2,) and np.isfinite(change).all()
                         and change[self.receiver_index] <= .0002, 'Receiver grasp/finger stability not confirmed')

    def capture_receiver(self, confirmation: list[dict[str, Any]]) -> np.ndarray:
        _require(self.owner_index == self.giver_index and self.phase in ('receiver_contact', 'receiver_confirmed')
                 and self.capture_counts[self.receiver_index] == 0, 'Receiver capture cannot be repeated or precede contact phase')
        self._evidence(confirmation, 5, release=False)
        _require(all(_flag(a.is_grasping(self.target_actor)) for a in self.env.agent.agents), 'Fresh actual double grasp required')
        before, clock = _state(self.env), _step(self.env)
        checker = self._receiver_factory(); self._check_unique(checker)
        ref = np.asarray(checker.attachment_T_tcp_object).copy()
        _require(ref.shape == (4, 4) and np.isfinite(ref).all(), 'Invalid actual receiver relation')
        after = _state(self.env)
        _require(clock == _step(self.env) and set(before) == set(after)
                 and all(np.array_equal(before[k], after[k]) for k in before), 'Receiver capture changed physical state')
        self._owned[self.receiver_index] = checker; self._frozen[self.receiver_index] = ref
        self.capture_counts[self.receiver_index] = 1
        self.set_phase('receiver_confirmed')
        self.events.append({'event': 'receiver_reference_captured', 'environment_step': clock,
                            'T_tcp_object': ref.tolist(), 'physical_attachment': False})
        return ref.copy()

    def commit_receiver_owner(self, confirmation: list[dict[str, Any]]) -> None:
        _require(self.owner_index == self.giver_index and self.phase == 'giver_release'
                 and self.capture_counts[self.receiver_index] == 1, 'Invalid prediction ownership transition')
        self._evidence(confirmation, 3, release=True)
        agents = self.env.agent.agents
        _require(not _flag(agents[self.giver_index].is_grasping(self.target_actor))
                 and _flag(agents[self.receiver_index].is_grasping(self.target_actor)), 'Fresh native release/grasp mismatch')
        self.owner_index = self.receiver_index; self.set_phase('giver_retreat')
        self.events.append({'event': 'prediction_owner_transferred', 'environment_step': _step(self.env),
                            'owner_index': self.owner_index, 'physical_attachment': False})

    def refresh(self) -> dict[str, Any]:
        for index, checker in self._owned.items():
            checker.refresh(); self._check_unique(checker)
            _require(np.array_equal(checker.attachment_T_tcp_object, self._frozen[index]), 'Original owner relation was recaptured')
        self.coverage = copy.deepcopy(self.active.coverage)
        self.coverage.update(scope=self.scope, unique_payload_actor_count=1, owner_index=self.owner_index,
            phase=self.phase, capture_counts=dict(self.capture_counts), physical_attachment=False,
            owner_references={str(k): v.tolist() for k, v in self._frozen.items()},
            owner_support_evidence={str(k): {field: copy.deepcopy(v.coverage.get(field)) for field in
                ('support_state', 'support_state_transition_history', 'support_validation_history', 'initial_support_geometry')}
                for k, v in self._owned.items()})
        return self.coverage

    def check(self, left7: np.ndarray, right7: np.ndarray) -> list[dict[str, Any]]:
        result = self.active.check(left7, right7)
        self.last_allowed_contacts = copy.deepcopy(self.active.last_allowed_contacts)
        return result

    def measured_check(self) -> list[dict[str, Any]]:
        result = self.active.measured_check()
        self.last_allowed_contacts = copy.deepcopy(self.active.last_allowed_contacts)
        return result


def self_test() -> dict[str, Any]:
    from unittest.mock import patch
    tests = []; role = _RoleAttachedChecker.__new__(_RoleAttachedChecker)
    role.active_robot = 'right'; role._target_names = {'banana::shape_0'}; role.live = NS(allowed_mount_contacts=set())
    record = {'object_name1': 'left', 'link_name1': 'panda_leftfinger', 'object_name2': 'banana::shape_0'}
    assert not role._allowed_robot_contact(record)
    role._transfer_finger_robots = frozenset(ROBOTS)
    assert role._allowed_robot_contact(record)
    assert not role._allowed_robot_contact({**record, 'link_name1': 'panda_hand'})
    assert not role._allowed_robot_contact({**record, 'object_name2': 'table'})
    assert not role._allowed_robot_contact({'object_name1': 'left', 'link_name1': 'panda_leftfinger', 'object_name2': 'right'})
    tests.append('only explicit fingertip-to-unique-target contact pairs allowed')
    class FakeHeld:
        def __init__(self, index: int):
            self._normal_names = ('banana::shape_0', 'table'); self._target_names = {'banana::shape_0'}
            self.live = NS(target_actor_name='banana'); self.attachment_T_tcp_object = np.eye(4)
            self.attachment_T_tcp_object[0, 3] = .01*index
            self.models = (); self.world = NS(); self.coverage = {}; self.last_allowed_contacts = []
        def refresh(self) -> dict: return {}
        def check(self, *joints: Any) -> list: return []
        def measured_check(self) -> list: return []
    flags = [True, True]; raw = NS(elapsed_steps=np.array([5]), agent=NS(agents=[NS(is_grasping=lambda t, i=i: np.array([flags[i]])) for i in range(2)]))
    raw.get_state_dict = lambda: {'state': np.array([[1., 2., 3.]])}
    checker = TransferCollisionChecker(raw, object(), 1, FakeHeld(1), lambda: FakeHeld(0))
    checker._owned[1].coverage = {'support_state': {'released_after': True},
        'support_state_transition_history': [{'control_step': 4}],
        'support_validation_history': [{'status': 'allowed'}]}
    checker.set_phase('receiver_contact')
    confirm = [{'environment_step': s, 'grasp_flags': [True, True], 'finger_change_m': [0., 0.]} for s in range(1, 6)]
    receiver = checker.capture_receiver(confirm)
    assert checker.owner_index == 1 and checker.capture_counts == {1: 1, 0: 1}
    try: checker.capture_receiver(confirm)
    except ValueError: pass
    else: raise AssertionError('Receiver recapture accepted')
    checker.set_phase('giver_release'); raw.elapsed_steps[0] = 8; flags[1] = False
    release = [{'environment_step': s, 'grasp_flags': [True, False], 'fingers_open': [False, True]} for s in range(6, 9)]
    checker.commit_receiver_owner(release); checker.refresh()
    assert checker.owner_index == 0 and np.array_equal(checker.attachment_T_tcp_object, receiver)
    assert checker.coverage['unique_payload_actor_count'] == 1
    evidence = checker.coverage['owner_support_evidence']
    assert set(evidence) == {'0', '1'} and evidence['1']['support_state']['released_after']
    checker._owned[1].coverage['support_state_transition_history'][0]['control_step'] = 99
    assert evidence['1']['support_state_transition_history'][0]['control_step'] == 4
    evidence['1']['support_validation_history'][0]['status'] = 'snapshot-only'
    assert checker._owned[1].coverage['support_validation_history'][0]['status'] == 'allowed'
    tests.append('giver support evidence survives receiver ownership; history snapshots are deep copies')
    checker.set_phase('receiver_hold')
    assert checker.active._transfer_finger_robots == {'left'}
    assert checker.allowed_finger_robots == ('left',) and checker.allows_target_contact(0, 'panda_leftfinger')
    assert not checker.allows_target_contact(1, 'panda_leftfinger') and not checker.allows_target_contact(0, 'panda_hand')
    tests.append('one capture per actual grasp and ownership only after3tick release')
    rejected = 0
    for callback in (lambda: checker.set_phase('giver_hold'),
                     lambda: checker.commit_receiver_owner(release),
                     lambda: checker._evidence(confirm[:-1], 5, release=False),
                     lambda: checker._evidence(confirm, 5, release=False)):
        try: callback()
        except ValueError: rejected += 1
        else: raise AssertionError('Invalid/stale phase transition accepted')
    assert rejected == 4
    target, table, unrelated = object(), object(), object()
    def native_contact(separation: float = .001, impulse: Any = (0., 0., 0.), *, reverse: bool = False) -> Any:
        bodies = (table, target) if reverse else (target, table)
        return NS(bodies=[NS(entity=e) for e in bodies], points=[NS(position=[0., 0., 0.], normal=[0., 0., 1.],
            impulse=impulse, separation=separation)])
    assert not _source_contact_evidence([native_contact()], target, table)['physical_contact']
    for contact in (native_contact(0.), native_contact(-1e-7), native_contact(.001, (1e-12, 0., 0.)),
                    native_contact(.001, (0., -1e-12, 0.), reverse=True)):
        assert _source_contact_evidence([contact], target, table)['physical_contact']
    other = native_contact(0.); other.bodies[1].entity = unrelated
    assert not _source_contact_evidence([other], target, table)['physical_contact']
    tests.append('native source entity pair in either order; nonpositive separation or any nonzero impulse is physical contact')
    def fixture() -> tuple[Any, dict[str, Any], dict[str, Any]]:
        held = _RoleAttachedChecker.__new__(_RoleAttachedChecker)
        state = {'gap': 2e-6, 'touching': False, 'contacts': [], 'reads': 0}
        def get_contacts() -> list:
            state['reads'] += 1
            return state['contacts']
        held.env = NS(elapsed_steps=np.array([10]), scene=NS(px=NS(get_contacts=get_contacts)))
        held._entity = target; held.live = NS(_shape_refs={TABLE_NAME: (table, object())})
        held._legacy_support_mode = None; held._initial_support_exists = True; held.support_released = False
        held._initial_object = np.eye(4); held._initial_object[2, 3] = .01
        held._table_pose = np.eye(4); held._table_pose[2, 3] = -.5
        held._table_side = np.array([2., 2., 1.]); held._target_names = frozenset(('banana::shape_0',))
        held.support_band_m = .003; held.support_penetration_m = .003; held.support_depth_slop_m = .0002
        held._initial_object_depth_m = 1e-5; held._support_state_history = []; held._support_validation_history = []
        held.coverage = {}; held._table = lambda objects: (held._table_pose, held._table_side)
        held._pieces = lambda pose: [{'bottom_minus_table_top_m': state['gap']}]
        held._local_vertices = {'banana::shape_0': np.array([[x, y, z] for x in (-.01, .01)
            for y in (-.01, .01) for z in (-.01, .01)])}
        held._shape_local = {'banana::shape_0': np.eye(4)}
        return held, state, {TABLE_NAME: object(), 'banana::shape_0': object()}
    def refresh_at(held: Any, state: dict, objects: dict, dz: float) -> None:
        actual = held._initial_object.copy(); actual[2, 3] += dz
        with patch(__name__+'._pair_result', return_value=(state['touching'], [])):
            held._update_support_release(objects, actual)
    held, state, objects = fixture()
    held._initial_object[2, 3] = 0.  # exact representable dz==.003 boundary, without subtraction cancellation
    for dz in (83.219e-6, .003):
        refresh_at(held, state, objects, dz); assert not held.support_released
    state['gap'] = NUMERICAL_TOP_TOLERANCE_M
    refresh_at(held, state, objects, .0031); assert not held.support_released
    state['gap'] = NUMERICAL_TOP_TOLERANCE_M+1e-12; state['touching'] = True
    refresh_at(held, state, objects, .0031); assert not held.support_released
    state['touching'] = False
    for contact in (native_contact(0.), native_contact(.001, (0., 0., 1e-12))):
        state['contacts'] = [contact]; refresh_at(held, state, objects, .0031); assert not held.support_released
    state['contacts'] = [native_contact()]
    refresh_at(held, state, objects, .0031); assert held.support_released
    assert held._support_state_receipt['release_height_comparison'] == 'strict_gt'
    state['contacts'] = [native_contact(-1e-4)]; state['gap'] = -1e-4; state['touching'] = True
    refresh_at(held, state, objects, 0.); assert held.support_released
    contact = {'depth': 1e-6, 'position': [0., 0., 0.], 'normal': [0., 0., 1.]}
    assert not held._support_allowed('banana::shape_0', TABLE_NAME, held._initial_object, [copy.deepcopy(contact)])
    tests.append('strict >3mm and >1um plus no FCL/native contact; irreversible release still rejects later recontact')
    pending, pending_state, pending_objects = fixture(); reads = pending_state['reads']
    assert pending._support_allowed('banana::shape_0', TABLE_NAME, pending._initial_object, [copy.deepcopy(contact)])
    assert not pending.support_released and pending_state['reads'] == reads and not pending._support_state_history
    bad_support = pending._initial_object.copy(); bad_support[0, 3] = .0011
    assert not pending._support_allowed('banana::shape_0', TABLE_NAME, bad_support, [copy.deepcopy(contact)])
    bad_support = pending._initial_object.copy(); bad_support[2, 3] += .0031
    assert not pending._support_allowed('banana::shape_0', TABLE_NAME, bad_support, [copy.deepcopy(contact)])
    bad_support = pending._initial_object.copy(); angle = .011
    bad_support[:3, :3] = [[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]]
    assert not pending._support_allowed('banana::shape_0', TABLE_NAME, bad_support, [copy.deepcopy(contact)])
    deep = {**contact, 'depth': .000211}
    assert not pending._support_allowed('banana::shape_0', TABLE_NAME, pending._initial_object, [deep])
    assert not pending.support_released and pending_state['reads'] == reads
    pending._initial_support_exists = False
    refresh_at(pending, pending_state, pending_objects, .1)
    assert not pending.support_released and pending_state['reads'] == reads
    assert pending._support_state_receipt['reason'] == 'no_initial_support'
    tests.append('candidate support queries do not advance release; original XY/height/rotation/depth bounds and no-initial-support semantics retained')
    invalid_evidence = 0
    for bad in (native_contact(float('nan')), native_contact(.001, [0., float('nan'), 0.]),
                native_contact(.001, [0., 0.])):
        try: _source_contact_evidence([bad], target, table)
        except ValueError: invalid_evidence += 1
        else: raise AssertionError('Invalid native source evidence accepted')
    broken, broken_state, broken_objects = fixture()
    broken.live._shape_refs = {}
    try: refresh_at(broken, broken_state, broken_objects, .0031)
    except KeyError: invalid_evidence += 1
    else: raise AssertionError('Missing table identity accepted')
    assert not broken.support_released
    broken, broken_state, broken_objects = fixture()
    def advancing_contact_query() -> list:
        broken.env.elapsed_steps[0] += 1
        return []
    broken.env.scene.px.get_contacts = advancing_contact_query
    try: refresh_at(broken, broken_state, broken_objects, .0031)
    except ValueError: invalid_evidence += 1
    else: raise AssertionError('Advancing native contact query accepted')
    assert not broken.support_released and invalid_evidence == 5
    tests.append('invalid/absent native evidence and a clock-changing query fail closed before release commit')
    return {'status': 'passed', 'checks': tests, 'invalid_transition_cases_rejected': rejected,
            'invalid_departure_evidence_rejected': invalid_evidence,
            'source_departure_policy': 'transfer_height_geometry_native_contact_v1',
            'unique_payload_actor_count': 1, 'environment_steps': 0, 'physics_steps': 0,
            'test_scope': 'CPU synthetic checker delegation and permission/state contracts; native geometry integration NOT_RUN'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--self-test', action='store_true')
    parsed = parser.parse_args()
    if not parsed.self_test: parser.error('Library only; use --self-test for CPU synthetic checks')
    print(json.dumps(self_test(), indent=2))
