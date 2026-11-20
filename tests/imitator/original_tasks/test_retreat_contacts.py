"""Initial release contacts disappear once; no new pair or re-entry is allowed."""

from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from imitator.planning._tig_sync import replay
from imitator.planning._tig_sync.observation import ObservedEnvironment
from imitator.planning._tig_sync.physical_observation import contact_allowed
from imitator.planning._tig_sync.release_contact import SeparationState
from imitator.planning._tig_sync.runtime_policy import DEFAULT_POLICY


LEFT, RIGHT = 'panda_leftfinger', 'panda_rightfinger'


def sides(index=0, link=LEFT, target_index=0):
    return [{'kind': 'robot', 'robot_index': index, 'link_name': link},
            {'kind': 'payload', 'target_index': target_index}]


def test_only_armed_own_finger_pair_is_allowed_in_retreat():
    armed = {(0, LEFT)}
    assert contact_allowed(sides(), 'retreat', {}, retreat_contacts=armed)
    assert contact_allowed(sides()[::-1], 'retreat', {}, retreat_contacts=armed)
    for pair in (sides(link=RIGHT), sides(link='panda_hand'), sides(target_index=1),
                 sides(index=1, target_index=1)):
        assert not contact_allowed(pair, 'retreat', {}, retreat_contacts=armed)
    assert not contact_allowed(sides(), 'retreat', {})
    assert not contact_allowed(sides(), 'retreat', {}, retreat_contacts=set())
    for phase in ('descend', 'close_and_confirm', 'lift_and_hold', 'carry_lower', 'release'):
        assert contact_allowed(sides(), phase, {}, retreat_contacts=set())
    assert not contact_allowed(sides(), 'verify_placement', {}, retreat_contacts=armed)


class Entity:
    def __init__(self, name):
        self.name = name


@pytest.fixture
def capture(tmp_path):
    urdf = tmp_path/'panda.urdf'
    urdf.with_suffix('.srdf').write_text('<robot/>')
    fingers = {(index, link): Entity(f'{index}:{link}') for index in (0, 1) for link in (LEFT, RIGHT)}
    agents = [NS(urdf_path=str(urdf), robot=NS(get_links=lambda index=index: [
        NS(name=link, _objs=[NS(entity=fingers[index, link])]) for link in (LEFT, RIGHT)]))
        for index in (0, 1)]
    targets = [NS(_objs=[Entity(f'target{index}')], pose=NS(p=np.zeros((1, 3)))) for index in (0, 1)]
    task = NS(target_names=('target0', 'target1'), targets=targets,
        initial_supports=(NS(name='source0'), NS(name='source1')),
        destination_supports=(NS(name='destination0'), NS(name='destination1')))
    current, reads = [], []

    def get_contacts():
        reads.append(True)
        return current

    raw = NS(agent=NS(agents=agents), scene=NS(actors=dict(zip(task.target_names, targets)),
        px=NS(get_contacts=get_contacts)), _after_simulation_step=lambda: None)
    capture = ObservedEnvironment(NS(unwrapped=raw), task, lambda: None, DEFAULT_POLICY)
    capture.phase = 'retreat'

    def contact(index, link, target=None, *, separation=-.0001, impulse=0.):
        return NS(bodies=[NS(entity=fingers[index, link]), NS(entity=targets[index if target is None else target]._objs[0])],
            points=[NS(position=np.zeros(3), normal=np.array([0., 1., 0.]),
                       separation=separation, impulse=np.array([impulse, 0., 0.]))])

    return capture, current, reads, contact


def test_substep_uses_contact_union_and_latches_each_missing_pair(capture):
    observer, current, reads, contact = capture
    assert not observer.release_separation_complete
    observer.begin_release_separation({(0, LEFT), (0, RIGHT), (1, LEFT)})
    assert not observer.release_separation_complete
    current[:] = [contact(0, LEFT), contact(0, LEFT), contact(0, RIGHT)]
    observer.record_substep()
    assert len(reads) == 1 and not observer.pending_errors
    assert observer._retreat_contacts == {(0, LEFT), (0, RIGHT)}
    current[:] = [contact(0, LEFT, separation=.0001), contact(0, RIGHT)]
    observer.record_substep()
    assert observer._retreat_contacts == {(0, RIGHT)}
    assert not observer.pending_errors  # Positive separation and zero impulse is proximity.
    current[:] = [contact(0, LEFT), contact(0, RIGHT)]
    observer.record_substep()
    assert [error['reason'] for error in observer.pending_errors] == ['forbidden_robot_contact']
    assert observer._retreat_contacts == {(0, RIGHT)}
    observer.end_release_separation()
    observer.end_release_separation()
    assert observer._retreat_contacts is None
    assert not observer.release_separation_complete
    assert observer.summary()['release_separation'] == {
        'initial_pairs': [[0, LEFT], [0, RIGHT], [1, LEFT]],
        'separated_pairs': [[0, LEFT], [1, LEFT]], 'remaining_pairs': [[0, RIGHT]]}


def test_new_pair_and_peer_target_stay_forbidden(capture):
    observer, current, _, contact = capture
    observer.begin_release_separation({(0, LEFT)})
    current[:] = [contact(0, LEFT), contact(0, RIGHT), contact(0, LEFT, target=1)]
    observer.record_substep()
    assert observer._retreat_contacts == {(0, LEFT)}
    assert len(observer.pending_errors) >= 2
    assert any(error['reason'] == 'forbidden_robot_contact' for error in observer.pending_errors)


def test_positive_separation_with_impulse_remains_actual_contact(capture):
    observer, current, _, contact = capture
    assert 'release_separation' not in observer.summary()
    observer.begin_release_separation({(0, LEFT)})
    current[:] = [contact(0, LEFT, separation=.0001, impulse=.001)]
    observer.record_substep()
    assert observer._retreat_contacts == {(0, LEFT)} and not observer.pending_errors
    current.clear()
    observer.record_substep()
    assert not observer._retreat_contacts
    assert observer.release_separation_complete
    observer.end_release_separation()
    assert not observer.release_separation_complete


def test_begin_unions_only_the_last_release_physical_contacts(capture):
    observer, current, reads, contact = capture
    observer.phase = 'release'
    current[:] = [contact(0, LEFT, separation=.0001, impulse=.001)]
    observer.record_substep()
    observer.phase = 'retreat'
    observer.begin_release_separation({(1, RIGHT)})
    assert len(reads) == 1  # Arming reuses the preceding substep; no extra read.
    assert observer._retreat_contacts == {(0, LEFT), (1, RIGHT)}
    observer.record_substep()
    assert observer._retreat_contacts == {(0, LEFT)} and not observer.pending_errors
    observer.end_release_separation()
    observer.phase = 'descend'
    observer.record_substep()
    observer.phase = 'retreat'
    observer.begin_release_separation(set())
    assert not observer._retreat_contacts  # A stale approach contact cannot re-arm retreat.


@pytest.mark.parametrize('fail_preflight, geometry_clear, physical_clear', [
    (False, True, True), (True, True, True), (False, False, True), (False, True, False)])
def test_replay_requires_separation_before_zero_step_outcome_and_always_cleans_up(
        monkeypatch, fail_preflight, geometry_clear, physical_clear):
    events = []
    armed = {(0, LEFT)}
    separation = SeparationState({'initial-pair': -.0001})
    agents = [NS(control_mode='pd_joint_pos', robot=NS(pose=None),
                 tcp=NS(pose=NS(p=np.zeros((1, 3))))) for _ in range(2)]
    raw = NS(num_envs=1, control_timestep=.05, sim_timestep=.01, agent=NS(agents=agents),
        robot_uids=('panda_wristcam',)*2,
        elapsed_steps=np.array([0]), get_state_dict=lambda: {},
        set_state_dict=lambda state: None,
        evaluate=lambda: {'success': np.array([True])})
    env = NS(unwrapped=raw, reset=lambda **kwargs: None, close=lambda: events.append('env.close'))
    task = NS(targets=(object(), object()), official_predicate_keys=('left', 'right'),
        region_predicates=lambda: (lambda actor: {'inside': True},)*2)

    class Checker:
        release_separation = separation

        def set_measured_attachment_prediction(self, value):
            pass

        def set_target_contacts_allowed(self, value):
            pass

        def begin_release_separation(self):
            events.append('checker.begin')
            return armed

        def end_release_separation(self):
            events.append('checker.end')
            self.release_separation = None

    class Capture:
        def __init__(self, *args):
            self.phase, self.last_step = None, None
            self.release_separation_complete = False

        def install(self):
            pass

        def restore(self):
            events.append('capture.restore')

        def begin_release_separation(self, allowed):
            assert allowed is armed
            events.append('capture.begin')

        def end_release_separation(self):
            events.append('capture.end')

        def summary(self):
            return {'max_held_position_slip_m': [0., 0.], 'max_held_rotation_slip_rad': [0., 0.]}

    class Phase:
        def __init__(self, capture, checker, name, **kwargs):
            self.name = name
            self.capture = capture
            self.arms = [NS(key='left'), NS(key='right')]
            self.row = {'measurements': {arm.key: {'finger_positions_tcp_m': [[0., 0., 0.]]*2}
                                        for arm in self.arms},
                'objects': [{'is_grasping': False} for _ in self.arms], 'tracking_error_rad': 0.}
            self.report = {'initial_observation': self.row, 'initial_measurements': self.row['measurements'],
                'trace': [], 'status': 'running'}

        def preflight(self):
            events.append(f'{self.name}.preflight')
            return not (fail_preflight and self.name == 'retreat')

        def tick(self, *args, native_actions=False):
            assert native_actions  # Saved commands are already in native controller coordinates.
            for agent in agents:
                agent.tcp.pose.p[0, 2] += .081
            raw.elapsed_steps += 1
            if geometry_clear:
                assert not separation.check({'initial-pair': .001})
            self.capture.release_separation_complete = physical_clear
            return self.row

        def opened(self, *args):
            return True

    policy = replace(DEFAULT_POLICY, require_final_stability=False)
    native = dict(robot_uids=['panda_wristcam']*2, action_order=[], model_fingerprints=['fixture']*2, mounts=[])
    monkeypatch.setattr(replay, 'native_contract', lambda *args: native)
    monkeypatch.setattr(replay, 'load_episode', lambda *args: (
        {'env_info': {'env_id': replay.TASK_ID, 'env_kwargs': {}}, 'native_data_contract': native,
         'motion_planning': {'action_order': list(replay.ACTION_ORDER), 'planner_backend': 'curobo'}},
        {'episode_seed': 401, 'reset_kwargs': {}, 'dual_sync': {'backend': 'curobo'}}, policy,
        [('pregrasp', 0), ('retreat', 1), ('observe_task_outcome', 0)],
        np.zeros((1, 16), dtype=np.float32), {}))
    monkeypatch.setattr(replay.gym, 'make', lambda *args, **kwargs: env)
    monkeypatch.setattr(replay, 'task_components', lambda task_id: (
        lambda raw: task, NS(from_solvers=lambda *args, **kwargs: Checker())))
    monkeypatch.setattr(replay, 'pill_planners', lambda *args, **kwargs: (object(), object()))
    monkeypatch.setattr(replay, 'ObservedEnvironment', Capture)
    monkeypatch.setattr(replay, '_Phase', Phase)
    result = replay.replay('unused.h5')
    assert 'checker.begin' in events, (result, events)
    assert events.index('checker.begin') < events.index('capture.begin') < events.index('retreat.preflight')
    assert events.index('retreat.preflight') < events.index('checker.end') < events.index('env.close')
    assert events.index('retreat.preflight') < events.index('capture.end') < events.index('capture.restore')
    passed = not fail_preflight and geometry_clear and physical_clear
    assert result['execution']['status'] == ('passed' if passed else 'failed')
    assert ('observe_task_outcome.preflight' in events) == passed
    if not fail_preflight and not geometry_clear:
        assert 'geometry separation' in result['execution']['reason']
    if not fail_preflight and geometry_clear and not physical_clear:
        assert 'physical contact separation' in result['execution']['reason']
    assert raw.elapsed_steps.item() == (0 if fail_preflight else 1)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
