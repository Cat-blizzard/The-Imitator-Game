"""Adversarial measured-array cases for the new release event semantics.

These synthetic arrays test the evaluator contract; they are not simulation
or evidence that low-drop control works on any particular asset.
"""
import copy

import numpy as np
import pytest

from resimpair_pickplace import drop_event_protocol_v1 as revised
from corl2026_workshop.experiments.autonomous_pipeline_v4.event_protocol import compile_events


@pytest.fixture
def inputs():
    objects = [{'id': name, 'semantic_description': name, 'shape': 'box',
                'half_size': [.02, .02, .02], 'color_name': 'red'} for name in ('payload', 'bystander')]
    scene = {'objects': objects, 'robot_roots': [[-.4, -.3, 0], [-.4, .3, 0]],
        'supports': [{'id': 'bin', 'semantic_description': 'bin', 'shape': 'mesh',
                      'half_size': [.15, .15, .04], 'position': [.25, 0., 0.], 'top_z': .01}]}
    task = {'id': 'test_drop', 'family': 'pick_place', 'required_skills': ['pick_place'],
        'object_ids': ['payload'], 'equivalence_groups': [], 'required_operation_precedence': [],
        'placement_contract': {'schema': 'supported_or_low_drop_v1',
            'maximum_release_height_over_goal_m': .15, 'xy_tolerance_m': .015,
            'continuous_hold_until_release': True},
        'operations': [{'id': 'move', 'object': 'payload', 'primitive': 'pick_place', 'arm': 0,
            'target': {'support': 'bin'}, 'release': {'mode': 'drop_from_above'}}]}
    return scene, task, ['payload', 'bystander'], ['payload', 'bystander', 'bin', 'table'], {'payload': [.25, 0., .03]}


def measurements(supported=False):
    n = 48
    data = {'timestamps_seconds': np.arange(n) * .05, 'object_states': np.zeros((n, 2, 13)),
            'grasp': np.zeros((n, 2, 2), dtype=bool), 'support_impulses': np.zeros((n, 2, 4)),
            'robot_object_force_N': np.zeros((n, 2, 2)), 'cross_robot_force_N': np.zeros(n),
            'tcp_world': np.tile(np.eye(4), (n, 2, 1, 1))}
    data['object_states'][:, :, 3] = 1.
    data['object_states'][:, 1, :3] = [-.2, -.2, .03]
    data['object_states'][:, 0, :3] = [.25, 0, .03]
    data['object_states'][:3, 0, :3] = [0, 0, .03]
    data['object_states'][3, 0, :3] = [0, 0, .10]
    data['object_states'][4:8, 0, 2] = .10
    data['object_states'][8, 0, 2] = .06
    if supported: data['object_states'][5:9, 0, 2] = .03
    data['grasp'][2:7, 0, 0] = True
    data['support_impulses'][7 if supported else 9:, 0, 2] = .02
    data['tcp_world'][:, :, 0, 3] = -.5
    return data


def check(protocol, data):
    return compile_events(protocol, data)['qualification']['passed']


def test_drop_changes_event_semantics_but_all_final_goals_remain_identical(inputs):
    baseline = revised._retained_builder(*inputs)
    candidate = revised.pick_place(*inputs)
    assert candidate['goals'] == baseline['goals']
    assert candidate['terminal_window_frames'] == baseline['terminal_window_frames'] == 20
    assert check(candidate, measurements())
    assert not check(baseline, measurements())


def test_supported_control_and_drop_use_exact_same_protocol(inputs):
    first = revised.pick_place(*inputs)
    other = copy.deepcopy(inputs)
    other[1]['operations'][0]['release']['mode'] = 'supported_place'
    second = revised.pick_place(*other)
    assert first == second
    assert check(first, measurements(supported=True))
    assert check(revised._retained_builder(*inputs), measurements(supported=True))


@pytest.mark.parametrize('failure', ['early_release', 'release_too_high', 'release_lateral',
    'final_position', 'final_tilt', 'final_speed', 'final_angular_speed', 'missing_support',
    'bystander_moved', 'bystander_contact', 'robot_collision'])
def test_bad_measured_outcomes_do_not_pass(inputs, failure):
    data = measurements()
    if failure == 'early_release': data['grasp'][5, 0, 0] = False
    elif failure == 'release_too_high': data['object_states'][6:8, 0, 2] = .23
    elif failure == 'release_lateral': data['object_states'][6:8, 0, 1] = .08
    elif failure == 'final_position': data['object_states'][-20:, 0, 0] += .016
    elif failure == 'final_tilt': data['object_states'][-20:, 0, 3:7] = [np.sqrt(.5), np.sqrt(.5), 0, 0]
    elif failure == 'final_speed': data['object_states'][-20:, 0, 7] = .026
    elif failure == 'final_angular_speed': data['object_states'][-20:, 0, 10] = .41
    elif failure == 'missing_support': data['support_impulses'][:] = 0
    elif failure == 'bystander_moved': data['object_states'][6:, 1, 0] += .009
    elif failure == 'bystander_contact': data['robot_object_force_N'][6, 0, 1] = .6
    elif failure == 'robot_collision': data['cross_robot_force_N'][6] = .6
    try: passed = check(revised.pick_place(*inputs), data)
    except ValueError: passed = False
    assert not passed, failure


def test_height_and_final_tolerances_cannot_be_relaxed(inputs):
    for field, bad in [('maximum_release_height_over_goal_m', .151), ('xy_tolerance_m', .02),
                       ('continuous_hold_until_release', False)]:
        bad_inputs = copy.deepcopy(inputs)
        bad_inputs[1]['placement_contract'][field] = bad
        with pytest.raises(ValueError): revised.pick_place(*bad_inputs)
