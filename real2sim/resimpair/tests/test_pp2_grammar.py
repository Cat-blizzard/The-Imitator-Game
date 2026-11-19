import numpy as np
import pytest

from resimpair_pp2 import grammar as g


ITEMS = [
    {'id': 'o0', 'category': 'cup', 'color': 'red'},
    {'id': 'o1', 'category': 'cup', 'color': 'blue'},
    {'id': 'o2', 'category': 'lemon', 'color': 'yellow'},
    {'id': 'r0', 'category': 'bowl', 'color': 'white'},
    {'id': 'r1', 'category': 'plate', 'color': 'blue'},
]


def test_minimal_descriptors_add_color_only_when_needed():
    d = g.minimal_descriptors(ITEMS)
    assert d['o2'] == {'category': 'lemon'}
    assert d['o0'] == {'category': 'cup', 'color': 'red'}
    assert g.resolve_descriptor({'category': 'cup', 'color': 'blue'}, ITEMS) == 'o1'
    with pytest.raises(ValueError):
        g.resolve_descriptor({'category': 'cup'}, ITEMS)
    same = ITEMS + [{'id': 'o3', 'category': 'cup', 'color': 'red'}]
    assert g.minimal_descriptors(same)['o0'] is None


def test_order_only_programs_are_distinct_but_unordered_is_confusable():
    d = g.minimal_descriptors(ITEMS)
    a, b = g.step(d['o0'], 'in', d['r0']), g.step(d['o2'], 'in', d['r0'])
    ab = g.program([a, b], True, 'multi_ordered')
    ba = g.program([b, a], True, 'multi_ordered')
    any_order = g.program([a, b], False, 'multi_same')
    assert not g.confusable(ab, ba)
    assert g.difference(ab, ba) == 'order'
    assert g.confusable(ab, any_order) and g.confusable(ba, any_order)
    assert g.instruction(ab) == 'First put the red cup in the bowl, then put the lemon in the bowl.'


def test_next_to_requires_side_and_classifies_differences():
    d = g.minimal_descriptors(ITEMS)
    with pytest.raises(ValueError):
        g.step(d['o0'], 'next_to', d['r0'])
    left = g.program([g.step(d['o0'], 'next_to', d['r0'], 'left')], False, 'single_next_to')
    right = g.program([g.step(d['o0'], 'next_to', d['r0'], 'right')], False, 'single_next_to')
    other = g.program([g.step(d['o1'], 'next_to', d['r0'], 'left')], False, 'single_next_to')
    assert g.difference(left, right) == 'destination'
    assert g.difference(left, other) == 'object'


def test_select_task_set_meets_rules():
    d = g.minimal_descriptors(ITEMS)
    rng = np.random.default_rng(3)
    programs, report = g.select_task_set(['o0', 'o1', 'o2'], {'r0': 'in', 'r1': 'on'}, d, 6, rng,
                                         fits=lambda r, objs: r == 'r0', feasible=lambda p: True)
    assert len(programs) >= 6  # k is a floor: reserved contrast and next_to slots may add more
    assert g.validate_task_set(programs) == []
    assert report['coverage_complete']
    kinds = {k for row in report['pairwise_differences'] for k in row}
    assert {'object', 'destination', 'order'} <= kinds
