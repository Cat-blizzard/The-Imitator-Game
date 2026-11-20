"""CPU original-surface geometry and bounded one-way release contact state."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

SOURCE = (Path(__file__).resolve().parents[3] /
    'imitator/planning/_tig_sync/release_contact.py')
SPEC = spec_from_file_location('release_contact_cpu', SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
SeparationState = MODULE.SeparationState
triangle_box_gap_m = MODULE.triangle_box_gap_m


def cube_triangles(half: float = 1.) -> np.ndarray:
    vertices = np.array([[x, y, z] for x in (-half, half) for y in (-half, half) for z in (-half, half)])
    faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
        [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return vertices[faces]


def test_sat_clear_touching_intersect_and_both_containments() -> None:
    triangles = cube_triangles()
    assert triangle_box_gap_m([1, 1, 1], triangles+[3., 0, 0]) == 1.
    assert triangle_box_gap_m([1, 1, 1], triangles+[2., 0, 0]) == 0.
    assert triangle_box_gap_m([1, 1, 1], triangles+[1.5, 0, 0]) < 0
    assert triangle_box_gap_m([.25, .25, .25], triangles) < 0
    assert triangle_box_gap_m([2, 2, 2], triangles) < 0
    assert triangle_box_gap_m([.25, .25, .25], triangles[:, ::-1]) < 0


def test_small_release_gap_changes_have_correct_sign() -> None:
    triangles = cube_triangles(.01)
    assert triangle_box_gap_m([.01]*3, triangles+[.02001, 0, 0]) > 0
    assert triangle_box_gap_m([.01]*3, triangles+[.01999, 0, 0]) < 0


def test_improvement_and_numeric_band_do_not_accumulate_deepening() -> None:
    state = SeparationState({'pair': -20e-6})
    assert state.check({'pair': -10e-6}) == []
    assert state.check({'pair': -10.5e-6}) == []
    assert state.best_gaps['pair'] == -10e-6
    assert state.check({'pair': -11.1e-6})[0]['reason'] == 'release_contact_deepened'
    assert state.best_gaps['pair'] == -10e-6


def test_clear_latches_permanently_and_initial_clear_pair_never_gets_permission() -> None:
    state = SeparationState({('left', 'finger'): -1e-4, ('right', 'finger'): 1e-3})
    assert state.active_pairs == frozenset({('left', 'finger')})
    assert state.check({('left', 'finger'): 1e-9, ('right', 'finger'): 1e-3}) == []
    assert state.all_clear and state.finish() == []
    rejected = state.check({('left', 'finger'): 0., ('right', 'finger'): -1e-9})
    assert {row['reason'] for row in rejected} == {'release_contact_reentry', 'release_contact_new_collision'}


def test_clone_and_failed_check_do_not_consume_measured_latches() -> None:
    state = SeparationState({'one': -1e-4, 'two': 1e-3})
    candidate = state.clone()
    assert candidate.check({'one': 1e-4, 'two': 1e-3}) == []
    assert candidate.all_clear and not state.all_clear
    assert state.finish()[0]['reason'] == 'release_contact_not_separated'
    assert state.check({'one': 1e-4, 'two': -1e-3})
    assert state.active_pairs == frozenset({'one'}) and state.best_gaps['one'] == -1e-4


def test_pair_identity_and_input_shapes_are_explicit() -> None:
    with pytest.raises(ValueError, match='physical pair'):
        SeparationState({'one': 1.}).check({'other': 1.})
    with pytest.raises(ValueError, match='triangles'):
        triangle_box_gap_m([1, 1, 1], np.zeros((0, 3, 3)))
    with pytest.raises(ValueError, match='finite'):
        SeparationState({'one': float('nan')})


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
