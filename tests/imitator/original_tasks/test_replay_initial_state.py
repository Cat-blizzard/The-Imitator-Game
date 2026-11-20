"""Initial replay state writes must not perturb an already matching reset."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from imitator.planning._tig_sync.trajectory_execution import initialize_replay_state


class RawEnvironment:
    """Expose state reads and an intentionally non-idempotent physical setter."""

    def __init__(self, state):
        self.state = state
        self.state_writes = []
        self.fixed_root_z = np.float32(0.)

    def get_state_dict(self):
        return self.state

    def set_state_dict(self, state):
        self.state_writes.append(state)
        # A pose setter may round a fixed root even when values already match.
        self.fixed_root_z = np.nextafter(self.fixed_root_z, np.float32(1.))

    def step(self, action):
        raise AssertionError("Initial-state selection must not advance physics")


def assert_tree_equal(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_tree_equal(actual[key], expected[key])
    elif isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    else:
        np.testing.assert_array_equal(actual, expected)


def test_matching_nested_torch_and_numpy_state_never_calls_physical_setter():
    current = {
        "actors": {"payload": torch.arange(13, dtype=torch.float32)[None, :]},
        "articulations": {"robot": np.arange(31, dtype=np.float32)[None, :]},
    }
    recorded = {
        "actors": {"payload": np.arange(13, dtype=np.float32)},
        "articulations": {"robot": torch.arange(31, dtype=torch.float32)},
    }
    before_current, before_recorded = deepcopy(current), deepcopy(recorded)
    raw = RawEnvironment(current)

    result = initialize_replay_state(raw, recorded)

    assert result == {"method": "seeded_reset", "state_write_count": 0}
    assert raw.state_writes == []
    assert raw.fixed_root_z == np.float32(0.)
    assert_tree_equal(current, before_current)
    assert_tree_equal(recorded, before_recorded)


@pytest.mark.parametrize("current_shape,recorded_shape", [
    ((4,), (4,)),
    ((1, 4), (4,)),
    ((4,), (1, 4)),
    ((1, 4), (1, 4)),
    ((1,), ()),
])
def test_only_single_environment_leading_axis_may_differ(current_shape, recorded_shape):
    count = int(np.prod(current_shape))
    current = np.arange(count, dtype=np.float32).reshape(current_shape)
    recorded = np.arange(count, dtype=np.float32).reshape(recorded_shape)
    raw = RawEnvironment({"state": current})

    assert initialize_replay_state(raw, {"state": recorded}) == {
        "method": "seeded_reset", "state_write_count": 0}
    assert raw.state_writes == []


def test_single_ulp_difference_restores_original_recorded_object_once():
    current = {"actors": {"payload": np.array([[1., 2.]], dtype=np.float32)}}
    recorded = {"actors": {"payload": np.array([1., 2.], dtype=np.float32)}}
    recorded["actors"]["payload"][1] = np.nextafter(np.float32(2.), np.float32(3.))
    before_current, before_recorded = deepcopy(current), deepcopy(recorded)
    raw = RawEnvironment(current)

    assert initialize_replay_state(raw, recorded) == {
        "method": "recorded_state", "state_write_count": 1}
    assert len(raw.state_writes) == 1
    assert raw.state_writes[0] is recorded
    assert_tree_equal(current, before_current)
    assert_tree_equal(recorded, before_recorded)


@pytest.mark.parametrize("current,recorded", [
    ({"actors": {"payload": np.zeros(4)}}, {"actors": {"other": np.zeros(4)}}),
    ({"actors": {"payload": np.zeros(4)}}, {"articulations": {"payload": np.zeros(4)}}),
    ({"actors": {"payload": np.zeros(4)}}, {"actors": {"payload": np.zeros(4)}, "extra": {}}),
    ({"state": np.zeros((2, 4))}, {"state": np.zeros(4)}),
    ({"state": np.zeros((1, 4))}, {"state": np.zeros((4, 1))}),
    ({"state": np.zeros((1, 2, 2))}, {"state": np.zeros(4)}),
    ({"state": {"payload": np.zeros(4)}}, {"state": np.zeros(4)}),
])
def test_key_or_non_batch_shape_mismatch_uses_one_original_state_restore(current, recorded):
    before_current, before_recorded = deepcopy(current), deepcopy(recorded)
    raw = RawEnvironment(current)

    assert initialize_replay_state(raw, recorded) == {
        "method": "recorded_state", "state_write_count": 1}
    assert len(raw.state_writes) == 1
    assert raw.state_writes[0] is recorded
    assert_tree_equal(current, before_current)
    assert_tree_equal(recorded, before_recorded)


def test_empty_matching_state_preserves_reset_without_any_write():
    raw = RawEnvironment({})
    assert initialize_replay_state(raw, {}) == {
        "method": "seeded_reset", "state_write_count": 0}
    assert raw.state_writes == []


def test_state_read_exception_propagates_without_fallback_write():
    raw = RawEnvironment({})
    error = RuntimeError("state read failed")

    def fail_read():
        raise error

    raw.get_state_dict = fail_read
    with pytest.raises(RuntimeError, match="state read failed") as caught:
        initialize_replay_state(raw, {})
    assert caught.value is error
    assert raw.state_writes == []


def test_state_restore_exception_propagates_after_single_original_object_write():
    raw = RawEnvironment({"state": np.array([0.], dtype=np.float32)})
    recorded = {"state": np.array([1.], dtype=np.float32)}
    error = RuntimeError("state restore failed")

    def fail_restore(state):
        raw.state_writes.append(state)
        raise error

    raw.set_state_dict = fail_restore
    with pytest.raises(RuntimeError, match="state restore failed") as caught:
        initialize_replay_state(raw, recorded)
    assert caught.value is error
    assert len(raw.state_writes) == 1
    assert raw.state_writes[0] is recorded


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
