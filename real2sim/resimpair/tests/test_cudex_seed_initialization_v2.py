import numpy as np
import pytest
from resimpair_grasp.cudex_seed_initialization_v2 import pool_action, SeedInitializationError


def test_only_actual_single_environment_empty_pool_triggers():
    assert pool_action(np.empty((0, 3)), [], [], 1) == 'repair_measured_empty_pool'
    assert pool_action(np.empty((0, 3)), [0], [0], 1) == 'repair_measured_empty_pool'
    with pytest.raises(SeedInitializationError): pool_action(np.empty((0, 3)), [2], [2], 1)
    with pytest.raises(SeedInitializationError): pool_action(np.empty((0, 3)), [], [], 2)


def test_valid_pool_and_numpy_random_state_remain_exact():
    positions = np.arange(15, dtype=float).reshape(5, 3); before = positions.copy()
    np.random.seed(128); expected = np.random.random(5)
    np.random.seed(128)
    assert pool_action(positions, [0], [5], 1) == 'preserve_original_nonempty_pool'
    assert np.array_equal(positions, before)
    assert np.array_equal(np.random.random(5), expected)


def test_corruption_is_not_empty_pool_recovery():
    for values in [np.full((5, 3), np.nan), np.ones((5, 2))]:
        with pytest.raises(SeedInitializationError): pool_action(values, [0], [5], 1)
    with pytest.raises(SeedInitializationError): pool_action(np.ones((5, 3)), [0], [4], 1)
