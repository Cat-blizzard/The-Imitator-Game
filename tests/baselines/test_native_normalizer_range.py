"""Opt-in bounds expansion must preserve native values and old checkpoints."""
from copy import deepcopy

import pytest
import torch

from imitator.dataset.normalizer import ActionNormalizer


def normalizer(min_range=0.0):
    result = ActionNormalizer(min_range=min_range)
    bounds = {"min": [2., -.0001, -3.], "max": [2., .0001, 3.],
              "q01": [2., -.00008, -2.], "q99": [2., .00008, 2.]}
    result.add_dataset_stats(0, "pill", {"state": deepcopy(bounds), "action": deepcopy(bounds)},
                             "state", "action")
    return result


@pytest.mark.parametrize("method", ["bounds", "bounds_q99"])
def test_default_matches_original_arithmetic_bitwise(method):
    old = normalizer()
    values = torch.tensor([[2., -.00003, .31], [2.1, .00004, -.72]])
    stats = old.dataset_stats[0]["action"]
    low, high = ("min", "max") if method == "bounds" else ("q01", "q99")
    lower, upper = torch.tensor(stats[low]), torch.tensor(stats[high])
    denom = upper - lower
    denom[denom == 0] = 1.0
    expected = torch.clamp(2.0 * (values - lower) / denom - 1.0, -1.0, 1.0)
    for normalize in (old.normalize_state, old.normalize_action):
        assert torch.equal(normalize(values, 0, method), expected)
    for denormalize in (old.denormalize_state, old.denormalize_action):
        assert torch.equal(denormalize(expected, 0, method),
                           0.5 * (expected + 1.0) * (upper - lower) + lower)


@pytest.mark.parametrize("method", ["bounds", "bounds_q99"])
def test_constant_tiny_and_large_ranges_round_trip_without_mutating_statistics(method):
    expanded = normalizer(.01)
    before = deepcopy(expanded.dataset_stats)
    values = torch.tensor([[2., -.00003, .31], [2.001, .00004, -.72]])
    for normalize, denormalize in ((expanded.normalize_state, expanded.denormalize_state),
                                  (expanded.normalize_action, expanded.denormalize_action)):
        normalized = normalize(values, 0, method)
        torch.testing.assert_close(denormalize(normalized, 0, method), values, rtol=0, atol=2e-7)
        # Subtracting expanded float32 bounds can introduce sub-ULP native error.
        torch.testing.assert_close(normalized[0, 0], torch.tensor(0.), rtol=0, atol=2e-5)
        torch.testing.assert_close(normalized[:, 1], values[:, 1] / .005, rtol=0, atol=1e-5)
        assert torch.equal(normalized[:, 2], normalizer().normalize_action(values, 0, method)[:, 2])
    assert expanded.dataset_stats == before


@pytest.mark.parametrize("minimum", [-.1, float("nan"), float("inf")])
def test_invalid_range_is_rejected(minimum):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        ActionNormalizer(min_range=minimum)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
