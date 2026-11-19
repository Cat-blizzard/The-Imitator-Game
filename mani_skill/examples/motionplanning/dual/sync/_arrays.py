"""Shared array conversion; callers own shape, finite-value and copy policies."""
from __future__ import annotations

from typing import Any

import numpy as np


def as_float_array(value: Any) -> np.ndarray:
    """Convert array-like or tensor input to float64, retaining NumPy views.

    Tensor inputs use detach -> CPU -> NumPy without importing a tensor library.
    This does not promise ownership or finite values; callers validate those
    requirements or copy before mutating a returned view.
    """
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


if __name__ == "__main__":
    source = np.zeros((1, 3), dtype=np.float64)
    result = as_float_array(source)
    assert result.shape == (1, 3) and np.shares_memory(result, source)
    converted = as_float_array([0, 1, 2])
    assert converted.dtype == np.float64 and converted.shape == (3,)
    assert np.isnan(as_float_array([np.nan])).all()
    print("array conversion CPU check passed: batch (1,3), vector (3,)")
