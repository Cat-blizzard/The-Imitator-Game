"""Read-only measurements of normalized, requested, and executed actions.

All vectors have shape ``(D,)`` in the caller's existing action order. The
caller supplies coordinate names and units: this module does not interpret or
convert them. Clipping is defined by exact comparisons of requested values with
the supplied execution bounds, without a tolerance or an execution decision.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


class ActionLimitDiagnostics:
    """Keep snapshots for later inspection without changing policy execution."""

    def __init__(self, coordinates: list[dict[str, Any]], low: np.ndarray,
                 high: np.ndarray) -> None:
        self._coordinates = deepcopy(coordinates)
        self._low = np.array(low, copy=True)
        self._high = np.array(high, copy=True)
        if self._low.ndim != 1 or self._high.shape != self._low.shape:
            raise ValueError("Action bounds must both have shape (D,)")
        if len(self._coordinates) != self._low.size:
            raise ValueError("Coordinate metadata must have one entry per action dimension")
        self._normalized: list[np.ndarray] = []
        self._requested: list[np.ndarray] = []
        self._executed: list[np.ndarray] = []

    def record(self, normalized: np.ndarray, requested: np.ndarray,
               executed: np.ndarray) -> None:
        """Copy the three existing action stages, each with shape ``(D,)``."""
        values = tuple(np.array(value, copy=True)
                       for value in (normalized, requested, executed))
        if any(value.shape != self._low.shape for value in values):
            raise ValueError("Action snapshots must each have shape (D,)")
        self._normalized.append(values[0])
        self._requested.append(values[1])
        self._executed.append(values[2])

    def arrays(self) -> dict[str, np.ndarray]:
        """Return independent arrays suitable for ``np.savez`` (no pickle)."""
        def stack(values: list[np.ndarray]) -> np.ndarray:
            return (np.stack(values, axis=0) if values else
                    np.empty((0, self._low.size), dtype=self._low.dtype))

        return {
            "normalized": stack(self._normalized),
            "requested": stack(self._requested),
            "executed": stack(self._executed),
            "low": self._low.copy(),
            "high": self._high.copy(),
        }

    def summary(self) -> dict[str, Any]:
        """Report per-coordinate excess; conditional statistics exclude legal values.

        Bounds comparisons use the recorded dtype. Excess arithmetic uses
        float64 to retain small float32 excursions. Conditional statistics are
        ``None`` when that coordinate never exceeded a bound; relative excess
        is also ``None`` for zero-width or unbounded intervals.
        """
        requested = self.arrays()["requested"]
        below = requested < self._low
        above = requested > self._high
        clipped = below | above
        by_coordinate = []
        for index, metadata in enumerate(self._coordinates):
            values = requested[:, index]
            values64 = values.astype(np.float64)
            low = float(self._low[index])
            high = float(self._high[index])
            excess = np.maximum(np.maximum(low - values64, values64 - high), 0.0)
            conditional = excess[clipped[:, index]]
            span = high - low
            stats: dict[str, Any] = {
                **deepcopy(metadata), "index": index,
                "low": low, "high": high,
                "lower_clip_count": int(below[:, index].sum()),
                "upper_clip_count": int(above[:, index].sum()),
                "total_clip_count": int(clipped[:, index].sum()),
                "requested_min": float(values.min()) if values.size else None,
                "requested_max": float(values.max()) if values.size else None,
                "max_abs_excess_on_clipped": float(conditional.max()) if conditional.size else None,
                "p50_abs_excess_on_clipped": float(np.percentile(conditional, 50)) if conditional.size else None,
                "p95_abs_excess_on_clipped": float(np.percentile(conditional, 95)) if conditional.size else None,
                "mean_abs_excess_on_clipped": float(conditional.mean()) if conditional.size else None,
                "bound_span_fraction_max": (float(conditional.max() / span)
                                            if conditional.size and span > 0 and np.isfinite(span)
                                            else None),
            }
            by_coordinate.append(stats)
        return {
            "total_steps": int(requested.shape[0]),
            "any_clip_steps": int(clipped.any(axis=1).sum()),
            "total_clipped_coordinates": int(clipped.sum()),
            "by_coordinate": by_coordinate,
        }


if __name__ == "__main__":
    low = np.array([-2.0, -1.0], dtype=np.float32)
    high = np.array([2.0, 1.0], dtype=np.float32)
    diagnostics = ActionLimitDiagnostics(
        [{"joint": "joint1", "unit": "rad"},
         {"joint": "gripper", "unit": "normalized"}], low, high)
    requested = np.array([2.25, np.nextafter(np.float32(-1), np.float32(-2))],
                         dtype=np.float32)
    normalized = np.array([1.1, -1.01], dtype=np.float32)
    executed = np.clip(requested, low, high)
    diagnostics.record(normalized, requested, executed)
    summary = diagnostics.summary()
    assert summary["total_steps"] == 1 and summary["any_clip_steps"] == 1
    assert summary["by_coordinate"][0]["max_abs_excess_on_clipped"] == 0.25
    assert summary["by_coordinate"][1]["lower_clip_count"] == 1
    assert diagnostics.arrays()["requested"].shape == (1, 2)
    assert requested[0] == 2.25 and executed[0] == 2.0
    print({"shape": list(diagnostics.arrays()["requested"].shape), "summary": summary})
