"""Select a direct held transport after measured source departure.

Release targets are finite world-frame XYZ [2,3] in metres. Only an
intermediate stop is removed: original release targets, native timing, motion
limits, collision checks, grasp/lift and release/retreat/stability gates remain.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

TASKS = ('TwoRobotPickAppleBananaToBaskets-v1', 'TwoRobotPickPillToRegions-v1',
         'TwoRobotPickFruitsToPlate-v1')
DIRECT_RELEASE_RECIPE = 'direct-release-v2'


def direct_release_stages(task_id: str, route: dict[str, Any],
                          departure: dict[str, Any]) -> tuple[tuple[str, np.ndarray], ...]:
    """Return the original release targets after both measured departures.

    The caller has already checked real grasp, >=10cm lift and physical source
    departure. This selector proposes a route; the native planner, combined
    collision checker and physical held executor must still accept it.
    """
    if task_id not in TASKS or not isinstance(route, dict) or not isinstance(departure, dict):
        raise ValueError('A supported task, measured route and departure are required')
    released = departure.get('effective_measured_departure')
    if (not isinstance(released, (list, tuple)) or len(released) != 2
            or any(value is not True for value in released)):
        raise ValueError('Both measured source departures must be confirmed')
    release = np.asarray(route.get('release'), dtype=np.float64)
    if release.shape != (2, 3) or not np.isfinite(release).all():
        raise ValueError('Original release targets must be finite world XYZ [2,3]')
    return (('carry_lower', release.copy()),)


def self_test() -> dict[str, Any]:
    """Check target preservation and measured source-departure requirements."""
    release = np.array([[.1, .2, .3], [-.1, .4, .5]])
    route = {'release': release, 'above_baskets': release + [0., 0., .06]}
    departed = {'effective_measured_departure': [True, True]}
    for task in TASKS:
        result = direct_release_stages(task, route, departed)
        assert len(result) == 1 and result[0][0] == 'carry_lower'
        assert np.array_equal(result[0][1], release)
        result[0][1][0, 0] = 4.
        assert release[0, 0] == .1
    for task, candidate, state in (
        ('unknown', route, departed), (TASKS[0], route, {}),
        (TASKS[0], route, {'effective_measured_departure': [True, False]}),
        (TASKS[0], route, {'effective_measured_departure': [1, 1]}),
        (TASKS[0], {'release': [[0., 0., 0.]]}, departed),
        (TASKS[0], {'release': np.full((2, 3), np.nan)}, departed),
    ):
        try:
            direct_release_stages(task, candidate, state)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError('Invalid direct transport candidate accepted')
    return {'status': 'passed', 'recipe': DIRECT_RELEASE_RECIPE,
            'environment_steps': 0, 'release_targets_preserved': True}


if __name__ == '__main__':
    print(json.dumps(self_test(), allow_nan=False))
