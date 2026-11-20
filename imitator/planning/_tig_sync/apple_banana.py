"""Original apple/banana task using the shared synchronous executor."""
from __future__ import annotations

from .apple_banana_task import TASK_ID
from .runtime_policy import DEFAULT_POLICY
from .solve import solve as solve_task


def solve(env, seed=None, debug=False, vis=False, *, policy=DEFAULT_POLICY):
    """Reuse the original entry's caller-owned environment and recorder."""
    return solve_task(env, seed=seed, debug=debug, vis=vis, policy=policy, task_id=TASK_ID)
