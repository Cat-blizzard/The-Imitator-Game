"""Optional synchronous experts for the original TIG task scenes.

The default 1.0 runner remains serial. This module does not change evaluation
success or terminate learned-policy rollouts; its checks apply to experts.
"""
from __future__ import annotations

from typing import Any

from ..embodiments.native import make_native_env, native_contract


def solve(env: Any, seed: int | None = None, debug: bool = False, vis: bool = False,
          **kwargs: Any) -> tuple[Any, Any]:
    """Execute separate arm plans on one clock in the caller-owned environment."""
    from ._tig_sync.solve import solve as _solve
    from ..envs.original_task_profiles import get_task_profile
    raw = env.unwrapped
    if getattr(raw, "embodiment_profile", None) is None:
        raw.embodiment_profile = get_task_profile(raw.spec.id, raw.robot_uids[0])
    return _solve(env, seed=seed, debug=debug, vis=vis, task_id=raw.spec.id, **kwargs)


def main(argv: list[str] | None = None) -> None:
    import sys
    from mani_skill.examples.motionplanning.dual.two_robot_run import main as run, parse_args
    values = sys.argv[1:] if argv is None else argv
    if values and values[0] == "replay":
        from ._tig_sync.replay import main as replay_main
        replay_main(values[1:])
        return
    args = parse_args(["--execution-mode", "sync", *values])
    run(args)


def replay(path: str, episode_id: int = 0) -> dict[str, Any]:
    from ._tig_sync.replay import replay as _replay
    return _replay(path, episode_id)


if __name__ == "__main__":
    main()
