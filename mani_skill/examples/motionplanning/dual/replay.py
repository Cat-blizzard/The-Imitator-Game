"""Replay original TIG actions from their recorded initial state and policy.

Use --input-dir and --output-dir; strict source checking remains the default.
The replay entry never reselects a source trajectory's task quality profile.
"""
from __future__ import annotations

from .validate_curobo_complete_replay import main


if __name__ == '__main__':
    raise SystemExit(main())
