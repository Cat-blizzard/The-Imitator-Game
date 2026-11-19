"""Fully open only for explicit above-rim drop, without changing drive gains.

The full aperture is checked in geometric preview and the live opening sweep.
The physical PD controller moves the joints; no actor or joint is teleported.
This context is for one isolated worker process, not a shared thread pool.
"""
from contextlib import contextmanager
from .above_rim_release_v1 import policy

VERSION = 'checked_full_aperture_drop_v2'


@contextmanager
def install(task):
    from . import whole_path_motion_v1 as preview, stability_motion as execution
    if not task['operations'] or any(policy(op) is None for op in task['operations']):
        raise ValueError('full_open_policy_requires_explicit_container_drop_for_every_operation')
    old_preview, old_width, old_command = preview.opening_width, execution.opening_width, execution.aperture_command
    def width(current):
        old_width(current)  # Retain width validation, including finite bounds.
        return .08
    def command(requested_width, lower, upper):
        return old_command(.08, lower, upper)
    preview.opening_width = execution.opening_width = width
    execution.aperture_command = command
    try: yield
    finally:
        preview.opening_width, execution.opening_width = old_preview, old_width
        execution.aperture_command = old_command
