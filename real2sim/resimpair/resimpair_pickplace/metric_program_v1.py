"""Bounded 1..8-object programs and operation-local low-drop opening.

The original rolling state machine, collision guards and physical goals are
unchanged. Different release contracts may coexist in one scene program.
This is process-local and intended for one isolated physics worker.
"""
from contextlib import contextmanager

from .above_rim_release_v1 import policy

VERSION = 'metric_one_to_eight_object_program_v1'


def ready_operations(task, completed):
    operations = task['operations']; ids = [o['id'] for o in operations]
    if not 1 <= len(ids) <= 8 or len(set(ids)) != len(ids):
        raise ValueError('one_to_eight_unique_operations_required')
    if len({o['object'] for o in operations}) != len(ids):
        raise ValueError('regrasp_not_implemented')
    if not set(completed) <= set(ids):
        raise ValueError('unknown_completed_operation')
    edges = task.get('required_operation_precedence', [])
    if any(len(e) != 2 or e[0] not in ids or e[1] not in ids or e[0] == e[1] for e in edges):
        raise ValueError('invalid_required_precedence')
    remaining = [o for o in operations if o['id'] not in completed]
    ready = [o for o in remaining if all(a in completed for a, b in edges if b == o['id'])]
    if remaining and not ready:
        raise ValueError('cyclic_or_unsatisfied_precedence')
    return ready


@contextmanager
def install():
    from . import rolling_pickplace_v3 as rolling, stability_motion as execution
    from .full_open_drop_v2 import install as full_open
    original_ready, original_select, original_solve = rolling.ready_operations, rolling.select_operation, execution.solve

    def select(rec, task, operation, directory, original_caches):
        one = rolling.operation_task(task, operation)
        if policy(operation) is None:
            return original_select(rec, task, operation, directory, original_caches)
        with full_open(one):
            return original_select(rec, task, operation, directory, original_caches)

    def solve(rec, one, *args, **kwargs):
        if len(one['operations']) != 1:
            raise ValueError('operation_local_release_requires_rolling_single_operation')
        if policy(one['operations'][0]) is None:
            return original_solve(rec, one, *args, **kwargs)
        with full_open(one):
            return original_solve(rec, one, *args, **kwargs)

    rolling.ready_operations, rolling.select_operation, execution.solve = ready_operations, select, solve
    try:
        yield
    finally:
        rolling.ready_operations, rolling.select_operation, execution.solve = original_ready, original_select, original_solve
