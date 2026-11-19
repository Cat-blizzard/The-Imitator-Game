"""Scene-relative declarations for sampled mechanisms; old tolerances retained.

The interpreter and historical contracts remain unchanged. New task ordering is
an explicit extra constraint, rather than treating reordered scripts as new data.
"""
from copy import deepcopy
from .core import require
from . import planning_protocols as p
from .event_protocol import validate


def protocol(scene, task):
    if scene.get('contact_extension'):
        from .contact_extensions import protocol as extension
        return extension(scene, task)
    from .novel_protocols import protocol as baseline
    plan = baseline(scene, task)
    if not scene.get('batch_generation'):
        return plan
    kind = task['scene_kind']
    params = task['planning_parameters']
    objects = {o['id']: o for o in scene['objects']}
    if kind in {'hook_retrieval', 'granule_sweeping'}:
        selected = params['selected']
        groups = [[f'cargo_{k}'] if kind == 'hook_retrieval' else [f'grain_{k}_{i}' for i in range(3)] for k in selected]
        previous_goals = []
        for k, group in zip(selected, groups):
            marker = objects[f'cargo_destination_{k}' if kind == 'hook_retrieval' else f'collector_{k}']
            cx = marker['position'][0]
            lo, hi = (cx-.07, cx+.065) if kind == 'hook_retrieval' else (cx-.10, cx+.09)
            for oid in group:
                position = p.expr('position', p.signal(oid+'.pose'))
                inside = p.expr('all', p.expr('gt', p.component(position, 0), p.const(lo)),
                    p.expr('lt', p.component(position, 0), p.const(hi)),
                    p.expr('lt', p.expr('abs', p.expr('sub', p.component(position, 1), p.const(objects[oid]['position'][1]))), p.const(.09)),
                    p.expr('lt', p.component(position, 2), p.const(.08)))
                for event in plan['events']:
                    if event['id'] == oid+'.goal': event['when'] = deepcopy(inside)
                    if event['id'] == oid+'.contact':
                        event['strictly_after'] += previous_goals
                        plan['precedence'] += [[before, event['id']] for before in previous_goals]
                        for before in previous_goals:
                            # Searching after a predecessor is not proof that
                            # contact did not already occur and persist earlier.
                            p.invariant(plan, oid+'.not_early_contact_before_'+before,
                                'no_contact', [oid], p.expr('not', event['when']),
                                until=before, include_end=True)
                            p.invariant(plan, oid+'.not_early_motion_before_'+before,
                                'unchanged_pose', [oid], p.expr('lt', p.expr('norm',
                                    p.expr('sub', position, p.expr('initial', position))), p.const(.008)),
                                until=before, position_tolerance_m=.008)
                goal = next(g for g in plan['goals'] if g['id'] == oid+'.goal')
                goal['predicate'] = inside
                goal['parameters']['region_x_m'] = [lo, hi]
            previous_goals = [oid+'.goal' for oid in group]
    if scene['joints']:
        previous = None
        for index in params.get('selected', [0]):
            oid = scene['joints'][index]['id']
            if previous:
                p.invariant(plan, oid+'.not_early_joint_motion', 'joint_range', [oid],
                    p.expr('lt', p.expr('abs', p.signal(oid+'.q')), p.const(.01)), until=previous)
                for arm in range(2):
                    p.invariant(plan, oid+f'.arm{arm}.not_early_contact', 'no_contact', [oid],
                        p.expr('le', p.signal(f'arm{arm}.{oid}.contact'), p.const(1e-6)), until=previous)
            previous = oid + ('.rebound' if kind == 'spring_console' else
                              '.return' if params.get('return_to_origin') else '.actuated')
    # Explicit displacement remains required even for broad legacy target bands.
    # This adds a no-op guard without relaxing the measured joint targets.
    for g in plan['goals']:
        if g['relation'] == 'joint_range' and abs(g['parameters'].get('target_joint_position', 0)) > .01:
            oid = g['entities'][0]
            g['predicate'] = p.expr('all', g['predicate'],
                p.expr('gt', p.expr('abs', p.expr('sub', p.signal(oid+'.q'),
                    p.expr('initial', p.signal(oid+'.q')))), p.const(.02)))
    validate(plan)
    return plan
