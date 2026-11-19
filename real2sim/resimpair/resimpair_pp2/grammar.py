"""Semantic pick-and-place programs, scene task sets and identifiability checks.

A program refers to objects only through descriptors {category, color?}, so the
same program can be grounded in another scene. Pure Python; no simulator.
"""
from __future__ import annotations

from itertools import permutations
import json

SCHEMA = 'pp2_program_v1'
RELATIONS = ('in', 'on', 'next_to')
# Operator frame: the robot base sits at x=-0.6 facing +x, so its left is +y.
SIDES = {'left': (0., 1.), 'right': (0., -1.)}


def descriptor_text(d):
    return 'the ' + (d['color'] + ' ' if d.get('color') else '') + d['category']


def minimal_descriptors(items):
    """items: [{'id', 'category', 'color'}] -> {id: descriptor or None (ambiguous)}."""
    result = {}
    for item in items:
        same = [x for x in items if x['category'] == item['category']]
        if len(same) == 1:
            result[item['id']] = {'category': item['category']}
        elif sum(x['color'] == item['color'] for x in same) == 1:
            result[item['id']] = {'category': item['category'], 'color': item['color']}
        else:
            result[item['id']] = None
    return result


def resolve_descriptor(descriptor, items):
    matches = [x['id'] for x in items if x['category'] == descriptor['category']
               and ('color' not in descriptor or x['color'] == descriptor['color'])]
    if len(matches) != 1:
        raise ValueError('descriptor_not_unique:' + descriptor_text(descriptor))
    return matches[0]


def step(obj, relation, target, side=None):
    if relation not in RELATIONS or (relation == 'next_to') != (side is not None):
        raise ValueError('invalid_relation_or_side')
    if side is not None and side not in SIDES:
        raise ValueError('unknown_side')
    value = {'object': obj, 'relation': relation, 'target': target}
    if side is not None:
        value['side'] = side
    return value


def program(steps, ordered, family):
    objects = [json.dumps(s['object'], sort_keys=True) for s in steps]
    if not steps or len(set(objects)) != len(objects):
        raise ValueError('one_step_per_object_required')
    return {'schema': SCHEMA, 'family': family, 'ordered': bool(ordered) and len(steps) > 1, 'steps': steps}


def step_key(s):
    return json.dumps([s['object'], s['relation'], s['target'], s.get('side')], sort_keys=True)


def canonical(p):
    keys = [step_key(s) for s in p['steps']]
    return json.dumps({'ordered': p['ordered'], 'steps': keys if p['ordered'] else sorted(keys)})


def demonstrations(p):
    """Every step sequence that would be a valid demonstration of the program."""
    keys = [step_key(s) for s in p['steps']]
    return {tuple(keys)} if p['ordered'] else set(permutations(keys))


def confusable(p, q):
    return bool(demonstrations(p) & demonstrations(q))


def instruction(p):
    def phrase(s):
        if s['relation'] == 'next_to':
            return f"put {descriptor_text(s['object'])} to the {s['side']} of {descriptor_text(s['target'])}"
        return f"put {descriptor_text(s['object'])} {s['relation']} {descriptor_text(s['target'])}"
    parts = [phrase(s) for s in p['steps']]
    if len(parts) == 1:
        return parts[0][0].upper() + parts[0][1:] + '.'
    if p['ordered']:
        text = 'First ' + parts[0] + ', then ' + ', then '.join(parts[1:]) + '.'
    else:
        text = parts[0][0].upper() + parts[0][1:] + ' and ' + ' and '.join(parts[1:]) + ' (in any order).'
    return text


def difference(p, q):
    """Classify how two programs differ, for contrast-pair accounting."""
    if canonical(p) == canonical(q):
        return 'identical'
    if len(p['steps']) == len(q['steps']) == 1:
        a, b = p['steps'][0], q['steps'][0]
        same_object = a['object'] == b['object']
        same_destination = (a['relation'], a['target'], a.get('side')) == (b['relation'], b['target'], b.get('side'))
        if same_destination and not same_object:
            return 'object'
        if same_object and not same_destination:
            return 'destination'
    if p['ordered'] and q['ordered'] and sorted(map(step_key, p['steps'])) == sorted(map(step_key, q['steps'])):
        return 'order'
    return 'other'


def _family(s):
    return 'single_next_to' if s['relation'] == 'next_to' else 'single_' + s['relation']


def select_task_set(objects, receptacles, descriptors, k, rng, *, fits, feasible, geometric_confusable=None,
                    max_next_to=None, multi_quota=0, statics=()):
    """Choose at least k programs over one scene satisfying the v2 contrast and coverage rules.

    objects / receptacles: id lists; receptacles map to their relation via
    `receptacles[rid]`. fits(rid, [object ids]) -> bool checks shared capacity.
    feasible(program) -> bool runs cheap geometry checks of a compiled program.
    """
    d = descriptors
    objs = list(objects); recs = list(receptacles)
    rng.shuffle(objs); rng.shuffle(recs)
    if len(objs) < 2 or len(recs) < 2 or any(d[x] is None for x in objs + recs):
        raise ValueError('two_objects_two_receptacles_and_unique_descriptors_required')
    single = lambda o, r: program([step(d[o], receptacles[r], d[r])], False, 'single_' + receptacles[r])
    chosen, notes = [], []

    def add(p, note):
        if p is None or any(confusable(p, q) for q in chosen) or not feasible(p):
            return False
        if max_next_to is not None and p['family'] == 'single_next_to' and \
                sum(q['family'] == 'single_next_to' for q in chosen) >= max_next_to:
            return False
        # Symbolically different programs can still share one end layout (e.g. "right of the jar" and
        # "left of the block" when the two landmarks flank the same spot).
        if geometric_confusable and any(geometric_confusable(p, q) for q in chosen):
            return False
        chosen.append(p); notes.append(note); return True

    # Anchor (a, b, r1, r2): a and b both feasible in r1 (object contrast), a also in r2 (destination contrast).
    # Prefer an r1 that holds a and b together, so an order contrast exists too.
    ok = {(x, r): feasible(single(x, r)) for x in objs for r in recs}
    anchors = [(x, y, r, s) for r in recs for s in recs if s != r for x in objs for y in objs
               if x != y and ok[x, r] and ok[y, r] and ok[x, s]]
    if not anchors:
        raise ValueError('task_set_rules:no_contrast_anchor')
    joint = [t for t in anchors if fits(t[2], [t[0], t[1]])]
    a, b, r1, r2 = (joint or anchors)[int(rng.integers(len(joint or anchors)))]
    objs = [a, b] + [x for x in objs if x not in (a, b)]
    recs = [r1, r2] + [r for r in recs if r not in (r1, r2)]
    add(single(a, r1), 'anchor')
    add(single(b, r1), 'object_contrast')
    add(single(a, r2), 'destination_contrast')
    for r in (r1, r2):
        if fits(r, [a, b]):
            first = program([step(d[a], receptacles[r], d[r]), step(d[b], receptacles[r], d[r])], True, 'multi_ordered')
            second = program([step(d[b], receptacles[r], d[r]), step(d[a], receptacles[r], d[r])], True, 'multi_ordered')
            if feasible(first) and feasible(second) and add(first, 'order_pair_a'):
                add(second, 'order_contrast')
                break
    # Spatial-relation tasks get their own slots (they used to come only from the fill, which a successful
    # anchor plus the multi quota leaves no room for); a side contrast on one landmark comes first.
    lands = lambda o: recs + [x for x in objs if x != o]
    nexts = [(o, l) for o in objs for l in lands(o)]; rng.shuffle(nexts)
    next_to = lambda o, l, side: program([step(d[o], 'next_to', d[l], side)], False, 'single_next_to')
    for o, l in nexts:
        if sum(q['family'] == 'single_next_to' for q in chosen) >= (max_next_to or 2):
            break
        sides = list(SIDES); rng.shuffle(sides)
        if add(next_to(o, l, sides[0]), 'next_to'):
            add(next_to(o, l, sides[1]), 'side_contrast')
    # Multi-object tasks (v3): an order-contrast pair across two receptacles, a mixed two-object task and a
    # three-object task. Landmarks of next_to steps inside a multi-object task never move in that task.
    if multi_quota:
        rel = lambda o, r: step(d[o], receptacles[r], d[r])
        pairs = [(x, y) for x in objs for y in objs if x != y]
        rng.shuffle(pairs)
        for x, y in pairs:
            if sum(len(q['steps']) > 1 for q in chosen) >= multi_quota:
                break
            first = program([rel(x, r1), rel(y, r2)], True, 'multi_ordered')
            second = program([rel(y, r2), rel(x, r1)], True, 'multi_ordered')
            if feasible(first) and feasible(second) and add(first, 'cross_order_a'):
                add(second, 'cross_order_contrast'); break
        landmarks = list(recs) + list(statics)
        room = lambda: sum(len(q['steps']) > 1 for q in chosen) < multi_quota
        other = {'left': 'right', 'right': 'left'}
        # A mixed task, then its minimal contrast: the same steps with the next_to side flipped.
        mixed = lambda x, r, y, l, side: program([rel(x, r), step(d[y], 'next_to', d[l], side)], False, 'multi_mixed')
        found = None
        for x, y in pairs:
            if found or not room():
                break
            for r in recs:
                lm = [l for l in landmarks if l != r] + [o for o in objs if o not in (x, y)]
                rng.shuffle(lm)
                found = next(((x, r, y, l, side) for l in lm[:3] for side in SIDES if add(mixed(x, r, y, l, side), 'mixed')), None)
                if found:
                    break
        if found and room():
            x, r, y, l, side = found
            add(mixed(x, r, y, l, other[side]), 'mixed_side_contrast')
        # A three-object task, then its minimal contrast: the two receptacle steps swap destinations.
        triple = lambda x, a, y, b, z, l, side: program([rel(x, a), rel(y, b), step(d[z], 'next_to', d[l], side)],
                                                        False, 'multi_triple')
        triples = [(x, y, z) for x in objs for y in objs for z in objs if len({x, y, z}) == 3]
        rng.shuffle(triples); found = None
        for x, y, z in triples[:40]:
            if found or not room():
                break
            lm = [l for l in landmarks if l not in (x, y, z)]
            rng.shuffle(lm)
            found = next(((x, y, z, l, side) for l in lm[:2] for side in SIDES
                          if add(triple(x, r1, y, r2, z, l, side), 'triple')), None)
        if found and room():
            x, y, z, l, side = found
            add(triple(x, r2, y, r1, z, l, side), 'triple_assignment_contrast')
    # Fill: coverage of unused objects/receptacles first, then family variety.
    pool = []
    rest = objs[2:]
    for o in objs:
        for r in recs:
            pool.append((single(o, r), 'single'))
        for landmark in recs + [x for x in objs if x != o]:
            for side in SIDES:
                pool.append((program([step(d[o], 'next_to', d[landmark], side)], False, 'single_next_to'), 'next_to'))
    for o1 in objs:
        for o2 in objs:
            if o1 < o2:
                for r in recs:
                    if fits(r, [o1, o2]):
                        pool.append((program([step(d[o1], receptacles[r], d[r]), step(d[o2], receptacles[r], d[r])],
                                             False, 'multi_same'), 'multi_same'))
                for x, y in ((o1, o2), (o2, o1)):
                    pool.append((program([step(d[x], receptacles[r1], d[r1]), step(d[y], receptacles[r2], d[r2])],
                                         False, 'multi_distribute'), 'multi_distribute'))
    order = list(range(len(pool))); rng.shuffle(order)

    def covered():
        used = {json.dumps(s['object'], sort_keys=True) for p in chosen for s in p['steps']}
        used |= {json.dumps(s['target'], sort_keys=True) for p in chosen for s in p['steps']}
        return used

    def gain(item):
        p = item[0]; used = covered()
        new = {json.dumps(s['object'], sort_keys=True) for s in p['steps']} | {json.dumps(s['target'], sort_keys=True) for s in p['steps']}
        families = {q['family'] for q in chosen}
        return (len(new - used), p['family'] not in families)

    while len(chosen) < k:
        candidates = sorted((pool[i] for i in order), key=gain, reverse=True)
        if not any(add(p, 'fill_' + kind) for p, kind in candidates):
            break
    names = [descriptor_text(d[x]) for x in objs + recs]
    report = {'roles': {'a': a, 'b': b, 'r1': r1, 'r2': r2, 'rest': rest}, 'notes': notes,
              'coverage_complete': all(json.dumps(d[x], sort_keys=True) in covered() for x in objs + recs),
              'pairwise_differences': [[difference(p, q) for q in chosen] for p in chosen],
              'descriptors': names}
    return chosen, report


def validate_task_set(programs, k_min=4):
    """Spec rules 1, 4 and the demonstration-disjointness rule."""
    problems = []
    if len(programs) < k_min:
        problems.append('too_few_tasks')
    for i, p in enumerate(programs):
        for q in programs[i + 1:]:
            if confusable(p, q):
                problems.append('confusable:' + canonical(p) + '|' + canonical(q))
    kinds = {difference(p, q) for i, p in enumerate(programs) for q in programs[i + 1:]}
    for needed in ('object', 'destination'):
        if needed not in kinds:
            problems.append('missing_contrast:' + needed)
    return problems
