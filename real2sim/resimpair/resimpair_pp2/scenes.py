"""Sample pp2 scenes from the canonical catalog and compile their task sets.

Layout and task selection never look at physical outcomes. The retained
request compiler, cached-TCP workspace check and nominal endpoint IK gate run
unchanged; a scene whose task set cannot pass them is resampled and the
rejection is kept.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import time
from unittest import mock

import numpy as np

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write
from . import grammar

VERSION = 'pp2_scene_factory_v1'
CATALOG = ROOT / 'runs/pp2_catalog_20261101_v2/catalog.json'
TABLE_XY = np.array([[-.46, .46], [-.90, .90]])
OBJECT_REGION = np.array([[-.33, .10], [-.46, .46]])
NEXT_TO_GAP_M = .03  # bounding circles apart; the retained compiler requires > .025
NEXT_TO_MAX_PLANNED_GAP_M = .085  # acceptance-measured planned gap; leaves 1.5 cm of placement error below 10 cm
HOME_QPOS = np.array([0., -.4, 0., -2., 0., 1.8, np.pi / 4, .04, .04])  # retained env's reset pose
# Per-joint uniform range of the sampled start. The full range ([.25, .2, .25, .25, .25, .25, .4]) cost 107 vs 134
# passes of 160 tasks against the fixed pose (retained reachability seeds IK from it); half costs 128.
START_JITTER_RAD = np.array([.125, .1, .125, .125, .125, .125, .2])
MIN_VISIBLE_MOVE_M = .08  # a reference video must show a clearly visible change
PACK_GAP_M, PACK_MARGIN_M = .02, .012  # compiler rejects separation at exact equality
LOW_CONTRAST = .10  # object vs receptacle luminance difference below this: placement hard to see


def footprint(dimensions):
    """Bounding-circle radius used by the retained compiler's separation checks."""
    return float(np.linalg.norm(np.asarray(dimensions[:2]) / 2))


def cavity_radius(descriptor):
    c = descriptor['cavity']
    return float(min(c['flat_floor_radius_m'], c.get('minimum_wall_radius_m', c['flat_floor_radius_m'])))


def load_catalog(path=CATALOG):
    catalog = read(path)
    manip = [a for a in catalog['assets'] if a['role'] == 'manipulable' and a['grasps'] >= 4]
    recs = [a for a in catalog['assets'] if a['role'] == 'receptacle' and max(a['dimensions_m'][:2]) <= .30]
    for r in recs:
        r['cavity_radius_m'] = cavity_radius(read(resolve(r['descriptor'])))
    return catalog, manip, recs


def container_fit(objects, recs):
    """Objects that fit alone in at least one chosen receptacle, and whether some pair shares one."""
    fit = lambda o, r: footprint(o['dimensions_m']) + PACK_MARGIN_M <= r['cavity_radius_m']
    alone = [o for o in objects if any(fit(o, r) for r in recs)]
    # Cross-receptacle tasks need distinct objects that fit each of the two receptacles.
    cross = len(recs) >= 2 and any(fit(x, recs[0]) and fit(y, recs[1]) for x in objects for y in objects if x is not y)
    return len(alone), cross


def contrast_anchor(objects, recs, luminance_rule=True):
    """Layout-independent precondition of the task-set rules: two objects that each fit and stand out in one
    receptacle (object contrast), one of them also in a second receptacle (destination contrast).

    Without it every layout trial of the asset set fails `missing_contrast` (half the q6 scenes did)."""
    def ok(o, r):
        if footprint(o['dimensions_m']) + PACK_MARGIN_M > r['cavity_radius_m']:
            return False
        a, b = o.get('luminance'), r.get('luminance')
        return not luminance_rule or a is None or b is None or abs(a - b) >= LOW_CONTRAST
    for r1 in recs:
        good = [o for o in objects if ok(o, r1)]
        if len(good) >= 2 and any(ok(o, r2) for o in good for r2 in recs if r2 is not r1):
            return True
    return False


def choose_assets(manip, recs, partition, rng, n_objects, pair_probability=.6, usage=None, metric_preference=0.):
    recs = [r for r in recs if r['partition'] == partition]
    manip = [m for m in manip if m['partition'] == partition]
    # Prefer receptacles with authored metric size (GSO/YCB/ABO/HSSD/Poly Haven) over category-scaled ones.
    metric = [r for r in recs if not r['source_id'].startswith('objaverse')]
    if metric_preference and len({r['category'] for r in metric}) >= 3 and rng.random() < metric_preference:
        recs = metric
    rec_by_category = {}
    for r in recs:
        rec_by_category.setdefault(r['category'], []).append(r)
    used = usage if usage is not None else {}
    least = lambda opts: min(opts, key=lambda a: (used.get(a['source_id'], 0), rng.random()))
    # Categories weighted by sqrt(asset count): rare categories still appear, but their few assets are not
    # drawn in every scene (uniform category sampling reused a single basket 11 times in one batch).
    names = sorted(rec_by_category)
    weights = np.sqrt([len(rec_by_category[c]) for c in names]); weights = weights / weights.sum()
    picked = list(rng.choice(len(names), size=len(names), replace=False, p=weights))
    order = [least(rec_by_category[names[i]]) for i in picked] + [recs[i] for i in rng.permutation(len(recs))]
    chosen_recs = []
    for r in order:
        if r in chosen_recs:
            continue
        if all((r['category'], r['color']) != (x['category'], x['color']) for x in chosen_recs):
            chosen_recs.append(r)
        if len(chosen_recs) == 2:
            break
    objects = []
    by_category = {}
    for m in manip:
        by_category.setdefault(m['category'], []).append(m)
    if rng.random() < pair_probability:
        pairable = [c for c, rows in by_category.items() if len({r['color'] for r in rows}) >= 2]
        if pairable:
            category = pairable[rng.integers(len(pairable))]
            rows = list(by_category[category]); rng.shuffle(rows)
            for r in rows:
                if all(r['color'] != x['color'] for x in objects):
                    objects.append(r)
                if len(objects) == 2:
                    break
    # Uniform over categories, then over assets within a category, so frequent nouns do not dominate scenes.
    rows = []
    for category in rng.permutation(sorted(by_category)):
        rows.append(least(by_category[category]))
    for r in rows:
        if len(objects) == n_objects:
            break
        if r['source_id'] in {x['source_id'] for x in objects}:
            continue
        if any((r['category'], r['color']) == (x['category'], x['color']) for x in objects):
            continue
        if sum(x['category'] == r['category'] for x in objects) >= 2:
            continue
        objects.append(r)
    if len(chosen_recs) < 2 or len(objects) < n_objects:
        raise ValueError('not_enough_distinct_assets_in_partition')
    colors = {x['color'] for x in objects + chosen_recs}
    if any(x['category'] in colors for x in objects + chosen_recs):
        raise ValueError('category_name_equals_a_scene_color')  # e.g. "the orange on the orange plate"
    return objects, chosen_recs


def choose_distractors(manip, partition, used, rng, count):
    """Static non-target clutter whose category differs from every task item (keeps descriptors short)."""
    taken = {x['category'] for x in used}; picks = []
    rows = [m for m in manip if m['partition'] == partition and m['source_id'] not in {x['source_id'] for x in used}]
    for i in rng.permutation(len(rows)):
        if len(picks) == count:
            break
        if rows[i]['category'] not in taken and rows[i]['category'] not in {x['color'] for x in used}:
            picks.append(rows[i]); taken.add(rows[i]['category'])
    return picks


def sample_layout(objects, recs, rng, attempts=400, distractors=()):
    sides = [1, -1]; rng.shuffle(sides)
    placed = []  # (id, xy, radius)
    rec_rows = []
    for i, (r, sign) in enumerate(zip(recs, sides)):
        xy = np.array([rng.uniform(-.06, .08), sign * rng.uniform(.17, .30)])
        rec_rows.append({'id': f'receptacle_{i}', 'asset': r, 'xy': xy, 'yaw': float(rng.uniform(-math.pi, math.pi))})
        placed.append((f'receptacle_{i}', xy, footprint(r['dimensions_m'])))
    obj_rows = []
    for i, o in enumerate(objects):
        radius = footprint(o['dimensions_m'])
        for _ in range(attempts):
            xy = np.array([rng.uniform(*OBJECT_REGION[0]), rng.uniform(*OBJECT_REGION[1])])
            if all(np.linalg.norm(xy - p) > radius + q + .04 for _, p, q in placed):
                break
        else:
            raise ValueError('object_layout_not_placeable')
        obj_rows.append({'id': f'object_{i}', 'asset': o, 'xy': xy, 'yaw': float(rng.uniform(-math.pi, math.pi))})
        placed.append((f'object_{i}', xy, radius))
    static_rows = []
    for i, o in enumerate(distractors):
        radius = footprint(o['dimensions_m'])
        for _ in range(attempts):
            xy = np.array([rng.uniform(-.36, .14), rng.uniform(-.52, .52)])
            if all(np.linalg.norm(xy - p) > radius + q + .05 for _, p, q in placed):
                break
        else:
            continue
        static_rows.append({'id': f'clutter_{i}', 'asset': o, 'xy': xy, 'yaw': float(rng.uniform(-math.pi, math.pi))})
        placed.append((f'clutter_{i}', xy, radius))
    if distractors:
        return obj_rows, rec_rows, static_rows
    return obj_rows, rec_rows


class SceneGeometry:
    """Cheap per-program feasibility mirroring the retained compiler's rules."""

    def __init__(self, obj_rows, rec_rows, receptacle_descriptors, static_rows=()):
        self.objects = {r['id']: r for r in obj_rows}
        self.recs = {r['id']: r for r in rec_rows}
        self.static = {r['id']: r for r in static_rows}
        self.contrast_rule = False
        self.cavity = {rid: cavity_radius(receptacle_descriptors[rid]) for rid in self.recs}
        self.cavity_center = {}
        for rid, row in self.recs.items():
            c, s_ = np.cos(row['yaw']), np.sin(row['yaw'])
            local = np.asarray(receptacle_descriptors[rid]['cavity']['center_xy_m'])
            self.cavity_center[rid] = row['xy'] + np.array([[c, -s_], [s_, c]]) @ local
        self.items = [{'id': r['id'], 'category': r['asset']['category'], 'color': r['asset']['color']}
                      for r in list(obj_rows) + list(rec_rows) + list(static_rows)]

    def radius(self, oid):
        return footprint(self.objects[oid]['asset']['dimensions_m'])

    def fits(self, rid, oids):
        from resimpair_tasks.multitask_scene_factory_v1 import simple_slots
        return simple_slots([self.radius(o) for o in oids], self.cavity[rid], gap=PACK_GAP_M, margin=PACK_MARGIN_M)['passed']

    def row(self, rid):
        return self.objects.get(rid) or self.recs.get(rid) or self.static[rid]

    def landmark(self, lid):
        row = self.row(lid)
        return row['xy'], footprint(row['asset']['dimensions_m'])

    def next_to_xy(self, oid, lid, side):
        """Bounding circles NEXT_TO_GAP_M apart (the retained compiler's separation rule). Acceptance measures the
        gap between half longest sides, which for square items is up to 40% of their size larger, so placements
        next to big square landmarks landed just past the 10 cm limit (all 6 next_to failures of q6): such
        programs are infeasible here instead of failing after execution."""
        xy, radius = self.landmark(lid)
        distance = radius + self.radius(oid) + NEXT_TO_GAP_M
        half = lambda row: .5 * max(row['asset']['dimensions_m'][:2])
        if distance - half(self.objects[oid]) - half(self.row(lid)) > NEXT_TO_MAX_PLANNED_GAP_M:
            raise ValueError('next_to_gap_too_wide_for_acceptance')
        return xy + np.asarray(grammar.SIDES[side]) * distance

    def placements(self, p):
        """Program -> retained request placements, or raise ValueError."""
        from resimpair_tasks.multitask_scene_factory_v1 import simple_slots
        ids = {}
        for s in p['steps']:
            ids[grammar.step_key(s)] = (grammar.resolve_descriptor(s['object'], self.items),
                                        grammar.resolve_descriptor(s['target'], self.items))
        by_rec = {}
        for s in p['steps']:
            oid, tid = ids[grammar.step_key(s)]
            if s['relation'] != 'next_to':
                by_rec.setdefault(tid, []).append(oid)
        offsets = {}
        for rid, oids in by_rec.items():
            # Slots follow object identity, not step order, so order-contrast tasks share one end state.
            oids = sorted(oids)
            slots = simple_slots([self.radius(o) for o in oids], self.cavity[rid], gap=PACK_GAP_M, margin=PACK_MARGIN_M)
            if not slots['passed']:
                raise ValueError('receptacle_capacity')
            offsets.update(zip(oids, slots['offsets_local_xy_m']))
        result, moved = [], set()
        for s in p['steps']:
            oid, tid = ids[grammar.step_key(s)]
            if s['relation'] == 'next_to':
                if tid in moved:
                    raise ValueError('landmark_moved_before_use')
                if oid not in moved and self.next_to_holds(oid, tid, s['side']):
                    raise ValueError('next_to_already_satisfied_initially')
                xy = self.next_to_xy(oid, tid, s['side'])
                if np.linalg.norm(xy - self.objects[oid]['xy']) < MIN_VISIBLE_MOVE_M:
                    raise ValueError('next_to_goal_too_close_to_start')
                self.check_table_goal(oid, xy, exclude={oid})
                result.append({'object': oid, 'target': {'kind': 'table', 'xy': [float(v) for v in xy]}})
            else:
                result.append({'object': oid, 'target': {'kind': 'container', 'container': tid,
                                                         'offset_xy_m': [float(v) for v in offsets[oid]]}})
            moved.add(oid)
        return result

    def next_to_holds(self, oid, lid, side):
        """Same predicate as acceptance, on the initial layout."""
        from .accept import RULES
        xy, _ = self.landmark(lid); delta = self.objects[oid]['xy'] - xy; distance = float(np.linalg.norm(delta))
        half = lambda row: .5 * max(row['asset']['dimensions_m'][:2])
        landmark = self.row(lid)
        gap = distance - half(self.objects[oid]) - half(landmark)
        angle = float(np.degrees(np.arccos(np.clip(delta @ np.asarray(grammar.SIDES[side]) / max(distance, 1e-9), -1, 1))))
        return RULES['next_to_min_edge_gap_m'] <= gap <= RULES['next_to_max_edge_gap_m'] and angle <= RULES['next_to_max_angle_deg']

    def check_table_goal(self, oid, xy, exclude):
        radius = self.radius(oid)
        if np.any(xy - radius < TABLE_XY[:, 0]) or np.any(xy + radius > TABLE_XY[:, 1]):
            raise ValueError('next_to_goal_outside_table')
        if not (OBJECT_REGION[0, 0] - .04 <= xy[0] <= OBJECT_REGION[0, 1] + .06 and abs(xy[1]) <= .55):
            raise ValueError('next_to_goal_outside_reach_region')
        for rid, row in list(self.objects.items()) + list(self.recs.items()) + list(self.static.items()):
            if rid in exclude:
                continue
            if np.linalg.norm(xy - row['xy']) <= radius + footprint(row['asset']['dimensions_m']) + .025:
                raise ValueError('next_to_goal_overlap')

    def end_positions(self, p):
        positions = {oid: row['xy'].copy() for oid, row in self.objects.items()}
        for placement in self.placements(p):
            target = placement['target']
            if target['kind'] == 'table':
                positions[placement['object']] = np.asarray(target['xy'])
            else:
                row = self.recs[target['container']]; c, s_ = np.cos(row['yaw']), np.sin(row['yaw'])
                positions[placement['object']] = self.cavity_center[target['container']] + \
                    np.array([[c, -s_], [s_, c]]) @ np.asarray(target['offset_xy_m'])
        return positions

    def satisfied_by(self, q, positions):
        """Does the end layout `positions` satisfy every step of program q (planning-level predicate)?"""
        from .accept import RULES
        half = lambda row: .5 * max(row['asset']['dimensions_m'][:2])
        for s in q['steps']:
            oid = grammar.resolve_descriptor(s['object'], self.items); tid = grammar.resolve_descriptor(s['target'], self.items)
            if s['relation'] == 'next_to':
                anchor = positions.get(tid, self.row(tid)['xy'])
                delta = positions[oid] - anchor; distance = float(np.linalg.norm(delta))
                landmark = self.row(tid)
                gap = distance - half(self.objects[oid]) - half(landmark)
                angle = float(np.degrees(np.arccos(np.clip(delta @ np.asarray(grammar.SIDES[s['side']]) / max(distance, 1e-9), -1, 1))))
                if not (RULES['next_to_min_edge_gap_m'] <= gap <= RULES['next_to_max_edge_gap_m'] and angle <= RULES['next_to_max_angle_deg']):
                    return False
            elif tid not in self.recs or np.linalg.norm(positions[oid] - self.cavity_center[tid]) > self.cavity[tid] + RULES['cavity_margin_m']:
                return False
        return True

    def geometric_confusable(self, p, q):
        """Same moved objects and p's planned end layout also satisfies q (or vice versa)."""
        moved = lambda x: {grammar.resolve_descriptor(s['object'], self.items) for s in x['steps']}
        if moved(p) != moved(q):
            return False
        # Same steps in a different required order: one end state, but the order is visible in the video.
        if grammar.difference(p, q) == 'order':
            return False
        try:
            return self.satisfied_by(q, self.end_positions(p)) or self.satisfied_by(p, self.end_positions(q))
        except ValueError:
            return False

    def low_contrast(self, p, minimum=LOW_CONTRAST):
        """Object and receptacle textures too close in luminance: the placement would be hard to see."""
        for st in p['steps']:
            if st['relation'] == 'next_to':
                continue
            oid = grammar.resolve_descriptor(st['object'], self.items); rid = grammar.resolve_descriptor(st['target'], self.items)
            a, b = self.objects[oid]['asset'].get('luminance'), self.recs[rid]['asset'].get('luminance')
            if a is not None and b is not None and abs(a - b) < minimum:
                return True
        return False

    def feasible(self, p):
        if self.contrast_rule and self.low_contrast(p):
            return False
        try:
            self.placements(p); return True
        except ValueError:
            return False


def robot_start(seed):
    """Per-scene start joints of both arms around the retained reset pose (shared by every task of the scene, so
    the robot observation still cannot tell its tasks apart); applied by resimpair_pp2.worker."""
    rng = np.random.default_rng([int(seed) % 2**32, 7411])
    return [np.r_[HOME_QPOS[:7] + rng.uniform(-1, 1, 7) * START_JITTER_RAD, HOME_QPOS[7:]].round(4).tolist() for _ in range(2)]


def compile_programs(scene_id, obj_rows, rec_rows, programs, geometry, seed, catalog, static_rows=()):
    from resimpair_pickplace import requested_tasks
    spec = {'id': scene_id,
            'objects': [{'id': r['id'], 'source_id': r['asset']['source_id'], 'xy': r['xy'].tolist(), 'yaw': r['yaw']}
                        for r in obj_rows],
            'containers': [{'id': r['id'], 'source_id': r['asset']['source_id'], 'xy': r['xy'].tolist(), 'yaw': r['yaw']}
                           for r in rec_rows],
            'tasks': [{'id': f'task_{i}', 'placements': geometry.placements(p)} for i, p in enumerate(programs)]}
    entries = [{'asset': r['asset']['descriptor'], 'cache': r['asset']['cache']} for r in obj_rows]
    containers = [r['asset']['descriptor'] for r in rec_rows]
    # Distractors from the retained catalog are not used in pp2 scenes.
    with mock.patch.object(requested_tasks, 'add_distractors', return_value=[]):
        scene, tasks, checks = requested_tasks.compile_scene(spec, entries, containers, seed)
    from resimpair_grasp.scenes import item as scene_item
    for r in static_rows:
        d = read(resolve(r['asset']['descriptor']))
        scene['objects'].append(scene_item(r['asset']['descriptor'], d, r['id'], r['xy'], r['yaw'], False))
    by_id = {r['id']: r for r in list(obj_rows) + list(rec_rows) + list(static_rows)}
    descriptors = grammar.minimal_descriptors(geometry.items)
    for item in scene['objects'] + scene['supports']:
        row = by_id[item['id']]; a = row['asset']
        item['pp2'] = {'source_id': a['source_id'], 'category': a['category'], 'color': a['color'],
                       'descriptor': descriptors[item['id']]}
        item['semantic_description'] = grammar.descriptor_text(descriptors[item['id']])
        item['color_name'] = a['color']
    for task, p in zip(tasks, programs):
        task['pp2_program'] = p
        task['instruction'] = grammar.instruction(p)
        ops = task['operations']
        task['required_operation_precedence'] = ([[ops[i]['id'], ops[i + 1]['id']] for i in range(len(ops) - 1)]
                                                 if p['ordered'] else [])
        task['incidental_controller_order_is_required'] = False
    scene['pp2_robot_init_qpos'] = robot_start(seed)
    return spec, scene, tasks, checks


def build_scene(number, partition, seed, output, catalog_path=CATALOG, k=6, n_objects=None, max_trials=24,
                factors=False, multi_quota=0, usage=None, asset_rounds=4):
    """factors=True samples 3-5 objects and 0-2 static distractors (v2 factor combination).

    The trials are split over `asset_rounds` asset sets: an asset set whose layouts keep failing is
    replaced rather than retried for every trial."""
    rng = np.random.default_rng(seed)
    n_objects = n_objects or int(rng.choice([3, 4, 5] if factors else [3, 4]))
    n_static = int(rng.choice([0, 1, 2])) if factors else 0
    rounds, per_round = [], max(1, max_trials // asset_rounds)
    for round_ in range(asset_rounds):
        row = _build_scene_assets(number, partition, seed, output, rng, catalog_path, k, n_objects, n_static,
                                  per_round, factors, multi_quota, usage, trial_offset=round_ * per_round)
        rounds.append({'scene_id': row['scene_id'], 'tasks': row['tasks'], 'rejected': row.get('rejected')})
        if row['tasks']:
            break
    return {**row, 'asset_rounds': rounds}


def _build_scene_assets(number, partition, seed, output, rng, catalog_path, k, n_objects, n_static, max_trials,
                        factors, multi_quota, usage, trial_offset=0):
    from resimpair_tasks.spatial_reachability_v4 import workspace_checks
    from resimpair_tasks.multitask_scene_factory_v3 import endpoint_qualification
    catalog, manip, recs = load_catalog(catalog_path)
    for attempt in range(256):
        try:
            objects, receptacles = choose_assets(manip, recs, partition, rng, n_objects, usage=usage,
                                                 metric_preference=.7 if factors else 0.)
        except ValueError as error:
            if 'category_name_equals' not in str(error):
                raise
            continue
        # Container tasks need objects that physically fit; ask for >=2 fitting objects (factor mode) and the
        # object/destination contrast structure the task-set rules demand.
        fitting, cross = container_fit(objects, receptacles)
        if not factors or (fitting >= 2 and (cross or not multi_quota) and contrast_anchor(objects, receptacles)) \
                or attempt == 255:
            break
    else:
        raise ValueError('no_asset_choice_without_name_conflict')
    clutter = choose_distractors(manip, partition, objects + receptacles, rng, n_static) if n_static else []
    scene_id = f'pp2_scene_{number:03d}_' + digest([seed, [o['source_id'] for o in objects]])[:8]
    directory = Path(output) / 'scenes' / scene_id
    attempts = []
    for trial in range(trial_offset, trial_offset + max_trials):
        trial_seed = seed + 7919 * (trial + 1)
        trial_rng = np.random.default_rng(trial_seed)
        trial_dir = directory / 'trials' / f'{trial:02d}'
        try:
            layout = sample_layout(objects, receptacles, trial_rng, distractors=clutter)
            obj_rows, rec_rows = layout[:2]; static_rows = layout[2] if len(layout) == 3 else []
            rec_desc = {r['id']: read(resolve(r['asset']['descriptor'])) for r in rec_rows}
            geometry = SceneGeometry(obj_rows, rec_rows, rec_desc, static_rows)
            geometry.contrast_rule = factors
            descriptors = grammar.minimal_descriptors(geometry.items)
            relation = {r['id']: r['asset']['relation'] for r in rec_rows}
            programs, report = grammar.select_task_set([r['id'] for r in obj_rows], relation, descriptors, k,
                                                       trial_rng, fits=geometry.fits, feasible=geometry.feasible,
                                                       geometric_confusable=geometry.geometric_confusable,
                                                       max_next_to=2 if factors else None,
                                                       multi_quota=multi_quota, statics=[r['id'] for r in static_rows])
            problems = grammar.validate_task_set(programs)
            if problems:
                raise ValueError('task_set_rules:' + ';'.join(problems[:3]))
            spec, scene, tasks, checks = compile_programs(scene_id, obj_rows, rec_rows, programs, geometry, trial_seed,
                                                          catalog, static_rows)
            workspace = workspace_checks(scene, tasks, .90)
            write(trial_dir / 'workspace.json', workspace)
            failing = {r['task'] for r in workspace['operations'] if not r['passed']}
            endpoint = endpoint_qualification(scene, tasks, trial_dir / 'endpoint_IK')
            failing |= {r['task'] for r in endpoint['operations'] if not r['passed']}
            if failing:
                keep = [i for i, t in enumerate(tasks) if t['requested_task_id'] not in failing]
                kept = [programs[i] for i in keep]
                if grammar.validate_task_set(kept):
                    raise ValueError('reachability_failed_tasks:' + ','.join(sorted(failing)))
                programs = kept
                spec, scene, tasks, checks = compile_programs(scene_id, obj_rows, rec_rows, programs, geometry, trial_seed,
                                                              catalog, static_rows)
            write(directory / 'request.json', spec)
            write(directory / 'compiled_scene.json', scene)
            write(directory / 'compiled_tasks.json', {'tasks': tasks})
            write(directory / 'geometry_checks.json', checks)
            factor_record = {'objects': len(obj_rows), 'receptacles': len(rec_rows), 'static_distractors': len(static_rows),
                             'categories': sorted({r['asset']['category'] for r in obj_rows}),
                             'same_category_pair': len({r['asset']['category'] for r in obj_rows}) < len(obj_rows),
                             'object_longest_m': [round(max(r['asset']['dimensions_m']), 3) for r in obj_rows],
                             'asset_sources': sorted({r['asset']['source_id'].split(':')[0] for r in obj_rows})}
            write(directory / 'selection.json', {**report, 'factors': factor_record, 'trial': trial, 'seed': trial_seed,
                  'dropped_for_reachability': sorted(failing), 'workspace': binding(trial_dir / 'workspace.json'),
                  'endpoint_IK': binding(trial_dir / 'endpoint_IK/qualification.json')})
            attempts.append({'trial': trial, 'accepted': True})
            write(directory / 'attempts.json', {'attempts': attempts})
            if usage is not None:
                for r in list(obj_rows) + list(rec_rows) + list(static_rows):
                    usage[r['asset']['source_id']] = usage.get(r['asset']['source_id'], 0) + 1
            return {'scene_id': scene_id, 'directory': str(directory), 'partition': partition, 'tasks': len(tasks),
                    'families': [p['family'] for p in programs], 'trials': trial + 1, 'factors': factor_record,
                    'assets_used': [r['asset']['source_id'] for r in list(obj_rows) + list(rec_rows) + list(static_rows)]}
        except (ValueError, KeyError) as error:
            attempts.append({'trial': trial, 'accepted': False, 'reason': getattr(error, 'code', str(error))[:300]})
    write(directory / 'attempts.json', {'attempts': attempts})
    return {'scene_id': scene_id, 'directory': str(directory), 'partition': partition, 'tasks': 0,
            'rejected': attempts[-1]['reason'], 'trials': max_trials}


def _scene_job(job):
    from resimpair_runtime import configure
    configure()
    number, partition, seed, output, kwargs = job
    return build_scene(number, partition, seed, output, **kwargs)


def build(output, scenes, partition, seed, k=6, catalog_path=CATALOG, factors=False, multi_quota=0, usage=None, workers=1):
    """Scenes 0..scenes-1, in waves of `workers` processes. Asset-usage balancing sees every earlier wave (not the
    current one), so the factory is deterministic for a given worker count; workers=1 is the sequential factory."""
    from concurrent.futures import ProcessPoolExecutor
    output = Path(output).resolve()
    if (output / 'factory.json').exists():
        raise ValueError('preserve_existing_factory')
    started = time.monotonic(); rows = []; usage = dict(usage or {})
    with ProcessPoolExecutor(workers) as pool:
        for wave in range(0, scenes, workers):
            kwargs = {'catalog_path': catalog_path, 'k': k, 'factors': factors, 'multi_quota': multi_quota, 'usage': dict(usage)}
            jobs = [(n, partition, seed + 1009 * n, output, kwargs) for n in range(wave, min(scenes, wave + workers))]
            for row in pool.map(_scene_job, jobs):
                rows.append(row); print(json.dumps(row), flush=True)
                for source in row.get('assets_used', []):
                    usage[source] = usage.get(source, 0) + 1
    write(output / 'factory.json', {'schema': VERSION, 'catalog': binding(catalog_path), 'partition': partition,
          'factors': factors, 'multi_quota': multi_quota, 'asset_usage': usage, 'workers': workers,
          'seed': seed, 'scenes': rows, 'implementation': {p.name: binding(p) for p in Path(__file__).parent.glob('*.py')},
          'seconds': time.monotonic() - started, 'physical_executions': 0, 'API_calls': 0})
    return {'scenes': len(rows), 'scenes_with_tasks': sum(bool(r['tasks']) for r in rows),
            'tasks': sum(r['tasks'] for r in rows), 'seconds': time.monotonic() - started}



def build_twin(source_directory, number, seed, output, catalog_path=CATALOG, max_trials=24):
    """Same assets and programs as a source scene, new layout and yaw: a cross-scene pairing partner."""
    from resimpair_tasks.spatial_reachability_v4 import workspace_checks
    from resimpair_tasks.multitask_scene_factory_v3 import endpoint_qualification
    catalog, manip, recs = load_catalog(catalog_path)
    by_source = {a['source_id']: a for a in manip + recs}
    source_directory = Path(source_directory)
    request = read(source_directory / 'request.json')
    objects = [by_source[o['source_id']] for o in request['objects']]
    receptacles = [by_source[c['source_id']] for c in request['containers']]
    programs = [t['pp2_program'] for t in read(source_directory / 'compiled_tasks.json')['tasks']]
    scene_id = f'pp2_scene_{number:03d}_twin_' + request['id'].split('_')[-1]
    directory = Path(output) / 'scenes' / scene_id; attempts = []
    for trial in range(max_trials):
        trial_seed = seed + 7919 * (trial + 1); trial_rng = np.random.default_rng(trial_seed)
        trial_dir = directory / 'trials' / f'{trial:02d}'
        try:
            obj_rows, rec_rows = sample_layout(objects, receptacles, trial_rng)
            rec_desc = {r['id']: read(resolve(r['asset']['descriptor'])) for r in rec_rows}
            geometry = SceneGeometry(obj_rows, rec_rows, rec_desc)
            kept = [p for p in programs if geometry.feasible(p)]
            if grammar.validate_task_set(kept):
                raise ValueError('twin_task_set_rules')
            spec, scene, tasks, checks = compile_programs(scene_id, obj_rows, rec_rows, kept, geometry, trial_seed, catalog)
            workspace = workspace_checks(scene, tasks, .90); write(trial_dir / 'workspace.json', workspace)
            endpoint = endpoint_qualification(scene, tasks, trial_dir / 'endpoint_IK')
            failing = {r['task'] for r in workspace['operations'] + endpoint['operations'] if not r['passed']}
            if failing:
                keep = [p for t, p in zip(tasks, kept) if t['requested_task_id'] not in failing]
                if grammar.validate_task_set(keep):
                    raise ValueError('reachability_failed_tasks:' + ','.join(sorted(failing)))
                kept = keep
                spec, scene, tasks, checks = compile_programs(scene_id, obj_rows, rec_rows, kept, geometry, trial_seed, catalog)
            write(directory / 'request.json', spec); write(directory / 'compiled_scene.json', scene)
            write(directory / 'compiled_tasks.json', {'tasks': tasks}); write(directory / 'geometry_checks.json', checks)
            write(directory / 'selection.json', {'twin_of': str(source_directory), 'trial': trial, 'seed': trial_seed,
                  'programs_kept': len(kept), 'programs_source': len(programs)})
            attempts.append({'trial': trial, 'accepted': True}); write(directory / 'attempts.json', {'attempts': attempts})
            return {'scene_id': scene_id, 'directory': str(directory), 'partition': objects[0]['partition'],
                    'tasks': len(tasks), 'families': [p['family'] for p in kept], 'trials': trial + 1,
                    'twin_of': source_directory.name}
        except (ValueError, KeyError) as error:
            attempts.append({'trial': trial, 'accepted': False, 'reason': str(error)[:300]})
    write(directory / 'attempts.json', {'attempts': attempts})
    return {'scene_id': scene_id, 'directory': str(directory), 'tasks': 0, 'rejected': attempts[-1]['reason'],
            'twin_of': source_directory.name}


def build_slice(output, scenes, twins, partition, seed, k=6):
    """`scenes` new scenes, then twins of the first `twins` accepted ones."""
    output = Path(output).resolve()
    if (output / 'factory.json').exists():
        raise ValueError('preserve_existing_factory')
    started = time.monotonic(); rows = []
    for number in range(scenes):
        row = build_scene(number, partition, seed + 1009 * number, output, k=k)
        rows.append(row); print(json.dumps(row)[:400], flush=True)
    accepted = [r for r in rows if r['tasks']]
    for i, source in enumerate(accepted[:twins]):
        row = build_twin(source['directory'], scenes + i, seed + 500009 + 1009 * i, output)
        rows.append(row); print(json.dumps(row)[:400], flush=True)
    write(output / 'factory.json', {'schema': VERSION, 'catalog': binding(CATALOG), 'partition': partition,
          'seed': seed, 'scenes': rows, 'implementation': {p.name: binding(p) for p in Path(__file__).parent.glob('*.py')},
          'seconds': time.monotonic() - started, 'physical_executions': 0, 'API_calls': 0})
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--scenes', type=int, default=6)
    p.add_argument('--partition', default='development'); p.add_argument('--seed', type=int, default=20261101)
    p.add_argument('--k', type=int, default=6); p.add_argument('--twins', type=int, default=0)
    p.add_argument('--catalog', type=Path, default=CATALOG); p.add_argument('--factors', action='store_true')
    p.add_argument('--multi-quota', type=int, default=0)
    p.add_argument('--usage-from', type=Path, help='factory.json whose asset usage seeds the balancing')
    p.add_argument('--workers', type=int, default=1)
    a = p.parse_args()
    if a.twins: build_slice(a.output, a.scenes, a.twins, a.partition, a.seed, a.k)
    else: build(a.output, a.scenes, a.partition, a.seed, a.k, a.catalog, a.factors, a.multi_quota,
                read(a.usage_from)['asset_usage'] if a.usage_from else None, a.workers)
