"""Semantic acceptance for pp2 executions (spec section 5).

Reads only exported measurements, the effective scene and the task program;
never planner internals. Every check is reported separately; missing evidence
is 'unknown' and does not pass.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from resimpair_grasp.common import ROOT, binding, read, resolve, write
from . import grammar

VERSION = 'pp2_semantic_acceptance_v3'  # v2: not_satisfied_initially; v3: visible_displacement
RULES = {'initial_shift_m': .005, 'initial_tilt_deg': 5., 'settle_window_s': .5, 'settle_shift_m': .002,
         'settle_rotation_deg': 2., 'non_target_shift_m': .02, 'non_target_rotation_deg': 15.,
         'next_to_max_edge_gap_m': .10, 'next_to_min_edge_gap_m': -.005, 'next_to_max_angle_deg': 45.,
         'cavity_margin_m': .01, 'min_visible_move_m': .08, 'support_contact_fraction': .5, 'off_table_z_m': -.02}


def rotation(state):
    w, x, y, z = state[3:7]
    return Rotation.from_quat([x, y, z, w])


def angle_deg(a, b):
    return float(np.degrees((a.inv() * b).magnitude()))


def half_extent(item):
    return .5 * max(2 * item['half_size'][0], 2 * item['half_size'][1])


def evaluate(directory, program=None):
    """program: optional substitute program, used for counterfactual checks."""
    directory = Path(directory)
    checks, failures, unknown = {}, [], []
    def record(name, passed, **evidence):
        checks[name] = {'passed': passed, **evidence}
        if passed is None: unknown.append(name)
        elif not passed: failures.append(name)
    terminal = read(directory / 'terminal.json') if (directory / 'terminal.json').exists() else {}
    worker = read(directory / 'worker_result.json') if (directory / 'worker_result.json').exists() else {}
    task = read(directory / 'effective_task.json') if (directory / 'effective_task.json').exists() else None
    if task is None:
        task = read(resolve(read(directory / 'reservation.json')['case']['task']))
    program = program or task['pp2_program']
    result = {'schema': VERSION, 'case': directory.name, 'family': program['family'], 'instruction': task.get('instruction'),
              'rules': RULES}
    record('process_completed', terminal.get('returncode') == 0 and bool(worker.get('bundle')) and not worker.get('failure'),
           returncode=terminal.get('returncode'), failure=(worker.get('failure') or {}).get('message'))
    if failures:
        return {**result, 'passed': False, 'checks': checks, 'failures': failures, 'unknown': unknown}
    scene = read(directory / 'effective_scene.json'); bundle = read(directory / 'execution_bundle.json')
    m = np.load(directory / 'measurements.npz')
    states, t = m['object_states'], m['timestamps_seconds']
    object_ids, support_ids = bundle['object_ids'], bundle['support_ids']
    items = {o['id']: o for o in scene['objects'] + scene['supports']}
    catalog_items = [{'id': i['id'], 'category': i['pp2']['category'], 'color': i['pp2']['color']}
                     for i in scene['objects'] + scene['supports'] if 'pp2' in i]
    index = {oid: k for k, oid in enumerate(object_ids)}
    window = t >= t[-1] - RULES['settle_window_s']
    final = states[-1]

    # Initial state equals the requested layout (no toppling at reset).
    worst_shift, worst_tilt = 0., 0.
    for oid in object_ids:
        req = items[oid]; s0 = states[0, index[oid]]
        worst_shift = max(worst_shift, float(np.linalg.norm(s0[:2] - np.asarray(req['position'][:2]))))
        worst_tilt = max(worst_tilt, float(np.degrees(np.arccos(np.clip(rotation(s0).apply([0, 0, 1])[2], -1, 1)))))
    record('initial_layout_stable', worst_shift <= RULES['initial_shift_m'] and worst_tilt <= RULES['initial_tilt_deg'],
           max_shift_m=worst_shift, max_tilt_deg=worst_tilt)

    # Everything at rest in the final window.
    moves = {oid: float(np.linalg.norm(states[window, index[oid], :3] - final[index[oid], :3], axis=1).max()) for oid in object_ids}
    turns = {oid: max(angle_deg(rotation(s), rotation(final[index[oid]])) for s in states[window, index[oid]]) for oid in object_ids}
    record('final_at_rest', max(moves.values()) <= RULES['settle_shift_m'] and max(turns.values()) <= RULES['settle_rotation_deg'],
           max_shift_m=max(moves.values()), max_rotation_deg=max(turns.values()))
    record('nothing_off_table', bool(final[:, 2].min() > RULES['off_table_z_m']), min_z_m=float(final[:, 2].min()))

    impulses = m['support_impulses']  # (T, objects, supports)
    table = support_ids.index('table')
    moved = set()
    for k, s in enumerate(program['steps']):
        oid = grammar.resolve_descriptor(s['object'], catalog_items)
        tid = grammar.resolve_descriptor(s['target'], catalog_items)
        moved.add(oid); p = final[index[oid], :3]; name = f'step_{k}_{s["relation"]}'
        if s['relation'] in ('in', 'on'):
            support = items[tid]; cavity = read(resolve(support['container_descriptor']))['cavity']
            yaw = support['yaw']; c, sn = np.cos(yaw), np.sin(yaw)
            center = np.asarray(support['position'][:2]) + np.array([[c, -sn], [sn, c]]) @ np.asarray(cavity['center_xy_m'])
            radius = float(min(cavity['flat_floor_radius_m'], cavity.get('minimum_wall_radius_m', cavity['flat_floor_radius_m'])))
            offset = float(np.linalg.norm(p[:2] - center))
            on_target = float((impulses[window, index[oid], support_ids.index(tid)] > 0).mean())
            on_table = float((impulses[window, index[oid], table] > 0).mean())
            record(name, offset <= radius + RULES['cavity_margin_m'] and on_target >= RULES['support_contact_fraction']
                   and on_table == 0., object=oid, target=tid, center_offset_m=offset, cavity_radius_m=radius,
                   target_contact_fraction=on_target, table_contact_fraction=on_table)
        else:
            lp = final[index[tid], :3] if tid in index else np.asarray(items[tid]['position'])
            delta = p[:2] - lp[:2]; distance = float(np.linalg.norm(delta))
            gap = distance - half_extent(items[oid]) - half_extent(items[tid])
            direction = np.asarray(grammar.SIDES[s['side']])
            angle = float(np.degrees(np.arccos(np.clip(delta @ direction / max(distance, 1e-9), -1, 1))))
            on_table = float((impulses[window, index[oid], table] > 0).mean())
            record(name, RULES['next_to_min_edge_gap_m'] <= gap <= RULES['next_to_max_edge_gap_m']
                   and angle <= RULES['next_to_max_angle_deg'] and on_table > 0, object=oid, landmark=tid,
                   edge_gap_m=gap, angle_deg=angle, table_contact_fraction=on_table)
    # A task whose every step already holds in the initial state is not informative.
    initially = []
    for s in program['steps']:
        oid = grammar.resolve_descriptor(s['object'], catalog_items); tid = grammar.resolve_descriptor(s['target'], catalog_items)
        p0 = states[0, index[oid], :3]
        if s['relation'] == 'next_to':
            lp = states[0, index[tid], :3] if tid in index else np.asarray(items[tid]['position'])
            delta = p0[:2] - lp[:2]; distance = float(np.linalg.norm(delta))
            gap = distance - half_extent(items[oid]) - half_extent(items[tid])
            angle = float(np.degrees(np.arccos(np.clip(delta @ np.asarray(grammar.SIDES[s['side']]) / max(distance, 1e-9), -1, 1))))
            initially.append(RULES['next_to_min_edge_gap_m'] <= gap <= RULES['next_to_max_edge_gap_m'] and angle <= RULES['next_to_max_angle_deg'])
        else:
            initially.append(bool((impulses[:5, index[oid], support_ids.index(tid)] > 0).any()))
    record('not_satisfied_initially', not all(initially), steps_already_holding=initially)
    travel = {oid: float(np.linalg.norm(final[index[oid], :2] - states[0, index[oid], :2])) for oid in moved}
    record('visible_displacement', min(travel.values()) >= RULES['min_visible_move_m'], travel_m=travel)
    worst = (0., 0., None)
    for oid in object_ids:
        if oid in moved:
            continue
        shift = float(np.linalg.norm(final[index[oid], :3] - states[0, index[oid], :3]))
        turn = angle_deg(rotation(states[0, index[oid]]), rotation(final[index[oid]]))
        if shift > worst[0] or turn > worst[1]:
            worst = (max(shift, worst[0]), max(turn, worst[1]), oid)
    record('non_targets_preserved', worst[0] <= RULES['non_target_shift_m'] and worst[1] <= RULES['non_target_rotation_deg'],
           max_shift_m=worst[0], max_rotation_deg=worst[1], object=worst[2])
    if program['ordered']:
        grasp = m['grasp'].any(axis=1)  # (T, objects)
        releases = []
        for s in program['steps']:
            oid = grammar.resolve_descriptor(s['object'], catalog_items)
            held = np.flatnonzero(grasp[:, index[oid]])
            releases.append(int(held[-1]) if len(held) else None)
        if None in releases:
            record('order', None, release_frames=releases)
        else:
            record('order', all(a < b for a, b in zip(releases, releases[1:])), release_frames=releases)
    return {**result, 'passed': not failures and not unknown, 'checks': checks, 'failures': failures, 'unknown': unknown}


def evaluate_cohort(root, output=None):
    root = Path(root).resolve(); output = Path(output or root / 'acceptance')
    rows = []
    for case in read(root / 'cohort.json')['cases']:
        directory = root / 'executions' / case['id']
        if not (directory / 'terminal.json').exists():
            continue
        try:
            value = evaluate(directory)
        except Exception as error:  # evidence problem, never a pass
            value = {'schema': VERSION, 'case': case['id'], 'passed': False, 'failures': ['evaluation_error'],
                     'unknown': [], 'checks': {}, 'error': repr(error)[:400], 'family': case.get('pp2_family')}
        path = output / 'cases' / (case['id'] + '.json')
        if path.exists():
            path.unlink()  # acceptance is derived; recomputed from immutable execution evidence
        write(path, value); rows.append(value)
    summary = {'schema': VERSION, 'implementation': binding(Path(__file__)), 'cohort': str(root),
               'executed': len(rows), 'passed': sum(r['passed'] for r in rows),
               'by_family': {f: [sum(r['passed'] for r in rows if r['family'] == f), sum(r['family'] == f for r in rows)]
                             for f in sorted({r['family'] for r in rows})},
               'failure_counts': dict(Counter(x for r in rows for x in r['failures'])),
               'unknown_counts': dict(Counter(x for r in rows for x in r['unknown']))}
    (output / 'summary.json').unlink(missing_ok=True)
    write(output / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args(); print(json.dumps(evaluate_cohort(a.root), indent=1))


def counterfactual_matrix(root):
    """Evaluate each completed execution against every program of its scene.

    Off-diagonal passes would mean the acceptance cannot tell two tasks of the
    same scene apart from their end states and event order.
    """
    root = Path(root).resolve(); cases = read(root / 'cohort.json')['cases']
    by_scene = {}
    for case in cases:
        by_scene.setdefault(case['scene_id'], []).append(case)
    rows = []
    for scene_id, group in by_scene.items():
        programs = {c['id']: read(resolve(c['task']))['pp2_program'] for c in group}
        for case in group:
            directory = root / 'executions' / case['id']
            if not (directory / 'terminal.json').exists() or not evaluate(directory)['passed']:
                continue
            for other, program in programs.items():
                if other == case['id']:
                    continue
                verdict = evaluate(directory, program)
                rows.append({'execution': case['id'], 'program_of': other, 'passed': verdict['passed'],
                             'failures': verdict['failures']})
    return {'pairs': len(rows), 'false_passes': [r for r in rows if r['passed']]}
