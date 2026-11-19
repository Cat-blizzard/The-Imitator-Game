"""Grasp bench: one asset alone on the table, one allowed grasp, the real pick-carry-place skill.

Each run gives the controller a single-candidate cache, so the outcome belongs to that
grasp. Per asset it tests the lowest-lever candidates and, as a within-asset contrast,
the highest-lever ones. Lever = distance from the asset centre of mass to the finger
closing axis. Results feed a measured grasp ordering shared by all assets.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time
from unittest import mock

import numpy as np

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write

VERSION = 'pp2_grasp_bench_v1'
OBJECT_XY, GOAL_XY = (-.12, -.25), (.02, -.30)


def lever(tcp, com):
    tcp = np.asarray(tcp, float); offset = np.asarray(com, float) - tcp[:3, 3]; axis = tcp[:3, 1]
    return float(np.linalg.norm(offset - axis * (offset @ axis)))


def features(asset):
    descriptor = read(resolve(asset['descriptor']))
    com = descriptor.get('native_COM_m') or [0., 0., 0.]
    rows = []
    for c in read(resolve(asset['cache']))['candidates']:
        t = np.asarray(c['tcp_in_asset'], float)
        yaw = float(np.arctan2(t[1, 1], t[0, 1]))
        rows.append({'candidate': c, 'lever_m': lever(t, com), 'yaw': yaw,
                     'bucket': (*np.round(t[:3, 3] / .01).astype(int).tolist(), int(np.round(yaw / .52)))})
    return rows


def choose(rows, low, high):
    seen, unique = set(), []
    for row in sorted(rows, key=lambda r: r['lever_m']):
        if row['bucket'] not in seen:
            seen.add(row['bucket']); unique.append(row)
    lows = unique[:low]
    edge = (lows[-1]['lever_m'] if lows else 0.) + .01
    highs = [r for r in unique[low:] if r['lever_m'] >= edge][-high:] if high else []
    return [(r, 'low') for r in lows] + [(r, 'high') for r in highs]


def build_case(args):
    asset, row, group, index, output = args
    from resimpair_pickplace import requested_tasks
    candidate = row['candidate']; sid = asset['source_id']
    scene_id = f"bench_{digest(sid)[:12]}_{index:02d}"
    directory = output / 'scenes' / scene_id
    signature = {'descriptor': asset['descriptor'], 'gripper': read(resolve(asset['cache']))['signature']['gripper'],
                 'parent_cache': asset['cache'], 'candidate': candidate['id']}
    write(directory / 'cache.json', {'schema': 'resimpair_mesh_grasp_cache_v1', 'signature': signature,
          'key': digest(signature), 'candidates': [candidate], 'physical_qualification': 'unknown'})
    spec = {'id': scene_id, 'objects': [{'id': 'object_0', 'source_id': sid, 'xy': list(OBJECT_XY), 'yaw': 0.}],
            'containers': [], 'tasks': [{'id': 'task_0', 'placements': [{'object': 'object_0',
             'target': {'kind': 'table', 'xy': list(GOAL_XY)}}]}]}
    entry = {'asset': asset['descriptor'], 'cache': binding(directory / 'cache.json')}
    with mock.patch.object(requested_tasks, 'add_distractors', return_value=[]):
        scene, tasks, _ = requested_tasks.compile_scene(spec, [entry], [], int(digest(scene_id)[:8], 16))
    scene['objects'][0]['pp2'] = {'source_id': sid, 'category': asset['category'], 'color': asset['color'],
                                  'descriptor': {'category': asset['category']}}
    task = tasks[0]
    task.update(pp2_program={'schema': VERSION, 'family': 'grasp_bench', 'ordered': False, 'steps': []},
                instruction=f"Bench {asset['category']} with {candidate['id']}")
    write(directory / 'compiled_scene.json', scene); write(directory / 'compiled_tasks.json', {'tasks': [task]})
    return {'scene_id': scene_id, 'directory': str(directory), 'partition': asset['partition'], 'tasks': 1,
            'source_id': sid, 'candidate': candidate['id'], 'group': group, 'lever_m': row['lever_m'],
            'width_m': candidate['width_m'], 'source': candidate.get('source', 'antipodal'), 'category': asset['category']}


def prepare(catalog, output, low=5, high=3, workers=32):
    output = Path(output).resolve()
    if (output / 'factory.json').exists():
        raise ValueError('preserve_existing_bench')
    assets = [a for a in read(catalog)['assets'] if a['role'] == 'manipulable']
    jobs = []
    for asset in assets:
        for index, (row, group) in enumerate(choose(features(asset), low, high)):
            jobs.append((asset, row, group, index, output))
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(build_case, jobs))
    write(output / 'factory.json', {'schema': VERSION, 'catalog': binding(catalog), 'scenes': rows,
          'low': low, 'high': high, 'physical_executions': 0, 'API_calls': 0})
    return {'assets': len(assets), 'cases': len(rows), 'groups': dict(Counter(r['group'] for r in rows))}


def score(cohort, factory):
    cohort = Path(cohort); rows = read(Path(factory) / 'factory.json')['scenes']
    out = []
    for row in rows:
        d = cohort / 'executions' / (row['scene_id'] + '__task_0')
        result = read(d / 'worker_result.json') if (d / 'worker_result.json').exists() else {}
        failure = (result.get('failure') or {}).get('message')
        if failure:
            outcome = failure.split(':')[0]
        elif not (d / 'measurements.npz').exists():
            outcome = 'no_measurements'
        else:
            states = np.load(d / 'measurements.npz')['object_states']
            outcome = 'success' if np.linalg.norm(states[-1, 0, :2] - np.asarray(GOAL_XY)) <= .03 else 'missed_goal'
        out.append({**row, 'outcome': outcome})
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--low', type=int, default=5); p.add_argument('--high', type=int, default=3)
    a = p.parse_args(); print(json.dumps(prepare(a.catalog, a.output, a.low, a.high)))
