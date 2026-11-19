"""Stored form of a pp2 run: one record per accepted execution, plus the reference-group manifest.

dataset/index.jsonl                      one line per sample (accepted execution)
dataset/samples/<case>/meta.json         task, program, scene, assets, robot start, render variants, group link
dataset/samples/<case>/trajectory.npz    timestamps, joint states, joint-position actions, TCP and object poses,
                                         grasp flags, and the saved physics states that re-render it
Videos stay where the render stage wrote them (renders/<case>/v<k>/<camera>.mp4); meta.json points at them.

groups/manifest.json (reference mode) is the "one human video, many robot samples" map: per reference
group and program, the source execution a human video is made from (layout 0) and every accepted robot
sample of the same program in the group's other layouts, each with its render variants.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from resimpair_grasp.common import ROOT, binding, read, resolve, write
from . import grammar

VERSION = 'pp2_dataset_v1'
TRAJECTORY_KEYS = ('timestamps_seconds', 'states', 'native_commands', 'tcp_world', 'object_states', 'grasp',
                   'robot_roots_world')


def rewrite(path, value):
    """Derived records (dataset meta, manifests) are rebuilt from immutable evidence, so they may be replaced."""
    Path(path).unlink(missing_ok=True); write(Path(path), value)


def relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def renders_of(renders, case):
    rows = []
    for result in sorted((Path(renders) / case).glob('v*/render_result.json')):
        r = read(result); out = result.parent
        rows.append({'variant': out.name, 'room': r['room'], 'room_seed': r['room_seed'], 'shader': r['render']['shader'],
                     'frames': r['frames'], 'fps': r['fps'],
                     'videos': {name: relative(out / f'{name}.mp4') for name in r['videos']},
                     'keyframes': {p.stem: relative(p) for p in sorted(out.glob('*_initial.png')) + sorted(out.glob('*_final.png'))},
                     'object_masks': {p.stem: relative(p) for p in sorted(out.glob('*_object_masks.npz'))},
                     'result': relative(result)})
    return rows


def sample(cohort, case, verdict, renders, output):
    execution = Path(cohort) / 'executions' / case['id']
    scene, task = read(execution / 'effective_scene.json'), read(execution / 'effective_task.json')
    measured = np.load(execution / 'measurements.npz')
    states = np.load(execution / 'deferred_render_states.npz')
    target = Path(output) / 'samples' / case['id']; target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target / 'trajectory.npz', **{k: measured[k] for k in TRAJECTORY_KEYS if k in measured.files},
                        physics_states=states['physics_states'])
    items = scene['objects'] + scene['supports']
    meta = {'schema': VERSION, 'sample_id': case['id'], 'scene_id': case['scene_id'], 'partition': case['partition'],
            'family': verdict['family'], 'instruction': task['instruction'], 'program': task['pp2_program'],
            'program_key': grammar.canonical(task['pp2_program']),
            'reference_group': task.get('reference_group'),
            'assets': {o['id']: {k: o.get('pp2', {}).get(k) for k in ('source_id', 'category', 'color')} for o in items},
            'robot_start_qpos': [measured['states'][0][9 * i:9 * i + 9].round(4).tolist() for i in range(2)],  # as executed
            # step boundaries (trajectory rows): per-step keyframes for human-video conditioning, subtask labels
            'steps': [{k: op.get(k) for k in ('operation_id', 'first_frame', 'last_frame')}
                      for op in read(execution / 'executed_operations.json')['operations']],
            'action_space': 'pd_joint_pos: per arm 7 joint targets + gripper (native_commands, one row per control step)',
            'trajectory': relative(target / 'trajectory.npz'), 'renders': renders_of(renders, case['id']),
            'execution': relative(execution), 'acceptance': verdict['schema']}
    rewrite(target / 'meta.json', meta)
    return meta


def index(cohort, renders, output):
    cohort, output = Path(cohort).resolve(), Path(output).resolve()
    rows = []
    for case in read(cohort / 'cohort.json')['cases']:
        verdict = cohort / 'acceptance/cases' / (case['id'] + '.json')
        if verdict.exists() and read(verdict)['passed']:
            rows.append(sample(cohort, case, read(verdict), renders, output))
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'index.jsonl').open('w') as stream:
        for meta in rows:
            stream.write(json.dumps({k: meta[k] for k in ('sample_id', 'scene_id', 'family', 'instruction', 'program_key',
                                                          'reference_group', 'trajectory')} |
                                    {'render_variants': len(meta['renders'])}, ensure_ascii=False) + '\n')
    return {'samples': len(rows), 'with_renders': sum(bool(m['renders']) for m in rows),
            'render_variants': sum(len(m['renders']) for m in rows), 'index': relative(output / 'index.jsonl')}


def group_manifest(source_dataset, group_dataset, output):
    """Per reference group and program: the human-video source sample and the robot samples sharing its program."""
    load = lambda d: [read(p) for p in sorted(Path(d).glob('samples/*/meta.json'))]
    sources = {(m['scene_id'], m['program_key']): m for m in load(source_dataset)}
    robots = defaultdict(list)
    for m in load(group_dataset):
        robots[(m['reference_group'], m['program_key'])].append(m)
    groups = defaultdict(list)
    for (group, key), members in sorted(robots.items()):
        source = sources.get((group, key))
        groups[group].append({
            'program_key': key, 'instruction': members[0]['instruction'], 'family': members[0]['family'],
            'human_video_source': None if source is None else {'sample_id': source['sample_id'], 'renders': source['renders']},
            'human_video': {'status': 'pending', 'path': None},
            'robot_samples': [m['sample_id'] for m in members],
            'robot_sample_renders': sum(max(1, len(m['renders'])) for m in members)})
    manifest = {'schema': VERSION + '_groups', 'groups': dict(groups),
                'programs': sum(len(v) for v in groups.values()),
                'robot_sample_renders': sum(p['robot_sample_renders'] for v in groups.values() for p in v)}
    rewrite(Path(output), manifest)
    return {k: manifest[k] for k in ('programs', 'robot_sample_renders')} | {'groups': len(groups)}
