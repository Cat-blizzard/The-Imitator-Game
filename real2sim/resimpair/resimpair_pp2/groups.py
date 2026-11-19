"""Reference groups: one human reference video per task program, hundreds of robot samples per video.

A group fixes the task-relevant object instances (targets, receptacles, landmarks) and the K programs
of a source scene. Layout 0 is reserved for the reference video. Every other layout re-samples all
object positions and yaws and a fresh distractor set (never sharing a category with a task item), and
keeps all K programs that stay feasible there, so each robot observation still admits several tasks.
Every layout also samples its own robot start (scenes.robot_start). Visual variation (room, lighting,
materials, camera) is added later by re-rendering saved states (render variants).
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from resimpair_grasp.common import binding, digest, read, resolve, write
from . import grammar, scenes

VERSION = 'pp2_reference_group_v1'


def build_layout(source, number, seed, output, catalog_path, distractor_range=(0, 3), max_trials=24):
    from resimpair_tasks.spatial_reachability_v4 import workspace_checks
    from resimpair_tasks.multitask_scene_factory_v3 import endpoint_qualification
    catalog, manip, recs = scenes.load_catalog(catalog_path)
    by_source = {a['source_id']: a for a in manip + recs}
    request = read(Path(source) / 'request.json')
    objects = [by_source[o['source_id']] for o in request['objects']]
    receptacles = [by_source[c['source_id']] for c in request['containers']]
    programs = [t['pp2_program'] for t in read(Path(source) / 'compiled_tasks.json')['tasks']]
    rng = np.random.default_rng(seed)
    clutter = scenes.choose_distractors(manip, objects[0]['partition'], objects + receptacles, rng,
                                        int(rng.integers(distractor_range[0], distractor_range[1] + 1)))
    group_id = Path(source).name
    scene_id = f'{group_id}__L{number:03d}'
    directory = Path(output) / 'scenes' / scene_id; attempts = []
    # Score layout candidates by the programs they keep (multi-step ones first: they are the most layout-sensitive
    # and the most valuable), then compile and check reachability best-first.
    candidates = []
    for trial in range(max_trials):
        trial_seed = seed + 7919 * (trial + 1); trial_rng = np.random.default_rng(trial_seed)
        try:
            layout = scenes.sample_layout(objects, receptacles, trial_rng, distractors=clutter)
        except ValueError as error:
            attempts.append(str(error)[:160]); continue
        obj_rows, rec_rows = layout[:2]; static_rows = layout[2] if len(layout) == 3 else []
        rec_desc = {r['id']: read(resolve(r['asset']['descriptor'])) for r in rec_rows}
        geometry = scenes.SceneGeometry(obj_rows, rec_rows, rec_desc, static_rows)
        descriptors = grammar.minimal_descriptors(geometry.items)
        if any(descriptors[r['id']] is None for r in obj_rows + rec_rows):
            attempts.append('distractor_made_a_descriptor_ambiguous'); continue
        kept = [p for p in programs if geometry.feasible(p)]
        candidates.append(((sum(len(p['steps']) > 1 for p in kept), len(kept)), trial, trial_seed,
                           obj_rows, rec_rows, static_rows, geometry, kept))
    candidates.sort(key=lambda c: (c[0], -c[1]), reverse=True)
    for _, trial, trial_seed, obj_rows, rec_rows, static_rows, geometry, kept in candidates[:6]:
        try:
            if len(kept) < max(2, len(programs) // 2):
                raise ValueError('too_few_programs_feasible_in_layout')
            spec, scene, tasks, checks = scenes.compile_programs(scene_id, obj_rows, rec_rows, kept, geometry,
                                                                 trial_seed, catalog, static_rows)
            workspace = workspace_checks(scene, tasks, .90)
            endpoint = endpoint_qualification(scene, tasks, directory / 'trials' / f'{trial:02d}' / 'endpoint_IK')
            failing = {r['task'] for r in workspace['operations'] + endpoint['operations'] if not r['passed']}
            if failing:
                kept = [p for t, p in zip(tasks, kept) if t['requested_task_id'] not in failing]
                if len(kept) < max(2, len(programs) // 2):
                    raise ValueError('reachability_left_too_few_programs')
                spec, scene, tasks, checks = scenes.compile_programs(scene_id, obj_rows, rec_rows, kept, geometry,
                                                                     trial_seed, catalog, static_rows)
            for task in tasks:
                task['reference_group'] = group_id
                task['reference_program_key'] = grammar.canonical(task['pp2_program'])
            write(directory / 'request.json', spec); write(directory / 'compiled_scene.json', scene)
            write(directory / 'compiled_tasks.json', {'tasks': tasks})
            write(directory / 'selection.json', {'reference_group': group_id, 'layout': number, 'seed': trial_seed,
                  'programs_kept': len(kept), 'programs_source': len(programs), 'distractors': len(static_rows),
                  'distractor_categories': sorted(r['asset']['category'] for r in static_rows),
                  'candidates_scored': len(candidates)})
            return {'scene_id': scene_id, 'directory': str(directory), 'partition': objects[0]['partition'],
                    'tasks': len(tasks), 'families': [p['family'] for p in kept], 'layout': number,
                    'reference_group': group_id, 'distractors': len(static_rows)}
        except (ValueError, KeyError) as error:
            attempts.append(str(error)[:160])
    return {'scene_id': scene_id, 'directory': str(directory), 'tasks': 0, 'rejected': (attempts or ['no_layout'])[-1],
            'layout': number, 'reference_group': group_id}


def _layout_job(job):
    from resimpair_runtime import configure
    configure()
    return build_layout(*job)


def build(sources, layouts, seed, output, catalog_path, workers=1):
    """`layouts` re-laid-out scenes per source scene, `workers` processes at a time."""
    from concurrent.futures import ProcessPoolExecutor
    output = Path(output).resolve()
    if (output / 'factory.json').exists():
        raise ValueError('preserve_existing_group_factory')
    started = time.monotonic()
    jobs = [(source, n, seed + 100003 * g + 1009 * n, output, catalog_path)
            for g, source in enumerate(sources) for n in range(1, layouts + 1)]
    rows = []
    with ProcessPoolExecutor(workers) as pool:
        for row in pool.map(_layout_job, jobs):
            rows.append(row); print(json.dumps({k: row.get(k) for k in ('scene_id', 'tasks', 'distractors', 'rejected')}), flush=True)
    write(output / 'factory.json', {'schema': VERSION, 'sources': [str(s) for s in sources], 'layouts_per_group': layouts,
          'reference_layout': 'the source scene itself (layout 0) is reserved for the human video',
          'catalog': binding(Path(catalog_path)), 'seed': seed, 'workers': workers, 'scenes': rows,
          'seconds': time.monotonic() - started})
    return {'groups': len(sources), 'layouts': len(rows), 'layouts_with_tasks': sum(bool(r['tasks']) for r in rows),
            'tasks': sum(r['tasks'] for r in rows), 'seconds': time.monotonic() - started}


def pairs(cohorts):
    """Robot samples per reference program: accepted executions across layouts sharing the program key."""
    per = Counter()
    for cohort in map(Path, cohorts):
        for case in read(cohort / 'cohort.json')['cases']:
            verdict = cohort / 'acceptance/cases' / (case['id'] + '.json')
            if verdict.exists() and read(verdict)['passed']:
                task = read(resolve(case['task']))
                per[(task.get('reference_group'), task.get('reference_program_key'))] += 1
    return per


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources', type=Path, nargs='+', required=True); p.add_argument('--layouts', type=int, default=20)
    p.add_argument('--seed', type=int, default=20262501); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--catalog', type=Path, required=True); p.add_argument('--workers', type=int, default=1)
    a = p.parse_args(); build(a.sources, a.layouts, a.seed, a.output, a.catalog, a.workers)
