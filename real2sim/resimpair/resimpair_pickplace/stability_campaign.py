"""Register exact-input controller comparisons against retained executions."""
import argparse
from pathlib import Path
import subprocess
from resimpair_grasp.common import ROOT, read, write, binding, resolve
from corl2026_workshop.experiments.autonomous_pipeline_v4 import core
from .runner import code, verify
from . import evaluator
from .support_surface import check_support


def register(parent, output, case_ids, qualified_index=None):
    if output.exists():
        raise ValueError('Preserve old comparison, do not overwrite')
    spec = verify(parent)
    source = read(parent / 'cohort.json')['cases']
    cases = [c for c in source if c['id'] in case_ids]
    if len(cases) > 24:
        raise ValueError('Register at most 24 comparison cases at once')
    if len(cases) != len(set(case_ids)) or not cases:
        raise ValueError('Missing case or empty comparison')
    if any(not c['eligible'] for c in cases):
        raise ValueError('This comparison requires unchanged eligible inputs')
    output.mkdir(parents=True)
    for c in cases:
        core.checked(c['scene']); core.checked(c['task'])
    # Inspect support geometry before selecting a grasp or spending a rollout.
    import trimesh
    active = []; rejected = []
    for case in cases:
        scene = core.read(core.checked(case['scene']))
        task = core.read(core.checked(case['task']))
        checks = []
        for op in task['operations']:
            support = next((s for s in scene['supports'] if s['id'] == op['target']['support']), None)
            if support is None or 'container_descriptor' not in support: continue
            d = read(resolve(support['container_descriptor']))
            mesh = trimesh.load(resolve(d['collision']), force='mesh', process=False)
            checks.append({'support': support['id'], 'descriptor': support['container_descriptor'],
                           **check_support(mesh, d['cavity'])})
        path = output/'inputs'/case['id']/'support_preflight.json'
        write(path, {'checks': checks, 'physical_executions': 0})
        if any(not r['passed'] for r in checks):
            rejected.append({**case, 'eligible': False,
                'preflight_reason': 'declared_floor_not_a_consistent_upward_collision_surface',
                'support_preflight': binding(path)})
        else: active.append(case)
    cases = active
    if qualified_index is None and cases:
        # By default run the closing-sweep check, preserving cached proposals.
        from .closure_quality import qualify
        entries = []
        for case in cases:
            scene = core.read(core.checked(case['scene']))
            task = core.read(core.checked(case['task']))
            for op in task['operations']:
                item = next(o for o in scene['objects'] if o['id'] == op['object'])
                e = {'asset': item['asset_descriptor'], 'cache': scene['grasp_caches'][item['id']]}
                if e not in entries: entries.append(e)
        index = output/'closure_input.json'; write(index, {'entries': entries, 'failures': []})
        qualify(index, output/'closure')
        qualified_index = output/'closure'/'cache_index.json'
        first = read(qualified_index)
        if first['failures']:
            # One bounded geometry search on missing assets, never a physical
            # retry. Do not rerun successful sources or hide failed proposals.
            failed = {r['source_id'] for r in first['failures']}
            retry = [e for e in entries if read(resolve(e['asset']))['source_id'] in failed]
            retry_index = output/'closure_resample_input.json'
            write(retry_index, {'entries': retry, 'failures': [], 'first_attempt': binding(qualified_index)})
            qualify(retry_index, output/'closure_resampled', resample=True)
            second = read(output/'closure_resampled'/'cache_index.json')
            merged = output/'qualified_index.json'
            write(merged, {'entries': first['entries']+second['entries'], 'failures': second['failures'],
                           'first_attempt_failures_preserved': first['failures'],
                           'maximum_resampling_rounds': 1, 'physical_retries': 0})
            qualified_index = merged
    if qualified_index is not None:
        import copy
        from .reachability import bind_selection
        candidates = read(qualified_index)['entries']
        derived = []
        for case in cases:
            scene = copy.deepcopy(core.read(core.checked(case['scene'])))
            task = copy.deepcopy(core.read(core.checked(case['task'])))
            missing = False
            for op in task['operations']:
                item = next(o for o in scene['objects'] if o['id'] == op['object'])
                matched = [e for e in candidates if e['asset'] == item['asset_descriptor']]
                if not matched:
                    missing = True; break
                # Several scene-specific caches may bind the same source.
                # Either has already been qualified against identical geometry.
                scene['grasp_caches'][item['id']] = matched[0]['cache']
            if missing:
                rejected.append({**case, 'eligible': False, 'preflight_reason': 'no_closing_sweep_qualified_grasp'})
                continue
            folder = output/'inputs'/case['id']; folder.mkdir(parents=True, exist_ok=True)
            scene, task = bind_selection(scene, task, folder)
            if scene is None:
                rejected.append({**case, 'eligible': False, 'preflight_reason': 'no_reachable_qualified_grasp'})
                continue
            write(folder/'scene.json', scene); write(folder/'task.json', task)
            derived.append({**case, 'scene': core.bind(folder/'scene.json'),
                            'task': core.bind(folder/'task.json'), 'parent_case': case,
                            'qualified_index': binding(qualified_index)})
        cases = derived
    cases += rejected
    write(output / 'experiment.json', {**spec, 'schema': 'measured_stability_comparison_v1',
          'code': code(), 'legacy_code': core.implementation(),
          'parent_experiment': binding(parent / 'experiment.json'),
          'controller': 'measured_stability_v3_oriented_support_and_closure',
          'baseline_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          'controlled_changes': 'measured attachment refresh, settled windows, opening and retreat checks, gradual bounded opening',
          'new_budget_created': False, 'api_limit': 0})
    write(output / 'rules.json', {'version': evaluator.VERSION, 'files': evaluator.bindings(),
          'frozen_before_attempts': True, 'retained_rules_unchanged': True})
    write(output / 'cohort.json', {'cases': cases, 'parent': binding(parent / 'cohort.json'),
          'same_physical_scenes_assets_and_initial_seeds': True,
          'grasp_cache_changed': qualified_index is not None, 'failures_retained': True})
    return {'registered': len(cases), 'eligible': sum(c['eligible'] for c in cases), 'budget': spec['budget']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--case', action='append', required=True)
    p.add_argument('--qualified-index', type=Path)
    args = p.parse_args()
    print(register(args.parent, args.output, args.case, args.qualified_index))
