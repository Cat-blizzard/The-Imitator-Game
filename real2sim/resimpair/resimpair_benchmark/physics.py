"""Matched native rollouts, retaining failed materialization in denominators."""
import argparse
import copy
import subprocess
import time

from resimpair_grasp.common import ROOT, binding, read, resolve, write
from .campaign import BASE, LEDGER, METHODS, proposal_folder


def register(split):
    from resimpair_pickplace.cudex_benchmark import scene_for
    from resimpair_pickplace.cartesian_reachability import bind_selection
    from resimpair_pickplace.runner import code
    from resimpair_pickplace import evaluator
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import core, resources
    from .proposals import qualify
    root = BASE / ('physics_' + split)
    if root.exists(): raise ValueError('Preserve registered physics campaign')
    from resimpair_pickplace.native_catalog import geometry_entries
    converted = geometry_entries(BASE / 'native_conversion', BASE / 'native_geometry_index')
    native = {read(resolve(e['asset']))['source_id']: e for e in converted}
    selection = [e for e in read(BASE / 'selection.json')['entries']
                 if e['physical_subset'] and e['split'] == split]
    snapshot = resources.gpu_snapshot(2)
    write(root / 'experiment.json', {'schema': 'general_pickplace_experiment_v1',
        'budget': str(LEDGER), 'gpu': 2, 'gpu_uuid': snapshot['uuid'], 'timeout_seconds': 360,
        'code': code(), 'legacy_code': core.implementation(), 'api_limit': 0,
        'protocol': binding(BASE / 'protocol.json'), 'registration_code': binding(__file__),
        'seed': 2026102600, 'controller': 'retained_measured_stability',
        'baseline_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()})
    write(root / 'rules.json', {'version': evaluator.VERSION, 'files': evaluator.bindings(),
        'frozen_before_attempts': True, 'retained_rules_unchanged': True})
    cases = []
    for entry in selection:
        native_entry = entry.get('native_entry', native.get(entry['source_id']))
        base_scene = task = None
        scene_error = 'native_collision_materialization_failed'
        if native_entry:
            try:
                base_scene, task = scene_for(native_entry, int(entry['id'][1:]))
                write(root / 'scenes' / (entry['id'] + '.json'), base_scene)
                write(root / 'scenes' / (entry['id'] + '_task.json'), task)
            except ValueError as error: scene_error = str(error)
        for method in METHODS:
            identity = entry['id'] + '_' + method
            folder = root / 'inputs' / identity
            case = {'id': identity, 'execution_id': identity, 'source_id': entry['source_id'],
                    'partition': entry['split'], 'experiment_split': entry['split'],
                    'split_group': entry['source_id'], 'method': method, 'layout': 0,
                    'source_coverage': 'synthetic_only', 'eligible': False,
                    'offline_asset': entry['asset']}
            if base_scene is None:
                case['preflight_reason'] = scene_error
                cases.append(case); continue
            result_path = proposal_folder(method, entry['id']) / 'result.json'
            if not result_path.exists():
                case['preflight_reason'] = 'proposal_generation_failed'
                cases.append(case); continue
            qualified = qualify(native_entry['asset'], read(result_path)['raw'],
                                 BASE / 'qualified_native' / method / entry['id'], geometry='native')
            case.update(asset=native_entry['asset'], shared_scene=binding(root / 'scenes' / (entry['id'] + '.json')),
                        native_qualification=binding(BASE / 'qualified_native' / method / entry['id'] / 'decisions.json'))
            if not qualified['retained']:
                case['preflight_reason'] = 'no_common_native_geometry_candidate'
                cases.append(case); continue
            scene, intent = copy.deepcopy(base_scene), copy.deepcopy(task)
            scene['grasp_caches']['target_object'] = qualified['cache']; intent['id'] = identity
            write(folder / 'role_intent.json', intent)
            started = time.monotonic()
            selected, planned = bind_selection(scene, intent, folder)
            case['endpoint_seconds'] = time.monotonic() - started
            case['arm_selection'] = binding(folder / 'arm_selection.json')
            if selected is not None:
                write(folder / 'scene.json', selected); write(folder / 'task.json', planned)
                case.update(eligible=True, scene=core.bind(folder / 'scene.json'), task=core.bind(folder / 'task.json'))
            else: case['preflight_reason'] = 'no_single_arm_source_and_destination_grasp'
            cases.append(case)
            print(identity, case['eligible'], flush=True)
    write(root / 'cohort.json', {'cases': cases, 'failures_retained': True,
        'same_physical_scenes_assets_initial_seeds_and_controller': True,
        'qualifier_rule_unchanged_across_methods': True, 'scene_rejections': []})
    print({'cases': len(cases), 'eligible': sum(c['eligible'] for c in cases)})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--split', required=True, choices=['development', 'heldout_asset'])
    register(p.parse_args().split)
