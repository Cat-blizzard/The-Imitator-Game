"""Finite matched CPU pilot of supported placement versus above-rim release."""
import argparse
import copy
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from resimpair_grasp.common import ROOT, binding, read, resolve, write

VERSION = 'drop_release_matched_pilot_v1'
ALLOCATION = 'above_rim_release_20261030_v1_six_attempts'
REGISTRY = ROOT / 'runtime/drop_release_registry_v1'
PARENT = ROOT / 'runs/multitask_10x8_20261030_v3'
CASES = ('multitask_scene_008_1a0d29dd_selected_subset',
         'multitask_scene_000_79ef2169_collect_group',
         'multitask_scene_003_e4d15ed3_collect_group')


def prepare(root, event_builder):
    from resimpair_pickplace.runner import code
    root = Path(root).resolve()
    if root.exists(): raise ValueError('preserve_existing_pilot')
    parent = PARENT / 'candidate'
    previous = read(PARENT / 'budget/limits.json')
    prior_actual = previous['prior_lineage']['prior_observed_attempts'] + len(list((PARENT / 'budget/reservations').glob('*.json')))
    if previous['cumulative_scope_limit'] != 298 or prior_actual != 132:
        raise ValueError('reconcile_parent_132_actual_298_cap_before_new_allocation')
    cases = {c['id']: c for c in read(parent / 'cohort.json')['cases']}
    implementations = code()
    for file in ('resimpair_tasks/drop_release_pilot_v1.py', 'resimpair_tasks/multitask_worker_v3.py',
                 'resimpair_tasks/multitask_campaign_v3.py', 'resimpair_tasks/headless_physics_v1.py'):
        implementations[file] = binding(ROOT / file)
    limits = {'schema': VERSION, 'physical_attempt_limit': 6, 'parallel_workers': 1,
        'timeout_seconds': 360, 'GPU_jobs': 0, 'API_calls': 0, 'retries': 0,
        'parent_budget': binding(PARENT / 'budget/limits.json'), 'prior_scope_limit': previous['cumulative_scope_limit'],
        'cumulative_scope_limit': previous['cumulative_scope_limit'] + 6,
        'prior_actual_attempts': prior_actual, 'unspent_old_allowance_not_reused': True}
    REGISTRY.mkdir(parents=True, exist_ok=True)
    with (REGISTRY / '.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipt = REGISTRY / (ALLOCATION + '.json')
        if receipt.exists(): raise ValueError('allocation_already_registered_recover_original_root')
        write(root / 'budget/limits.json', {**limits, 'allocation_id': ALLOCATION})
        write(receipt, {'allocation_id': ALLOCATION, 'root': str(root), 'limit': 6,
                       'budget': binding(root / 'budget/limits.json'), 'registered_unix': time.time()})
    for variant in ('supported', 'drop'):
        out = root / variant; cohort = []
        for cid in CASES:
            original = cases[cid]; task = copy.deepcopy(read(resolve(original['task'])))
            task['placement_contract'] = {'schema': 'supported_or_low_drop_v1',
                'maximum_release_height_over_goal_m': .15, 'xy_tolerance_m': .015,
                'continuous_hold_until_release': True}
            for op in task['operations']:
                if op['target']['support'] == 'table': raise ValueError('container_only_pilot')
                op['release'] = {'mode': 'drop_from_above' if variant == 'drop' else 'supported_place',
                    'allow_drop': variant == 'drop', 'fragile': False,
                    'material_status': 'rigid_nonbreakable_simulation_assumption_not_real_item_claim',
                    'bottom_above_rim_m': .03, 'hand_above_rim_m': .01, 'maximum_fall_m': .15}
            write(out / 'inputs' / cid / 'task.json', task)
            cohort.append({**original, 'task': binding(out / 'inputs' / cid / 'task.json'),
                           'parent_task': original['task'], 'paired_variant': variant,
                           'pilot_role': 'diagnostic' if cid == CASES[0] else 'regression' if cid == CASES[1] else 'retained_validation'})
        spec = copy.deepcopy(read(parent / 'experiment.json'))
        spec.update(schema=VERSION, code=implementations, budget=str(root / 'budget'),
                    drop_adapter=binding(ROOT / 'resimpair_pickplace/above_rim_release_v1.py'),
                    drop_event_builder=binding(event_builder), parent_experiment=binding(parent / 'experiment.json'))
        write(out / 'experiment.json', spec); write(out / 'rules.json', read(parent / 'rules.json'))
        write(out / 'cohort.json', {'cases': cohort, 'selection_before_execution': True})
        write(out / 'protocol.json', {'schema': VERSION, 'driver': binding(__file__),
            'budget': binding(root / 'budget/limits.json'), 'original_rules': binding(parent / 'rules.json'),
            'event_builder': binding(event_builder), 'variant': variant,
            'same_final_goals_and_physical_guards': True, 'no_layout_or_size_or_mass_change': True,
            'full_SE3_selection_shared_by_both_variants': True})
    write(root / 'protocol.json', {'schema': VERSION, 'implementation': binding(__file__),
        'event_builder': binding(event_builder), 'budget': binding(root / 'budget/limits.json'),
        'allocation_receipt': binding(REGISTRY / (ALLOCATION + '.json')),
        'frozen_inputs': [binding(root / v / f) for v in ('supported', 'drop')
                          for f in ('experiment.json', 'cohort.json', 'rules.json', 'protocol.json')],
        'purpose': 'test_release_strategy_on_same_frozen_legacy_size_inputs_not_metric_size_improvement',
        'same_new_event_semantics_for_both_variants': True,
        'original_event_audit_also_retained': True})
    return {'root': str(root), 'attempt_limit': 6, 'prior_actual': prior_actual}


def worker(root, case_id):
    from resimpair_runtime import configure
    configure()
    from resimpair_pickplace.above_rim_release_v1 import install
    from resimpair_pickplace import drop_event_protocol_v1 as events
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import planning_protocols
    from .multitask_worker_v3 import worker as retained
    spec = read(Path(root) / 'experiment.json'); resolve(spec['drop_event_builder'])
    case = next(c for c in read(Path(root) / 'cohort.json')['cases'] if c['id'] == case_id)
    task = read(resolve(case['task']))
    original = planning_protocols.pick_place
    # The independent builder calls its frozen baseline function, never this wrapper.
    planning_protocols.pick_place = events.pick_place
    try:
        with install(task): return retained(root, case_id)
    finally:
        planning_protocols.pick_place = original
        directory = Path(root) / 'executions' / case_id
        if (directory / 'effective_scene.json').exists() and (directory / 'effective_task.json').exists():
            from corl2026_workshop.experiments.autonomous_pipeline_v4.simulation import targets
            scene, actual_task = read(directory / 'effective_scene.json'), read(directory / 'effective_task.json')
            ids = [o['id'] for o in scene['objects']]
            supports = ids + [s['id'] for s in scene['supports']] + ['table']
            write(directory / 'retained_event_protocol.json', original(scene, actual_task, ids, supports, targets(scene, actual_task)))


def run(root):
    from .multitask_campaign_v3 import admission_check
    from resimpair_pickplace.runner import verify
    root = Path(root).resolve(); protocol = read(root / 'protocol.json')
    for ref in protocol['frozen_inputs']: resolve(ref)
    resolve(protocol['implementation']); resolve(protocol['event_builder']); resolve(protocol['budget'])
    receipt = read(resolve(protocol['allocation_receipt']))
    if receipt['root'] != str(root) or receipt['limit'] != 6 or receipt['allocation_id'] != ALLOCATION:
        raise ValueError('registered_allocation_mismatch')
    locks = ROOT / 'runtime/multitask_campaign_registry_v3'; locks.mkdir(exist_ok=True, parents=True)
    rows = []
    with (locks / '.execution.lock').open('a+') as shared, (root / '.supervisor.lock').open('a+') as owned:
        fcntl.flock(shared, fcntl.LOCK_EX | fcntl.LOCK_NB); fcntl.flock(owned, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'admission_stop.json').exists(): raise ValueError('preserved_stop')
        for variant in ('supported', 'drop'):
            out = root / variant; verify(out)
            for case in read(out / 'cohort.json')['cases']:
                directory = out / 'executions' / case['id']
                if (directory / 'terminal.json').exists(): continue
                if (directory / 'reservation.json').exists(): raise ValueError('unknown_attempt_recover_without_retry')
                used = len(list((root / 'budget/reservations').glob('*.json')))
                if used >= 6: raise ValueError('budget_exhausted')
                receipt = {'case': case, 'campaign': str(out), 'variant': variant,
                    'started_unix': time.time(), 'protocol': binding(root / 'protocol.json')}
                write(root / 'budget/reservations' / f'{used:04d}.json', receipt)
                write(directory / 'reservation.json', receipt)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none',
                    VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/lvp_icd.x86_64.json', OMP_NUM_THREADS='1',
                    OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONPATH=str(ROOT),
                    RESIMPAIR_HEADLESS_RESERVATION=str(directory / 'reservation.json'))
                began = time.monotonic()
                with (directory / 'worker.log').open('x') as stream:
                    child = subprocess.Popen([sys.executable, '-u', '-m', 'resimpair_tasks.drop_release_pilot_v1',
                        'worker', '--root', str(out), '--case', case['id']], cwd=ROOT, env=env,
                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    write(directory / 'process.json', {'pid': child.pid,
                        'start_ticks': int(Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]),
                        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()})
                    try: rc = child.wait(timeout=360)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGTERM)
                        try: child.wait(timeout=8)
                        except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()
                        rc = 'timeout'
                check = admission_check(directory, rc)
                row = {'variant': variant, 'case': case['id'], 'physical_attempts': 1,
                       'returncode': rc, 'elapsed_seconds': time.monotonic() - began,
                       'admission': check, 'API_calls': 0, 'GPU_jobs': 0}
                write(directory / 'terminal.json', row); rows.append(row)
                print({'variant': variant, 'case': case['id'], 'seconds': row['elapsed_seconds'],
                       'failure': check['reported_failure']}, flush=True)
                if check['stop_admission']:
                    write(root / 'admission_stop.json', row)
                    return rows
    write(root / 'completion.json', {'actual_attempts': len(list((root / 'budget/reservations').glob('*.json'))),
                                   'new_rows': rows, 'API_calls': 0, 'GPU_jobs': 0})
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('command', choices=['prepare', 'run', 'worker'])
    p.add_argument('--root', type=Path, required=True); p.add_argument('--case')
    a = p.parse_args()
    if a.command == 'prepare': print(prepare(a.root, ROOT / 'resimpair_pickplace/drop_event_protocol_v1.py'))
    elif a.command == 'worker': print(worker(a.root, a.case))
    else: print(run(a.root))
