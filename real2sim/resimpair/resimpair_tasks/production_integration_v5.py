"""Finite CPU integration campaigns sharing one non-resettable allocation.

Uses the existing measured-state worker and frozen independent physical rules.
The first cohort is an exact paired prefix regression; later metric cohorts
share the same budget. Asset preparation, renders and API calls have separate
ledgers and cannot be counted as successful physical trajectories.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from resimpair_grasp.common import ROOT, binding, read, resolve, write

VERSION = 'metric_asset_production_integration_v5'
PARENT = ROOT / 'runs/drop_release_full_open_20261030_v2'
REGISTRY = ROOT / 'runtime/production_integration_v1'
ALLOCATION = 'full_asset_metric_integration_20261030_v1'


def register(root):
    root = Path(root).resolve()
    if root.exists():
        raise ValueError('existing_integration_campaign_use_resume')
    previous = read(PARENT / 'budget/limits.json')
    prior_used = len(list((PARENT / 'budget/reservations').glob('*.json')))
    if prior_used != 3 or previous['cumulative_scope_limit'] != 307:
        raise ValueError('reconcile_completed_parent_before_new_allocation')
    for case in read(PARENT / 'cohort.json')['cases']:
        if not (PARENT / 'executions' / case['id'] / 'terminal.json').exists():
            raise ValueError('parent_attempt_unknown')
    limits = {'schema': VERSION, 'physical_attempt_limit': 128, 'parallel_workers': 1,
        'timeout_seconds': 480, 'retries_per_request': 0, 'API_calls': 0, 'GPU_jobs': 0,
        'parent_budget': binding(PARENT / 'budget/limits.json'),
        'prior_actual_attempts': previous['prior_actual_attempts'] + prior_used,
        'prior_scope_limit': 307, 'cumulative_scope_limit': 435,
        'allocation_id': ALLOCATION, 'unspent_prior_allowance_reassigned': False}
    REGISTRY.mkdir(parents=True, exist_ok=True)
    receipt = REGISTRY / (ALLOCATION + '.json')
    with (REGISTRY / '.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt.exists():
            raise ValueError('allocation_registered_recover_original_root')
        write(root / 'budget/limits.json', limits)
        write(receipt, {'root': str(root), 'budget': binding(root / 'budget/limits.json'),
                        'allocation_id': ALLOCATION, 'limit': 128, 'registered_unix': time.time()})
    write(root / 'allocation.json', {'registration': binding(receipt), 'initial_independent_protocol':
        binding(ROOT / 'validation/paired_assets_20261030/independent/PROTOCOL.json'),
        'scope': 'paired prefix regression, metric asset and multiobject integration; all attempts retained'})
    return limits


def prepare_cohort(root, name, cases, *, prefix_reuse=True, full_open=True):
    root = Path(root).resolve(); output = root / name
    if output.exists():
        raise ValueError('preserve_existing_cohort')
    limits = read(root / 'budget/limits.json')
    if not cases or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('finite_unique_cases_required')
    allocated = sum(len(read(p)['cases']) for p in root.glob('*/cohort.json'))
    if allocated + len(cases) > limits['physical_attempt_limit']:
        raise ValueError('finite_cohort_count_exceeds_campaign_limit')
    spec = deepcopy(read(PARENT / 'experiment.json'))
    spec.update(schema=VERSION, budget=str(root / 'budget'), timeout_seconds=480,
                integration_prefix_reuse=bool(prefix_reuse), integration_full_open=bool(full_open),
                parent_experiment=binding(PARENT / 'experiment.json'))
    spec['code'].update({p: binding(ROOT / p) for p in (
        'resimpair_tasks/production_integration_v5.py', 'resimpair_pickplace/certified_prefix_v1.py',
        'resimpair_pickplace/metric_program_v1.py', 'resimpair_tasks/metric_task_factory_v2.py',
        'resimpair_pickplace/metric_body_evidence_v1.py', 'resimpair_pickplace/checked_withdrawal_v3.py',
        'resimpair_pickplace/contact_latched_closure_v1.py')})
    for c in cases:
        resolve(c['scene']); resolve(c['task'])
    write(output / 'experiment.json', spec)
    write(output / 'rules.json', read(PARENT / 'rules.json'))
    write(output / 'cohort.json', {'cases': cases, 'selected_before_execution': True,
                                 'failed_cases_replaced': False})
    write(output / 'protocol.json', {'schema': VERSION, 'driver': binding(__file__),
        'campaign_root': str(root), 'allocation': binding(root / 'allocation.json'),
        'budget': binding(root / 'budget/limits.json'), 'original_rules': binding(PARENT / 'rules.json'),
        'event_builder': binding(ROOT / 'resimpair_pickplace/drop_event_protocol_v1.py'),
        'prefix_reuse': prefix_reuse, 'release_policy': 'supported_with_native_geometry_checked_aperture_and_withdrawal',
        'maximum_unique_operations': 8, 'closure_policy': 'cached_width_contact_latched_closure_v1', 'API_calls': 0, 'GPU_jobs': 0})
    write(output / 'frozen_inputs.json', {k: binding(output / (k + '.json'))
        for k in ('experiment', 'cohort', 'rules', 'protocol')})
    return {'root': str(output), 'requests': len(cases)}


def prepare_regression(root):
    limits = register(root)
    cases = read(PARENT / 'cohort.json')['cases']
    rows = [prepare_cohort(root, name, deepcopy(cases), prefix_reuse=enabled, full_open=True)
            for name, enabled in [('prefix_baseline', False), ('prefix_candidate', True)]]
    return {'allocation': limits, 'cohorts': rows}


def worker(root, case_id):
    from resimpair_runtime import configure
    configure()
    from resimpair_pickplace import above_rim_release_v1 as drop, full_open_drop_v2 as opening
    from resimpair_pickplace import certified_prefix_v1 as prefix, drop_event_protocol_v1 as events
    from resimpair_pickplace import metric_program_v1 as program
    from resimpair_pickplace import metric_body_evidence_v1 as body_evidence
    from resimpair_pickplace import checked_withdrawal_v3 as withdrawal
    from resimpair_pickplace import contact_latched_closure_v1 as closure, rolling_pickplace_v3 as rolling
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import planning_protocols, simulation
    from .multitask_worker_v3 import worker as original_worker
    root = Path(root); spec = read(root / 'experiment.json')
    c = next(c for c in read(root / 'cohort.json')['cases'] if c['id'] == case_id)
    task = read(resolve(c['task'])); original_events = planning_protocols.pick_place
    planning_protocols.pick_place = events.pick_place
    original_execute = rolling.execute
    def contact_execute(rec, active_task, **kwargs):
        with closure.install(rec, active_task):
            return original_execute(rec, active_task, **kwargs)
    rolling.execute = contact_execute
    try:
        with ExitStack() as stack:
            stack.enter_context(body_evidence.install())
            stack.enter_context(withdrawal.install())
            stack.enter_context(drop.install(task))
            if spec['integration_prefix_reuse']:
                stack.enter_context(prefix.install(task))
            stack.enter_context(program.install())
            return original_worker(root, case_id)
    finally:
        rolling.execute = original_execute
        planning_protocols.pick_place = original_events
        folder = root / 'executions' / case_id
        if (folder / 'effective_scene.json').exists() and (folder / 'effective_task.json').exists():
            scene = read(folder / 'effective_scene.json'); executed = read(folder / 'effective_task.json')
            ids = [o['id'] for o in scene['objects']]
            supports = ids + [s['id'] for s in scene['supports']] + ['table']
            write(folder / 'retained_event_protocol.json', original_events(scene, executed, ids, supports,
                                                                         simulation.targets(scene, executed)))


def run(output, limit=None):
    from resimpair_pickplace.runner import verify
    from .multitask_campaign_v3 import admission_check
    output = Path(output).resolve(); verify(output)
    for ref in read(output / 'frozen_inputs.json').values(): resolve(ref)
    protocol = read(output / 'protocol.json'); campaign = Path(protocol['campaign_root'])
    limits = read(resolve(protocol['budget'])); allocation = read(resolve(protocol['allocation']))
    receipt = read(resolve(allocation['registration']))
    if receipt['root'] != str(campaign) or receipt['limit'] != limits['physical_attempt_limit']:
        raise ValueError('campaign_allocation_binding_mismatch')
    lock_path = ROOT / 'runtime/multitask_campaign_registry_v3/.execution.lock'
    rows = []
    with lock_path.open('a+') as shared, (campaign / '.supervisor.lock').open('a+') as owned:
        fcntl.flock(shared, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(owned, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (campaign / 'admission_stop.json').exists(): raise ValueError('preserved_admission_stop')
        for case in read(output / 'cohort.json')['cases']:
            if limit is not None and len(rows) >= limit: break
            directory = output / 'executions' / case['id']
            if (directory / 'terminal.json').exists(): continue
            if (directory / 'reservation.json').exists(): raise ValueError('unknown_attempt_recover_no_retry')
            used = len(list((campaign / 'budget/reservations').glob('*.json')))
            if used >= limits['physical_attempt_limit']: raise ValueError('physical_budget_exhausted')
            reservation = {'case': case, 'cohort': output.name, 'campaign': str(output),
                'protocol': binding(output / 'protocol.json'), 'started_unix': time.time()}
            write(campaign / 'budget/reservations' / f'{used:04d}.json', reservation)
            write(directory / 'reservation.json', reservation)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none',
                VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/lvp_icd.x86_64.json', OMP_NUM_THREADS='1',
                OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONPATH=str(ROOT),
                RESIMPAIR_HEADLESS_RESERVATION=str(directory / 'reservation.json'))
            start = time.monotonic()
            with (directory / 'worker.log').open('x') as stream:
                child = subprocess.Popen([sys.executable, '-u', '-m', 'resimpair_tasks.production_integration_v5',
                    'worker', '--root', str(output), '--case', case['id']], cwd=ROOT, env=env,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                write(directory / 'process.json', {'pid': child.pid,
                    'start_ticks': int(Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]),
                    'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()})
                try: rc = child.wait(timeout=limits['timeout_seconds'])
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    try: child.wait(timeout=8)
                    except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()
                    rc = 'timeout'
            check = admission_check(directory, rc)
            result = {'case': case['id'], 'cohort': output.name, 'physical_attempts': 1,
                'returncode': rc, 'elapsed_seconds': time.monotonic() - start, 'admission': check,
                'API_calls': 0, 'GPU_jobs': 0}
            write(directory / 'terminal.json', result); rows.append(result)
            print({'cohort': output.name, 'case': case['id'], 'seconds': round(result['elapsed_seconds'], 3),
                   'failure': check['reported_failure']}, flush=True)
            if check['stop_admission']:
                write(campaign / 'admission_stop.json', result); break
    cases = read(output / 'cohort.json')['cases']
    if all((output / 'executions' / c['id'] / 'terminal.json').exists() for c in cases):
        write(output / 'completion.json', {'requests': len(cases), 'all_attempts_terminal': True,
            'campaign_attempts': len(list((campaign / 'budget/reservations').glob('*.json')))})
    return {'new_attempts': len(rows)}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare-regression', 'run', 'worker'])
    p.add_argument('--root', type=Path, required=True); p.add_argument('--case'); p.add_argument('--limit', type=int)
    a = p.parse_args()
    print(prepare_regression(a.root) if a.command == 'prepare-regression' else
          run(a.root, a.limit) if a.command == 'run' else worker(a.root, a.case))
