"""Three fixed follow-ups, changing only the explicit drop opening schedule."""
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

VERSION = 'full_open_drop_followup_v2'
PARENT = ROOT / 'runs/drop_release_20261030_v1'
REGISTRY = ROOT / 'runtime/drop_release_registry_v1'
ALLOCATION = 'above_rim_full_open_20261030_v2_three_attempts'


def prepare(root):
    from resimpair_pickplace.runner import code
    root = Path(root).resolve()
    if root.exists(): raise ValueError('preserve_existing_followup')
    completion = read(PARENT / 'completion.json'); prior = read(PARENT / 'budget/limits.json')
    if completion['actual_attempts'] != 6 or prior['prior_actual_attempts'] != 132 or prior['cumulative_scope_limit'] != 304:
        raise ValueError('completed_six_attempt_parent_required')
    for variant in ('supported', 'drop'):
        for c in read(PARENT / variant / 'cohort.json')['cases']:
            if read(PARENT / variant / 'executions' / c['id'] / 'terminal.json')['physical_attempts'] != 1:
                raise ValueError('parent_attempt_unresolved')
    limits = {'schema': VERSION, 'physical_attempt_limit': 3, 'parallel_workers': 1,
        'timeout_seconds': 360, 'API_calls': 0, 'GPU_jobs': 0, 'retries': 0,
        'parent_budget': binding(PARENT / 'budget/limits.json'), 'parent_completion': binding(PARENT / 'completion.json'),
        'prior_actual_attempts': 138, 'cumulative_scope_limit': 307, 'allocation_id': ALLOCATION}
    REGISTRY.mkdir(parents=True, exist_ok=True)
    receipt = REGISTRY / (ALLOCATION + '.json')
    with (REGISTRY / '.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt.exists(): raise ValueError('allocation_registered_recover_original_root')
        write(root / 'budget/limits.json', limits)
        write(receipt, {'allocation_id': ALLOCATION, 'root': str(root), 'limit': 3,
                       'budget': binding(root / 'budget/limits.json')})
    source = PARENT / 'drop'; spec = copy.deepcopy(read(source / 'experiment.json'))
    implementation = spec['code']; implementation.update(code()); implementation[str(Path(__file__).relative_to(ROOT))] = binding(__file__)
    spec.update(schema=VERSION, code=implementation, budget=str(root / 'budget'),
                opening_adapter=binding(ROOT / 'resimpair_pickplace/full_open_drop_v2.py'))
    write(root / 'experiment.json', spec); write(root / 'rules.json', read(source / 'rules.json'))
    write(root / 'cohort.json', read(source / 'cohort.json'))
    write(root / 'protocol.json', {'schema': VERSION, 'driver': binding(__file__),
        'parent_protocol': binding(PARENT / 'protocol.json'), 'budget': binding(root / 'budget/limits.json'),
        'allocation_receipt': binding(receipt), 'same_inputs_and_event_rules': True,
        'changed_only': 'full_aperture_checked_in_preview_and_execution_then_commanded_from_first_open_step',
        'arm_grasp_choices_may_change_due_to_full_open_sweep': True,
        'API_calls': 0, 'GPU_jobs': 0})
    write(root / 'frozen_inputs.json', {n: binding(root / (n + '.json')) for n in ['experiment', 'rules', 'cohort', 'protocol']})
    return {'root': str(root), 'attempt_limit': 3}


def worker(root, case_id):
    from resimpair_runtime import configure
    configure()
    from resimpair_pickplace.full_open_drop_v2 import install
    from .drop_release_pilot_v1 import worker as retained
    c = next(c for c in read(Path(root) / 'cohort.json')['cases'] if c['id'] == case_id)
    with install(read(resolve(c['task']))): return retained(root, case_id)


def run(root):
    from resimpair_pickplace.runner import verify
    from .multitask_campaign_v3 import admission_check
    root = Path(root).resolve(); spec = verify(root); protocol = read(root / 'protocol.json')
    for ref in read(root / 'frozen_inputs.json').values(): resolve(ref)
    receipt = read(resolve(protocol['allocation_receipt']))
    if receipt['root'] != str(root) or receipt['limit'] != 3: raise ValueError('allocation_mismatch')
    lock_path = ROOT / 'runtime/multitask_campaign_registry_v3/.execution.lock'
    rows = []
    with lock_path.open('a+') as shared, (root / '.supervisor.lock').open('a+') as owned:
        fcntl.flock(shared, fcntl.LOCK_EX | fcntl.LOCK_NB); fcntl.flock(owned, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'admission_stop.json').exists(): raise ValueError('preserved_stop')
        for case in read(root / 'cohort.json')['cases']:
            out = root / 'executions' / case['id']
            if (out / 'terminal.json').exists(): continue
            if (out / 'reservation.json').exists(): raise ValueError('unknown_attempt_recover_without_retry')
            used = len(list((root / 'budget/reservations').glob('*.json')))
            if used >= 3: raise ValueError('budget_exhausted')
            reservation = {'case': case, 'campaign': str(root), 'protocol': binding(root / 'protocol.json'), 'started_unix': time.time()}
            write(root / 'budget/reservations' / f'{used:04d}.json', reservation); write(out / 'reservation.json', reservation)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none',
                VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/lvp_icd.x86_64.json', OMP_NUM_THREADS='1',
                OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONPATH=str(ROOT),
                RESIMPAIR_HEADLESS_RESERVATION=str(out / 'reservation.json'))
            start = time.monotonic()
            with (out / 'worker.log').open('x') as stream:
                child = subprocess.Popen([sys.executable, '-u', '-m', 'resimpair_tasks.drop_release_followup_v2',
                    'worker', '--root', str(root), '--case', case['id']], cwd=ROOT, env=env, stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True)
                write(out / 'process.json', {'pid': child.pid, 'start_ticks': int(Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]),
                    'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()})
                try: rc = child.wait(timeout=360)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    try: child.wait(timeout=8)
                    except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()
                    rc = 'timeout'
            check = admission_check(out, rc)
            row = {'variant': 'full_open_drop', 'case': case['id'], 'physical_attempts': 1, 'returncode': rc,
                'elapsed_seconds': time.monotonic() - start, 'admission': check, 'API_calls': 0, 'GPU_jobs': 0}
            write(out / 'terminal.json', row); rows.append(row)
            print({'case': case['id'], 'seconds': row['elapsed_seconds'], 'failure': check['reported_failure']}, flush=True)
            if check['stop_admission']:
                write(root / 'admission_stop.json', row); return rows
    write(root / 'completion.json', {'actual_attempts': len(list((root / 'budget/reservations').glob('*.json'))), 'new_rows': rows})
    return {'new_attempts': len(rows)}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('command', choices=['prepare', 'run', 'worker'])
    p.add_argument('--root', type=Path, required=True); p.add_argument('--case'); a = p.parse_args()
    print(prepare(a.root) if a.command == 'prepare' else run(a.root) if a.command == 'run' else worker(a.root, a.case))
