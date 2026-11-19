"""Six frozen native reruns for the continuity/retiming candidate.

CPU prepare changes no scene, task, cached grasp, contact, or acceptance rule.
Run waits for a genuinely unused GPU for rendering; physics remains PhysX CPU.
All launches share one finite six-attempt ledger and preserve prior 679 attempts.
"""
import argparse
from contextlib import ExitStack
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from resimpair_grasp.common import ROOT, binding, read, resolve, write
from resimpair_pickplace.runner import code, verify
from corl2026_workshop.experiments.autonomous_pipeline_v4 import resources, core

VERSION = 'continuation_physics_regression_six_v1'


def prepare(root, inputs):
    root = Path(root).resolve()
    if root.exists():
        raise ValueError('existing_physics_campaign_resume_only')
    source = read(inputs)
    for name in ('candidate_verdict', 'candidate_controller', 'candidate_retimer', 'retained_guard'):
        resolve(source[name])
    cases, parent_specs = [], []
    for row in source['inputs']:
        parent = ROOT / row['parent_campaign']
        parent_specs.append(verify(parent))
        case = next(c for c in read(parent / 'cohort.json')['cases'] if c['id'] == row['case'])
        if not case['eligible']:
            raise ValueError('original_eligible_native_input_required')
        for name in ('scene', 'task'):
            core.checked(case[name])
        cases.append(case)
    if len(cases) != 6 or len({c['id'] for c in cases}) != 6:
        raise ValueError('exact_six_frozen_cases_required')
    baseline = ROOT / 'runs/batch500_20261027/asset_capacity_audit_20261028/FINAL_SUMMARY.json'
    spent = read(baseline)['physical_budget']
    prior = spent['prior_spent'] + spent['reserved']
    if prior != 679 or spent['unknown_reservations']:
        raise ValueError('expected_closed_prior_physics_ledger')
    budget = root / 'budget'
    write(budget / 'limits.json', {'physical_attempt_limit': 6, 'prior_spent': prior,
        'cumulative_physical_limit': prior+6, 'prior_evidence': binding(baseline), 'api_limit': 0})
    proposed = {**parent_specs[0], 'schema': VERSION, 'budget': str(budget),
        'gpu': None, 'gpu_uuid': None, 'timeout_seconds': 180,
        'code': code(), 'legacy_code': core.implementation(),
        'candidate_motion_override': 'bounded_continuation_motion_v3',
        'motion_policy': 'transport_fallback_v2', 'release_policy': 'legacy_v1',
        'source_inputs': binding(inputs), 'validation_scope': 'previously_exposed_failures',
        'default_promoted': False, 'contact_policy': parent_specs[0].get('contact_policy', 'native_default')}
    if len({p.get('contact_policy', 'native_default') for p in parent_specs}) != 1:
        raise ValueError('do_not_mix_contact_policies')
    write(root / 'proposed_experiment.json', proposed)
    rules_parent = ROOT / source['inputs'][0]['parent_campaign'] / 'rules.json'
    write(root / 'rules.json', read(rules_parent))
    write(root / 'cohort.json', {'cases': cases, 'frozen_inputs': binding(inputs),
        'same_physical_scenes_assets_initial_seeds_and_grasp_caches': True, 'no_physical_retries': True})
    write(root / 'protocol.json', {'schema': VERSION, 'prepared_unix': time.time(),
        'proposed_experiment': binding(root / 'proposed_experiment.json'),
        'cohort': binding(root / 'cohort.json'), 'rules': binding(root / 'rules.json'),
        'driver': binding(__file__), 'new_physical_attempt_limit': 6, 'api_calls': 0,
        'gpu_requirement': '120_seconds_continuously_idle_and_project_exclusive_lock',
        'max_wait_seconds': 7200, 'prior_physical_spent': prior,
        'counting': 'separate_regression_reruns_do_not_change_original500_denominator'})
    return {'prepared_cases': 6, 'physical_executions': 0, 'api_calls': 0}


def run(root):
    import json
    root = Path(root).resolve(); protocol = read(root / 'protocol.json')
    for key in ('proposed_experiment', 'cohort', 'rules', 'driver'):
        resolve(protocol[key])
    spec = read(root / 'proposed_experiment.json')
    for b in spec['code'].values():
        resolve(b)
    def status(value):
        tmp = root / ('status.'+str(os.getpid())+'.tmp')
        tmp.write_text(json.dumps({**value, 'updated_unix': time.time(), 'pid': os.getpid()}, indent=2)+'\n')
        tmp.replace(root / 'status.json')
    begin = time.time(); quiet = {}
    with (root / '.supervisor.lock').open('a+') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(root / 'supervisor_sessions' / (str(time.time_ns())+'.json'),
            {'pid': os.getpid(), 'started_unix': begin, 'protocol': binding(root / 'protocol.json')})
        for case in read(root / 'cohort.json')['cases']:
            folder = root / 'executions' / case['id']
            if (folder / 'terminal.json').exists():
                continue
            if (folder / 'reservation.json').exists():
                raise RuntimeError('unknown_existing_physics_recover_do_not_repeat')
            chosen = None
            while time.time()-begin < protocol['max_wait_seconds']:
                candidates = [read(root/'experiment.json')['gpu']] if (root/'experiment.json').exists() else range(4)
                for i in candidates:
                    snap = resources.gpu_snapshot(i)
                    try:
                        resources.check_idle(snap)
                    except RuntimeError:
                        quiet.pop(i, None); continue
                    quiet.setdefault(i, time.time())
                    if time.time()-quiet[i] >= 120:
                        chosen = i; break
                if chosen is not None:
                    break
                status({'status': 'waiting_unused_gpu', 'physical_executions': len(list(root.glob('executions/*/reservation.json'))),
                    'api_calls': 0})
                time.sleep(30)
            if chosen is None:
                status({'status': 'wait_deadline_no_resource', 'api_calls': 0}); return
            with ExitStack() as locks:
                for path in resources.lock_paths(chosen):
                    lock = locks.enter_context(path.open('r+'))
                    # The lock is shared with cuDex and all retained project runners.
                    fcntl.flock(lock, fcntl.LOCK_EX)
                snapshot = resources.gpu_snapshot(chosen); resources.check_idle(snapshot)
                actual = {**spec, 'gpu': chosen, 'gpu_uuid': snapshot['uuid']}
                write(root / 'experiment.json', actual)
                budget = Path(spec['budget'])
                used = len(list((budget / 'reservations').glob('*.json')))
                if used >= 6:
                    raise ValueError('shared_six_attempt_budget_exhausted')
                reservation = {'case': case, 'campaign': str(root), 'started_unix': time.time(), 'gpu': snapshot}
                write(budget / 'reservations' / f'{used:04d}.json', reservation)
                write(folder / 'reservation.json', reservation)
                env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                    PYTHONPATH=str(ROOT), RESIMPAIR_PICKPLACE_RESERVATION=str(folder / 'reservation.json'))
                env.pop('CUDA_VISIBLE_DEVICES', None)
                start = time.monotonic()
                with (folder / 'worker.log').open('x') as stream:
                    child = subprocess.Popen([sys.executable, '-u', '-m', 'resimpair_pickplace.continuation_motion_v3',
                        '--root', str(root), '--case', case['id']], cwd=ROOT, env=env, stdout=stream,
                        stderr=subprocess.STDOUT, start_new_session=True)
                    write(folder / 'process.json', {'pid': child.pid, 'started_unix': time.time()})
                    try:
                        result = child.wait(timeout=180)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=8)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL); child.wait()
                        result = 'timeout'
                write(folder / 'terminal.json', {'case': case['id'], 'returncode': result,
                    'elapsed_seconds': time.monotonic()-start, 'physical_attempts': 1, 'api_calls': 0})
                if result != 0:
                    status({'status': 'worker_error_needs_diagnosis', 'case': case['id'], 'api_calls': 0}); return
                from resimpair_pickplace.requested_goal_audit import evaluate
                audit = evaluate(folder, case)
                status({'status': 'running', 'last_case': case['id'], 'requested_goal_passed': audit['passed'],
                    'physical_executions': used+1, 'api_calls': 0})
        status({'status': 'terminal_requires_independent_review', 'physical_executions': 6,
            'api_calls': 0, 'default_promoted': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    p = parser.add_subparsers(dest='command', required=True)
    q = p.add_parser('prepare'); q.add_argument('--root', type=Path, required=True); q.add_argument('--inputs', type=Path, required=True)
    q = p.add_parser('run'); q.add_argument('--root', type=Path, required=True)
    a = parser.parse_args()
    print(prepare(a.root, a.inputs) if a.command == 'prepare' else run(a.root))
