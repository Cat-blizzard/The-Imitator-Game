"""Six finite regression executions of the tighter path-proposal candidate."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from resimpair_grasp.common import ROOT, binding, read, resolve, write


def prepare(root, parent, independent):
    from resimpair_pickplace.runner import code
    from resimpair_pickplace.clearance_margin_v2 import VERSION
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import core
    root, parent, independent = map(lambda p: Path(p).resolve(), (root, parent, independent))
    if root.exists(): raise ValueError('preserve_existing_comparison')
    protocol = read(independent)
    for ref in protocol['frozen_evaluation_dependencies'].values(): resolve(ref)
    cases = [c for c in read(parent / 'candidate/cohort.json')['cases']
             if c['experiment_role'] == 'development_regression']
    if len(cases) != 6: raise ValueError('six_prespecified_regressions_required')
    old = read(parent / 'candidate/experiment.json')
    out = root / 'candidate'
    deps = code(); deps['resimpair_tasks/whole_path_campaign_v1.py'] = binding(ROOT / 'resimpair_tasks/whole_path_campaign_v1.py')
    write(root / 'budget/limits.json', {'physical_attempt_limit': 6, 'api_limit': 0, 'gpu_jobs': 0,
          'parallel_workers': 1, 'per_attempt_timeout_seconds': 240,
          'parent_comparison_budget': binding(parent / 'budget/limits.json'),
          'independent_protocol': binding(independent), 'this_turn_cumulative_scope_limit': 26,
          'allocation_this_turn': {'v1_comparison': 14, 'expanded_factory_integration': 6, 'v2_regression': 6}})
    write(out / 'experiment.json', {**old, 'code': deps, 'legacy_code': core.implementation(),
          'budget': str(root / 'budget'), 'candidate_motion_override': VERSION,
          'independent_protocol': binding(independent), 'parent_v1': binding(parent / 'candidate/experiment.json')})
    write(out / 'rules.json', read(parent / 'candidate/rules.json'))
    write(out / 'cohort.json', {'cases': cases, 'all_cases_previously_exposed': True})
    write(out / 'protocol.json', {'driver': binding(__file__), 'worker': binding(ROOT / 'resimpair_tasks/whole_path_campaign_v1.py'),
          'independent_protocol': binding(independent), 'budget': binding(root / 'budget/limits.json'),
          'change': '3mm_interarm_proposal_margin_only', 'original_acceptance_unchanged': True})
    write(out / 'frozen_inputs.json', {n: binding(out / (n + '.json')) for n in ('experiment', 'rules', 'cohort', 'protocol')})
    write(root / 'comparison_manifest.json', {'rows': [{'case_id': c['id'], 'arm': 'candidate_v2',
          'directory': str(out / 'executions' / c['id']), 'case': c} for c in cases]})
    return {'registered': 6, 'new_physical_executions': 0}


def run(root, limit):
    from resimpair_pickplace.runner import verify
    root = Path(root).resolve(); out = root / 'candidate'; verify(out)
    for ref in read(out / 'frozen_inputs.json').values(): resolve(ref)
    for key in ('driver', 'worker', 'independent_protocol', 'budget'): resolve(read(out / 'protocol.json')[key])
    limits = read(root / 'budget/limits.json')
    if not 1 <= limit <= 6: raise ValueError('bounded_six_case_limit')
    launched = []
    with (root / '.supervisor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for case in read(out / 'cohort.json')['cases']:
            folder = out / 'executions' / case['id']
            if (folder / 'terminal.json').exists(): continue
            if len(launched) >= limit: break
            if (folder / 'reservation.json').exists(): raise ValueError('recover_unknown_attempt_no_repeat')
            used = len(list((root / 'budget/reservations').glob('*.json')))
            if used >= limits['physical_attempt_limit']: raise ValueError('finite_budget_exhausted')
            reservation = {'case': case, 'started_unix': time.time(), 'api_calls': 0, 'gpu_jobs': 0,
                           'protocol': binding(out / 'protocol.json'), 'controller': 'whole_path_interarm_margin_v2'}
            write(root / 'budget/reservations' / f'{used:04d}.json', reservation); write(folder / 'reservation.json', reservation)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none',
                VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/lvp_icd.x86_64.json', OMP_NUM_THREADS='1',
                OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONPATH=str(ROOT),
                RESIMPAIR_HEADLESS_RESERVATION=str(folder / 'reservation.json'))
            start = time.monotonic()
            with (folder / 'worker.log').open('x') as stream:
                child = subprocess.Popen([sys.executable, '-u', '-m', 'resimpair_tasks.clearance_campaign_v2', 'worker',
                    '--root', str(out), '--case', case['id']], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                write(folder / 'process.json', {'pid': child.pid, 'started_unix': time.time()})
                try: rc = child.wait(timeout=limits['per_attempt_timeout_seconds'])
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    try: child.wait(timeout=8)
                    except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()
                    rc = 'timeout'
            terminal = {'case': case['id'], 'returncode': rc, 'physical_attempts': 1,
                        'elapsed_seconds': time.monotonic() - start, 'api_calls': 0, 'gpu_jobs': 0}
            write(folder / 'terminal.json', terminal); launched.append(terminal); print(terminal, flush=True)
            if rc != 0: break
    return {'new_attempts': len(launched)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'worker'])
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--parent', type=Path)
    parser.add_argument('--independent', type=Path); parser.add_argument('--limit', type=int, default=1); parser.add_argument('--case')
    a = parser.parse_args()
    if a.command == 'prepare': print(prepare(a.root, a.parent, a.independent))
    elif a.command == 'run': print(run(a.root, a.limit))
    else:
        from resimpair_pickplace.clearance_margin_v2 import install
        from .whole_path_campaign_v1 import worker
        with install(a.root): print(worker(a.root, a.case))
