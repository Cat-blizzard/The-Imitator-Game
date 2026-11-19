"""Continue a frozen cuDex queue past individually evidenced asset failures.

No optimizer change: known invalid inputs are skipped before GPU admission;
known terminal numerical/empty-seed failures remain failed without resubmission.
Original manifests, qualification, budget, timeout, GPU locks and deadline bind
this opt-in driver. Unknown/infrastructure failures stop admission and drain.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np

from resimpair_grasp.common import ROOT, binding, read, resolve, write
from resimpair_grasp import cudex_finite_input_v4 as finite
from . import cudex_bulk_queue_v2 as queue
from . import cudex_multigpu_v2 as multi
from . import cudex_seed_repair_campaign_v4 as retained
from .cudex_quarantine_v3 import EMPTY_INPUT_ERROR

VERSION = 'cudex_evidenced_asset_skip_v5'
NONFINITE_ERRORS = (
    'ValueError: expected finite floating hand pose and two fingers',
    'ValueError: expected_256_finite_native_tcp_poses',
)


def _terminal_evidence(protocol, ref):
    if ref not in protocol['bulk_jobs']:
        raise ValueError('only_frozen_original_bulk_job_may_be_skipped')
    folder = resolve(ref).parent
    terminal = read(folder / 'terminal.json')
    optimizer = read(folder / 'optimizer_terminal.json')
    if (terminal.get('status') != 'failed' or terminal.get('failure_stage') != 'gpu'
            or terminal.get('returncode') != 1 or optimizer.get('returncode') != 1
            or optimizer.get('error') is not None
            or optimizer.get('complete_output_present', False) is not False
            or read(folder / 'reservation.json').get('job') != ref):
        raise ValueError('known_terminal_GPU_failure_required')
    if any((folder / name).exists() for name in
           ('optimizer_result.json', 'result.json', 'export_reservation.json', 'raw.npz')):
        raise ValueError('ambiguous_or_complete_output_requires_recovery')
    for name in ('gpu_process.json', 'gpu_worker_process.json', 'cpu_process.json', 'cpu_worker_process.json'):
        path = folder / name
        if path.exists() and queue.process_alive(read(path).get('identity')):
            raise RuntimeError('asset_process_still_alive_no_skip')
    return folder, {name: binding(folder / name) for name in
                    ('terminal.json', 'optimizer_terminal.json', 'reservation.json', 'worker.log')}


def classify_terminal(protocol, ref):
    """Only evidence-backed terminal asset errors; unknown errors raise."""
    folder, evidence = _terminal_evidence(protocol, ref)
    log = (folder / 'worker.log').read_text()
    tail = log.rstrip().splitlines()[-1] if log.strip() else ''
    extra = {}
    if (tail == EMPTY_INPUT_ERROR and '/curobo/opt/newton/newton_base.py' in log
            and 'q = q.view(self.n_problems, self.action_horizon * self.d_action)' in log):
        reason = 'empty_newton_input_runtime_error'
    elif tail in NONFINITE_ERRORS:
        extra['input_preflight'] = finite.inspect_job(resolve(ref))
        extra['solver_state'] = finite.inspect_unsorted(folder / 'solver_state_before_diagnostics.npz')
        if not extra['solver_state']['nonfinite_rows']:
            raise ValueError('finite_output_does_not_match_nonfinite_failure')
        reason = ('nonfinite_mesh_COM_propagated_to_optimizer'
                  if 'nonfinite_mesh_center_mass' in extra['input_preflight']['reasons']
                  else 'nonfinite_optimizer_output_with_finite_input_or_other_invalid_geometry')
    else:
        raise ValueError('unrecognized_terminal_error_requires_diagnosis')
    return {'schema': VERSION, 'job': ref, 'reason_code': reason, 'evidence': evidence, **extra,
            'disposition': 'failed_asset_skipped_no_retry', 'retry': False,
            'cache_admitted': False, 'physical_success': 'unmeasured',
            'original_attempt_consumed': True}


def validate_success(ref):
    result = queue.validate_result(resolve(ref))
    with np.load(resolve(result['raw']), allow_pickle=False) as raw:
        if raw['poses'].shape != (256, 4, 4) or not np.isfinite(raw['poses']).all():
            raise ValueError('completed_cache_has_invalid_raw_poses')
    return result


def dispatch(refs, execute, maximum, deadline, stop, finished, poll, classify):
    """No error-count quota for known bad assets; finite manifest bounds cost."""
    if type(maximum) is not int or maximum < 1:
        raise ValueError('positive_parallelism_required')
    pending = iter(refs); futures = {}; exhausted = False; error = None
    submitted = completed = skipped = 0
    with ThreadPoolExecutor(max_workers=maximum, thread_name_prefix='cudex-v5') as pool:
        try:
            while futures or not exhausted:
                while not exhausted and not stop.is_set() and len(futures) < maximum:
                    if time.time() >= deadline:
                        error = error or 'original_deadline_no_new_submission'
                        stop.set(); exhausted = True; break
                    ref = next(pending, None)
                    if ref is None: exhausted = True; break
                    futures[pool.submit(execute, ref)] = ref; submitted += 1
                if not futures: break
                done, _ = wait(futures, timeout=5, return_when=FIRST_COMPLETED)
                for future in done:
                    ref = futures.pop(future)
                    try:
                        result = future.result()
                        if result['status'] == 'failed':
                            classify(ref, result); skipped += 1
                        elif result['status'] == 'asset_skipped':
                            # This status is produced only by the internal preflight path.
                            classify(ref, result); skipped += 1
                        elif result['status'] != 'completed':
                            raise RuntimeError(result.get('error', 'unknown_submission_requires_recovery'))
                    except Exception as exc:
                        result = {'status': 'needs_recovery', 'error': f'{type(exc).__name__}: {exc}'}
                        error = error or result['error']; stop.set(); exhausted = True
                    completed += 1; finished(ref, result)
                poll()
        except BaseException:
            stop.set()
            raise
    return {'submitted': submitted, 'completed_futures': completed, 'known_asset_failures': skipped,
            'error': error, 'failed_results_remain_failed': True, 'automatic_retries': 0}


def prepare(parent_policy, output):
    parent_policy, output = Path(parent_policy).resolve(), Path(output).resolve()
    if output.exists(): raise ValueError('preserve_existing_skip_candidate')
    parent, original, _ = retained.validate(parent_policy)
    gate = retained.qualify(parent_policy)
    if gate['decision'] != 'proceed_continuation': raise ValueError('parent_qualification_required')
    counts = queue.status(parent['campaign_root'])
    if counts['active_processes'] or counts['unresolved_stages']:
        raise ValueError('recover_original_jobs_before_preparing')
    write(output / 'original_counts.json', counts)
    write(output / 'policy.json', {'schema': VERSION, 'driver': binding(__file__),
        'finite_input': binding(finite.__file__), 'parent_policy': binding(parent_policy),
        'parent_gate': binding(parent_policy.parent / 'qualification_gate.json'),
        'parent_stop': binding(parent_policy.parent / 'continuation_stop.json'),
        'campaign_root': parent['campaign_root'], 'protocol': parent['original_protocol'],
        'initial_counts': binding(output / 'original_counts.json'),
        'resource_policy': parent['continuation_resource_policy'],
        'deadline_unix': parent['deadline_unix'], 'gpu_job_limit': original['gpu_job_limit'],
        'lineage_gpu_job_limit': original['lineage_gpu_job_limit'],
        'continuation': parent['continuation'], 'deferred': parent['deferred_to_preserve_original_budget'],
        'diagnostic_jobs': parent['prior_diagnostic_jobs'] + [r['job'] for r in parent['qualification']],
        'no_retries': True, 'unknown_errors_stop': True,
        'known_asset_errors_continue': True, 'preflight_skips_consume_manifest_slot': True,
        'solver_changed': False, 'native_acceptance_changed': False,
        'user_request': '遇到这种错误就先记录然后跳过下一个', 'API_calls': 0})
    return {'policy': binding(output / 'policy.json'), 'new_GPU_jobs': 0}


def validate(policy_path):
    p = read(policy_path)
    if (p.get('schema') != VERSION or p['driver'] != binding(__file__)
            or p['finite_input'] != binding(finite.__file__)
            or p.get('no_retries') is not True or p.get('unknown_errors_stop') is not True
            or p.get('known_asset_errors_continue') is not True
            or p.get('preflight_skips_consume_manifest_slot') is not True
            or p.get('solver_changed') is not False or p.get('native_acceptance_changed') is not False):
        raise ValueError('invalid_asset_skip_policy')
    parent_path = resolve(p['parent_policy'])
    parent, original, _ = retained.validate(parent_path)
    for key in ('campaign_root', 'deadline_unix', 'gpu_job_limit', 'lineage_gpu_job_limit', 'continuation'):
        if p[key] != parent[key]: raise ValueError('original_scope_budget_or_deadline_changed:' + key)
    if (p['protocol'] != parent['original_protocol']
            or p['resource_policy'] != parent['continuation_resource_policy']
            or p['deferred'] != parent['deferred_to_preserve_original_budget']
            or p['diagnostic_jobs'] != parent['prior_diagnostic_jobs'] + [r['job'] for r in parent['qualification']]
            or resolve(p['parent_gate']) != parent_path.parent / 'qualification_gate.json'
            or read(resolve(p['parent_gate']))['decision'] != 'proceed_continuation'):
        raise ValueError('qualified_original_scope_required')
    resolve(p['parent_stop']); resolve(p['initial_counts'])
    resource = multi.validate_policy(p['campaign_root'], resolve(p['resource_policy']))
    return p, original, resource


def run(policy_path, max_new_assets=None, parallelism=None):
    policy_path = Path(policy_path).resolve(); p, original, resource = validate(policy_path)
    out, root = policy_path.parent, Path(p['campaign_root'])
    if max_new_assets is not None and (type(max_new_assets) is not int or max_new_assets < 1):
        raise ValueError('positive_session_asset_limit_required')
    maximum = resource['maximum_parallel_gpu_jobs'] if parallelism is None else parallelism
    if type(maximum) is not int or not 1 <= maximum <= resource['maximum_parallel_gpu_jobs']:
        raise ValueError('parallelism_may_only_reduce_original_limit')
    policy_ref = binding(policy_path); stop = threading.Event(); mutex = threading.RLock()
    active = {}; last = [0.]
    def emit(_root, state, **extra):
        with mutex:
            if extra.get('job'): active[extra['job']['path']] = {'state': state, **extra}
            queue.atomic(out / 'status.json', {'schema': VERSION, 'state': state,
                'identity': queue.process_identity(os.getpid()), 'updated_unix': time.time(),
                'active': list(active.values()), 'policy': policy_ref, **({} if extra.get('job') else extra)})
    adapter = multi.PoolResources(root, resolve(p['resource_policy']), emit, stop)
    def launch(job, stage, timeout, env, lock_fds=()):
        # Keep the qualified conditional seed initializer and native exporter.
        cmd = [sys.executable, '-u', '-m', retained.MODULE, 'execute',
               '--policy', str(resolve(p['parent_policy'])), '--job', str(job),
               '--stage', stage, '--timeout', str(timeout)]
        for fd in lock_fds: cmd += ['--lock-fd', str(fd)]
        with (job.parent / (stage + '_runner.log')).open('x') as stream:
            return subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True, pass_fds=tuple(lock_fds))
    def skip_path(ref): return out / 'skips' / (Path(ref['path']).parent.name + '.json')
    def save_skip(ref, entry):
        write(skip_path(ref), {'policy': policy_ref, 'entry': entry})
    def classify(ref, result):
        if result['status'] == 'asset_skipped':
            saved = read(skip_path(ref))
            if (saved['policy'] != policy_ref or saved['entry']['job'] != ref
                    or saved['entry']['input_preflight']['status'] != 'asset_invalid'):
                raise ValueError('unbound_preflight_skip')
            if saved['entry']['input_preflight'] != finite.inspect_job(resolve(ref)):
                raise ValueError('preflight_input_changed')
        else: save_skip(ref, classify_terminal(original, ref))
    def refresh(force=False):
        if force or time.monotonic() - last[0] >= 60:
            counts = queue.status(root)
            diagnostics = sum((resolve(ref).parent / 'reservation.json').exists() for ref in p['diagnostic_jobs'])
            cumulative = counts['gpu_reserved'] + diagnostics
            if cumulative > p['gpu_job_limit']: raise ValueError('original_cumulative_budget_exceeded')
            skips = [read(x)['entry'] for x in (out / 'skips').glob('*.json')]
            counts.update(diagnostic_jobs_reserved=diagnostics, cumulative_gpu_jobs=cumulative,
                candidate_asset_skips=len(skips),
                preflight_skips_without_GPU=sum(x.get('GPU_jobs') == 0 for x in skips),
                deferred_assets=len(p['deferred']))
            queue.atomic(out / 'counts.json', counts)
            # Emit the consumable index beside this candidate, leaving the
            # stopped predecessor's live status/index snapshots untouched.
            queue.update_candidate_index(out, original)
            last[0] = time.monotonic()
    session = out / 'sessions' / f'{time.time_ns()}_{os.getpid()}'
    def finished(ref, result):
        with mutex: active.pop(ref['path'], None)
        write(session / 'results' / (Path(ref['path']).parent.name + '.json'),
              {'job': ref, 'result': result, 'policy': policy_ref})
        emit(root, 'continuation_running')
    def execute(ref):
        job = resolve(ref)
        if not (job.parent / 'reservation.json').exists():
            preflight = finite.inspect_job(job)
            write(out / 'preflight' / (job.parent.name + '.json'), preflight)
            if preflight['status'] == 'asset_invalid':
                entry = {'schema': VERSION, 'job': ref, 'input_preflight': preflight,
                    'reason_code': 'invalid_numerical_solver_input', 'disposition': 'asset_skipped_before_GPU',
                    'cache_admitted': False, 'retry': False, 'GPU_jobs': 0,
                    'original_manifest_slot_consumed': True, 'physical_success': 'unmeasured'}
                save_skip(ref, entry)
                return {'status': 'asset_skipped', 'receipt': binding(skip_path(ref)), 'GPU_jobs': 0}
        result = queue._run_job(root, original, ref, p['deadline_unix'], {})
        if result['status'] == 'completed': validate_success(ref)
        return result
    previous = queue._acquire_gpu, queue._launch_stage, queue._state
    with (root / '.supervisor.lock').open('a+') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        counts = queue.status(root)
        if counts['active_processes'] or counts['unresolved_stages']:
            raise ValueError('recover_unknown_original_jobs_before_new_admission')
        write(session / 'start.json', {'policy': policy_ref, 'max_new_assets': max_new_assets,
            'parallelism': maximum, 'identity': queue.process_identity(os.getpid()),
            'original_deadline': p['deadline_unix'], 'prior_counts': counts})
        queue._acquire_gpu, queue._launch_stage, queue._state = adapter.acquire, launch, emit
        try:
            pending, recover = [], []
            for ref in p['continuation']:
                folder = resolve(ref).parent
                if (folder / 'terminal.json').exists():
                    result = read(folder / 'terminal.json')
                    if result['status'] == 'completed': validate_success(ref)
                    else: classify(ref, result)
                elif skip_path(ref).exists():
                    if (folder / 'reservation.json').exists(): raise ValueError('preflight_skip_has_reservation')
                    classify(ref, {'status': 'asset_skipped'})
                elif (folder / 'reservation.json').exists(): recover.append(ref)
                else: pending.append(ref)
            refresh(True)
            for ref in recover:
                result = execute(ref)
                if result['status'] != 'completed': classify(ref, result)
                finished(ref, result)
            selected = pending if max_new_assets is None else pending[:max_new_assets]
            write(session / 'admission.json', {'new_assets': selected, 'recovered': recover,
                'remaining_not_admitted': len(pending) - len(selected),
                'same_manifest_only': True, 'no_budget_increase': True})
            result = dispatch(selected, execute, maximum, p['deadline_unix'], stop, finished, refresh, classify)
            refresh(True)
            result['remaining_not_admitted_this_session'] = len(pending) - len(selected)
            result['session_limit_reached'] = len(selected) < len(pending)
            write(session / 'terminal.json', result)
            emit(root, 'stopped_for_recovery' if result['error'] else 'bounded_session_finished', result=result)
            return result
        except BaseException as exc:
            stop.set(); refresh(True)
            emit(root, 'stopped_for_recovery', error=f'{type(exc).__name__}: {exc}')
            raise
        finally: queue._acquire_gpu, queue._launch_stage, queue._state = previous


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run'])
    parser.add_argument('--parent-policy', type=Path); parser.add_argument('--output', type=Path)
    parser.add_argument('--policy', type=Path); parser.add_argument('--max-new-assets', type=int)
    parser.add_argument('--parallelism', type=int)
    a = parser.parse_args()
    print(prepare(a.parent_policy, a.output) if a.command == 'prepare'
          else run(a.policy, a.max_new_assets, a.parallelism))
