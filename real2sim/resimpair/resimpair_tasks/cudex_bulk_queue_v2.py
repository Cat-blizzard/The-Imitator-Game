"""Finite, restart-safe cuDex generation for the complete native-ready index.

Eight fixed qualification jobs precede the full library. Four candidate jobs
also serve the bulk catalog, so an N-asset campaign reserves at most N+4 GPU
attempts. GPU optimization and CPU native qualification are separate durable
stages. Unknown attempts are recovered, never silently resubmitted. The native
cache is an offline geometric result, not a physical-success certificate.
"""
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

import yaml

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write
from resimpair_grasp.cudex_native import UPSTREAM
from corl2026_workshop.experiments.autonomous_pipeline_v4 import resources

VERSION = 'idle_only_cudex_bulk_queue_v2'
WORKER_MODULE = 'resimpair_grasp.cudex_constraint_v3'
QUEUE_MODULE = 'resimpair_tasks.cudex_bulk_queue_v2'


def atomic(path, value):
    """Only derived status/index files are mutable; evidence uses write()."""
    path = Path(path)
    temp = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def process_identity(pid):
    """PID alone is insufficient: a recycled PID must never justify waiting/killing."""
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return {'pid': int(pid), 'start_ticks': int(fields[19]),
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    except (FileNotFoundError, ProcessLookupError):
        return None


def process_alive(identity):
    return bool(identity and process_identity(identity['pid']) == identity)


def _configuration_files(manip_paths):
    """Freeze the used native configs and upstream task config references."""
    found = set()
    pending = list(map(Path, manip_paths))
    while pending:
        path = pending.pop().resolve()
        if path in found:
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
        found.add(path)
        if path.suffix not in ('.yml', '.yaml'):
            continue
        data = yaml.safe_load(path.read_text())
        def visit(value):
            if isinstance(value, dict):
                for item in value.values():
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)
            elif isinstance(value, str) and '{$' not in value:
                candidate = Path(value)
                if candidate.suffix not in ('.yml', '.yaml', '.urdf', '.xml'):
                    return
                choices = [candidate] if candidate.is_absolute() else [
                    path.parent / candidate,
                    UPSTREAM / 'src/curobo/content/configs/task' / candidate,
                    UPSTREAM / 'src/curobo/content/configs/robot' / candidate,
                    UPSTREAM / 'src/curobo/content/configs/world' / candidate]
                matches = [item for item in choices if item.is_file()]
                if matches:
                    pending.append(matches[0])
                elif candidate.is_absolute():
                    raise FileNotFoundError(candidate)
        visit(data)
    return sorted(found)


def implementation_files():
    import xml.etree.ElementTree as ET
    panda = ROOT / 'PolicyGen/mani_skill/assets/robots/panda'
    urdf = panda / 'panda_v3.urdf'
    geometry = [urdf] + [panda / item.attrib['filename']
        for item in ET.parse(urdf).findall('.//collision/geometry/mesh')]
    return sorted(set(geometry + [
        Path(__file__), ROOT / 'resimpair_runtime.py',
        ROOT / 'resimpair_grasp/cudex_constraint_v3.py',
        ROOT / 'resimpair_grasp/common.py', ROOT / 'resimpair_grasp/cudex_native.py',
        ROOT / 'resimpair_grasp/cartesian_clearance.py', ROOT / 'resimpair_grasp/grasps.py',
        ROOT / 'resimpair_pickplace/closure_quality.py', ROOT / 'resimpair_benchmark/proposals.py',
        ROOT / 'resimpair_tabletop/geometry.py',
        ROOT / 'data_generation_methods/human_to_robot_pipeline/strict_motion.py',
        Path(resources.__file__),
        UPSTREAM / 'src/curobo/geom/sdf/world.py',
        UPSTREAM / 'src/curobo/geom/sdf/world_mesh.py',
        UPSTREAM / 'src/curobo/rollout/cost/primitive_collision_cost.py',
        UPSTREAM / 'src/curobo/wrap/reacher/grasp_solver.py',
        UPSTREAM / 'src/curobo/curobolib/cpp/sphere_obb_kernel.cu',
    ]))


def prepare(root, index, original, candidate, qualification_inputs, gate_validator,
            campaign_seconds=7 * 86400, code_paths=None, registry=None):
    """Freeze every canonical input; existing roots cannot create fresh budgets."""
    root = Path(root).resolve()
    if root.exists():
        raise ValueError('existing_bulk_campaign_resume_only')
    if not 0 < campaign_seconds <= 7 * 86400:
        raise ValueError('campaign_deadline_must_be_finite_at_most_seven_days')
    entries = read(index)['entries']
    frozen = read(qualification_inputs)['entries']
    by_key = {e['content_key']: e for e in entries}
    if len(entries) != len(by_key) or len({e['source_id'] for e in entries}) != len(entries):
        raise ValueError('canonical_index_contains_duplicates')
    if len(frozen) != 4 or len({e['content_key'] for e in frozen}) != 4:
        raise ValueError('exactly_four_fixed_qualification_assets_required')
    for e in frozen:
        if e['content_key'] not in by_key or e['asset'] != by_key[e['content_key']]['asset']:
            raise ValueError('fixed_qualification_asset_changed_or_absent')
    code = [binding(p) for p in (implementation_files() if code_paths is None else code_paths)]
    configuration = [binding(p) for p in _configuration_files([original, candidate])]
    for e in entries:
        resolve(e['asset'])
    gate = binding(gate_validator)
    registry = Path(registry or ROOT / 'runtime/cudex_bulk_registry_v2').resolve()
    registry.mkdir(parents=True, exist_ok=True)
    # Scheduler/report changes cannot turn the same pose experiment into a
    # fresh allowance. Different solve inputs require an explicit new recipe.
    recipe = digest({'canonical_assets': sorted((e['content_key'], e['asset']['sha256']) for e in entries),
        'seed': 20261028, 'seed_count': 256,
        'configuration_sha256': sorted(r['sha256'] for r in configuration),
        'generator_and_qualifier': [r for r in code if not r['path'].endswith('cudex_bulk_queue_v2.py')
                                  and not r['path'].endswith('resources.py')]})
    registry_record = registry / (recipe + '.json')
    with (registry / '.registry.lock').open('a+') as registry_lock:
        fcntl.flock(registry_lock, fcntl.LOCK_EX)
        if registry_record.exists():
            raise ValueError('existing_recipe_campaign_resume_at:' + read(registry_record)['root'])
        root.mkdir(parents=True)
        write(registry_record, {'schema': VERSION, 'recipe': recipe, 'root': str(root),
                               'reserved_unix': time.time(), 'gpu_jobs_started_at_prepare': 0})
    write(root / 'frozen_inputs.json', {'index': binding(index), 'entries': entries,
        'qualification_inputs': binding(qualification_inputs),
        'qualification_entries': frozen, 'native_geometry_only': True})
    qualification_jobs, bulk_jobs, candidates = [], [], {}
    for i, entry in enumerate(frozen):
        for variant, manip in [('retained_signed_constraint', original),
                               ('nonnegative_constraint', candidate)]:
            folder = root / 'qualification' / f'{i:02d}_{variant}'
            job = folder / 'job.json'
            write(job, {'asset': entry['asset'], 'manip': binding(manip),
                'source_id': entry['source_id'], 'content_key': entry['content_key'],
                'seed': 20261028, 'output': str(folder), 'variant': variant,
                'phase': 'qualification', 'source_aliases': entry.get('source_aliases', [])})
            qualification_jobs.append(binding(job))
            if variant == 'nonnegative_constraint':
                candidates[entry['content_key']] = binding(job)
    for entry in sorted(entries, key=lambda e: (e['source_id'].split(':')[0], e['content_key'])):
        if entry['content_key'] in candidates:
            continue
        folder = root / 'bulk' / entry['content_key']
        job = folder / 'job.json'
        write(job, {'asset': entry['asset'], 'manip': binding(candidate),
            'source_id': entry['source_id'], 'content_key': entry['content_key'],
            'seed': 20261028, 'output': str(folder), 'variant': 'nonnegative_constraint',
            'phase': 'bulk', 'source_aliases': entry.get('source_aliases', [])})
        ref = binding(job)
        bulk_jobs.append(ref)
        candidates[entry['content_key']] = ref
    manifest = [{'source_id': e['source_id'], 'content_key': e['content_key'],
                 'source_aliases': e.get('source_aliases', []), 'asset': e['asset'],
                 'geometry_evidence': e.get('geometry_evidence'),
                 'job': candidates[e['content_key']], 'physical_status': 'unmeasured'}
                for e in entries]
    write(root / 'candidate_manifest.json', {'entries': manifest,
        'qualification_candidates_reused': 4, 'canonical_assets': len(manifest),
        'source_ids_including_aliases': len(manifest) + sum(len(e['source_aliases']) for e in manifest)})
    protocol = {'schema': VERSION, 'created_unix': time.time(),
        'frozen_inputs': binding(root / 'frozen_inputs.json'),
        'candidate_manifest': binding(root / 'candidate_manifest.json'),
        'qualification_jobs': qualification_jobs, 'bulk_jobs': bulk_jobs,
        'gpu_job_limit': len(qualification_jobs) + len(bulk_jobs),
        'canonical_assets': len(entries), 'qualification_candidates_reused': 4,
        'maximum_parallel_gpu_jobs': 1, 'maximum_parallel_cpu_exports': 1,
        'per_gpu_job_timeout_seconds': 180, 'per_cpu_export_timeout_seconds': 180,
        'maximum_campaign_seconds': campaign_seconds, 'gpu_indices': [0, 1, 2, 3],
        'quiet_seconds': 120, 'poll_seconds': 30, 'min_free_gib': 10,
        'gpu_ownership': 'no_compute_owner_memory_le_512MiB_utilization_le_10_then_project_lock',
        'quiet_scope': 'initial_external_handoff_120s; own_serial_jobs_reuse_after_fresh_idle_check; observed_external_owner_resets_clock',
        'gpu_locks_inherited_by_worker': True, 'api_calls': 0,
        'worker_module': WORKER_MODULE, 'code': code, 'configuration': configuration,
        'registry': binding(registry_record), 'recipe': recipe,
        'prior_lineage_gpu_jobs': 142, 'lineage_gpu_job_limit': 142 + len(qualification_jobs) + len(bulk_jobs),
        'prior_lineage_scope': 'retained_expansion141_plus_failed_alignment1_not_all_project_history',
        'gate_validator': gate, 'gate_timeout_seconds': 120,
        'unknown_never_resubmitted': True, 'physical_success': 'unmeasured',
        'default_promoted': False}
    write(root / 'protocol.json', protocol)
    return {'root': str(root), 'canonical_assets': len(entries),
            'gpu_job_limit': protocol['gpu_job_limit'], 'prepared_qualification_jobs': 8,
            'prepared_bulk_jobs': len(bulk_jobs), 'gpu_jobs_started': 0, 'api_calls': 0}


def validate_optimizer(job_path):
    job = read(job_path)
    result = read(Path(job['output']) / 'optimizer_result.json')
    if (result.get('status') != 'optimizer_complete'
            or result.get('job') != binding(job_path) or result.get('asset') != job['asset']
            or result.get('seed_count') != 256
            or result.get('nonnegative_semantics_valid') is not True):
        raise ValueError('optimizer_output_integrity_failed')
    resolve(result['raw']); resolve(result['components'])
    return result


def validate_result(job_path):
    job = read(job_path)
    result = read(Path(job['output']) / 'result.json')
    validate_optimizer(job_path)
    if (result.get('status') != 'completed' or result.get('job') != binding(job_path)
            or result.get('asset') != job['asset'] or result.get('seed_count') != 256
            or result.get('nonnegative_semantics_valid') is not True
            or result.get('physical_success') != 'unmeasured'):
        raise ValueError('final_output_integrity_failed')
    resolve(result['raw']); resolve(result['decisions'])
    cache = read(resolve(result['cache']))
    if (cache.get('schema') != 'resimpair_mesh_grasp_cache_v1'
            or cache['signature']['descriptor'] != job['asset']
            or result.get('retained') != len(cache['candidates'])):
        raise ValueError('cache_integrity_failed')
    return result


def validate_gate(root, protocol):
    gate = read(root / 'independent_gate.json')
    expected = [binding(resolve(j).parent / 'result.json') for j in protocol['qualification_jobs']]
    if gate.get('protocol') != binding(root / 'protocol.json') or gate.get('qualification_results') != expected:
        raise ValueError('independent_gate_does_not_bind_exact_qualification_outputs')
    if gate.get('decision') != 'proceed_bulk':
        raise ValueError('independent_gate_did_not_approve_bulk')
    if not gate.get('reviewer'):
        raise ValueError('independent_reviewer_identity_missing')
    return gate


def _verify_frozen(protocol):
    for ref in protocol['code'] + protocol['configuration'] + [protocol['gate_validator']]:
        resolve(ref)


def status(root, write_status=False):
    root = Path(root).resolve()
    protocol = read(root / 'protocol.json')
    refs = protocol['qualification_jobs'] + protocol['bulk_jobs']
    counts = {'gpu_reserved': 0, 'gpu_completed': 0, 'terminal_jobs': 0,
              'completed_jobs': 0, 'failed_jobs': 0, 'nonempty_job_caches': 0,
              'candidate_caches': 0, 'candidate_nonempty_caches': 0,
              'candidate_poses': 0}
    unknown, active = [], []
    for ref in refs:
        folder = (ROOT / ref['path']).parent if not Path(ref['path']).is_absolute() else Path(ref['path']).parent
        if (folder / 'reservation.json').exists():
            counts['gpu_reserved'] += 1
        if (folder / 'optimizer_terminal.json').exists():
            counts['gpu_completed'] += 1
        if (folder / 'terminal.json').exists():
            counts['terminal_jobs'] += 1
            terminal = read(folder / 'terminal.json')
            if terminal['status'] == 'completed':
                counts['completed_jobs'] += 1
                counts['nonempty_job_caches'] += int(read(folder / 'result.json')['retained'] > 0)
            else:
                counts['failed_jobs'] += 1
            continue
        for stage, reservation in [('gpu', 'reservation.json'), ('cpu', 'export_reservation.json')]:
            if not (folder / reservation).exists() or (stage == 'gpu' and (folder / 'optimizer_terminal.json').exists()):
                continue
            process = folder / (stage + '_process.json')
            record = read(process) if process.exists() else {}
            entry = {'job': ref, 'stage': stage, 'identity': record.get('identity')}
            if process_alive(record.get('identity')):
                active.append(entry)
            else:
                expected = folder / ('optimizer_result.json' if stage == 'gpu' else 'result.json')
                entry['complete_output_present_recovery_required'] = expected.exists()
                unknown.append(entry)
    for row in read(resolve(protocol['candidate_manifest']))['entries']:
        job_path = Path(row['job']['path'])
        folder = (ROOT / job_path).parent if not job_path.is_absolute() else job_path.parent
        if (folder / 'terminal.json').exists() and read(folder / 'terminal.json')['status'] == 'completed':
            value = read(folder / 'result.json')
            counts['candidate_caches'] += 1
            counts['candidate_nonempty_caches'] += int(value['retained'] > 0)
            counts['candidate_poses'] += value['retained']
    report = {'schema': VERSION, 'updated_unix': time.time(), **counts,
        'canonical_assets': protocol['canonical_assets'], 'gpu_job_limit': protocol['gpu_job_limit'],
        'active_processes': active, 'unresolved_stages': unknown,
        'physical_success': 'unmeasured', 'api_calls': 0, 'default_promoted': False}
    if write_status:
        atomic(root / 'counts.json', report)
    return report


def _state(root, state, **extra):
    atomic(root / 'status.json', {'schema': VERSION, 'status': state,
        'pid': os.getpid(), 'identity': process_identity(os.getpid()),
        'updated_unix': time.time(), 'api_calls': 0, **extra})


def _environment(job_path, gpu_uuid=None):
    deps = ROOT / 'runtime/cudex_deps'
    return dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid or '',
        RESIMPAIR_CUDEX_ALIGNMENT_JOB=str(job_path), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
        PYTHONPATH=os.pathsep.join(map(str, [ROOT, deps, deps / 'cmeel.prefix/lib/python3.12/site-packages',
            UPSTREAM / 'src', UPSTREAM / 'src/curobo/geom/cpp'])),
        LD_LIBRARY_PATH=str(deps / 'cmeel.prefix/lib') + os.pathsep + os.environ.get('LD_LIBRARY_PATH', ''))


def execute_stage(job_path, stage, timeout, lock_fds=()):
    """Detached durable runner: survives supervisor exit and keeps GPU locks."""
    job_path = Path(job_path).resolve()
    out = job_path.parent
    process_path = out / (stage + '_process.json')
    if process_path.exists():
        raise ValueError('stage_already_launched_no_implicit_retry')
    started = time.time()
    write(process_path, {'identity': process_identity(os.getpid()), 'started_unix': started,
        'job': binding(job_path), 'stage': stage})
    command = 'worker' if stage == 'gpu' else 'export'
    log_name = 'worker.log' if stage == 'gpu' else 'export.log'
    returncode, error = None, None
    try:
        with (out / log_name).open('x') as stream:
            child = subprocess.Popen([sys.executable, '-u', '-m', WORKER_MODULE,
                command, '--job', str(job_path)], cwd=ROOT, env=os.environ.copy(),
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                pass_fds=tuple(lock_fds))
            write(out / (stage + '_worker_process.json'), {
                'identity': process_identity(child.pid), 'started_unix': time.time()})
            try:
                returncode = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # This runner owns exactly this process group, never an unrelated GPU owner.
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
                returncode = 'timeout'
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        for fd in lock_fds:
            os.close(fd)
    terminal_path = out / ('optimizer_terminal.json' if stage == 'gpu' else 'export_terminal.json')
    terminal = {'returncode': returncode, 'error': error, 'stage': stage,
        'started_unix': started, 'finished_unix': time.time(),
        'elapsed_seconds': time.time() - started, 'api_calls': 0,
        'complete_output_present': (out / ('optimizer_result.json' if stage == 'gpu' else 'result.json')).exists()}
    write(terminal_path, terminal)
    return terminal


def _launch_stage(job_path, stage, timeout, env, lock_fds=()):
    command = [sys.executable, '-u', '-m', QUEUE_MODULE, 'execute', '--job', str(job_path),
               '--stage', stage, '--timeout', str(timeout)]
    for fd in lock_fds:
        command += ['--lock-fd', str(fd)]
    with (job_path.parent / (stage + '_runner.log')).open('x') as stream:
        return subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True, pass_fds=tuple(lock_fds))


def _recover_or_wait(job_path, stage, poll_seconds=5, timeout_seconds=180):
    """Return a terminal stage, wait for its exact process, or fail closed."""
    folder = job_path.parent
    terminal_path = folder / ('optimizer_terminal.json' if stage == 'gpu' else 'export_terminal.json')
    process_path = folder / (stage + '_process.json')
    while True:
        if terminal_path.exists():
            return read(terminal_path)
        record = read(process_path) if process_path.exists() else {}
        if process_alive(record.get('identity')):
            if time.time() - record['started_unix'] > timeout_seconds + 60:
                raise RuntimeError('stage_runner_exceeded_timeout_requires_diagnosis_no_resubmit:' + str(job_path))
            time.sleep(poll_seconds)
            continue
        worker_path = folder / (stage + '_worker_process.json')
        worker = read(worker_path) if worker_path.exists() else {}
        if process_alive(worker.get('identity')):
            if time.time() - worker['started_unix'] > timeout_seconds + 60:
                raise RuntimeError('orphaned_live_worker_exceeded_timeout_requires_diagnosis_no_resubmit:' + str(job_path))
            time.sleep(poll_seconds)
            continue
        reservation_path = folder / ('reservation.json' if stage == 'gpu' else 'export_reservation.json')
        reservation = read(reservation_path)
        if time.time() - reservation['started_unix'] < 30:
            time.sleep(min(2, poll_seconds)); continue
        result_path = folder / ('optimizer_result.json' if stage == 'gpu' else 'result.json')
        if result_path.exists():
            (validate_optimizer if stage == 'gpu' else validate_result)(job_path)
            recovery = {'returncode': 'recovered_from_complete_output', 'stage': stage,
                'original_process_exit_status': 'unknown', 'recovered_unix': time.time(),
                'result': binding(result_path), 'api_calls': 0}
            write(terminal_path, recovery)
            return recovery
        raise RuntimeError('unknown_reserved_stage_no_live_process_or_complete_output:' + str(job_path))


def _acquire_gpu(root, protocol, deadline, quiet_since, job_ref):
    while time.time() < deadline:
        observations = []
        for i in protocol['gpu_indices']:
            try:
                snap = resources.gpu_snapshot(i)
                observations.append(snap)
                resources.check_idle(snap)
            except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as exc:
                quiet_since.pop(i, None)
                if not observations or observations[-1].get('gpu') != i:
                    observations.append({'gpu': i, 'resource_error': str(exc)})
                continue
            quiet_since.setdefault(i, time.monotonic())
            if time.monotonic() - quiet_since[i] < protocol['quiet_seconds']:
                continue
            locks = ExitStack()
            try:
                fds = []
                for lock_path in resources.lock_paths(i):
                    lock = locks.enter_context(lock_path.open('r+'))
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fds.append(lock.fileno())
                latest = resources.gpu_snapshot(i); resources.check_idle(latest)
                if latest['uuid'] != snap['uuid']:
                    raise RuntimeError('GPU_identity_changed_during_lock')
                return latest, locks, fds
            except (BlockingIOError, RuntimeError):
                locks.close(); quiet_since.pop(i, None)
        _state(root, 'waiting_unused_gpu', job=job_ref, observations=observations,
               deadline_unix=deadline, quiet_seconds_required=protocol['quiet_seconds'])
        time.sleep(min(protocol['poll_seconds'], max(0, deadline - time.time())))
    raise TimeoutError('finite_campaign_deadline_reached_waiting_for_gpu')


def _finish_job(job_path, stage, terminal):
    ok = terminal['returncode'] in (0, 'recovered_from_complete_output')
    if stage == 'cpu' and ok:
        result = validate_result(job_path)
        final = {'status': 'completed', 'result': binding(job_path.parent / 'result.json'),
                 'retained': result['retained'], 'physical_success': 'unmeasured', 'api_calls': 0}
    else:
        final = {'status': 'failed', 'failure_stage': stage, 'returncode': terminal['returncode'],
                 'api_calls': 0, 'physical_success': 'unmeasured'}
    write(job_path.parent / 'terminal.json', final)
    return final


def _run_job(root, protocol, job_ref, deadline, quiet_since):
    job_path = resolve(job_ref); folder = job_path.parent
    if (folder / 'terminal.json').exists():
        terminal = read(folder / 'terminal.json')
        if terminal['status'] == 'completed':
            validate_result(job_path)
        return terminal
    _verify_frozen(protocol)
    if shutil.disk_usage(root).free < protocol['min_free_gib'] * 1024 ** 3:
        raise RuntimeError('insufficient_disk_no_historical_files_removed')
    gpu_runner = cpu_runner = None
    if not (folder / 'reservation.json').exists():
        snap, locks, fds = _acquire_gpu(root, protocol, deadline, quiet_since, job_ref)
        with locks:
            # Each member of the frozen manifest can reserve once. Membership
            # and total size are verified once at startup, not by scanning
            # thousands of result directories before every GPU dispatch.
            write(folder / 'reservation.json', {'job': job_ref, 'gpu': snap,
                'started_unix': time.time(), 'protocol': binding(root / 'protocol.json')})
            gpu_runner = _launch_stage(job_path, 'gpu', protocol['per_gpu_job_timeout_seconds'],
                                      _environment(job_path, snap['uuid']), fds)
        # Child inherited lock file descriptions, so closing parent copies does
        # not release the exclusive lock while optimization still runs.
    _state(root, 'optimizer_running_or_recovering', job=job_ref)
    gpu = _recover_or_wait(job_path, 'gpu', timeout_seconds=protocol['per_gpu_job_timeout_seconds'])
    if gpu_runner is not None:
        gpu_runner.wait(timeout=10)
    if gpu['returncode'] not in (0, 'recovered_from_complete_output'):
        return _finish_job(job_path, 'gpu', gpu)
    validate_optimizer(job_path)
    if not (folder / 'export_reservation.json').exists():
        write(folder / 'export_reservation.json', {'job': job_ref, 'started_unix': time.time(),
            'optimizer_result': binding(folder / 'optimizer_result.json'), 'api_calls': 0})
        cpu_runner = _launch_stage(job_path, 'cpu', protocol['per_cpu_export_timeout_seconds'], _environment(job_path))
    _state(root, 'native_cpu_export_running_or_recovering', job=job_ref)
    cpu = _recover_or_wait(job_path, 'cpu', timeout_seconds=protocol['per_cpu_export_timeout_seconds'])
    if cpu_runner is not None:
        cpu_runner.wait(timeout=10)
    return _finish_job(job_path, 'cpu', cpu)


def _gate(root, protocol):
    expected = [binding(resolve(j).parent / 'result.json') for j in protocol['qualification_jobs']]
    for ref in protocol['qualification_jobs']:
        validate_result(resolve(ref))
    write(root / 'qualification_complete.json', {'protocol': binding(root / 'protocol.json'),
        'qualification_results': expected, 'physical_success': 'unmeasured', 'api_calls': 0})
    if not (root / 'independent_gate.json').exists():
        if (root / 'gate_reservation.json').exists():
            raise RuntimeError('unknown_gate_execution_recover_validator_without_rerun')
        validator = resolve(protocol['gate_validator'])
        write(root / 'gate_reservation.json', {'started_unix': time.time(),
            'validator': protocol['gate_validator'], 'api_calls': 0})
        with (root / 'gate_validator.log').open('x') as stream:
            completed = subprocess.run([sys.executable, str(validator), '--queue-root', str(root),
                '--output', str(root / 'independent_gate.json')], cwd=ROOT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'),
                stdout=stream, stderr=subprocess.STDOUT, timeout=protocol['gate_timeout_seconds'])
        write(root / 'gate_terminal.json', {'returncode': completed.returncode, 'api_calls': 0})
        if completed.returncode != 0:
            raise RuntimeError('independent_validator_failed')
    return validate_gate(root, protocol)


def update_candidate_index(root, protocol):
    entries = []
    for row in read(resolve(protocol['candidate_manifest']))['entries']:
        path = Path(row['job']['path'])
        out = (ROOT / path).parent if not path.is_absolute() else path.parent
        if not (out / 'terminal.json').exists() or read(out / 'terminal.json')['status'] != 'completed':
            continue
        result = read(out / 'result.json')
        entries.append({**row, 'cache': result['cache'], 'result': binding(out / 'result.json'),
                       'retained': result['retained'], 'optimizer_status': 'completed',
                       'physical_status': 'unmeasured'})
    atomic(root / 'candidate_cache_index.json', {'schema': VERSION, 'entries': entries,
        'canonical_assets_requested': protocol['canonical_assets'], 'updated_unix': time.time(),
        'native_pose_qualification_only': True, 'default_promoted': False})
    atomic(root / 'native_cache_index.json', {'schema': VERSION,
        'entries': [e for e in entries if e['retained'] > 0],
        'canonical_assets_requested': protocol['canonical_assets'],
        'candidate_outputs_completed': len(entries), 'updated_unix': time.time(),
        'native_pose_qualification_only': True, 'physical_success': 'unmeasured', 'default_promoted': False})


def run(root):
    root = Path(root).resolve()
    protocol = read(root / 'protocol.json')
    _verify_frozen(protocol)
    registry = read(resolve(protocol['registry']))
    if Path(registry['root']).resolve() != root or registry['recipe'] != protocol['recipe']:
        raise ValueError('campaign_registry_binding_mismatch')
    all_jobs = protocol['qualification_jobs'] + protocol['bulk_jobs']
    if len({r['path'] for r in all_jobs}) != len(all_jobs) or len(all_jobs) != protocol['gpu_job_limit']:
        raise ValueError('finite_frozen_job_manifest_invalid')
    resolve(protocol['frozen_inputs']); resolve(protocol['candidate_manifest'])
    with (root / '.supervisor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        session_id = time.strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:10]
        write(root / 'sessions' / (session_id + '.json'), {'identity': process_identity(os.getpid()),
            'started_unix': time.time(), 'protocol': binding(root / 'protocol.json')})
        clock_path = root / 'campaign_clock.json'
        if not clock_path.exists():
            started = time.time()
            write(clock_path, {'started_unix': started,
                'deadline_unix': started + protocol['maximum_campaign_seconds']})
        deadline = read(clock_path)['deadline_unix']
        quiet_since, consecutive_timeouts = {}, 0
        status(root, write_status=True)
        update_candidate_index(root, protocol)
        if (root / 'completion.json').exists():
            _state(root, 'completed', counts=status(root))
            return
        last_index_update = 0
        try:
            for phase in ('qualification', 'bulk'):
                if phase == 'bulk':
                    _state(root, 'independent_qualification_review')
                    _gate(root, protocol)
                for ref in protocol[phase + '_jobs']:
                    job_path = resolve(ref)
                    # Recover existing reservations even after the submission deadline.
                    if time.time() >= deadline and not (job_path.parent / 'reservation.json').exists():
                        raise TimeoutError('finite_campaign_deadline_no_new_submission')
                    result = _run_job(root, protocol, ref, deadline, quiet_since)
                    # Live indexes are derived at most once per minute. The
                    # immutable per-job results remain immediately available.
                    if time.monotonic() - last_index_update >= 60 or phase == 'qualification':
                        status(root, write_status=True)
                        update_candidate_index(root, protocol)
                        last_index_update = time.monotonic()
                    if result['status'] != 'completed':
                        consecutive_timeouts += int(result.get('returncode') == 'timeout')
                        if (phase == 'qualification' or result.get('returncode') != 'timeout'
                                or consecutive_timeouts >= 2):
                            raise RuntimeError('job_failure_needs_diagnosis:' + str(job_path))
                    else:
                        consecutive_timeouts = 0
            status(root, write_status=True)
            update_candidate_index(root, protocol)
            write(root / 'completion.json', {'finished_unix': time.time(),
                'protocol': binding(root / 'protocol.json'), 'counts': status(root), 'api_calls': 0,
                'physical_success': 'unmeasured', 'default_promoted': False})
            _state(root, 'completed', counts=status(root))
        except Exception as exc:
            write(root / 'sessions' / (session_id + '_stopped.json'), {
                'stopped_unix': time.time(), 'error': f'{type(exc).__name__}: {exc}', 'api_calls': 0})
            status(root, write_status=True)
            update_candidate_index(root, protocol)
            _state(root, 'stopped_for_diagnosis' if not isinstance(exc, TimeoutError) else 'deadline_reached',
                   error=f'{type(exc).__name__}: {exc}', counts=status(root))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    for name in ('root', 'index', 'original', 'candidate', 'qualification-inputs', 'gate-validator'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--campaign-seconds', type=int, default=7 * 86400)
    for name in ('run', 'status'):
        sub.add_parser(name).add_argument('--root', type=Path, required=True)
    p = sub.add_parser('execute'); p.add_argument('--job', type=Path, required=True)
    p.add_argument('--stage', choices=('gpu', 'cpu'), required=True)
    p.add_argument('--timeout', type=int, required=True)
    p.add_argument('--lock-fd', type=int, action='append', default=[])
    a = parser.parse_args()
    if a.command == 'prepare':
        result = prepare(a.root, a.index, a.original, a.candidate, a.qualification_inputs,
                         a.gate_validator, a.campaign_seconds)
    elif a.command == 'run':
        result = run(a.root)
    elif a.command == 'status':
        result = status(a.root)
    else:
        result = execute_stage(a.job, a.stage, a.timeout, a.lock_fd)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
