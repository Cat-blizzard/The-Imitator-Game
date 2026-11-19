"""Explicit isolated CPU compilation with confirmed persistent collision OBJs.

Prepare a fresh single-source job, then run it in a separate process. Only that
worker temporarily replaces metric_native_v3's imported cooking helper; all
raw ingestion, refinement, surface, mass and settling code remains unchanged.
This module does not register, publish or resume any existing production queue.
"""
import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from resimpair_grasp.common import ROOT, binding, read, resolve, write

VERSION = 'raw_manipulation_confirmed_cook_v2'
WORKER_JOB_ENV = 'RESIMPAIR_CONFIRMED_COOK_WORKER_JOB_SHA256'
IMPLEMENTATIONS = (
    'resimpair_assets/raw_manipulation_v2.py',
    'resimpair_assets/stable_cook_confirmed_v2.py',
    'resimpair_assets/raw_manipulation_v1.py',
    'resimpair_assets/raw_mesh_io_v1.py',
    'resimpair_assets/raw_size_policy_v1.py',
    'resimpair_assets/metric_variants_v1.py',
    'resimpair_assets/metric_variants_v2.py',
    'resimpair_assets/metric_native_v3.py',
    'resimpair_assets/metric_native_v4.py',
    'resimpair_assets/metric_native_v5.py',
    'resimpair_assets/metric_native_v6.py',
    'resimpair_assets/metric_admission_v1.py',
    'resimpair_grasp/assets.py',
    'resimpair_grasp/evaluator.py',
    'resimpair_grasp/common.py',
    'resimpair_tabletop/native_mass.py',
)


def prepare(input_path, output, *, kind='raw_job', timeout_seconds=900):
    """Freeze one new attempt without running a native cook or changing inputs.

    metric_descriptor takes the *pre-native* metric variant. To repair an old
    descriptor, pass its native_compiler.input_metric_descriptor; old cooked
    outputs must not silently become the new source geometry.
    """
    input_path, output = Path(input_path).resolve(), Path(output).resolve()
    if kind not in {'raw_job', 'metric_descriptor'}:
        raise ValueError('raw_job_or_metric_descriptor_required')
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 7200:
        raise ValueError('one_to_7200_second_worker_budget_required')
    if output.exists():
        raise ValueError('fresh_single_source_output_required_no_retry')
    source = read(input_path)
    if kind == 'raw_job':
        for key in ('raw_row', 'seed', 'ingestion_group'):
            if key not in source:
                raise ValueError('original_raw_source_job_required')
        source_id = source['raw_row']['id']
    else:
        if source.get('semantic_role') != 'manipulation':
            raise ValueError('manipulation_metric_descriptor_required')
        if any(key in source for key in ('native_compiler', 'native_properties', 'cooking')):
            raise ValueError('use_original_pre_native_metric_descriptor_not_old_cooked_geometry')
        source_id = source['source_id']
        for ref in [source['visual'], *source['collisions']]:
            resolve(ref)
    implementation_refs = [binding(ROOT / path) for path in IMPLEMENTATIONS]
    original_input = binding(input_path)
    if kind == 'raw_job':
        derived = {**source, 'output': str(output / 'compiled')}
        write(output / 'raw_job.json', derived)
        compile_input = binding(output / 'raw_job.json')
    else:
        compile_input = original_input
    job = {'schema': VERSION, 'kind': kind, 'source_id': source_id,
        'original_input': original_input, 'compile_input': compile_input,
        'output': str(output / 'compiled'), 'timeout_seconds': timeout_seconds,
        'implementations': implementation_refs, 'maximum_asset_compile_attempts': 1}
    write(output / 'job.json', job)
    protocol = {'schema': VERSION, 'job': binding(output / 'job.json'),
        'original_input': original_input, 'implementation_bindings': implementation_refs,
        'hook_target': 'resimpair_assets.metric_native_v3._stable_cook',
        'replacement': 'resimpair_assets.stable_cook_confirmed_v2.stable_cook',
        'hook_scope': 'separate CPU worker process only; restored on success or exception',
        'unchanged_convergence_threshold_m': 1e-6, 'unchanged_runtime_replay_threshold_m': 1e-5,
        'unchanged_visual_surface_p95_threshold_m': .003,
        'maximum_conversions_per_part_per_native_stage': 3,
        'maximum_fresh_process_probes_per_part_per_native_stage': 4,
        'native_refinement_and_canonicalization': 'original metric_native_v6 through v3 rules',
        'timeout_seconds': timeout_seconds, 'maximum_asset_compile_attempts': 1,
        'new_grasps_required_for_changed_descriptor': True,
        'grasp_or_robot_execution': False, 'API_calls': 0, 'GPU_jobs': 0,
        'existing_queue_or_cache_modified': False}
    write(output / 'PROTOCOL.json', protocol)
    return {'job': binding(output / 'job.json'), 'protocol': binding(output / 'PROTOCOL.json'),
            'source_id': source_id, 'status': 'prepared_not_executed'}


@contextmanager
def _confirmed_cook_hook():
    from resimpair_grasp import assets
    from . import metric_native_v3, stable_cook_confirmed_v2
    if (stable_cook_confirmed_v2.CONVERGENCE_M != 1e-6
            or stable_cook_confirmed_v2.RUNTIME_REPLAY_M != 1e-5
            or stable_cook_confirmed_v2.MAX_CONVERSIONS != 3):
        raise ValueError('confirmed_cook_original_thresholds_and_budget_required')
    previous = metric_native_v3._stable_cook
    if previous is not assets._stable_cook:
        raise ValueError('unexpected_existing_native_cook_override')
    metric_native_v3._stable_cook = stable_cook_confirmed_v2.stable_cook
    try:
        yield
    finally:
        metric_native_v3._stable_cook = previous


def accounting(output):
    """Count every helper attempt and OBJ export, including incomplete parts."""
    output = Path(output)
    jobs = sorted(output.glob('**/hull_*_confirmation_v2/probe_*/job.json'))
    complete = [path.parent / 'result.json' for path in jobs if (path.parent / 'result.json').exists()]
    confirmations = sorted(output.glob('**/hull_*_confirmation_v2/RESULT.json'))
    settles = [path for pattern in ('**/initial_settle.json', '**/canonical_settle.json')
               for path in output.glob(pattern)]
    return {'fresh_process_probe_attempts': len(jobs), 'completed_native_probes': len(complete),
        'probe_outcomes': [binding(path) for path in complete],
        'native_geometry_conversion_exports': len(list(output.glob('**/hull_*_cooked_*.obj'))),
        'initial_source_OBJ_exports': len(list(output.glob('**/hull_*_source.obj'))),
        'completed_part_confirmations': len(confirmations),
        'confirmed_parts': sum(read(path)['status'] == 'confirmed_native_fixed_file' for path in confirmations),
        'recorded_settle_steps': sum(read(path)['frames'] for path in settles),
        'settle_steps_scope': 'completed evidence files only; interrupted unrecorded steps are unknown',
        'robot_physical_executions': 0, 'GPU_jobs': 0, 'API_calls': 0}


def _validated_job(job_path):
    job_path = Path(job_path).resolve()
    job = read(job_path)
    if job.get('schema') != VERSION or job.get('maximum_asset_compile_attempts') != 1:
        raise ValueError('registered_single_confirmed_cook_attempt_required')
    protocol = read(job_path.parent / 'PROTOCOL.json')
    if protocol['job'] != binding(job_path) or protocol['implementation_bindings'] != job['implementations']:
        raise ValueError('confirmed_cook_protocol_job_binding_mismatch')
    if Path(job['output']).resolve() != job_path.parent / 'compiled':
        raise ValueError('fresh_bound_compile_directory_required')
    for ref in [job['original_input'], job['compile_input'], *job['implementations']]:
        resolve(ref)
    return job_path, job


def worker(job_path):
    """Internal entry point; run() supplies the marker in a newly spawned process."""
    job_path, job = _validated_job(job_path)
    if os.environ.get(WORKER_JOB_ENV) != binding(job_path)['sha256']:
        raise ValueError('separate_confirmed_cook_worker_required_use_run')
    folder = job_path.parent
    if (folder / 'worker_started.json').exists() or Path(job['output']).exists():
        raise ValueError('single_worker_attempt_required_no_retry')
    write(folder / 'worker_started.json', {'job': binding(job_path),
        'pid': os.getpid(), 'parent_pid': os.getppid(), 'unix': time.time()})
    began = time.monotonic()
    try:
        with _confirmed_cook_hook():
            if job['kind'] == 'raw_job':
                from . import raw_manipulation_v1
                result = raw_manipulation_v1.compile_raw(resolve(job['compile_input']))
            elif job['kind'] == 'metric_descriptor':
                from . import metric_native_v6
                native = metric_native_v6.compile_variant(resolve(job['compile_input']), Path(job['output']))
                result = {'source_id': job['source_id'], 'asset': native['asset'],
                    'status': 'native_geometry_ready' if native['geometry_qualified'] else 'native_qualification_failed',
                    'native_result': binding(Path(job['output']) / 'result.json'),
                    'geometry_qualified': native['geometry_qualified'],
                    'semantic_scale_admission': 'not_repeated_by_metric_descriptor_entry',
                    'physical_success': 'unmeasured', 'new_grasp_required': True}
            else:
                raise ValueError('unknown_confirmed_cook_job_kind')
        for ref in job['implementations']:
            resolve(ref)
    except Exception as error:
        from scipy.spatial import QhullError
        status = 'geometry_or_policy_failed' if isinstance(error, (ValueError, QhullError)) else 'unknown_system_failure'
        result = {'source_id': job['source_id'], 'status': status,
            'reason': type(error).__name__ + ':' + str(error), 'traceback': traceback.format_exc()}
    result = {**result, 'adapter': VERSION, 'protocol': binding(folder / 'PROTOCOL.json'),
        'worker_pid': os.getpid(), 'parent_pid': os.getppid(),
        'worker_wall_seconds': time.monotonic() - began, 'cpu_accounting': accounting(job['output']),
        'grasp_cache_reuse_authorized': False, 'existing_queue_or_cache_modified': False}
    write(folder / 'cpu_result.json', result)
    return result


def run(job_path):
    """Launch exactly one CPU worker; a timeout kills its probe subprocess group."""
    job_path, job = _validated_job(job_path)
    folder = job_path.parent
    if any((folder / name).exists() for name in ('LAUNCH.json', 'worker_started.json', 'SUPERVISOR.json')):
        raise ValueError('single_confirmed_cook_launch_required_no_retry')
    write(folder / 'LAUNCH.json', {'job': binding(job_path), 'supervisor_pid': os.getpid(),
        'unix': time.time(), 'timeout_seconds': job['timeout_seconds'], 'attempt': 1})
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env[WORKER_JOB_ENV] = binding(job_path)['sha256']
    began = time.monotonic()
    command = [sys.executable, '-m', 'resimpair_assets.raw_manipulation_v2', '_worker', '--job', str(job_path)]
    status = 'worker_completed'
    with (folder / 'stdout.log').open('x') as stdout, (folder / 'stderr.log').open('x') as stderr:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                   start_new_session=True)
        write(folder / 'WORKER_PROCESS.json', {'pid': process.pid, 'supervisor_pid': os.getpid(),
            'job': binding(job_path), 'process_group_id': process.pid})
        try:
            process.wait(timeout=job['timeout_seconds'])
        except subprocess.TimeoutExpired:
            status = 'worker_timeout'
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
            # A probe belongs to the worker's group even if its parent exits
            # first. Do not leave such a probe running after the job timeout.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    result_path = folder / 'cpu_result.json'
    if status == 'worker_completed' and (process.returncode != 0 or not result_path.exists()):
        status = 'worker_failed_without_complete_result'
    if result_path.exists():
        result = read(result_path)
        if result['worker_pid'] != process.pid or result['parent_pid'] != os.getpid():
            raise ValueError('independent_asset_worker_identity_mismatch')
    report = {'schema': VERSION, 'status': status, 'job': binding(job_path),
        'returncode': process.returncode, 'worker_pid': process.pid, 'supervisor_pid': os.getpid(),
        'worker_result': binding(result_path) if result_path.exists() else None,
        'wall_seconds': time.monotonic() - began, 'timeout_seconds': job['timeout_seconds'],
        'cpu_accounting': accounting(job['output']), 'API_calls': 0, 'GPU_jobs': 0,
        'robot_physical_executions': 0, 'existing_queue_or_cache_modified': False}
    write(folder / 'SUPERVISOR.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    prepare_parser = subparsers.add_parser('prepare')
    prepare_parser.add_argument('--kind', choices=('raw_job', 'metric_descriptor'), required=True)
    prepare_parser.add_argument('--input', type=Path, required=True)
    prepare_parser.add_argument('--output', type=Path, required=True)
    prepare_parser.add_argument('--timeout-seconds', type=int, default=900)
    for name in ('run', '_worker'):
        command_parser = subparsers.add_parser(name)
        command_parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.input, args.output, kind=args.kind, timeout_seconds=args.timeout_seconds)
    elif args.command == 'run':
        result = run(args.job)
    else:
        result = worker(args.job)
    print(result)
    if result.get('status') in {'worker_timeout', 'worker_failed_without_complete_result', 'unknown_system_failure'}:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
