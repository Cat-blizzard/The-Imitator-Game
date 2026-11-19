"""Frozen-list supervision for at most 24 sources, using the untouched v1 worker.

The v1 worker accepts six-source protocol chunks. This supervisor binds those
chunks into one unique-source campaign, two concurrent workers and one absolute
60-minute deadline. Preparation never launches a worker. Execution requires an
explicit parent review bound to this exact protocol and job list.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from resimpair_grasp.common import ROOT, binding, read, resolve, write
from . import raw_native_input_repairs_v1 as retained

VERSION = 'raw_native_input_repairs_campaign_v2'
RELEASE_SCHEMA = 'raw_native_input_repairs_parent_release_v2'
MAX_SOURCES = 24
WORKERS = 2
SOURCE_SECONDS = 900
CAMPAIGN_SECONDS = 3600


def prepare(cohort_path, report_root, run_root):
    """Read/bind sources and create job-only directories; no source processing."""
    cohort_path, report_root, run_root = map(lambda p: Path(p).resolve(), (cohort_path, report_root, run_root))
    if run_root.exists() or (report_root / 'PROTOCOL.json').exists():
        raise ValueError('fresh_campaign_preparation_required_no_reset')
    cohort = read(cohort_path); selected = cohort['selected']
    ids = [item['source']['source_id'] for item in selected]
    if not 1 <= len(ids) <= MAX_SOURCES or len(set(ids)) != len(ids):
        raise ValueError('one_to_24_unique_frozen_sources_required')
    excluded = {item['source']['source_id'] for item in read(resolve(cohort['excluded_six']))['selected']}
    if set(ids) & excluded:
        raise ValueError('prior_six_source_pilot_overlap_forbidden')
    resolve(cohort['snapshot_source_rows'])
    for item in selected:
        row = item['source']; original = read(resolve(row['job']))
        resolve(row['cpu_result'])
        if original['raw_row']['id'] != row['source_id'] or original['raw_row']['partition'] != row['partition']:
            raise ValueError('original_source_identity_and_partition_required')
    implementation_refs = [binding(ROOT / p) for p in retained.IMPLEMENTATIONS] + [binding(__file__)]
    chunks = []; all_jobs = []; indexed = {}
    for number, start in enumerate(range(0, len(selected), 6)):
        items = selected[start:start + 6]
        report = report_root / 'worker_chunks' / f'{number:02d}'
        prereg = {**cohort, 'selected': items, 'source_count': len(items),
            'global_cohort': binding(cohort_path), 'chunk_index': number,
            'chunk_is_only_a_worker_protocol_compatibility_boundary': True}
        write(report / 'PREREGISTRATION.json', prereg)
        refs = retained.prepare(report / 'PREREGISTRATION.json', report, run_root / f'chunk_{number:02d}')
        chunks.append({'protocol': binding(report / 'PROTOCOL.json'), 'jobs': binding(report / 'JOBS.json')})
        for item, ref in zip(items, refs):
            indexed[item['source']['source_id']] = ref
    # Interleave failure strata by frozen within-stratum rank so a deadline
    # does not systematically leave the final failure class unobserved.
    strata = {name: number for number, name in enumerate(cohort['strata'])}
    ordered = sorted(selected, key=lambda item: (item['rank_in_stratum'], strata[item['stratum']]))
    for number, item in enumerate(ordered):
        row = item['source']
        all_jobs.append({'schedule_index': number, 'source_id': row['source_id'],
            'failure_stratum': item['stratum'], 'partition': row['partition'],
            'selection_hash': item['selection_hash'], 'job': indexed[row['source_id']]})
    protocol = {'schema': VERSION, 'cohort': binding(cohort_path), 'report_root': str(report_root),
        'run_root': str(run_root), 'implementation_bindings': implementation_refs,
        'worker_chunks': chunks, 'maximum_sources': len(ids), 'maximum_CPU_workers': WORKERS,
        'source_timeout_seconds': SOURCE_SECONDS, 'campaign_timeout_seconds': CAMPAIGN_SECONDS,
        'deadline_policy': 'one monotonic absolute campaign deadline; each source gets min(900, remaining seconds); at deadline kill active worker process groups and never start remaining jobs',
        'unstarted_deadline_state': 'not_launched_campaign_deadline; source_attempts=0',
        'maximum_source_attempts_per_source': 1, 'automatic_retries': 0,
        'maximum_representation_proposals_per_source': 2, 'maximum_native_settle_steps_per_source': 1440,
        'retained_worker': 'resimpair_assets.raw_native_input_repairs_v1._worker',
        'retained_worker_scientific_logic_or_qualifiers_changed': False,
        'qualification_scope': 'unchanged original code-gate observations; no automatic promotion or GPU launch',
        'parent_release_required': True, 'release_schema': RELEASE_SCHEMA,
        'phase_at_freeze': 'prepared_only_pending_parent_review',
        'old_main_population_or_six_source_attempts_modified': False,
        'new_source_attempts_during_preparation': 0, 'new_native_cooks_during_preparation': 0,
        'new_pose_candidates': 0, 'robot_executions': 0, 'GPU_jobs': 0, 'API_calls': 0,
        'training_runs': 0, 'physical_success': 'unmeasured', 'created_unix': time.time()}
    write(report_root / 'PROTOCOL.json', protocol)
    write(report_root / 'JOBS.json', {'schema': VERSION, 'protocol': binding(report_root / 'PROTOCOL.json'), 'jobs': all_jobs})
    write(report_root / 'PARENT_RELEASE_TEMPLATE.json', {'schema': RELEASE_SCHEMA, 'reviewer_role': 'parent',
        'passed': False, 'protocol': binding(report_root / 'PROTOCOL.json'), 'jobs': binding(report_root / 'JOBS.json'),
        'authorized_unique_sources': len(ids), 'maximum_source_attempts_per_source': 1,
        'maximum_CPU_workers': WORKERS, 'maximum_source_seconds': SOURCE_SECONDS,
        'maximum_campaign_seconds': CAMPAIGN_SECONDS, 'automatic_retries': 0,
        'GPU_jobs': 0, 'robot_executions': 0, 'API_calls': 0,
        'review_note': 'Template only. Parent must issue a separate passed=true exact-bound review after inspecting the prepared cohort and executable plan.'})
    return {'protocol': binding(report_root / 'PROTOCOL.json'), 'jobs': binding(report_root / 'JOBS.json'),
            'registered_sources': len(ids), 'phase': 'prepared_not_executed'}


def validate(report_root):
    root = Path(report_root).resolve(); protocol = read(root / 'PROTOCOL.json')
    planned = read(root / 'JOBS.json')
    if (protocol['schema'] != VERSION or planned['protocol'] != binding(root / 'PROTOCOL.json')
            or protocol['report_root'] != str(root) or protocol['maximum_CPU_workers'] != WORKERS
            or protocol['source_timeout_seconds'] != SOURCE_SECONDS
            or protocol['campaign_timeout_seconds'] != CAMPAIGN_SECONDS):
        raise ValueError('frozen_two_worker_900s_3600s_campaign_required')
    rows = planned['jobs']; ids = [r['source_id'] for r in rows]
    if not 1 <= len(ids) <= MAX_SOURCES or len(ids) != len(set(ids)) or len(ids) != protocol['maximum_sources']:
        raise ValueError('unique_bound_campaign_population_required')
    for ref in [protocol['cohort'], *protocol['implementation_bindings']]:
        resolve(ref)
    for row in rows:
        path, job, _ = retained._validate_job(resolve(row['job']))
        if job['source_id'] != row['source_id']:
            raise ValueError('bound_campaign_source_identity_required')
    return root, protocol, rows


def require_release(root, protocol, release_path):
    if release_path is None:
        raise ValueError('parent_review_required_before_any_source_launch')
    ref = binding(release_path); release = read(resolve(ref))
    expected = {'schema': RELEASE_SCHEMA, 'reviewer_role': 'parent', 'passed': True,
        'protocol': binding(root / 'PROTOCOL.json'), 'jobs': binding(root / 'JOBS.json'),
        'authorized_unique_sources': protocol['maximum_sources'], 'maximum_source_attempts_per_source': 1,
        'maximum_CPU_workers': WORKERS, 'maximum_source_seconds': SOURCE_SECONDS,
        'maximum_campaign_seconds': CAMPAIGN_SECONDS, 'automatic_retries': 0,
        'GPU_jobs': 0, 'robot_executions': 0, 'API_calls': 0}
    if any(release.get(key) != value for key, value in expected.items()):
        raise ValueError('exact_bound_parent_review_required_before_source_launch')
    return ref


def source_window(deadline, now):
    """Pure budget calculation, used both before and after input validation."""
    return min(float(SOURCE_SECONDS), max(0., deadline - now))


def supervise_one(row, campaign_root, deadline, release_ref):
    root = Path(campaign_root)
    result_path = root / 'source_supervisors' / f'{row["schedule_index"]:02d}.json'
    remaining = source_window(deadline, time.monotonic())
    if remaining <= 0:
        value = {'source_id': row['source_id'], 'job': row['job'], 'source_attempts': 0,
                 'status': 'not_launched_campaign_deadline', 'automatic_retries': 0}
        write(result_path, value)
        return value
    job_path, job, _ = retained._validate_job(resolve(row['job'])); folder = job_path.parent
    if any((folder / name).exists() for name in ('LAUNCH.json', 'worker_started.json', 'SUPERVISOR.json', 'compiled')):
        raise ValueError('existing_source_attempt_is_never_resubmitted')
    remaining = source_window(deadline, time.monotonic())
    if remaining <= 0:
        value = {'source_id': row['source_id'], 'job': row['job'], 'source_attempts': 0,
                 'status': 'not_launched_campaign_deadline', 'automatic_retries': 0}
        write(result_path, value)
        return value
    began = time.monotonic()
    write(folder / 'LAUNCH.json', {'job': row['job'], 'source_attempt': 1, 'unix': time.time(),
        'supervisor_pid': os.getpid(), 'timeout_seconds': remaining,
        'global_campaign_protocol': binding(root / 'PROTOCOL.json'), 'parent_release': release_ref})
    env = retained._env(); env[retained.WORKER_ENV] = row['job']['sha256']
    status = 'worker_completed'
    with (folder / 'stdout.log').open('x') as stdout, (folder / 'stderr.log').open('x') as stderr:
        process = subprocess.Popen([sys.executable, '-m', 'resimpair_assets.raw_native_input_repairs_v1',
            '_worker', '--job', str(job_path)], cwd=ROOT, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        write(folder / 'WORKER_PROCESS.json', {'pid': process.pid, 'supervisor_pid': os.getpid(), 'process_group_id': process.pid})
        try:
            process.wait(timeout=source_window(deadline, time.monotonic()))
        except subprocess.TimeoutExpired:
            status = 'worker_timeout'
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
    observed = folder / 'cpu_result.json'
    if status == 'worker_completed' and (process.returncode != 0 or not observed.exists()):
        status = 'worker_failed_without_complete_result'
    value = {'schema': VERSION, 'source_id': row['source_id'], 'job': row['job'], 'source_attempts': 1,
        'status': status, 'returncode': process.returncode, 'worker_pid': process.pid,
        'worker_result': binding(observed) if observed.exists() else None,
        'wall_seconds': time.monotonic() - began, 'granted_timeout_seconds': remaining,
        'cpu_accounting': retained.raw_manipulation_v2.accounting(job['output']),
        'automatic_retries': 0, 'GPU_jobs': 0, 'robot_executions': 0, 'API_calls': 0}
    write(folder / 'SUPERVISOR.json', value); write(result_path, value)
    return value


def run(report_root, release_path=None):
    root, protocol, rows = validate(report_root)
    release_ref = require_release(root, protocol, release_path)
    if (root / 'RUN_STARTED.json').exists():
        raise ValueError('single_campaign_launch_no_reset_or_automatic_resume')
    began = time.monotonic(); deadline = began + CAMPAIGN_SECONDS
    write(root / 'RUN_STARTED.json', {'protocol': binding(root / 'PROTOCOL.json'),
        'jobs': binding(root / 'JOBS.json'), 'parent_release': release_ref, 'pid': os.getpid(),
        'unix': time.time(), 'deadline_unix': time.time() + CAMPAIGN_SECONDS,
        'monotonic_deadline_enforced': True})
    reports = []
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = [executor.submit(supervise_one, row, root, deadline, release_ref) for row in rows]
        for future in as_completed(futures):
            report = future.result(); reports.append(report)
            print(report['source_id'], report['status'], flush=True)
    for ref in protocol['implementation_bindings']:
        resolve(ref)
    result = {'schema': VERSION, 'protocol': binding(root / 'PROTOCOL.json'), 'reports': reports,
        'source_attempts': sum(r['source_attempts'] for r in reports),
        'not_launched_due_deadline': sum(r['source_attempts'] == 0 for r in reports),
        'wall_seconds_including_kill_and_record_cleanup': time.monotonic() - began,
        'maximum_active_source_workers': WORKERS, 'automatic_retries': 0,
        'GPU_jobs': 0, 'robot_executions': 0, 'API_calls': 0, 'physical_success': 'unmeasured'}
    write(root / 'CAMPAIGN_RESULT.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run'))
    parser.add_argument('--cohort', type=Path); parser.add_argument('--report-root', type=Path, required=True)
    parser.add_argument('--run-root', type=Path); parser.add_argument('--release', type=Path)
    args = parser.parse_args()
    result = prepare(args.cohort, args.report_root, args.run_root) if args.command == 'prepare' else run(args.report_root, args.release)
    print(result)


if __name__ == '__main__':
    main()
