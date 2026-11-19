"""Resume only unattempted cases of a frozen, human-grounded simulation plan.

The frozen native collectors are unchanged. Each finite batch is independently
verified before the next one; generated candidates still require visual review.
"""
import argparse
import collections
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path('/mnt/sdf/dataset/zhouxunzhe/zhiqi/PolicyGen')
PYTHON = REPO / '.venv/bin/python'
METHODS = Path(__file__).resolve().parents[1]
LIBRARY = Path('/mnt/sdf/datasets/hypericl_human_robot_task_library_20261019_v1')
GEN = LIBRARY / 'generation'
PLAN = GEN / 'scale192_prepared_v1/plan.json'
RESERVE = 150 * 2**30
TASKS = ['apple_bowl_transfer', 'closed_book_to_open_box', 'right_cabinet_open_only']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def bind(path):
    return {'path': str(path), 'sha256': sha(path)}


def save(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open('x') as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write('\n')
    else:
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        tmp.replace(path)


def remaining_jobs(plan):
    remaining, attempted = [], []
    for job in plan['jobs']:
        touched = Path(job['output']).exists() or (GEN / 'scale192_bindings_v1' / job['id']).exists()
        (attempted if touched else remaining).append(job)
    return remaining, attempted


def balanced_order(jobs):
    bins = collections.defaultdict(collections.deque)
    for job in jobs:
        bins[(job['asset']['id'], job['acting_arm_index'])].append(job)
    ordered = []
    while any(bins.values()):
        for key in sorted(bins):
            if bins[key]:
                ordered.append(bins[key].popleft())
    return ordered


def gpu_snapshot(gpu):
    raw = subprocess.check_output([
        'nvidia-smi', '-i', str(gpu), '--query-gpu=uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=20).strip()
    uuid, memory, util = [v.strip() for v in raw.split(',')]
    apps = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
        '--format=csv,noheader,nounits'], text=True, timeout=20)
    owners = [line for line in apps.splitlines() if line.split(',')[0].strip() == uuid]
    return dict(gpu=gpu, uuid=uuid, memory_MiB=int(memory), utilization=int(util), compute_owners=owners)


def assert_idle(snapshot):
    if snapshot['compute_owners'] or snapshot['memory_MiB'] > 512 or snapshot['utilization'] > 10:
        raise RuntimeError('Assigned GPU is occupied; preserving other processes and stopping: ' + str(snapshot))


def source_check(plan):
    for path, digest in plan['source_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Frozen source mismatch: ' + path)
    if sha(plan['spec']['path']) != plan['spec']['sha256']:
        raise ValueError('Human task specification changed')
    spec = read(plan['spec']['path'])
    for task in spec['tasks']:
        for human in task['human_windows']:
            if sha(human['video']['path']) != human['video']['sha256']:
                raise ValueError('Human video changed: ' + human['id'])
    return spec


def prepare(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    owner_audit = read(args.owner_audit)
    if (not owner_audit.get('all_same_identity_processes_closed') or
            not owner_audit.get('no_numeric_old_groups_or_sessions_have_non_zombie_members') or
            owner_audit.get('GPU2_compute_apps') != [] or args.gpu != 2):
        raise ValueError('This continuation requires the reviewed GPU2 owner closure evidence')
    if not read(args.prior_qc/'report.json')['passed']:
        raise ValueError('Prior attempted cases have unresolved verification failures')
    plan = read(PLAN)
    spec = source_check(plan)
    remaining, attempted = remaining_jobs(plan)
    if not remaining:
        raise ValueError('No unattempted identities remain')
    snapshot = gpu_snapshot(args.gpu)
    assert_idle(snapshot)
    reserve_check = shutil.disk_usage(GEN).free
    estimate = read(GEN / 'scale192_prepared_v1/CPU_preflight.json')['scale192_storage_budget_bytes']
    if reserve_check - estimate < RESERVE:
        raise RuntimeError('Insufficient space for measured batch budget plus 150 GiB reserve')
    for job in remaining:
        task = next(t for t in spec['tasks'] if t['id'] == job['task'])
        valid = {h['id'] for h in task['human_windows'] if h['split'] == job['split']}
        if not job['human_window_ids'] or not set(job['human_window_ids']) <= valid:
            raise ValueError('Missing complete same-split human task reference')
    args.output.mkdir(parents=True)
    queues = {task: collections.deque(balanced_order([j for j in remaining if j['task'] == task]))
              for task in TASKS}
    batches = []
    while any(queues.values()):
        for task in TASKS:
            jobs = [queues[task].popleft() for _ in range(min(args.batch_size, len(queues[task])))]
            if not jobs:
                continue
            name = f'{len(batches):02d}_{task}'
            auth = dict(phase='scale', native_launch_authorized=True,
                pilot_plan_sha256=sha(PLAN), GPUs=[args.gpu], GPU_UUIDs={str(args.gpu):snapshot['uuid']},
                task_GPU_assignment={task:args.gpu}, case_ids=[j['id'] for j in jobs],
                existing_native_owners_reviewed_closed=True,
                quarantine_constraints_reviewed_for_assigned_GPU=True,
                owner_and_quarantine_evidence=bind(args.owner_audit),
                user_instruction='2026-10-20: 按之前要求生成仿真，严格从人类视频造对应仿真。',
                resume_of_previously_paused_data_generation=True,
                no_retry_of_any_touched_identity=True, no_model_training=True)
            auth_path = args.output / 'authorizations' / (name + '.json')
            save(auth_path, auth, exclusive=True)
            batches.append(dict(name=name, task=task, job_ids=auth['case_ids'],
                authorization=bind(auth_path), queue=str(args.output/'queues'/name),
                qc=str(args.output/'qc'/name)))
    pins = {str(Path(__file__).resolve()):sha(__file__),
            str(METHODS/'human_to_robot_pipeline/verify_batch.py'):sha(METHODS/'human_to_robot_pipeline/verify_batch.py')}
    schedule = dict(schema='human_first_resume_finite_v1', created_unix=time.time(), plan=bind(PLAN),
        human_spec=plan['spec'], GPU=args.gpu, GPU_UUID=snapshot['uuid'], source_sha256=pins,
        owner_and_quarantine_evidence=bind(args.owner_audit),
        initial_gpu_snapshot=snapshot, batches=batches, native_attempt_budget=len(remaining),
        previously_attempted_ids=[j['id'] for j in attempted],
        new_attempts_by_task=dict(collections.Counter(j['task'] for j in remaining)),
        initial_free_bytes=reserve_check, measured_full192_storage_budget_bytes=estimate,
        disk_reserve_bytes=RESERVE, native_CPU_cores=args.cpu_cores, QC_CPU_core=args.qc_core,
        prior_qc_report=bind(args.prior_qc/'report.json'),
        prior_qc_records=bind(args.prior_qc/'verified_robot_records.json'),
        pilot_records=bind(GEN/'pilot_review_CPU_v1/verified_robot_records.json'),
        acceptance='Machine-verified candidates; actual RGB review required before training admission',
        reference_policy='Only task-spec reviewed, same-split windows. Never infer the loader pool from goal name.',
        synthetic_human_generation=False, model_training=False)
    save(args.output/'schedule.json', schedule, exclusive=True)
    print(json.dumps({'schedule':str(args.output/'schedule.json'), 'attempts':len(remaining),
                      'batches':len(batches), 'by_task':schedule['new_attempts_by_task']}), flush=True)


def run(args):
    path = args.output/'schedule.json'
    schedule = read(path)
    original_digest = sha(path)
    guard = (args.output/'supervisor.lock').open('a')
    fcntl.flock(guard.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if sha(schedule['plan']['path']) != schedule['plan']['sha256']:
        raise ValueError('Frozen plan changed')
    for key in ('owner_and_quarantine_evidence', 'prior_qc_report', 'prior_qc_records', 'pilot_records'):
        evidence = schedule[key]
        if sha(evidence['path']) != evidence['sha256']:
            raise ValueError('Changed bound evidence: ' + key)
    source_check(read(schedule['plan']['path']))
    prior_records = [schedule['pilot_records']['path'], schedule['prior_qc_records']['path']]
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    # Keep user/global CUDA environment out of CPU verification; native worker owns its GPU selection.
    env.pop('CUDA_VISIBLE_DEVICES', None)
    started = time.time()
    completed = []
    for batch in schedule['batches']:
        for source, digest in schedule['source_sha256'].items():
            if sha(source) != digest:
                raise ValueError('Supervisor/verification source changed while collecting')
        if sha(path) != original_digest:
            raise ValueError('Schedule changed')
        auth = batch['authorization']
        if sha(auth['path']) != auth['sha256']:
            raise ValueError('Exact batch authorization changed')
        queue, qc = Path(batch['queue']), Path(batch['qc'])
        if not queue.exists():
            snap = gpu_snapshot(schedule['GPU'])
            if snap['uuid'] != schedule['GPU_UUID']:
                raise ValueError('GPU mapping changed')
            assert_idle(snap)
            if shutil.disk_usage(GEN).free < RESERVE:
                raise RuntimeError('150 GiB disk reserve reached')
            command = ['taskset','-c',schedule['native_CPU_cores'],'nice','-n','15',str(PYTHON),'-B','-u','-m',
                'examples.baselines.hypericl_task_library.generation.scale192_v1.queue',
                '--plan',schedule['plan']['path'],'--authorization',auth['path'],
                '--gpu',str(schedule['GPU']),'--task',batch['task'],'--output',str(queue)]
            save(args.output/'state.json', dict(status='native_batch_running', batch=batch['name'],
                completed_batches=completed, planned_attempts=schedule['native_attempt_budget'],updated_unix=time.time()))
            with (args.output/(batch['name']+'.log')).open('x') as log:
                result = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'Native queue exited {result.returncode}; no automatic retry: {queue}')
        terminal_path = queue/'summary.json'
        if not terminal_path.exists():
            raise RuntimeError('Queue lacks normal terminal summary; preserve attempted identities: ' + str(queue))
        terminal = read(terminal_path)
        if {r['job_id'] for r in terminal['completed']} != set(batch['job_ids']):
            raise ValueError('Completed batch identities differ')
        if not (qc/'report.json').exists():
            if qc.exists():
                raise RuntimeError('Interrupted QC output requires separate recovery output; no overwrite: '+str(qc))
            command = ['taskset','-c',str(schedule['QC_CPU_core']),'nice','-n','15',str(PYTHON),'-B','-u',
                str(METHODS/'human_to_robot_pipeline/verify_batch.py'), '--plan',schedule['plan']['path'],
                '--queue',str(queue),'--output',str(qc)]
            for previous in prior_records:
                command += ['--prior-records',previous]
            save(args.output/'state.json', dict(status='CPU_verifying_batch',batch=batch['name'],
                completed_batches=completed,updated_unix=time.time()))
            with (args.output/(batch['name']+'.qc.log')).open('x') as log:
                result = subprocess.run(command,cwd=REPO,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'CPU verification exited {result.returncode}: {qc}')
        report = read(qc/'report.json')
        if not report['passed']:
            raise RuntimeError('Batch failed independent verification; stopping expansion: '+str(qc))
        record_path = qc/'verified_robot_records.json'
        prior_records.append(str(record_path))
        completed.append(dict(batch=batch['name'], task=batch['task'],attempted=terminal['attempted'],
            native_successes=terminal['native_successes'], CPU_passed=len(read(record_path)['records']),
            qc_report=bind(qc/'report.json'), training_admitted=0))
        save(args.output/'state.json',dict(status='batch_verified',completed_batches=completed,
            new_attempted=sum(b['attempted'] for b in completed),
            new_CPU_passed=sum(b['CPU_passed'] for b in completed),updated_unix=time.time()))
        print(json.dumps(completed[-1]),flush=True)
    save(args.output/'summary.json',dict(status='finite_generation_and_CPU_verification_complete',
        completed_batches=completed, new_attempted=sum(b['attempted'] for b in completed),
        new_CPU_passed=sum(b['CPU_passed'] for b in completed),elapsed_seconds=time.time()-started,
        whole_batch_visual_review_pending=True, training_admitted=0, model_training=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--output',type=Path,required=True)
    prep.add_argument('--owner-audit',type=Path,required=True)
    prep.add_argument('--prior-qc',type=Path,required=True)
    prep.add_argument('--gpu',type=int,default=2)
    prep.add_argument('--batch-size',type=int,default=8,choices=range(1,17))
    prep.add_argument('--cpu-cores',default='216,217')
    prep.add_argument('--qc-core',type=int,default=218)
    runner = sub.add_parser('run')
    runner.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    try:
        (prepare if args.command=='prepare' else run)(args)
    except Exception as exc:
        if args.command=='run':
            save(args.output/'STOPPED.json',dict(reason=str(exc),time_unix=time.time(),
                no_automatic_retry=True, original_outputs_preserved=True))
        raise


if __name__=='__main__':
    main()
