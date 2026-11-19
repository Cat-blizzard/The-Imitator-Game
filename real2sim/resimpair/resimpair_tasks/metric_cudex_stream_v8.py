"""Finite source-bound cuDex consumer for a live CPU metric-asset campaign.

Parent pilot reservations count in the same 3,080-job allowance. A source's
current base variant gets at most one optimizer attempt. Known asset failures
are terminal; unknown process/output states stop admission until recovered.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from contextlib import ExitStack
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from resimpair_grasp.common import ROOT,binding,digest,read,resolve,write
from resimpair_grasp.cudex_native_com_v4 import condition
from resimpair_assets.metric_admission_v1 import inspect
from resimpair_assets.metric_batch_v3 import equivalent_geometry
from resimpair_tasks import cudex_bulk_queue_v2 as base
from corl2026_workshop.experiments.autonomous_pipeline_v4 import resources

VERSION='metric_cudex_source_stream_v8'
WORKER='resimpair_grasp.cudex_native_com_v4'


def prepare(root,cpu_root,source_plan,parent):
    root,cpu_root,parent=map(lambda p:Path(p).resolve(),[root,cpu_root,parent])
    if root.exists():raise FileExistsError(root)
    states=read(source_plan);eligible=[r['source_id'] for r in states['entries'] if
        r.get('role')=='manipulation' and r.get('semantic_admission')=='engineered_role_and_scale_candidate']
    if len(eligible)!=3080:raise ValueError('root_authorized_3080_source_plan_required')
    parent_protocol=read(parent/'protocol.json');parents=[];spent=0
    for path in sorted((parent/'jobs').glob('*/job.json')):
        folder=path.parent
        if not (folder/'reservation.json').exists():continue
        spent+=1
        if not (folder/'terminal.json').exists():raise ValueError('recover_parent_unknown_jobs_before_extending')
        if read(folder/'terminal.json')['status']!='completed':raise ValueError('parent_failed_job_requires_explicit_recovery_audit')
        result=base.validate_result(path)
        parents.append({'job':binding(path),'result':binding(folder/'result.json'),
            'source_id':read(path)['source_id'],'asset':read(path)['asset'],'retained':result['retained']})
    if spent>24:raise ValueError('parent_allowance_exceeded')
    config=parent_protocol['manip'];resolve(config)
    recipe=digest({'source_plan':binding(source_plan),'worker':binding(ROOT/'resimpair_grasp/cudex_native_com_v4.py'),
        'manip':config,'seed':20261030,'base_variant_only':True})
    registry=ROOT/'runtime/metric_cudex_stream_registry_v8';registry.mkdir(parents=True,exist_ok=True)
    with (registry/'.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        record=registry/(recipe+'.json')
        if record.exists():raise ValueError('same_recipe_resume_existing_root:'+read(record)['root'])
        root.mkdir(parents=True)
        write(record,{'recipe':recipe,'root':str(root),'parent':str(parent),'maximum_cumulative_optimizer_jobs':3080})
    code=[binding(__file__),binding(ROOT/'resimpair_grasp/cudex_native_com_v4.py'),
        binding(ROOT/'resimpair_grasp/cudex_seed_initialization_v2.py'),
        binding(ROOT/'resimpair_assets/metric_admission_v1.py'),binding(ROOT/'resimpair_assets/metric_batch_v3.py')]
    code += [binding(p) for p in base.implementation_files()]
    write(root/'protocol.json',{'schema':VERSION,'source_plan':binding(source_plan),
        'eligible_sources':eligible,'cpu_root':str(cpu_root),'cpu_protocol':binding(cpu_root/'protocol.json'),
        'parent_protocol':binding(parent/'protocol.json'),'parent_jobs':parents,
        'parent_original_allowance':24,'parent_GPU_spent':spent,'parent_unsubmitted_allowance_reclaimed':24-spent,
        'additional_authorized_allowance':3056,'maximum_cumulative_optimizer_jobs':3080,
        'maximum_new_optimizer_jobs':3080-spent,'per_source_current_base_attempt_limit':1,
        'gpu_indices':[3,1,0],'maximum_gpu_parallelism':3,'minimum_free_gib':12,
        'gpu_timeout_seconds':120,'export_timeout_seconds':120,'poll_seconds':10,
        'deadline_unix':read(cpu_root/'protocol.json')['deadline_unix'],
        'created_unix':time.time(),'manip':config,'code':code,
        'configuration':[binding(p) for p in base._configuration_files([resolve(config)])],
        'registry':binding(record),'prior_older_GPU_jobs':142,'prior_old_scale_GPU_jobs':3724,
        'new_lineage_does_not_modify_old_queue':True,'API_calls':0,'unknown_never_resubmitted':True,
        'empty_or_numerical_asset_results_terminal':True,'default_physical_success_inferred':False})
    return {'root':str(root),'parent_spent':spent,'additional_authorized':3056,
        'reclaimed_unsubmitted_parent_slots':24-spent,'remaining_total':3080-spent}


def validate(root):
    p=read(root/'protocol.json')
    if p['schema']!=VERSION or p['maximum_cumulative_optimizer_jobs']!=3080 or p['maximum_gpu_parallelism']>3:
        raise ValueError('registered_finite_scope_changed')
    for ref in p['code']+p['configuration']+[p['source_plan'],p['cpu_protocol'],p['parent_protocol'],p['registry']]:resolve(ref)
    return p


def status(root):
    root=Path(root);p=read(root/'protocol.json');rows=[];reserved=0
    for path in sorted((root/'jobs').glob('*/job.json')):
        folder=path.parent;job=read(path);reservation=(folder/'reservation.json').exists();reserved+=reservation
        terminal=read(folder/'terminal.json') if (folder/'terminal.json').exists() else None
        rows.append({'source_id':job['source_id'],'variant_id':job['variant_id'],'job':binding(path),
            'reserved_optimizer':reservation,'terminal':terminal,
            'result':binding(folder/'result.json') if (folder/'result.json').exists() else None})
    if reserved+p['parent_GPU_spent']>3080:raise ValueError('cumulative_budget_exceeded')
    return {'schema':VERSION,'new_GPU_reservations':reserved,'parent_GPU_spent':p['parent_GPU_spent'],
        'cumulative_new_scale_GPU_jobs':reserved+p['parent_GPU_spent'],'limit':3080,
        'historical_GPU_jobs_including_old_scales':reserved+p['parent_GPU_spent']+p['prior_old_scale_GPU_jobs']+p['prior_older_GPU_jobs'],
        'registered_sources':len(rows),'terminal_sources':sum(r['terminal'] is not None for r in rows),
        'nonempty_caches':sum((r['terminal'] or {}).get('retained',0)>0 for r in rows),
        'poses':sum((r['terminal'] or {}).get('retained',0) for r in rows),'rows':rows,'API_calls':0}


def refresh(root,state='running'):
    root=Path(root);s=status(root);entries=[]
    for row in s['rows']:
        if not row['result'] or (row['terminal'] or {}).get('status')!='completed':continue
        r=read(resolve(row['result']));job=read(resolve(row['job']))
        if not r.get('retained',0):continue
        e=job['entry'];entries.append({**e,'cache':r['cache'],'retained':r['retained'],
            'job':row['job'],'result':row['result'],'optimizer_status':r.get('optimizer_status','completed'),
            'physical_status':'unmeasured','semantic_scale_admission':job['semantic_scale_admission']})
    base.atomic(root/'native_cache_index.json',{'schema':'resimpair_grasp_cache_index_v1','entries':entries,
        'protocol':binding(root/'protocol.json')})
    base.atomic(root/'status.json',{**s,'state':state,'updated_unix':time.time(),
        'identity':base.process_identity(os.getpid())})
    return s


def _run_process(command,folder,stage,timeout,env,fds=()):
    # A reservation is written before the child. Interrupted attempts are
    # recovered from process identity + outputs, never re-launched implicitly.
    with (folder/(stage+'.log')).open('x') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,pass_fds=tuple(fds))
        write(folder/(stage+'_process.json'),{'identity':base.process_identity(child.pid)})
        start=time.monotonic()
        try:code=child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            code=124
    write(folder/(stage+'_terminal.json'),{'returncode':code,'elapsed_seconds':time.monotonic()-start})
    return code


def _asset_error(folder,code):
    lines=(folder/'worker.log').read_text(errors='replace').rstrip().splitlines();tail=lines[-1] if lines else ''
    tokens=('expected_256_finite_native_tcp_poses','expected finite floating hand pose',
        'cannot reshape tensor of 0 elements','empty_surface_seed_pool')
    known=code==124 or any(t in tail for t in tokens)
    return {'status':'asset_timeout' if code==124 else 'asset_numeric_failure' if known else 'needs_recovery',
        'stage':'gpu','returncode':code,'error_tail':tail,'retry':False,'retained':0}


def _parent_requalification(job,path,parent):
    from resimpair_benchmark.proposals import qualify
    old=read(resolve(parent['result']));folder=path.parent
    start=time.monotonic()
    value=qualify(job['asset'],old['raw'],folder/'parent_raw_requalification',geometry='native',maximum=256)
    result={'schema':'parent_raw_native_requalification_v1','asset':job['asset'],'cache':value['cache'],
        'retained':value['retained'],'raw':old['raw'],'parent_result':parent['result'],
        'optimizer_status':'not_rerun_equivalent_current_base_geometry','export_seconds':time.monotonic()-start,
        'worker_seconds':0,'GPU_jobs':0,'API_calls':0,'physical_success':'unmeasured'}
    write(folder/'result.json',result)
    terminal={'status':'completed','retained':value['retained'],'new_GPU_jobs':0,
        'parent_raw_requalified':True,'result':binding(folder/'result.json')}
    write(folder/'terminal.json',terminal);return terminal


def run(root):
    root=Path(root).resolve();p=validate(root);source_state={r['source_id']:r for r in read(resolve(p['source_plan']))['entries']}
    eligible=set(p['eligible_sources']);stop=threading.Event();mutex=threading.RLock();parents={}
    for row in p['parent_jobs']:
        d=read(resolve(row['asset']))
        if abs(d['size_provenance']['selected_relative_scale']-1)<1e-6:parents[row['source_id']]=row
    with (root/'.supervisor.lock').open('a+') as supervisor:
        fcntl.flock(supervisor,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Explicitly recover only completed saved optimizer/export artifacts.
        for path in (root/'jobs').glob('*/job.json'):
            folder=path.parent
            if (folder/'terminal.json').exists():
                if read(folder/'terminal.json')['status']=='needs_recovery':
                    raise RuntimeError('prior_unknown_terminal_requires_recovery:'+str(path))
                continue
            if any((folder/(name+'_process.json')).exists() and base.process_alive(read(folder/(name+'_process.json'))['identity']) for name in ['worker','export']):
                raise RuntimeError('existing_live_stage_wait_then_resume:'+str(path))
            if (folder/'reservation.json').exists() or (folder/'worker_started.json').exists():
                if (folder/'result.json').exists():
                    r=base.validate_result(path);write(folder/'terminal.json',{'status':'completed','retained':r['retained'],'recovered_saved_result':True});continue
                if (folder/'optimizer_result.json').exists() and not (folder/'export_process.json').exists():
                    base.validate_optimizer(path)
                    code=_run_process([sys.executable,'-u','-m',WORKER,'export','--job',str(path)],folder,'export',p['export_timeout_seconds'],base._environment(path))
                    if not code:
                        r=base.validate_result(path);write(folder/'terminal.json',{'status':'completed','retained':r['retained'],'recovered_saved_optimizer':True});continue
                if (folder/'worker_terminal.json').exists() and read(folder/'worker_terminal.json')['returncode']:
                    t=_asset_error(folder,read(folder/'worker_terminal.json')['returncode'])
                    write(folder/'terminal.json',t)
                    if t['status']!='needs_recovery':continue
                raise RuntimeError('unknown_stage_recovery_required_no_optimizer_resubmission:'+str(path))
        def discover():
            index=Path(p['cpu_root'])/'pending_grasp_index.json'
            if not index.exists():return
            snapshot=read(index)
            for e in snapshot['entries']:
                sid=e['source_id'];folder=root/'jobs'/digest(sid)[:24];path=folder/'job.json'
                if path.exists():
                    if read(path)['asset']!=e['asset']:raise ValueError('source_current_base_changed_requires_new_plan')
                    continue
                if sid not in eligible:raise ValueError('source_outside_frozen_dynamic_scope')
                d=read(resolve(e['asset']));gate=inspect(e,source_state[sid]);condition(e['asset'])
                if abs(d['size_provenance']['selected_relative_scale']-1)>1e-6:raise ValueError('stream_scope_current_base_only')
                if not gate['passed']:
                    write(root/'preflight_skips'/(digest(sid)[:24]+'.json'),{'source_id':sid,'asset':e['asset'],'gate':gate});continue
                write(path,{'source_id':sid,'variant_id':e['variant_id'],'asset':e['asset'],'entry':e,
                    'output':str(folder),'content_key':e['variant_id'],'manip':p['manip'],'seed':20261030,
                    'phase':'new_metric_base_stream','source_aliases':e.get('source_aliases',[]),
                    'semantic_scale_admission':gate,'source_plan':p['source_plan'],'protocol':binding(root/'protocol.json')})
        def execute(path,gpu):
            job=read(path);folder=path.parent;parent=parents.get(job['source_id'])
            if parent and equivalent_geometry(parent['asset'],job['asset']):return _parent_requalification(job,path,parent)
            with ExitStack() as stack:
                fds=[]
                try:
                    for lock_path in resources.lock_paths(gpu):
                        lock=stack.enter_context(lock_path.open('a+'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fds.append(lock.fileno())
                except BlockingIOError:return {'status':'resource_deferred_no_submission'}
                free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                if free<p['minimum_free_gib']*1024:return {'status':'resource_deferred_no_submission'}
                snap=resources.gpu_snapshot(gpu)
                with mutex:
                    spent=status(root)['new_GPU_reservations']
                    if spent>=p['maximum_new_optimizer_jobs']:stop.set();return {'status':'budget_exhausted'}
                    write(folder/'reservation.json',{'job':binding(path),'unix':time.time(),'gpu':snap,
                        'cumulative_new_scale_ordinal':p['parent_GPU_spent']+spent+1})
                code=_run_process([sys.executable,'-u','-m',WORKER,'worker','--job',str(path)],folder,'worker',p['gpu_timeout_seconds'],base._environment(path,snap['uuid']),fds)
            if code:
                t=_asset_error(folder,code);write(folder/'terminal.json',t)
                if t['status']=='needs_recovery':stop.set()
                return t
            base.validate_optimizer(path)
            code=_run_process([sys.executable,'-u','-m',WORKER,'export','--job',str(path)],folder,'export',p['export_timeout_seconds'],base._environment(path))
            if code:
                t={'status':'needs_recovery','stage':'export','returncode':code,'retry':False};stop.set()
            else:
                result=base.validate_result(path);t={'status':'completed','retained':result['retained'],
                    'result':binding(folder/'result.json'),'optimizer_seconds':result['worker_seconds'],'export_seconds':result['export_seconds']}
            write(folder/'terminal.json',t);return t
        active={};busy=set();last=0.;finished=False;deferred_until={}
        with ThreadPoolExecutor(max_workers=3) as pool:
            while not finished:
                if time.time()>=p['deadline_unix'] or (root/'STOP').exists():stop.set()
                if not stop.is_set():
                    discover();active_paths=set(active.values())
                    pending=[path for path in (root/'jobs').glob('*/job.json') if path not in active_paths
                        and not (path.parent/'terminal.json').exists() and time.time()>=deferred_until.get(path,0)]
                    for gpu in p['gpu_indices']:
                        if gpu in busy or not pending:continue
                        path=pending.pop(0);f=pool.submit(execute,path,gpu);active[f]=path;f.gpu=gpu;busy.add(gpu)
                if active:
                    done,_=wait(active,timeout=5,return_when=FIRST_COMPLETED)
                    for f in done:
                        path=active.pop(f);busy.remove(f.gpu)
                        try:
                            result=f.result()
                            if result['status']=='resource_deferred_no_submission':deferred_until[path]=time.time()+p['poll_seconds']
                        except Exception as exc:
                            write(root/'supervisor_failure.json',{'error':repr(exc),'unix':time.time()});stop.set()
                else:time.sleep(p['poll_seconds'])
                if time.monotonic()-last>=30:
                    refresh(root,'draining_or_needs_recovery' if stop.is_set() else 'running');last=time.monotonic()
                if stop.is_set() and not active:finished=True
                cpu_status=Path(p['cpu_root'])/'status.json'
                if cpu_status.exists() and not active:
                    counts=read(cpu_status)['counts']
                    pending_jobs=any(not (path.parent/'terminal.json').exists() for path in (root/'jobs').glob('*/job.json'))
                    if not counts.get('pending',0) and not counts.get('running',0) and not pending_jobs:finished=True
        return refresh(root,'stopped_pending_recovery_or_budget' if stop.is_set() else 'completed_current_CPU_scope')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','run','status'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--cpu-root',type=Path)
    p.add_argument('--source-plan',type=Path);p.add_argument('--parent',type=Path);a=p.parse_args()
    print(prepare(a.root,a.cpu_root,a.source_plan,a.parent) if a.command=='prepare' else run(a.root) if a.command=='run' else status(a.root))
