"""Resume the same finite cuDex ledger with bound resource/input amendments.

No budget reset, source identity change, resubmission or original protocol edit.
Already-running jobs must drain under v8 before this supervisor takes the lock.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from contextlib import ExitStack
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from resimpair_grasp.common import ROOT,binding,digest,read,resolve,write
from resimpair_tasks import metric_cudex_stream_v8 as retained
from resimpair_tasks.metric_cudex_stream_v8 import base,resources,WORKER,status,refresh,_run_process,_asset_error,_parent_requalification,equivalent_geometry,condition,inspect


def amended_config(root):
    root=Path(root)
    p=retained.validate(root)
    receipt=read(root/'resource_continuation_v9.json')
    if receipt['parent_protocol']!=binding(root/'protocol.json') or receipt['source_plan']!=p['source_plan']:
        raise ValueError('continuation_scope_binding_mismatch')
    for ref in [receipt['parent_protocol'],receipt['source_plan'],receipt['driver'],receipt['retained_supervisor']]+receipt['supplemental_native_indexes']:
        resolve(ref)
    if receipt['driver']!=binding(__file__) or receipt['retained_supervisor']!=binding(retained.__file__):
        raise ValueError('continuation_driver_binding_mismatch')
    if receipt['maximum_cumulative_optimizer_jobs']!=3080 or receipt['additional_optimizer_allowance']!=0:
        raise ValueError('resource_amendment_cannot_increase_budget')
    gpus=receipt['gpu_indices'];parallel=receipt['maximum_gpu_parallelism']
    if len(gpus)!=len(set(gpus)) or not set(gpus).issubset({0,1,2,3}) or not 1<=parallel<=4 or parallel>len(gpus):
        raise ValueError('bounded_per_card_resources_required')
    if 2 in gpus and not receipt.get('GPU2_explicit_root_admission',False):
        raise ValueError('GPU2_requires_explicit_resource_admission')
    return {**p,'gpu_indices':gpus,'maximum_gpu_parallelism':parallel,
        'supplemental_native_indexes':receipt['supplemental_native_indexes']}


def run(root):
    root=Path(root).resolve();p=amended_config(root);source_state={r['source_id']:r for r in read(resolve(p['source_plan']))['entries']}
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
            indexes=[Path(p['cpu_root'])/'pending_grasp_index.json']+[resolve(ref) for ref in p['supplemental_native_indexes']]
            snapshot={'entries':[entry for index in indexes if index.exists() for entry in read(index)['entries']]}
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
                    'semantic_scale_admission':gate,'source_plan':p['source_plan'],'protocol':binding(root/'protocol.json'),
                    'continuation_receipt':binding(root/'resource_continuation_v9.json')})
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
        with ThreadPoolExecutor(max_workers=p['maximum_gpu_parallelism']) as pool:
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['run','status'])
    p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    result=run(a.root) if a.command=='run' else status(a.root)
    print({k:v for k,v in result.items() if k!='rows'})
