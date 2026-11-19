"""Finite new-lineage metric cuDex jobs, durable single submission per variant.

One shared immutable allowance covers incremental input batches. Known asset
errors continue; unknown jobs stop admission and require output recovery.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write
from resimpair_grasp.cudex_native_com_v4 import condition
from resimpair_tasks import cudex_bulk_queue_v2 as retained
from corl2026_workshop.experiments.autonomous_pipeline_v4 import resources

VERSION='metric_grasp_campaign_v6'
MODULE='resimpair_grasp.cudex_native_com_v4'
MANIP=ROOT/'runs/simulator_alignment_20261028/constraint_candidate_v2r1/manip_nonnegative.yml'


def prepare(root, limit=24):
    root=Path(root).resolve()
    if root.exists(): raise FileExistsError(root)
    if type(limit) is not int or not 1<=limit<=24: raise ValueError('registered_pilot_limit_1_to_24')
    root.mkdir(parents=True)
    code=[binding(ROOT/'resimpair_grasp/cudex_native_com_v4.py'), binding(__file__),
          binding(ROOT/'resimpair_assets/metric_native_v3.py'),
          binding(ROOT/'resimpair_grasp/cudex_seed_initialization_v2.py')]
    code += [binding(p) for p in retained.implementation_files()]
    protocol={'schema':VERSION,'gpu_job_limit':limit,'cpu_export_timeout_seconds':120,
        'gpu_timeout_seconds':120,'gpu_indices':[1,3], 'parallelism':2,
        'minimum_free_gib':12, 'deadline_unix':time.time()+86400,
        'created_unix':time.time(),'prior_scope_cumulative_gpu_jobs':3724,
        'prior_scope':str(ROOT/'runs/cudex_bulk_5524_20261028'),
        'prior_scope_limit':5528,'prior_scope_reused_budget':False,
        'older_lineage_gpu_jobs':142,'API_calls':0,'no_automatic_retries':True,
        'unknown_stops_admission':True,'manip':binding(MANIP),'code':code,
        'configuration':[binding(p) for p in retained._configuration_files([MANIP])],
        'new_lineage_reason':'source_aware_metric_geometry_and_native_COM_condition'}
    write(root/'protocol.json',protocol); return protocol


def _validate(root):
    p=read(root/'protocol.json')
    if p.get('schema')!=VERSION or p['gpu_job_limit']>24 or p['gpu_indices']!=[1,3]:
        raise ValueError('invalid_frozen_pilot_allowance')
    for r in p['code']+p['configuration']: resolve(r)
    if p['no_automatic_retries'] is not True: raise ValueError('retries_not_allowed')
    return p


def status(root):
    root=Path(root); p=read(root/'protocol.json'); jobs=[]
    for path in sorted((root/'jobs').glob('*/job.json')):
        folder=path.parent
        r={'job':binding(path),'source_id':read(path)['source_id'],
           'variant_id':read(path)['variant_id'], 'reserved':(folder/'reservation.json').exists()}
        for name in ['terminal','result']:
            if (folder/(name+'.json')).exists(): r[name]=read(folder/(name+'.json'))
        jobs.append(r)
    return {'schema':VERSION,'new_gpu_reservations':sum(x['reserved'] for x in jobs),
        'limit':p['gpu_job_limit'],'completed':sum(x.get('terminal',{}).get('status')=='completed' for x in jobs),
        'nonempty_caches':sum(x.get('result',{}).get('retained',0)>0 for x in jobs),
        'poses':sum(x.get('result',{}).get('retained',0) for x in jobs),
        'prior_scope_cumulative_gpu_jobs':p['prior_scope_cumulative_gpu_jobs'],
        'historical_total_including_older_lineage':p['prior_scope_cumulative_gpu_jobs']+p['older_lineage_gpu_jobs']+sum(x['reserved'] for x in jobs),
        'jobs':jobs,'API_calls':0}


def run(root, index):
    root=Path(root).resolve(); p=_validate(root); stop=threading.Event(); mutex=threading.RLock()
    entries=read(index)['entries']
    with (root/'.supervisor.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pending=[]
        for e in entries:
            d=read(resolve(e['asset'])); variant=e.get('variant_id',d.get('variant_id'))
            if not variant: raise ValueError('stable_variant_id_required')
            folder=root/'jobs'/variant[:24]; job=folder/'job.json'
            if job.exists():
                if read(job)['asset']!=e['asset']: raise ValueError('variant_identity_changed')
                if not (folder/'terminal.json').exists():
                    raise RuntimeError('existing_unknown_job_recover_without_resubmission:'+str(job))
                continue
            if not e.get('geometry_qualified', True): continue
            condition(e['asset'])
            write(job,{'source_id':d['source_id'],'variant_id':variant,'asset':e['asset'],
                'native_support':d['native_support'],'manip':p['manip'],'seed':20261030,
                'output':str(folder),'content_key':variant,'source_aliases':e.get('source_aliases',[]),
                'phase':'new_metric_qualification','protocol':binding(root/'protocol.json')})
            pending.append(job)
        existing=status(root)['new_gpu_reservations']
        if existing+len(pending)>p['gpu_job_limit']: raise ValueError('total_registered_allowance_exceeded')
        def execute(job,gpu):
            folder=job.parent
            if stop.is_set(): return None
            with ExitStack() as stack:
                lock_fds=[]
                for path in resources.lock_paths(gpu):
                    f=stack.enter_context(path.open('a+')); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    lock_fds.append(f.fileno())
                snap=resources.gpu_snapshot(gpu); resources.check_idle(snap)
                if time.time()>=p['deadline_unix']: raise RuntimeError('campaign_deadline_reached')
                with mutex:
                    n=status(root)['new_gpu_reservations']
                    if n>=p['gpu_job_limit']: raise ValueError('cumulative_allowance_exhausted')
                    write(folder/'reservation.json',{'job':binding(job),'unix':time.time(),
                        'gpu':snap,'historical_ordinal':p['prior_scope_cumulative_gpu_jobs']+n+1})
                env=retained._environment(job,snap['uuid']); start=time.monotonic()
                with (folder/'worker.log').open('x') as log:
                    child=subprocess.Popen([sys.executable,'-u','-m',MODULE,'worker','--job',str(job)],
                        cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                        pass_fds=tuple(lock_fds))
                    write(folder/'process.json',{'identity':retained.process_identity(child.pid)})
                    try: code=child.wait(timeout=p['gpu_timeout_seconds'])
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGTERM)
                        try: child.wait(timeout=5)
                        except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL); child.wait()
                        code=124
                write(folder/'gpu_terminal.json',{'returncode':code,'elapsed_seconds':time.monotonic()-start})
            if code:
                log=(folder/'worker.log').read_text(); tail=log.rstrip().splitlines()[-1] if log.strip() else ''
                known=any(token in tail for token in ['expected_256_finite_native_tcp_poses',
                    'expected finite floating hand pose','empty_surface_seed_pool',
                    'cannot reshape tensor of 0 elements'])
                result={'status':'asset_failed' if known else 'needs_recovery',
                        'stage':'gpu','returncode':code,'error_tail':tail,'retry':False}
                if not known: stop.set()
            else:
                retained.validate_optimizer(job)
                with (folder/'export.log').open('x') as log:
                    try:
                        code=subprocess.run([sys.executable,'-u','-m',MODULE,'export','--job',str(job)],
                            cwd=ROOT,env=retained._environment(job),stdout=log,stderr=subprocess.STDOUT,
                            timeout=p['cpu_export_timeout_seconds']).returncode
                    except subprocess.TimeoutExpired: code=124
                if code: stop.set(); result={'status':'needs_recovery','stage':'cpu_export','returncode':code}
                else:
                    value=retained.validate_result(job)
                    result={'status':'completed','retained':value['retained'],'result':binding(folder/'result.json'),
                        'optimizer_seconds':value['worker_seconds'],'export_seconds':value['export_seconds']}
            write(folder/'terminal.json',result)
            with mutex: refresh(root)
            return result
        # Persistent per-GPU serial lanes prevent a second thread taking the same card.
        def lane(gpu,jobs):
            for job in jobs:
                if stop.is_set(): break
                execute(job,gpu)
        with ThreadPoolExecutor(max_workers=p['parallelism']) as pool:
            futures=[pool.submit(lane,gpu,pending[i::p['parallelism']]) for i,gpu in enumerate(p['gpu_indices'])]
            for f in as_completed(futures):
                try: f.result()
                except Exception:
                    stop.set(); raise
        refresh(root)
    return status(root)


def refresh(root):
    root=Path(root); s=status(root); entries=[]
    for item in s['jobs']:
        if item.get('result',{}).get('retained',0)<=0: continue
        job=read(resolve(item['job'])); result=item['result']; d=read(resolve(job['asset']))
        entries.append({'source_id':job['source_id'],'variant_id':job['variant_id'],
            'relative_scale':d['size_provenance']['selected_relative_scale'],'asset':job['asset'],
            'cache':result['cache'],'retained':result['retained'],'job':item['job'],
            'native_support':d['native_support'],'native_properties':d['native_properties'],
            'qualification_stage':'native_geometry_support_and_grasp','physical_status':'unmeasured',
            'source_aliases':job['source_aliases'],'optimizer_status':'completed'})
    retained.atomic(root/'native_cache_index.json',{'schema':'resimpair_grasp_cache_index_v1',
        'entries':entries,'protocol':binding(root/'protocol.json'),'physical_success_inferred':False})
    retained.atomic(root/'status.json',s)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','run','status'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--index',type=Path)
    p.add_argument('--limit',type=int,default=24);a=p.parse_args()
    print(prepare(a.root,a.limit) if a.command=='prepare' else run(a.root,a.index) if a.command=='run' else status(a.root))
