"""Resume bounded CPU compilation of every canonical native source.

No GPU jobs are launched here. Geometry-equivalent old proposal sets are run
through the unchanged native qualifier, never copied across changed geometry.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import Counter
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import numpy as np
import trimesh
from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write
from resimpair_assets import metric_variants_v2, metric_native_v6
from resimpair_tasks.cudex_bulk_queue_v2 import atomic, process_identity, process_alive

VERSION='metric_population_CPU_continuation_v6'


def equivalent_geometry(parent_ref,new_ref):
    parent,new=read(resolve(parent_ref)),read(resolve(new_ref))
    if not np.allclose(parent['source_to_asset'],new['source_to_asset'],rtol=1e-6,atol=1e-8):return False
    if len(parent['collisions'])!=len(new['collisions']):return False
    for left,right in zip(parent['collisions'],new['collisions']):
        a=trimesh.load(resolve(left),force='mesh',process=False).vertices
        b=trimesh.load(resolve(right),force='mesh',process=False).vertices
        distance=np.linalg.norm(a[:,None]-b[None],axis=-1)
        if max(distance.min(0).max(),distance.min(1).max())>1e-6:return False
    return True



def selected_job(root,key):
    candidate=Path(root)/'recoveries_v6'/key/'job.json'
    return candidate if candidate.exists() else Path(root)/'jobs'/key/'job.json'


def initialize_continuation(root):
    root=Path(root).resolve();record=root/'continuation_v6.json'
    if record.exists():
        value=read(record)
        if value['driver']!=binding(__file__):raise ValueError('continuation_driver_changed')
        return value
    recover=[]
    for terminal in (root/'jobs').glob('*/terminal.json'):
        if read(terminal)['status']!='unknown_system_failure':continue
        failure=read(terminal.parent/'failure.json')
        if failure['error_type']!='FileNotFoundError' or '/metric/catalog.json' not in failure['error']:
            raise RuntimeError('different_unknown_system_error_requires_diagnosis')
        job=read(terminal.parent/'job.json')
        if not job.get('previous_compilation_result'):raise ValueError('bound_previous_lineage_required')
        recover.append({'old_job':binding(terminal.parent/'job.json'),'old_terminal':binding(terminal),'key':terminal.parent.name})
    value={'schema':VERSION,'original_protocol':binding(root/'protocol.json'),'driver':binding(__file__),
        'recovery_scope':recover,'reason':'follow_bound_transitive_compilation_lineage_before_finding_metric_catalog',
        'new_optimizer_jobs':0,'same_input_asset_failure_retries':0,'original_limits_and_deadline_unchanged':True,
        'created_unix':time.time()}
    write(record,value)
    for row in recover:
        old=read(resolve(row['old_job']));folder=root/'recoveries_v6'/row['key']
        write(folder/'job.json',{**old,'output':str(folder),'infrastructure_recovery':row,
            'continuation':binding(record)})
    return value

def worker(job_path):
    job_path=Path(job_path);job=read(job_path);out=Path(job['output']);start=time.monotonic()
    if (out/'worker_started.json').exists():raise ValueError('previous_worker_started_no_automatic_retry')
    write(out/'worker_started.json',{'job':binding(job_path),'identity':process_identity(os.getpid()),'unix':time.time()})
    if job['decision'].get('semantic_admission')=='pending':
        result={'schema':VERSION,'job':binding(job_path),'source_id':job['source_id'],'entries':[],
            'deferred_reason':'role_or_scale_evidence_pending','GPU_jobs':0,'API_calls':0}
        write(out/'result.json',result);return result
    if job.get('previous_failed_terminal'):
        result={'schema':VERSION,'job':binding(job_path),'source_id':job['source_id'],'entries':[],
            'deferred_reason':'unchanged_known_invalid_metric_input_no_retry','previous_failure':job['previous_failed_terminal'],
            'GPU_jobs':0,'API_calls':0}
        resolve(job['previous_failed_terminal']);write(out/'result.json',result);return result
    if job.get('previous_compilation_result'):
        previous_path=resolve(job['previous_compilation_result']);previous=read(previous_path)
        seen=set()
        while not (previous_path.parent/'metric/catalog.json').exists() and previous.get('previous_compilation_result'):
            if str(previous_path) in seen:raise ValueError('cyclic_previous_compilation_lineage')
            seen.add(str(previous_path));previous_path=resolve(previous['previous_compilation_result']);previous=read(previous_path)
        if previous['source_id']!=job['source_id']:raise ValueError('previous_compilation_source_binding_changed')
        for i,item in enumerate(previous['entries']):
            if item.get('role')=='manipulation' and not item.get('geometry_qualified'):
                old_native=previous_path.parent/f'native_{i}/result.json'
                metric=read(previous_path.parent/'metric/catalog.json')['assets'][i]
                refined=metric_native_v6.compile_variant(resolve(metric),out/f'refined_previous_{i}',old_native)
                previous['entries'][i]={**item,**refined,'status':'needs_new_grasp_optimization' if refined['geometry_qualified'] else 'native_refinement_still_failed'}
        for item in previous['entries']:
            if item['source_id']!=job['source_id']:raise ValueError('reused_compilation_source_mismatch')
            resolve(item['asset'])
            if item.get('cache'):resolve(item['cache'])
        result={**previous,'job':binding(job_path),'previous_compilation_result':job['previous_compilation_result'],
            'CPU_compilation_reused':True,'elapsed_seconds':time.monotonic()-start}
        write(out/'result.json',result);return result
    value=metric_variants_v2.materialize(resolve(job['parent']),resolve(job['policy']),out/'metric')
    catalog=read(out/'metric/catalog.json');items=[]
    for i,ref in enumerate(catalog['assets']):
        d=read(resolve(ref));role=d['semantic_role']
        if role!='manipulation':
            items.append({'source_id':d['source_id'],'variant_id':d['geometry_variant_key'],
                'asset':ref,'relative_scale':d['size_provenance']['selected_relative_scale'],
                'role':role,'status':'static_numeric_geometry_only','native_physics':'pending_if_required',
                'physical_status':'unmeasured','grasp_cache':None})
            continue
        compiled=metric_native_v6.compile_variant(resolve(ref),out/f'native_{i}')
        entry={**compiled,'source_aliases':job.get('source_aliases',[]),'role':role,
            'status':'needs_new_grasp_optimization' if compiled['geometry_qualified'] else 'native_qualification_failed'}
        old=job.get('prior_entry')
        if old and compiled['geometry_qualified'] and equivalent_geometry(job['parent'],compiled['asset']):
            from resimpair_benchmark.proposals import qualify
            cache=read(resolve(old['cache']))
            if cache['signature']['descriptor']!=job['parent']:raise ValueError('old_cache_parent_binding_mismatch')
            raw=cache['signature']['raw_proposals'];resolve(raw)
            result=qualify(compiled['asset'],raw,out/f'requalified_cache_{i}',geometry='native',maximum=256)
            entry.update(cache=result['cache'],retained=result['retained'],
                prior_cache=old['cache'],prior_descriptor=job['parent'],
                pose_origin='unchanged_geometry_old_raw_proposals_requalified_with_new_descriptor',
                qualification_stage='native_cook_COM_inertia_CPU_settle_and_grasp_requalification',
                optimizer_status='not_rerun_geometry_equivalent',
                status='executable_offline_cache' if result['retained'] else 'empty_requalified_old_cache')
        items.append(entry)
    result={'schema':VERSION,'job':binding(job_path),'source_id':job['source_id'],
        'entries':items,'elapsed_seconds':time.monotonic()-start,'GPU_jobs':0,'API_calls':0}
    write(out/'result.json',result);return result


def prepare(manifest_path,root):
    root=Path(root).resolve()
    if root.exists():raise FileExistsError(root)
    manifest=read(manifest_path);root.mkdir(parents=True)
    code=[binding(__file__),binding(metric_native_v6.__file__),binding(metric_variants_v2.__file__),
          binding(ROOT/'resimpair_assets/metric_native_v3.py'),binding(ROOT/'resimpair_assets/metric_native_v4.py'),binding(ROOT/'resimpair_assets/metric_native_v5.py'),binding(ROOT/'resimpair_assets/metric_variants_v1.py'),binding(ROOT/'runtime/collision_cpu_deps/vhacdx.cpython-312-x86_64-linux-gnu.so'),binding(ROOT/'resimpair_assets/metric_semantics_v1.py'),binding(ROOT/'configs/metric_population_v7.json')]
    protocol={'schema':VERSION,'manifest':binding(manifest_path),'code':code,
        'maximum_cpu_workers':4,'per_source_timeout_seconds':120,'deadline_unix':manifest['deadline_unix'],
        'maximum_sources':len(manifest['jobs']),'GPU_jobs':0,'API_calls':0,'automatic_retries':False,
        'each_source_single_immutable_terminal':True,'created_unix':time.time()}
    write(root/'protocol.json',protocol)
    for entry in manifest['jobs']:
        folder=root/'jobs'/entry['key']
        prior=ROOT/'runs/metric_assets_20261030_v4/batch/jobs'/entry['key']/'terminal.json'
        if prior.exists() and read(prior)['status']=='asset_numeric_or_geometry_failed':entry={**entry,'previous_failed_terminal':binding(prior)}
        write(folder/'job.json',{**entry,'output':str(folder),'protocol':binding(root/'protocol.json')})
    return protocol


def refresh(root, cudex_index=None):
    root=Path(root);p=read(root/'protocol.json');manifest=read(resolve(p['manifest']))
    states=read(resolve(manifest['source_state_index']));by_state={x['source_id']:x for x in states['entries']}
    ready=[];cache=[];static=[];pending_grasp=[];outcomes=Counter();statuses=[]
    for job in manifest['jobs']:
        folder=selected_job(root,job['key']).parent;path=folder/'terminal.json';sid=job['source_id']
        result=read(folder/'result.json') if (folder/'result.json').exists() else None
        term=read(path) if path.exists() else {'status':'running' if (folder/'worker_started.json').exists() else 'pending'}
        outcomes[term['status']]+=1
        by_state[sid].update(batch_status=term['status'],terminal=binding(path) if path.exists() else None)
        if result:
            by_state[sid]['variants']=result['entries']
            for e in result['entries']:
                if e['role']!='manipulation':static.append(e['asset']);continue
                if e['geometry_qualified']:
                    ready.append(e);static.append(e['asset'])
                    if e.get('retained',0)>0:cache.append(e)
                    elif e['status']=='needs_new_grasp_optimization':pending_grasp.append(e)
        statuses.append({'source_id':sid,**term})
    if cudex_index and Path(cudex_index).exists():
        existing={x['variant_id'] for x in cache}
        cache += [e for e in read(cudex_index)['entries'] if e['variant_id'] not in existing]
    summary={'schema':VERSION,'counts':dict(outcomes),'native_ready':len(ready),'static_descriptors':len(static),
        'nonempty_cache_entries':len(cache),'poses':sum(x['retained'] for x in cache),
        'new_GPU_candidates':len(pending_grasp),'updated_unix':time.time(),'GPU_jobs':0,'API_calls':0}
    atomic(root/'native_ready_index.json',{'schema':'resimpair_grasp_cache_index_v1','entries':ready})
    atomic(root/'native_cache_index.json',{'schema':'resimpair_grasp_cache_index_v1','entries':cache})
    atomic(root/'pending_grasp_index.json',{'schema':'resimpair_grasp_cache_index_v1','entries':pending_grasp})
    atomic(root/'static_catalog.json',{'schema':'resimpair_mesh_catalog_v1','assets':static})
    atomic(root/'source_state_index.json',{**states,'batch_summary':summary})
    atomic(root/'status.json',summary)
    return summary


def run(root,maximum=None):
    root=Path(root).resolve();p=read(root/'protocol.json')
    for ref in p['code']:resolve(ref)
    amendment=read(root/'continuation_v6.json')
    if amendment['original_protocol']!=binding(root/'protocol.json') or amendment['driver']!=binding(__file__):raise ValueError('bound_continuation_amendment_required')
    manifest=read(resolve(p['manifest']));pending=[]
    with (root/'.run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for job in manifest['jobs']:
            path=selected_job(root,job['key']);folder=path.parent
            if (folder/'terminal.json').exists():continue
            if (folder/'worker_started.json').exists():
                started=read(folder/'worker_started.json')
                if (folder/'result.json').exists():
                    write(folder/'terminal.json',{'status':'completed','recovered_existing_result':True,
                        'result':binding(folder/'result.json')});continue
                raise RuntimeError('unknown_prior_source_job_recover_without_resubmission:'+str(path))
            pending.append(path)
        groups={}
        for path in pending:
            source=read(path)['source_id'].split(':')[0];groups.setdefault(source,[]).append(path)
        pending=[]
        while any(groups.values()):
            for source in sorted(groups):
                if groups[source]:pending.append(groups[source].pop(0))
        if maximum:pending=pending[:maximum]
        def execute(path):
            folder=path.parent
            if time.time()>=p['deadline_unix'] or (root/'STOP').exists():return None
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            with (folder/'worker.log').open('x') as log:
                child=subprocess.Popen([sys.executable,'-u','-m','resimpair_assets.metric_batch_continue_v6','worker','--job',str(path)],
                    cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                write(folder/'process.json',{'identity':process_identity(child.pid),'unix':time.time()})
                try:code=child.wait(timeout=p['per_source_timeout_seconds'])
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    code=124
            if not code and (folder/'result.json').exists():
                result=read(folder/'result.json')
                terminal={'status':result.get('deferred_reason','completed'),'result':binding(folder/'result.json')}
            elif code==124:terminal={'status':'source_timeout','returncode':code,'retry':False}
            else:
                failure=read(folder/'failure.json') if (folder/'failure.json').exists() else {}
                terminal={'status':failure.get('classification','unknown_system_failure'),
                    'returncode':code,'failure':binding(folder/'failure.json') if failure else None,'retry':False}
            write(folder/'terminal.json',terminal);return terminal
        completed=0
        # Only four futures may own a source; a STOP file drains them and leaves
        # every not-started source pending without a reservation or retry.
        iterator=iter(pending); active={}; exhausted=False; failed=False
        with ThreadPoolExecutor(max_workers=4) as pool:
            while active or not exhausted:
                while len(active)<4 and not exhausted and not failed:
                    if (root/'STOP').exists() or time.time()>=p['deadline_unix']:
                        exhausted=True;break
                    path=next(iterator,None)
                    if path is None:exhausted=True;break
                    active[pool.submit(execute,path)]=path
                if not active:break
                done,_=wait(active,timeout=5,return_when=FIRST_COMPLETED)
                for future in done:
                    active.pop(future);value=future.result();completed+=1
                    if completed%25==0:print(refresh(root),flush=True)
                    if value and value['status']=='unknown_system_failure':
                        failed=True;exhausted=True
        if failed:raise RuntimeError('unknown_system_failure_paused_new_CPU_admission')
        return refresh(root)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','continue','run','refresh','worker'])
    p.add_argument('--root',type=Path);p.add_argument('--manifest',type=Path);p.add_argument('--job',type=Path)
    p.add_argument('--maximum',type=int);p.add_argument('--cudex-index',type=Path);a=p.parse_args()
    if a.command=='worker':
        try:print(worker(a.job))
        except Exception as exc:
            # Binding/infrastructure/program faults stop; bounded numeric and mesh
            # faults are durable per-source outcomes and do not kill a campaign.
            message=str(exc);typ=type(exc).__name__
            known=(isinstance(exc,ValueError) and not any(x in message for x in ['hash changed','binding_mismatch','previous_worker']))
            write(a.job.parent/'failure.json',{'error_type':typ,'error':message,'traceback':traceback.format_exc(),
                'classification':'asset_numeric_or_geometry_failed' if known else 'unknown_system_failure'})
            raise
    elif a.command=='continue':
        initialize_continuation(a.root);print(run(a.root,a.maximum))
    else:print(prepare(a.manifest,a.root) if a.command=='prepare' else run(a.root,a.maximum) if a.command=='run' else refresh(a.root,a.cudex_index))
