"""Finite, resumable raw-asset CPU and cuDex queues over one frozen population.

The CPU and GPU supervisors have separate locks. One failed asset is terminal;
an unknown infrastructure state stops admission. No implicit resubmission.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write
from resimpair_tasks.cudex_bulk_queue_v2 import atomic, process_identity, process_alive, _environment, validate_result, validate_optimizer
from resimpair_tasks.metric_cudex_stream_v8 import _run_process, _asset_error
from corl2026_workshop.experiments.autonomous_pipeline_v4 import resources

VERSION='raw_asset_population_rollout_v1'
SOURCE_PLAN=ROOT/'runs/metric_assets_20261030_v7/inventory_r1/source_state_index.json'
BASELINE=ROOT/'runs/semantic_scale_consumer_20261030_v1/final_consumer_v2/BUNDLE.json'
CUDA_PARENT=ROOT/'runs/metric_assets_20261030_v7/cudex_stream/protocol.json'
CPU_MODULE='resimpair_assets.raw_manipulation_v1'
GPU_MODULE='resimpair_grasp.cudex_native_com_v4'


def code_bindings():
    files=[__file__]
    files += [ROOT/p for p in [
        'resimpair_assets/raw_manipulation_v1.py','resimpair_assets/raw_mesh_io_v1.py',
        'resimpair_assets/raw_size_policy_v1.py','resimpair_assets/raw_metric_containers_v1.py',
        'resimpair_assets/metric_inventory_v7.py','resimpair_assets/metric_semantics_v1.py',
        'resimpair_assets/metric_visual_precision_v3.py','resimpair_assets/metric_variants_v1.py',
        'resimpair_assets/metric_variants_v2.py','resimpair_assets/metric_native_v3.py',
        'resimpair_assets/metric_native_v4.py','resimpair_assets/metric_native_v5.py','resimpair_assets/metric_native_v6.py',
        'resimpair_assets/metric_admission_v1.py','resimpair_grasp/assets.py','resimpair_grasp/common.py',
        'resimpair_grasp/cudex_native_com_v4.py','resimpair_grasp/cudex_constraint_v3.py',
        'resimpair_grasp/cudex_seed_initialization_v2.py','resimpair_benchmark/proposals.py',
        'resimpair_tasks/metric_cudex_stream_v8.py','resimpair_tasks/cudex_bulk_queue_v2.py',
        'corl2026_workshop/experiments/autonomous_pipeline_v4/resources.py',
        'resimpair_tabletop/native_mass.py','resimpair_tabletop/geometry.py',
        'configs/metric_population_v7.json','configs/metric_numeric_bounds_v2.json']]
    return [binding(p) for p in files]


def prepare(root):
    root=Path(root).resolve()
    if root.exists():raise FileExistsError('preserve_existing_population')
    if shutil.disk_usage(root.parent).free<200*2**30:raise ValueError('200GiB_free_reserve_required')
    states=read(SOURCE_PLAN)['entries'];by_id={r['source_id']:r for r in states}
    baseline=read(BASELINE);old_entries=read(resolve(baseline['native_cache_index']))['entries']
    old_ids={e['source_id'] for e in old_entries}
    database=ROOT/'assets/expanded_library/raw_roles_v4.sqlite'
    with sqlite3.connect(f'file:{database}?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row;raw=[dict(r) for r in db.execute('SELECT * FROM assets')]
    aliases=defaultdict(list)
    for r in raw:
        if r['source_sha']:aliases[r['source_sha']].append({'source_id':r['id'],'partition':r['partition']})
    candidates=[];skips=[]
    for r in raw:
        state=by_id[r['id']]
        if state.get('role')!='manipulation' or state.get('status')=='canonical_alias':continue
        if not r['path'] or not Path(r['path']).is_file():
            skips.append({'source_id':r['id'],'reason':'missing_raw_file'});continue
        groups=aliases[r['source_sha']]
        if len({x['partition'] for x in groups})!=1:
            skips.append({'source_id':r['id'],'reason':'raw_alias_partition_conflict','aliases':groups});continue
        source_record=json.loads(r['record']);r['source_categories']=source_record.get('categories',[])
        candidates.append((r,state,source_record))
    # Content aliases have one job; never count a recollection as a new object.
    groups=defaultdict(list)
    for item in candidates:groups[item[0]['source_sha']].append(item)
    canonical=[]
    for sha,items in groups.items():
        items.sort(key=lambda x:(x[0]['id'] not in old_ids,x[0]['source'] not in {'ycb_official','google_scanned_objects_original_fuel'},x[0]['id']))
        canonical.append(items[0])
        skips.extend({'source_id':r['id'],'reason':'canonical_raw_alias','canonical':items[0][0]['id']} for r,_,_ in items[1:])
    def rank(item):return digest(['raw-rollout-20261031',item[0]['id']])
    canonical.sort(key=rank)
    # Twelve predeclared cross-source inputs. Selection does not inspect a new
    # mesh, size outcome, CPU success or grasp result.
    pilot=[]
    for partition in ('development','heldout_asset'):
        for group in ('prior_admitted','raw_metric','raw_engineered'):
            def matches(item):
                r,s,_=item
                if r['partition']!=partition:return False
                if group=='prior_admitted':return r['id'] in old_ids and r['source']=='objaverse'
                if s.get('native_asset'):return False
                return (r['source'] in {'ycb_official','google_scanned_objects_original_fuel'}) if group=='raw_metric' else r['source']=='objaverse'
            selected=[x for x in canonical if matches(x)][:2]
            if len(selected)!=2:raise ValueError('predeclared_pilot_stratum_has_fewer_than_two_sources:'+partition+':'+group)
            pilot.extend(selected)
    pilot_ids={x[0]['id'] for x in pilot};ordered=pilot+[x for x in canonical if x[0]['id'] not in pilot_ids]
    metadata=read(ROOT/'runs/metric_assets_20261030_v7/inventory_r1/abo_metric_evidence.json')
    root.mkdir();paths=[];rows=[]
    for row,state,source_record in ordered:
        sid=row['id'];folder=root/'cpu'/digest(sid)[:24]
        slim={k:row[k] for k in ['id','source','name','category','partition','path','source_sha','source_categories']}
        group='size_corrected' if state.get('native_asset') else 'new_raw'
        job={'raw_row':slim,'output':str(folder),'seed':20261031,'ingestion_group':group,
            'prior_native':state.get('native_asset'),'source_aliases':[x['source_id'] for x in aliases[row['source_sha']] if x['source_id']!=sid],
            'license':source_record.get('license'),'metadata_records':metadata['matches'].get(sid.split(':',1)[1],[]) if row['source']=='abo' else []}
        write(folder/'job.json',job);paths.append(str(folder/'job.json'))
        rows.append({'source_id':sid,'partition':row['partition'],'source_type':row['source'],
            'ingestion_group':group,'prior_admitted_cache':sid in old_ids,'raw_sha256':row['source_sha'],
            'job':binding(folder/'job.json'),'qualification_pilot':sid in pilot_ids})
    write(root/'population.json',{'schema':VERSION,'sources':rows,'skipped_source_rows':skips,
        'raw_source_plan':binding(SOURCE_PLAN),'no_new_outcomes_used_in_selection':True})
    prior=read(CUDA_PARENT)
    completed_parent=list((CUDA_PARENT.parent/'jobs').glob('*/reservation.json'))
    protocol={'schema':VERSION,'population':binding(root/'population.json'),'job_paths':paths,
        'qualification_job_paths':paths[:12],'maximum_CPU_sources':len(paths),'CPU_parallelism':8,
        'CPU_timeout_seconds':240,'GPU_indices':[1,2,3],'GPU_parallelism':3,'minimum_free_GPU_GiB':12,
        'GPU_timeout_seconds':120,'export_timeout_seconds':120,'maximum_optimizer_jobs':len(paths),
        'maximum_wall_seconds':43200,'maximum_attempts_per_source_and_stage':1,'retries':0,
        'minimum_free_disk_GiB':200,'manip':prior['manip'],'solver_configuration':prior['configuration'],
        'implementation_bindings':code_bindings(),'baseline_bundle':binding(BASELINE),
        'parent_optimizer_protocol':binding(CUDA_PARENT),'parent_stream_reserved_jobs':len(completed_parent),
        'prior_old_scale_GPU_jobs':prior['prior_old_scale_GPU_jobs'],'prior_older_GPU_jobs':prior['prior_older_GPU_jobs'],
        'budget_scope':'new original-source/size-policy geometry; not a retry or reopening of the terminal parent queue',
        'current_native_success_not_all_assets_physically_certified':True,'API_calls':0,'robot_physical_executions':0,
        'CPU_settle_steps_per_attempt_upper_bound':1440,'API_credentials_read':False,
        'maximum_initial_plus_conditional_convex_decompositions_per_CPU_source':2,
        'created_unix':time.time()}
    write(root/'protocol.json',protocol)
    for ref in protocol['implementation_bindings']:
        p=resolve(ref);dest=root/'code_snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    atomic(root/'cpu_status.json',{'state':'prepared','counts':{'pending':len(paths)}})
    return {'sources':len(paths),'pilot_sources':12,'ingestion_groups':dict(Counter(r['ingestion_group'] for r in rows)),
        'source_types':dict(Counter(r['source_type'] for r in rows)),'excluded_rows':len(skips)}


def validate(root):
    root=Path(root).resolve();p=read(root/'protocol.json')
    for ref in p['implementation_bindings']+p['solver_configuration']+[p['population'],p['manip']]:resolve(ref)
    if len(p['job_paths'])!=p['maximum_CPU_sources'] or p['maximum_optimizer_jobs']!=len(p['job_paths']):
        raise ValueError('frozen_population_or_optimizer_cap_changed')
    return root,p


def selected_paths(root,p,phase):
    if phase=='qualification':return [Path(x) for x in p['qualification_job_paths']]
    gate=read(root/'continuation_gate.json')
    if gate.get('continue_population') is not True or gate['protocol']!=binding(root/'protocol.json'):
        raise ValueError('independent_continuation_gate_required')
    resolve(gate['independent_review'])
    return [Path(x) for x in p['job_paths']]


def deadline(root,p):
    path=root/'START.json'
    with (root/'.start.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not path.exists():
            now=time.time();write(path,{'unix':now,'deadline_unix':now+p['maximum_wall_seconds']})
        return read(path)['deadline_unix']


def disk_ok(root,p):
    if shutil.disk_usage(root).free<p['minimum_free_disk_GiB']*2**30:
        raise RuntimeError('disk_reserve_stop_no_new_submission')


def cpu_run(root,phase):
    root,p=validate(root);paths=selected_paths(root,p,phase);end=deadline(root,p)
    frozen={str(resolve(row['job'])):row['job'] for row in read(resolve(p['population']))['sources']}
    with (root/'.cpu.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pending=[];done={}
        for path in paths:
            folder=path.parent;terminal=folder/'cpu_terminal.json'
            if terminal.exists():
                value=read(terminal)
                if value.get('status')=='unknown_system_failure':
                    raise RuntimeError('unresolved_CPU_failure_requires_audit')
                done[str(path)]=value;continue
            if (folder/'cpu_reservation.json').exists():
                process=read(folder/'compile_process.json') if (folder/'compile_process.json').exists() else {}
                if process.get('identity') and process_alive(process['identity']):raise RuntimeError('live_CPU_worker_recover_before_admission')
                if (folder/'cpu_result.json').exists():
                    value=read(folder/'cpu_result.json');atomic(terminal,value)
                    if value.get('status')=='unknown_system_failure':
                        raise RuntimeError('unresolved_CPU_failure_requires_audit')
                    done[str(path)]=value;continue
                raise RuntimeError('unknown_CPU_reservation_never_resubmit')
            pending.append(path)
        def one(path):
            folder=path.parent;disk_ok(root,p)
            if binding(path)!=frozen[str(path)]:raise ValueError('CPU_job_changed_since_population_freeze')
            write(folder/'cpu_reservation.json',{'job':binding(path),'protocol':binding(root/'protocol.json'),
                'attempt':1,'unix':time.time()})
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_THREAD_LIMIT='2')
            code=_run_process([sys.executable,'-u','-m',CPU_MODULE,'--job',str(path)],folder,'compile',
                min(p['CPU_timeout_seconds'],max(1,end-time.time())),env)
            result_path=folder/'cpu_result.json'
            if result_path.exists():result=read(result_path)
            else:result={'source_id':read(path)['raw_row']['id'],'status':'CPU_timeout' if code==124 else 'unknown_system_failure',
                'returncode':code,'recorded_settle_steps':'unknown_check_saved_partial_outputs','robot_physical_executions':0}
            result={**result,'returncode':code,'terminal_recorded_unix':time.time()}
            # The process receipt and source outcome have distinct filenames.
            atomic(folder/'cpu_terminal.json',result)
            return result
        stopped=False;iterator=iter(pending);active={};last=0
        with ThreadPoolExecutor(max_workers=p['CPU_parallelism']) as pool:
            while True:
                if (root/'STOP').exists():stopped=True
                while not stopped and time.time()<end and len(active)<p['CPU_parallelism']:
                    path=next(iterator,None)
                    if path is None:break
                    active[pool.submit(one,path)]=path
                if not active:break
                complete,_=wait(active,timeout=5,return_when=FIRST_COMPLETED)
                for future in complete:
                    path=active.pop(future)
                    try:value=future.result()
                    except Exception as exc:
                        write(root/'cpu_supervisor_error.json',{'error':repr(exc),'job':binding(path)});stopped=True;continue
                    done[str(path)]=value
                    if value['status']=='unknown_system_failure':stopped=True
                if time.monotonic()-last>=15:
                    _cpu_refresh(root,p,paths,done,active,'running' if not stopped else 'draining_unknown');last=time.monotonic()
        return _cpu_refresh(root,p,paths,done,{},'completed_scope' if len(done)==len(paths) else 'stopped_or_deadline')


def _cpu_refresh(root,p,paths,done,active,state):
    entries=[];states=[]
    for path,value in done.items():
        if value['status']=='native_ready':
            entry=read(resolve(value['entry']));entries.append(entry);states.append(read(resolve(value['state'])))
    atomic(root/'pending_grasp_index.json',{'schema':'raw_metric_native_ready_index_v1','entries':entries,'protocol':binding(root/'protocol.json')})
    atomic(root/'source_state_index.json',{'schema':VERSION,'entries':states,'scope':'ready native subset; entire population and failures remain in population/cpu jobs'})
    counts=Counter(v['status'] for v in done.values());counts['running']=len(active);counts['pending']=len(paths)-len(done)-len(active)
    value={'state':state,'phase_source_count':len(paths),'full_source_count':len(p['job_paths']),
        'counts':dict(counts),'native_ready':len(entries),'completed_sources':len(done),'unix':time.time(),
        'API_calls':0,'robot_physical_executions':0}
    atomic(root/'cpu_status.json',value);return value


def validate_gpu_job(path,root,p,selected,frozen_jobs=None):
    path=Path(path).resolve();job=read(path);sid=job['source_id']
    if sid not in selected or path != root/'gpu'/digest(sid)[:24]/'job.json':
        raise ValueError('GPU_job_outside_frozen_population')
    if job.get('protocol')!=binding(root/'protocol.json') or Path(job['output']).resolve()!=path.parent:
        raise ValueError('GPU_job_protocol_or_output_mismatch')
    cpu_job=root/'cpu'/digest(sid)[:24]/'job.json'
    if str(cpu_job) not in p['job_paths'] or read(cpu_job)['raw_row']['id']!=sid:
        raise ValueError('GPU_source_not_registered_for_CPU')
    frozen=frozen_jobs if frozen_jobs is not None else {row['source_id']:row['job'] for row in read(resolve(p['population']))['sources']}
    if binding(cpu_job)!=frozen[sid]:
        raise ValueError('GPU_parent_CPU_job_changed_since_population_freeze')
    entry=read(cpu_job.parent/'native_entry.json')
    if job['asset']!=entry['asset'] or job['entry']!=entry:
        raise ValueError('GPU_job_not_bound_to_current_native_result')
    return job


def unfinished_gpu_jobs(paths):
    return [x for x in paths if not (x.parent/'terminal.json').exists()]


def gpu_run(root,phase):
    root,p=validate(root);selected={read(x)['raw_row']['id'] for x in selected_paths(root,p,phase)};end=deadline(root,p)
    frozen={row['source_id']:row['job'] for row in read(resolve(p['population']))['sources']}
    with (root/'.gpu.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        active={};busy=set();stopped=False;deferred={};last=0;known_jobs=set();memo={}
        def execute(path,gpu):
            folder=path.parent;disk_ok(root,p)
            validate_gpu_job(path,root,p,selected,frozen)
            with ExitStack() as stack:
                fds=[]
                try:
                    for lp in resources.lock_paths(gpu):
                        handle=stack.enter_context(lp.open('a+'));fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);fds.append(handle.fileno())
                except BlockingIOError:return {'status':'resource_deferred'}
                free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                if free<p['minimum_free_GPU_GiB']*1024:return {'status':'resource_deferred'}
                snapshot=resources.gpu_snapshot(gpu)
                write(folder/'reservation.json',{'job':binding(path),'protocol':binding(root/'protocol.json'),'gpu':snapshot,'attempt':1,'unix':time.time()})
                code=_run_process([sys.executable,'-u','-m',GPU_MODULE,'worker','--job',str(path)],folder,'worker',
                    min(p['GPU_timeout_seconds'],max(1,end-time.time())),_environment(path,snapshot['uuid']),fds)
            if code:value=_asset_error(folder,code)
            else:
                validate_optimizer(path)
                code=_run_process([sys.executable,'-u','-m',GPU_MODULE,'export','--job',str(path)],folder,'export',
                    min(p['export_timeout_seconds'],max(1,end-time.time())),_environment(path))
                if code:value={'status':'needs_recovery','stage':'export','returncode':code,'retained':0}
                else:
                    r=validate_result(path);value={'status':'completed','retained':r['retained'],'result':binding(folder/'result.json'),
                        'optimizer_seconds':r['worker_seconds'],'export_seconds':r['export_seconds']}
            atomic(folder/'terminal.json',value);return value
        # Recovery can export an already-paid completed optimizer, never run it again.
        for path in (root/'gpu').glob('*/job.json'):
            validate_gpu_job(path,root,p,selected,frozen);known_jobs.add(path)
            f=path.parent
            if (f/'terminal.json').exists():
                if read(f/'terminal.json')['status']=='needs_recovery':raise RuntimeError('unresolved_GPU_terminal_requires_audit')
                continue
            if (f/'reservation.json').exists():
                for stage in ('worker','export'):
                    receipt=f/(stage+'_process.json')
                    if receipt.exists() and process_alive(read(receipt).get('identity')):
                        raise RuntimeError('live_GPU_stage_recover_before_admission')
                if (f/'result.json').exists():
                    r=validate_result(path);atomic(f/'terminal.json',{'status':'completed','retained':r['retained'],'result':binding(f/'result.json'),'recovered':True})
                elif (f/'optimizer_result.json').exists() and not (f/'export_process.json').exists():
                    validate_optimizer(path)
                    code=_run_process([sys.executable,'-u','-m',GPU_MODULE,'export','--job',str(path)],f,'export',p['export_timeout_seconds'],_environment(path))
                    if code:raise RuntimeError('saved_optimizer_export_needs_recovery')
                    r=validate_result(path);atomic(f/'terminal.json',{'status':'completed','retained':r['retained'],'result':binding(f/'result.json'),'recovered':True})
                else:raise RuntimeError('unknown_GPU_submission_never_resubmit')
        with ThreadPoolExecutor(max_workers=p['GPU_parallelism']) as pool:
            while True:
                if time.time()>=end or (root/'STOP').exists():stopped=True
                if not stopped and (root/'pending_grasp_index.json').exists():
                    for entry in read(root/'pending_grasp_index.json')['entries']:
                        if entry['source_id'] not in selected:continue
                        folder=root/'gpu'/digest(entry['source_id'])[:24];path=folder/'job.json'
                        if path.exists():
                            if path not in known_jobs:
                                validate_gpu_job(path,root,p,selected,frozen);known_jobs.add(path)
                            if read(path)['asset']!=entry['asset']:raise ValueError('current_asset_changed_in_registered_source')
                            continue
                        job={'source_id':entry['source_id'],'variant_id':entry['variant_id'],'asset':entry['asset'],'entry':entry,
                            'output':str(folder),'content_key':entry['variant_id'],'manip':p['manip'],'seed':20261031,
                            'semantic_scale_admission':entry['semantic_scale_admission'],'source_aliases':entry.get('source_aliases',[]),
                            'protocol':binding(root/'protocol.json'),'source_state':entry['source_state']}
                        write(path,job)
                        validate_gpu_job(path,root,p,selected,frozen);known_jobs.add(path)
                    if len(known_jobs)>p['maximum_optimizer_jobs']:
                        raise ValueError('frozen_optimizer_job_cap_exceeded')
                pending=[x for x in sorted(known_jobs) if x not in active.values() and not (x.parent/'terminal.json').exists()
                    and time.time()>=deferred.get(x,0)]
                if not stopped:
                    for gpu in p['GPU_indices']:
                        if gpu in busy or not pending:continue
                        path=pending.pop(0)
                        if (path.parent/'reservation.json').exists():raise RuntimeError('unknown_GPU_reservation')
                        f=pool.submit(execute,path,gpu);f.gpu=gpu;active[f]=path;busy.add(gpu)
                if active:
                    completed,_=wait(active,timeout=5,return_when=FIRST_COMPLETED)
                    for f in completed:
                        path=active.pop(f);busy.remove(f.gpu)
                        try:
                            value=f.result()
                            if value['status']=='resource_deferred':deferred[path]=time.time()+15
                            if value['status']=='needs_recovery':stopped=True
                        except Exception as exc:
                            write(root/'gpu_supervisor_error.json',{'error':repr(exc),'job':binding(path)});stopped=True
                else:time.sleep(2)
                if time.monotonic()-last>=15:gpu_refresh(root,'running',memo);last=time.monotonic()
                cpu=read(root/'cpu_status.json')
                # Resource-deferred work remains pending after a future returns;
                # the pre-launch pending list is not a completion criterion.
                unfinished=unfinished_gpu_jobs(known_jobs)
                if not active and (stopped or (cpu['state'] in {'completed_scope','stopped_or_deadline'} and not unfinished)):
                    break
        return gpu_refresh(root,'stopped_needs_recovery_or_deadline' if stopped else 'completed_CPU_scope',memo)


def gpu_refresh(root,state,memo=None):
    if memo is None:memo={}
    rows=[];entries=[]
    for path in (root/'gpu').glob('*/job.json'):
        job=read(path);terminal=path.parent/'terminal.json'
        if not terminal.exists():rows.append({'source_id':job['source_id'],'status':'active_or_pending'});continue
        value=read(terminal);rows.append({'source_id':job['source_id'],**value})
        if value['status']=='completed' and value['retained']>0:
            if path not in memo:memo[path]=validate_result(path)
            r=memo[path]
            entries.append({'source_id':job['source_id'],'variant_id':job['variant_id'],'asset':job['asset'],'cache':r['cache'],
                'retained':r['retained'],'method':'cudex_native_com_v4','relative_scale':1.,
                'semantic_scale_admission':job['semantic_scale_admission'],'source_aliases':job.get('source_aliases',[]),
                'source_state':job['source_state'],'ingestion_group':job['entry']['ingestion_group'],'optimizer_result':binding(path.parent/'result.json')})
    atomic(root/'native_cache_index.json',{'schema':'incremental_native_grasp_index_v1','entries':entries,
        'physical_success_inferred':False,'protocol':binding(root/'protocol.json')})
    value={'state':state,'counts':dict(Counter(x['status'] for x in rows)),
        'GPU_reserved_jobs':len(list((root/'gpu').glob('*/reservation.json'))),'nonempty_sources':len(entries),
        'poses':sum(e['retained'] for e in entries),'API_calls':0,'robot_physical_executions':0,'unix':time.time()}
    atomic(root/'gpu_status.json',value);return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','cpu','gpu','status'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=['qualification','population'],default='qualification')
    a=p.parse_args()
    if a.command=='prepare':result=prepare(a.root)
    elif a.command=='cpu':result=cpu_run(a.root,a.phase)
    elif a.command=='gpu':result=gpu_run(a.root,a.phase)
    else:result={stage:read(a.root/(stage+'_status.json')) if (a.root/(stage+'_status.json')).exists() else None for stage in ['cpu','gpu']}
    print(json.dumps(result,ensure_ascii=False))
