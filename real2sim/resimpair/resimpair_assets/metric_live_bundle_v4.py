"""Immutable role-gated snapshots of live CPU and cuDex asset campaigns."""
from __future__ import annotations
import argparse
from collections import Counter
import fcntl
from pathlib import Path
import time
import trimesh
from resimpair_grasp.common import binding,digest,read,resolve,write
from resimpair_grasp.grasps import load
from resimpair_assets.metric_admission_v1 import inspect
from resimpair_tasks.cudex_bulk_queue_v2 import atomic

VERSION='live_source_metric_bundle_v4'


def snapshot(root,source_plan,cpu_root,stream_root,extra_indexes,container_catalogs):
    root,source_plan,cpu_root,stream_root=map(Path,[root,source_plan,cpu_root,stream_root])
    states=read(source_plan);source_ref=binding(source_plan);by_source={s['source_id']:s for s in states['entries']}
    cpu=read(cpu_root/'source_state_index.json');stream=read(stream_root/'native_cache_index.json')
    cpu_status=read(cpu_root/'status.json');gpu_status=read(stream_root/'status.json')
    terminal=sum(v for k,v in cpu_status['counts'].items() if k not in {'pending','running'})
    folder=root/'snapshots'/f'{terminal:05d}_{gpu_status["terminal_sources"]:05d}_{time.time_ns()}'
    folder.mkdir(parents=True)
    write(folder/'cpu_state_snapshot.json',cpu);write(folder/'gpu_index_snapshot.json',stream)
    write(folder/'producer_status.json',{'cpu':cpu_status,'gpu':{k:v for k,v in gpu_status.items() if k!='rows'}})
    candidates=[];static_candidates=[]
    for record in cpu['entries']:
        current=by_source[record['source_id']]
        for key in ['batch_status','terminal','variants']:
            if key in record:current[key]=record[key]
        for e in record.get('variants',[]):
            if e.get('retained',0)>0:candidates.append(e)
            if e.get('role')!='manipulation' or e.get('geometry_qualified'):static_candidates.append(e['asset'])
    candidates+=stream['entries']
    extra=[]
    for path in extra_indexes:
        value=read(path);candidates+=value['entries'];extra.append(binding(path))
    cache_dir=root/'admission_cache';gate_ref=binding(Path(__file__).with_name('metric_admission_v1.py'))
    unique={}
    for e in candidates:unique[(e['source_id'],e['variant_id'])]=e
    entries=[];audit=[];partitions={};aliases_rejected=0
    for e in unique.values():
        state=by_source[e['source_id']];key=digest({'asset':e['asset'],'source_plan':source_ref,'gate':gate_ref})
        path=cache_dir/(key+'.json')
        if not path.exists():write(path,inspect(e,state))
        gate=read(path);d=read(resolve(e['asset']))
        alias_parts={by_source[s]['partition'] for s in [e['source_id']]+e.get('source_aliases',[]) if s in by_source}
        split_ok=len(alias_parts)<=1
        audit.append({'source_id':e['source_id'],'variant_id':e['variant_id'],'asset':e['asset'],
            'name':d['name'],'admission':gate,'admission_evidence':binding(path),'source_alias_split_consistent':split_ok})
        if not split_ok:aliases_rejected+=1
        if not gate['passed'] or not split_ok:continue
        result=load(e['cache'],e['asset'])
        if len(result['candidates'])!=e['retained']:raise ValueError('cache_count_or_signature_changed')
        if read(resolve(e['native_support']))['status']!='pass':raise ValueError('native_support_gate_failed')
        entry={**e,'semantic_scale_admission':gate,'semantic_scale_admission_evidence':binding(path),
            'source_alias_split_consistent':True,'partition':d['partition']}
        entries.append(entry);partitions.setdefault(d['partition'],set()).add(e['source_id'])
        state.setdefault('executable_variants',[]).append({'variant_id':e['variant_id'],'asset':e['asset'],
            'cache':e['cache'],'retained':e['retained'],'partition':d['partition']})
    static=[];background=[];static_audit=[]
    seen=set()
    for ref in static_candidates:
        if ref['sha256'] in seen:continue
        seen.add(ref['sha256']);d=read(resolve(ref));state=by_source[d['source_id']]
        key=digest({'static_asset':ref,'source_plan':source_ref,'implementation':binding(__file__)})
        path=cache_dir/(key+'.json')
        if not path.exists():
            mesh=trimesh.load(resolve(d['visual']),force='scene',process=False).to_geometry()
            base=state.get('base_longest_m');ratio=float(max(mesh.bounding_box_oriented.primitive.extents))/base if base else None
            value={'passed':state.get('semantic_admission')!='pending' and state.get('role')==d.get('semantic_role')
                   and ratio is not None and .9-1e-4<=ratio<=1.1+1e-4,'ratio_to_current_base':ratio,
                   'current_role':state.get('role'),'physical_success_inferred':False}
            write(path,value)
        value=read(path)
        if not value['passed']:
            static_audit.append({'source_id':d['source_id'],'admission':value});continue
        static.append(ref)
        if state['role']=='background':background.append(ref)
    containers=[];container_rows=[];container_refs=[]
    for path in container_catalogs:
        cat=read(path);containers+=cat['assets'];container_rows+=cat['rows'];container_refs.append(binding(path))
    container_sources={read(resolve(ref))['source_id'] for ref in containers}
    output={'native_cache_index.json':{'schema':'resimpair_grasp_cache_index_v1','entries':entries},
        'containers.json':{'schema':'stationary_container_catalog_v1','assets':containers,'rows':container_rows,
            'parents':container_refs,'source_id_and_partition_preserved':True,'physics_executions':0,'api_calls':0},
        'static_catalog.json':{'schema':'resimpair_mesh_catalog_v1','assets':static},
        'background_catalog.json':{'schema':'resimpair_mesh_catalog_v1','assets':background},
        'source_state_index.json':states,'admission_audit.json':{'cache_entries':audit,'static_rejections':static_audit}}
    for name,value in output.items():write(folder/name,value)
    partition_counts={k:len(v) for k,v in partitions.items()}
    summary={'raw_sources':len(states['entries']),'native_role_counts':states['summary']['roles'],
        'semantic_pending_native_sources':states['summary']['semantic_admission']['pending'],
        'CPU_terminal_sources':terminal,'GPU_terminal_sources':gpu_status['terminal_sources'],
        'GPU_cumulative_new_scale_jobs':gpu_status['cumulative_new_scale_GPU_jobs'],'GPU_limit':3080,
        'admitted_grasp_variants':len(entries),'admitted_grasp_sources':len({e['source_id'] for e in entries}),
        'partition_source_counts':partition_counts,'poses':sum(e['retained'] for e in entries),
        'source_alias_split_rejections':aliases_rejected,
        'container_variants':len(containers),'container_sources':len(container_sources),
        'static_descriptors':len(static),'background_descriptors':len(background),
        'minimum_20_sources_and_8_each_split':len({e['source_id'] for e in entries})>=20 and
            partition_counts.get('development',0)>=8 and partition_counts.get('heldout_asset',0)>=8,
        'physical_success_inferred':False,'API_calls':0}
    bundle={'schema':VERSION,'summary':summary,
        **{key:binding(folder/name) for key,name in [('native_cache_index','native_cache_index.json'),('containers','containers.json'),
            ('static_catalog','static_catalog.json'),('background_catalog','background_catalog.json'),('source_state_index','source_state_index.json'),('admission_audit','admission_audit.json')]},
        'source_semantics_checked_against_current_policy':{'schema':'source_semantic_scale_gate_v1',
            'source_state_index':source_ref,'gate':gate_ref,'all_cache_entries_passed':True},
        'input_snapshots':[binding(folder/'cpu_state_snapshot.json'),binding(folder/'gpu_index_snapshot.json')],
        'extra_cache_indexes':extra,'implementation':binding(__file__)}
    write(folder/'BUNDLE.json',bundle)
    latest={'bundle':binding(folder/'BUNDLE.json'),'summary':summary,'created_unix':time.time()}
    atomic(root/'LATEST.json',latest);return latest


def watch(config_path):
    config_path=Path(config_path);c=read(config_path);root=Path(c['root']);root.mkdir(parents=True,exist_ok=True)
    with (root/'.snapshot.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=read(root/'LATEST.json') if (root/'LATEST.json').exists() else None
        while time.time()<c['deadline_unix']:
            cpu_path=Path(c['cpu_root'])/'status.json';gpu_path=Path(c['stream_root'])/'status.json'
            if cpu_path.exists() and gpu_path.exists():
                cpu,gpu=read(cpu_path),read(gpu_path)
                terminal=sum(v for k,v in cpu['counts'].items() if k not in {'pending','running'})+gpu['terminal_sources']
                prior_terminal=(previous['summary']['CPU_terminal_sources']+previous['summary']['GPU_terminal_sources']) if previous else 0
                if previous is None or terminal-prior_terminal>=200 or time.time()-previous['created_unix']>=300:
                    previous=snapshot(root,c['source_plan'],c['cpu_root'],c['stream_root'],c['extra_indexes'],c['container_catalogs'])
                    print(previous['summary'],flush=True)
                if not cpu['counts'].get('pending',0) and not cpu['counts'].get('running',0) and gpu.get('state')=='completed_current_CPU_scope':
                    if terminal!=prior_terminal:snapshot(root,c['source_plan'],c['cpu_root'],c['stream_root'],c['extra_indexes'],c['container_catalogs'])
                    return
            time.sleep(30)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    a=p.parse_args();watch(a.config)
