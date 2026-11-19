"""Lazy native-collider materialization for selected tasks or physics samples.

The large offline pose library does not wait for this expensive stage.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import defaultdict
import argparse,json,os,subprocess,sys,time,hashlib,signal
import numpy as np
import trimesh
from resimpair_grasp.common import ROOT,read,write,resolve,binding,digest


def select_sample(catalog,output,count,partition,seed,mode):
    if output.exists():raise ValueError('Sample already exists')
    population=[r for r in read(catalog)['rows'] if r['status']=='offline_tabletop_feasible' and (not partition or r['partition']==partition)]
    if count>len(population):raise ValueError('Sample exceeds available population')
    population.sort(key=lambda r:hashlib.sha256(f"{seed}:{r['source_id']}".encode()).hexdigest())
    if mode=='random':chosen=population[:count]
    else:
        groups=defaultdict(list)
        for r in population:groups[r['source']].append(r)
        chosen=[]
        while len(chosen)<count:
            for key in sorted(groups):
                if groups[key]:chosen.append(groups[key].pop(0))
                if len(chosen)==count:break
    output.mkdir(parents=True)
    value={'schema':'tabletop_sample_plan_v1','library':binding(catalog),'population_count':len(population),'sample_count':count,
        'partition':partition,'seed':seed,'sampling':mode,'sample':chosen,'code':binding(__file__),
        'frozen_before_native_materialization':True,'automatic_replacement_of_failed_assets':False,'api_calls':0}
    write(output/'plan.json',value)
    return {'sample':str(output),'count':count,'population':len(population),'sampling':mode}


def prepare(asset_binding,output):
    from resimpair_grasp.assets import _stable_cook,key
    if output.exists():raise ValueError('Materialization exists')
    output.mkdir(parents=True);d=read(resolve(asset_binding));v=next(v for v in d['variants'] if v['id']==d['selected_variant'])
    mesh=trimesh.load(resolve(d['visual']),force='scene',process=True).to_geometry();hull=mesh.convex_hull
    points,_=trimesh.sample.sample_surface(mesh,1800,seed=71);distance=trimesh.proximity.closest_point(hull,points)[1]
    if np.quantile(distance,.95)<=.002:
        parts=[hull];policy='near_convex_hull_then_independent_surface_check'
    else:
        import coacd
        coacd.set_log_level('error');proxy=mesh
        if len(proxy.faces)>6000:proxy=proxy.simplify_quadric_decimation(face_count=6000)
        parts=[trimesh.Trimesh(vertices=vs,faces=fs,process=True) for vs,fs in coacd.run_coacd(coacd.Mesh(proxy.vertices,proxy.faces),
            threshold=.04,max_convex_hull=16,resolution=1500,mcts_nodes=10,mcts_iterations=30,mcts_max_depth=3,max_ch_vertex=64,seed=0)]
        policy='coacd_bounded_then_independent_surface_check'
    if not parts or len(parts)>16:raise ValueError('collision_part_budget')
    collisions=[];cooking=[]
    for i,part in enumerate(parts):
        if len(part.faces)>96:part=part.simplify_quadric_decimation(face_count=96).convex_hull
        b,evidence=_stable_cook(part,output,i);collisions.append(b);cooking.append(evidence)
    cooked=[trimesh.load(resolve(b),force='mesh',process=True) for b in collisions]
    distances=np.min([trimesh.proximity.closest_point(p,points)[1] for p in cooked],axis=0)
    union=trimesh.util.concatenate(cooked);native_points,_=trimesh.sample.sample_surface(union,1800,seed=73)
    back=trimesh.proximity.closest_point(mesh,native_points)[1];volume=sum(float(x.volume) for x in cooked)
    if volume<=0:raise ValueError('nonpositive_collider_volume')
    descriptor={'schema':'resimpair_mesh_asset_v1','id':'bulk_'+d['id'],'source_id':d['source_id'],'name':d['name'] or d['source_id'],
        'category':d['category'],'partition':d['partition'],'provenance':{'source':d['source'],'original_sha256':d['source_mesh']['sha256'],'license':d['license']},
        'visual':d['visual'],'collisions':collisions,'cooking':cooking,'source_to_asset':v['source_to_asset'],'dimensions_m':v['dimensions_m'],
        'half_size':v['half_size'],'bounds_m':v['bounds_m'],'collision_bounds_m':union.bounds.tolist(),
        'size_provenance':'predeclared synthetic gripper-scaled library profile, not recovered real size','mass_kg':.08,'density_kg_m3':.08/volume,
        'mass_provenance':'engineering assumption','collision_policy':policy,'surface_comparison':{'sample_count_each_direction':1800,
            'visual_to_collision_p95_m':float(np.quantile(distances,.95)),'visual_to_collision_max_m':float(distances.max()),
            'collision_to_visual_p95_m':float(np.quantile(back,.95)),'collision_to_visual_max_m':float(back.max())},
        'roles':{'static_distractor':True,'rigid_manipulation_candidate':True,'container_support':False,'articulated':False},
        'qualification':{'geometry':'pending_independent_check','physical_execution':'unknown','training_admitted':False},
        'offline_asset':asset_binding,'semantic_status':'source_metadata_unverified'}
    path=output/'descriptor.json';write(path,descriptor);write(output/'catalog.json',{'schema':'resimpair_mesh_catalog_v1','assets':[binding(path)],'failures':[]})
    return {'catalog':str(output/'catalog.json')}


def validate_and_cache(asset_binding,output):
    from resimpair_grasp.assets import refine_catalog
    from resimpair_grasp.grasps import GRIPPER
    result=refine_catalog(output/'prepared/catalog.json',output/'refined',include_household=False)
    b=read(result['catalog'])['assets'][0];descriptor=read(resolve(b))
    if not descriptor['roles']['rigid_manipulation_candidate']:raise ValueError('independent_collision_geometry_rejected')
    d=read(resolve(asset_binding));v=next(v for v in d['variants'] if v['id']==d['selected_variant']);candidates=[]
    for c in v['candidates']:
        if c['tcp_in_asset'][2][3]+v['half_size'][2]<GRIPPER['minimum_tcp_above_table_m']:continue
        candidates.append({**c,'id':f'g{len(candidates):03d}','proposal':'offline_mesh_antipodal','contact_geometry':'full_visual_mesh',
            'physical_status':'unknown','source_offline_candidate_status':'unmeasured'})
    if not candidates:raise ValueError('legacy_controller_height_gate_has_no_candidate')
    cache={'schema':'resimpair_mesh_grasp_cache_v1','signature':{'descriptor':b,'gripper':GRIPPER,'offline_asset':asset_binding,
        'selected_variant':v['id'],'materializer':binding(__file__)},'candidates':candidates,'physical_qualification':'unknown','api_calls':0}
    cache['key']=digest(cache['signature']);path=output/'grasp_cache.json';write(path,cache)
    entry={'asset':b,'cache':binding(path)};write(output/'entry.json',entry);return entry


def run(root,workers,timeout):
    plan=read(root/'plan.json');resolve(plan['code'])
    if not 1<=workers<=4 or not 30<=timeout<=300:raise ValueError('Finite materialization bounds required')
    def one(row):
        folder=root/'materialized'/row['id'];terminal=folder/'terminal.json'
        if terminal.exists():return read(terminal)
        if (folder/'reservation.json').exists():raise ValueError('Unknown old sample materialization; do not repeat')
        folder.mkdir(parents=True);write(folder/'reservation.json',{'asset':row['asset'],'started_unix':time.time(),'plan':binding(root/'plan.json')})
        start=time.monotonic();status={'source_id':row['source_id'],'status':'failed','api_calls':0}
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MALLOC_ARENA_MAX='2',OMP_THREAD_LIMIT='2')
        for phase in ['prepare','verify']:
            with (folder/(phase+'.log')).open('x') as log:
                p=subprocess.Popen([sys.executable,'-m','resimpair_tabletop.materialize',phase,'--asset',str(resolve(row['asset'])),'--output',str(folder.resolve())],
                    cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=p.wait(timeout=max(1,timeout-(time.monotonic()-start)))
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait();code='timeout'
            if code!=0:
                lines=(folder/(phase+'.log')).read_text(errors='replace').splitlines();status.update(phase=phase,returncode=code,reason=lines[-1] if lines else str(code));break
        else:status.update(status='native_geometry_ready',entry=read(folder/'entry.json'))
        status['elapsed_seconds']=time.monotonic()-start;write(terminal,status);return status
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(one,row) for row in plan['sample']]):
            t=future.result();print(json.dumps({k:t.get(k) for k in ['source_id','status','reason','elapsed_seconds']},ensure_ascii=False),flush=True)
    entries=[];failures=[]
    for row in plan['sample']:
        t=read(root/'materialized'/row['id']/'terminal.json')
        if t['status']=='native_geometry_ready':entries.append(t['entry'])
        else:failures.append(t)
    write(root/'cache_index.json',{'schema':'resimpair_grasp_cache_index_v1','entries':entries,'failures':failures})
    write(root/'catalog.json',{'schema':'resimpair_mesh_catalog_v1','assets':[e['asset'] for e in entries],'failures':failures})
    return {'sample':str(root),'planned':len(plan['sample']),'native_geometry_ready':len(entries),'failed':len(failures),'cache_index':str(root/'cache_index.json')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('sample');q.add_argument('--catalog',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    q.add_argument('--count',type=int,default=20);q.add_argument('--partition',choices=['development','heldout_asset']);q.add_argument('--seed',type=int,required=True)
    q.add_argument('--mode',choices=['random','source_balanced'],default='random')
    q=s.add_parser('run');q.add_argument('--root',type=Path,required=True);q.add_argument('--workers',type=int,default=2);q.add_argument('--timeout',type=int,default=180)
    for name in ['prepare','verify']:
        q=s.add_parser(name);q.add_argument('--asset',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='sample':result=select_sample(a.catalog,a.output,a.count,a.partition,a.seed,a.mode)
    elif a.command=='run':result=run(a.root,a.workers,a.timeout)
    elif a.command=='prepare':result=prepare(binding(a.asset),a.output/'prepared')
    else:result=validate_and_cache(binding(a.asset),a.output)
    print(json.dumps(result,indent=2,ensure_ascii=False))
