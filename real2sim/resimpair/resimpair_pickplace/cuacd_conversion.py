"""Weld texture seams before bounded convex conversion; same independent rules."""
import argparse
import contextlib
import fcntl
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import trimesh
from resimpair_grasp.common import ROOT,read,write,resolve,binding


def prepare(asset_binding,output):
    from resimpair_grasp.assets import _stable_cook
    from resimpair_tabletop.native_mass import calibrated_density
    if output.exists():raise ValueError('Existing conversion')
    output.mkdir(parents=True);d=read(resolve(asset_binding))
    variant=next(v for v in d['variants'] if v['id']==d['selected_variant'])
    mesh=trimesh.load(resolve(d['visual']),force='scene',process=True).to_geometry()
    raw_vertices=len(mesh.vertices)
    # Render meshes preserve UV/normal seams. Physics has no texture topology.
    # Only weld coincident vertices on a geometry copy; preserve the visual.
    mesh=trimesh.Trimesh(vertices=mesh.vertices,faces=mesh.faces,process=True,validate=True)
    mesh.merge_vertices(merge_tex=True,merge_norm=True);mesh.fix_normals(multibody=True)
    write(output/'topology.json',{'raw_vertices':raw_vertices,'welded_vertices':len(mesh.vertices),'watertight':bool(mesh.is_watertight),'visual_unchanged':True})
    hull=mesh.convex_hull;points,_=trimesh.sample.sample_surface(mesh,1800,seed=71)
    distance=trimesh.proximity.closest_point(hull,points)[1]
    if np.quantile(distance,.95)<=.0015:
        parts=[hull];policy='near_convex_then_retained_independent_surface_check'
    else:
        sys.path.insert(0,str(ROOT/'runtime/cuacd_deps'))
        import cuacd
        lo,hi=mesh.bounds;center=(lo+hi)/2;scale=float((hi-lo).max())/2
        vertices=np.asarray((mesh.vertices-center)/scale,dtype=np.float32)
        triangles=np.asarray(mesh.faces,dtype=np.int32)
        started=time.monotonic()
        with cuacd.Context(device=0,pool_bytes=1024**3) as ctx:
            topology=ctx.check_mesh(vertices,triangles)
            print('cuacd_topology',topology,flush=True)
            if not topology.get('ok',False):
                vertices,triangles=ctx.preprocess(vertices,triangles,resolution=128)
            raw=ctx.lookahead_decompose(vertices,triangles,threshold=.035,max_iters=40,
                width=20,width2=5,depth=2,max_n_cutting=8,n_concave_edges=16,
                decompose_components=True,merge_hulls=True)
            parts=[trimesh.Trimesh(vertices=hv*scale+center,faces=ht,process=True) for _,_,hv,ht in raw]
        write(output/'decomposition.json',{'backend':'cuacd==0.1.0','threshold':.035,'parts':len(parts),
            'gpu_seconds':time.monotonic()-started,'topology':topology,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')})
        policy='cuacd_0.035_32parts_then_retained_independent_surface_check'

    if not parts or len(parts)>32:raise ValueError('collision_part_budget_32')
    collisions=[];cooking=[]
    for i,part in enumerate(parts):
        if len(part.faces)>180:part=part.simplify_quadric_decimation(face_count=180).convex_hull
        b,record=_stable_cook(part,output,i);collisions.append(b);cooking.append(record)
    cooked=[trimesh.load(resolve(b),force='mesh',process=True) for b in collisions]
    distances=np.min([trimesh.proximity.closest_point(p,points)[1] for p in cooked],axis=0)
    union=trimesh.util.concatenate(cooked);native_points,_=trimesh.sample.sample_surface(union,1800,seed=73)
    back=trimesh.proximity.closest_point(mesh,native_points)[1];volume=sum(float(p.volume) for p in cooked)
    if volume<=0:raise ValueError('nonpositive_collider_volume')
    density,mass_history=calibrated_density(collisions,.08/volume,.08)
    descriptor={'schema':'resimpair_mesh_asset_v1','id':'fine_'+d['id'],'source_id':d['source_id'],
        'name':d['name'] or d['source_id'],'category':d['category'],'partition':d['partition'],
        'provenance':{'source':d['source'],'original_sha256':d['source_mesh']['sha256'],'license':d['license']},
        'visual':d['visual'],'collisions':collisions,'cooking':cooking,'source_to_asset':variant['source_to_asset'],
        'dimensions_m':variant['dimensions_m'],'half_size':variant['half_size'],'bounds_m':variant['bounds_m'],
        'collision_bounds_m':union.bounds.tolist(),'size_provenance':'same fixed synthetic size as parent offline library',
        'mass_kg':.08,'density_kg_m3':density,'mass_provenance':'engineering assumption, native density calibrated',
        'native_mass_preflight':mass_history,'collision_policy':policy,
        'surface_comparison':{'sample_count_each_direction':1800,
            'visual_to_collision_p95_m':float(np.quantile(distances,.95)),'visual_to_collision_max_m':float(distances.max()),
            'collision_to_visual_p95_m':float(np.quantile(back,.95)),'collision_to_visual_max_m':float(back.max())},
        'roles':{'static_distractor':True,'rigid_manipulation_candidate':True,'container_support':False,'articulated':False},
        'qualification':{'geometry':'pending_independent_check','physical_execution':'unknown','training_admitted':False},
        'offline_asset':asset_binding,'semantic_status':'source_metadata_unverified','generator':binding(__file__)}
    path=output/'descriptor.json';write(path,descriptor)
    write(output/'catalog.json',{'schema':'resimpair_mesh_catalog_v1','assets':[binding(path)],'failures':[]})


def run(plan,output):
    if output.exists():raise ValueError('New conversion batch required')
    jobs=read(plan)['sample']
    if len(jobs)>24:raise ValueError('At most 24 conversion samples')
    output.mkdir(parents=True)
    write(output/'plan.json',{'parent_sample':binding(plan),'sample':jobs,'code':binding(__file__),
        'workers':1,'timeout_seconds':120,'old_acceptance':binding(ROOT/'resimpair_grasp/assets.py'),
        'same_3mm_external_and_10um_native_thresholds':True,'api_limit':0})
    def one(row):
        folder=output/'materialized'/row['id'];folder.mkdir(parents=True)
        write(folder/'reservation.json',{'asset':row['asset'],'started_unix':time.time()});start=time.monotonic()
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_THREAD_LIMIT='2')
        with (folder/'worker.log').open('x') as log:
            p=subprocess.Popen([sys.executable,'-m','resimpair_pickplace.cuacd_conversion','worker','--asset',str(resolve(row['asset'])),'--output',str(folder)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=p.wait(timeout=120)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait();code='timeout'
        result={'source_id':row['source_id'],'returncode':code,'elapsed_seconds':time.monotonic()-start,
            'status':'native_geometry_ready' if code==0 else 'rejected','api_calls':0}
        if code==0:result['entry']=read(folder/'entry.json')
        else:
            lines=(folder/'worker.log').read_text(errors='replace').splitlines();result['reason']=lines[-1] if lines else str(code)
        write(folder/'terminal.json',result);return result
    with ThreadPoolExecutor(max_workers=1) as pool:
        for f in as_completed([pool.submit(one,row) for row in jobs]):print(json.dumps(f.result()),flush=True)
    rows=[read(output/'materialized'/r['id']/'terminal.json') for r in jobs]
    entries=[r['entry'] for r in rows if r['status']=='native_geometry_ready']
    write(output/'cache_index.json',{'schema':'resimpair_grasp_cache_index_v1','entries':entries,'failures':[r for r in rows if r['status']!='native_geometry_ready']})
    write(output/'report.json',{'attempted':len(rows),'native_ready':len(entries),'rows':rows,'api_calls':0,'physical_executions':0})
    return {'attempted':len(rows),'native_ready':len(entries),'index':str(output/'cache_index.json')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('run');q.add_argument('--plan',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    q=s.add_parser('worker');q.add_argument('--asset',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='run':
        from corl2026_workshop.experiments.autonomous_pipeline_v4 import resources
        with contextlib.ExitStack() as stack:
            for path in resources.lock_paths(2):
                lock=stack.enter_context(path.open('r'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            snapshot=resources.gpu_snapshot(2);resources.check_idle(snapshot)
            os.environ['CUDA_VISIBLE_DEVICES']=snapshot['uuid']
            r=run(a.plan,a.output)
    else:
        prepare(binding(a.asset),a.output/'prepared')
        from resimpair_tabletop.materialize import validate_and_cache
        r=validate_and_cache(binding(a.asset),a.output)
    print(json.dumps(r,indent=2))
