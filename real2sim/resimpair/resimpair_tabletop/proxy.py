"""Conservative exterior proxy for rigid grasp/place, never cavity interaction.

Unlike the retained whole-surface 3 mm profile, this profile certifies enclosure,
contact-patch fidelity and the open-hand approach. Qualification is explicitly
different; the physical trajectory/event evaluator remains unchanged.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse,hashlib,itertools,json,os,subprocess,sys,time,signal
import numpy as np
import trimesh
from scipy.spatial import ConvexHull,HalfspaceIntersection
from resimpair_grasp.common import ROOT,read,write,binding,resolve,digest

PROFILE={'version':'external_grasp_proxy_v1','envelope_margin_m':.00015,'max_enclosure_violation_m':.00001,
         'center_contact_error_m':.00075,'patch_p95_error_m':.0015,'patch_max_error_m':.003,
         'minimum_patch_rays':5,'patch_offsets_m':[-.006,0,.006],
         'roles':['external_rigid_grasp','free_space_transfer','external_surface_placement'],
         'forbidden_roles':['insertion','container_interior','pouring','articulation','precision_mating']}


def build(asset_binding,folder):
    from .geometry import Clearance
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    from resimpair_grasp.assets import _stable_cook
    from resimpair_grasp.grasps import GRIPPER
    if folder.exists():raise ValueError('Proxy output exists')
    folder.mkdir(parents=True);a=read(resolve(asset_binding));v=next(x for x in a['variants'] if x['id']==a['selected_variant'])
    mesh=trimesh.load(resolve(a['visual']),force='scene',process=True).to_geometry()
    directions=[np.asarray(d,float) for d in itertools.product([-1,0,1],repeat=3) if any(d)]
    directions += [np.asarray(n,float) for c in v['candidates'] for n in c['contact_normals']]
    directions=np.asarray([d/np.linalg.norm(d) for d in directions]);directions=np.unique(np.round(directions,6),axis=0)
    support=(mesh.vertices@directions.T).max(0)+PROFILE['envelope_margin_m'];planes=np.c_[directions,-support]
    interior=mesh.convex_hull.center_mass
    vertices=HalfspaceIntersection(planes,interior).intersections;envelope=trimesh.convex.convex_hull(vertices)
    collider,cooking=_stable_cook(envelope,folder,0)
    native=trimesh.load(resolve(collider),force='mesh',process=True);equations=ConvexHull(native.vertices).equations
    violation=float((mesh.vertices@equations[:,:3].T+equations[:,3]).max())
    if violation>PROFILE['max_enclosure_violation_m']:raise ValueError('proxy_does_not_enclose_visual')
    raw_ray=RayMeshIntersector(mesh);proxy_ray=RayMeshIntersector(native);guard=Clearance(native)
    kept=[];rejections=[];patch_rows=[]
    for c in v['candidates']:
        T=np.asarray(c['tcp_in_asset']);checks=[]
        if T[2,3]+v['half_size'][2]<GRIPPER['minimum_tcp_above_table_m']:
            rejections.append('legacy_minimum_height');continue
        for point,normal in zip(c['contacts_in_asset'],c['contact_normals']):
            point=np.asarray(point);normal=np.asarray(normal);tangent=T[:3,0]
            tangent=tangent-normal*(tangent@normal);tangent/=np.linalg.norm(tangent);other=np.cross(normal,tangent)
            origins=np.asarray([point+normal*.02+x*tangent+y*other for x,y in itertools.product(PROFILE['patch_offsets_m'],repeat=2)])
            rays=np.tile(-normal,(len(origins),1))
            raw,ri,_=raw_ray.intersects_location(origins,rays,multiple_hits=False)
            proxy,pi,_=proxy_ray.intersects_location(origins,rays,multiple_hits=False)
            rp={int(i):p for i,p in zip(ri,raw)};pp={int(i):p for i,p in zip(pi,proxy)};joint=sorted(rp.keys()&pp.keys())
            errors=[float(np.linalg.norm(rp[i]-pp[i])) for i in joint]
            center=float(np.linalg.norm(rp[4]-pp[4])) if 4 in joint else None
            okay=(center is not None and center<=PROFILE['center_contact_error_m'] and len(joint)>=PROFILE['minimum_patch_rays']
                and np.quantile(errors,.95)<=PROFILE['patch_p95_error_m'] and max(errors)<=PROFILE['patch_max_error_m'])
            checks.append({'passed':bool(okay),'rays':len(joint),'center_error_m':center,'p95_error_m':float(np.quantile(errors,.95)) if errors else None})
        clear=guard.check(T,native.bounds[0,2],a['profile']['table_clearance_m'])
        if all(x['passed'] for x in checks) and clear[0]:
            kept.append({**c,'id':f'g{len(kept):03d}','physical_status':'unknown','proxy_contact_checks':checks})
            patch_rows.append(checks)
        else:rejections.append('contact_patch_mismatch' if not all(x['passed'] for x in checks) else clear[2])
    if not kept:
        write(folder/'rejection.json',{'reason':'no_proxy_compatible_grasp','checks':rejections,'profile':PROFILE})
        raise ValueError('no_proxy_compatible_grasp')
    density=float(np.float32(.08/native.volume));mass=float(density*native.volume)
    descriptor={'schema':'resimpair_mesh_asset_v1','id':'proxy_'+a['id'],'source_id':a['source_id'],'name':a['name'] or a['source_id'],
        'category':a['category'],'partition':a['partition'],'provenance':{'source':a['source'],'license':a['license'],'original_sha256':a['source_mesh']['sha256']},
        'visual':a['visual'],'collisions':[collider],'cooking':[cooking],'source_to_asset':v['source_to_asset'],'dimensions_m':v['dimensions_m'],
        'half_size':v['half_size'],'bounds_m':v['bounds_m'],'collision_bounds_m':native.bounds.tolist(),'mass_kg':mass,'density_kg_m3':density,
        'mass_provenance':'synthetic engineering assumption','size_provenance':a['size'],'collision_policy':'external_grasp_proxy_v1',
        'roles':{'static_distractor':True,'rigid_manipulation_candidate':True,'container_support':False,'articulated':False},
        'qualification':{'geometry':'external_grasp_proxy_only','global_3mm_surface_qualification':'not_claimed','physical_execution':'unknown','training_admitted':False},
        'proxy_profile':PROFILE,'enclosure_max_violation_m':violation,'contact_patch_checks':patch_rows,'offline_asset':asset_binding,
        'semantic_status':'source_metadata_unverified','role_scope':'external rigid grasp/place only; no cavity interaction'}
    path=folder/'descriptor.json';write(path,descriptor);db=binding(path)
    cache={'schema':'resimpair_mesh_grasp_cache_v1','signature':{'descriptor':db,'gripper':GRIPPER,'offline_asset':asset_binding,
        'proxy_profile':PROFILE,'implementation':binding(__file__)},'candidates':kept,'physical_qualification':'unknown','api_calls':0}
    cache['key']=digest(cache['signature']);cp=folder/'grasp_cache.json';write(cp,cache)
    # A fresh native collider load must match the cached shape without changing
    # the retained 10 um native replay threshold.
    import sapien
    shape=sapien.physx.PhysxCollisionShapeConvexMesh(filename=str(resolve(collider)),scale=[1,1,1],material=sapien.physx.PhysxMaterial(.5,.5,0))
    distance=np.linalg.norm(native.vertices[:,None]-np.asarray(shape.vertices)[None],axis=-1)
    error=float(max(distance.min(0).max(),distance.min(1).max()))
    if error>=1e-5:raise ValueError('native_proxy_replay_mismatch')
    write(folder/'proxy_qualification.json',{'profile':PROFILE,'source_binding':asset_binding,'retained_grasps':len(kept),
        'native_replay_error_m':error,'enclosure_max_violation_m':violation,'physical_success':'unmeasured','rejections':rejections})
    entry={'asset':db,'cache':binding(cp)};write(folder/'entry.json',entry);return entry


def initialize(input_path,root,sample):
    if root.exists():raise ValueError('Proxy campaign exists')
    data=read(input_path)
    rows=data['sample'] if sample else [r for r in data['rows'] if r['status']=='offline_tabletop_feasible']
    root.mkdir(parents=True)
    plan={**data,'sample':rows,'proxy_profile':PROFILE,'source_plan':binding(input_path),'code':binding(__file__),
          'schema':'tabletop_proxy_materialization_plan_v1','physics_executions':0,'api_calls':0}
    write(root/'plan.json',plan)
    # Preserve sampled outcome protocol shape for the same-case proxy ablation.
    return {'planned':len(rows),'root':str(root),'sample_ablation':sample}


def run(root,workers,timeout):
    plan=read(root/'plan.json');resolve(plan['code'])
    if not 1<=workers<=16 or not 10<=timeout<=120:raise ValueError('Finite CPU/timeout limits required')
    def one(row):
        folder=root/'materialized'/row['id'];terminal=folder/'terminal.json'
        if terminal.exists():return read(terminal)
        if folder.exists():raise ValueError('Unresolved proxy reservation')
        folder.mkdir(parents=True);write(folder/'reservation.json',{'row':row,'started_unix':time.time()});start=time.monotonic()
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_THREAD_LIMIT='1',MALLOC_ARENA_MAX='2')
        with (folder/'worker.log').open('x') as log:
            p=subprocess.Popen([sys.executable,'-m','resimpair_tabletop.proxy','one','--asset',str(resolve(row['asset'])),'--root',str(folder.resolve())],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait();code='timeout'
        if code==0:
            result={'source_id':row['source_id'],'status':'native_geometry_ready','entry':read(folder/'proxy/entry.json')}
        else:
            lines=(folder/'worker.log').read_text(errors='replace').splitlines()
            result={'source_id':row['source_id'],'status':'failed','returncode':code,'reason':lines[-1] if lines else str(code)}
        result.update(elapsed_seconds=time.monotonic()-start,physics_executions=0,api_calls=0);write(terminal,result);return result
    complete=0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(one,row) for row in plan['sample']]):
            result=future.result();complete+=1
            if len(plan['sample'])<=32 or complete%50==0:print(json.dumps({'completed':complete,'last':result['status']},ensure_ascii=False),flush=True)
    entries=[];failures=[]
    for row in plan['sample']:
        t=read(root/'materialized'/row['id']/'terminal.json')
        if t['status']=='native_geometry_ready':entries.append(t['entry'])
        else:failures.append(t)
    write(root/'cache_index.json',{'schema':'resimpair_grasp_cache_index_v1','entries':entries,'failures':failures})
    write(root/'catalog.json',{'schema':'resimpair_mesh_catalog_v1','assets':[e['asset'] for e in entries],'failures':failures,'qualification_profile':PROFILE})
    return {'planned':len(plan['sample']),'proxy_ready':len(entries),'failed':len(failures),'physics_executions':0,'api_calls':0}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('init');q.add_argument('--input',type=Path,required=True);q.add_argument('--root',type=Path,required=True);q.add_argument('--sample',action='store_true')
    q=s.add_parser('run');q.add_argument('--root',type=Path,required=True);q.add_argument('--workers',type=int,default=8);q.add_argument('--timeout',type=int,default=60)
    q=s.add_parser('one');q.add_argument('--root',type=Path,required=True);q.add_argument('--asset',type=Path,required=True)
    a=p.parse_args()
    if a.command=='init':result=initialize(a.input,a.root,a.sample)
    elif a.command=='run':result=run(a.root,a.workers,a.timeout)
    else:result=build(binding(a.asset),a.root/'proxy')
    print(json.dumps(result,indent=2,ensure_ascii=False))
