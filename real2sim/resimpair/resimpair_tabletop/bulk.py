"""Full raw-pool offline processing with finite workers, timeouts and durable state."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import Counter,defaultdict
from pathlib import Path
import fcntl,hashlib,json,os,resource,shutil,signal,sqlite3,subprocess,sys,threading,time
from resimpair_grasp.common import ROOT,read,write,resolve,binding,sha,digest


def code():
    paths=list((ROOT/'resimpair_tabletop').glob('*.py'))+[ROOT/'resimpair_assets/source_mesh.py',ROOT/'resimpair_assets/batch.py',ROOT/'resimpair_grasp/common.py']
    return {str(p.relative_to(ROOT)):binding(p) for p in paths}


def plan(a):
    if a.root.exists():raise ValueError('Existing batch: resume/report, never replace')
    if not 1<=a.limit<=60000 or not 1<=a.workers<=24 or not 10<=a.timeout<=180:raise ValueError('Finite asset/CPU/time limits required')
    if not 1<=a.disk_gib<=200 or not 60<=a.wall_seconds<=14400:raise ValueError('Finite storage/wall-clock budget required')
    profile=read(a.profile);pool=[];excluded=Counter();duplicates=0;seen=set()
    forbidden={'furniture_seating','furniture_storage','furniture_tables_beds','home_fixtures','CHAIR','SOFA','TABLE','BED','DESK','DRESSER','RUG','SHELF','CABINET','OTTOMAN','STOOL_SEATING','HEADBOARD','BENCH','BED_FRAME','LIGHT_FIXTURE'}
    with sqlite3.connect(f'file:{a.index.resolve()}?mode=ro',uri=True) as db:
        for sid,source,name,data,oldcat,source_sha in db.execute('SELECT id,source,name,record,category,source_sha FROM assets ORDER BY id'):
            row=json.loads(data);cats=row.get('categories',[]);cats=[cats] if isinstance(cats,str) else cats or []
            path=Path(row['primary_path']) if row.get('primary_path') else None
            reason=None
            if row.get('kind')!='model':reason='not_mesh_model'
            elif set(cats)&forbidden:reason='non_tabletop_source_category'
            elif path is None or not path.is_file():reason='mesh_missing'
            elif path.suffix.lower() not in {'.glb','.gltf','.obj','.ply','.stl'}:reason='mesh_format'
            elif path.stat().st_size>profile['max_input_bytes']:reason='input_size_budget'
            if reason:excluded[reason]+=1;continue
            identity=source_sha or sha(path)
            if identity in seen:duplicates+=1;continue
            seen.add(identity)
            part='heldout_asset' if int(hashlib.sha256(identity.encode()).hexdigest()[:8],16)%5==0 else 'development'
            if a.partition and part!=a.partition:continue
            category=oldcat or (cats[0] if cats else 'unclassified')
            pool.append({'source_id':sid,'source':source,'source_sha256':identity,'row':row,'category':category,'partition':part,
                'id':hashlib.sha256(sid.encode()).hexdigest()[:24]})
    # Balance raw source/category cohorts, without a manipulation category whitelist.
    groups=defaultdict(list)
    for j in pool:groups[j['category'],j['source']].append(j)
    for group in groups.values():group.sort(key=lambda j:hashlib.sha256(f"{a.seed}:{j['source_id']}".encode()).hexdigest(),reverse=True)
    jobs=[]
    while len(jobs)<a.limit and any(groups.values()):
        for group in sorted(groups):
            if groups[group]:jobs.append(groups[group].pop())
            if len(jobs)>=a.limit:break
    if not jobs:raise ValueError('No raw meshes in requested cohort')
    a.root.mkdir(parents=True);bindings=code()
    for rel,b in bindings.items():
        dest=a.root/'release'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(resolve(b),dest)
    from .geometry import panda_meshes
    gripper=panda_meshes()[1]
    plan={'schema':'tabletop_bulk_plan_v1','jobs':jobs,'source_index':binding(a.index),'profile':profile,'profile_binding':binding(a.profile),
        'code':bindings,'gripper_bindings':gripper,'pool_count':len(pool),'raw_exclusions':dict(excluded),'exact_byte_duplicates':duplicates,
        'maximum_assets':len(jobs),'workers':a.workers,'timeout_seconds':a.timeout,'wall_seconds_per_run':a.wall_seconds,
        'output_limit_bytes':a.disk_gib*2**30,'per_asset_reserve_bytes':48*2**20,'min_free_bytes':50*2**30,
        'seed':a.seed,'api_calls_limit':0,'gpu_execution_limit':0,'physical_simulation_per_asset':False,
        'physics_sample_must_be_separate':True,'automatic_retry':False,'sample_population_frozen_before_results':True}
    write(a.root/'plan.json',plan)
    write(a.root/'execution_config.json',{k:v for k,v in plan.items() if k!='jobs'})
    for n,job in enumerate(jobs):write(a.root/'jobs'/f'{n:06d}.json',job)
    return {'root':str(a.root),'raw_mesh_pool':len(pool),'scheduled':len(jobs),'categories':len(groups),'sources':dict(Counter(j['source'] for j in jobs)),
        'workers':a.workers,'physics_executions':0,'api_calls':0}


def verify(root,include_jobs=False):
    p=read(root/('plan.json' if include_jobs else 'execution_config.json'))
    for b in list(p['code'].values())+p['gripper_bindings']:resolve(b)
    resolve(p['profile_binding']);return p


def worker(a):
    # Apply affinity before numerical/native libraries start worker threads.
    available=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{available[-1-(a.slot%min(24,len(available)))]})
    resource.setrlimit(resource.RLIMIT_AS,(8*2**30,8*2**30));resource.setrlimit(resource.RLIMIT_FSIZE,(40*2**20,40*2**20))
    spec=verify(a.root);job=read(a.root/'jobs'/f'{a.number:06d}.json');folder=a.root/'assets'/job['id'];p=spec['profile']
    if os.environ.get('RESIMPAIR_TABLETOP_RESERVATION')!=str((folder/'reservation.json').resolve()):raise ValueError('Supervisor required')
    import numpy as np
    from .geometry import prepare_mesh,propose
    from .validation import check_pose,VERSION
    from resimpair_assets.source_mesh import load
    from resimpair_assets.batch import reduce_textures
    start=time.perf_counter();source=Path(job['row']['primary_path'])
    if sha(source)!=job['source_sha256']:raise ValueError('Source changed')
    scene,appearance=load(job['row']);loaded=time.perf_counter();raw=scene.to_geometry()
    mesh,S,poses,info=prepare_mesh(raw,p);prepared=time.perf_counter();variants=[];rejections=[]
    for i,(pose,prob) in enumerate(poses):
        m=mesh.copy();m.apply_transform(pose)
        # The exact visual bounds define tabletop height and actor origin.
        shift=-m.bounds.mean(0);C=np.eye(4);C[:3,3]=shift;m.apply_transform(C)
        if m.extents[2]<2*p['min_tcp_height_m']:
            rejections.append({'pose':i,'reason':'too_flat_in_support_pose'});continue
        candidates,stat=propose(m,p,spec['seed']+a.number*31+i)
        for c in candidates:check_pose(c,p,m.bounds)
        if candidates:
            variants.append({'id':f'p{i:02d}','source_to_asset':(C@pose@S).tolist(),'geometric_support_probability':float(prob),
                'dimensions_m':m.extents.tolist(),'bounds_m':m.bounds.tolist(),'half_size':(m.extents/2).tolist(),
                'candidates':candidates,'search':stat})
        else:rejections.append({'pose':i,'reason':'no_tabletop_grasp','search':stat})
    result={'schema':'tabletop_asset_v1','id':job['id'],'source_id':job['source_id'],'source':job['source'],'name':job['row'].get('name'),
        'category':job['category'],'partition':job['partition'],'source_mesh':binding(source),'license':job['row'].get('license'),
        'appearance':appearance,'size':info,'variants':variants,'pose_rejections':rejections,'profile':p,'validation_version':VERSION,
        'qualification':'offline_tabletop_feasible' if variants else 'no_feasible_pose_in_budget','physical_execution':'unmeasured',
        'semantic_identity':'source_metadata_unverified','api_calls':0,'timing_seconds':{'load':loaded-start,'normalize_and_support':prepared-loaded}}
    if variants:
        selected=max(variants,key=lambda v:(len(v['candidates']),v['geometric_support_probability']))
        scene.apply_transform(np.asarray(selected['source_to_asset']))
        # Preserve complete visuals; resize textures without simplifying the contact surface.
        for g in scene.geometry.values():
            mat=getattr(g.visual,'material',None)
            for k in ['image','baseColorTexture','normalTexture','metallicRoughnessTexture','emissiveTexture','occlusionTexture']:
                im=getattr(mat,k,None)
                if im is not None and hasattr(im,'size') and max(im.size)>p['texture_max_pixels']:
                    copy=im.copy();copy.thumbnail((p['texture_max_pixels'],p['texture_max_pixels']));setattr(mat,k,copy)
        visual=folder/'visual.glb';scene.export(visual)
        if visual.stat().st_size>p['max_visual_bytes']:raise ValueError('visual_output_budget')
        result.update(visual=binding(visual),selected_variant=selected['id'])
    result['timing_seconds']['total']=time.perf_counter()-start
    write(folder/'asset.json',result)
    return {'qualification':result['qualification'],'asset':binding(folder/'asset.json'),
        'pose_count':len(variants),'grasp_count':sum(len(v['candidates']) for v in variants)}


def run(a):
    spec=verify(a.root,True);lock=a.root/'.supervisor.lock';start=time.monotonic();mutex=threading.Lock();active=[0];used=[0];counter=[0]
    plan_binding=binding(a.root/'plan.json');local=threading.local();next_slot=[0]
    if not 1<=a.limit<=spec['maximum_assets']:raise ValueError('Finite invocation limit required')
    with lock.open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pending=[]
        for n,j in enumerate(spec['jobs']):
            folder=a.root/'assets'/j['id'];terminal=folder/'terminal.json'
            if terminal.exists():used[0]+=read(terminal).get('artifact_bytes',0);continue
            if (folder/'reservation.json').exists():raise ValueError('Unresolved prior reservation; recover without resubmission: '+j['id'])
            pending.append((n,j))
        def launch(pair):
            n,j=pair;folder=a.root/'assets'/j['id'];t=time.monotonic()
            with mutex:
                if not hasattr(local,'slot'):local.slot=next_slot[0];next_slot[0]+=1
                if time.monotonic()-start>spec['wall_seconds_per_run']:return
                if used[0]+(active[0]+1)*spec['per_asset_reserve_bytes']>spec['output_limit_bytes']:return
                if shutil.disk_usage(a.root).free<spec['min_free_bytes']:return
                folder.mkdir(parents=True);active[0]+=1
                write(folder/'reservation.json',{'number':n,'job':j,'plan':plan_binding,'started_unix':time.time()})
            env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MALLOC_ARENA_MAX='2',OMP_THREAD_LIMIT='1',TBB_NUM_THREADS='1',
                PYTHONDONTWRITEBYTECODE='1',RESIMPAIR_TABLETOP_RESERVATION=str((folder/'reservation.json').resolve()))
            terminal={'source_id':j['source_id'],'api_calls':0,'physics_executions':0}
            with (folder/'worker.log').open('x') as log:
                proc=subprocess.Popen([sys.executable,'-m','resimpair_tabletop','worker','--root',str(a.root.resolve()),'--number',str(n),'--slot',str(local.slot)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                write(folder/'process.json',{'pid':proc.pid,'started_unix':time.time()})
                try:returncode=proc.wait(timeout=spec['timeout_seconds'])
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait();returncode='timeout'
            result=folder/'worker_result.json'
            if returncode==0 and result.exists():terminal.update(read(result));terminal['status']='completed'
            else:
                lines=(folder/'worker.log').read_text(errors='replace').splitlines()
                terminal.update(status='failed',returncode=returncode,reason=lines[-1] if lines else str(returncode))
            terminal['elapsed_seconds']=time.monotonic()-t;terminal['artifact_bytes']=sum(x.stat().st_size for x in folder.rglob('*') if x.is_file())
            write(folder/'terminal.json',terminal)
            with mutex:
                used[0]+=terminal['artifact_bytes'];active[0]-=1;counter[0]+=1
                if counter[0]%25==0:print(json.dumps({'completed_this_run':counter[0],'elapsed_seconds':time.monotonic()-start,'output_GiB':used[0]/2**30}),flush=True)
        with ThreadPoolExecutor(max_workers=spec['workers']) as pool:
            for future in as_completed([pool.submit(launch,x) for x in pending[:a.limit]]):future.result()
    return report(a.root)


def report(root):
    spec=read(root/'plan.json');rows=[];counts=Counter();timings=[];sources=defaultdict(Counter)
    for j in spec['jobs']:
        folder=root/'assets'/j['id'];p=folder/'terminal.json'
        if not p.exists():counts['pending']+=1;continue
        t=read(p);status=t.get('qualification',t['status']);counts[status]+=1;sources[j['source']][status]+=1;timings.append(t['elapsed_seconds'])
        rows.append({'id':j['id'],'source_id':j['source_id'],'source':j['source'],'category':j['category'],'partition':j['partition'],
            'status':status,'asset':t.get('asset'),'pose_count':t.get('pose_count',0),'grasp_count':t.get('grasp_count',0),
            'reason':t.get('reason'),'terminal':binding(p)})
    value={'schema':'tabletop_library_catalog_v1','plan':binding(root/'plan.json'),'rows':rows,'counts':dict(counts),'sources':dict(sources),
        'scheduled':len(spec['jobs']),'attempted':len(rows),'offline_eligible':sum(r['status']=='offline_tabletop_feasible' for r in rows),
        'pose_count':sum(r['pose_count'] for r in rows),'grasp_count':sum(r['grasp_count'] for r in rows),'sum_worker_seconds':sum(timings),
        'physics_executions':0,'api_calls':0,'physical_success_inferred_per_asset':False}
    output=root/'reports'/(digest(value)[:16]+'.json');write(output,value)
    return {'catalog':str(output),**{k:value[k] for k in ['scheduled','attempted','offline_eligible','pose_count','grasp_count','counts','physics_executions','api_calls']}}
