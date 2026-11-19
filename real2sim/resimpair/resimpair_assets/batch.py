"""Bounded, resumable CPU preparation, with role-specific cached qualification."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import threading
import time

import numpy as np

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, sha, write
from . import catalog, acceptance, source_mesh


def implementation():
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += [ROOT / 'resimpair_grasp/assets.py', ROOT / 'resimpair_grasp/common.py']
    return {str(p.relative_to(ROOT)): binding(p) for p in sorted(paths)}


def plan(a):
    if a.output.exists(): raise ValueError('Batch exists; resume it without rewriting its plan')
    if not 1 <= a.limit <= 1000 or not 1 <= a.workers <= 4 or not 10 <= a.timeout <= 180:
        raise ValueError('Need finite asset, CPU and timeout limits')
    if not 256 <= a.disk_mib <= 8192: raise ValueError('Need bounded output storage')
    excluded = set()
    for path in a.exclude_catalog:
        excluded.update(read(resolve(b))['source_id'] for b in read(path)['assets'])
    rows = catalog.search(a.index, role=a.role, category=a.category, source=a.source,
                          partition=a.partition, query=a.query, limit=None)
    chosen = catalog.balanced(rows, a.limit, a.seed, exclude=excluded)
    if not chosen: raise ValueError('No matching metadata candidates')
    code = implementation(); spec = []
    for row in chosen:
        actual_sha = row['source_sha'] or sha(row['path'])
        if row['source_sha'] and sha(row['path']) != actual_sha: raise ValueError('Raw source changed: ' + row['id'])
        # Exact-byte duplicates must share a split even when source names differ.
        partition = 'heldout_asset' if int(hashlib.sha256(actual_sha.encode()).hexdigest()[:8], 16) % 5 == 0 else 'development'
        if a.partition and partition != a.partition: continue
        classification = row['classification']
        external=source_mesh.resources(row['record'])
        signature = {'source_sha256': actual_sha, 'external_resources': external, 'category': row['category'], 'role': a.role,
                     'longest_m': classification['longest_m'], 'partition': partition,
                     'support': 'geometric_stable_pose', 'texture_max_px': 1024,
                     'code_sha256': {k:v['sha256'] for k,v in code.items()}}
        spec.append({'source_id': row['id'], 'source': row['source'], 'row': row['record'],
                     'classification': classification, 'signature': signature, 'cache_key': digest(signature)})
    if not spec: raise ValueError('No candidate remains after exact-content split checking')
    a.output.mkdir(parents=True); a.store.mkdir(parents=True, exist_ok=True)
    for name,b in code.items():
        target=a.output/'release'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(resolve(b),target)
    value = {'schema': 'resimpair_asset_batch_v1', 'index': binding(a.index), 'code': code,
             'acceptance': binding(acceptance.__file__), 'role': a.role, 'seed': a.seed,
             'cache_store': str(a.store.resolve()), 'maximum_assets': a.limit, 'jobs': spec,
             'workers': a.workers, 'asset_timeout_seconds': a.timeout, 'output_limit_mib': a.disk_mib,
             'max_artifact_bytes_per_job': 64 * 2**20, 'min_free_bytes': 10 * 2**30,
             'virtual_memory_limit_GiB':8,'worker_cpu_affinity_count':2,'malloc_arena_max':2,
             'api_calls_limit': 0, 'gpu_execution_limit': 0, 'automatic_retry': False,
             'excluded_catalogs': [binding(p) for p in a.exclude_catalog]}
    write(a.output / 'plan.json', value)
    return {'root': str(a.output), 'scheduled': len(spec), 'categories': dict(Counter(j['classification']['category'] for j in spec)),
            'sources': dict(Counter(j['source'] for j in spec)), 'role': a.role, 'workers': a.workers, 'api_calls': 0}


def verify(root):
    spec = read(root / 'plan.json')
    for b in spec['code'].values(): resolve(b)
    resolve(spec['acceptance'])
    return spec


def bytes_under(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())


def run(a):
    root = a.root.resolve(); spec = verify(root); store = Path(spec['cache_store'])
    if not 1 <= a.limit <= spec['maximum_assets']: raise ValueError('Bounded launch count required')
    if shutil.disk_usage(store).free < spec['min_free_bytes']: raise ValueError('Insufficient free disk')
    mutex = threading.Lock(); active = [0]; produced = [0]
    lock = store / '.batch.lock'
    with lock.open('a+') as exclusive:
        fcntl.flock(exclusive, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pending = []
        for job in spec['jobs']:
            folder = store / job['cache_key']
            if (folder / 'terminal.json').exists(): continue
            if (folder / 'reservation.json').exists(): raise ValueError('Unfinished asset reservation; recover, do not repeat: ' + job['cache_key'])
            pending.append(job)
        pending = pending[:a.limit]
        existing = sum(bytes_under(store / j['cache_key']) for j in spec['jobs'] if (store / j['cache_key']).exists())

        def launch(job):
            folder = store / job['cache_key']; start = time.monotonic()
            with mutex:
                upper = existing + produced[0] + (active[0] + 1) * spec['max_artifact_bytes_per_job']
                if upper > spec['output_limit_mib'] * 2**20: return {'source_id': job['source_id'], 'status': 'storage_reserve_exhausted'}
                if shutil.disk_usage(store).free < spec['min_free_bytes']: return {'source_id': job['source_id'], 'status': 'free_disk_stop'}
                folder.mkdir(); active[0] += 1
                write(folder / 'reservation.json', {'job': job, 'plan': binding(root / 'plan.json'), 'started_unix': time.time()})
            result = {'source_id': job['source_id'], 'status': 'failed', 'api_calls': 0, 'cache_key': job['cache_key']}
            env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       OMP_THREAD_LIMIT='2',MALLOC_ARENA_MAX='2',NUMEXPR_NUM_THREADS='1',
                       PYTHONDONTWRITEBYTECODE='1', RESIMPAIR_ASSET_RESERVATION=str(folder / 'reservation.json'))
            try:
                for phase in ['prepare', 'validate']:
                    left = spec['asset_timeout_seconds'] - (time.monotonic() - start)
                    if left <= 0: result['reason'] = 'timeout'; break
                    with (folder / (phase + '.log')).open('x') as log:
                        command = [sys.executable, '-m', 'resimpair_assets', 'worker', '--root', str(root),
                                   '--key', job['cache_key'], '--phase', phase]
                        p = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        write(folder / (phase + '_process.json'), {'pid': p.pid, 'started_unix': time.time()})
                        try: code = p.wait(timeout=left)
                        except subprocess.TimeoutExpired:
                            os.killpg(p.pid, signal.SIGKILL); p.wait(); code = 'timeout'
                    if code != 0:
                        result.update(reason=f'{phase}:{code}', log=binding(folder / (phase + '.log'))); break
                else:
                    result.update(status='qualified_for_role', entry=read(folder / 'entry.json'))
            except Exception as exc:
                result['reason'] = repr(exc)
            finally:
                result['elapsed_seconds'] = time.monotonic() - start
                result['artifact_bytes'] = bytes_under(folder)
                with mutex:
                    active[0] -= 1; produced[0] += result['artifact_bytes']
                    write(folder / 'terminal.json', result)
                print(json.dumps({k:result.get(k) for k in ['source_id','status','reason','elapsed_seconds']}, ensure_ascii=False), flush=True)
            return result

        with ThreadPoolExecutor(max_workers=spec['workers']) as pool:
            for future in as_completed([pool.submit(launch,j) for j in pending]): future.result()
    return report(root)


def report(root):
    spec = read(root / 'plan.json'); store = Path(spec['cache_store'])
    assets, failures, pending = [], [], []
    for j in spec['jobs']:
        p = store / j['cache_key'] / 'terminal.json'
        if not p.exists(): pending.append(j['source_id']); continue
        t = read(p)
        if t['status'] == 'qualified_for_role':
            resolve(t['entry']['descriptor']); resolve(t['entry']['validation'])
            assets.append(t['entry']['descriptor'])
        else: failures.append({'source_id': j['source_id'], 'reason': t.get('reason'), 'terminal': binding(p)})
    value = {'schema':'resimpair_mesh_catalog_v1','assets':assets,'failures':failures,'pending':pending,
             'plan':binding(root/'plan.json'),'attempted_assets':len(assets)+len(failures),
             'scheduled_assets':len(spec['jobs']),'api_calls':0,'physics_qualified_count':0,
             'role':spec['role'],'scope':'role eligibility only, not manipulation execution or visual semantic proof'}
    path = root / 'reports' / (digest(value)[:16]+'.json'); write(path,value)
    return {'catalog':str(path),'qualified':len(assets),'failed':len(failures),'pending':len(pending),
            'role':spec['role'],'api_calls':0}


def reduce_textures(scene):
    changed = 0
    for geom in scene.geometry.values():
        material = getattr(geom.visual, 'material', None)
        if material is None: continue
        for name in ['image','baseColorTexture','normalTexture','metallicRoughnessTexture','emissiveTexture','occlusionTexture']:
            im = getattr(material,name,None)
            if im is not None and hasattr(im,'size') and max(im.size)>1024:
                copy = im.copy(); copy.thumbnail((1024,1024)); setattr(material,name,copy); changed+=1
    return changed


def static_prepare(job, folder):
    import struct
    import trimesh
    from resimpair_grasp.assets import _stable_cook, key
    row, s = job['row'], job['signature']; source = Path(row['primary_path'])
    scene,appearance = source_mesh.load(row,s.get('external_resources'))
    if not len(scene.geometry): raise ValueError('Empty geometry')
    mesh = scene.to_geometry()
    if not np.isfinite(mesh.vertices).all() or len(mesh.faces)>160000 or len(mesh.faces)==0: raise ValueError('Visual complexity or finite-geometry limit')
    if mesh.extents.min() <= 1e-9: raise ValueError('Degenerate geometry')
    poses, probabilities = trimesh.poses.compute_stable_poses(mesh.convex_hull, n_samples=1, sigma=0., threshold=.01)
    if not len(poses): raise ValueError('No geometric support pose')
    transform = poses[int(np.argmax(probabilities))]
    scene.apply_transform(transform); mesh = scene.to_geometry()
    scale = s['longest_m'] / mesh.extents.max(); metric = np.diag([scale,scale,scale,1.])
    scene.apply_transform(metric); transform = metric @ transform
    shift = -scene.bounds.mean(0); move = np.eye(4); move[:3,3]=shift
    scene.apply_transform(move); transform = move @ transform
    changed = reduce_textures(scene); visual = folder / 'visual.glb'; scene.export(visual)
    if visual.stat().st_size>32*2**20: raise ValueError('Prepared visual exceeds storage bound')
    raw=visual.read_bytes(); length,kind=struct.unpack_from('<II',raw,12)
    gltf=json.loads(raw[20:20+length]);
    if any(x.get('uri') and not x['uri'].startswith('data:') for k in ['buffers','images'] for x in gltf.get(k,[])):
        raise ValueError('Exported GLB has external resources')
    mesh = scene.to_geometry(); proxy = trimesh.creation.box(extents=mesh.extents)
    collision,cooking = _stable_cook(proxy,folder,0)
    value={'schema':'resimpair_mesh_asset_v1','id':key(row['asset_id']), 'source_id':row['asset_id'],
       'name':row.get('name',row['asset_id']),'category':job['classification']['category'],'partition':s['partition'],
       'provenance':{'source':row['source'],'original_sha256':s['source_sha256'],'license':row.get('license'),
                     'original_location_for_provenance_only':str(source)},
       'visual':binding(visual),'collisions':[collision],'cooking':[cooking],
       'bounds_m':mesh.bounds.tolist(),'half_size':(mesh.extents/2).tolist(),'dimensions_m':mesh.extents.tolist(),
       'source_to_asset':transform.tolist(),'design_longest_m':s['longest_m'],
       'support_pose':'most probable geometric convex support; no semantic upright inference',
       'size_provenance':'category engineering size fixed before processing, not metric GT',
       'mass_kg':.08,'density_kg_m3':.08/np.prod(mesh.extents),'mass_provenance':'unused mass assumption for static asset',
       'collision_policy':'static_only_conservative_AABB','roles':{'static_distractor':True,'rigid_manipulation_candidate':False,'container_support':False,'articulated':False},
       'qualification':{'geometry':'static_proxy_pending_independent_check','native_collision_replay':'unknown','physical_execution':'unknown','training_admitted':False},
       'semantic_evidence':job['classification'],'source_appearance':appearance,'texture_resizes':changed,'max_texture_pixels':1024,
       'interaction_prohibition':'not a grasp object or container/support interface; cavity filling is deliberate conservative avoidance'}
    write(folder/'descriptor.json',value)


def worker(a):
    spec=verify(a.root); job=next(j for j in spec['jobs'] if j['cache_key']==a.key)
    folder=Path(spec['cache_store'])/a.key
    if os.environ.get('RESIMPAIR_ASSET_RESERVATION')!=str(folder/'reservation.json'): raise ValueError('Supervisor reservation required')
    resource.setrlimit(resource.RLIMIT_AS,(8*2**30,8*2**30))
    resource.setrlimit(resource.RLIMIT_FSIZE,(64*2**20,64*2**20))
    available=sorted(os.sched_getaffinity(0))[-max(2,spec['workers']*2):]
    group=int(job['cache_key'][:8],16)%max(1,len(available)//2)
    os.sched_setaffinity(0,available[2*group:2*group+2])
    source=Path(job['row']['primary_path'])
    if sha(source)!=job['signature']['source_sha256']: raise ValueError('Raw source hash changed')
    target=folder/'prepared'
    if a.phase=='prepare':
        target.mkdir()
        if spec['role']=='static': static_prepare(job,target)
        else:
            from resimpair_grasp.assets import _prepare
            s=job['signature']; selection=[job['source_id'],s['category'],s['longest_m'],'stable',s['partition']]
            scene,appearance=source_mesh.load(job['row'],s.get('external_resources'))
            resized=reduce_textures(scene);staged=target/'source_with_resources.glb';scene.export(staged)
            geometry=target/'geometry';geometry.mkdir()
            b=_prepare({**job['row'],'primary_path':str(staged)},selection,geometry)
            d=read(resolve(b));d['previous_descriptor']=b
            d['provenance']={**d['provenance'],'original_sha256':s['source_sha256'],
                'original_location_for_provenance_only':job['row']['primary_path']}
            d.update(source_appearance=appearance,texture_resizes=resized,semantic_evidence=job['classification'])
            write(target/'descriptor.json',d)
    else:
        descriptor=target/'descriptor.json'
        if spec['role']=='static':
            result=acceptance.evaluate(descriptor)
        else:
            from resimpair_grasp.assets import refine_catalog
            write(folder/'candidate_catalog.json',{'assets':[binding(descriptor)],'failures':[]})
            refined=refine_catalog(folder/'candidate_catalog.json',folder/'refined',include_household=False)
            descriptor=resolve(read(refined['catalog'])['assets'][0]);d=read(descriptor)
            result={'passed':d['roles']['rigid_manipulation_candidate'],'qualification':d['qualification'],
                    'version':'generic_mesh_acceptance_v1_geometry_preflight','physical_execution':'unknown'}
            if not result['passed']:
                write(folder/'validation.json',result);raise ValueError('Independent manipulation geometry rejected')
        write(folder/'validation.json',result)
        if spec['role']=='static':
            d=read(descriptor)
            d['qualification']={**d['qualification'],'geometry':'static_conservative_proxy',
                                'native_collision_replay':'pass','visual_semantics':'metadata_candidate'}
            d['independent_eligibility']=binding(folder/'validation.json')
            descriptor=folder/'qualified_descriptor.json';write(descriptor,d)
        write(folder/'entry.json',{'descriptor':binding(descriptor),'validation':binding(folder/'validation.json'),
                                  'cache_key':job['cache_key'],'role':spec['role']})


def merge(catalogs, output):
    if output.exists(): raise ValueError('Merged catalog exists')
    selected={}; failures=[]
    for path in catalogs:
        c=read(path); failures.extend(c.get('failures',[]))
        for b in c['assets']:
            d=read(resolve(b));key=d['source_id']
            if key not in selected or (d['roles'].get('rigid_manipulation_candidate') and not selected[key][1]['roles'].get('rigid_manipulation_candidate')):
                selected[key]=(b,d)
    value={'schema':'resimpair_mesh_catalog_v1','assets':[b for b,d in selected.values()],
           'source_catalogs':[binding(p) for p in catalogs],'failures':failures,'physics_qualified_count':0,'api_calls':0}
    write(output,value)
    return {'catalog':str(output),'assets':len(selected),'static':sum(d['roles']['static_distractor'] for b,d in selected.values()),
            'manipulation_geometry_candidates':sum(d['roles'].get('rigid_manipulation_candidate',False) for b,d in selected.values())}


def relabel(catalog_path,index,output):
    """Version metadata without reprocessing or altering existing geometry."""
    import sqlite3
    if output.exists():raise ValueError('Metadata revision exists')
    output.mkdir(parents=True);entries=[];changes=[];index_binding=binding(index)
    with sqlite3.connect(f'file:{index.resolve()}?mode=ro',uri=True) as db:
        for bound in read(catalog_path)['assets']:
            d=read(resolve(bound))
            if d['roles']['rigid_manipulation_candidate']:raise ValueError('This revision is for static-only descriptors')
            row=db.execute('SELECT classification FROM assets WHERE id=?',(d['source_id'],)).fetchone()
            classification=json.loads(row[0]) if row else None
            updated={**d,'previous_descriptor':bound,'semantic_evidence':classification,
                     'category':classification['category'] if classification else 'unclassified',
                     'semantic_revision':{'index_sha256':index_binding['sha256'],'geometry_roles_and_engineering_size_unchanged':True}}
            assert all(updated[k]==d[k] for k in ['visual','collisions','roles','dimensions_m','design_longest_m','source_to_asset'])
            p=output/(d['id']+'.json');write(p,updated);entries.append(binding(p))
            if d['category']!=updated['category'] or (classification or {}).get('semantic_ambiguities'):
                changes.append({'id':d['source_id'],'old_category':d['category'],'new_category':updated['category'],
                                'ambiguities':(classification or {}).get('semantic_ambiguities',[])})
    result={'schema':'resimpair_mesh_catalog_v1','assets':entries,'failures':read(catalog_path).get('failures',[]),
            'previous_catalog':binding(catalog_path),'classification_index':index_binding,'semantic_changes':changes,
            'scope':'metadata correction only; no new visual or physical qualification','api_calls':0}
    write(output/'catalog.json',result)
    return {'catalog':str(output/'catalog.json'),'assets':len(entries),'semantic_changes_or_ambiguities':len(changes)}
