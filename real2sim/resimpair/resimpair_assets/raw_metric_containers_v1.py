"""Bounded raw GSO/YCB container intake at the source metric convention.

Never consumes normalized or native-ready asset descriptors. Uses the retained
cavity, stable-pose and oriented-support rules. Static PhysX mesh cooking is
CPU geometry preparation: no scene, dynamics, renderer, GPU, API or grasp run.
Outputs are candidate geometry in a separate catalog, never default admission.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import traceback

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write

VERSION = 'raw_metric_container_intake_v1'
ALLOWED_SOURCES = {'google_scanned_objects_original_fuel', 'ycb_official'}
MAX_SOURCES, WORKERS, ASSET_SECONDS, WALL_SECONDS = 37, 2, 120, 1200
RULE_FILES = ['resimpair_pickplace/containers.py', 'resimpair_pickplace/support_surface.py',
              'resimpair_assets/metric_visual_precision_v3.py', 'resimpair_tabletop/geometry.py',
              'resimpair_grasp/common.py', 'resimpair_runtime.py']


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z': return None
        return {'pid': int(pid), 'start_ticks': int(fields[19]),
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    except (OSError, ProcessLookupError): return None


@contextmanager
def campaign_lock(root):
    with (Path(root) / '.run.lock').open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try: yield
        finally: fcntl.flock(stream, fcntl.LOCK_UN)


def sufficient_disk(root):
    free = shutil.disk_usage(root).free
    if free < 50 * 2**30: raise RuntimeError('minimum_50GiB_free_required')
    return free


def raw_dependencies(row):
    """Bind OBJ/MTL/textures; GSO Fuel textures live outside meshes/.

    A resolver fixes that package-relative lookup without editing raw MTL bytes.
    No directory-wide search or network material fetch is performed.
    """
    source = Path(row['path']).resolve()
    if not source.is_file(): raise ValueError('raw_source_file_missing')
    if source.suffix.lower() != '.obj': raise ValueError('this_raw_intake_requires_OBJ')
    if source.stat().st_size > 64 * 2**20: raise ValueError('bounded_raw_OBJ_byte_limit')
    source_ref = binding(source)
    if source_ref['sha256'] != row['source_sha']: raise ValueError('raw_source_hash_changed')
    base = source.parent.parent if row['source'] == 'google_scanned_objects_original_fuel' else source.parent
    resolver = {}
    for line in source.read_text(errors='strict').splitlines():
        if not line.startswith('mtllib '): continue
        name = line.split(maxsplit=1)[1].strip()
        mtl = (source.parent / name).resolve()
        if not mtl.is_relative_to(base) or not mtl.is_file(): raise ValueError('source_MTL_missing_or_outside_package')
        if mtl.stat().st_size > 2**20: raise ValueError('bounded_source_MTL_byte_limit')
        resolver[name] = binding(mtl)
        for material_line in mtl.read_text().splitlines():
            fields = material_line.strip().split(maxsplit=1)
            if len(fields) != 2 or not (fields[0].lower().startswith('map_') or fields[0].lower() in {'bump', 'disp'}): continue
            texture_name = fields[1].strip()
            choices = [(mtl.parent / texture_name).resolve(), (base / 'materials/textures' / texture_name).resolve()]
            texture = next((x for x in choices if x.is_relative_to(base) and x.is_file()), None)
            if texture is None: raise ValueError('source_material_texture_missing')
            if texture.stat().st_size > 128 * 2**20: raise ValueError('bounded_source_texture_byte_limit')
            if texture_name in resolver and resolver[texture_name] != binding(texture):
                raise ValueError('ambiguous_source_material_resource')
            resolver[texture_name] = binding(texture)
    if len(resolver) > 32: raise ValueError('bounded_source_material_resource_count')
    return {'source': source_ref, 'resolver': resolver, 'raw_files_modified': False}


def prepare(root, source_selection, database=None):
    root = Path(root).resolve(); selection = read(source_selection)
    source_ids = selection['source_ids']
    if not 1 <= len(source_ids) <= MAX_SOURCES or len(set(source_ids)) != len(source_ids):
        raise ValueError('bounded_unique_source_selection_required')
    if root.exists(): raise FileExistsError('preserve_existing_intake_root')
    free = sufficient_disk(root.parent)
    database = Path(database or ROOT / 'assets/expanded_library/raw_roles_v4.sqlite').resolve()
    with sqlite3.connect(f'file:{database}?mode=ro', uri=True) as connection:
        connection.row_factory = sqlite3.Row
        selected = []
        for source_id in source_ids:
            row = connection.execute('SELECT * FROM assets WHERE id=?', (source_id,)).fetchone()
            if row is None: raise ValueError('selected_source_not_in_raw_database')
            row = dict(row)
            if row['source'] not in ALLOWED_SOURCES: raise ValueError('only_declared_metric_source_conventions_allowed')
            aliases = [dict(x) for x in connection.execute('SELECT id,partition FROM assets WHERE source_sha=?', (row['source_sha'],))]
            row['alias_group'] = aliases
            selected.append(row)
    root.mkdir()
    jobs = []; seen = {}
    for row in selected:
        key = digest(row['id'])[:24]; folder = root / 'jobs' / key
        preflight_error = None; dependencies = None
        try: dependencies = raw_dependencies(row)
        except (OSError, ValueError) as error: preflight_error = type(error).__name__ + ':' + str(error)
        duplicate = seen.get(row['source_sha']); seen.setdefault(row['source_sha'], row['id'])
        job = {'schema': VERSION, 'source_id': row['id'], 'source_type': row['source'],
            'name': row['name'], 'partition': row['partition'], 'source_record': json.loads(row['record']),
            'raw_metadata_record_sha256': digest({k: v for k,v in row.items() if k != 'alias_group'}),
            'dependencies': dependencies, 'preflight_error': preflight_error,
            'source_aliases': row['alias_group'], 'alias_split_consistent': len({x['partition'] for x in row['alias_group']}) == 1,
            'duplicate_selected_source': duplicate, 'source_scale_factor': 1.0,
            'scale_evidence': 'original_source_metric_convention_not_per_object_measurement',
            'output': str(folder)}
        write(folder / 'job.json', job); jobs.append(binding(folder / 'job.json'))
    protocol = {'schema': VERSION, 'source_selection': binding(source_selection), 'database': binding(database),
        'source_count': len(jobs), 'source_type_counts': dict(Counter(r['source'] for r in selected)), 'jobs': jobs,
        'maximum_sources': MAX_SOURCES, 'maximum_CPU_workers': WORKERS, 'per_source_timeout_seconds': ASSET_SECONDS,
        'maximum_wall_seconds_from_first_run': WALL_SECONDS, 'per_source_attempts': 1, 'retries': 0,
        'rule_bindings': [binding(ROOT / p) for p in RULE_FILES], 'implementation': binding(__file__),
        'rules': {'stable_pose_candidates': 10, 'stable_pose_probability_threshold': .005,
            'stable_hull_face_limit': 400, 'source_mesh_face_limit': 300000,
            'cavity_floor_tolerance_m': .003, 'native_triangle_vertex_error_limit_m': 1e-5,
            'export_bounds_rtol':1e-6, 'export_bounds_atol_m':1e-7,
            'cavity_and_support_functions_reused_without_change': True},
        'resources': {'free_disk_bytes_at_prepare': free, 'minimum_free_bytes': 50 * 2**30,
                      'worker_threads': 1, 'CUDA_VISIBLE_DEVICES': ''},
        'source_scale_factor': 1.0, 'default_catalog_modified': False, 'dynamic_scene_created': False,
        'unknown_reservations_never_resubmitted': True, 'API_calls': 0, 'GPU_jobs': 0, 'physics_steps': 0}
    write(root / 'PROTOCOL.json', protocol)
    return status(root)


def validate(root):
    protocol = read(Path(root) / 'PROTOCOL.json')
    if (protocol.get('maximum_sources'), protocol.get('maximum_CPU_workers'), protocol.get('per_source_timeout_seconds'),
            protocol.get('maximum_wall_seconds_from_first_run'), protocol.get('retries')) != (37,2,120,1200,0):
        raise ValueError('frozen_finite_budget_changed')
    for ref in protocol['rule_bindings'] + [protocol['implementation'], protocol['source_selection']]: resolve(ref)
    if len(protocol['jobs']) != protocol['source_count'] or len(protocol['jobs']) > MAX_SOURCES:
        raise ValueError('registered_source_count_changed')
    return protocol


def texture_summary(scene):
    import numpy as np
    rows = []
    for name, mesh in scene.geometry.items():
        visual = mesh.visual; material = getattr(visual, 'material', None)
        image = getattr(material, 'baseColorTexture', None)
        if image is None: image = getattr(material, 'image', None)
        texture = None
        if image is not None:
            rgba = np.asarray(image.convert('RGBA'))
            texture = {'shape': list(rgba.shape), 'pixels_sha256':hashlib.sha256(rgba.tobytes()).hexdigest()}
        rows.append({'geometry': name, 'vertices':len(mesh.vertices), 'faces':len(mesh.faces),
            'visual_kind':visual.kind, 'material_class': type(material).__name__,
            'has_UV':getattr(visual, 'uv', None) is not None, 'texture':texture})
    return rows


def compile_one(job_path):
    import numpy as np
    import trimesh
    import sapien
    from scipy.spatial import cKDTree
    from resimpair_assets.metric_visual_precision_v3 import bake_visual_world
    from resimpair_pickplace.containers import cavity
    from resimpair_pickplace.support_surface import check_support

    job = read(job_path); output = Path(job['output']); start = time.monotonic()
    deps = job['dependencies']
    source = resolve(deps['source'])
    resolver = {name:resolve(ref).read_bytes() for name,ref in deps['resolver'].items()}
    scene = trimesh.load(source, force='scene', process=False, resolver=resolver)
    original_materials = texture_summary(scene)
    scene = bake_visual_world(scene); original = scene.to_geometry()
    if not isinstance(original, trimesh.Trimesh) or not len(original.faces): raise ValueError('no_triangle_geometry')
    if len(original.faces) > 300000: raise ValueError('container_face_budget')
    if not np.isfinite(original.vertices).all(): raise ValueError('nonfinite_container')
    source_geometry = {'source': deps['source'], 'raw_bounds_m':original.bounds.tolist(),
        'raw_dimensions_m':original.extents.tolist(), 'source_scale_factor':1.0,
        'mesh_vertices':len(original.vertices), 'mesh_faces':len(original.faces), 'materials':original_materials,
        'scale_evidence':job['scale_evidence'], 'source_geometry_node_transforms_baked_without_rescaling':True}
    write(output / 'SOURCE_GEOMETRY.json', source_geometry)
    hull = original.convex_hull
    if len(hull.faces) > 400: hull = hull.simplify_quadric_decimation(face_count=400).convex_hull
    poses, probabilities = trimesh.poses.compute_stable_poses(hull, n_samples=1, sigma=0, threshold=.005)
    accepted = []; search = []
    for index, (pose, probability) in enumerate(zip(poses[:10], probabilities[:10])):
        placed = original.copy(); placed.apply_transform(pose)
        centre = placed.bounds.mean(axis=0); centre[2] = placed.bounds[0,2]
        placed.apply_translation(-centre)
        translation = np.eye(4); translation[:3,3] = -centre; transform = translation @ pose
        try: opening = cavity(placed)
        except ValueError as error:
            search.append({'index':index, 'status':'no_open_supported_cavity', 'reason':str(error)})
            continue
        search.append({'index':index, 'status':'geometry_candidate', 'cavity':opening})
        accepted.append((opening['flat_floor_radius_m'], placed, transform, opening, float(probability), index))
    write(output / 'orientation_search.json', {'maximum_stable_candidates':10, 'rows':search, 'scale_factor':1.0})
    if not accepted: raise ValueError('no_upright_container_support_pose_at_original_metric_scale')
    _, mesh, transform, opening, probability, pose_index = max(accepted, key=lambda x:x[0])
    singular = np.linalg.svd(transform[:3,:3], compute_uv=False)
    if not np.allclose(singular, 1, rtol=0, atol=1e-10) or np.linalg.det(transform[:3,:3]) <= 0:
        raise ValueError('source_to_asset_must_be_rigid_without_rescaling')
    visual = scene.copy(); visual.apply_transform(transform); visual = bake_visual_world(visual)
    visual_path = output / 'visual.glb'; visual.export(visual_path)
    reloaded_scene = trimesh.load(visual_path, force='scene', process=False)
    reloaded = reloaded_scene.to_geometry()
    if not np.allclose(reloaded.bounds, mesh.bounds, rtol=1e-6, atol=1e-7):
        raise ValueError('container_visual_export_changed_scale_or_origin')
    reloaded_materials = texture_summary(reloaded_scene)
    textures = lambda rows: Counter(x['texture']['pixels_sha256'] for x in rows if x['texture'])
    if textures(original_materials) != textures(reloaded_materials): raise ValueError('source_texture_pixels_not_preserved')
    if sum(x['has_UV'] for x in original_materials) != sum(x['has_UV'] for x in reloaded_materials):
        raise ValueError('source_texture_UV_presence_changed')
    write(output / 'MATERIAL_EXPORT.json', {'source':original_materials,'reloaded':reloaded_materials,
        'texture_pixel_multiset_equal':True,'UV_presence_preserved':True,
        'shading_equivalence':'unknown; OBJ material to GLTF PBR format conversion is not a lighting/render equivalence claim'})
    collision_path = output / 'collision.obj'; mesh.export(collision_path, digits=12)
    native = sapien.physx.PhysxCollisionShapeTriangleMesh(filename=str(collision_path), scale=[1,1,1],
        material=sapien.physx.PhysxMaterial(.6,.6,0.))
    native_vertices = np.asarray(native.vertices)
    error = max(cKDTree(mesh.vertices).query(native_vertices)[0].max(), cKDTree(native_vertices).query(mesh.vertices)[0].max())
    if not np.isfinite(error) or error > 1e-5: raise ValueError('native_triangle_geometry_changed')
    # Same consumer check, on the exact saved collision surface subsequently
    # consumed by the stationary-triangle loader. No winding reversal or repair.
    saved_collision = trimesh.load(collision_path, force='mesh', process=False)
    support = check_support(saved_collision, opening)
    write(output / 'NATIVE_GEOMETRY_AND_SUPPORT.json', {'native_triangle_vertex_error_m':float(error),
        'native_triangle_vertex_count':len(native_vertices), 'source_to_asset_singular_values':singular.tolist(),
        'native_oriented_support':support, 'scene_created':False,'physics_steps':0,'GPU_jobs':0})
    variant_id = digest({'raw_source':deps['source'],'transform':transform.tolist(), 'scale':1.,'generator':binding(__file__)})
    descriptor = {'schema':'stationary_mesh_container_v1', 'id':variant_id[:24], 'variant_id':variant_id,
        'source_id':job['source_id'], 'name':job['name'], 'partition':job['partition'],
        'source':deps['source'], 'license':job['source_record'].get('license'), 'source_dependencies':deps['resolver'],
        'source_aliases':job['source_aliases'], 'source_alias_split_consistent':job['alias_split_consistent'],
        'visual':binding(visual_path), 'collision':binding(collision_path), 'source_to_asset':transform.tolist(),
        'bounds_m':mesh.bounds.tolist(), 'dimensions_m':mesh.extents.tolist(), 'cavity':opening,
        'stable_pose_index':pose_index,'geometric_support_probability':probability,
        'native_triangle_vertex_error_m':float(error), 'native_policy':'static_triangle_mesh_only; cavity preserved',
        'roles':{'stationary_container':bool(support['passed']),'graspable_container':False},
        'metric_size':{'mode':job['scale_evidence'],'source_unit_to_m':1.0,'additional_scale_in_container_stage':1.0,
                       'source_geometry':binding(output/'SOURCE_GEOMETRY.json'),'per_object_metric_truth':'unknown'},
        'semantic_status':'retrieved_container_role_with_geometry_only; semantic identity and payload fit remain unverified',
        'physical_status':'unmeasured','relative_scale':1.0,'generator':binding(__file__),
        'native_support_check':binding(output/'NATIVE_GEOMETRY_AND_SUPPORT.json')}
    write(output / 'descriptor.json', descriptor)
    state = 'candidate_geometry_ready' if support['passed'] else 'native_support_failed'
    if not job['alias_split_consistent']: state = 'candidate_geometry_split_conflict'
    return {'source_id':job['source_id'],'status':state,'asset':binding(output/'descriptor.json'),
        'dimensions_m':mesh.extents.tolist(), 'cavity':opening, 'support':support,
        'native_triangle_vertex_error_m':float(error), 'elapsed_seconds':time.monotonic()-start,
        'source_scale_factor':1.,'physical_success_inferred':False,'default_admitted':False,'API_calls':0,'GPU_jobs':0,'physics_steps':0}


def worker(root, job_path):
    root, job_path = Path(root).resolve(), Path(job_path).resolve()
    protocol = validate(root)
    matches = [ref for ref in protocol['jobs'] if (ROOT / ref['path']).resolve() == job_path]
    if len(matches) != 1:
        raise ValueError('worker_job_not_in_frozen_source_scope')
    job_path = resolve(matches[0]); job = read(job_path); output = Path(job['output']).resolve()
    if output != job_path.parent or output.parent != root / 'jobs' or job_path.name != 'job.json':
        raise ValueError('worker_output_must_equal_registered_job_folder')
    reservation_path = output / 'reservation.json'
    if not reservation_path.is_file(): raise ValueError('worker_requires_prior_registered_reservation')
    reservation = read(reservation_path)
    if (reservation.get('job') != matches[0] or reservation.get('source_id') != job['source_id']
            or reservation.get('attempt') != 1):
        raise ValueError('worker_reservation_does_not_match_frozen_job')
    if (output/'worker_started.json').exists(): raise ValueError('existing_attempt_requires_recovery_not_retry')
    write(output/'worker_started.json',{'identity':identity(os.getpid()),'job':binding(job_path),'unix':time.time()})
    start = time.monotonic()
    try: result = compile_one(job_path)
    except Exception as error:
        from scipy.spatial import QhullError
        known = isinstance(error,(ValueError,QhullError))
        result = {'source_id':job['source_id'], 'status':'geometry_or_input_failed' if known else 'unknown_system_failure',
            'reason':type(error).__name__+':'+str(error), 'trace':traceback.format_exc(),
            'elapsed_seconds':time.monotonic()-start,'physical_success_inferred':False,'API_calls':0,'GPU_jobs':0,'physics_steps':0}
    write(output/'result.json',result)
    write(output/'worker_terminal.json',{'result':binding(output/'result.json'),'identity':identity(os.getpid()),'unix':time.time()})
    return result


def status(root):
    root = Path(root).resolve(); protocol = read(root/'PROTOCOL.json'); rows = []
    for ref in protocol['jobs']:
        job_path = resolve(ref); job = read(job_path); output = job_path.parent
        terminal = output/'terminal.json'; reservation = output/'reservation.json'
        if terminal.is_file(): row = read(terminal)
        elif (output/'worker_terminal.json').is_file():
            receipt = read(output/'worker_terminal.json'); result = read(resolve(receipt['result']))
            row = {'source_id':job['source_id'],'status':'worker_result_complete_awaiting_supervisor_receipt','result_status':result['status']}
        elif reservation.is_file():
            process = read(output/'process.json') if (output/'process.json').is_file() else {}
            old = process.get('identity'); live = bool(old and identity(old['pid']) == old)
            row = {'source_id':job['source_id'],'status':'running' if live else 'unknown_reserved_no_terminal'}
        else: row = {'source_id':job['source_id'],'status':'not_submitted'}
        rows.append(row)
    return {'schema':VERSION,'root':str(root),'registered':len(rows),'counts':dict(Counter(x['status']for x in rows)),
        'rows':rows,'API_calls':0,'GPU_jobs':0,'physics_steps':0,'new_default_assets':0}


def run(root):
    root = Path(root).resolve(); protocol = validate(root)
    with campaign_lock(root):
        sufficient_disk(root)
        if not (root/'START.json').exists():
            write(root/'START.json',{'identity':identity(os.getpid()),'unix':time.time(),'deadline_unix':time.time()+WALL_SECONDS})
        deadline = read(root/'START.json')['deadline_unix']
        work = []
        for ref in protocol['jobs']:
            path = resolve(ref); job = read(path); folder = path.parent
            if (folder/'terminal.json').exists(): continue
            if (folder/'worker_terminal.json').exists():
                receipt = read(folder/'worker_terminal.json'); result = read(resolve(receipt['result']))
                write(folder/'terminal.json',{'source_id':job['source_id'],'status':result['status'],
                    'result':receipt['result'],'recovered_from_complete_worker_receipt':True,'optimizer_or_asset_resubmitted':False})
                continue
            if (folder/'reservation.json').exists():
                raise RuntimeError('unknown_or_live_reservation_recover_before_admission:'+job['source_id'])
            work.append((path,job))
        def one(item):
            path,job = item; folder = path.parent
            if time.time() >= deadline: return None
            sufficient_disk(root)
            write(folder/'reservation.json',{'source_id':job['source_id'],'job':binding(path),'unix':time.time(),'attempt':1})
            if job['preflight_error'] or job['duplicate_selected_source']:
                result = {'source_id':job['source_id'],'status':'input_preflight_failed' if job['preflight_error'] else 'duplicate_selected_source',
                    'reason':job['preflight_error'],'canonical_source':job['duplicate_selected_source']}
                write(folder/'terminal.json',result); return result
            environment = dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            started = time.monotonic()
            with (folder/'worker.log').open('x') as log:
                process = subprocess.Popen([sys.executable,'-m','resimpair_assets.raw_metric_containers_v1','worker',
                    '--root',str(root),'--job',str(path)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=environment,start_new_session=True)
                write(folder/'process.json',{'identity':identity(process.pid),'unix':time.time()})
                try: code = process.wait(timeout=min(ASSET_SECONDS,max(.001,deadline-time.time())))
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGTERM)
                    try: process.wait(timeout=5)
                    except subprocess.TimeoutExpired: os.killpg(process.pid,signal.SIGKILL);process.wait()
                    code = 'timeout'
            if (folder/'worker_terminal.json').exists():
                receipt = read(folder/'worker_terminal.json'); result = read(resolve(receipt['result']))
                state = result['status']
            else: receipt = {}; state = 'source_timeout' if code == 'timeout' else 'unknown_system_failure'
            terminal = {'source_id':job['source_id'],'status':state,'returncode':code,
                'result':receipt.get('result'),'elapsed_seconds':time.monotonic()-started,'retries':0}
            write(folder/'terminal.json',terminal)
            print(job['source_id'],state,flush=True)
            return terminal
        # Only two admitted jobs at once. Unknown infrastructure failures stop
        # further admission; queued unsubmitted inputs keep their original IDs.
        pending = iter(work)
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            active = {executor.submit(one,item) for item in [next(pending,None),next(pending,None)] if item is not None}
            halt = False
            while active:
                future = next(as_completed(active)); active.remove(future); terminal = future.result()
                halt = halt or bool(terminal and terminal['status']=='unknown_system_failure')
                item = next(pending,None) if not halt and time.time()<deadline else None
                if item is not None: active.add(executor.submit(one,item))
        summary = status(root); rows = summary['rows']; assets = []
        for row in rows:
            if row['status']=='candidate_geometry_ready': assets.append(read(resolve(row['result']))['asset'])
        complete = not any(x['status'] in {'not_submitted','running','unknown_reserved_no_terminal',
                                          'worker_result_complete_awaiting_supervisor_receipt'} for x in rows)
        if complete:
            write(root/'CATALOG.json',{'schema':'stationary_container_catalog_v1','assets':assets,
                'scope':'candidate geometry only, no default or physical admission','rows':rows,'protocol':binding(root/'PROTOCOL.json'),
                'API_calls':0,'GPU_jobs':0,'physics_steps':0})
            write(root/'FINAL.json',summary)
        else:
            write(root/'snapshots'/f'{time.time_ns()}.json',summary)
        return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run','status','worker'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--sources',type=Path)
    parser.add_argument('--job',type=Path)
    args = parser.parse_args()
    if args.command=='prepare': result = prepare(args.root,args.sources)
    elif args.command=='run': result = run(args.root)
    elif args.command=='worker': result = worker(args.root,args.job)
    else: result = status(args.root)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False))
