"""Batch qualify stationary mesh containers without filling their cavities.

Names only retrieve candidates. Geometric rays establish a conservative flat
floor disk and surrounding walls. Native static triangle meshes preserve the
opening; no convex hull is substituted. This is not a physical certificate.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sqlite3
import subprocess
import sys
import time

import numpy as np
import trimesh
from resimpair_grasp.common import ROOT, binding, read, resolve, write
# Install the isolated Embree path, not a baseline environment modification.
from resimpair_tabletop import geometry


def load_mesh(path):
    scene = trimesh.load(path, force='scene', process=False)
    for name, part in list(scene.geometry.items()):
        if not isinstance(part, trimesh.Trimesh):
            scene.delete_geometry(name)
    mesh = scene.to_geometry()
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError('no_triangle_geometry')
    if len(mesh.faces) > 300000:
        raise ValueError('container_face_budget')
    if not np.isfinite(mesh.vertices).all():
        raise ValueError('nonfinite_container')
    return mesh


def cavity(mesh):
    """Return a disk supported by a near-flat floor and enclosed horizontally.

    Finite ray samples are an offline qualification, never a guarantee for an
    arbitrary payload. Actual payload/hand meshes and native paths still check.
    """
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    ray = RayMeshIntersector(mesh)
    lower, upper = mesh.bounds
    center = (lower + upper) / 2
    angles = np.linspace(0, 2*np.pi, 64, endpoint=False)
    directions = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(64)])
    best = None
    for dx, dy in [(0, 0), (-.015, 0), (.015, 0), (0, -.015), (0, .015)]:
        xy = center[:2] + [dx, dy]
        hit, _, face = ray.intersects_location([[*xy, upper[2]+.02]], [[0, 0, -1]], multiple_hits=False)
        if not len(hit):
            continue
        floor = float(hit[0, 2])
        depth = float(upper[2]-floor)
        if not .008 <= depth <= .15 or abs(mesh.face_normals[face[0], 2]) < .90:
            continue
        radii = []
        # Check the whole usable lower cavity, with 64 angular rays per level.
        for z in np.linspace(floor+.004, min(upper[2]-.003, floor+.055), 5):
            starts = np.tile([*xy, z], (64, 1))
            loc, ids, _ = ray.intersects_location(starts, directions, multiple_hits=False)
            if len(set(ids)) != 64:
                break
            radii.append(float(np.min(np.linalg.norm(loc[:, :2]-xy, axis=1))))
        if len(radii) != 5:
            continue
        radius = min(radii)-.006
        if radius < .045:
            continue
        # Shrink until a horizontal disk is actually supported. Bowl interiors
        # with steep sides cannot claim their full rim width as a flat floor.
        for _ in range(10):
            pts = np.vstack([xy, xy+directions[:, :2]*radius, xy+directions[:, :2]*(radius/2)])
            starts = np.column_stack([pts, np.full(len(pts), upper[2]+.02)])
            loc, ids, _ = ray.intersects_location(starts, np.tile([0, 0, -1], (len(pts), 1)), multiple_hits=False)
            if len(set(ids)) == len(pts) and np.max(np.abs(loc[:, 2]-floor)) <= .003:
                break
            radius *= .88
        else:
            continue
        if radius < .04:
            continue
        value = {'center_xy_m': xy.tolist(), 'floor_z_m': floor,
                 'flat_floor_radius_m': radius, 'minimum_wall_radius_m': min(radii),
                 'rim_z_m': float(upper[2]), 'depth_m': depth,
                 'rays_per_level': 64, 'wall_levels': 5, 'floor_tolerance_m': .003,
                 'qualification': 'sampled_mesh_geometry_only'}
        if best is None or radius > best['flat_floor_radius_m']:
            best = value
    if best is None:
        raise ValueError('no_supported_open_cavity')
    return best


def prepare(row, folder):
    start = time.monotonic()
    source = Path(row['path'])
    if binding(source)['sha256'] != row['source_sha']:
        raise ValueError('container_source_changed')
    original = load_mesh(source)
    hull = original.convex_hull
    lengths = np.sort(hull.bounding_box_oriented.primitive.extents)
    scale = .28 / lengths[-1]
    mesh = original.copy(); mesh.apply_scale(scale)
    stable_hull = mesh.convex_hull
    if len(stable_hull.faces) > 400:
        stable_hull = stable_hull.simplify_quadric_decimation(face_count=400).convex_hull
    poses, probabilities = trimesh.poses.compute_stable_poses(stable_hull, n_samples=1, sigma=0, threshold=.005)
    accepted = []
    for i, (pose, probability) in enumerate(zip(poses[:10], probabilities[:10])):
        placed = mesh.copy(); placed.apply_transform(pose)
        center = placed.bounds.mean(0); center[2] = placed.bounds[0, 2]
        placed.apply_translation(-center)
        try:
            opening = cavity(placed)
        except ValueError:
            continue
        T = np.eye(4); T[:3, 3] = -center
        accepted.append((opening['flat_floor_radius_m'], placed, T@pose@np.diag([scale, scale, scale, 1]), opening, float(probability), i))
    if not accepted:
        raise ValueError('no_upright_container_support_pose')
    _, mesh, transform, opening, probability, pose_index = max(accepted, key=lambda x:x[0])
    folder.mkdir(parents=True, exist_ok=True)
    mesh.export(folder/'visual.glb'); mesh.export(folder/'collision.obj')
    record = json.loads(row['record'])
    d = {'schema':'stationary_mesh_container_v1', 'id':hashlib.sha256(row['source_sha'].encode()).hexdigest()[:16],
         'source_id':row['id'], 'name':row['name'], 'partition':row['partition'],
         'source':binding(source), 'license':record.get('license'),
         'visual':binding(folder/'visual.glb'), 'collision':binding(folder/'collision.obj'),
         'source_to_asset':transform.tolist(), 'bounds_m':mesh.bounds.tolist(),
         'dimensions_m':mesh.extents.tolist(), 'cavity':opening,
         'stable_pose_index':pose_index, 'geometric_support_probability':probability,
         'native_policy':'static_triangle_mesh_only; preserve cavity',
         'roles':{'stationary_container':True, 'graspable_container':False},
         'metric_size':'synthetic 0.28 m longest dimension; not recovered real scale',
         'semantic_status':'name retrieved, cavity measured, task-specific identity unverified',
         'physical_status':'unmeasured', 'elapsed_seconds':time.monotonic()-start}
    write(folder/'descriptor.json', d)
    return binding(folder/'descriptor.json')


def plan(index, root, limit, seed):
    if root.exists() or not 1 <= limit <= 2000:
        raise ValueError('New bounded container batch required')
    c = sqlite3.connect(index); c.row_factory=sqlite3.Row
    words = re.compile(r'\b(bowl|tray|basket|basin|bin|dish|tub|container)s?\b', re.I)
    candidates = [dict(r) for r in c.execute('select * from assets')
                  if r['path'] and words.search((r['name'] or '').replace('_', ' '))
                  and Path(r['path']).is_file() and Path(r['path']).suffix.lower() in {'.glb','.gltf','.obj','.ply','.stl'}]
    c.close()
    candidates.sort(key=lambda r:hashlib.sha256(f'{seed}:{r["source_sha"]}'.encode()).hexdigest())
    seen = set(); unique = []
    for row in candidates:
        if row['source_sha'] not in seen:
            seen.add(row['source_sha']); unique.append(row)
    root.mkdir(parents=True)
    jobs = unique[:limit]
    write(root/'plan.json', {'schema':'container_batch_v1','code':binding(__file__),
          'index':binding(index),'seed':seed,'retrieved':len(unique),'jobs':jobs,
          'finite_timeout_seconds':60,'maximum_workers':4,'api_calls':0,
          'physics_executions':0,'failures_retained':True})
    return {'planned':len(jobs),'name_candidates':len(unique),'root':str(root)}


def run(root, workers):
    spec=read(root/'plan.json');resolve(spec['code'])
    if not 1 <= workers <= spec['maximum_workers']: raise ValueError('Worker limit')
    def one(i,row):
        folder=root/'assets'/row['source_sha'][:16]
        if (folder/'terminal.json').exists(): return read(folder/'terminal.json')
        folder.mkdir(parents=True,exist_ok=True)
        if (folder/'reservation.json').exists(): raise ValueError('Unknown previous job; recover first')
        write(folder/'reservation.json',{'start_unix':time.time(),'job':row,'code':spec['code']})
        start=time.monotonic()
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with (folder/'worker.log').open('x') as log:
            p=subprocess.Popen([sys.executable,'-m','resimpair_pickplace.containers','worker','--root',str(root),'--number',str(i)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try: code=p.wait(timeout=spec['finite_timeout_seconds'])
            except subprocess.TimeoutExpired: os.killpg(p.pid,signal.SIGKILL);p.wait();code='timeout'
        d=folder/'descriptor.json'
        row={'source_id':row['id'],'status':'geometry_ready' if code==0 and d.exists() else 'rejected',
             'returncode':code,'elapsed_seconds':time.monotonic()-start}
        if row['status']=='geometry_ready': row['descriptor']=binding(d)
        else:
            lines=(folder/'worker.log').read_text(errors='replace').splitlines()
            row['reason']=lines[-1] if lines else str(code)
        write(folder/'terminal.json',row);return row
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(one,i,r) for i,r in enumerate(spec['jobs'])]):
            result=f.result();print(json.dumps(result),flush=True)
    rows=[read(root/'assets'/r['source_sha'][:16]/'terminal.json') for r in spec['jobs']]
    value={'schema':'stationary_container_catalog_v1','plan':binding(root/'plan.json'),
           'assets':[r['descriptor'] for r in rows if r['status']=='geometry_ready'],
           'rows':rows,'api_calls':0,'physics_executions':0}
    write(root/'catalog.json',value)
    return {'catalog':str(root/'catalog.json'),'qualified':len(value['assets']),'attempted':len(rows)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('plan');q.add_argument('--index',type=Path,default=ROOT/'assets/expanded_library/raw_roles_v4.sqlite');q.add_argument('--root',type=Path,required=True);q.add_argument('--limit',type=int,default=128);q.add_argument('--seed',type=int,default=2026102671)
    q=s.add_parser('run');q.add_argument('--root',type=Path,required=True);q.add_argument('--workers',type=int,default=4)
    q=s.add_parser('worker');q.add_argument('--root',type=Path,required=True);q.add_argument('--number',type=int,required=True)
    a=p.parse_args()
    if a.command=='plan':r=plan(a.index,a.root,a.limit,a.seed)
    elif a.command=='run':r=run(a.root,a.workers)
    else:
        spec=read(a.root/'plan.json');resolve(spec['code']);row=spec['jobs'][a.number]
        r=prepare(row,a.root/'assets'/row['source_sha'][:16])
    print(json.dumps(r,indent=2))
