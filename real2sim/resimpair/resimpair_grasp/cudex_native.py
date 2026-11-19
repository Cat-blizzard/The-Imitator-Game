"""Opt-in adapter to the pinned, actual cuDex solver; never fabricates solutions.

Preparation is CPU-only. GPU work requires a finite campaign and exclusive GPU
lease. Optimizer feasibility and native physical success are separate outputs.
"""
from pathlib import Path
import copy
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import yaml

from .common import ROOT, binding, read, resolve, write, digest
from .grasps import GRIPPER

UPSTREAM = ROOT / 'runtime/upstream/cudex-native-panda'
COMMIT = 'e3d84aea7ee6c87df99b4f980cac8b6a52b2d362'


def prepare(directory):
    """Build a native gripper config from the simulator URDF, with TCP binding.

    Removing the mimic tag makes both fingers explicit optimization variables;
    the upstream tendon constraint enforces their equality. Meshes and joint
    origins are unchanged. Sphere coverage is only an optimizer approximation;
    the existing native-mesh path/closure guards remain mandatory downstream.
    """
    from resimpair_tabletop.geometry import panda_meshes
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'PolicyGen/mani_skill/assets/robots/panda/panda_v3.urdf'
    root = ET.parse(source).getroot()
    for mesh in root.findall('.//mesh'):
        mesh.set('filename', str((source.parent / mesh.get('filename')).resolve()))
    for joint in root.findall('joint'):
        mimic = joint.find('mimic')
        if mimic is not None:
            joint.remove(mimic)
    urdf = directory / 'native_panda.urdf'
    ET.ElementTree(root).write(urdf, encoding='unicode')

    # Native collision geometry in TCP coordinates, opening zero. Fixed camera
    # attachments are merged into the hand approximation; no phantom fingers.
    hand_meshes = []
    finger_meshes = {'panda_leftfinger': [], 'panda_rightfinger': []}
    for name, _, mesh in panda_meshes(opening=0)[0]:
        mesh = mesh.copy()
        if name in finger_meshes:
            mesh.apply_translation([0, 0, .1034 - .0584])
            finger_meshes[name].append(mesh)
        else:
            mesh.apply_translation([0, 0, .1034])
            hand_meshes.append(mesh)
    def covering_spheres(meshes, pitch):
        mesh = trimesh.util.concatenate(meshes)
        voxels = mesh.voxelized(pitch)
        return [{'center': p.tolist(), 'radius': float(np.sqrt(3)*pitch/2)}
                for p in voxels.points]
    spheres = {'panda_hand': covering_spheres(hand_meshes, .022)}
    for name, meshes in finger_meshes.items():
        # Contact sphere 0 is inscribed in the actual 17.5 x 15.2 x 18.5 mm pad.
        sign = 1 if name == 'panda_leftfinger' else -1
        spheres[name] = [{'center': [0, sign*.00758, .04525], 'radius': .0076}]
        spheres[name] += covering_spheres(meshes, .008)
    sphere_path = directory / 'spheres.yml'
    sphere_path.write_text(yaml.safe_dump({'collision_spheres': spheres}))
    transfer = directory / 'transfer.yml'
    transfer.write_text(yaml.safe_dump({'panda_hand': {
        'r': [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
        't': [-.1034, 0, 0]}}))
    robot = yaml.safe_load((UPSTREAM/'src/curobo/content/configs/robot/franka_ext_ee.yml').read_text())
    k = robot['robot_cfg']['kinematics']
    k.update(urdf_path=str(urdf), asset_root_path=str(source.parent),
             hand_pose_transfer_path=str(transfer), collision_spheres=str(sphere_path),
             ee_link='panda_hand_tcp', link_names=['panda_leftfinger', 'panda_rightfinger', 'panda_hand_tcp'],
             mesh_link_names=['panda_hand', 'panda_leftfinger', 'panda_rightfinger'])
    robot_path = directory/'robot.yml'
    robot_path.write_text(yaml.safe_dump(robot))
    manip = yaml.safe_load((UPSTREAM/'src/curobo/content/configs/manip/parallel_gripper/franka_ext.yml').read_text())
    manip['robot_file'] = str(robot_path)
    # Floating-gripper optimization does not use the optional arm IK solver.
    # Full native dual-arm IK is checked against the actual scene downstream.
    manip['robot_file_with_arm'] = None
    manip['grasp_contact_strategy']['contact_points_name'] = ['panda_leftfinger/0', 'panda_rightfinger/0']
    # Keep upstream seeds, iterations, DFC, distance schedule and thresholds.
    manip_path = directory/'manip.yml'
    manip_path.write_text(yaml.safe_dump(manip))
    write(directory/'manifest.json', {'schema': 'cudex_native_panda_config_v1',
        'upstream_commit': COMMIT, 'native_urdf': binding(source),
        'files': [binding(p) for p in (urdf, sphere_path, transfer, robot_path, manip_path)],
        'sphere_counts': {n: len(s) for n, s in spheres.items()},
        'native_hand_to_tcp_m': [0, 0, .1034],
        'native_collision_required': True, 'published_extended_finger_reproduction': False})
    return manip_path


def worker(job):
    """One actual upstream optimization; CUDA_VISIBLE_DEVICES set by supervisor."""
    import torch
    from curobo.wrap.reacher.grasp_solver import GraspSolver, GraspSolverConfig
    job = read(job)
    output = Path(job['output'])
    descriptor = read(resolve(job['asset']))
    start = time.monotonic()
    torch.manual_seed(job['seed'])
    np.random.seed(job['seed'])
    mesh = trimesh.load(resolve(descriptor['visual']), force='scene', process=False).to_geometry()
    mesh_path = output/'object.obj'
    mesh.export(mesh_path)
    # Object-frame tabletop, exactly matching the retained asset support pose.
    z = float(descriptor['collision_bounds_m'][0][2])
    world = {'mesh': {'payload': {'file_path': str(mesh_path), 'pose': [0, 0, 0, 1, 0, 0, 0]}},
             'cuboid': {'table': {'dims': [2, 2, .10], 'pose': [0, 0, z-.05, 1, 0, 0, 0]}}}
    manip = yaml.safe_load(Path(job['manip']).read_text())
    write(output/'effective_input.json', {'world': world, 'manip': manip,
        'asset': job['asset'], 'seed': job['seed'], 'grasp_threshold': .001,
        'distance_threshold': .01, 'metric': 'upstream_DFC',
        'physical_success': 'unmeasured'})
    config = GraspSolverConfig.load_from_robot_config(world_model=[world],
        manip_name_list=['payload'], obj_gravity_center=[mesh.center_mass.tolist()],
        obj_obb_length=[[float(np.linalg.norm(mesh.bounding_box_oriented.primitive.extents)/2)]],
        manip_config_data=manip, use_cuda_graph=False)
    solver = GraspSolver(config)
    result = solver.solve_batch_env(return_seeds=solver.num_seeds)
    torch.cuda.synchronize()
    arrays = {name: getattr(result, name).detach().cpu().numpy() for name in
              ('solution', 'success', 'grasp_error', 'dist_error', 'contact_point', 'contact_frame')}
    np.savez_compressed(output/'raw_result.npz', **arrays)
    write(output/'optimizer_result.json', {'schema': 'actual_cudex_result_v1',
        'arrays': binding(output/'raw_result.npz'), 'shapes': {n:list(v.shape) for n,v in arrays.items()},
        'optimizer_feasible_count': int(arrays['success'].sum()),
        'seed_count': solver.num_seeds, 'elapsed_seconds': time.monotonic()-start,
        'solve_seconds': float(result.solve_time), 'physics_status': 'unmeasured'})


def run_job(campaign, asset, manip, name, seed=20261026, gpu=2):
    """Bounded, single-submit job; unknown outcomes are never resubmitted."""
    import fcntl
    from contextlib import ExitStack
    from corl2026_workshop.experiments.autonomous_pipeline_v4.resources import lock_paths, gpu_snapshot, check_idle
    campaign = Path(campaign).resolve()
    protocol = read(ROOT/'validation/cudex_physical_20261026/protocol.json')
    with ExitStack() as stack:
        for path in lock_paths(gpu):
            lock = stack.enter_context(path.open('r+'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output = campaign/name
        if (output/'terminal.json').exists():
            return read(output/'terminal.json')
        if (output/'reservation.json').exists():
            raise RuntimeError('Unresolved optimizer submission; query existing process before retry')
        if len(list(campaign.glob('*/reservation.json'))) >= protocol['gpu_optimization_limit']:
            raise RuntimeError('Optimizer job budget exhausted')
        snapshot = gpu_snapshot(gpu)
        check_idle(snapshot)
        output.mkdir(parents=True, exist_ok=True)
        write(output/'job.json', {'asset': asset, 'manip': str(Path(manip).resolve()),
            'seed': seed, 'output': str(output)})
        write(output/'reservation.json', {'start_unix': time.time(), 'gpu': snapshot,
            'input': binding(output/'job.json'), 'code': binding(__file__),
            'timeout_s': protocol['gpu_optimization_seconds_per_asset']})
        deps = ROOT/'runtime/cudex_deps'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=snapshot['uuid'],
            PYTHONPATH=os.pathsep.join(map(str, [ROOT, deps, deps/'cmeel.prefix/lib/python3.12/site-packages',
                UPSTREAM/'src', UPSTREAM/'src/curobo/geom/cpp'])),
            LD_LIBRARY_PATH=str(deps/'cmeel.prefix/lib')+os.pathsep+os.environ.get('LD_LIBRARY_PATH',''),
            OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        start = time.monotonic()
        with (output/'worker.log').open('w') as log:
            p = subprocess.Popen([sys.executable, '-m', 'resimpair_grasp.cudex_native',
                'worker', str(output/'job.json')], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            write(output/'process.json', {'pid':p.pid, 'start_unix':time.time()})
            try:
                code = p.wait(timeout=protocol['gpu_optimization_seconds_per_asset'])
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(p.pid, signal.SIGTERM)
                try: p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL); p.wait()
                code = 'timeout'
        terminal = {'returncode':code, 'elapsed_seconds':time.monotonic()-start,
                    'result_exists':(output/'optimizer_result.json').exists()}
        write(output/'terminal.json', terminal)
        return terminal


def hand_to_tcp(solution):
    """cuDex root xyz + wxyz + two joints -> native TCP; preserve 6D pose."""
    q = np.asarray(solution, float)
    if q.shape != (9,) or not np.isfinite(q).all():
        raise ValueError('expected finite floating hand pose and two fingers')
    if abs(np.linalg.norm(q[3:7])-1) > .01:
        raise ValueError('nonunit_optimizer_quaternion')
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(q[3:7], scalar_first=True).as_matrix()
    T[:3, 3] = q[:3] + T[:3, :3] @ [0, 0, .1034]
    return T


def export_cache(job_directory, output, max_checks=128, policy='strict'):
    """Export strict successes or explicitly unqualified optimized proposals.

    No low-friction proxy, snapping to vertical, or hidden geometric fallback.
    All failures stay in the asset denominator even if no cache can be emitted.
    """
    from .cartesian_clearance import VerticalClearance
    from resimpair_pickplace.closure_quality import ClosingSweep
    job_directory, output = Path(job_directory), Path(output)
    job = read(job_directory/'job.json')
    descriptor = read(resolve(job['asset']))
    result = read(job_directory/'optimizer_result.json')
    arrays = np.load(resolve(result['arrays']), allow_pickle=False)
    q = arrays['solution'][0]
    # Upstream returns initial/intermediate/final states for each sorted seed.
    final = q[:, -1, :] if q.ndim == 3 else q
    success = arrays['success'][0].astype(bool)
    mesh = trimesh.util.concatenate([trimesh.load(resolve(b), force='mesh') for b in descriptor['collisions']])
    clearance, closure = VerticalClearance(mesh), ClosingSweep(mesh)
    candidates, decisions, seen = [], [], set()
    if policy not in {'strict', 'optimized_proposals'}:
        raise ValueError('unknown_optimizer_export_policy')
    indices = np.flatnonzero(success) if policy == 'strict' else np.arange(len(final))
    for i in indices[:max_checks]:
        pose = hand_to_tcp(final[i])
        # Optimized fingers can retain a finite contact gap. The cache width is
        # the opposing OBJECT contact span (same meaning as geometric caches),
        # not the open/partially closed optimizer joint state.
        contacts = arrays['contact_point'][0, i].reshape(2, 3)
        width = float(abs((contacts[1]-contacts[0]) @ pose[:3, 1]))
        row = {'upstream_index':int(i), 'width_m':width,
               'optimizer_joint_opening_m':float(final[i, 7:].sum()),
               'optimizer_success':bool(success[i])}
        if not .012 <= width <= .077:
            decisions.append({**row, 'rejected':'native_opening_range'}); continue
        bucket = tuple(np.round(np.r_[pose[:3, 3]/.004,
            Rotation.from_matrix(pose[:3, :3]).as_rotvec()/.12]).astype(int))
        if bucket in seen:
            decisions.append({**row, 'rejected':'duplicate'}); continue
        seen.add(bucket)
        ok, gap, reason = clearance.check(pose, float(mesh.bounds[0, 2]), .002)
        row['open_hand_check'] = {'passed':bool(ok), 'clearance_m':gap, 'reason':reason}
        if not ok:
            decisions.append({**row, 'rejected':reason}); continue
        closing = closure.measure(pose, width)
        row['closing_sweep'] = closing
        if not closing['contact_pair_reachable']:
            decisions.append({**row, 'rejected':'native_first_contact_width_mismatch'}); continue
        candidates.append({'id':f'cudex_{i:04d}', 'tcp_in_asset':pose.tolist(), 'width_m':width,
            'physical_status':'unmeasured', 'optimizer_success':bool(success[i]), 'closing_sweep':closing,
            'optimizer_rank':int(i), 'grasp_error':arrays['grasp_error'][0, i].tolist()})
        decisions.append({**row, 'rejected':None})
    signature = {'descriptor':job['asset'], 'gripper':GRIPPER, 'upstream_commit':COMMIT,
        'optimizer_output':binding(job_directory/'optimizer_result.json'),
        'adapter':binding(__file__), 'export_policy':policy,
        'native_clearance':binding(ROOT/'resimpair_grasp/cartesian_clearance.py')}
    write(output/'decisions.json', {'upstream_success':int(success.sum()), 'checked':len(decisions),
        'maximum_checks':max_checks, 'retained':len(candidates), 'rows':decisions, 'policy':policy,
        'rejected_not_removed_from_asset_denominator':True})
    write(output/'grasp_cache.json', {'schema':'resimpair_mesh_grasp_cache_v1',
        'signature':signature, 'key':digest(signature), 'candidates':candidates})
    return binding(output/'grasp_cache.json')


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker','run','export'])
    p.add_argument('path');p.add_argument('--asset');p.add_argument('--manip');p.add_argument('--name')
    p.add_argument('--output');p.add_argument('--policy',default='strict',choices=['strict','optimized_proposals']);a=p.parse_args()
    if a.mode=='prepare': print(prepare(a.path))
    elif a.mode=='worker': worker(a.path)
    elif a.mode=='export': print(export_cache(a.path,a.output,policy=a.policy))
    else: print(run_job(a.path, binding(a.asset), a.manip, a.name))


if __name__=='__main__': main()
