"""Optional GPU workers. The cached-pose runtime never imports this module."""
import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np

from resimpair_grasp.common import binding, read, resolve, write
from .ledger import check_request


def check(job):
    if os.environ.get('RESIMPAIR_BENCHMARK_WORKER') != str(Path(job).resolve()):
        raise ValueError('A reserved GPU supervisor is required')
    spec = read(job)
    for item in spec['jobs']:
        check_request(spec['ledger'], item['identity'], item['request'], item['output'])
        resolve(item['request']['protocol'])
        for value in item['request']['code'].values(): resolve(value)
    return spec


def graspgenx(spec):
    import torch
    import trimesh
    from graspgenx import get_checkpoints_version_dir
    from graspgenx.utils.checkpoint_io import load_model_cfg
    from graspgenx.grasp_server import GraspGenXSampler
    from resimpair_pickplace.graspgenx import UPSTREAM, tcp_transform
    torch.set_num_threads(2)
    start = time.monotonic()
    checkpoint = get_checkpoints_version_dir()
    cfg = load_model_cfg(str(checkpoint / 'gen'), str(checkpoint / 'dis'))
    sampler = GraspGenXSampler(cfg, 'franka_panda', assets_dir=str(UPSTREAM / 'assets'))
    urdf = UPSTREAM / 'ext/gripper_descriptions/gripper_descriptions/assets/x_grippers/franka_panda/gripper.urdf'
    conversion = tcp_transform(urdf)
    write(Path(spec['batch_output']) / 'model.json', {
        'generator': binding(cfg.eval.gen_checkpoint), 'discriminator': binding(cfg.eval.dis_checkpoint),
        'load_seconds': time.monotonic() - start, 'base_to_native_tcp': conversion.tolist(),
        'urdf': binding(urdf), 'torch_version': torch.__version__})
    for item in spec['jobs']:
        out, request = Path(item['output']), item['request']
        started = time.monotonic()
        write(out / 'worker_started.json', {'unix': time.time(), 'pid': os.getpid()})
        try:
            seed = request['seed']
            torch.manual_seed(seed); np.random.seed(seed)
            descriptor = read(resolve(request['asset']))
            mesh = trimesh.load(resolve(descriptor['visual']), force='scene').dump(concatenate=True)
            points, _ = trimesh.sample.sample_surface(mesh, 3500, seed=seed)
            input_seconds = time.monotonic() - started
            torch.cuda.synchronize(); start = time.monotonic()
            poses, scores = GraspGenXSampler.run_inference(np.asarray(points), sampler, grasp_threshold=-1.,
                num_grasps=256, topk_num_grasps=256, min_grasps=1, max_tries=1, remove_outliers=False)
            torch.cuda.synchronize()
            inference_seconds = time.monotonic() - start
            model_poses, scores = poses.detach().cpu().numpy(), scores.detach().cpu().numpy()
            order = np.argsort(-scores.reshape(-1), kind='stable')
            model_poses, scores = model_poses[order], scores[order]
            native = model_poses @ conversion
            np.savez_compressed(out / 'raw.npz', poses=native, scores=scores, model_base_poses=model_poses)
            write(out / 'result.json', {'method': 'graspgenx', 'asset': request['asset'],
                'raw': binding(out / 'raw.npz'), 'proposal_count': len(native),
                'input_seconds': input_seconds, 'inference_seconds': inference_seconds,
                'worker_seconds': time.monotonic() - started, 'physical_success': 'unmeasured',
                'native_tcp_no_vertical_snapping': True})
        except Exception as error:
            write(out / 'error.json', {'error': repr(error), 'worker_seconds': time.monotonic() - started})
        print(item['identity'], 'finished', flush=True)


def cudex(spec):
    import torch
    import trimesh
    import yaml
    from curobo.wrap.reacher.grasp_solver import GraspSolver, GraspSolverConfig
    from resimpair_grasp.cudex_native import hand_to_tcp
    from .proposals import tcp_to_hand_seed
    for item in spec['jobs']:
        out, request = Path(item['output']), item['request']
        started = time.monotonic()
        write(out / 'worker_started.json', {'unix': time.time(), 'pid': os.getpid()})
        try:
            torch.manual_seed(request['seed']); np.random.seed(request['seed'])
            descriptor = read(resolve(request['asset']))
            mesh = trimesh.load(resolve(descriptor['visual']), force='scene', process=False).to_geometry()
            mesh_path = out / 'object.obj'; mesh.export(mesh_path)
            z = float(mesh.bounds[0, 2])
            world = {'mesh': {'payload': {'file_path': str(mesh_path), 'pose': [0, 0, 0, 1, 0, 0, 0]}},
                'cuboid': {'table': {'dims': [2, 2, .10], 'pose': [0, 0, z - .05, 1, 0, 0, 0]}}}
            manip = yaml.safe_load(resolve(request['manip']).read_text())
            manip['seed_num'] = 256
            config = GraspSolverConfig.load_from_robot_config(world_model=[world],
                manip_name_list=['payload'], obj_gravity_center=[mesh.center_mass.tolist()],
                obj_obb_length=[[float(np.linalg.norm(mesh.bounding_box_oriented.primitive.extents) / 2)]],
                manip_config_data=manip, use_cuda_graph=False)
            solver = GraspSolver(config)
            seed_config = None
            if request['method'] == 'hybrid':
                raw = np.load(resolve(request['learned_raw']), allow_pickle=False)
                if len(raw['poses']) != 256:
                    raise ValueError('Hybrid requires exactly 256 learned seeds; no random fallback')
                seeds = tcp_to_hand_seed(raw['poses'])
                np.savez_compressed(out / 'initial_seeds.npz', solution=seeds)
                seed_config = torch.as_tensor(seeds[None], device='cuda', dtype=torch.float32)
            setup_seconds = time.monotonic() - started
            torch.cuda.synchronize(); start = time.monotonic()
            # n_iters=500 is already in the pinned gradient config. The solver's
            # newton_iters argument sets OUTER iterations (each has 50 steps).
            result = solver.solve_batch_env(seed_config=seed_config, return_seeds=256, num_seeds=256)
            torch.cuda.synchronize(); measured_solve_seconds = time.monotonic() - start
            arrays = {name: getattr(result, name).detach().cpu().numpy() for name in
                ('solution', 'success', 'grasp_error', 'dist_error', 'contact_point', 'contact_frame')}
            q = arrays['solution'][0]
            final = q[:, -1, :] if q.ndim == 3 else q
            native = np.array([hand_to_tcp(value) for value in final])
            np.savez_compressed(out / 'raw.npz', poses=native, **arrays)
            write(out / 'result.json', {'method': request['method'], 'asset': request['asset'],
                'raw': binding(out / 'raw.npz'), 'proposal_count': len(native),
                'strict_optimizer_success': int(arrays['success'].sum()),
                'setup_seconds': setup_seconds, 'solve_seconds': measured_solve_seconds,
                'upstream_solve_seconds': float(result.solve_time),
                'worker_seconds': time.monotonic() - started, 'physical_success': 'unmeasured',
                'seed_count': 256, 'iterations': 500, 'native_tcp_no_vertical_snapping': True,
                'seed_origin': 'learned_only' if seed_config is not None else 'upstream_sampler'})
        except Exception as error:
            import traceback
            traceback.print_exc()
            write(out / 'error.json', {'error': repr(error), 'worker_seconds': time.monotonic() - started})
        print(item['identity'], 'finished', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('job'); a = p.parse_args()
    spec = check(a.job)
    if spec['method'] == 'graspgenx': graspgenx(spec)
    else: cudex(spec)
