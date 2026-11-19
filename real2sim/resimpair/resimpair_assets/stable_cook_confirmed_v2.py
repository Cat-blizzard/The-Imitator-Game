"""Return only an OBJ whose exact bytes have passed a fresh-process native cook.

The old helper returned an untested final export. This helper instead tests
the file it will return. At most three native-geometry exports are allowed;
the fourth read is confirmation only. No renderer, scene or dynamics is used.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from resimpair_grasp.common import ROOT, binding, read, resolve, write

VERSION = 'stable_cook_confirmed_v2'
MAX_CONVERSIONS = 3
CONVERGENCE_M = 1e-6
RUNTIME_REPLAY_M = 1e-5
PROBE_TIMEOUT_SECONDS = 60


def pointset_error(a, b):
    """Same symmetric nearest-vertex distance; not a surface-distance proxy."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if (a.ndim != 2 or b.ndim != 2 or a.shape[1] != 3 or b.shape[1] != 3
            or not len(a) or not len(b) or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError('finite_nonempty_three_dimensional_vertex_arrays_required')
    return float(max(cKDTree(a).query(b)[0].max(), cKDTree(b).query(a)[0].max()))


def probe_worker(job_path):
    """One independent CPU load/cook; invoked by the bounded parent helper."""
    began=time.monotonic();job_path=Path(job_path).resolve();job=read(job_path)
    path=resolve(job['input']);out=job_path.parent
    if job['implementation'] != binding(__file__):
        raise ValueError('native_probe_implementation_binding_mismatch')
    if (out/'result.json').exists():raise ValueError('preserve_existing_native_probe')
    source=trimesh.load(path,force='mesh',process=False)
    import sapien
    material=sapien.physx.PhysxMaterial(.5,.5,0.)
    cook_start=time.monotonic()
    shape=sapien.physx.PhysxCollisionShapeConvexMesh(filename=str(path),scale=[1.,1.,1.],material=material)
    cook_seconds=time.monotonic()-cook_start
    vertices=np.asarray(shape.vertices).copy();faces=np.asarray(shape.triangles).copy()
    error=pointset_error(source.vertices,vertices)
    arrays=out/'native_arrays.npz'
    if arrays.exists():raise ValueError('preserve_existing_native_arrays')
    np.savez(arrays,vertices=vertices,faces=faces)
    result={'schema':VERSION,'job':binding(job_path),'input':binding(path),
        'native_arrays':binding(arrays),'worker_pid':os.getpid(),'parent_pid':job['parent_pid'],
        'input_vertex_count':len(source.vertices),'native_vertex_count':len(vertices),
        'input_face_count':len(source.faces),'native_face_count':len(faces),
        'vertex_error_m':error,'convergence_threshold_m':CONVERGENCE_M,
        'runtime_threshold_m':RUNTIME_REPLAY_M,'convergence_passed':error<CONVERGENCE_M,
        'runtime_replay_passed':error<RUNTIME_REPLAY_M,
        'scale':np.asarray(shape.scale).tolist(),
        'local_pose':np.asarray(shape.local_pose.to_transformation_matrix()).tolist(),
        'density_kg_m3':float(shape.density),
        'sapien_version':getattr(sapien,'__version__','unknown'),
        'wall_seconds':time.monotonic()-began,'native_cook_seconds':cook_seconds,
        'scene_created':False,'dynamic_steps':0,'GPU_jobs':0,'API_calls':0}
    write(out/'result.json',result)
    return result


def probe_file(path, output):
    """Expose an exact-file replay check for CPU regression and read-only audit."""
    path=Path(path).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('fresh_native_confirmation_directory_required')
    job={'schema':VERSION,'input':binding(path),'implementation':binding(__file__),
         'parent_pid':os.getpid(),'timeout_seconds':PROBE_TIMEOUT_SECONDS}
    write(output/'job.json',job)
    env=dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    start=time.monotonic()
    try:
        result=subprocess.run([sys.executable,'-m','resimpair_assets.stable_cook_confirmed_v2',
            '--probe',str(output/'job.json')],cwd=ROOT,env=env,capture_output=True,text=True,
            timeout=PROBE_TIMEOUT_SECONDS,check=False)
        (output/'stdout.log').write_text(result.stdout)
        (output/'stderr.log').write_text(result.stderr)
        write(output/'SUPERVISOR.json',{'returncode':result.returncode,'wall_seconds':time.monotonic()-start,
            'child_timeout_seconds':PROBE_TIMEOUT_SECONDS,'parent_pid':os.getpid()})
        if result.returncode:raise RuntimeError('independent_native_confirmation_process_failed')
    except subprocess.TimeoutExpired:
        write(output/'SUPERVISOR.json',{'status':'timeout','wall_seconds':time.monotonic()-start,
            'child_timeout_seconds':PROBE_TIMEOUT_SECONDS,'parent_pid':os.getpid()})
        raise RuntimeError('independent_native_confirmation_timeout')
    value=read(output/'result.json')
    if (value['job']!=binding(output/'job.json') or value['input']!=job['input']
            or value['input']!=binding(path) or value['parent_pid']!=os.getpid()
            or value['worker_pid']==os.getpid()):
        raise ValueError('independent_native_confirmation_binding_failed')
    resolve(value['native_arrays'])
    return value


def stable_cook(mesh, directory, number):
    """Drop-in `(binding, passes)` for metric_native_v3._stable_cook.

    Every pass describes a file actually loaded by a fresh native process.
    The final pass's vertex_error_m is therefore the original caller's replay
    criterion on the returned file. No export is performed after a pass.
    """
    started=time.monotonic();directory=Path(directory)
    if type(number)is not int or number<0:raise ValueError('nonnegative_part_number_required')
    pointset_error(mesh.vertices,mesh.vertices)
    directory.mkdir(parents=True,exist_ok=True)
    folder=directory/f'hull_{number}_confirmation_v2'
    path=directory/f'hull_{number}_source.obj'
    if folder.exists() or path.exists() or any(directory.glob(f'hull_{number}_cooked_*.obj')):
        raise ValueError('preserve_existing_cooked_part_no_overwrite')
    mesh.export(path)
    write(folder/'PROTOCOL.json',{'schema':VERSION,'implementation':binding(__file__),
        'source':binding(path),'part_number':number,'maximum_conversions':MAX_CONVERSIONS,
        'maximum_fresh_process_probes':MAX_CONVERSIONS+1,
        'convergence_threshold_m':CONVERGENCE_M,'runtime_threshold_m':RUNTIME_REPLAY_M,
        'acceptance_metric':'symmetric_nearest_vertex_distance','scene_created':False,
        'dynamic_steps':0,'GPU_jobs':0,'API_calls':0})
    passes=[]
    try:
        for conversions in range(MAX_CONVERSIONS+1):
            probe=probe_file(path,folder/f'probe_{conversions}')
            record={'input':binding(path),'output':binding(path),
                'vertex_error_m':probe['vertex_error_m'],'phase':'exact_return_file_confirmation',
                'confirmation':binding(folder/f'probe_{conversions}/result.json'),
                'conversion_count_before_probe':conversions,
                'returned_file_exported_after_check':False}
            passes.append(record)
            if probe['convergence_passed'] and probe['runtime_replay_passed']:
                write(folder/'RESULT.json',{'schema':VERSION,'status':'confirmed_native_fixed_file',
                    'returned_file':binding(path),'conversions':conversions,'fresh_process_probes':len(passes),
                    'passes':passes,'wall_seconds':time.monotonic()-started,'dynamic_steps':0,'GPU_jobs':0,'API_calls':0})
                return binding(path),passes
            if conversions==MAX_CONVERSIONS:break
            with np.load(resolve(probe['native_arrays']),allow_pickle=False)as arrays:
                cooked=trimesh.Trimesh(vertices=arrays['vertices'],faces=arrays['faces'],process=False)
            following=directory/f'hull_{number}_cooked_{conversions}.obj'
            cooked.export(following)
            record.update(output=binding(following),phase='nonconverged_file_native_export')
            path=following
        write(folder/'RESULT.json',{'schema':VERSION,'status':'native_file_not_confirmed_within_budget',
            'conversions':MAX_CONVERSIONS,'fresh_process_probes':len(passes),'passes':passes,
            'wall_seconds':time.monotonic()-started,'returned_file':None,'dynamic_steps':0,'GPU_jobs':0,'API_calls':0})
        raise ValueError('PhysX returned file failed confirmation within 3 conversions')
    except Exception as error:
        if not (folder/'RESULT.json').exists():
            write(folder/'ERROR.json',{'schema':VERSION,'status':'unknown_system_failure',
                'error_type':type(error).__name__,'message':str(error),'passes':passes,
                'wall_seconds':time.monotonic()-started,'returned_file':None})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe',type=Path,required=True)
    args=parser.parse_args();probe_worker(args.probe)
