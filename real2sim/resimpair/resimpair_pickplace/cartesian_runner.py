"""Explicit Cartesian lift candidate runner; pending physical qualification."""
import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from resimpair_grasp.common import ROOT,binding,code_bindings,read,resolve,write
from corl2026_workshop.experiments.autonomous_pipeline_v4 import core,resources


from .runner import verify as retained_verify,code,audit


def verify(root):
    spec=retained_verify(root)
    if spec.get('controller')!='cartesian_lift_v1':
        raise ValueError('Candidate controller requires a separately registered campaign')
    for name in ['cartesian_runner.py','cartesian_motion.py','cartesian_lift.py']:
        key='resimpair_pickplace/'+name
        if spec['code'].get(key)!=binding(ROOT/key):
            raise ValueError('Candidate controller is not bound in the campaign')
    return spec

def run(a):
    root=a.root.resolve();spec=verify(root);cases=read(root/'cohort.json')['cases'];results=[]
    if a.case:cases=[c for c in cases if c['id'] in a.case]
    if not 1<=a.limit<=24:raise ValueError('Finite launch count')
    for case in cases:
        if len(results)>=a.limit:break
        directory=root/'executions'/case['execution_id']
        if (directory/'terminal.json').exists():continue
        if not case['eligible']:
            terminal={'case':case['id'],'physical_attempts':0,'returncode':'preflight_rejected',
                'reason':case.get('preflight_reason','no_single_arm_source_and_destination_grasp'),'api_calls':0,'elapsed_seconds':0}
            write(directory/'terminal.json',terminal);results.append(terminal);continue
        with contextlib.ExitStack() as stack:
            for path in resources.lock_paths(spec['gpu']):
                f=stack.enter_context(path.open('r'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            snapshot=resources.gpu_snapshot(spec['gpu']);resources.check_idle(snapshot)
            if snapshot['uuid']!=spec['gpu_uuid']:raise ValueError('GPU changed')
            if (directory/'reservation.json').exists():raise ValueError('Unknown old execution; recover without resubmission')
            budget=Path(spec['budget']);limits=read(budget/'limits.json')
            reservations=list((budget/'reservations').glob('*.json'))
            if len(reservations)>=limits['physical_attempt_limit']:raise ValueError('Shared physical budget exhausted')
            if shutil.disk_usage(root).free<20*2**30:raise ValueError('Insufficient space')
            reservation={'case':case,'campaign':str(root),'started_unix':time.time(),'gpu':snapshot}
            write(budget/'reservations'/f'{len(reservations):04d}.json',reservation)
            write(directory/'reservation.json',reservation)
            env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                     RESIMPAIR_PICKPLACE_RESERVATION=str(directory/'reservation.json'))
            env.pop('CUDA_VISIBLE_DEVICES',None)
            start=time.monotonic()
            with (directory/'worker.log').open('x') as log:
                process=subprocess.Popen([sys.executable,'-u','-m','resimpair_pickplace.cartesian_runner','worker','--root',str(root),'--case',case['id']],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                write(directory/'process.json',{'pid':process.pid,'started_unix':time.time()})
                try:returncode=process.wait(timeout=spec['timeout_seconds'])
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=8)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                    returncode='timeout'
            terminal={'case':case['id'],'returncode':returncode,'elapsed_seconds':time.monotonic()-start,'physical_attempts':1,'api_calls':0}
            write(directory/'terminal.json',terminal);results.append(terminal);print(json.dumps(terminal),flush=True)
    return {'rows':results}


def worker(a):
    from . import environment
    from resimpair_grasp.runner import native_geometry
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import simulation,planning_protocols
    from corl2026_workshop.experiments.autonomous_pipeline_v4.timing import Accumulator,span
    from examples.baselines.hypericl_task_expansion.common import bind_gpu
    root=a.root.resolve();spec=verify(root);case=next(c for c in read(root/'cohort.json')['cases'] if c['id']==a.case[0])
    directory=root/'executions'/case['id']
    if os.environ.get('RESIMPAIR_PICKPLACE_RESERVATION')!=str(directory/'reservation.json'):raise ValueError('Supervisor required')
    gpu=bind_gpu(spec['gpu'])
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    import gymnasium as gym
    scene=core.read(core.checked(case['scene']));task=core.read(core.checked(case['task']))
    ids=[o['id'] for o in scene['objects']];supports=ids+[s['id'] for s in scene['supports']]+['table']
    write(directory/'event_protocol.json',planning_protocols.pick_place(scene,task,ids,supports,simulation.targets(scene,task)))
    write(directory/'planning_declaration.json',{'event_protocol':core.bind(directory/'event_protocol.json'),
        'case':case,'scope':'before_physical_execution','declared_unix':time.time(),'implementation':spec['code']})
    performance=Accumulator();env=rec=None;failure=bundle=None
    try:
        with span(directory,'simulation_environment_setup'):
            environment.register(scene,performance)
            env=gym.make(environment.ENV_ID,sim_backend='physx_cpu',render_backend=gpu['render_backend'],control_mode='pd_joint_pos',obs_mode='rgb',reward_mode='none',render_mode='rgb_array',sensor_configs={'shader_pack':os.environ.get('RESIMPAIR_SHADER','rt-fast')})
        with span(directory,'simulation_reset_and_recorders'):
            from .seed_compatibility import seed_receipt
            reset_seed=seed_receipt(scene['seed']);write(directory/'simulation_seed.json',reset_seed)
            performance.phase='reset';rec=simulation.Recorder(env,directory);rec.reset(reset_seed['simulator_reset_seed'])
            native_geometry(rec,directory);environment.native_evidence(rec,directory)
        with span(directory,'planning_execution_and_rendering'):
            from resimpair_pickplace.cartesian_motion import solve
            performance.phase='execution';solve(rec,task,'cached_antipodal')
    except Exception as exc:
        import traceback
        failure={'type':type(exc).__name__,'message':str(exc),'trace':traceback.format_exc()};write(directory/'planning_failure.json',failure)
    finally:
        if rec is not None:
            with span(directory,'simulation_finalize'):
                performance.phase='finalize';rec.finish();bundle=simulation.export(rec,root,case,task,scene,directory,failure)
        if env is not None:env.close()
        performance.persist(directory/'timing_breakdown.json')
    result={'bundle':bundle,'failure':failure,'api_calls':0};write(directory/'worker_result.json',result)
    return {'bundle':bundle,'failure':failure['message'] if failure else None}



if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['run','worker','audit']);p.add_argument('--root',type=Path,required=True);p.add_argument('--case',action='append',default=[]);p.add_argument('--limit',type=int,default=1)
    a=p.parse_args();print(json.dumps(globals()[a.command](a),indent=2))
