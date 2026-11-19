"""Finite shared-skill campaigns; endpoint rejections and failed runs persist."""
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


def code():
    values=code_bindings()
    for package in ['resimpair_pickplace','resimpair_tabletop']:
        values.update({str(p.relative_to(ROOT)):binding(p) for p in (ROOT/package).glob('*.py')})
    return values


def init(a):
    from . import evaluator
    from .scenes import sample,container_choices
    from .reachability import bind_selection
    if a.root.exists():raise ValueError('Campaign exists; preserve it')
    cfg=read(a.config);snapshot=resources.gpu_snapshot(cfg['gpu'])
    budget=a.budget.resolve()
    if not (budget/'limits.json').exists():
        write(budget/'limits.json',{'physical_attempt_limit':cfg['physical_attempt_limit'],
            'api_limit':0,'source_config':binding(a.config),'created_unix':time.time()})
    a.root.mkdir(parents=True)
    write(a.root/'experiment.json',{'schema':'general_pickplace_experiment_v1','config':binding(a.config),
        'budget':str(budget),'gpu':cfg['gpu'],'gpu_uuid':snapshot['uuid'],
        'timeout_seconds':cfg['per_attempt_timeout_seconds'],'code':code(),'legacy_code':core.implementation(),
        'asset_index':binding(a.index),'containers':binding(a.containers),'clutter':binding(a.clutter),
        'seed':a.seed,'api_limit':0,'baseline_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
    write(a.root/'rules.json',{'version':evaluator.VERSION,'files':evaluator.bindings(),
                              'frozen_before_attempts':True,'retained_rules_unchanged':True})
    entries=read(a.index)['entries']
    if a.asset:entries=[e for e in entries if read(resolve(e['asset']))['source_id'] in a.asset]
    cases=[];rejections=[]
    for ai,entry in enumerate(entries):
        d=read(resolve(entry['asset']))
        choices=container_choices(a.containers,d['partition'],d['half_size'])
        if a.container:choices=[pair for pair in choices if pair[1]['id'] in a.container]
        if not choices:
            rejections.append({'source_id':d['source_id'],'reason':'no_size_and_partition_compatible_container'});continue
        # Different task destinations share one initial scene. The choices are
        # frozen before physical outcomes; no replacement for failed attempts.
        choices.sort(key=lambda x:x[1]['id'])
        for li in range(a.layouts):
            chosen=[choices[(ai+li)%len(choices)]]
            if len(choices)>1:chosen.append(choices[(ai+li+1)%len(choices)])
            seed=a.seed+ai*100+li
            try:scene,tasks=sample(entry,chosen,a.clutter,seed,li)
            except ValueError as exc:
                rejections.append({'source_id':d['source_id'],'layout':li,'reason':str(exc)});continue
            scene_path=a.root/'scenes'/(scene['id']+'.json');write(scene_path,scene)
            for task in tasks:
                folder=a.root/'inputs'/task['id'];folder.mkdir(parents=True)
                write(folder/'role_intent.json',task)
                selected,planned=bind_selection(scene,task,folder)
                case={'id':task['id'],'execution_id':task['id'],'asset':entry['asset'],
                    'layout':li,'partition':d['partition'],'source_id':d['source_id'],
                    'split_group':d['source_id'],'shared_scene':binding(scene_path),
                    'method':'cached_antipodal','source_coverage':'synthetic_only',
                    'arm_selection':binding(folder/'arm_selection.json'),'eligible':selected is not None}
                if selected is not None:
                    write(folder/'scene.json',selected);write(folder/'task.json',planned)
                    case.update(scene=core.bind(folder/'scene.json'),task=core.bind(folder/'task.json'))
                cases.append(case)
                print(json.dumps({'case':task['id'],'eligible':case['eligible']},ensure_ascii=False),flush=True)
    write(a.root/'cohort.json',{'cases':cases,'scene_rejections':rejections,'failures_retained':True,
        'no_physical_outcomes_used_for_registration':True,'coverage':'one movable, multiple destination tasks per scene'})
    return {'root':str(a.root),'registered':len(cases),'eligible':sum(c['eligible'] for c in cases),'scene_rejections':len(rejections)}


def verify(root):
    spec=read(root/'experiment.json')
    for b in spec['code'].values():resolve(b)
    for b in spec['legacy_code'].values():core.checked(b)
    for b in read(root/'rules.json')['files'].values():resolve(b)
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
                'reason':'no_single_arm_source_and_destination_grasp','api_calls':0,'elapsed_seconds':0}
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
                process=subprocess.Popen([sys.executable,'-u','-m','resimpair_pickplace.runner','worker','--root',str(root),'--case',case['id']],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
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
            from resimpair_grasp.motion import solve
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


def audit(a):
    from .evaluator import evaluate
    root=a.root.resolve();verify(root);rows=[]
    for c in read(root/'cohort.json')['cases']:
        directory=root/'executions'/c['id'];terminal=directory/'terminal.json'
        if not terminal.exists():continue
        t=read(terminal);r=evaluate(directory) if t['physical_attempts'] else {'passed':False,'errors':[t['reason']]}
        rows.append({'case':c['id'],'partition':c['partition'],'passed':bool(r['passed']),
            'physical_attempts':t['physical_attempts'],'elapsed_seconds':t['elapsed_seconds'],
            'errors':r['errors'],'audit':binding(directory/'container_audit.json') if t['physical_attempts'] else None})
    result={'rows':rows,'attempts':len(rows),'physical_executions':sum(r['physical_attempts'] for r in rows),
            'passed':sum(r['passed'] for r in rows),'api_calls':0,'heldout_not_used_for_tuning':True}
    from resimpair_grasp.common import digest
    output=root/'audits'/(digest(result)[:16]+'.json');write(output,result)
    return {'report':str(output),**result}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('init');q.add_argument('--root',type=Path,required=True);q.add_argument('--index',type=Path,required=True)
    q.add_argument('--containers',type=Path,required=True);q.add_argument('--clutter',type=Path,default=ROOT/'assets/expanded_library/scene_catalog_v3.json')
    q.add_argument('--config',type=Path,default=ROOT/'configs/general_pickplace_v1.json')
    q.add_argument('--budget',type=Path,default=ROOT/'runs/general_pickplace_budget_20261026')
    q.add_argument('--seed',type=int,default=2026102677);q.add_argument('--layouts',type=int,default=1)
    q.add_argument('--asset',action='append',default=[]);q.add_argument('--container',action='append',default=[])
    for name in ['run','worker','audit']:
        q=s.add_parser(name);q.add_argument('--root',type=Path,required=True)
        if name in {'run','worker'}:q.add_argument('--case',action='append',default=[])
        if name=='run':q.add_argument('--limit',type=int,default=3)
    a=p.parse_args();print(json.dumps(globals()[a.command](a),ensure_ascii=False,indent=2))
