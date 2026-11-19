#!/usr/bin/env python3
"""Standalone entry point for the retained Real→Sim→synthetic-human pipeline."""
import argparse
from collections import Counter
import fcntl
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from resimpair_runtime import ROOT, RUNS, REGISTRY, configure
configure()
PACKAGE = 'corl2026_workshop.experiments.autonomous_pipeline_v4'


def module(name):
    return importlib.import_module(PACKAGE + '.' + name)


def doctor(gpu=None):
    from resimpair_runtime import prepare_asset_catalog
    prepare_asset_catalog()
    core = module('core'); activation = core.activate()
    dependencies = {}
    for name in ['numpy', 'scipy', 'torch', 'sapien', 'gymnasium', 'mplib', 'mani_skill',
                 'cv2', 'av', 'PIL', 'jsonschema', 'trimesh']:
        m = importlib.import_module(name)
        dependencies[name] = {'file': str(Path(m.__file__).resolve()),
                              'version': str(getattr(m, '__version__', 'unknown'))}
    tools = {name: shutil.which(name) for name in ['ffmpeg', 'ffprobe', 'nvidia-smi']}
    core.require(all(tools.values()), 'ffmpeg_ffprobe_and_nvidia_smi_required')
    config = json.loads((ROOT/'configs/providers.json').read_text())
    transport = module('review_transport').generation_config(config, {'type':'object',
        'properties': {'ok': {'type':'boolean'}}, 'required':['ok']},
        wire_schema_adapter=config['gemini'].get('wire_schema_adapter'))
    credential = ROOT/config['credentials_file']
    return {'status':'ready', 'repository':str(ROOT), 'activation':activation,
            'dependencies':dependencies, 'system_tools':tools,
            'credentials_file_present':credential.is_file(), 'credential_values_printed':False,
            'provider_schema_evidence_verified':bool(transport),
            'gpu':module('resources').gpu_snapshot(gpu) if gpu is not None else None,
            'api_calls':0, 'simulation_started':False}


def historical_lineage():
    """Carry old attempts forward; every local campaign is included, even failed ones."""
    c = module('core'); b = module('budget')
    original = ROOT/'provenance/lineage_snapshot.json'
    value = c.read(original)
    total = dict(value['lineage_started_calls'])
    evidence = [c.bind(original)]
    # A fresh clone must retain the paid validation done while assembling this
    # repository. On this host those same markers still live in runtime ledgers;
    # account for them exactly once and verify their original bytes.
    carried = {}
    portable_snapshot = ROOT/'provenance/portable_validation_lineage.json'
    if portable_snapshot.exists():
        saved = c.read(portable_snapshot)
        c.require(saved['base_snapshot_sha256'] == c.bind(original)['sha256']
                  and saved['unresolved_submissions'] == 0, 'portable_lineage_requires_recovery')
        observed = Counter()
        for campaign in saved['campaigns']:
            for marker in campaign['markers']:
                key = (campaign['campaign_id'], marker['stage'])
                c.require(key not in carried and marker['resolved'] is True
                          and marker['kind'] in total, 'portable_lineage_ambiguous')
                carried[key] = marker
                observed[marker['kind']] += 1
        c.require(set(saved['additional_started_calls']) == set(total)
                  and all(observed[k] == saved['additional_started_calls'][k] for k in total),
                  'portable_lineage_count_mismatch')
        total = {k: total[k]+observed[k] for k in total}
        evidence.append(c.bind(portable_snapshot))
    for p in sorted(REGISTRY.glob('*.json')):
        spec = c.read(p); ledger = Path(spec['ledger_path'])
        c.require((ledger/'budget.json').is_file(), 'registered_ledger_missing')
        c.require(c.read(ledger/'budget.json') == spec, 'registered_budget_changed')
        for marker in sorted((ledger/'stages').glob('*/call_started.json')):
            started = c.read(marker); stage=marker.parent
            prior = carried.get((spec.get('campaign_id', p.stem), stage.name))
            if prior:
                c.require(c.bind(marker)['sha256'] == prior['sha256']
                          and started['kind'] == prior['kind'], 'carried_call_marker_changed')
            else:
                total[started['kind']] += 1
            failure = c.read(stage/'failure.json') if (stage/'failure.json').exists() else {}
            if (stage/'result.json').exists():
                if started['kind']=='video':
                    terminal=c.read(stage/'video_terminal.json') if (stage/'video_terminal.json').exists() else {}
                    c.require(terminal.get('status') in {'succeeded','failed','expired','cancelled'},
                              'prior_video_pending_recover_before_new_campaign')
            else:
                c.require(failure.get('provider_outcome')=='definite_failure'
                          or module('billing').validate_retired_billed_review(stage),
                          'prior_submission_unknown_recover_before_new_campaign')
            evidence.append(c.bind(marker))
    return total, evidence


def initialize(args):
    from resimpair_runtime import prepare_asset_catalog
    prepare_asset_catalog()
    c = module('core'); c.activate(); name = c.identifier(args.name)
    root = RUNS/name
    c.require(not root.exists(), 'campaign_exists_resume_do_not_reset')
    c.require(1 <= args.layouts <= 5 and 1 <= args.tasks_per_scene <= 4, 'invalid_cohort_size')
    kinds=module('batch_sampling').KINDS+module('contact_extensions').KINDS
    selected=args.kinds or list(kinds)
    c.require(selected and len(selected)==len(set(selected)) and set(selected)<=set(kinds), 'invalid_mechanisms')
    limits={'image':args.image_limit,'video':args.video_limit,'review':args.review_limit}
    c.require(all(type(v) is int and 0 <= v <= 100 for v in limits.values()), 'finite_call_limits_required')
    c.require(0 < args.usd_limit <= 60, 'finite_nominal_credit_limit_required')
    gpu=module('resources').gpu_snapshot(args.gpu)
    c.require(shutil.disk_usage(ROOT).free > 10*2**30, 'need_10GiB_free')
    (ROOT/'runtime').mkdir(exist_ok=True)
    with (ROOT/'runtime/.new_campaign.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lineage,evidence=historical_lineage()
        registration={'schema':'resimpair_portable_campaign_v1','name':name,
            'seed':args.seed,'mechanisms':selected,'layouts':args.layouts,'tasks_per_scene':args.tasks_per_scene,
            'new_call_limits':limits,'nominal_usd_admission_limit':args.usd_limit,
            'observed_quota_stop_limit':int(args.usd_limit*500000),
            'gpu_index':args.gpu,'gpu_uuid':gpu['uuid'],'api_concurrency':1,'automatic_retry':False}
        c.write(root/'portable_registration.json',registration)
        budget={'schema':'resimpair_cumulative_budget_v4','campaign_id':name,
            'ledger_path':str(root/'campaign_ledger'),'portable_campaign':True,
            'portable_registration':c.bind(root/'portable_registration.json'),
            'new_call_limits':limits,'ancestor_started_calls':lineage,'ancestor_evidence':evidence,
            'lineage_cumulative_limits':{k:lineage[k]+limits[k] for k in limits},
            'max_concurrent':1,'timeout_seconds':600,'automatic_retry':False,
            'currency_budget':None,'observed_quota_stop_limit':registration['observed_quota_stop_limit'],
            'nominal_usd_admission_limit':args.usd_limit,
            'reserve_quota_before_new_call':{'image':500000,'video':8000000,'review':500000},
            'authorization':'Standing user data-generation authorization; preserve all prior failed/unknown calls.'}
        c.write(root/'budget.json',budget)
        c.write(root/'config.json',c.read(ROOT/'configs/providers.json'))
        from corl2026_workshop.experiments.autonomous_pipeline_v4.batch_campaign import register
        bs=module('batch_sampling');ce=module('contact_extensions')
        groups=[];contacts=[];smoke=[]
        caps=Counter({'base':0,'qualified_base_trajectory_variants':0,'development_and_new_contacts':0})
        for ki,kind in enumerate(selected):
            for li in range(args.layouts):
                seed=args.seed+ki*100+li
                if kind in bs.KINDS:
                    scene=bs.sample_scene(kind,seed,li)
                    tasks=[bs.vary_task(t,0) for t in bs.enumerate_tasks(scene)][:args.tasks_per_scene]
                    phase='base'
                else:
                    scene=ce.scene(kind,seed)
                    scene.update(scene_instance_id=f'{kind}_L{li}',split_group=f'{kind}_L{li}',
                                 partition='development' if li==0 else 'heldout_layout')
                    tasks=[ce.task(scene,i) for i in range(min(3,args.tasks_per_scene))]
                    for i,t in enumerate(tasks):t['id']=scene['scene_instance_id']+'__t'+str(i)
                    phase='contact'
                members=[register(root,scene,t,phase,kind) for t in tasks]
                caps['base' if phase=='base' else 'development_and_new_contacts']+=len(members)
                group={'kind':kind,'layout_index':li,'cases':members,
                       'split_group':scene['scene_instance_id'],'partition':scene['partition']}
                (groups if phase=='base' else contacts).append(group)
                # Two tasks in the first scene establish shared-initial-state capability.
                if li==0:smoke.extend(r['id'] for r in members[:2])
        caps['qualified_base_trajectory_variants']=0
        c.write(root/'experiment.json',{'schema':'resimpair_parameterized_batch_v1','sources':[],'cases':[],
            'simulation_limit':sum(caps.values()),'physical_caps':dict(caps),'simulation_timeout_seconds':240,
            'gpu':args.gpu,'gpu_uuid':gpu['uuid'],'protected_gpus':[],'no_training':True,'automatic_retry':False,
            'source_frame_budget':24,'review_frame_budget':49,'new_api_limits':limits,
            'scope':'Retained finite mechanism templates; no claim of arbitrary real-video adaptation.',
            'human_queue_separate':True,'source_independent_holdout_claimed':False})
        c.write(root/'cohort.json',{'groups':groups,'contact_groups':contacts,'smoke_case_ids':smoke,
            'base_cases':caps['base'],'contact_cases':caps['development_and_new_contacts'],
            'same_initial_scene_reset_per_task':True,'failed_inputs_remain_in_denominator':True,
            'human_generation_is_separate':True,'source_coverage':'missing','formal_pair_count':0})
        c.write(root/'human_runtime.json',{'schema':'resimpair_human_runtime_v1','budget':c.bind(root/'budget.json'),
            'media_upload_limit':limits['video'],'provider_application':c.bind(ROOT/'provenance/provider_app.js'),
            'per_video_poll_deadline_seconds':900,'automatic_retry':False})
        module('budget').Budget(root/'budget.json')
        auditor=importlib.import_module('corl2026_workshop.tools.audit_batch_generation_20261025')
        c.write(root/'evaluator_bindings.json',{'files':auditor.evaluator_files(),
            'acceptance_rules_unchanged':True,'frozen_before_first_execution':True})
        module('experiment').freeze(root,'portable_r1')
    return {'root':str(root),'cases':sum(caps.values()),'smoke_case_ids':smoke,
            'new_call_limits':limits,'lineage_started_calls':lineage,'generation_started':False}


def run_frozen(root, name, arguments):
    root=Path(root).resolve(); release=root/'releases/portable_r1'
    env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(release),str(ROOT),str(ROOT/'PolicyGen')]),
             PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    subprocess.run([sys.executable,'-B','-u','-m',name,*map(str,arguments)],cwd=ROOT,env=env,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    d=sub.add_parser('doctor');d.add_argument('--gpu',type=int)
    g=sub.add_parser('grasp',help='Opt-in asset preparation, cached grasps and independently audited mesh campaigns')
    g.add_argument('arguments',nargs=argparse.REMAINDER)
    for name in ['assets','tasks','tabletop','pickplace']:
        q=sub.add_parser(name,help='Full-library assets or reusable task composition')
        q.add_argument('arguments',nargs=argparse.REMAINDER)
    q=sub.add_parser('random-pickplace',help='Bounded agent-free CPU batches from frozen expanded asset factories')
    q.add_argument('arguments',nargs=argparse.REMAINDER)
    i=sub.add_parser('init');i.add_argument('--name',required=True);i.add_argument('--gpu',type=int,default=0)
    i.add_argument('--seed',type=int,default=2026102500);i.add_argument('--kinds',nargs='+')
    i.add_argument('--layouts',type=int,default=1);i.add_argument('--tasks-per-scene',type=int,default=2)
    for kind,default in [('image',1),('video',2),('review',6)]:i.add_argument('--'+kind+'-limit',type=int,default=default)
    i.add_argument('--usd-limit',type=float,default=20)
    for name in ['native','audit','export','human-prepare','human','status','pipeline']:
        q=sub.add_parser(name);q.add_argument('--root',type=Path,required=True)
        if name=='native':
            q.add_argument('--case-id',action='append',default=[])
            q.add_argument('--phase',default='smoke')
            q.add_argument('--audit',type=Path)
        if name=='audit':q.add_argument('--output',type=Path)
        if name in ['export','human-prepare']:q.add_argument('--audit',type=Path,required=True)
        if name=='export':q.add_argument('--output',type=Path,required=True)
        if name=='human-prepare':q.add_argument('--max-scenes',type=int,default=1)
        if name=='human':q.add_argument('--phase',choices=['qualification','expansion'],default='qualification')
        if name=='pipeline':q.add_argument('arguments',nargs=argparse.REMAINDER)
    a=p.parse_args();result=None
    if a.command=='grasp':
        subprocess.run([sys.executable,'-m','resimpair_grasp',*a.arguments],cwd=ROOT,check=True)
        return
    if a.command in {'assets','tasks','tabletop','pickplace'}:
        subprocess.run([sys.executable,'-m','resimpair_'+a.command,*a.arguments],cwd=ROOT,check=True)
        return
    if a.command=='random-pickplace':
        subprocess.run([sys.executable,'-m','resimpair_tasks.random_pickplace_v1',*a.arguments],cwd=ROOT,check=True)
        return
    if a.command=='doctor':result=doctor(a.gpu)
    elif a.command=='init':result=initialize(a)
    elif a.command=='native':
        args=['run','--root',a.root,'--phase',a.phase]
        if a.audit:args+=['--audit',a.audit]
        for cid in a.case_id:args+=['--case-id',cid]
        run_frozen(a.root,'autonomous_pipeline_v4.batch_campaign',args)
    elif a.command=='audit':
        output=a.output or a.root/'audits/native.json'
        run_frozen(a.root,'corl2026_workshop.tools.audit_batch_generation_20261025',
            ['--root',a.root,'--output',output,'--evaluator-manifest',a.root/'evaluator_bindings.json'])
    elif a.command=='export':
        run_frozen(a.root,'autonomous_pipeline_v4.batch_dataset',['--root',a.root,'--audit',a.audit,'--output',a.output])
    elif a.command=='human-prepare':
        run_frozen(a.root,'autonomous_pipeline_v4.batch_human_queue',
            ['prepare','--root',a.root,'--audit',a.audit,'--max-scenes',a.max_scenes])
    elif a.command=='human':
        run_frozen(a.root,'autonomous_pipeline_v4.batch_human_queue',['run','--root',a.root,'--phase',a.phase])
    elif a.command=='status':result=module('budget').Budget(a.root/'budget.json').usage()
    elif a.command=='pipeline':
        extra=a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments
        run_frozen(a.root,'autonomous_pipeline_v4',['--root',a.root,*extra])
    if result is not None:print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
