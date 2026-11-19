"""One scene, several declared programs; existing bounded runner executes them."""
from pathlib import Path
from collections import Counter
import json
import shutil
import subprocess
import numpy as np

from resimpair_grasp.common import ROOT,binding,code_bindings,read,resolve,write
from resimpair_grasp import evaluator,runner
from corl2026_workshop.experiments.autonomous_pipeline_v4 import core,resources
from . import programs


def entries(indexes,partition,categories=()):
    candidates=[];rejections=[];seen=set()
    for path in indexes:
        for entry in read(path)['entries']:
            d=read(resolve(entry['asset']))
            if d['partition']!=partition or d['source_id'] in seen:continue
            if d.get('semantic_role_eligible') is False:
                rejections.append({'source_id':d['source_id'],'reason':'ambiguous_asset_semantics'});continue
            if categories and d['category'] not in categories:continue
            seen.add(d['source_id'])
            from resimpair_grasp.grasps import load
            cache=load(entry['cache'],entry['asset'])
            if not d['roles']['rigid_manipulation_candidate'] or not cache['candidates']:
                rejections.append({'source_id':d['source_id'],'reason':'no_qualified_geometry_or_grasp_proposals'});continue
            candidates.append(entry)
    return candidates,rejections


def initialize(a):
    if a.root.exists():raise ValueError('Campaign exists; new version required')
    if not 1<=a.scenes<=50 or not 1<=a.physical_limit<=300:raise ValueError('Finite scene/physical budget required')
    if set(a.families)-programs.FAMILIES:raise ValueError('Unsupported task composition')
    recipe=read(a.recipe) if a.recipe else None
    if recipe:
        if recipe.get('schema')!='resimpair_task_recipe_v1' or len(recipe.get('role_queries',[]))!=2:
            raise ValueError('Expected two explicit role queries')
        if any(not programs.CAPABILITIES.get(q.get('required_skill'),{}).get('supported') for q in recipe['role_queries']):
            raise ValueError('Recipe requests unsupported contact mechanism')
        a.partition=recipe['partition'];a.families=recipe['families']
        if set(a.families)-programs.FAMILIES:raise ValueError('Recipe requests unregistered composition')
    pool,rejected=entries(a.cache_index,a.partition,a.category)
    if len(pool)<2:raise ValueError('Need two eligible assets in the requested partition')
    snapshot=resources.gpu_snapshot(a.gpu)
    if shutil.disk_usage(ROOT).free<10*2**30:raise ValueError('Insufficient disk space')
    a.root.mkdir(parents=True);inputs=a.root/'inputs';inputs.mkdir()
    code=code_bindings()
    code.update({str(p.relative_to(ROOT)):binding(p) for p in Path(__file__).parent.glob('*.py')})
    for name,b in code.items():
        target=a.root/'release'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(resolve(b),target)
    write(a.root/'rules.json',{'version':evaluator.VERSION,'files':evaluator.bindings(),'frozen_before_attempts':True})
    write(a.root/'experiment.json',{'schema':'resimpair_generic_grasp_campaign_v1','gpu':a.gpu,'gpu_uuid':snapshot['uuid'],
        'physical_limit':a.physical_limit,'timeout_seconds':300,'max_concurrent':1,'api_limit':0,'api_cost_usd':0,
        'automatic_retry':False,'source_coverage':'synthetic_only','new_training_runs':0,'code':code,
        'legacy_code':core.implementation(),'baseline_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'initial_snapshot':snapshot,'seed':a.seed,'catalog':binding(a.catalog),
        'cache_indexes':[binding(p) for p in a.cache_index],'task_factory':'asset_roles_scene_programs_v1',
        'partition':a.partition,'families':a.families,'scene_limit':a.scenes,'recipe':binding(a.recipe) if a.recipe else None})
    rng=np.random.default_rng(a.seed);cases=[];scene_rejections=[];sources=[]
    for i in range(a.scenes):
        # Selection is deterministic and recorded before observing physics.
        if recipe:
            pair=[]
            aliases={'food_can':'can','food_box':'box','food_container':'jar','block':'toy'}
            for query in recipe['role_queries']:
                choices=[]
                for entry in pool:
                    d=read(resolve(entry['asset']));category=aliases.get(d['category'],d['category'])
                    if category in query['categories'] and entry not in pair:choices.append(entry)
                if not choices:raise ValueError('Missing qualified role candidates: '+query['role'])
                pair.append(choices[int(rng.integers(len(choices)))])
        else:
            indices=rng.choice(len(pool),2,replace=False);pair=[pool[int(x)] for x in indices]
        try:scene,tasks=programs.create_scene(a.catalog,pair,a.seed+i*101,i,inputs,a.families)
        except ValueError as e:
            scene_rejections.append({'scene_index':i,'assets':pair,'reason':str(e)});continue
        scene_path=inputs/(scene['id']+'.json');write(scene_path,scene)
        sources.append({'scene':binding(scene_path),'active_source_ids':scene['active_source_ids'],
                        'task_count':len(tasks),'split_group':scene['split_group'],'partition':scene['partition']})
        for task in tasks:
            task_path=inputs/(task['id']+'.json');write(task_path,task)
            cases.append({'id':task['id'],'execution_id':task['id']+'__cached_antipodal',
                'scene':core.bind(scene_path),'task':core.bind(task_path),'method':'cached_antipodal',
                'semantic_key':core.digest(task),'split_group':scene['split_group'],'partition':scene['partition'],
                'source_coverage':'synthetic_only','asset':pair[0]['asset'],'active_assets':[e['asset'] for e in pair],
                'layout':i,'composition_template':task['composition_template']})
    write(a.root/'cohort.json',{'cases':cases,'scenes':sources,'asset_preflight_rejections':rejected,
        'scene_preflight_rejections':scene_rejections,'failures_remain_in_denominator':True,
        'same_scene_across_tasks':True,'physical_budget_counts_failed_and_unknown':True})
    return {'root':str(a.root),'registered_scenes':len(sources),'registered_tasks':len(cases),
            'scene_preflight_rejections':scene_rejections,'asset_preflight_rejections':len(rejected),
            'contact_mechanisms':['pick_place'],'api_limit':0,'physical_limit':a.physical_limit}


def run(a):
    if not a.case:
        a.case=[c['execution_id'] for c in read(a.root/'cohort.json')['cases']]
    return runner.run(a)


def audit(a):
    result=runner.audit(a);cohort=read(a.root/'cohort.json');lookup={c['execution_id']:c for c in cohort['cases']}
    per_template={}
    for row in result['rows']:
        c=lookup[row['case']];kind=c['composition_template'];d=per_template.setdefault(kind,{'attempted':0,'passed':0})
        d['attempted']+=1;d['passed']+=int(row['passed'])
    result['task_factory_coverage']={'registered_scenes':len(cohort['scenes']),'registered_tasks':len(cohort['cases']),
        'scene_preflight_rejected':len(cohort['scene_preflight_rejections']),'by_composition':per_template,
        'contact_mechanisms':['pick_place'],'new_contact_mechanisms':0,
        'counting_rule':'semantic goals, physical executions, source assets and camera views are separate dimensions'}
    path=a.root/'audits'/('factory_'+core.digest(result)[:16]+'.json');write(path,result)
    return {'factory_summary':str(path),**result}
