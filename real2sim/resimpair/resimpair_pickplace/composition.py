"""Compose task relations, not motion scripts: one scene, several programs.

Each object is transferred at most once per program. Regrasp/handover remains
explicitly unsupported. The same retained controller executes every operation.
"""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation
from resimpair_grasp.common import ROOT,read,resolve,write,binding,digest
from resimpair_grasp.scenes import item
from .scenes import sample,container_choices,container_fits
from .reachability import bind_selection
from . import runner,evaluator
from corl2026_workshop.experiments.autonomous_pipeline_v4 import core,resources


def compose(scene,name,assignments,ordered=False):
    objects={o['id']:o for o in scene['objects']};supports={s['id']:s for s in scene['supports']}
    if not assignments or len(assignments)>6:raise ValueError('Finite 1..6 transfers required')
    if len(assignments)!=len(set(oid for oid,_ in assignments)):raise ValueError('Repeated object needs explicit regrasp/handover skill')
    operations=[]
    for oid,sid in assignments:
        if oid not in objects or objects[oid]['kind']!='movable':raise ValueError('Movable role not bound')
        if sid not in supports:raise ValueError('Container role not bound')
        d=read(resolve(supports[sid]['container_descriptor']))
        if not container_fits(d,objects[oid]['half_size']):raise ValueError('Payload does not fit qualified cavity')
        if sum(s==sid for _,s in assignments)>1:
            raise ValueError('Multiple payloads in one container require explicit capacity/packing offsets')
        c=np.r_[d['cavity']['center_xy_m'],0.]
        offset=(Rotation.from_euler('z',supports[sid]['yaw']).as_matrix()@c)[:2]
        operations.append({'id':'move_'+oid,'primitive':'pick_place','object':oid,'arm':None,
            'target':{'support':sid,'offset_xy':offset.tolist(),'yaw':objects[oid]['yaw']}})
    precedence=[(operations[i]['id'],operations[i+1]['id']) for i in range(len(operations)-1)] if ordered else []
    return {'schema':'resimpair_task_intent_v4','id':scene['id']+'_'+name,'family':'sort' if len(operations)>1 else 'relocate',
        'required_skills':['pick_place'],'object_ids':[o['object'] for o in operations],
        'goal':name.replace('_',' '),'roles':{'assignments':[list(a) for a in assignments]},'operations':operations,
        'required_operation_precedence':precedence,'equivalence_groups':[],
        'incidental_controller_order_is_required':False,'human_hand_policy':'preserve_robot_body_side',
        'non_target_policy':'preserve_identity_and_pose','release_precondition_control':True,
        'scope':'combinations of one rigid pick/place skill; object names do not grant other contact mechanisms'}


def pair_scene(index,containers,clutter,seed,container_ids):
    entries=read(index)['entries']
    if len(entries)!=2:raise ValueError('Pair demonstration requires two entries; generic composer accepts 1..6 roles')
    descriptors=[read(resolve(e['asset'])) for e in entries]
    if len({d['source_id'] for d in descriptors})!=2 or len({d['partition'] for d in descriptors})!=1:
        raise ValueError('Distinct sources from the same partition required')
    half=np.max([d['half_size'] for d in descriptors],axis=0)
    choices=container_choices(containers,descriptors[0]['partition'],half)
    if container_ids:choices=[p for p in choices if p[1]['id'] in container_ids]
    choices.sort(key=lambda p:p[1]['id'])
    if len(choices)<2:raise ValueError('Two compatible containers required')
    scene,_=sample(entries[0],choices[:2],clutter,seed,0)
    scene['id']='pair_'+digest([d['source_id'] for d in descriptors]+[seed])[:16]
    scene['scene_instance_id']=scene['id'];scene['active_source_ids']=[d['source_id'] for d in descriptors]
    scene['split_group']='sources_'+digest(sorted(scene['active_source_ids']))[:16]
    first=scene['objects'][0];first['id']='object_0';scene['grasp_caches']={'object_0':entries[0]['cache']}
    radius=float(np.linalg.norm(descriptors[1]['half_size'][:2]));rng=np.random.default_rng(seed+811)
    others=[(np.asarray(o['position'][:2]),float(np.linalg.norm(o['half_size'][:2]))) for o in scene['objects']+scene['supports']]
    for _ in range(100):
        xy=np.array([rng.uniform(-.34,.16),rng.uniform(-.54,.54)])
        if all(np.linalg.norm(xy-p)>r+radius+.025 for p,r in others):break
    else:raise ValueError('No distinct second source position')
    scene['objects'].insert(1,item(entries[1]['asset'],descriptors[1],'object_1',xy,rng.uniform(-np.pi,np.pi),True))
    scene['grasp_caches']['object_1']=entries[1]['cache'];scene['randomization']['second_source_xy_m']=xy.tolist()
    tasks=[compose(scene,'select_0',[('object_0','container_0')]),
           compose(scene,'select_1',[('object_1','container_1')]),
           compose(scene,'sort_pair',[('object_0','container_0'),('object_1','container_1')]),
           compose(scene,'reverse_sort',[('object_0','container_1'),('object_1','container_0')]),
           compose(scene,'ordered_pair',[('object_1','container_1'),('object_0','container_0')],ordered=True)]
    return scene,tasks


def register(a):
    if a.root.exists():raise ValueError('Preserve existing campaign')
    cfg=read(ROOT/'configs/general_pickplace_v1.json');snapshot=resources.gpu_snapshot(cfg['gpu'])
    budget=ROOT/'runs/general_pickplace_budget_20261026'
    if not (budget/'limits.json').exists():raise ValueError('Initialize shared budget with runner first')
    scene,tasks=pair_scene(a.index,a.containers,a.clutter,a.seed,a.container)
    a.root.mkdir(parents=True)
    write(a.root/'experiment.json',{'schema':'general_pickplace_composition_v1','budget':str(budget),'gpu':cfg['gpu'],
        'gpu_uuid':snapshot['uuid'],'timeout_seconds':cfg['per_attempt_timeout_seconds'],'code':runner.code(),
        'legacy_code':core.implementation(),'asset_index':binding(a.index),'containers':binding(a.containers),
        'clutter':binding(a.clutter),'seed':a.seed,'api_limit':0})
    write(a.root/'rules.json',{'version':evaluator.VERSION,'files':evaluator.bindings(),'frozen_before_attempts':True,'retained_rules_unchanged':True})
    path=a.root/'scene.json';write(path,scene);cases=[]
    for task in tasks:
        folder=a.root/'inputs'/task['id'];write(folder/'role_intent.json',task)
        selected,planned=bind_selection(scene,task,folder)
        case={'id':task['id'],'execution_id':task['id'],'partition':scene['partition'],'split_group':scene['split_group'],
            'source_coverage':'synthetic_only','active_source_ids':scene['active_source_ids'],'shared_scene':binding(path),
            'arm_selection':binding(folder/'arm_selection.json'),'eligible':selected is not None,'method':'cached_antipodal'}
        if selected is not None:
            from resimpair_tasks.programs import validate_task
            validate_task(selected,planned)
            write(folder/'scene.json',selected);write(folder/'task.json',planned)
            case.update(scene=core.bind(folder/'scene.json'),task=core.bind(folder/'task.json'))
        cases.append(case);print(json.dumps({'case':case['id'],'eligible':case['eligible']}),flush=True)
    write(a.root/'cohort.json',{'cases':cases,'failures_retained':True,'shared_scene':binding(path),
        'independent_contact_mechanisms':1,'task_compositions':len(cases),'semantics_defined_before_execution':True})
    return {'registered':len(cases),'eligible':sum(c['eligible'] for c in cases),'root':str(a.root)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--index',type=Path,required=True);p.add_argument('--containers',type=Path,required=True)
    p.add_argument('--clutter',type=Path,default=ROOT/'assets/expanded_library/scene_catalog_v3.json')
    p.add_argument('--seed',type=int,default=2026102690);p.add_argument('--container',action='append',default=[])
    a=p.parse_args();print(json.dumps(register(a),indent=2))
