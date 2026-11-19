#!/usr/bin/env python3
"""Register four finite follow-up programs; does not execute them."""
from pathlib import Path
import copy
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from resimpair_grasp.common import ROOT,read,resolve,write,binding,digest
from resimpair_grasp.scenes import item
from resimpair_pickplace.cudex_benchmark import SELECTION,BASE,scene_for
from resimpair_pickplace.scenes import container_choices
from resimpair_pickplace.packing import compose_packing
from resimpair_pickplace.cartesian_reachability import bind_selection
from resimpair_pickplace import runner,evaluator
from resimpair_tasks.programs import validate_task
from corl2026_workshop.experiments.autonomous_pipeline_v4 import core,resources


def register():
    output=BASE/'packing_physics'
    if output.exists():raise ValueError('Preserve existing pilot')
    entries=read(SELECTION)['entries'];groups=[('development',[0,1]),('heldout_layouts',[2,3])]
    ok={r['descriptor']['sha256'] for r in read(ROOT/'runs/support_surface_audit_20261026_v1.json')['rows'] if r['passed']}
    write(ROOT/'validation/cudex_physical_20261026/packing_protocol.json',{
        'schema':'two_scene_multi_payload_followup_v1','groups':groups,'programs':['pack_0_then_1','pack_1_then_0'],
        'physical_limit':4,'same_shared_budget':binding(ROOT/'runs/cudex_physical_budget_20261026/limits.json'),
        'decision':'Paired grasp/place gave geometry 7/8 and native cuDex proposals 8/8; test next composition bottleneck.',
        'asset_status':'Previously compared assets, new layouts and multi-object task composition; not unseen-asset validation.',
        'container_policy':'shallowest native upward-support-qualified compatible container with floor radius >= sum of payload radii + 20 mm',
        'controller':'same retained stability_motion','acceptance':'same frozen native evaluator; no relaxations',
        'api_limit':0,'no_retries_after_physical_failure':True})
    output.mkdir(parents=True)
    code=runner.code();code['scripts/register_packing_pilot.py']=binding(__file__)
    snapshot=resources.gpu_snapshot(2)
    write(output/'experiment.json',{'schema':'multi_payload_packing_pilot_v1',
        'budget':str(ROOT/'runs/cudex_physical_budget_20261026'),'gpu':2,'gpu_uuid':snapshot['uuid'],
        'timeout_seconds':360,'code':code,'legacy_code':core.implementation(),'api_limit':0,
        'protocol':binding(ROOT/'validation/cudex_physical_20261026/packing_protocol.json')})
    write(output/'rules.json',{'version':evaluator.VERSION,'files':evaluator.bindings(),
        'frozen_before_attempts':True,'retained_rules_unchanged':True})
    cases=[]
    for label,indices in groups:
        es=[entries[i] for i in indices];ds=[read(resolve(e['asset'])) for e in es]
        if len({d['partition'] for d in ds})!=1:raise ValueError('source split mismatch')
        radii=[np.linalg.norm(d['half_size'][:2]) for d in ds]
        choices=[(b,d) for b,d in container_choices(ROOT/'runs/pickplace_containers_20261026_catalog_v2.json',
            ds[0]['partition'],np.max([d['half_size'] for d in ds],axis=0))
            if b['sha256'] in ok and d['cavity']['flat_floor_radius_m']>=sum(radii)+.020]
        choices.sort(key=lambda x:(x[1]['dimensions_m'][2],x[1]['id']))
        if not choices:raise ValueError('no_capacity_compatible_container')
        cb,cd=choices[0]
        scene,_=scene_for(es[0],indices[0]);scene['id']='packing_'+label
        scene['scene_instance_id']=scene['id'];scene['active_source_ids']=[d['source_id'] for d in ds]
        scene['split_group']='sources_'+digest(sorted(scene['active_source_ids']))[:16]
        scene['objects']=scene['objects'][1:];scene['grasp_caches']={}
        for k,(i,e,d) in enumerate(zip(indices,es,ds)):
            name=f'object_{k}';sign=-1 if k==0 else 1
            scene['objects'].insert(k,item(e['asset'],d,name,[-.22,sign*.46],.3 if k==0 else -.4,True))
            key='dev_00' if i==0 else f'asset_{i:02d}'
            scene['grasp_caches'][name]=binding(BASE/'proposal_caches'/key/'grasp_cache.json')
        support=scene['supports'][0]
        support.update(container_descriptor=cb,position=[-.10,0.,0.],yaw=0.,
            half_size=(np.asarray(cd['dimensions_m'])/2).tolist(),top_z=cd['cavity']['floor_z_m'],
            semantic_description='the '+cd['name'])
        scene['randomization'].update(source_and_goal='two opposite-side source positions, common container',
            composition='two payloads share one destination with automatic packing offsets')
        path=output/'scenes'/(scene['id']+'.json');write(path,scene)
        for order in [[0,1],[1,0]]:
            name=f'pack_{order[0]}_then_{order[1]}';identity=scene['id']+'_'+name
            folder=output/'inputs'/identity
            derived,task,reason=compose_packing(scene,[f'object_{i}' for i in order],support['id'],folder,name)
            if derived is not None:
                write(folder/'role_intent.json',task)
                selected,planned=bind_selection(derived,task,folder)
                if selected is None:reason='no_single_arm_endpoint_solution'
            else:selected=planned=None
            case={'id':identity,'execution_id':identity,'partition':ds[0]['partition'],
                'experiment_split':label,'active_source_ids':scene['active_source_ids'],
                'split_group':scene['split_group'],'shared_scene':binding(path),
                'method':'cudex_proposals_with_packing','source_coverage':'synthetic_only',
                'eligible':selected is not None,'packing_allocation':binding(folder/'packing_allocation.json')}
            if selected is not None:
                validate_task(selected,planned)
                write(folder/'scene.json',selected);write(folder/'task.json',planned)
                case.update(scene=core.bind(folder/'scene.json'),task=core.bind(folder/'task.json'),
                    arm_selection=binding(folder/'arm_selection.json'))
            else:case['preflight_reason']=reason
            cases.append(case);print(identity,case['eligible'],reason,flush=True)
    write(output/'cohort.json',{'cases':cases,'failures_retained':True,
        'physical_attempt_limit':4,'same_sources_cannot_cross_splits':True,
        'independent_contact_mechanisms':1,'shared_scenes':2,'programs':4,
        'new_heldout_assets':0,'new_heldout_layouts_and_compositions':2})


if __name__=='__main__':register()
