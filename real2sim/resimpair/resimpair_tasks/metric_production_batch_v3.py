"""Unified finite metric candidate batch; retained execution and audit rules.

Explicit immutable semantic/container consumer snapshots feed sampler v4,
target-side source proposals and the same-arm local corridor condition. The
validated withdrawal driver4 is used normally. Driver6 closure is opt-in only.
This entry does not register budgets, call APIs or promote a default pipeline.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import fcntl
from pathlib import Path

from resimpair_grasp.common import binding,digest,read,resolve,write
from .metric_production_batch_v2 import GOALS

VERSION='metric_production_batch_v3'


def consumer_inputs(bundle,container_catalog):
    bundle,container_catalog=Path(bundle).resolve(),Path(container_catalog).resolve()
    b=read(bundle)
    if b.get('schema')!='live_source_metric_bundle_v4':
        raise ValueError('immutable_metric_live_bundle_required')
    reports={}
    for key in ('consumer_semantic_revision','container_affordance_consumer'):
        ref=b.get(key)
        if not isinstance(ref,dict) or set(ref)!={'path','sha256'}:
            raise ValueError('final_bound_semantic_and_container_consumer_required:'+key)
        report=read(resolve(ref))
        if report.get('consumer_qualified') is not True:
            raise ValueError('consumer_qualification_report_required:'+key)
        reports[key]=report
    if b['containers']!=binding(container_catalog):
        raise ValueError('explicit_strict_container_catalog_must_match_final_bundle')
    semantic=reports['consumer_semantic_revision'];container=reports['container_affordance_consumer']
    if semantic.get('native_cache_index')!=b['native_cache_index'] or semantic.get('static_catalog')!=b['static_catalog']:
        raise ValueError('semantic_report_must_bind_the_exact_consumed_indices')
    if container.get('catalog')!=b['containers']:
        raise ValueError('container_report_must_bind_the_exact_consumed_catalog')
    # The original sampler still independently checks per-entry semantic gates,
    # native cache bindings, physical-size variant identities and source splits.
    for key in ('native_cache_index','static_catalog','containers'):
        resolve(b[key])
    return b


def campaign_inputs(campaign):
    campaign=Path(campaign).resolve()
    limits=read(campaign/'budget/limits.json')
    allocation=read(campaign/'allocation.json');receipt=read(resolve(allocation['registration']))
    if (limits['physical_attempt_limit']!=128 or receipt['limit']!=128
            or Path(receipt['root']).resolve()!=campaign
            or receipt['budget']!=binding(campaign/'budget/limits.json')
            or limits['API_calls']!=0 or limits['GPU_jobs']!=0):
        raise ValueError('existing_bound_128_attempt_CPU_allocation_required')
    return limits


def verify_compiled(factory):
    """Check actual compiled IDs and source partitions, including all clutter."""
    factory=Path(factory)
    if not (factory/'cohort.json').exists():return {'cases':0,'scenes':[],'source_partitions':{}}
    cases=read(factory/'cohort.json')['cases'];seen_ids=set();scenes={};source_partitions={}
    for case in cases:
        scene=read(resolve(case['scene']));task=read(resolve(case['task']))
        if case['id'] in seen_ids:raise ValueError('duplicate_compiled_task_id')
        seen_ids.add(case['id'])
        expected=scene['id']+'_'+case['requested_task_id']
        if (case['id']!=expected or task['id']!=expected or case['execution_id']!=expected
                or case['partition']!=scene['partition']):
            raise ValueError('compiled_scene_task_identity_or_partition_mismatch')
        if scene['id'] not in scenes:
            all_sources=[]
            for item in list(scene['objects'])+list(scene['supports']):
                ref=item.get('asset_descriptor') or item.get('container_descriptor')
                if ref is None:raise ValueError('metric_source_descriptor_required')
                descriptor=read(resolve(ref));sid=descriptor['source_id'];partition=descriptor['partition']
                if partition!=scene['partition']:
                    raise ValueError('target_container_or_distractor_source_crosses_scene_split')
                if sid in source_partitions and source_partitions[sid]!=partition:
                    raise ValueError('one_original_source_crosses_batch_split')
                source_partitions[sid]=partition;all_sources.append(sid)
            scenes[scene['id']]={'scene':case['scene'],'partition':scene['partition'],
                'target_object_count':len(scene['metric_variants']),
                'total_physical_objects_including_clutter':len(scene['objects']),
                'all_original_sources_including_containers_clutter':sorted(set(all_sources)),
                'task_ids':[]}
        scenes[scene['id']]['task_ids'].append(case['id'])
    return {'cases':len(cases),'scenes':list(scenes.values()),'source_partitions':source_partitions,
        'unique_task_IDs':True,'original_source_partition_preserved':True,
        'physical_success':'unmeasured'}


def commit(campaign,name):
    """Recover a compiled, unsubmitted preparation without resampling it."""
    from . import production_integration_v4 as stable_driver,production_integration_v6 as closure_driver
    from resimpair_pickplace.requested_tasks import identity
    campaign=Path(campaign).resolve();identity(name);campaign_inputs(campaign)
    staging=campaign/'prepared_inputs'/name
    request=read(staging/'preparation_request.json')
    if request['schema']!=VERSION or request['implementation']!=binding(__file__):
        raise ValueError('compiled_preparation_code_binding_changed')
    consumer_inputs(resolve(request['bundle']),resolve(request['container_catalog']))
    prepared=read(staging/'prepared_cases.json')
    if prepared['preparation_request']!=binding(staging/'preparation_request.json'):
        raise ValueError('prepared_cases_contract_binding_mismatch')
    cases=prepared['cases']
    for case in cases:
        for key in ('scene','task','sampling_protocol','generation_contract','identity_and_split_check'):
            resolve(case[key])
    driver=closure_driver if request['experimental_closure'] else stable_driver
    if request['execution_driver']!=binding(driver.__file__):
        raise ValueError('prepared_execution_driver_changed')
    with (campaign/'.supervisor.lock').open('a+') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        output=campaign/name
        if output.exists():
            if (not (output/'frozen_inputs.json').exists()
                    or read(output/'cohort.json')['cases']!=cases):
                raise ValueError('partial_or_different_cohort_requires_inspection_no_overwrite')
            for ref in read(output/'frozen_inputs.json').values():resolve(ref)
            result={'root':str(output),'requests':len(cases)}
        else:
            result=driver.prepare_cohort(campaign,name,cases,prefix_reuse=True,full_open=False)
    write(staging/'commit_result.json',{'schema':VERSION,'prepared_cases':binding(staging/'prepared_cases.json'),
        'result':result,'physical_attempts':0,'API_calls':0})
    return result


def prepare(campaign,name,bundle,scenes,seed,*,container_catalog,count_schedule=None,
            minimum=3,maximum=8,goals=('all',),experimental_closure=False):
    from . import metric_population_sampler_v4 as sampler
    from .metric_operation_corridor_v2 import qualify_scene
    from .metric_target_conditioned_pose_v1 import propose
    from . import production_integration_v4 as stable_driver,production_integration_v6 as closure_driver
    from resimpair_pickplace.requested_tasks import identity
    campaign=Path(campaign).resolve();identity(name)
    if not goals or len(set(goals))!=len(goals) or (tuple(goals)!=('all',) and set(goals)-set(GOALS)):
        raise ValueError('explicit_unique_supported_goals_or_all_required')
    if type(experimental_closure) is not bool:raise ValueError('explicit_boolean_closure_switch_required')
    b=consumer_inputs(bundle,container_catalog);limits=campaign_inputs(campaign)
    allocated=sum(len(read(p)['cases']) for p in campaign.glob('*/cohort.json'))
    maximum_cases=scenes*(len(GOALS) if tuple(goals)==('all',) else len(goals))
    if allocated+maximum_cases>limits['physical_attempt_limit']:
        raise ValueError('requested_goals_exceed_existing_cumulative_allocation')
    staging=campaign/'prepared_inputs'/name
    if staging.exists() or (campaign/name).exists():
        raise ValueError('existing_batch_preserve_and_recover_do_not_resample')
    driver=closure_driver if experimental_closure else stable_driver
    request={'schema':VERSION,'implementation':binding(__file__),'bundle':binding(bundle),
        'container_catalog':binding(container_catalog),'requested_scenes':scenes,'seed':seed,
        'count_schedule':count_schedule,'object_count_range':[minimum,maximum],
        'requested_execution_goals':list(goals),'sampler':binding(sampler.__file__),
        'object_pose_proposal':sampler.callback_binding(propose),'layout_qualifier':sampler.callback_binding(qualify_scene),
        'execution_driver':binding(driver.__file__),'experimental_closure':experimental_closure,
        'closure_default_promoted':False,'pipeline_default_promoted':False,
        'maximum_physical_cases':maximum_cases,'physical_attempts':0,'API_calls':0,
        'cumulative_budget':binding(campaign/'budget/limits.json'),'maximum_sampling_proposals_per_scene':16,
        'consumer_semantic_revision':b['consumer_semantic_revision'],
        'container_affordance_consumer':b['container_affordance_consumer']}
    write(staging/'preparation_request.json',request)
    try:
        sampled=sampler.build(bundle,staging/'sampled',scenes=scenes,seed=seed,
            minimum=minimum,maximum=maximum,container_catalog=container_catalog,count_schedule=count_schedule,
            object_pose_proposal=propose,layout_qualifier=qualify_scene)
        factory=staging/'sampled/factory';checks=verify_compiled(factory)
        write(staging/'compiled_identity_split_check.json',checks)
        cases=[]
        if (factory/'cohort.json').exists():
            for source in read(factory/'cohort.json')['cases']:
                if tuple(goals)!=('all',) and source['requested_task_id'] not in goals:continue
                task=deepcopy(read(resolve(source['task'])))
                for op in task['operations']:
                    op['release']['mode']='supported_place';op['release']['allow_drop']=False
                path=staging/'release_contracts'/source['id']/'task.json';write(path,task)
                cases.append({**source,'task':binding(path),'parent_task':source['task'],
                    'release_selection':'supported_native_clearance','experimental_closure':experimental_closure,
                    'batch_preparer':binding(__file__),'preparation_request':binding(staging/'preparation_request.json'),
                    'sampling_protocol':binding(staging/'sampled/protocol.json'),
                    'generation_contract':binding(staging/'sampled/generation_contract.json'),
                    'identity_and_split_check':binding(staging/'compiled_identity_split_check.json')})
        if not cases:result={'prepared':False,'physical_cases':0,'reason':'no_requested_goals_prepared'}
        else:
            write(staging/'prepared_cases.json',{'preparation_request':binding(staging/'preparation_request.json'),
                'cases':cases})
            # A busy existing supervisor can be recovered with commit after
            # it releases the shared lock; no scenes or proposals are redrawn.
            result=commit(campaign,name)
        final={'schema':VERSION,'request':request,'sampled':sampled,'prepared':result,
            'compiled_identity_split_check':binding(staging/'compiled_identity_split_check.json'),
            'physical_cases':len(cases),'physical_attempts':0,'API_calls':0,
            'qualification_scope':'semantic/geometry/nominal local kinematics only; full live paths and physical outcome unmeasured'}
        write(staging/'preparation_result.json',final)
        return final
    except Exception as exc:
        write(staging/'preparation_failure.json',{'schema':VERSION,'type':type(exc).__name__,'message':str(exc),
            'physical_attempts_by_preparer':0,'API_calls':0,'existing_outputs_preserved':True,
            'automatic_resampling_or_retry':False})
        raise


def run(cohort,limit=None):
    from . import production_integration_v4 as stable_driver,production_integration_v6 as closure_driver
    cohort=Path(cohort).resolve();spec=read(cohort/'experiment.json');cases=read(cohort/'cohort.json')['cases']
    if not cases:raise ValueError('nonempty_frozen_cohort_required')
    requests={digest(c['preparation_request']):c['preparation_request'] for c in cases}
    if len(requests)!=1:raise ValueError('one_bound_preparation_contract_required')
    ref=next(iter(requests.values()));request=read(resolve(ref))
    if request['schema']!=VERSION or request['implementation']!=binding(__file__):
        raise ValueError('candidate_batch_implementation_changed')
    consumer_inputs(resolve(request['bundle']),resolve(request['container_catalog']))
    driver=closure_driver if request['experimental_closure'] else stable_driver
    if spec['schema']!=driver.VERSION or request['execution_driver']!=binding(driver.__file__):
        raise ValueError('frozen_execution_driver_does_not_match_explicit_closure_choice')
    for key in ('sampler','cumulative_budget','consumer_semantic_revision','container_affordance_consumer'):
        resolve(request[key])
    for key in ('object_pose_proposal','layout_qualifier'):
        resolve(request[key]['implementation'])
    # Retained driver owns the original global/campaign locks, 128-attempt
    # accounting, finite timeout, failure records and unknown-submit stop.
    return driver.run(cohort,limit)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--campaign',type=Path,required=True);q.add_argument('--name',required=True)
    q.add_argument('--bundle',type=Path,required=True);q.add_argument('--container-catalog',type=Path,required=True)
    q.add_argument('--scenes',type=int,required=True);q.add_argument('--seed',type=int,required=True)
    q.add_argument('--min-objects',type=int,default=3);q.add_argument('--max-objects',type=int,default=8)
    q.add_argument('--count-schedule');q.add_argument('--goals',nargs='+',default=['all'])
    q.add_argument('--experimental-closure',action='store_true',help='explicit driver6 experiment; never the normal driver4 path')
    q=sub.add_parser('commit');q.add_argument('--campaign',type=Path,required=True);q.add_argument('--name',required=True)
    q=sub.add_parser('run');q.add_argument('--cohort',type=Path,required=True);q.add_argument('--limit',type=int)
    a=p.parse_args()
    if a.command=='prepare':
        value=prepare(a.campaign,a.name,a.bundle,a.scenes,a.seed,container_catalog=a.container_catalog,
            count_schedule=[int(x) for x in a.count_schedule.split(',')] if a.count_schedule else None,
            minimum=a.min_objects,maximum=a.max_objects,goals=tuple(a.goals),experimental_closure=a.experimental_closure)
        print({k:v for k,v in value.items() if k in ('schema','prepared','physical_cases','physical_attempts','API_calls')})
    elif a.command=='commit':print(commit(a.campaign,a.name))
    else:print(run(a.cohort,a.limit))

if __name__=='__main__':main()
