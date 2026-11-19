"""Bind every published cache and support to its actual registered producer.

This wraps the preserved v1 selection with an immutable input capture and
strict producer-chain checks. It does not change geometry, size, pose filters,
receiving profiles, or the original library.
"""
import argparse
from collections import defaultdict
from pathlib import Path

from resimpair_grasp.common import binding,digest,read,resolve,write
from resimpair_tasks.cudex_bulk_queue_v2 import validate_result
from . import raw_candidate_bundle_v1 as retained


def check_native_properties(entry,descriptor):
    for key in ('source_id','variant_id','native_support','native_properties'):
        if entry.get(key)!=descriptor.get(key):
            raise ValueError('published_native_property_source_mismatch:'+key)


def validate_grasp_chain(root,entry,protocol,population):
    sid=entry['source_id'];cpu=resolve(population[sid]['job']);folder=cpu.parent
    if str(cpu) not in protocol['job_paths'] or cpu != root/'cpu'/digest(sid)[:24]/'job.json':
        raise ValueError('CPU_job_not_original_registered_source')
    job=read(cpu);reservation=read(folder/'cpu_reservation.json')
    if reservation['job']!=binding(cpu) or reservation['protocol']!=binding(root/'protocol.json'):
        raise ValueError('CPU_reservation_not_bound_to_frozen_source')
    terminal=read(folder/'cpu_terminal.json');result=read(folder/'cpu_result.json')
    if (terminal.get('status')!='native_ready' or result.get('status')!='native_ready'
            or terminal['entry']!=result['entry'] or result['entry']!=binding(folder/'native_entry.json')
            or result['state']!=entry['source_state']):
        raise ValueError('CPU_source_did_not_produce_published_native_entry')
    native=read(resolve(result['entry']));descriptor=read(resolve(entry['asset']))
    if native['asset']!=entry['asset']:raise ValueError('published_asset_not_current_CPU_result')
    check_native_properties(native,descriptor)
    state=read(resolve(entry['source_state']))
    if (job['raw_row']['id']!=sid or state['source_id']!=sid
            or job['raw_row']['partition']!=state['partition'] or descriptor['partition']!=state['partition']
            or entry.get('source_aliases',[])!=job.get('source_aliases',[])):
        raise ValueError('published_source_split_or_alias_changed')
    gpu=root/'gpu'/digest(sid)[:24];gj=gpu/'job.json';gjob=read(gj)
    gres=read(gpu/'reservation.json');gt=read(gpu/'terminal.json')
    if (gjob['entry']!=native or gjob['asset']!=entry['asset'] or gjob['source_state']!=entry['source_state']
            or gres['job']!=binding(gj) or gres['protocol']!=binding(root/'protocol.json')
            or gt.get('status')!='completed' or gt['result']!=entry['optimizer_result']
            or gt['result']!=binding(gpu/'result.json')):
        raise ValueError('GPU_entry_or_reservation_not_bound_to_registered_CPU_result')
    actual=validate_result(gj)
    if actual['cache']!=entry['cache'] or actual['retained']!=entry['retained']:
        raise ValueError('published_cache_not_actual_GPU_export')
    return {'source_id':sid,'CPU_job':binding(cpu),'CPU_terminal':binding(folder/'cpu_terminal.json'),
        'native_entry':result['entry'],'GPU_job':binding(gj),'GPU_terminal':binding(gpu/'terminal.json'),
        'exact_native_properties_match':True,'exact_pose_export_match':True}


def reviewed_receivers(report_ref):
    report=read(resolve(report_ref));approved={}
    if report.get('parent_container_review'):approved.update(reviewed_receivers(report['parent_container_review']))
    approved.update(retained.prior_supports(report_ref))
    if report.get('new_container_review'):
        review=read(resolve(report['new_container_review']))
        if review.get('all_binding_and_numeric_checks_passed') is not True:
            raise ValueError('prior_raw_container_review_required')
        for row in review['rows']:
            if row['status']=='candidate_geometry_ready':
                result=read(resolve(row['result']));approved[result['asset']['sha256']]=row['result']
    return approved


def validate_receiver_chains(baseline,catalog):
    b=read(baseline);old=read(resolve(b['containers']))['assets']
    report=read(resolve(b['container_affordance_consumer']))
    if report['catalog']!=b['containers']:raise ValueError('prior_catalog_review_mismatch')
    approved=reviewed_receivers(b['container_affordance_consumer'])
    for ref in old:
        resolve(ref)
        if ref['sha256'] not in approved:raise ValueError('baseline_receiver_outside_reviewed_descriptor_set')
    cat=read(catalog);parent=read(resolve(cat['parent']));p=read(resolve(parent['protocol']))
    job_refs={read(resolve(ref))['source_id']:ref for ref in p['jobs']}
    original_refs={r['sha256']:r for r in parent['assets']};rows=[]
    for ref in cat['assets']:
        if original_refs.get(ref['sha256'])!=ref:raise ValueError('receiver_outside_original_geometry_catalog')
        d=read(resolve(ref));job_ref=job_refs[d['source_id']];job_path=resolve(job_ref);folder=job_path.parent
        reservation=read(folder/'reservation.json');terminal=read(folder/'terminal.json')
        if reservation['job']!=job_ref or terminal['status']!='candidate_geometry_ready':
            raise ValueError('receiver_did_not_finish_registered_job')
        result=read(resolve(terminal['result']))
        if (result['asset']!=ref or d['native_support_check']!=binding(folder/'NATIVE_GEOMETRY_AND_SUPPORT.json')
                or d['source']!=read(job_path)['expected_source']
                or d['partition']!=read(job_path)['raw_row']['partition']):
            raise ValueError('receiver_support_or_source_not_from_bound_job')
        rows.append({'descriptor':ref,'job':job_ref,'result':terminal['result'],'support':d['native_support_check']})
    partitions=defaultdict(set)
    for ref in old+cat['assets']:
        d=read(resolve(ref));partitions[d['source_id']].add(d['partition'])
        for alias in d.get('source_aliases',[]):
            if isinstance(alias,dict):
                partitions[alias.get('id',alias.get('source_id'))].add(alias['partition'])
                if alias['partition']!=d['partition']:raise ValueError('receiver_alias_split_conflict')
    if any(len(parts)!=1 for parts in partitions.values()):raise ValueError('receiver_source_split_conflict')
    return {'old_exact_reviewed_descriptors':len(old),'new_rows':rows,'all_source_splits_consistent':True}


def publish(raw_root,baseline,container_catalog,output):
    root=Path(raw_root).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('preserve_existing_validated_consumer')
    p=read(root/'protocol.json');population={r['source_id']:r for r in read(resolve(p['population']))['sources']}
    live=read(root/'native_cache_index.json')
    rows=[validate_grasp_chain(root,e,p,population) for e in live['entries']]
    receiving=validate_receiver_chains(baseline,container_catalog)
    # Capture rolling content before delegation. Registered completed producer
    # files are immutable; symlinks keep their original path/hash identities.
    stage=output/'captured_raw_root';stage.mkdir(parents=True)
    for name in ('protocol.json','cpu','gpu'):(stage/name).symlink_to(root/name,target_is_directory=name!='protocol.json')
    write(stage/'native_cache_index.json',live)
    write(output/'CHAIN_REVIEW.json',{'implementation':binding(__file__),'raw_protocol':binding(root/'protocol.json'),
        'captured_index':binding(stage/'native_cache_index.json'),'grasp_rows':rows,'receiver_review':receiving,
        'retained_publisher':binding(retained.__file__),'thresholds_changed':False,'physical_success_inferred':False})
    value=retained.publish(stage,baseline,container_catalog,output/'consumer')
    write(output/'RESULT.json',{**value,'producer_chain_review':binding(output/'CHAIN_REVIEW.json')})
    return {**value,'producer_chain_review':binding(output/'CHAIN_REVIEW.json')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('raw-root','baseline','container-catalog','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(publish(a.raw_root,a.baseline,a.container_catalog,a.output))
