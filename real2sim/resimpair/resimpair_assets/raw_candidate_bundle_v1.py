"""Snapshot corrected raw-source caches and receiving surfaces for task use.

The original library stays available separately. This input contains only new
exact-scale caches; it never fills an unsuccessful corrected source with its
old differently scaled cache. Native qualification is not a task success label.
"""
import argparse
from collections import Counter
from pathlib import Path
import time

import numpy as np

from resimpair_grasp.common import ROOT,binding,digest,read,resolve,write
from resimpair_grasp.grasps import load as load_cache
from .metric_admission_v1 import inspect
from .semantic_scale_consumer_v1 import classify
from resimpair_tasks.container_affordance_consumer_v1 import CANDIDATES,place_relation
from validation.paired_assets_20261030.independent.container_affordance_census_v1 import band


def receiving_geometry(descriptor):
    """Task-use classification, additional to the unchanged native support test.

    It distinguishes a low receiving recess from the top indentation of a large
    closed crate. It does not assert semantic identity or measured emptiness.
    """
    bounds=np.asarray(descriptor['bounds_m'],float);c=descriptor['cavity']
    h=float(bounds[1,2]-bounds[0,2]);z=float(c['floor_z_m']-bounds[0,2])
    depth=float(c['depth_m']);radius=float(c['flat_floor_radius_m'])
    if not np.isfinite([h,z,depth,radius]).all() or min(h,depth,radius)<=0:
        raise ValueError('finite_receiving_geometry_required')
    fraction=z/h
    group=band(h,fraction,depth)
    relation=CANDIDATES[group]
    automatic=group in {'shallow_tray_like_cavity','deeper_open_cavity_geometry_candidate'}
    return {'classification':group,'proposed_relation':relation,'automatic_receiving_use':automatic,
        'height_m':h,'floor_height_fraction':fraction,'depth_m':depth,
        'usable_floor_diameter_m':2*radius,'semantic_truth_verified':False,
        'assertable_by_R2H':False,'asserted_semantic_relation':'unknown',
        'reason':'Conservative task-input relation only; original geometry/support criteria unchanged.',
        'implementation':binding(__file__)}


def prior_supports(report_ref):
    report=read(resolve(report_ref));result={}
    if report.get('parent_container_review'):result.update(prior_supports(report['parent_container_review']))
    if report.get('affordance_index'):
        review=read(resolve(report['independent_review']))
        if (review.get('strict_subset_and_metadata_integrity_passed') is not True
                or review.get('affordance_index')!=report['affordance_index']):
            raise ValueError('original_independent_container_review_binding_required')
        for row in read(resolve(report['affordance_index']))['rows']:
            if row['retained_oriented_floor_support'].get('passed') is True:
                result[row['descriptor']['sha256']]=report['affordance_index']
    return result


def publish(raw_root,baseline,container_catalog,output):
    raw_root=Path(raw_root).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('preserve_existing_raw_candidate_snapshot')
    original=read(baseline);protocol=read(raw_root/'protocol.json')
    # The rolling index is captured once, then only this immutable copy is used.
    live=read(raw_root/'native_cache_index.json')
    if live['protocol']!=binding(raw_root/'protocol.json'):
        raise ValueError('rolling_cache_protocol_changed')
    write(output/'input_cache_snapshot.json',live)
    rows=[];entries=[];states=[]
    for e in live['entries']:
        sid=e['source_id'];job_path=raw_root/'gpu'/digest(sid)[:24]/'job.json'
        job=read(job_path);native=job['entry'];state=read(resolve(e['source_state']))
        descriptor=read(resolve(e['asset']));cache=load_cache(e['cache'],e['asset'])
        if (e['asset']!=job['asset'] or descriptor['source_id']!=sid
                or descriptor['partition']!=state['partition'] or len(cache['candidates'])!=e['retained']):
            raise ValueError('raw_snapshot_source_cache_or_partition_binding_changed')
        raw_job=read(raw_root/'cpu'/digest(sid)[:24]/'job.json')
        gate=inspect(e,state);revision=classify(raw_job['raw_row'],state)
        row={'source_id':sid,'partition':descriptor['partition'],'asset':e['asset'],'cache':e['cache'],
            'admission':gate,'consumer_revision':revision,'ingestion_group':e['ingestion_group'],
            'dimensions_m':descriptor['dimensions_m'],'retained_poses':e['retained']}
        accepted=gate['passed'] and not revision['automatic_manipulation_hold'];row['accepted']=accepted;rows.append(row)
        if accepted:
            entries.append({**native,**e,'semantic_scale_admission':gate,'partition':descriptor['partition'],
                            'source_alias_split_consistent':True})
            states.append(state)
    if len({e['source_id'] for e in entries})!=len(entries):
        raise ValueError('one_corrected_variant_per_source_snapshot_required')
    write(output/'native_cache_index.json',{'schema':'resimpair_grasp_cache_index_v1','entries':entries})
    write(output/'source_state_index.json',{'schema':'raw_candidate_source_states_v1','entries':states})
    write(output/'admission_audit.json',{'rows':rows,'raw_protocol':binding(raw_root/'protocol.json'),
        'native_admission':binding(ROOT/'resimpair_assets/metric_admission_v1.py'),
        'consumer_revision':binding(ROOT/'resimpair_assets/semantic_scale_consumer_v1.py')})
    # Keep the old reviewed candidates and new native-supported raw candidates.
    refs=read(resolve(original['containers']))['assets']+read(container_catalog)['assets']
    old_support=prior_supports(original['container_affordance_consumer'])
    seen=set();receivers=[];receiver_rows=[];source_partitions={e['source_id']:read(resolve(e['asset']))['partition'] for e in entries}
    for ref in refs:
        if ref['sha256'] in seen:continue
        seen.add(ref['sha256']);d=read(resolve(ref));sid=d['source_id']
        if sid in source_partitions and source_partitions[sid]!=d['partition']:
            raise ValueError('receiver_and_active_source_partition_conflict')
        if not d['roles'].get('stationary_container'):raise ValueError('original_container_role_required')
        if d.get('native_support_check'):
            check=read(resolve(d['native_support_check']))
            if check.get('native_oriented_support',{}).get('passed') is not True:
                raise ValueError('original_native_receiving_support_required')
            support_ref=d['native_support_check']
        else:
            if ref['sha256'] not in old_support:raise ValueError('old_receiver_missing_bound_native_support')
            support_ref=old_support[ref['sha256']]
        relation=receiving_geometry(d)
        receiver_rows.append({'source_id':sid,'descriptor':ref,'partition':d['partition'],'name':d['name'],
            'native_support_evidence':support_ref,**relation})
        if relation['automatic_receiving_use']:receivers.append(ref)
    write(output/'receiving_geometry.json',{'rows':receiver_rows,'semantic_certification':False,
        'input_catalogs':[original['containers'],binding(container_catalog)],
        'high_top_recesses_are_not_empty_containers':True})
    write(output/'containers.json',{'schema':'stationary_container_catalog_v1','assets':receivers,
        'receiving_geometry':binding(output/'receiving_geometry.json'),'physical_task_success':'unmeasured'})
    summary={'admitted_grasp_sources':len(entries),'admitted_grasp_variants':len(entries),
        'poses':sum(e['retained'] for e in entries),'new_raw_sources':sum(e['ingestion_group']=='new_raw' for e in entries),
        'size_corrected_sources':sum(e['ingestion_group']=='size_corrected' for e in entries),
        'partition_source_counts':dict(Counter(s['partition'] for s in states)),
        'container_sources':len({read(resolve(r))['source_id'] for r in receivers}),
        'container_variants':len(receivers),'receiving_profile_counts':dict(Counter(r['classification'] for r in receiver_rows)),
        'legacy_cache_fallbacks':0,'physical_success_inferred':False,'API_calls':0}
    b={'schema':'live_source_metric_bundle_v4','summary':summary,
        **{k:binding(output/(k+'.json')) for k in ['native_cache_index','source_state_index','containers','admission_audit']},
        'source_semantics_checked_against_current_policy':{'schema':'source_semantic_scale_gate_v1',
            'source_state_index':binding(output/'source_state_index.json'),'gate':binding(ROOT/'resimpair_assets/metric_admission_v1.py'),
            'all_cache_entries_passed':True},
        'static_catalog':original['static_catalog'],'background_catalog':original['background_catalog'],
        'receiving_geometry':binding(output/'receiving_geometry.json'),
        'parent_bundle':binding(baseline),'input_snapshot':binding(output/'input_cache_snapshot.json'),
        'raw_protocol':binding(raw_root/'protocol.json'),'implementation':binding(__file__),
        'publication_status':'opt_in_candidate; old default unchanged','created_unix':time.time()}
    write(output/'BUNDLE.json',b)
    from resimpair_tasks.metric_population_sampler_v4 import admitted_pool
    _,loaded,sources,bins=admitted_pool(output/'BUNDLE.json')
    if len(loaded)!=len(entries) or len(sources)!=len(entries) or len(bins)!=len(receivers):
        raise ValueError('actual_task_consumer_reload_changed_membership')
    result={**summary,'bundle':binding(output/'BUNDLE.json'),'actual_task_consumer_reload_passed':True}
    write(output/'RELOAD.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('raw-root','baseline','container-catalog','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(publish(a.raw_root,a.baseline,a.container_catalog,a.output))
