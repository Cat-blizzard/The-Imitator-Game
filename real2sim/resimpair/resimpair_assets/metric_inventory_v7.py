"""Trace every raw source through old preprocessing and a source-aware size plan.

Raw catalog membership is never confused with an executable grasp asset.
Missing metadata remains explicit; category scales are engineering hypotheses.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import re
import sqlite3
import time
import numpy as np
from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write

VERSION='source_metric_inventory_v7'
DEFAULT=ROOT/'configs/metric_population_v7.json'
RAW=ROOT/'assets/expanded_library/raw_roles_v4.sqlite'
NATIVE=ROOT/'runs/incremental_asset_library_20261028/native_ready_index.json'
POSE=ROOT/'runs/cudex_bulk_5524_20261028/asset_skip_v5/native_cache_index.json'
CONTAINERS=ROOT/'runs/pickplace_containers_20261026_catalog_v2.json'
ABO=Path('/mnt/sdg/datasets/open_assets/household_scaleup_20261020_v1/abo/metadata')
LENGTH={'inches':.0254,'inch':.0254,'centimeters':.01,'millimeters':.001,'meters':1.,'feet':.3048}
MASS={'pounds':.45359237,'ounces':.028349523125,'kilograms':1.,'grams':.001}


def unit_value(value, table):
    if not isinstance(value,dict): return None
    v=value.get('normalized_value',value)
    if v.get('unit') not in table or not isinstance(v.get('value'),(int,float)): return None
    x=float(v['value'])*table[v['unit']]
    return x if np.isfinite(x) and x>0 else None


def abo_metadata(wanted, output):
    matches=defaultdict(list); files=[]; start=time.monotonic()
    for path in sorted(ABO.glob('listings_*.json.gz')):
        ref=binding(path); files.append(ref)
        with gzip.open(path,'rb') as stream:
            for line_number,line in enumerate(stream,1):
                row=json.loads(line)
                ids={row.get('item_id'),row.get('3dmodel_id')} & wanted
                if not ids: continue
                dimensions={k:unit_value(v,LENGTH) for k,v in row.get('item_dimensions',{}).items()}
                item={'metadata_file':ref,'line_number':line_number,
                    'item_id':row.get('item_id'),'model_id':row.get('3dmodel_id'),
                    'dimensions_m':dimensions,'weights_kg':[v for x in row.get('item_weight',[])
                        if (v:=unit_value(x,MASS)) is not None],
                    'product_type':row.get('product_type'),'material':row.get('material')}
                for key in ids: matches[key].append(item)
    value={'schema':'all_local_ABO_metric_evidence_v1','matches':dict(matches),
        'files':files,'wanted':len(wanted),'matched':len(matches),'seconds':time.monotonic()-start}
    write(output,value); return value


def semantic_profile(row,config):
    from resimpair_assets.metric_semantics_v1 import profile
    return profile(row,config)


def metric_match(original_obb, records, tolerance=.1):
    original=np.sort(np.asarray(original_obb,float)); good=[]
    if original.shape!=(3,) or not np.isfinite(original).all() or min(original)<=0:
        return {'status':'original_geometry_extent_unavailable','matches':[]}
    for item in records:
        dims=[item['dimensions_m'].get(k) for k in ['height','length','width']]
        if not all(x is not None and x>0 for x in dims): continue
        ratios=np.sort(dims)/original; scale=float(np.median(ratios))
        error=float(max(abs(ratios/scale-1)))
        if error<=tolerance: good.append({'scale':scale,'relative_shape_error':error,'record':item})
    if not good: return {'status':'no_metadata_uniform_shape_match','matches':[]}
    scales=[x['scale'] for x in good]
    if max(scales)/min(scales)>1+tolerance:
        return {'status':'conflicting_metric_records','matches':good}
    best=min(good,key=lambda x:x['relative_shape_error'])
    return {'status':'matched_uniform_metric_metadata','matches':good,'selected':best}


def decide(row, d, old, metadata, evidence_ref, config):
    row={**row,'native_category':d.get('category')}
    profile=semantic_profile(row,config); role=profile['role']; source=row['source']
    old_size=old.get('semantic_notes',{}).get('size',{})
    original=old_size.get('original_obb_dimensions')
    transform=np.asarray(d.get('source_to_asset',np.eye(4)),float)
    scale=float(np.linalg.norm(transform[:3,0]))
    current_obb=np.asarray(original,float)*scale if original else np.sort(d['dimensions_m'])
    current_longest=float(max(current_obb))
    records=metadata.get(row['id'].split(':',1)[1],[]) if source=='abo' else []
    match=metric_match(original or [0,0,0],records,config['metadata_uniform_fit_tolerance'])
    evidence=[evidence_ref]
    if match['status']=='matched_uniform_metric_metadata':
        base=float(max(original))*match['selected']['scale']
        size={'mode':'source_metric','source_unit_to_m':match['selected']['scale'],
            'dimensions_m':sorted([match['selected']['record']['dimensions_m'][k] for k in ['height','length','width']]),
            'dimension_tolerance_relative':config['metadata_uniform_fit_tolerance'],
            'evidence':evidence,'rationale':'Local ABO item dimensions agree with original mesh OBB under a uniform metric transform.'}
        confidence='source_metadata_geometry_agreement'
    elif source in {'google_scanned_objects_original_fuel','ycb_official','polyhaven'} and original:
        base=float(max(original))
        # Source format/convention, not a new physical measurement.
        size={'mode':'inferred_category','base_longest_m':base,
            'plausible_longest_m':[base*.99,base*1.01], 'evidence':evidence,
            'rationale':'Retain original numeric source mesh scale using the metric simulation/scanned-source convention; not independently measured per object.'}
        confidence='source_metric_convention_engineering_inference'
    else:
        lower,upper=profile['longest_m']
        base=current_longest if lower<=current_longest<=upper else profile['base_longest_m']
        size={'mode':'inferred_category','base_longest_m':base,'plausible_longest_m':[lower,upper],
            'evidence':evidence,'rationale':f"Category/morphology engineering range {profile['name']}; retain an existing cached scale inside that range, otherwise use its explicit representative length. No gripper-width normalization."}
        confidence='engineering_category_or_morphology_assumption'
    # A metric-scanned small rigid proxy may be used with explicit unknown semantics.
    if profile['name']=='role_scale_pending' and confidence!='engineering_category_or_morphology_assumption' and base<=.4:
        role='manipulation';profile={**profile,'name':'metric_bounded_rigid_proxy','density_kg_m3':400}
    # Large metric sources stay full size and leave the dynamic grasp pool.
    if role=='manipulation' and (base>.6 or (max(original or [0])*scale>3)):
        role='background'
    weights=[]
    if match['status']=='matched_uniform_metric_metadata':
        weights=match['selected']['record']['weights_kg']
    if weights and max(weights)/min(weights)<=1.1:
        mass={'mode':'measured_base_mass','base_mass_kg':float(np.median(weights)),
            'evidence':evidence,'rationale':'Local ABO item_weight for the metadata geometry match; product specification, not a new laboratory weighing.'}
    else:
        mass={'mode':'effective_density','density_kg_m3':profile['density_kg_m3'],
            'rationale':f"Explicit {profile['name']} category engineering effective density over the cooked compound volume; hollow/part overlaps are approximate. No measured real mass inferred."}
    policy={'schema':'bounded_metric_variants_v2','source_id':d['source_id'],
        'semantic_role':role,'semantic_category':profile['name'],
        'source_object_role':'furniture' if profile['name']=='furniture' else 'household_movable',
        'role_rationale':'Name/category retrieval inference plus source metric envelope; not human verified.',
        'size':size,'mass':mass,'scale_factors':[1.0],'variant_ratio_bounds':[.9,1.1]}
    return {'profile':profile['name'],'role':role,'size_confidence':confidence,
        'classification_evidence':profile['classification_evidence'],
        'semantic_admission':'pending' if profile['name']=='role_scale_pending' else 'engineered_role_and_scale_candidate',
        'base_longest_m':base,'old_longest_m':current_longest,
        'base_multiplier_from_parent':base/current_longest,
        'unchanged_metric_geometry':abs(base/current_longest-1)<1e-6,
        'metric_metadata_status':match['status'],'mass_source':mass['mode'],
        'semantic_verified':False,'policy':policy}


def build(output, metadata_path=None, prior_manifest=None, prior_batch=None):
    start=time.monotonic(); output=Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True); config=read(DEFAULT)
    db=sqlite3.connect(RAW);db.row_factory=sqlite3.Row
    raw=[dict(r) for r in db.execute('SELECT id,source,name,category,partition,path,source_sha,eligible,reason,record FROM assets')];db.close()
    if metadata_path:
        metadata=read(metadata_path);write(output/'abo_metric_evidence.json',metadata)
    else:
        metadata=abo_metadata({r['id'].split(':',1)[1] for r in raw if r['source']=='abo'},output/'abo_metric_evidence.json')
    evidence_ref=binding(output/'abo_metric_evidence.json')
    for row in raw:
        record=json.loads(row.pop('record'));row['source_categories']=record.get('categories',[])
        row['source_kind']=record.get('kind')
    native=read(NATIVE)['entries']; poses=read(POSE)['entries']; containers=read(CONTAINERS)['assets']
    by_native={sid:e for e in native for sid in [e['source_id']]+e.get('source_aliases',[])}
    by_pose={sid:e for e in poses for sid in [e['source_id']]+e.get('source_aliases',[])}
    by_container=defaultdict(list)
    for ref in containers: by_container[read(resolve(ref))['source_id']].append(ref)
    offline={}; evidence=[]
    for name in ['v2','v3']:
        folder=ROOT/f'runs/tabletop_pose_library_20261026_{name}/reports'
        for path in folder.glob('*.json'):
            value=read(path)
            if 'rows' not in value:continue
            evidence.append(binding(path))
            for row in value['rows']:offline[row['source_id']]=row
    states=[]; jobs=[]; aliases=[]; counts=Counter()
    prior_jobs={j['source_id']:j for j in read(prior_manifest)['jobs']} if prior_manifest else {}
    def comparable(v):
        if isinstance(v,dict):
            if set(v)=={'path','sha256'}: return {'sha256':v['sha256']}
            return {k:comparable(x) for k,x in v.items()}
        if isinstance(v,list): return [comparable(x) for x in v]
        return v
    for row in raw:
        sid=row['id']; old=by_native.get(sid); pose=by_pose.get(sid); off=offline.get(sid)
        state={'source_id':sid,'source':row['source'],'name':row['name'],'category':row['category'],
            'partition':row['partition'],'source_categories':row['source_categories'],'source_kind':row['source_kind'],'raw_path':row['path'],'raw_sha256':row['source_sha'],
            'prior_raw_eligibility':row['eligible'],'prior_raw_reason':row['reason'],
            'offline_status':off.get('status') if off else 'not_attempted',
            'offline_asset':off.get('asset') if off else None,
            'native_asset':old['asset'] if old else None,
            'prior_pose_cache':pose.get('cache') if pose else None,
            'prior_pose_count':pose.get('retained',0) if pose else 0,
            'container_candidates':by_container.get(sid,[]),'physical_status':'unmeasured'}
        if old:
            if sid!=old['source_id']:
                state.update(status='canonical_alias',canonical_source_id=old['source_id'],
                    reason='exact geometry alias stays bound to canonical partition and source group')
                aliases.append({'source_id':sid,'canonical_source_id':old['source_id']})
            else:
                d=read(resolve(old['asset'])); decision=decide(row,d,old,metadata['matches'],evidence_ref,config)
                old_job=prior_jobs.get(sid); previous_result=None
                policy=decision.pop('policy')
                if old_job and comparable(read(resolve(old_job['policy'])))==comparable(policy):
                    key=old_job['key'];policy_path=resolve(old_job['policy'])
                    candidate=Path(prior_batch)/'jobs'/key/'result.json' if prior_batch else None
                    if candidate and candidate.exists():previous_result=binding(candidate)
                else:
                    key=digest({'source':sid,'parent':old['asset'],'policy':policy})[:24]
                    policy_path=output/'policies'/(key+'.json');write(policy_path,policy)
                state.update(decision,status=('role_scale_pending' if decision['semantic_admission']=='pending' else 'planned_metric_native' if decision['role']=='manipulation' else 'planned_static_role'),
                    policy=binding(policy_path),key=key,
                    reason='new metric/native/mass/support qualification before sampling; retained old cache remains historical')
                jobs.append({'source_id':sid,'key':key,'parent':old['asset'],'policy':binding(policy_path),
                    'prior_entry':pose,'source_aliases':old.get('source_aliases',[]),
                    'decision':decision,'previous_compilation_result':previous_result})
        else:
            profile=semantic_profile(row,config)
            state.update(role=profile['role'],semantic_profile=profile['name'],
                status='raw_or_offline_pending_native',
                reason=('historical_preprocessing_failed_or_no_offline_pose; source-aware preprocessing not yet executed'
                        if off else 'raw_source_not_yet_processed_by_source_aware_native_compiler'))
            if not row['path'] or not Path(row['path']).is_file():
                state.update(status='raw_missing',reason='raw_path_missing_on_disk')
            elif row['source']=='robocasa':
                state.update(status='articulated_package_pending_adapter',reason='MJCF/package source requires articulated adapter; not a rigid mesh discard')
        states.append(state);counts[state['status']]+=1
    summary={'raw_sources':len(states),'source_counts':dict(Counter(x['source'] for x in states)),
        'native_canonical':len(native),'native_sources_including_aliases':len(by_native),
        'nonempty_prior_cache_assets':len(poses),'prior_poses':sum(x.get('retained',0) for x in poses),
        'container_candidates':len(containers),'planned_native_jobs':len(jobs),
        'roles':dict(Counter(x['decision']['role'] for x in jobs)),
        'unchanged_geometry_plans':sum(x['decision']['unchanged_metric_geometry'] for x in jobs),
        'size_confidence':dict(Counter(x['decision']['size_confidence'] for x in jobs)),
        'semantic_admission':dict(Counter(x['decision']['semantic_admission'] for x in jobs)),
        'status_counts':dict(counts),'elapsed_seconds':time.monotonic()-start}
    write(output/'source_state_index.json',{'schema':VERSION,'entries':states,'summary':summary,
        'implementation':binding(__file__),
        'input_indexes':{'raw':binding(RAW),'native':binding(NATIVE),'poses':binding(POSE),'containers':binding(CONTAINERS)},
        'historical_offline_reports':evidence,'raw_not_processed_is_not_asset_unusable':True})
    write(output/'batch_manifest.json',{'schema':VERSION,'jobs':jobs,'config':binding(DEFAULT),
        'source_state_index':binding(output/'source_state_index.json'),'aliases':aliases,
        'cpu_workers':4,'per_source_timeout_seconds':120,'deadline_unix':time.time()+172800,
        'API_calls':0,'new_GPU_budget_separate':True})
    write(output/'summary.json',summary)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--metadata',type=Path);p.add_argument('--prior-manifest',type=Path);p.add_argument('--prior-batch',type=Path)
    a=p.parse_args();print(json.dumps(build(a.output,a.metadata,a.prior_manifest,a.prior_batch),ensure_ascii=False,indent=2))
