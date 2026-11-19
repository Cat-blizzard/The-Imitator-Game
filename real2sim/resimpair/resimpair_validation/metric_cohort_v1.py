"""Audit one completed metric cohort without generation, physics, or rendering.

Example:
  .venv/bin/python -m resimpair_validation.metric_cohort_v1 \
    --cohort runs/production_integration_20261030_v1/metric_group_supported \
    --output validation/new_independent_review

Outputs are immutable. Optional compatibility publication copies the actual
audit bytes into qualified native execution folders, never changing decisions.
"""
from pathlib import Path
from copy import deepcopy
import argparse
import importlib.util
import sys

import numpy as np
from resimpair_grasp.common import ROOT, binding, read, resolve, write
from resimpair_tabletop.sample_report import native_scalars

VERSION='independent_metric_cohort_acceptance_v1'
FROZEN=Path(__file__).with_name('frozen_metric_acceptance_v1.json')
FROZEN_SHA256='1f9b4e3ca9aac69d4a107e7741cec1c3faec55131d86e5f7edba276935359dfd'
MULTI=ROOT/'validation/multitask_20261030/independent/review_multitask_v3.py'
HELPER=ROOT/'validation/drop_release_20261030/independent/review_pilot_v2.py'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def canonical_protocol(value):
    value=deepcopy(value)
    rows=value['constraints'];ids=[r['id'] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('duplicate_event_constraint_identity')
    value['constraints']=sorted(rows,key=lambda r:r['id'])
    return value


def mass_evidence(source,bundle):
    path=source/'runtime_mass_properties.json'
    movable=[o for o in bundle['scene']['objects'] if o['kind']=='movable']
    if not path.exists():
        return {'status':'unknown','reason':'runtime_Body_readback_not_recorded',
            'expected_dynamic_objects':[o['id'] for o in movable],'rows':[]}
    try:
        report=read(path);resolve(report['implementation'])
        if report['actor_updates']!=0 or report['extra_physics_steps']!=0:
            raise ValueError('readback_must_not_change_native_world')
        rows=report['rows']
        if len(rows)!=len(movable) or {r['object'] for r in rows}!={o['id'] for o in movable}:
            raise ValueError('incomplete_or_duplicate_dynamic_body_readback')
        results=[];unknown=False;failed=False
        for row in rows:
            obj=next(o for o in movable if o['id']==row['object'])
            if row['asset']!=obj['asset_descriptor']:raise ValueError('native_body_descriptor_binding_changed')
            descriptor=read(resolve(row['asset']));actual=row['native_readback']
            required=('mass_kg','center_of_mass_m','cmass_pose_pq','principal_inertia_kg_m2','inertia_kg_m2')
            if any(not np.isfinite(np.asarray(actual[k],dtype=float)).all() for k in required):
                raise ValueError('nonfinite_runtime_native_body')
            if actual['mass_kg']<=0 or min(actual['principal_inertia_kg_m2'])<=0:
                raise ValueError('nonpositive_runtime_native_body')
            ref=descriptor.get('native_properties')
            status='unknown';equal={};differences={}
            if ref is not None:
                if row['offline_native_properties']!=ref:raise ValueError('offline_native_binding_changed')
                expected=read(resolve(ref))
                for key in required:
                    a=np.asarray(actual[key]);b=np.asarray(expected[key])
                    equal[key]=bool(a.shape==b.shape and np.array_equal(a,b))
                    differences[key]=float(np.max(np.abs(a-b))) if a.shape==b.shape else None
                status='confirmed' if all(equal.values()) else 'failed'
            unknown|=status=='unknown';failed|=status=='failed'
            results.append({'object':row['object'],'asset':row['asset'],'source_id':descriptor['source_id'],
                'source_name':descriptor['name'],'variant_id':descriptor.get('variant_id'),
                'dimensions_m':descriptor.get('dimensions_m'),'runtime_mass_kg':actual['mass_kg'],
                'status':status,'exact_native_field_matches':equal,'maximum_field_differences':differences})
        return {'status':'failed' if failed else 'unknown' if unknown else 'confirmed',
            'runtime_readback':binding(path),'rows':results,
            'scope':'Separate metric native-evidence dimension; original seven physical checks are unchanged.'}
    except Exception as error:
        return {'status':'failed','runtime_readback':binding(path),'rows':[],
            'error':type(error).__name__+':'+str(error)}


def render_evidence(case_id,source_binding,render_index):
    if case_id not in render_index:
        return {'status':'unknown','reason':'no_independent_render_review_supplied','full_pair_qualified':False}
    reference=render_index[case_id]
    try:
        report=read(resolve(reference))
        from resimpair_pickplace import render_lineage_validator_v5 as validator
        if report['validator']!=binding(validator.__file__):raise ValueError('unrecognized_or_changed_independent_render_validator')
        if report['source_native_execution']!=source_binding:raise ValueError('render_belongs_to_another_native_execution')
        # Resolve the original independently recorded provenance/camera/file
        # bindings; this does not run rendering or alter its visual thresholds.
        def verify_refs(value):
            if isinstance(value,dict):
                if 'path' in value and 'sha256' in value:resolve(value)
                else:
                    for item in value.values():verify_refs(item)
            elif isinstance(value,list):
                for item in value:verify_refs(item)
        verify_refs(report)
        return {'status':'confirmed' if report['passed'] else 'failed','review':reference,
            'selected_reference_visibility':report['pixel_readability']['status'],
            'all_multioperation_events_visible':'unknown_not_established_by_three_keyframes',
            'full_pair_qualified':False,'human_reference_qualification':'not_in_scope'}
    except Exception as error:
        return {'status':'failed','review':reference,'error':type(error).__name__+':'+str(error),
            'full_pair_qualified':False}


def publish(source,shadow,result_path):
    result=read(result_path)
    if not result['physical_qualified'] or result['runtime_native_mass_properties']['status']=='failed':return None
    files=[]
    for name in ('audit.json','container_audit.json','requested_goal_audit.json'):
        origin=shadow/name;target=source/name;payload=origin.read_bytes()
        if target.exists():
            if target.read_bytes()!=payload:raise ValueError('preserve_existing_source_audit:'+name)
        else:
            with target.open('xb') as f:f.write(payload)
        files.append({'filename':name,'independent_origin':binding(origin),'runtime_copy':binding(target),'byte_identical':True})
    bridge=source/'independent_metric_cli_audit_bridge_v1.json'
    value={'schema':'independent_metric_cli_evidence_bridge_v1','validator':binding(__file__),
        'independent_result':binding(result_path),'files':files,'physical_qualified':True,
        'metric_native_evidence':result['runtime_native_mass_properties']['status'],
        'full_pair_qualified':False,'all_audit_decisions_copied_without_modification':True,
        'new_physics_steps':0,'new_render_jobs':0,'API_calls':0}
    if bridge.exists():
        if read(bridge)!=value:raise ValueError('preserve_existing_metric_audit_bridge')
    else:write(bridge,value)
    return binding(bridge)


def review_case(cohort,case,output,multi,helper,render_index):
    source=cohort/'executions'/case['id'];shadow=output/case['id'];shadow.mkdir(parents=True,exist_ok=False)
    before={str(p.relative_to(source)):binding(p) for p in source.rglob('*') if p.is_file()}
    for p in source.iterdir():
        if p.name not in ('audit.json','container_audit.json','requested_goal_audit.json'):
            (shadow/p.name).symlink_to(p.resolve(),target_is_directory=p.is_dir())
    base={'case_id':case['id'],'physical_qualified':False,'metric_native_qualified':False,
        'full_pair_qualified':False,'native_simulation_record_qualified':False,'audit_error':None}
    try:
        expected={'case_id':case['id'],'source_request':case['source_request'],
            'requested_task_id':case['requested_task_id'],'dataset_partition':case['partition'],'experiment_role':cohort.name}
        physical=multi.import_retained().review(shadow,case,expected)
        base['retained_physical_review']=physical
        if not (source/'execution_bundle.json').exists():
            base.update(runtime_native_mass_properties={'status':'unknown','reason':'no_execution_bundle'},
                        RGB_synchronization={'status':'unknown','reason':'no_execution_bundle'})
        else:
            bundle=read(source/'execution_bundle.json');source_ref=binding(source/'execution_bundle.json')
            with np.load(resolve(bundle['measurements']),allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
            event_protocol=read(source/'event_protocol.json')
            from resimpair_pickplace.drop_event_protocol_v1 import pick_place
            from corl2026_workshop.experiments.autonomous_pipeline_v4.simulation import targets
            rebuilt=pick_place(bundle['scene'],bundle['task_intent'],bundle['object_ids'],bundle['support_ids'],targets(bundle['scene'],bundle['task_intent']))
            if canonical_protocol(rebuilt)!=canonical_protocol(event_protocol):raise ValueError('frozen_event_contract_differs_from_recorded_contract')
            event=helper.safe_compile(event_protocol,arrays);write(shadow/'new_event_audit.json',native_scalars(event))
            old_path=source/'retained_event_protocol.json';old_status='unknown_missing_retained_event_protocol'
            if old_path.exists():
                old=read(old_path)
                if old['goals']!=event_protocol['goals']:raise ValueError('original_final_goals_changed')
                old_event=helper.safe_compile(old,arrays);write(shadow/'retained_event_audit.json',native_scalars(old_event))
                old_status=bool(old_event['qualification']['passed'])
            paths=read(resolve(bundle['planned_paths']))['paths']
            operation=multi.original_operation_contract(read(resolve(case['task'])),bundle['task_intent'],
                [{'id':e['id'],'frame':e['frame']} for e in event['events']],paths)
            with np.load(resolve(read(source/'deferred_render_manifest.json')['physics_states']),allow_pickle=False) as z:
                commitments=multi.commitments(source,bundle,z['physics_states'])
            nested=multi.nested_planning_errors(source);prefixes=[]
            for path in paths:
                stage=path['stage'].split(':')[-1]
                if stage not in ('approach','grasp_reach'):continue
                q=arrays['states'][path['start_frame'],9*path['arm']:9*path['arm']+7]
                commands=np.asarray(path['positions']);delta=float(np.abs(commands[0]-q).max())
                guard=read(path['guard_evidence']);reused='reused_prefix' in guard
                if reused:
                    cert=read(resolve(guard['reused_prefix']))
                    np.testing.assert_array_equal(commands,cert['prefixes'][stage]['path']['position'])
                    if not guard['full_guard_repeated_without_threshold_change'] or not guard['report']['passed']:
                        raise ValueError('certified_prefix_original_guard_not_repeated')
                prefixes.append({'stage':path['stage'],'actual_start_max_error_rad':delta,
                    'retained_15mrad_bound_passed':delta<=.015,'certified_prefix_reused':reused})
            native=mass_evidence(source,bundle);rgb=render_evidence(case['id'],source_ref,render_index)
            integrity=bool(operation['passed'] and commitments['passed'] and not nested['integrity_failure']
                and all(p['retained_15mrad_bound_passed'] for p in prefixes))
            qualified=bool(physical['physical_subchecks_all_passed'] and integrity)
            base.update(source_execution=source_ref,requested_task_id=case['requested_task_id'],
                requested_operations=len(read(resolve(case['task']))['operations']),
                new_event_contract_passed=bool(event['qualification']['passed']),old_event_contract_passed=old_status,
                frozen_event_contract_integrity=True,
                final_and_release_measurements=helper.frame_statistics(bundle,arrays,event),
                initial_state=multi.initial_receipt(source,case),original_operation_contract=operation,
                preaction_commitments=commitments,nested_planning_errors=nested,measured_prefix_starts=prefixes,
                runtime_native_mass_properties=native,physical_qualified=qualified,
                metric_native_qualified=qualified and native['status']=='confirmed',RGB_synchronization=rgb,
                native_simulation_record_qualified=qualified and native['status']=='confirmed' and rgb['status']=='confirmed')
    except Exception as error:
        base['audit_error']=type(error).__name__+':'+str(error)
        base.setdefault('runtime_native_mass_properties',{'status':'unknown','reason':'review_stopped_on_integrity_error'})
        base.setdefault('RGB_synchronization',{'status':'unknown','reason':'review_stopped_on_integrity_error'})
    after={str(p.relative_to(source)):binding(p) for p in source.rglob('*') if p.is_file()}
    if before!=after:raise ValueError('source_changed_during_independent_review')
    base['source_files_unchanged_during_review']=True
    write(shadow/'independent_result.json',native_scalars(base));return base


def run(cohort,output,*,render_reviews=None,publish_native_audits=False):
    cohort=Path(cohort).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('independent_output_must_be_new')
    if output==cohort or output.is_relative_to(cohort):raise ValueError('independent_output_must_be_outside_source_cohort')
    if binding(FROZEN)['sha256']!=FROZEN_SHA256:raise ValueError('independent_acceptance_config_changed')
    frozen=read(FROZEN)
    for ref in frozen['dependencies'].values():resolve(ref)
    if not (cohort/'completion.json').exists():raise ValueError('all_registered_attempts_must_be_terminal_before_cohort_review')
    from resimpair_pickplace.runner import verify
    verify(cohort)
    for ref in read(cohort/'frozen_inputs.json').values():resolve(ref)
    if read(cohort/'rules.json')!=frozen['retained_rules']:raise ValueError('cohort_physical_rules_differ_from_frozen_acceptance')
    cases=read(cohort/'cohort.json')['cases'];ids=[c['id'] for c in cases]
    if not ids or len(ids)!=len(set(ids)):raise ValueError('nonempty_unique_case_denominator_required')
    for case in cases:
        if not (cohort/'executions'/case['id']/'terminal.json').exists():raise ValueError('registered_case_missing_terminal:'+case['id'])
    index={}
    if render_reviews is not None:
        manifest=read(render_reviews)
        if manifest.get('schema')!='independent_metric_render_review_index_v1':raise ValueError('render_review_index_schema_required')
        index=manifest['case_reviews']
        if not set(index)<=set(ids):raise ValueError('render_index_contains_foreign_case')
    multi=load('metric_cli_retained_multitask',MULTI);helper=load('metric_cli_retained_event_statistics',HELPER)
    write(output/'PROTOCOL.json',{'schema':VERSION,'validator':binding(__file__),'frozen_acceptance':binding(FROZEN),
        'cohort':binding(cohort/'cohort.json'),'execution_protocol':binding(cohort/'protocol.json'),
        'render_review_index':binding(render_reviews) if render_reviews else None,'publish_native_audits':publish_native_audits,
        'retained_evaluation_dependencies':frozen['dependencies'],'rules_relaxed':False,'GPU_jobs':0,'physical_attempts_by_audit':0,'API_calls':0})
    rows=[]
    for case in cases:
        result=review_case(cohort,case,output,multi,helper,index);rows.append(result)
        print(case['id'],'physical',result['physical_qualified'],'native',result['runtime_native_mass_properties']['status'],
            'error',result['audit_error'],flush=True)
    publication=[]
    if publish_native_audits:
        for result in rows:
            shadow=output/result['case_id'];source=cohort/'executions'/result['case_id']
            ref=publish(source,shadow,shadow/'independent_result.json')
            if ref:publication.append(ref)
    report={'schema':VERSION,'protocol':binding(output/'PROTOCOL.json'),'cohort_directory':str(cohort),
        'requested_attempts':len(rows),'actual_physical_attempts':sum(r.get('retained_physical_review',{}).get('physical_attempts',0) for r in rows),
        'physical_qualified':sum(r['physical_qualified'] for r in rows),'metric_native_qualified':sum(r['metric_native_qualified'] for r in rows),
        'native_simulation_records_qualified':sum(r['native_simulation_record_qualified'] for r in rows),
        'full_pairs_qualified':0,'audit_error_count':sum(r['audit_error'] is not None for r in rows),'rows':rows,
        'published_native_audit_bridges':publication,'rules_relaxed':False,'new_physics_attempts_by_audit':0,'GPU_jobs':0,'API_calls':0,
        'scope':'Original seven physical dimensions and requested goals; runtime metric native properties and independently audited RGB remain separate. Human-reference qualification is not performed.'}
    write(output/'FINAL_REVIEW.json',native_scalars(report));return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--render-reviews',type=Path,help='Bound independent v5 reviews indexed by case ID; no rendering is launched.')
    parser.add_argument('--publish-native-audits',action='store_true',help='Publish unchanged compatibility reports for qualified executions, refusing conflicts.')
    args=parser.parse_args();report=run(args.cohort,args.output,render_reviews=args.render_reviews,publish_native_audits=args.publish_native_audits)
    print({k:v for k,v in report.items() if isinstance(v,(int,bool))})
    if report['audit_error_count']:sys.exit(2)


if __name__=='__main__':main()
