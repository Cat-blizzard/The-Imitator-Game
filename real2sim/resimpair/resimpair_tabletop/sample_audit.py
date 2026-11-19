"""Frozen grasp/lift outcome, separate from placement success and offline poses."""
from pathlib import Path
import argparse,json,math
import numpy as np
from resimpair_grasp.common import binding,read,write,resolve

VERSION='tabletop_sample_grasp_lift_v1'
RULES={'minimum_lift_m':.12,'minimum_contiguous_hold_seconds':.12,
       'maximum_one_frame_translation_m':.04,'requires_measured_two_finger_grasp':True,
       'requires_independent_path_guard':True,'full_pick_place_qualified_is_separate':True}


def lift_measurements(arrays,arm,index):
    times=arrays['timestamps_seconds'];xyz=arrays['object_states'][:,index,:3]
    held=arrays['grasp'][:,arm,index].astype(bool)
    lifted=xyz[:,2]-xyz[0,2]>=RULES['minimum_lift_m'];valid=held&lifted
    best=0.;start=None
    for i,okay in enumerate(valid):
        if okay:
            if start is None:start=i
            best=max(best,float(times[i]-times[start]))
        else:start=None
    jumps=np.linalg.norm(np.diff(xyz,axis=0),axis=1)
    return {'passed':best>=RULES['minimum_contiguous_hold_seconds'] and (not len(jumps) or jumps.max()<=RULES['maximum_one_frame_translation_m']),
        'max_contiguous_measured_grasp_and_lift_seconds':best,'max_lift_m':float((xyz[:,2]-xyz[0,2]).max()),
        'maximum_frame_translation_m':float(jumps.max()) if len(jumps) else 0.}


def wilson(k,n):
    if not n:return None
    z=1.95996398454;p=k/n;d=1+z*z/n;center=(p+z*z/(2*n))/d
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0.,center-half),min(1.,center+half)]


def audit(sample,campaign,output):
    if output.exists():raise ValueError('Sample audit exists')
    frozen=read(sample/'outcome_rules.json')
    if frozen['version']!=VERSION or frozen['rules']!=RULES:raise ValueError('Sample outcome rules changed')
    resolve(frozen['implementation'])
    plan=read(sample/'plan.json');cases=read(campaign/'cohort.json')['cases'];by_source={}
    for c in cases:
        if not c['id'].endswith('tray_place'):continue
        sid=read(resolve(c['asset']))['source_id'];by_source[sid]=c
    rows=[]
    for selected in plan['sample']:
        row={'source_id':selected['source_id'],'partition':selected['partition'],'source':selected['source'],
            'offline_asset':selected['asset'],'native_executed':False,'grasp_lift_passed':False,'full_task_passed':False}
        terminal=read(sample/'materialized'/selected['id']/'terminal.json')
        if terminal['status']!='native_geometry_ready':
            row.update(failure_stage='native_geometry_materialization',reason=terminal.get('reason'));rows.append(row);continue
        case=by_source.get(selected['source_id']);directory=campaign/'executions'/case['execution_id'] if case else None
        if directory is None or not (directory/'execution_bundle.json').exists():
            row.update(failure_stage='native_execution_missing');rows.append(row);continue
        from resimpair_grasp.evaluator import evaluate
        result=evaluate(directory);b=read(directory/'execution_bundle.json');op=b['task_intent']['operations'][0]
        with np.load(resolve(b['measurements']),allow_pickle=False) as f:data={k:f[k] for k in f.files}
        measured=lift_measurements(data,op['arm'],b['object_ids'].index(op['object']))
        guarded=result.get('checks',{}).get('executed_paths_guarded',{}).get('passed',False)
        robot_checks=[r for r in result.get('checks',{}).get('task_events',{}).get('checks',[]) if r['id']=='constraint:robot_collision']
        safe_contact=bool(robot_checks) and all(r['passed'] for r in robot_checks)
        row.update(native_executed=True,grasp_lift_passed=bool(measured['passed'] and guarded and safe_contact),
            full_task_passed=result['passed'],measured=measured,paths_guarded=guarded,robot_contact_safe=safe_contact,
            independent_native_audit=binding(directory/'audit.json'),execution=binding(directory/'execution_bundle.json'),
            planning_failure=b['planning_failure'])
        rows.append(row)
    passed=sum(r['grasp_lift_passed'] for r in rows);n=len(rows);executed=sum(r['native_executed'] for r in rows)
    report={'schema':'tabletop_sample_audit_v1','version':VERSION,'rules':RULES,'sample_plan':binding(sample/'plan.json'),'rows':rows,
        'sample_size':n,'native_executed':executed,'grasp_lift_passed':passed,'full_pick_place_passed':sum(r['full_task_passed'] for r in rows),
        'sample_end_to_end_yield':passed/n if n else None,'grasp_lift_given_execution':passed/executed if executed else None,
        'wilson_95_interval_on_sample_yield':wilson(passed,n) if plan['sampling']=='random' else None,
        'population_inference_scope':'registered random sample only; preprocessing failures retained' if plan['sampling']=='random' else 'diagnostic source-balanced sample, no global rate inference',
        'all_unsampled_assets_physical_status':'unmeasured','api_calls':0}
    write(output,report);return {k:report[k] for k in ['sample_size','native_executed','grasp_lift_passed','full_pick_place_passed','sample_end_to_end_yield','wilson_95_interval_on_sample_yield']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('freeze');q.add_argument('--sample',type=Path,required=True)
    q=s.add_parser('audit');q.add_argument('--sample',type=Path,required=True);q.add_argument('--campaign',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='freeze':
        write(a.sample/'outcome_rules.json',{'version':VERSION,'rules':RULES,'implementation':binding(__file__),'before_native_execution':True});result={'frozen':True}
    else:result=audit(a.sample,a.campaign,a.output)
    print(json.dumps(result,indent=2))
