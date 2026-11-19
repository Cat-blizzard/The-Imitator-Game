"""Execute an independently reviewed subset without rewriting its frozen cohort.

Original preparation failures stay allocated and visible; they receive neither
a fabricated physical terminal nor a physical reservation. The original v2
worker, admission check, locks, timeout and cumulative ledger are retained.
"""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from resimpair_grasp.common import ROOT,binding,read,resolve,write
from resimpair_scene import execution
from resimpair_tasks.cudex_bulk_queue_v2 import process_identity


def checked_selection(cohort,review_path):
    cohort=Path(cohort).resolve();review=read(review_path)
    if (review.get('prepared_cohort')!=binding(cohort/'cohort.json')
            or review.get('prepared_execution_protocol')!=binding(cohort/'protocol.json')
            or review.get('exact_case_bindings_passed') is not True
            or review.get('unchanged_native_physics_acceptance') is not True):
        raise ValueError('independent_exact_cohort_and_rule_review_required')
    for key in ('independent_protocol','release_geometry_review','producer_chain_review'):resolve(review[key])
    cases=read(cohort/'cohort.json')['cases'];all_ids=[c['id'] for c in cases]
    good=review['geometry_eligible_case_ids'];bad=review['preexecution_contract_failure_case_ids']
    if (len(set(good))!=len(good) or len(set(bad))!=len(bad) or set(good)&set(bad)
            or set(good)|set(bad)!=set(all_ids) or good!=[cid for cid in all_ids if cid in good]):
        raise ValueError('review_must_partition_exact_original_cohort_without_reordering')
    return [c for c in cases if c['id'] in good],review


def run(cohort,review_path,limit):
    from resimpair_pickplace.runner import verify
    from resimpair_tasks.multitask_campaign_v3 import admission_check
    cohort=Path(cohort).resolve();cases,review=checked_selection(cohort,review_path)
    if type(limit) is not int or not 1<=limit<=len(cases):raise ValueError('finite_selected_launch_limit_required')
    protocol=read(cohort/'protocol.json');campaign=Path(protocol['campaign_root'])
    _,limits,refs=execution._campaign(campaign);execution._cohort(cohort,campaign,refs)
    selected_profile=read(cohort/'scene_execution.json')
    profile=read(resolve(selected_profile['request']))['profile']
    if profile['name']!='task_release' or profile['module']!='resimpair_tasks.production_integration_v2':
        raise ValueError('this_subset_supervisor_only_delegates_original_task_release_worker')
    verify(cohort)
    for ref in read(cohort/'frozen_inputs.json').values():resolve(ref)
    selection={'independent_review':binding(review_path),'cohort':binding(cohort/'cohort.json'),
        'implementation':binding(__file__),'worker':protocol['driver'],
        'eligible_case_ids':[c['id'] for c in cases],
        'preserved_preflight_failure_ids':review['preexecution_contract_failure_case_ids'],
        'physical_rules_changed':False,'allocated_requests_refunded':False}
    write(cohort/'selected_execution.json',selection)
    shared_path=ROOT/'runtime/multitask_campaign_registry_v3/.execution.lock';rows=[]
    with shared_path.open('a+') as shared,(campaign/'.supervisor.lock').open('a+') as owned:
        fcntl.flock(shared,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(owned,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status=execution.campaign_status(campaign)
        if status['unknown_or_active_attempts'] or status['admission_stopped']:
            raise ValueError('recover_unknown_or_stopped_campaign_before_submission')
        for case in cases:
            if len(rows)>=limit:break
            directory=cohort/'executions'/case['id']
            if (directory/'terminal.json').exists():continue
            if (directory/'reservation.json').exists():raise ValueError('unknown_attempt_never_resubmit')
            used=len(list((campaign/'budget/reservations').glob('*.json')))
            if used>=limits['physical_attempt_limit']:raise ValueError('original_physical_budget_exhausted')
            reservation={'case':case,'cohort':cohort.name,'campaign':str(cohort),
                'protocol':binding(cohort/'protocol.json'),'started_unix':time.time(),
                'independent_subset_selection':binding(cohort/'selected_execution.json')}
            write(campaign/'budget/reservations'/f'{used:04d}.json',reservation)
            write(directory/'reservation.json',reservation)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='none',
                VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/lvp_icd.x86_64.json',
                OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(ROOT),
                RESIMPAIR_HEADLESS_RESERVATION=str(directory/'reservation.json'))
            start=time.monotonic()
            with (directory/'worker.log').open('x') as log:
                child=subprocess.Popen([sys.executable,'-u','-m',profile['module'],'worker',
                    '--root',str(cohort),'--case',case['id']],cwd=ROOT,env=env,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                write(directory/'process.json',process_identity(child.pid))
                try:rc=child.wait(timeout=limits['timeout_seconds'])
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=8)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    rc='timeout'
            check=admission_check(directory,rc)
            result={'case':case['id'],'cohort':cohort.name,'physical_attempts':1,'returncode':rc,
                'elapsed_seconds':time.monotonic()-start,'admission':check,'API_calls':0,'GPU_jobs':0}
            write(directory/'terminal.json',result);rows.append(result)
            print({'case':case['id'],'seconds':round(result['elapsed_seconds'],3),
                'failure':check['reported_failure']},flush=True)
            if check['stop_admission']:write(campaign/'admission_stop.json',result);break
    if all((cohort/'executions'/c['id']/'terminal.json').exists() for c in cases):
        write(cohort/'selected_completion.json',{'selection':binding(cohort/'selected_execution.json'),
            'eligible_requests':len(cases),'all_selected_attempts_terminal':True,
            'unsubmitted_preflight_failures':review['preexecution_contract_failure_case_ids']})
    return {'new_attempts':len(rows),'selected_scope':len(cases),
        'original_cohort_requests':len(read(cohort/'cohort.json')['cases']),
        'campaign_status':execution.campaign_status(campaign)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort',type=Path,required=True);p.add_argument('--review',type=Path,required=True)
    p.add_argument('--limit',type=int,required=True);a=p.parse_args()
    print(run(a.cohort,a.review,a.limit))
