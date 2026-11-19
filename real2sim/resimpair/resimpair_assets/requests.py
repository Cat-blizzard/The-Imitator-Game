"""Resolve a task's asset demand from prepared data, preparing only shortages."""
from pathlib import Path
from types import SimpleNamespace
import hashlib

from resimpair_grasp.common import binding,read,resolve,write
from resimpair_grasp.assets import select
from . import batch


def fulfill(a):
    if a.output.exists():raise ValueError('Request output exists; inspect it instead of resubmitting')
    if not 1<=a.count<=256 or not 1<=a.preparation_limit<=256:raise ValueError('Finite request/preparation limits required')
    role='static_distractor' if a.role=='static' else 'rigid_manipulation_candidate'
    candidates=[]
    for b,d in select(a.catalog,role=role,max_dimension=a.max_dimension):
        if a.category and d['category'] not in a.category:continue
        if a.category and d.get('semantic_role_eligible') is False:continue
        if a.partition and d['partition'] not in {a.partition,'shared_distractor'}:continue
        candidates.append((b,d))
    candidates.sort(key=lambda p:hashlib.sha256(f"{a.seed}:{p[1]['source_id']}".encode()).hexdigest())
    chosen=candidates[:a.count];a.output.mkdir(parents=True)
    write(a.output/'request.json',{'schema':'asset_role_request_v1','catalog':binding(a.catalog),'role':a.role,
          'categories':a.category,'count':a.count,'max_dimension_m':a.max_dimension,'partition':a.partition,
          'prepare_missing':a.prepare_missing,'maximum_new_imports':a.preparation_limit,'api_limit':0,
          'semantics':'metadata retrieval; visual semantic certainty is not inferred'})
    preparation=None;failure=None
    shortage=a.count-len(chosen)
    if shortage and a.prepare_missing:
        # One finite attempt list. Failures do not trigger unbounded replenishment.
        options=SimpleNamespace(output=a.output/'preparation',index=a.index,store=a.store,
             role=a.role,limit=min(shortage,a.preparation_limit),workers=a.workers,timeout=a.timeout,
             disk_mib=a.disk_mib,seed=a.seed,category=a.category,source=[],partition=a.partition,
             query=None,exclude_catalog=[a.catalog])
        try:
            batch.plan(options)
            preparation=batch.run(SimpleNamespace(root=options.output,limit=options.limit))
            for b,d in select(Path(preparation['catalog']),role=role,max_dimension=a.max_dimension):
                if len(chosen)<a.count:chosen.append((b,d))
        except (ValueError,FileNotFoundError) as e:failure=repr(e)
    result={'schema':'resolved_asset_role_request_v1','requested':a.count,'resolved':len(chosen),
       'shortage':a.count-len(chosen),'status':'resolved' if len(chosen)==a.count else 'partial_or_unavailable',
       'assets':[b for b,d in chosen],'reused_existing':min(len(candidates),a.count),
       'preparation':preparation,'preparation_error':failure,'api_calls':0,
       'qualification_scope':'static eligibility' if a.role=='static' else 'geometry candidate; grasp cache and physical task qualification still required',
       'semantic_status':'metadata_candidate','physical_execution_success_inferred':False}
    write(a.output/'resolved.json',result)
    return result
