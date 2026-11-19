"""Plan a disjoint next batch; reuse earlier evidence without reprocessing assets."""
import argparse
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from . import bulk
from resimpair_grasp.common import read, write, binding, resolve


def previous_identities(previous):
    seen=set();parents=[]
    for catalog in previous:
        data=read(catalog);plan_path=resolve(data['plan']);plan=read(plan_path)
        if data['attempted']!=data['scheduled'] or data['counts'].get('pending',0):
            raise ValueError('Previous batch is incomplete; recover it before extending')
        rows={row['id']:row for row in data['rows']}
        for job in plan['jobs']:
            if job['id'] not in rows:raise ValueError('Previous result is missing a planned identity')
            resolve(rows[job['id']]['terminal'])
            seen.add(job['source_sha256'])
        # Parents form a chain of immutable snapshots, including failed searches.
        inherited=plan.get('reused_parent_catalogs',[])
        if inherited:
            identities, _=previous_identities([resolve(b) for b in inherited]);seen.update(identities)
        parents.append(binding(catalog))
    return seen,parents


def plan_extension(previous, root, limit):
    if root.exists():raise ValueError('Extension exists; resume instead of replacing')
    if not 1<=limit<=60000:raise ValueError('Finite next-batch limit required')
    seen,parents=previous_identities(previous)
    first=read(resolve(read(previous[0])['plan']))
    # Temporary planning contains metadata only. No mesh processing or physics.
    with tempfile.TemporaryDirectory(prefix='resimpair_tabletop_plan_') as tmp:
        temporary=Path(tmp)/'proposal'
        args=SimpleNamespace(root=temporary,index=resolve(first['source_index']),profile=resolve(first['profile_binding']),
            limit=60000,workers=first['workers'],timeout=first['timeout_seconds'],
            disk_gib=first['output_limit_bytes']//2**30,wall_seconds=first['wall_seconds_per_run'],
            seed=first['seed'],partition=None)
        bulk.plan(args);full=read(temporary/'plan.json')
        remaining=[job for job in full['jobs'] if job['source_sha256'] not in seen]
        selected=remaining[:limit]
        if not selected:raise ValueError('No unprocessed mesh identities remain')
        full.update(jobs=selected,maximum_assets=len(selected),reused_parent_catalogs=parents,
                    previous_processed_identities=len(seen),remaining_before_this_batch=len(remaining),
                    extension_policy='Exclude all completed identities, including negative/failed outcomes; no automatic retries')
        root.mkdir(parents=True)
        shutil.copytree(temporary/'release',root/'release')
        write(root/'plan.json',full)
        write(root/'execution_config.json',{k:v for k,v in full.items() if k!='jobs'})
        for i,job in enumerate(selected):write(root/'jobs'/f'{i:06d}.json',job)
    return {'root':str(root),'scheduled':len(selected),'previous_identities_not_repeated':len(seen),
            'remaining_after_this_batch':len(remaining)-len(selected),'physics_executions':0,'api_calls':0}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous',type=Path,action='append',required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=12000)
    args=parser.parse_args()
    print(json.dumps(plan_extension(args.previous,args.root,args.limit),indent=2))
