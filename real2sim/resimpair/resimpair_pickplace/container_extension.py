"""Disjoint metadata extension of a completed container batch, no retries."""
import argparse
import json
from pathlib import Path
import tempfile
from .containers import plan
from resimpair_grasp.common import read,resolve,binding,write


def extend(previous,output,count):
    if output.exists():raise ValueError('Existing extension')
    old=read(previous);p=read(resolve(old['plan']))
    if len(old['rows'])!=len(p['jobs']):raise ValueError('Prior batch incomplete')
    seen={r['source_sha'] for r in p['jobs']}
    with tempfile.TemporaryDirectory(prefix='container_metadata_') as tmp:
        root=Path(tmp)/'plan';plan(resolve(p['index']),root,2000,p['seed'])
        full=read(root/'plan.json');jobs=[r for r in full['jobs'] if r['source_sha'] not in seen][:count]
        full.update(jobs=jobs,reused_parent=binding(previous),previous_not_repeated=len(seen))
        output.mkdir(parents=True);write(output/'plan.json',full)
    return {'root':str(output),'new_jobs':len(jobs),'old_jobs_not_repeated':len(seen)}


def merge(catalogs,output):
    assets=[];rows=[];seen=set()
    for p in catalogs:
        c=read(p)
        for r in c['rows']:
            if r['source_id'] in seen:raise ValueError('Repeated container identity')
            seen.add(r['source_id']);rows.append(r)
        assets.extend(c['assets'])
    write(output,{'schema':'stationary_container_catalog_v1','assets':assets,'rows':rows,
                  'parents':[binding(p) for p in catalogs],'physics_executions':0,'api_calls':0})
    return {'attempted':len(rows),'geometry_ready':len(assets),'catalog':str(output)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('extend');q.add_argument('--previous',type=Path,required=True);q.add_argument('--output',type=Path,required=True);q.add_argument('--count',type=int,default=384)
    q=s.add_parser('merge');q.add_argument('--catalog',type=Path,action='append',required=True);q.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=extend(a.previous,a.output,a.count) if a.command=='extend' else merge(a.catalog,a.output)
    print(json.dumps(r,indent=2))
