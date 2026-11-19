import argparse
import json
from pathlib import Path
from . import batch,catalog
from resimpair_grasp.common import ROOT


def main():
    p=argparse.ArgumentParser(description='Full-library role retrieval and bounded asset preparation (no paid API)')
    s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('index');q.add_argument('--source',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    q.add_argument('--policy',type=Path,default=ROOT/'configs/asset_roles_v4.json')
    for name in ['search','plan']:
        q=s.add_parser(name);q.add_argument('--index',type=Path,required=True)
        q.add_argument('--category',action='append',default=[]);q.add_argument('--source',action='append',default=[])
        q.add_argument('--role',choices=['static','manipulation'],default='static')
        q.add_argument('--partition',choices=['development','heldout_asset']);q.add_argument('--query');q.add_argument('--limit',type=int,default=20)
        if name=='plan':
            q.add_argument('--output',type=Path,required=True);q.add_argument('--store',type=Path,default=ROOT/'assets/expanded_library/cache_v1')
            q.add_argument('--seed',type=int,default=2026102601);q.add_argument('--timeout',type=int,default=60)
            q.add_argument('--workers',type=int,default=4);q.add_argument('--disk-mib',type=int,default=2048)
            q.add_argument('--exclude-catalog',type=Path,action='append',default=[])
    q=s.add_parser('run');q.add_argument('--root',type=Path,required=True);q.add_argument('--limit',type=int,default=8)
    q=s.add_parser('report');q.add_argument('--root',type=Path,required=True)
    q=s.add_parser('worker');q.add_argument('--root',type=Path,required=True);q.add_argument('--key',required=True);q.add_argument('--phase',choices=['prepare','validate'],required=True)
    q=s.add_parser('merge');q.add_argument('--catalog',type=Path,action='append',required=True);q.add_argument('--output',type=Path,required=True)
    q=s.add_parser('relabel');q.add_argument('--catalog',type=Path,required=True);q.add_argument('--index',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='index':result=catalog.build(a.source,a.policy,a.output)
    elif a.command=='search':
        result=[{k:r[k] for k in ['id','name','source','category','partition','bytes','classification']} for r in catalog.search(a.index,category=a.category,source=a.source,role=a.role,partition=a.partition,query=a.query,limit=a.limit)]
    elif a.command=='plan':result=batch.plan(a)
    elif a.command=='run':result=batch.run(a)
    elif a.command=='report':result=batch.report(a.root)
    elif a.command=='worker':result=batch.worker(a)
    elif a.command=='merge':result=batch.merge(a.catalog,a.output)
    elif a.command=='relabel':result=batch.relabel(a.catalog,a.index,a.output)
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
