import argparse,json
from pathlib import Path
from resimpair_grasp.common import ROOT,write


def main():
    p=argparse.ArgumentParser(description='Bulk offline tabletop poses; simulator probes are a separate sample')
    s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('plan');q.add_argument('--root',type=Path,required=True);q.add_argument('--index',type=Path,default=ROOT/'assets/expanded_library/raw_roles_v4.sqlite')
    q.add_argument('--profile',type=Path,default=ROOT/'configs/tabletop_bulk_v1.json');q.add_argument('--limit',type=int,default=10000)
    q.add_argument('--workers',type=int,default=16);q.add_argument('--timeout',type=int,default=90);q.add_argument('--disk-gib',type=int,default=64)
    q.add_argument('--wall-seconds',type=int,default=3600);q.add_argument('--seed',type=int,default=2026102608)
    q.add_argument('--partition',choices=['development','heldout_asset'])
    for name in ['run','report','worker']:
        q=s.add_parser(name);q.add_argument('--root',type=Path,required=True)
        if name=='run':q.add_argument('--limit',type=int,default=10000)
        if name=='worker':q.add_argument('--number',type=int,required=True);q.add_argument('--slot',type=int,required=True)
    a=p.parse_args()
    from . import bulk
    if a.command=='plan':result=bulk.plan(a)
    elif a.command=='run':result=bulk.run(a)
    elif a.command=='report':result=bulk.report(a.root)
    else:
        result=bulk.worker(a);job=bulk.read(a.root/'jobs'/f'{a.number:06d}.json')
        write(a.root/'assets'/job['id']/'worker_result.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
