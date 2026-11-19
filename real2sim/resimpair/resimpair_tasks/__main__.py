import argparse
import json
from pathlib import Path
from . import factory,programs


def main():
    p=argparse.ArgumentParser(description='Reusable asset/scene/task programs with bounded native execution')
    s=p.add_subparsers(dest='command',required=True);s.add_parser('capabilities')
    q=s.add_parser('init');q.add_argument('--root',type=Path,required=True);q.add_argument('--catalog',type=Path,required=True)
    q.add_argument('--cache-index',type=Path,action='append',required=True);q.add_argument('--partition',default='development',choices=['development','heldout_asset'])
    q.add_argument('--category',action='append',default=[]);q.add_argument('--scenes',type=int,default=2)
    q.add_argument('--recipe',type=Path)
    q.add_argument('--families',nargs='+',default=['select','clear','arrange','ordered']);q.add_argument('--seed',type=int,default=2026102631)
    q.add_argument('--gpu',type=int,default=2);q.add_argument('--physical-limit',type=int,default=12)
    for name in ['run','audit']:
        q=s.add_parser(name);q.add_argument('--root',type=Path,required=True)
        if name=='run':q.add_argument('--limit',type=int,default=1);q.add_argument('--case',action='append',default=[])
    a=p.parse_args()
    if a.command=='capabilities':result=programs.CAPABILITIES
    elif a.command=='init':result=factory.initialize(a)
    else:result=getattr(factory,a.command)(a)
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
