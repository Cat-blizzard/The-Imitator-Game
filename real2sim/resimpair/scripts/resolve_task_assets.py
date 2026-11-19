#!/usr/bin/env python3
"""Resolve role/category/count requests, reusing prepared assets first."""
from pathlib import Path
import sys,argparse,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from resimpair_runtime import configure,ROOT
configure()
from resimpair_assets.requests import fulfill

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--index',type=Path,default=ROOT/'assets/expanded_library/raw_roles_v4.sqlite')
    p.add_argument('--store',type=Path,default=ROOT/'assets/expanded_library/on_demand_cache')
    p.add_argument('--role',choices=['static','manipulation'],default='static');p.add_argument('--category',action='append',default=[])
    p.add_argument('--count',type=int,default=6);p.add_argument('--max-dimension',type=float,default=.18)
    p.add_argument('--partition',choices=['development','heldout_asset']);p.add_argument('--seed',type=int,default=20261026)
    p.add_argument('--prepare-missing',action='store_true');p.add_argument('--preparation-limit',type=int,default=8)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--timeout',type=int,default=120);p.add_argument('--disk-mib',type=int,default=512)
    print(json.dumps(fulfill(p.parse_args()),indent=2,ensure_ascii=False))
