"""Prepare or explicitly execute new-content repairs; never retry unknown old POSTs."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from data_generation_methods.r2h_pipeline.posthoc_controlled_repair import (
    BRANCH_NAME, prepare_branch, execute, accept_terminal, aggregate_counts,
)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation",choices=("prepare","image","accept-terminal","video","poll","review","counts"))
    parser.add_argument("--root",type=Path,default=Path("/mnt/sdg/datasets/real_sim_human_pipeline/controlled_20261020e"))
    parser.add_argument("--case",choices=("apple","cabinet"),required=True)
    parser.add_argument("--arm",choices=("A","B"),required=True)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--reviewer")
    parser.add_argument("--reason")
    args=parser.parse_args()
    branch=args.root/args.case/args.arm/BRANCH_NAME
    if args.operation=="prepare":
        result=prepare_branch(args.root,args.case,args.arm)
    elif args.operation=="counts":
        result={"combined":aggregate_counts(branch.parent),"branch":aggregate_counts(branch)}
    elif args.operation=="accept-terminal":
        if not args.reviewer or not args.reason:
            parser.error("accept-terminal requires --reviewer and --reason")
        result=accept_terminal(branch,reviewer=args.reviewer,reason=args.reason)
    else:
        if args.operation!="poll" and not args.execute:
            parser.error("Paid operations require --execute after explicit authorization")
        result=execute(branch,args.operation)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
