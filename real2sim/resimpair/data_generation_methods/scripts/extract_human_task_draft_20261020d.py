"""Prepare blind real-human window observations; --extract makes one cached call."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from data_generation_methods.r2h_pipeline.human_task_draft import prepare_packet,extract_draft
from data_generation_methods.r2h_pipeline.common import load_env_file


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--video",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--start",type=int,default=0)
    p.add_argument("--end",type=int)
    p.add_argument("--frame-budget",type=int,default=24)
    p.add_argument("--extract",action="store_true")
    p.add_argument("--config",type=Path,default=Path(__file__).resolve().parents[1]/"config.echoflow.example.json")
    p.add_argument("--credentials",type=Path,default=Path("/home/xzzhou/.config/policygen/credentials/echoflow.env"))
    a=p.parse_args()
    if a.video:
        packet=prepare_packet(a.video,a.output,start=a.start,end=a.end,frame_budget=a.frame_budget)
        print(json.dumps({"prepared_frames":len(packet["frames"]),"output":str(a.output),"paid_calls":0}))
    if a.extract:
        load_env_file(a.credentials)
        draft=extract_draft(a.output,json.loads(a.config.read_text()))
        print(json.dumps({"status":draft["contract_status"],"output":str(a.output)}))
    if not a.video and not a.extract:p.error("Provide --video and/or --extract")


if __name__=="__main__":main()
