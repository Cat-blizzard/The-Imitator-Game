"""Poll a possible existing task after ambiguous submission; never POST/retry."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from data_generation_methods.r2h_pipeline.providers import SeedanceClient, ProviderError
from data_generation_methods.r2h_pipeline.common import load_env_file, write_json
from data_generation_methods.r2h_pipeline.cli import download_video


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task-id',required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--minutes',type=float,default=20)
    args=p.parse_args()
    load_env_file('/home/xzzhou/.config/policygen/credentials/echoflow.env')
    cfg=json.loads((Path(__file__).resolve().parents[1]/'config.echoflow.example.json').read_text())['seedance']
    cfg['timeout']=30
    client=SeedanceClient(cfg)
    directory=args.root/'seedance_recovery';directory.mkdir(exist_ok=True)
    deadline=time.monotonic()+args.minutes*60
    state={}
    while time.monotonic()<deadline:
        try:
            result=client.poll(args.task_id)
            write_json(directory/f'{time.time_ns()}.json',result)
            state={k:v for k,v in result.items() if k!='output_url'}
            state.update(candidate_task_binding='unconfirmed: gateway omitted timestamp and input echo',
                updated_unix=time.time(),no_POST_or_resubmission=True,training_admitted=False)
            if result['status']=='succeeded':
                media=download_video(result['output_url'],directory/'candidate_generated_human.mp4',
                    task_id=args.task_id,request_sha256=None)
                state['candidate_media']=media
                state['candidate_task_binding']='requires comparison of downloaded content to our submitted input'
            write_json(directory/'state.json',state,overwrite=True)
            print(json.dumps(state,ensure_ascii=False),flush=True)
            if result['status'] in {'succeeded','failed','expired','cancelled'}:return
        except ProviderError as exc:
            state=dict(status='read_only_query_failed',task_id=args.task_id,code=exc.code,
                http_status=exc.http_status,updated_unix=time.time(),no_POST_or_resubmission=True)
            write_json(directory/'state.json',state,overwrite=True)
            print(json.dumps(state),flush=True)
            if exc.http_status in {401,403,404}:return
        time.sleep(min(60,max(0,deadline-time.monotonic())))
    state.update(watcher_status='bounded_poll_window_ended',no_POST_or_resubmission=True)
    write_json(directory/'state.json',state,overwrite=True)


if __name__=='__main__':main()
