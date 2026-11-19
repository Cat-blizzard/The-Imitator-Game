"""Bounded, one-request-at-a-time real API smoke tests; never starts training."""
import argparse
import base64
import io
import json
from pathlib import Path
import re
import sys
import time
from urllib.error import HTTPError

WORKSPACE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(WORKSPACE))
from data_generation_methods.r2h_pipeline.common import load_env_file, file_sha256, write_json
from data_generation_methods.r2h_pipeline.providers import GeminiClient, SeedanceClient, ProviderError, _urllib_transport
from data_generation_methods.r2h_pipeline.cli import download_video

ROOT=Path('/mnt/sdg/datasets/real_sim_human_pipeline/pilot_20261020')
CONFIG=WORKSPACE/'data_generation_methods/config.echoflow.example.json'
HUMAN_SPEC=Path('/mnt/sdf/datasets/hypericl_human_robot_task_library_20261019_v1/generation/prepared_v2/human_grounded_task_spec.json')
ANCHOR=WORKSPACE/'data_generation_methods/pilot_inputs/simx_real_bowl_transfer_891911000_pick_first_to_right/first_frame000000.png'
BLIND_PROMPT='''Describe only visible evidence in this video; never infer an intended task from a filename or unseen instruction. Return JSON with keys: objects (stable descriptive identities), initial_relations, events (each has start_seconds,end_seconds,actor,object,action,visible_evidence), final_relations, uncertain_observations, task_description. Explicitly say unknown when grasp/contact/release/support is occluded. Distinguish moving a container from moving its contents. No numeric confidence is needed. Include whether hands withdraw and objects remain stationary before video ends. The JSON is an observation report, not a declaration that a target task succeeded.'''
ANCHOR_PROMPT='''Edit this simulation image into a realistic photograph of the same tabletop scene. Replace ONLY the robot arms/grippers at the left and right image boundaries with natural human forearms/hands resting outside the bowls. Preserve the exact camera, table, two bowls and their positions, the red apple currently in one bowl, the small green object outside the bowls, illumination and shadows as far as possible. Hands must not touch or move anything yet. Keep the occupied/empty bowl identities unchanged. No added object, no text, no arrows, no intended task cues. Do not add a person face. This is a task-independent starting frame.'''
VIDEO_PROMPT='''A continuous fixed-camera shot of natural human manipulation, starting exactly from the reference image. A human hand grasps the red apple that is inside the occupied bowl, lifts it clear of that bowl, moves it to the initially empty bowl, releases it inside, and withdraws completely. The apple settles and stays in the destination bowl. Both bowls and the small green object outside the bowls remain untouched and stationary. Show the entire grasp, transfer, release and final settled state. No cuts, no camera motion, no new objects, no duplication or disappearance. Do not move either bowl. Human hands only, no robot arms. Natural speed.'''
OBSERVATION_SCHEMA={
    'type':'object','properties':{
        'initial_relations':{'type':'array','items':{'type':'string'}},
        'final_relations':{'type':'array','items':{'type':'string'}},
        'events':{'type':'array','items':{'type':'object','properties':{
            'start_seconds':{'type':'number'},'end_seconds':{'type':'number'},
            'action':{'type':'string'},'visible_evidence':{'type':'string'}},
            'required':['start_seconds','end_seconds','action','visible_evidence']}},
        'uncertain_observations':{'type':'array','items':{'type':'string'}},
        'visible_discontinuities':{'type':'array','items':{'type':'string'}},
        'task_description':{'type':'string'}},
    'required':['initial_relations','final_relations','events','uncertain_observations','visible_discontinuities','task_description']}


def save(path,data):
    write_json(path,data,overwrite=False)


def diagnostic_transport(directory):
    def call(method,url,**kwargs):
        try:
            raw=_urllib_transport(method,url,**kwargs)
            # Persist a private response before semantic parsing, so an invalid
            # structured response can be diagnosed without another billable call.
            import os
            serialized=json.dumps(raw,ensure_ascii=False)
            key=os.environ.get('ECHOFLOW_API_KEY')
            if key:serialized=serialized.replace(key,'[REDACTED]')
            save(directory/'provider_response.json',json.loads(serialized))
            return raw
        except HTTPError as exc:
            # Never save auth headers or a complete provider error body.
            raw=exc.read(8192).decode('utf8','replace')
            try:
                body=json.loads(raw);error=body.get('error',body)
                message=error.get('message','') if isinstance(error,dict) else str(error)
                code=error.get('code') if isinstance(error,dict) else None
            except ValueError:message='non_json_http_error';code=None
            import os
            message=str(message).replace(os.environ.get('ECHOFLOW_API_KEY','\0'),'[REDACTED]')
            message=re.sub(r'sk-[A-Za-z0-9_-]{12,}','[REDACTED]',message)
            message=re.sub(r'https?://\S+','[URL_REDACTED]',message)
            save(directory/'http_diagnostic.json',{'http_status':exc.code,'provider_code':code,'message':message[:1200]})
            raise
    return call


def run(args):
    root=args.root;root.mkdir(parents=True,exist_ok=True)
    load_env_file(args.env_file)
    cfg=json.loads(CONFIG.read_text())
    directory=root/args.command
    directory.mkdir(exist_ok=True)
    if args.command=='poll-video':
        submission=json.loads((root/'submit-video/result.json').read_text())
        result=SeedanceClient(cfg['seedance']).poll(submission['task_id'])
        # GET polling is non-mutating. Keep each state observation.
        path=directory/f'{time.time_ns()}.json';save(path,result)
        if result['status']=='succeeded':
            result['media']=download_video(result['output_url'],root/'generated_human.mp4',
                task_id=result['task_id'],request_sha256=file_sha256(root/'submit-video/request_metadata.json'))
            save(directory/f'{time.time_ns()}_downloaded.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='output_url'},ensure_ascii=False));return
    result_path=directory/'result.json'
    if result_path.exists():
        print(json.dumps({'status':'already_completed_no_resubmit','result_path':str(result_path)}));return
    if args.command=='review-human':
        window=json.loads(HUMAN_SPEC.read_text())['tasks'][0]['human_windows'][0]
        media=Path(window['video']['path']);prompt=BLIND_PROMPT
        if file_sha256(media)!=window['video']['sha256']:raise ValueError('Human source changed')
        model=cfg['gemini']['model']
    elif args.command=='edit-anchor':
        media=ANCHOR;prompt=ANCHOR_PROMPT;model=cfg['anchor_model']
    elif args.command=='submit-video':
        media=root/'edit-anchor/human_initial.png';prompt=VIDEO_PROMPT;model=cfg['seedance']['model']
        if not media.exists():raise ValueError('Actual generated image required')
    else:
        media=root/'generated_human.mp4';prompt=BLIND_PROMPT;model=cfg['gemini']['model']
        if not media.exists():raise ValueError('Actual generated video required')
    metadata=dict(model=model,source_path=str(media),source_sha256=file_sha256(media),prompt=prompt,
        mode='first_frame_semantic_generation_not_video_to_video_edit' if args.command=='submit-video' else args.command,
        training_admitted=False,source_wrist_3d_labels_transferable=False)
    save(directory/'request_metadata.json',metadata)
    # Durable marker before POST; timeout/interruption must never auto-rebill.
    with (directory/'call_started.json').open('x') as f:
        json.dump({'started_unix':time.time(),'state':'response_unknown_until_persisted'},f)
    transport=diagnostic_transport(directory)
    started=time.monotonic()
    try:
        if args.command in {'review-human','review-output','review-output-schema'}:
            schema=OBSERVATION_SCHEMA if args.command=='review-output-schema' else None
            result=GeminiClient(cfg['gemini'],transport=transport).review(media,prompt,schema)
        elif args.command=='edit-anchor':
            result=GeminiClient(cfg['gemini'],transport=transport).generate_image(media,prompt,model=model)
            from PIL import Image
            image=Image.open(io.BytesIO(base64.b64decode(result['images'][0]['data'],validate=True)))
            image.load();image.save(directory/'human_initial.png')
        else:
            data='data:image/png;base64,'+base64.b64encode(media.read_bytes()).decode()
            payload=dict(model=model,content=[{'type':'text','text':prompt},
                {'type':'image_url','image_url':{'url':data},'role':'first_frame'}],
                ratio='adaptive',duration=8,resolution='720p',generate_audio=False)
            result=SeedanceClient(cfg['seedance'],transport=transport).submit(payload)
        result['elapsed_seconds']=round(time.monotonic()-started,3)
        save(result_path,result)
        summary={k:v for k,v in result.items() if k not in {'images','output_url'}}
        summary['result_path']=str(result_path)
        if 'images' in result:summary['image_count']=len(result['images'])
        print(json.dumps(summary,ensure_ascii=False))
    except ProviderError as exc:
        failure=dict(status='failed_or_unresolved',code=exc.code,http_status=exc.http_status,
            submission_unknown=exc.submission_unknown,elapsed_seconds=round(time.monotonic()-started,3))
        save(directory/'failure.json',failure);print(json.dumps(failure));raise SystemExit(2)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['review-human','edit-anchor','submit-video','poll-video','review-output','review-output-schema'])
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--env-file',type=Path,default=Path('/home/xzzhou/.config/policygen/credentials/echoflow.env'))
    run(p.parse_args())
