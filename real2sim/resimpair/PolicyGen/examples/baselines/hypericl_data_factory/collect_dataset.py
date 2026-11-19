"""Atomic qualified collection with native/224 media and honest failure counts."""
import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from .bootstrap import ROOT,sha,write,process
from ..hypericl_task_expansion.common import bind_gpu,source_pins

PROGRAMS=('pick_first_to_left','pick_first_to_right','pick_second_to_left','pick_second_to_right')


def seeds_for_assets(pool,base,count,condition='all',heldout=None):
    seeds=[]
    for seed in range(base,base+max(100,count*30)):
        rng=np.random.default_rng(seed+730001)
        ids={pool[int(i)]['uid']for i in rng.choice(len(pool),2,replace=False)}
        if condition=='exclude'and heldout in ids:continue
        if condition=='include'and heldout not in ids:continue
        seeds.append(seed)
        if len(seeds)==count:break
    if len(seeds)!=count:raise ValueError('insufficient disjoint asset-filtered layout seeds')
    if condition!='all'and heldout not in {r['uid']for r in pool}:raise ValueError('held-out asset is not in pool')
    return seeds


def visibility_quality(rec,program):
    index=0 if program.startswith('pick_first')else 1
    pixels=np.asarray([r['object_pixels_224']for r in rec.visibility])[:,index]
    in_frame=np.asarray([r['projected_entirely_in_frame']for r in rec.visibility])[:,index]
    projected=np.asarray([r['projected_bounds_224']for r in rec.visibility])[:,index]
    visible=np.asarray([r['visible_bounds_224']for r in rec.visibility])[:,index]
    present=pixels>0
    # GLB visuals must agree with the collider's projected hull bounds. A hole
    # or transparent material may reduce pixel area, but cannot move the mesh.
    aligned=(visible[present,:2]>=projected[present,:2]-3).all()and(visible[present,2:]<=projected[present,2:]+3).all()
    passed=bool(pixels[0]>=24 and pixels[-1]>=16 and in_frame.mean()>=.95 and aligned)
    return dict(passed=passed,selected_object=index,initial_pixels_224=int(pixels[0]),final_pixels_224=int(pixels[-1]),
        visible_pixels_quantiles=np.quantile(pixels,[0,.1,.5,.9,1]).tolist(),
        projected_in_frame_fraction=float(in_frame.mean()),segmentation_visual_within_collision_projection=bool(aligned),
        fraction_frames_detected=float((pixels>0).mean()),thresholds=dict(initial_pixels=24,final_pixels=16,in_frame_fraction=.95),
        segmentation_used_only_for_quality=True)


def verify_native(receipt):
    import av
    with np.load(receipt['qa_visibility'],allow_pickle=False)as f:q={k:f[k]for k in f.files}
    n=receipt['length']
    if q['object_pixels_224'].shape!=(n,2):raise ValueError('QA timeline length mismatch')
    expected=q['intrinsic_224'].copy();expected[:2]*=512/224
    np.testing.assert_allclose(q['intrinsic_512'],expected,atol=2e-4,rtol=0)
    if sha(receipt['reference_video']['path'])!=receipt['reference_video_sha256']:raise ValueError('native movie hash changed')
    pts=[]
    with av.open(receipt['reference_video']['path'])as c:
        for frame in c.decode(video=0):
            if(frame.width,frame.height)!=(512,512):raise ValueError('wrong native source size')
            pts.append(float(frame.pts*frame.time_base))
    if len(pts)!=n:raise ValueError('native movie/label count mismatch')
    np.testing.assert_allclose(pts,np.arange(n)/receipt['fps'],atol=1e-5,rtol=0)
    if not receipt['image_quality']['passed']:raise ValueError('image quality rejected')


def check_approval(root,family,programs,pins,profile):
    approval=json.loads((root/'review_approval.json').read_text())
    if not approval.get('approved_for_production'):raise ValueError('qualification lacks reviewed production gate')
    plan=json.loads((root/'plan.json').read_text())
    summary=json.loads((root/'summary.json').read_text())
    if sha(root/'summary.json')!=approval['summary_sha256']:raise ValueError('qualification summary changed')
    if plan['family']!=family or plan['view_profile']!=profile:raise ValueError('qualification configuration differs')
    if plan['source_sha256']!=pins:raise ValueError('production source differs from qualified source')
    reviewed=approval.get('reviewed_sample_ids',[])
    if not reviewed or not approval.get('reviewed_programs'):raise ValueError('missing actual visual sample review')
    if not set(programs).issubset(approval['reviewed_programs']):raise ValueError('some programs lack visual sample review')
    counts=Counter(r['program']for r in summary['results']if r['passed'])
    for program in programs:
        if counts[program]<6:raise ValueError('program has fewer than six independent qualification successes')
    return approval


def remap_paths(value,source,destination):
    if isinstance(value,dict):return {k:remap_paths(v,source,destination)for k,v in value.items()}
    if isinstance(value,list):return [remap_paths(v,source,destination)for v in value]
    if isinstance(value,str)and value.startswith(str(source)+'/'):return str(destination)+value[len(str(source)):]
    return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--family',choices=['bottle','cup'],required=True);p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--split',choices=['qualification','train','validation','test'],default='qualification')
    p.add_argument('--view-profile',choices=['close','balanced'],default='close')
    p.add_argument('--seed-base',type=int,required=True);p.add_argument('--layouts',type=int,default=8)
    p.add_argument('--programs',nargs='+',default=list(PROGRAMS));p.add_argument('--approval',type=Path)
    p.add_argument('--asset-condition',choices=['all','exclude','include'],default='all');p.add_argument('--heldout-uid')
    p.add_argument('--background-split',choices=['train','validation','test'])
    a=p.parse_args();gpu=bind_gpu(a.gpu)
    import gymnasium as gym
    from . import rich_runtime_v2,rich_collect,verify_rich
    from .rich_runtime_v2 import VisibilityRecorder,register,save_native,settling_equivalence
    from .framing_probe import asset_pool
    from ..hypericl_task_expansion.collect import solve,qualify,save,save_diagnostic,require
    if a.layouts<1 or any(x not in PROGRAMS for x in a.programs):raise ValueError('invalid production plan')
    pool=asset_pool(a.family);seeds=seeds_for_assets(pool,a.seed_base,a.layouts,a.asset_condition,a.heldout_uid)
    pins=source_pins()
    for path in [Path(__file__).resolve(),Path(rich_runtime_v2.__file__),Path(rich_collect.__file__),Path(verify_rich.__file__),
        Path(__file__).with_name('framing_probe.py'),ROOT/'asset_visual_review_v1.json',ROOT/'asset_adapters_v2/catalog.json',
        ROOT/'background_inventory_v1/approved.json',ROOT.parent/'open_assets/task_factory_20261016_v1/polyhaven_downloads.json']:
        pins[str(path.resolve())]=sha(path)
    if a.split!='qualification':
        if not a.approval:raise ValueError('a reviewed qualification gate is required')
        check_approval(a.approval,a.family,a.programs,pins,a.view_profile)
    a.root.mkdir(parents=True,exist_ok=False);identity=process(os.getpid());write(a.root/'process.json',identity)
    archive={}
    for filename,digest in pins.items():
        source=Path(filename);dest=a.root/'source_snapshot'/(digest+source.suffix)
        dest.parent.mkdir(exist_ok=True);shutil.copyfile(source,dest);archive[filename]=dict(path=str(dest),sha256=digest)
    write(a.root/'source_archive.json',archive)
    plan=dict(schema='rich_dual_resolution_collection_v2',family=a.family,split=a.split,
        view_profile=a.view_profile,layout_seeds=seeds,programs=a.programs,asset_uids=[r['uid']for r in pool],
        source_sha256=pins,native_settling_equivalence=settling_equivalence(),approval=str(a.approval)if a.approval else None,
        background_split=a.background_split or('train'if a.split=='qualification'else a.split),
        asset_condition=a.asset_condition,heldout_uid=a.heldout_uid,source_resolution=[512,512],policy_resolution=[224,224],
        initial_goal_leakage_forbidden=True,disk_reserve_gib=100,robot_action_dim=14,model_training_started=False)
    write(a.root/'plan.json',plan);register()
    env=gym.make('HyperICLRichAssetRelocation-v2',asset_pool=pool,family=a.family,view_profile=a.view_profile,
        data_split=plan['background_split'],program=a.programs[0],sim_backend='physx_cpu',render_backend=gpu['render_backend'],
        control_mode='pd_joint_pos',obs_mode='rgb+segmentation',reward_mode='none',render_mode='rgb_array',
        sensor_configs=dict(shader_pack='rt-fast'),human_render_camera_configs=dict(shader_pack='rt-fast'))
    results=[];started=time.time();initial={}
    try:
        for seed in seeds:
            for filename,digest in pins.items():
                if sha(filename)!=digest:raise RuntimeError('frozen collection source changed')
            for program in a.programs:
                if shutil.disk_usage(a.root).free<100*2**30:raise RuntimeError('disk reserve reached')
                rec=VisibilityRecorder(env);name=f'{seed}_{program}';folder=a.root/'staging'/name
                destination=a.root/'episodes'/name
                try:
                    rec.reset(seed,program);events=solve(rec,program)
                    metrics,final=qualify(rec,program);require(metrics['success'],'native goal qualification failed')
                    image_quality=visibility_quality(rec,program);require(image_quality['passed'],'image quality rejected: '+json.dumps(image_quality))
                    receipt=save(rec,folder,rec.unwrapped.scene_name,program,a.split,seed,events,metrics,final,pins)
                    receipt.update(save_native(rec,folder))
                    receipt.update(qualification_only=a.split=='qualification',production_approved=a.split!='qualification',
                        randomization=rec.unwrapped.randomization,initial_measured_states=rec.initial['states'].tolist(),
                        initial_measured_object_states=rec.initial['objects'].tolist(),fresh_recorder_per_episode=True,
                        image_quality=image_quality,asset_condition=a.asset_condition,heldout_uid=a.heldout_uid,
                        qa_privileged_fields_are_not_policy_inputs=True)
                    write(folder/'receipt.json',receipt)
                    _,check,_=verify_rich.verify(folder/'receipt.json');verify_native(receipt)
                    signature=(receipt['initial_physics_sha256'],receipt['initial_rgb_sha256'],receipt['reference_raw_initial_rgb_sha256'])
                    if seed in initial and initial[seed]!=signature:raise ValueError('initial observations encode the goal')
                    initial[seed]=signature
                    remapped=remap_paths(receipt,folder,destination);write(folder/'receipt.json',remapped)
                    write(folder/'verification.json',dict(numerical_and_physical=check,native_video_and_visibility_passed=True))
                    write(folder/'COMMITTED.json',dict(receipt_sha256=sha(folder/'receipt.json'),verified=True,
                        split=a.split,qualification_only=a.split=='qualification'))
                    destination.parent.mkdir(exist_ok=True);folder.rename(destination)
                    results.append(dict(seed=seed,program=program,passed=True,receipt=str(destination/'receipt.json'),
                        initial_pixels=image_quality['initial_pixels_224'],in_frame_fraction=image_quality['projected_in_frame_fraction']))
                except Exception as exc:
                    failed=a.root/'failures'/name;failed.parent.mkdir(exist_ok=True)
                    if folder.exists():folder.rename(failed)
                    if rec.rows:
                        try:save_diagnostic(rec,failed)
                        except Exception as d:write(failed/'diagnostic_failure.json',dict(error=repr(d)))
                    failure=dict(seed=seed,program=program,passed=False,error=repr(exc),
                        randomization=getattr(rec.unwrapped,'randomization',{}),stage=rec.stage)
                    write(failed/'failure.json',failure);results.append(failure)
                write(a.root/'state.json',dict(status='collecting_verified_native_clips',split=a.split,
                    completed=len(results),attempts=len(seeds)*len(a.programs),passed=sum(r['passed']for r in results),
                    elapsed_seconds=time.time()-started,process=identity,updated_unix=time.time()))
    finally:env.close()
    for filename,digest in pins.items():
        if sha(filename)!=digest:raise RuntimeError('source changed during collection')
    report=dict(status='batch_complete_review_pending',attempted=len(results),passed=sum(r['passed']for r in results),
        failed=sum(not r['passed']for r in results),by_program=dict(Counter(r['program']for r in results if r['passed'])),
        results=results,elapsed_seconds=time.time()-started,model_training_started=False,
        qualification_only=a.split=='qualification',production_release_review_pending=True)
    write(a.root/'summary.json',report);write(a.root/'state.json',report)
    print(json.dumps({k:v for k,v in report.items()if k!='results'}),flush=True)


if __name__=='__main__':main()
