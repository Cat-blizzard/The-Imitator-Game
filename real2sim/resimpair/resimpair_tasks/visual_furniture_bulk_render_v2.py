"""Finite batch screens and full native-clock dual-camera furniture videos.

Sources must already have independently accepted physical executions. The
adapter reuses the frozen renderer, shared GPU locks and state roundtrip checks.
Candidate video launches require a separate screen qualification receipt.
"""
from __future__ import annotations

import argparse
import fcntl
import html
import os
from pathlib import Path
import shutil
import time

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write
from . import visual_furniture_bulk_scene_v2 as furniture
from . import multitask_appearance_v5 as appearance

VERSION='bulk_furniture_native_video_batch_v2'
REGISTRY=ROOT/'runtime/visual_furniture_bulk_render_v2'
PRIOR_BUDGET=ROOT/'runs/visual_furniture_bulk_screen_20261030_v1/budget/limits.json'


def register_budget(output, prior_budget, allocation_id, screens=24, videos=12):
    if not allocation_id or type(screens) is not int or type(videos) is not int or not 1<=screens<=24 or not 0<=videos<=12:
        raise ValueError('current_authorized_allocation_is_at_most_24_screen_and_12_video_jobs')
    prior_path=Path(prior_budget); prior=read(prior_path)
    reservations=list((prior_path.parent/'reservations').glob('*.json'))
    for path in reservations:
        reservation=read(path); old_root=resolve(reservation['protocol']).parent
        if not (old_root/'cases'/reservation['case']['id']/'terminal.json').exists():
            raise ValueError('prior_render_submission_unresolved')
    historical=prior['historical_actual_render_jobs']+len(reservations); count=screens+videos
    REGISTRY.mkdir(parents=True,exist_ok=True)
    target=REGISTRY/'registrations'/(digest(allocation_id)+'.json')
    with (REGISTRY/'.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if target.exists(): raise ValueError('allocation_already_registered_recover_original_directory')
        limits={'schema':VERSION,'maximum_attempts':count,'allocation_id':allocation_id,'allocation_limit':count,
            'allocation':{'screen':screens,'video':videos},'prior_budget':binding(prior_path),
            'historical_actual_render_jobs':historical,
            'prior_lineage_limit':prior['maximum_lineage_render_jobs_including_history'],
            'maximum_lineage_render_jobs_including_history':prior['maximum_lineage_render_jobs_including_history']+count,
            'maximum_actual_jobs_including_history':historical+count,
            'unspent_prior_allowance_not_reassigned':True,'physical_execution_limit':0,'api_limit':0,
            'gpu':2,'minimum_free_gpu_mib':12288,'parallel_workers':1,
            'timeout_seconds':{'screen':480,'video':480},'resource_wait_seconds':180,
            'automatic_retries':0,'unknown_submission_must_be_recovered':True}
        write(output/'budget/limits.json',limits)
        write(target,{'allocation_id':allocation_id,'maximum_attempts':count,'root':str(output.resolve()),
            'budget':binding(output/'budget/limits.json'),'created_unix':time.time()})
    return binding(target),limits


def source_specs_from_cohort(cohort, output, *, seed=2026103022, repeats=3):
    """Prepare recipes from an existing accepted cohort; no agent per asset."""
    if Path(output).exists(): raise ValueError('new_source_recipe_file_required')
    original=read(cohort); rows=original['sources']; rooms=list(furniture.ROOMS); specs=[]
    for index,row in enumerate(rows):
        specs.append({'id':row['id']+'_'+rooms[index%len(rooms)],'execution':row['source_execution'],
            'partition':row['dataset_partition'],'seed':seed+index,'room':rooms[index%len(rooms)],
            'view_layout':'dual_view','count':4,'baseline_screen':True,'video':True})
    for i,row in enumerate(rows[:repeats]):
        room=rooms[(i+3)%len(rooms)]
        specs.append({'id':row['id']+'_variant_'+room,'execution':row['source_execution'],
            'partition':row['dataset_partition'],'seed':seed+100+i,'room':room,
            'view_layout':('opposite_high','robot_side','dual_view')[i%3],
            'count':4,'baseline_screen':True,'video':True})
    if len(specs)>12: raise ValueError('current_batch_video_allocation_requires_at_most_12_recipes')
    value={'schema':VERSION,'source_cohort':binding(cohort),'source_count':len(rows),
        'appearance_recipe_count':len(specs),'same_source_variants_are_not_new_physics':True,'sources':specs}
    write(output,value); return value


def prepare(source_specs, output, *, catalog=furniture.CATALOG, prior_budget=PRIOR_BUDGET,
            allocation_id='bulk_furniture_20261030_full_library_native_videos'):
    output=Path(output).resolve(); source_specs=Path(source_specs)
    if output.exists(): raise ValueError('new_batch_output_required_preserve_previous_results')
    if shutil.disk_usage(ROOT).free<10*2**30: raise ValueError('insufficient_disk_preserve_history')
    specs=read(source_specs)['sources']
    if not specs or len({s['id'] for s in specs})!=len(specs): raise ValueError('nonempty_unique_recipe_ids_required')
    screens=len(specs)+sum(bool(s.get('baseline_screen',True)) for s in specs)
    videos=sum(bool(s.get('video',True)) for s in specs)
    if screens>24 or videos>12: raise ValueError('registered_current_render_limits_exceeded')
    cases=[]; sources=[]; receipts=[]; cached_sources={}
    for i,spec in enumerate(specs):
        source_key=digest(spec['execution'])
        if source_key not in cached_sources:
            source=appearance.replay.validate_source(resolve(spec['execution']).parent)
            baseline,repair=appearance.render.baseline_scene(source)
            envelope=furniture.domain.build_motion_envelope(baseline,source['source']/'measurements.npz')
            cached_sources[source_key]=(source,baseline,repair,envelope)
        source,baseline,repair,envelope=cached_sources[source_key]
        source_partition=source['bundle']['case'].get('partition'); partition=spec.get('partition',source_partition)
        if partition!=source_partition: raise ValueError('background_partition_must_preserve_original_source')
        scene,receipt=furniture.apply_overlay(baseline,spec['seed'],motion_envelope=envelope,catalog_path=catalog,
            room=spec['room'],preset=spec.get('preset'),partition=partition,count=spec.get('count',4),
            view_layout=spec.get('view_layout','dual_view'),required_asset=spec.get('required_asset'))
        if not receipt['appearance_applied']: raise ValueError('visual_resources_missing_no_success_claim')
        appearance.assert_invariants(baseline,scene)
        appearance.render.validate_visual_derivative(source['scene'],scene,'environment_3d')
        source_id=f'source_{i:02d}'; count=len(source['states'])
        grasp=next(e['frame'] for e in read(source['source']/'audit.json')['event_frames'] if e['id'].endswith('.grasp'))
        source_row={'id':source_id,'source_execution':source['bindings']['execution_bundle.json'],
            'artifacts':source['bindings'],'source_execution_code':source['bundle']['code'],
            'frame_count':count,'fps':float(source['fps']),'source_spec':spec}
        sources.append(source_row)
        recipe_id=f'{i:02d}_{spec["id"]}'
        variants=[('screen','candidate',scene)]
        if spec.get('baseline_screen',True): variants.insert(0,('screen','baseline',baseline))
        if spec.get('video',True): variants.append(('video','candidate',scene))
        for stage,variant,render_scene in variants:
            case_id=f'{stage}_{recipe_id}_{variant}'; folder=output/'cases'/case_id
            write(folder/'render_scene.json',render_scene)
            write(folder/'motion_envelope.json',envelope)
            write(folder/'source_artifacts.json',source['bindings'])
            for name in appearance.replay.COPY_EVIDENCE: shutil.copy2(source['source']/name,folder/name)
            metadata_repair=None
            if repair is not None:
                write(folder/'source_metadata_repair.json',repair);metadata_repair=binding(folder/'source_metadata_repair.json')
            if variant=='candidate':
                write(folder/'appearance_receipt.json',receipt);receipts.append(binding(folder/'appearance_receipt.json'))
            case={'id':case_id,'source_id':source_id,'stage':stage,
                'variant':'baseline' if variant=='baseline' else 'environment_3d',
                'scene':binding(folder/'render_scene.json'),'source_execution':source['bindings']['execution_bundle.json'],
                'state_indices':[0,grasp,count-1] if stage=='screen' else None,
                'visualization_only':True,'R2H_ready':False,'predeclared_control':False,
                'screen_case_id':f'screen_{recipe_id}_candidate' if stage=='video' else None,
                'metadata_repair':metadata_repair,'recipe_id':recipe_id,'room':spec['room']}
            cases.append(case)
    registration,limits=register_budget(output,prior_budget,allocation_id)
    selection={'schema':VERSION,'author':'implementation_agent_with_independent_video_screen_gate_required',
        'input_sources':binding(source_specs),'catalog':binding(catalog),'sources':sources,'cases':cases,
        'predeclared_control_video_case_ids':[],'appearance_receipts':receipts,
        'qualification_scope':'complete saved-state native-clock visual derivatives, source physics and both cameras unchanged',
        'independent_source_executions':len(cached_sources),'visual_recipes':len(specs)}
    write(output/'selection_protocol.json',selection)
    modules=[Path(__file__),Path(furniture.__file__),Path(appearance.__file__),
        Path(appearance.render.__file__),Path(appearance.replay.__file__),Path(furniture.domain.__file__),
        ROOT/'resimpair_tasks/visual_furniture_bulk_scene_v1.py',ROOT/'resimpair_tasks/scene_quality_v3.py',
        ROOT/'resimpair_tasks/scene_material_policy_v3.py',ROOT/'resimpair_tasks/appearance_v1.py',
        ROOT/'resimpair_pickplace/environment.py',ROOT/'resimpair_tasks/cudex_shared_gpu_v1.py',
        ROOT/'corl2026_workshop/experiments/autonomous_pipeline_v4/simulation.py',
        ROOT/'corl2026_workshop/experiments/autonomous_pipeline_v4/resources.py',
        ROOT/'PolicyGen/mani_skill/render/shaders.py',ROOT/'PolicyGen/mani_skill/sensors/camera.py']
    protocol={'schema':appearance.render.VERSION,'bulk_background_adapter':binding(__file__),
        'independent_protocol':binding(output/'selection_protocol.json'),'selection_protocol_author':selection['author'],
        'sources':sources,'cases':cases,'budget':binding(output/'budget/limits.json'),'budget_registration':registration,
        'code':{str(p.relative_to(ROOT)):binding(p) for p in modules},'state_roundtrip_tolerance':1e-6,'TCP_tolerance_m':1e-5,
        'physical_execution_count':0,'native_execution':False,'new_trajectory':False,
        'api_calls':0,'R2H_ready':False,'default_promoted':False}
    write(output/'protocol.json',protocol)
    refs={'protocol':binding(output/'protocol.json')}
    for ref in appearance.render.bindings_in(protocol): resolve(ref);refs['dependency_'+digest(ref)]=ref
    for candidate in (read(resolve(c['scene'])) for c in cases):
        for ref in appearance.render.bindings_in(candidate): resolve(ref);refs['scene_resource_'+digest(ref)]=ref
    write(output/'frozen_inputs.json',refs);appearance.render.verify(output)
    result={'schema':VERSION,'source_executions':len(cached_sources),'visual_recipes':len(specs),
        'screen_cases':screens,'video_cases':videos,'actual_render_jobs':0,
        'selected_distinct_geometry':len({x for p in receipts for x in read(resolve(p))['bulk_policy']['selected']}),
        'historical_actual_render_jobs':limits['historical_actual_render_jobs'],
        'maximum_lineage_render_jobs':limits['maximum_lineage_render_jobs_including_history'],
        'native_video_frames_total_per_camera':sum(r['frame_count'] for r in sources),
        'new_physical_executions':0,'API_calls':0}
    write(output/'prepared.json',result);return result


def run(output, *, stage, limit, gate=None):
    """Isolate terminal asset/render failures; unknown jobs stop for recovery."""
    root=Path(output);protocol,limits=appearance.render.verify(root)
    if type(limit) is not int or not 1<=limit<=limits['allocation'][stage]: raise ValueError('finite_stage_limit_required')
    results=[]
    for _ in range(limit):
        pending=[c for c in protocol['cases'] if c['stage']==stage and not (root/'cases'/c['id']/'terminal.json').exists()]
        if not pending: break
        expected=pending[0];path=root/'cases'/expected['id']/'terminal.json'
        try: appearance.render.run(root,stage,1,gate=gate)
        except RuntimeError as exc:
            if not str(exc).startswith('render_failure_preserved_no_automatic_retry:') or not path.exists(): raise
        if not path.exists(): raise RuntimeError('render_unknown_recover_original_job')
        results.append({'id':expected['id'],**read(path)})
    result={'schema':VERSION,'stage':stage,'actual_new_attempts':len(results),'results':results,
        'failed':sum(r['returncode']!=0 for r in results),'new_physical_executions':0,'API_calls':0}
    write(root/f'batch_{stage}_{int(time.time())}.json',result);return result


def gallery(output, destination):
    root=Path(output);destination=Path(destination)
    if destination.exists(): raise ValueError('new_gallery_required_preserve_previous_review')
    protocol=read(root/'protocol.json');groups={}
    def href(path):return html.escape(os.path.relpath(path,destination.parent),quote=True)
    for case in protocol['cases']:groups.setdefault(case['recipe_id'],[]).append(case)
    blocks=[]
    for name,cases in groups.items():
        media=[]
        for camera in ('robot_side','opposite_high'):
            pair=[]
            for variant in ('baseline','environment_3d'):
                case=next((c for c in cases if c['stage']=='screen' and c['variant']==variant),None)
                if case:
                    folder=root/'cases'/case['id'];image=folder/(camera+'_state_000000.png')
                    if image.exists():pair.append(f'<figure><a href="{href(image)}"><img loading="lazy" src="{href(image)}"></a><figcaption>{camera} / {variant}</figcaption></figure>')
            media.append('<div class="pair">'+''.join(pair)+'</div>')
        video=next((c for c in cases if c['stage']=='video'),None)
        if video:
            folder=root/'cases'/video['id'];items=[]
            for camera in ('robot_side','opposite_high'):
                path=folder/(camera+'.mp4')
                if path.exists():items.append(f'<figure><video controls preload="none" poster="{href(folder/(camera+"_initial.png"))}" src="{href(path)}"></video><figcaption>{camera} / complete native timeline</figcaption></figure>')
            media.append('<div class="pair">'+''.join(items)+'</div>')
        statuses=[c['id']+': '+(str(read(root/'cases'/c['id']/'terminal.json')['returncode']) if (root/'cases'/c['id']/'terminal.json').exists() else 'not completed') for c in cases]
        blocks.append('<section><h2>'+html.escape(name)+'</h2>'+''.join(media)+'<details><summary>Execution records</summary><pre>'+html.escape('\n'.join(statuses))+'</pre></details></section>')
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text('<!doctype html><meta charset="utf-8"><title>Native furniture background batch</title><style>body{font:16px sans-serif;background:#edf1f4;color:#172331;max-width:1250px;margin:auto;padding:24px}.pair{display:flex;gap:12px}figure{margin:6px;flex:1}img,video{width:100%;border-radius:8px}section{background:white;padding:16px;margin:20px 0;border-radius:12px}figcaption{padding:8px}</style><h1>原背景 → 真实家具背景</h1><p>同一物理执行、同一动作状态时间轴、同一双相机。仅背景外观变化；不计为新增动作轨迹。全库可消费资格与本页实际抽样渲染数量分别统计。人类视频尚未生成。</p>'+''.join(blocks))
    return {'gallery':str(destination),'recipes':len(groups)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('recipes');p.add_argument('--cohort',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=3)
    p=sub.add_parser('prepare');p.add_argument('--sources',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--catalog',type=Path,default=furniture.CATALOG);p.add_argument('--prior-budget',type=Path,default=PRIOR_BUDGET)
    p.add_argument('--allocation-id',default='bulk_furniture_20261030_full_library_native_videos')
    p=sub.add_parser('run');p.add_argument('--output',type=Path,required=True);p.add_argument('--stage',choices=('screen','video'),required=True)
    p.add_argument('--limit',type=int,required=True);p.add_argument('--gate',type=Path)
    p=sub.add_parser('gallery');p.add_argument('--output',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='recipes':print(source_specs_from_cohort(args.cohort,args.output,repeats=args.repeats))
    elif args.command=='prepare':print(prepare(args.sources,args.output,catalog=args.catalog,prior_budget=args.prior_budget,allocation_id=args.allocation_id))
    elif args.command=='run':print(run(args.output,stage=args.stage,limit=args.limit,gate=args.gate))
    else:print(gallery(args.output,args.destination))
