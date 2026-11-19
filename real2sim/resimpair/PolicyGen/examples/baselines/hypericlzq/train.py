"""Mixed-source training. No training runs are launched by data curation.

python -m examples.baselines.hypericlzq.train --manifest ... --output ...
torchrun --nproc-per-node=4 -m examples.baselines.hypericlzq.train ...
"""
from __future__ import annotations
import argparse
import contextlib
from dataclasses import asdict,fields
import json
import hashlib
import math
import os
from pathlib import Path
import random
import time
import shutil
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from .data import CuratedDataset,SourceGoalSampler
from .model import HyperICLZQ,ZQConfig
from .modules.frame_tokenizer import FrozenFrameTokenizer
from .epochs import EpochSchedule,EpochMeter


def seed_worker(_):
    seed=torch.initial_seed()%2**32;random.seed(seed);np.random.seed(seed);torch.set_num_threads(1)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=20000);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--accumulate',type=int,default=1);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--warmup',type=int,default=500);p.add_argument('--aux-warmup',type=int,default=2000)
    p.add_argument('--num-workers',type=int,default=4);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--checkpoint-every',type=int,default=1000);p.add_argument('--validate-every',type=int,default=500)
    p.add_argument('--validation-batches',type=int,default=8);p.add_argument('--log-every',type=int,default=20)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--cached-tokens',action='store_true')
    p.add_argument('--tokenizer-batch-size',type=int,default=4)
    p.add_argument('--gpu-memory-fraction',type=float,default=1.)
    p.add_argument('--first-checkpoint',type=int,default=25)
    p.add_argument('--validation-batch-size',type=int,default=16)
    p.add_argument('--validation-workers',type=int,default=2)
    p.add_argument('--balanced-batches',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--cache-validation',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--keep-every',type=int,default=5000)
    p.add_argument('--save-every-epochs',type=int,default=0,help='Retain an immutable checkpoint every N nominal epochs; overrides step retention')
    p.add_argument('--validate-every-epochs',type=int,default=0,help='Validate every N nominal epochs; 0 retains step scheduling')
    p.add_argument('--external-support-exponent',type=float,default=0.)
    p.add_argument('--reference-cache-dir')
    p.add_argument('--tiers',default='Q0_anchor,Q1_action,Q2_video')
    p.add_argument('--tokenizer-name',default='facebook/dinov2-large')
    p.add_argument('--hf-cache-dir',default='/mnt/sdf/dataset/zhouxunzhe/huggingface')
    p.add_argument('--resume');p.add_argument('--init-from')
    p.add_argument('--source-ratios',default='imitator_sim=.6,imitator_human=.1,agibot=.15,egodex_pose=.1,egodex_video=.025,epic=.025')
    for f in fields(ZQConfig):p.add_argument('--'+f.name.replace('_','-'),type=type(f.default),default=f.default)
    return p.parse_args()


def load_weights(model,ckpt,config,spaces,resume=False):
    if ckpt['action_spaces']!=spaces:raise ValueError('Action-space schema/normalization changed')
    architecture={k:v for k,v in asdict(config).items() if not k.endswith('_weight')}
    old={k:v for k,v in ckpt['config'].items() if not k.endswith('_weight')}
    if old!=architecture or (resume and ckpt['config']!=asdict(config)):raise ValueError('Incompatible checkpoint configuration')
    result=model.load_state_dict(ckpt['model'],strict=False)
    allowed=('tokenizer.','rgb_head.','order_head.')
    if any(not k.startswith(allowed) for k in result.missing_keys+result.unexpected_keys):raise ValueError(str(result))
    return result


@torch.no_grad()
def validate(model,loader,device,batches):
    model.eval();sums={};count=0
    devices=[device.index or 0] if device.type=='cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(917)
        for i,batch in enumerate(loader):
            if i>=batches:break
            batch={k:v.to(device) for k,v in batch.items()};out=model(batch)
            for k,v in out.items():sums[k]=sums.get(k,0)+float(v)
            count+=1
    model.train()
    return {k:v/max(count,1) for k,v in sums.items()}


def main():
    a=parse_args()
    if a.resume and a.init_from:raise ValueError('Choose resume OR weights-only init')
    if min(a.steps,a.batch_size,a.accumulate,a.validate_every,a.checkpoint_every,a.log_every,a.validation_batches)<1:raise ValueError('Positive sizes required')
    torch.set_num_threads(4)
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    device=torch.device('cuda',int(os.environ.get('LOCAL_RANK',0))) if a.device=='cuda' else torch.device(a.device)
    if device.type=='cuda':
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(a.gpu_memory_fraction,device)
    if world>1:dist.init_process_group('nccl' if device.type=='cuda' else 'gloo')
    torch.manual_seed(a.seed+rank);np.random.seed(a.seed+rank);random.seed(a.seed+rank)
    c=ZQConfig(**{f.name:getattr(a,f.name) for f in fields(ZQConfig)})
    if a.reference_cache_dir:
        if not a.amp:raise ValueError('Reference cache uses the BF16 teacher configuration')
        from .reference_cache import setup_cache
        setup_cache(a.reference_cache_dir,c,a.tokenizer_name)
    data=CuratedDataset(a.manifest,c,cached_tokens=a.cached_tokens,tiers=a.tiers.split(','),reference_cache_dir=a.reference_cache_dir)
    epochs=EpochSchedule(len(data),a.batch_size*a.accumulate*world)
    if min(a.save_every_epochs,a.validate_every_epochs)<0:raise ValueError('Epoch intervals cannot be negative')
    if a.validate_every_epochs:a.validate_every=epochs.updates*a.validate_every_epochs
    if a.cached_tokens and data.raw.get('feature_spec'):
        spec=data.raw['feature_spec']
        if any(spec[k]!=v for k,v in dict(model=a.tokenizer_name,feat_dim=c.feat_dim,seq_patches=c.seq_patches,image_size=c.image_size).items()):raise ValueError('Feature cache tokenizer/configuration mismatch')
    ratios={k:float(v) for k,v in (part.split('=') for part in a.source_ratios.split(','))}
    sampler=SourceGoalSampler(data,a.steps*a.accumulate*a.batch_size,a.seed,rank,ratios,batch_size=a.batch_size if a.balanced_batches else 1,
        external_support_exponent=a.external_support_exponent)
    loader=DataLoader(data,batch_size=a.batch_size,sampler=sampler,num_workers=a.num_workers,pin_memory=device.type=='cuda',worker_init_fn=seed_worker)
    validation=None
    if rank==0 and any(r['split']=='validation' and r.get('tier','Q0_anchor') in a.tiers.split(',') for r in data.raw['episodes']):
        val_data=CuratedDataset(a.manifest,c,split='validation',cached_tokens=a.cached_tokens,tiers=a.tiers.split(','),reference_cache_dir=a.reference_cache_dir)
        # Fixed, reproducible coverage of sources/tasks; first-N manifest rows
        # would otherwise validate almost exclusively AgiBot after expansion.
        val_sampler=SourceGoalSampler(val_data,a.validation_batches*a.validation_batch_size,seed=917,
            ratios=dict(imitator_sim=1.,imitator_sim_extra=2.,imitator_human=.25,agibot=1.,oxe_franka=1.,egodex_pose=1.,egodex_video=.25,epic=.25),
            batch_size=a.validation_batch_size)
        validation=DataLoader(val_data,batch_size=a.validation_batch_size,sampler=val_sampler,num_workers=a.validation_workers,
            worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(917))
    tokenizer=None if a.cached_tokens else FrozenFrameTokenizer(backend='dinov2_hf',seq_patches=c.seq_patches,feat_dim=c.feat_dim,image_size=c.image_size,hf_model_name=a.tokenizer_name,hf_cache_dir=a.hf_cache_dir,frame_batch_size=a.tokenizer_batch_size)
    model=HyperICLZQ(c,data.spaces,tokenizer).to(device)
    model.reference_cache_dir=a.reference_cache_dir
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=a.lr,weight_decay=.01)
    def lr_scale(step):
        if step<a.warmup:return (step+1)/max(1,a.warmup)
        return .1+.9*.5*(1+math.cos(math.pi*min(1,(step-a.warmup)/max(1,a.steps-a.warmup))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_scale)
    start=0;previous_elapsed=0.;meter_state=None
    if a.resume or a.init_from:
        ckpt=torch.load(a.resume or a.init_from,map_location='cpu',weights_only=False)
        load_weights(model,ckpt,c,data.spaces,resume=bool(a.resume))
        if a.resume:
            optimizer.load_state_dict(ckpt['optimizer']);scheduler.load_state_dict(ckpt['scheduler']);start=ckpt['step']
            # Sampling restarts deterministically from a new stream; not exact mid-epoch continuation.
            sampler.set_epoch(start)
            previous_elapsed=float(ckpt.get('elapsed_seconds',0.))
            if ckpt.get('epoch_progress',{}).get('updates_per_epoch')==epochs.updates:meter_state=ckpt.get('epoch_meter')
    model.train();wrapped=DistributedDataParallel(model,device_ids=[device.index] if device.type=='cuda' else None,find_unused_parameters=True) if world>1 else model
    out=Path(a.output)
    if rank==0:
        out.mkdir(parents=True,exist_ok=True)
        # Pre-epoch checkpoints have elapsed time only in their JSONL log.
        if a.resume and not previous_elapsed and (out/'metrics.jsonl').exists():
            for line in (out/'metrics.jsonl').read_text().splitlines():
                try:old_record=json.loads(line)
                except json.JSONDecodeError:continue
                if old_record['step']<=start:previous_elapsed=old_record.get('elapsed_seconds',previous_elapsed)
        (out/'config.json').write_text(json.dumps(dict(args=vars(a),config=asdict(c),action_spaces=data.spaces,parameters=model.parameter_report(),
            epoch_schedule=dict(training_episodes=len(data),**epochs.progress(start),definition='ceil(training episodes / effective batch) optimizer updates, weighted sampling with replacement'),
            manifest_sha256=hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest()),indent=2))
        print(json.dumps(dict(event='training_segment_start',resume_step=start,**epochs.progress(start),source_ratios=ratios)),flush=True)
    iterator=iter(loader);started=time.monotonic();val_cached=None;best_panda=float('inf')
    meter=EpochMeter(start,state=meter_state)
    if rank==0 and (out/'best_panda.json').exists():
        previous=json.loads((out/'best_panda.json').read_text());best_panda=previous.get('selection_value',previous['position_mm'])
    for step in range(start,a.steps):
        update_started=time.monotonic()
        optimizer.zero_grad(set_to_none=True);record={}
        source_counts=torch.zeros(len(data.sampling_bucket_names),device=device,dtype=torch.long)
        for micro in range(a.accumulate):
            batch={k:v.to(device,non_blocking=True) for k,v in next(iterator).items()}
            source_counts+=torch.bincount(batch['sampling_bucket_id'],minlength=len(source_counts))
            sync=wrapped.no_sync() if world>1 and micro<a.accumulate-1 else contextlib.nullcontext()
            with sync:
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=a.amp and device.type=='cuda'):
                    losses=wrapped(batch);action=losses.get('action',losses['loss']*0)
                    aux_scale=min(1.,(step+1)/max(1,a.aux_warmup))
                    loss=c.action_weight*action+aux_scale*(losses['loss']-c.action_weight*action)
                    if world>1 and c.action_weight>0:
                        local=batch['has_actions'].sum().float();global_count=local.clone();dist.all_reduce(global_count)
                        loss=loss+c.action_weight*(world*local/global_count.clamp_min(1)-1)*action
                (loss/a.accumulate).backward()
            record['optimized_loss']=record.get('optimized_loss',0.)+float(loss.detach())/a.accumulate
            for k,v in losses.items():
                if k.startswith('action_raw/'):
                    count=float(losses['action_count/'+k.split('/',1)[1]])
                    record[k]=record.get(k,0.)+float(v.detach())*count
                elif k.startswith('action_count/'):record[k]=record.get(k,0.)+float(v.detach())
                else:record[k]=record.get(k,0.)+float(v.detach())/a.accumulate
        for k in record:
            if k.startswith('action_raw/'):record[k]/=max(record['action_count/'+k.split('/',1)[1]],1)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();scheduler.step()
        if world>1:dist.all_reduce(source_counts)
        sources=dict(zip(data.sampling_bucket_names,source_counts.tolist()))
        meter.add(step+1,record,sources,time.monotonic()-update_started)
        if rank==0 and ((step+1)%epochs.updates==0 or step+1==a.steps):
            epoch_record=meter.report(epochs)
            print(json.dumps(dict(event='epoch_summary',**epoch_record)),flush=True)
            with (out/'epochs.jsonl').open('a') as f:f.write(json.dumps(epoch_record)+'\n')
        if (step+1)%epochs.updates==0:meter=EpochMeter(step+1)
        if rank==0 and (step==start or (step+1)%a.log_every==0):
            record.update(step=step+1,**epochs.progress(step+1),source_counts=sources,grad_norm=float(norm),lr=scheduler.get_last_lr()[0],aux_scale=aux_scale,
                          elapsed_seconds=previous_elapsed+time.monotonic()-started,segment_elapsed_seconds=time.monotonic()-started,
                          seconds_per_step=(time.monotonic()-started)/(step+1-start),
                          gpu_peak_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else 0.)
            if step==start:
                record['nonzero_gradients']={name:any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in getattr(model,name).parameters())
                    for name in ('E','T','R','D','target','action_in','action_out')}
            print(json.dumps(record),flush=True)
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        epoch_checkpoint=epochs.due(step+1,a.save_every_epochs)
        periodic_checkpoint=epoch_checkpoint if a.save_every_epochs else (step+1)%a.checkpoint_every==0
        checkpoint_due=(periodic_checkpoint or (step+1)%a.validate_every==0 or step+1 in (a.steps,a.first_checkpoint) or (bool(a.resume) and step==start))
        if rank==0 and checkpoint_due:
            ckpt=dict(model={k:v for k,v in model.state_dict().items() if not k.startswith('tokenizer.')},
                      optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),config=asdict(c),action_spaces=data.spaces,args=vars(a),step=step+1,
                      epoch_progress=epochs.progress(step+1),epoch_meter=meter.state_dict(),elapsed_seconds=previous_elapsed+time.monotonic()-started)
            torch.save(ckpt,out/'last.pt.tmp');(out/'last.pt.tmp').replace(out/'last.pt')
            if epoch_checkpoint:
                name=epochs.checkpoint_name(step+1);destination=out/name
                if destination.exists():raise FileExistsError(f'Refusing to overwrite retained checkpoint {destination}')
                shutil.copyfile(out/'last.pt',out/(name+'.tmp'));(out/(name+'.tmp')).replace(destination)
                with (out/'checkpoints.jsonl').open('a') as f:f.write(json.dumps(dict(step=step+1,**epochs.progress(step+1),path=str(destination)))+'\n')
                print(json.dumps(dict(event='epoch_checkpoint',step=step+1,**epochs.progress(step+1),path=str(destination))),flush=True)
            elif not a.save_every_epochs and a.keep_every>0 and (step+1)%a.keep_every==0:shutil.copyfile(out/'last.pt',out/f'step_{step+1:07d}.pt')
        if (step+1)%a.validate_every==0:
            if world>1:dist.barrier()
            if rank==0 and validation is not None:
                if a.cache_validation:
                    from .validation import prepare_validation,validate_cached
                    if val_cached is None:
                        signature=hashlib.sha256(json.dumps(dict(manifest=hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),
                            config=asdict(c),batch=a.validation_batch_size,batches=a.validation_batches,tokenizer=a.tokenizer_name),sort_keys=True).encode()).hexdigest()
                        val_cached=prepare_validation(model,validation,device,a.validation_batches,out/'validation_tokens.pt',signature,a.amp)
                    values=validate_cached(model,val_cached,device,a.amp)
                else:values=validate(model,validation,device,a.validation_batches)
                report=dict(step=step+1,**epochs.progress(step+1),validation=values)
                print(json.dumps(report),flush=True)
                with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
                panda=values.get('panda_correct',{})
                selection=panda.get('moving_position_mm',panda.get('position_mm',float('inf')))
                if selection<best_panda:
                    best_panda=selection
                    shutil.copyfile(out/'last.pt',out/'best_panda.pt.tmp');(out/'best_panda.pt.tmp').replace(out/'best_panda.pt')
                    (out/'best_panda.json').write_text(json.dumps(dict(step=step+1,**epochs.progress(step+1),position_mm=panda['position_mm'],selection_value=best_panda,
                        selection_metric='Fixed heldout Panda moving-target position error in mm; NOT closed-loop success',
                        diagnostics=panda,hold_baseline=values.get('panda_hold',{})),indent=2))
            if world>1:dist.barrier()
    if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
