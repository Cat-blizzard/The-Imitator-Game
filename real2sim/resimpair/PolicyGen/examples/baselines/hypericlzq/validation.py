"""Fixed validation feature cache and physically interpretable Panda diagnostics.

Frozen teacher tokens are cached ONCE. Every validation uses identical clips,
references and random noise. Raw flow error is separated by action space, while
Panda sampled action errors use metres/radians before IK (not task success).
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation


@torch.no_grad()
def prepare_validation(model,loader,device,batches,path,signature,amp=True):
    path=Path(path)
    if path.exists():
        c=torch.load(path,map_location='cpu',weights_only=False)
        if c['signature']!=signature:raise ValueError('Validation feature cache mismatch')
        return c['batches']
    model.eval();cached=[]
    with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
        torch.manual_seed(917)
        for i,raw in enumerate(loader):
            if i>=batches:break
            b={k:v.to(device) for k,v in raw.items()}
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=amp and device.type=='cuda'):
                tokens={kind+'_tokens':model._tokens(b,kind).cpu().half() for kind in ('reference','reference2','obs','context','future')}
            c={k:v for k,v in raw.items() if not k.endswith(('_video','_cache_hit')) and k not in ('obs','future_rgb')}
            if model.config.rgb_weight>0:c['future_rgb']=raw['future_rgb']
            c.update(tokens);cached.append(c)
            print(json.dumps(dict(validation_cache_batch=i+1,total_batches=batches)),flush=True)
    torch.save(dict(signature=signature,batches=cached),str(path)+'.tmp');Path(str(path)+'.tmp').replace(path)
    model.train();return cached


def panda_errors(pred,batch,spec):
    target=batch['actions'][...,:14]*batch['actions'].new_tensor(spec['action_std'])+batch['actions'].new_tensor(spec['action_mean'])
    mask=batch['action_mask'][...,:14].reshape(-1,2,7).cpu().numpy()
    p=pred.reshape(-1,2,7).float().cpu().numpy();y=target.reshape(-1,2,7).float().cpu().numpy()
    valid=mask[...,:6].all(-1)
    mm=np.linalg.norm(p[...,:3]-y[...,:3],axis=-1)*1000
    Rp=Rotation.from_rotvec(p[...,3:6].reshape(-1,3)).as_matrix()
    Ry=Rotation.from_rotvec(y[...,3:6].reshape(-1,3)).as_matrix()
    deg=np.linalg.norm(Rotation.from_matrix(Rp.swapaxes(-1,-2)@Ry).as_rotvec(),axis=-1).reshape(valid.shape)*180/np.pi
    g=mask[...,6]
    moving=valid&((np.linalg.norm(y[...,:3],axis=-1)>.005)|(np.linalg.norm(y[...,3:6],axis=-1)>.035))
    return dict(position_mm_sum=float(mm[valid].sum()),rotation_deg_sum=float(deg[valid].sum()),pose_count=int(valid.sum()),
                moving_position_mm_sum=float(mm[moving].sum()),moving_rotation_deg_sum=float(deg[moving].sum()),moving_pose_count=int(moving.sum()),
                gripper_correct=int(((p[...,6]>.5)==(y[...,6]>.5))[g].sum()),gripper_count=int(g.sum()))


@torch.no_grad()
def validate_cached(model,cached,device,amp=True):
    model.eval();totals={};weights={};physical={};panda_name='panda_eef14_v3'
    devices=[device.index or 0] if device.type=='cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(917)
        for i,raw in enumerate(cached):
            b={k:v.to(device).float() if k.endswith('_tokens') else v.to(device) for k,v in raw.items()}
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=amp and device.type=='cuda'):
                out=model(b)
                for k,v in out.items():
                    if k.startswith('action_count/'):continue
                    n=float(out['action_count/'+k.split('/',1)[1]]) if k.startswith('action_raw/') else len(b['space_id'])
                    totals[k]=totals.get(k,0)+float(v)*n;weights[k]=weights.get(k,0)+n
                if panda_name not in model.space_names:continue
                which=b['has_actions']&(b['space_id']==model.space_names.index(panda_name))
                if not which.any():continue
                pb={k:v[which] for k,v in b.items()}
                torch.manual_seed(1700+i);pred=torch.stack(model.sample_actions(pb))
                stats=panda_errors(pred,pb,model.spaces[panda_name])
                for k,v in stats.items():physical['correct/'+k]=physical.get('correct/'+k,0)+v
                spec=model.spaces[panda_name];hold=torch.zeros_like(pred)
                state=pb['state'][:,:18]*pb['state'].new_tensor(spec['state_std'])+pb['state'].new_tensor(spec['state_mean'])
                hold[...,6]=((state[:,7]+state[:,8])/.08).clamp(0,1)[:,None]
                hold[...,13]=((state[:,16]+state[:,17])/.08).clamp(0,1)[:,None]
                hs=panda_errors(hold,pb,spec)
                for k,v in hs.items():physical['hold/'+k]=physical.get('hold/'+k,0)+v
                # Select a genuinely different task reference from the SAME batch.
                goals=pb['reference_goal_id'];targets=[];refs=[]
                for j in range(len(goals)):
                    other=torch.where(goals!=goals[j])[0]
                    if len(other):targets.append(j);refs.append(int(other[0]))
                if len(targets)==len(goals) and model.training_stage!='base':
                    wrong=dict(pb);wrong['reference_tokens']=pb['reference_tokens'][refs]
                    wrong['reference_domain']=pb['reference_domain'][refs]
                    torch.manual_seed(1700+i);pred_wrong=torch.stack(model.sample_actions(wrong))
                    ws=panda_errors(pred_wrong,pb,model.spaces[panda_name])
                    cs=panda_errors(pred,pb,model.spaces[panda_name])
                    for label,values in [('wrong',ws),('paired_correct',cs)]:
                        for k,v in values.items():physical[label+'/'+k]=physical.get(label+'/'+k,0)+v
    report={k:totals[k]/max(weights[k],1) for k in totals}
    report['action_space_example_counts']={k.split('/',1)[1]:int(n) for k,n in weights.items() if k.startswith('action_raw/')}
    for label in ('correct','wrong','paired_correct','hold'):
        count=physical.get(label+'/pose_count',0)
        if count:
            report['panda_'+label]=dict(position_mm=physical[label+'/position_mm_sum']/count,
                rotation_deg=physical[label+'/rotation_deg_sum']/count,
                gripper_accuracy=physical[label+'/gripper_correct']/max(physical[label+'/gripper_count'],1),pose_targets=count)
            moving=physical.get(label+'/moving_pose_count',0)
            if moving:report['panda_'+label].update(moving_position_mm=physical[label+'/moving_position_mm_sum']/moving,
                moving_rotation_deg=physical[label+'/moving_rotation_deg_sum']/moving,moving_pose_targets=moving)
    if 'panda_wrong' in report:
        report['panda_wrong_minus_correct_mm']=report['panda_wrong']['position_mm']-report['panda_paired_correct']['position_mm']
    model.train();return report
