"""Episode-balanced, goal-matched references with independently causal video targets."""
from __future__ import annotations
import collections
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from .media import frames_at
from .expert_bank import expert_key


def source_id(row):
    if row['source'].startswith('imitator'):
        return 0
    return {'egodex': 1, 'agibot': 2, 'epic': 3, 'oxe_franka': 4,
            'simulation_expansion': 5}[row['source']]


def sampling_bucket(row):
    source=row['source']
    if source=='egodex':source='egodex_pose' if row.get('action_space') else 'egodex_video'
    return row.get('sampling_bucket',source)


def pose_vector(transforms, pinch):
    xyz=transforms[...,:3,3]
    rot=transforms[...,:3,:2].swapaxes(-1,-2).reshape(*transforms.shape[:-2],6)
    return np.concatenate([xyz,rot,pinch[...,None]],-1).reshape(-1,20).astype('float32')


def ego_actions(arrays, t, horizon):
    """Observed future hand poses in camera(t), never in the moving future camera.

    This is kinematic supervision for a HUMAN action space, not robot motor control.
    """
    end=min(t+1+horizon,len(arrays['camera']))
    inv=np.linalg.inv(arrays['camera'][t])
    a=pose_vector(inv@arrays['wrists'][t+1:end],arrays['pinch'][t+1:end])
    s=pose_vector(inv@arrays['wrists'][t:t+1],arrays['pinch'][t:t+1])[0]
    valid=arrays['valid'][t+1:end] & arrays['valid'][t]
    mask=valid if valid.shape[-1]==20 else np.repeat(valid,10,axis=1)
    return a,s,mask


def agibot_motion(arrays,t,horizon):
    """Future measured joints + released gripper signal, not same-time copy.

    Local Alpha joint action fields duplicate state in the inspected release.
    This auxiliary target must not be advertised as validated native commands.
    """
    end=min(t+1+horizon,len(arrays['states']))
    a=arrays['actions'][t+1:end].copy()
    a[:,:14]=arrays['states'][t+1:end,:14]
    return a,arrays['states'][t],arrays['mask'][t+1:end]


class CuratedDataset(Dataset):
    def __init__(self, manifest, config, split='train', cached_tokens=False, tiers=None,reference_cache_dir=None,stage='joint',share_references=False):
        self.path=Path(manifest).resolve();self.root=self.path.parent;self.config=config
        self.raw=json.loads(self.path.read_text());self.spaces=self.raw['action_spaces']
        for r in self.raw['episodes']:
            if r['goal_key'] in self.raw.get('unseen_tasks',[]):
                if r['source'].startswith('imitator') or self.raw.get('protocol') not in ('external_task_transfer_v2','external_task_transfer_eef_v3'):
                    raise ValueError('Protected unseen task in training data')
        for spec in self.spaces.values():
            for kind,dim in [('action',spec['action_dim']),('state',spec['state_dim'])]:
                mean=np.asarray(spec[kind+'_mean']);std=np.asarray(spec[kind+'_std'])
                if mean.shape!=(dim,) or std.shape!=(dim,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std<=0).any():raise ValueError('Invalid normalization schema')
        self.space_names=sorted(self.spaces);self.space_ids={n:i for i,n in enumerate(self.space_names)}
        self.action_dim=max(x['action_dim'] for x in self.spaces.values())
        self.state_dim=max(x['state_dim'] for x in self.spaces.values())
        self.rows=[r for r in self.raw['episodes'] if r['split']==split and (tiers is None or r.get('tier','Q0_anchor') in tiers)]
        if not self.rows:raise ValueError(f'No episodes in split {split}')
        if len({r['id'] for r in self.raw['episodes']})!=len(self.raw['episodes']):raise ValueError('Duplicate episode IDs')
        # One recording/session may never occur on both sides of our validation split.
        groups=collections.defaultdict(set)
        for r in self.raw['episodes']:groups[(r['source'],r['group'])].add(r['split'])
        if any(len(s)>1 for s in groups.values()):raise ValueError('Recording group split leakage')
        self.goal_ids={g:i for i,g in enumerate(sorted({r['goal_key'] for r in self.rows}))}
        self.expert_keys=[expert_key(r) for r in self.rows]
        self.by_goal=collections.defaultdict(list)
        for i,r in enumerate(self.rows):
            self.by_goal[r['goal_key']].append(i)
            if r.get('action_space') and r['action_space'] not in self.spaces:raise ValueError('Unknown action space')
            if r.get('action_space') and not r.get('arrays'):raise ValueError('Labeled row without arrays')
            if cached_tokens and not r.get('tokens'):raise ValueError('Cached tokens missing from manifest')
        self.cached_tokens=cached_tokens
        self.stage=stage
        self.share_references=share_references
        if stage not in ('base','hyper','joint'):raise ValueError(stage)
        self.sampling_bucket_names=sorted({sampling_bucket(r) for r in self.raw['episodes']})
        self.sampling_bucket_ids={name:i for i,name in enumerate(self.sampling_bucket_names)}
        self.reference_cache_dir=reference_cache_dir
        if reference_cache_dir:
            from .reference_cache import row_key
            self.reference_keys={r['id']:row_key(r) for r in self.rows}
            if len(set(self.reference_keys.values()))!=len(self.reference_keys):raise ValueError('Reference key collision')
        self.array_cache=collections.OrderedDict()
        self.token_cache=collections.OrderedDict()
        self.reference_token_cache=collections.OrderedDict()
        self.by_family=collections.defaultdict(list)
        for i,r in enumerate(self.rows):
            if r.get('reference_family'):self.by_family[r['reference_family']].append(i)

    def reference_weight(self,query,reference):
        if query['goal_key']==reference['goal_key']:return 1.
        if query.get('reference_family') and query.get('reference_family')==reference.get('reference_family'):
            return min(query.get('family_confidence',0.),reference.get('family_confidence',0.),query['goal_confidence'],reference['goal_confidence'])
        return 0.

    def __len__(self):return len(self.rows)

    def _arrays(self,row):
        key=row['arrays']
        if key not in self.array_cache:
            with np.load(self.root/key,allow_pickle=False) as f:self.array_cache[key]={k:f[k] for k in f.files}
            if len(self.array_cache)>64:self.array_cache.popitem(last=False)
        self.array_cache.move_to_end(key)
        return self.array_cache[key]

    def choose_reference(self,i,exclude=None):
        row=self.rows[i];candidates=[j for j in self.by_goal[row['goal_key']] if j!=exclude]
        if exclude is None and row['source']!='imitator_sim' and torch.rand(())<self.config.self_reference_prob:
            return i
        # Only explicitly reviewed goal-equivalence families, never loose verb matches.
        # Close-object variants receive lower action weight; consistency and task
        # contrastive positives still use exact goal identity.
        cross=[j for j in self.by_family.get(row.get('reference_family'),[])
               if j not in (i,exclude) and self.rows[j]['source']!=row['source']
               and self.rows[j]['group']!=row['group'] and self.reference_weight(row,self.rows[j])>0]
        if cross and torch.rand(())<self.raw.get('cross_source_reference_probability',0.):
            return cross[int(torch.randint(len(cross),()))]
        others=[j for j in candidates if j!=i and self.rows[j]['group']!=row['group']]
        if row['source']=='imitator_sim':
            # Preserve the original human-reference -> robot-action bridge.
            human=[j for j in others if self.rows[j]['domain']=='human']
            if human and torch.rand(())<.7: return human[int(torch.randint(len(human),()))]
        cross_scene=[j for j in others if self.rows[j]['scene']!=row['scene']]
        pool=cross_scene or others or candidates or [i]
        return pool[int(torch.randint(len(pool),()))]

    def _video(self,row,idx):
        times=row['start_seconds']+np.asarray(idx)/row['fps']
        rgb=frames_at(row['video'],times,self.config.image_size)
        return torch.from_numpy(rgb).permute(0,3,1,2).float()/255

    def _tokens(self,row,idx):
        key=row['tokens']
        if key not in self.token_cache:
            self.token_cache[key]=np.load(self.root/key,mmap_mode='r',allow_pickle=False)
            if len(self.token_cache)>64:self.token_cache.popitem(last=False)
        self.token_cache.move_to_end(key)
        x=self.token_cache[key]
        if x.shape!=(row['length'],self.config.seq_patches,self.config.feat_dim):raise ValueError('Token cache shape mismatch')
        return torch.from_numpy(np.array(x[idx],copy=True))

    def __getitems__(self,indices):
        # Share two demonstrations within each original-platform task in this
        # microbatch; resample every batch. External/self/cross-source sampling
        # stays query-specific. Keep each query frame exactly once per epoch.
        if not self.share_references or self.stage=='base':return [self[i] for i in indices]
        pairs={};result=[]
        for request in indices:
            i,t=request if isinstance(request,tuple) else (request,None)
            row=self.rows[i];pair=None
            if row['source']=='imitator_sim':
                key=row['goal_key']
                if key not in pairs:
                    first=self.choose_reference(i);second=self.choose_reference(i,exclude=first)
                    pairs[key]=(first,second)
                candidate=pairs[key]
                if all(self.rows[j]['goal_key']==key and self.rows[j]['group']!=row['group'] for j in candidate):
                    pair=candidate
            result.append(self._item((i,t),pair))
        return result

    def __getitem__(self,i):return self._item(i)

    def _item(self,i,reference_pair=None):
        fixed_t=None
        if isinstance(i,tuple):i,fixed_t=i
        c=self.config;r=self.rows[i];T=r['length'];t=int(torch.randint(T-1,())) if fixed_t is None else int(fixed_t)
        if not 0<=t<T:raise IndexError(t)
        if fixed_t is None and r.get('action_space') and self.spaces[r['action_space']].get('representation')=='eef_relative_v3':
            arr=self._arrays(r)
            valid=np.flatnonzero(arr['pose_valid'][:-1].any(-1))
            if not len(valid):raise ValueError('No valid query state')
            t=int(valid[int(torch.randint(len(valid),()))])
        if reference_pair is not None:ri,r2i=reference_pair
        else:
            ri=i if self.stage=='base' else self.choose_reference(i)
            r2i=i if self.stage=='base' else self.choose_reference(i,exclude=ri)
        ref,ref2=self.rows[ri],self.rows[r2i]
        # Multiple horizons discourage solving every example by copying the
        # immediately preceding frame. Every context frame still precedes target.
        gaps=sorted({max(1,round(seconds*r['fps'])) for seconds in (.2,.5,1.,2.) if round(seconds*r['fps'])<T})
        gaps=gaps or [1]
        delta=gaps[int(torch.randint(len(gaps),()))]
        cutoff=int(torch.randint(T-delta,()))
        ctx=np.linspace(0,cutoff,c.frames).round().astype(int);future=cutoff+delta
        actions=np.zeros((c.horizon,self.action_dim),np.float32);mask=np.zeros_like(actions,dtype=bool)
        state=np.zeros(self.state_dim,np.float32);smask=np.zeros(self.state_dim,dtype=bool)
        sid=self.space_ids.get(r.get('action_space'),len(self.spaces));has=r.get('action_space') is not None
        if has:
            spec=self.spaces[r['action_space']];arr=self._arrays(r)
            if spec.get('representation')=='eef_relative_v3':
                from .eef import eef_actions
                a,s,m=eef_actions(r,arr,t,c.horizon)
            elif spec.get('representation')=='relative_v2':
                from .action_representation import relative_actions
                a,s,m=relative_actions(r,arr,t,c.horizon)
            elif r['source']=='egodex':a,s,m=ego_actions(arr,t,c.horizon)
            elif r.get('label_kind')=='future_joint_motion_and_gripper_signal':a,s,m=agibot_motion(arr,t,c.horizon)
            else:a,s,m=arr['actions'][t:t+c.horizon],arr['states'][t],arr['mask'][t:t+c.horizon]
            if not np.isfinite(a[m]).all() or not np.isfinite(s).all():raise ValueError('Non-finite supervision')
            dim=len(s);state[:dim]=(s-np.asarray(spec['state_mean']))/np.asarray(spec['state_std']);smask[:dim]=True
            if r['source']=='egodex':smask[:dim]=arr['valid'][t]
            if r['source']=='oxe_franka':smask[:dim]=arr['state_mask'][t]
            dim=spec['action_dim'];n=len(a)
            actions[:n,:dim]=(a-np.asarray(spec['action_mean']))/np.asarray(spec['action_std']);mask[:n,:dim]=m
            has=bool(mask.any())
        item=dict(actions=torch.from_numpy(actions),action_mask=torch.from_numpy(mask),state=torch.from_numpy(state),
                  expert_key=torch.tensor(self.expert_keys[i]),
                  sampling_bucket_id=torch.tensor(self.sampling_bucket_ids[sampling_bucket(r)]),
                  state_mask=torch.from_numpy(smask),has_actions=torch.tensor(has),space_id=torch.tensor(sid),
                  domain=torch.tensor(int(r['domain']=='robot')),reference_domain=torch.tensor(int(ref['domain']=='robot')),
                  reference2_domain=torch.tensor(int(ref2['domain']=='robot')),goal_id=torch.tensor(self.goal_ids[r['goal_key']]),
                  reference_row_id=torch.tensor(ri),reference2_row_id=torch.tensor(r2i),
                  source_id=torch.tensor(source_id(r)),
                  reference_goal_id=torch.tensor(self.goal_ids[ref['goal_key']]),
                  reference2_goal_id=torch.tensor(self.goal_ids[ref2['goal_key']]),
                  reference_source_id=torch.tensor(source_id(ref)),
                  reference2_source_id=torch.tensor(source_id(ref2)),
                  reference_match_weight=torch.tensor(self.reference_weight(r,ref)),
                  pair_valid=torch.tensor(ref['group']!=ref2['group'] and ref['goal_key']==ref2['goal_key']==r['goal_key']),
                  pair_confidence=torch.tensor(min(ref['goal_confidence'],ref2['goal_confidence'])),
                  loss_weight=torch.tensor(r['loss_weight']),action_dt=torch.tensor(1/r['fps']),
                  future_seconds=torch.tensor(delta/r['fps']),context_indices=torch.from_numpy(ctx),future_index=torch.tensor(future))
        selections=[('reference',ref,np.linspace(0,ref['length']-1,c.frames).round().astype(int)),
                    ('reference2',ref2,np.linspace(0,ref2['length']-1,c.frames).round().astype(int)),('context',r,ctx),('future',r,[future]),('obs',r,[t])]
        needed={'obs'}
        if self.stage!='base':
            needed.add('reference')
            if c.consistency_weight>0 or c.task_weight>0:needed.add('reference2')
            if any(w>0 for w in (c.future_weight,c.rgb_weight,c.order_weight,c.domain_weight)):needed.add('context')
            if c.future_weight>0:needed.add('future')
        for name,row,idx in selections:
            if name not in needed:continue
            if self.cached_tokens:
                if name in ('reference','reference2'):
                    key=row['id']
                    if key not in self.reference_token_cache:
                        self.reference_token_cache[key]=self._tokens(row,idx)
                        if len(self.reference_token_cache)>256:self.reference_token_cache.popitem(last=False)
                    self.reference_token_cache.move_to_end(key)
                    item[name+'_tokens']=self.reference_token_cache[key]
                else:item[name+'_tokens']=self._tokens(row,idx)
            elif self.reference_cache_dir and name in ('reference','reference2'):
                from .reference_cache import cache_path
                key=self.reference_keys[row['id']];path=cache_path(self.reference_cache_dir,key)
                hit=path.exists();shape=(c.frames,c.seq_patches,c.feat_dim)
                if hit:
                    tokens=np.load(path,allow_pickle=False)
                    if tokens.shape!=shape or not np.isfinite(tokens).all():raise ValueError('Invalid reference cache')
                    item[name+'_tokens']=torch.from_numpy(tokens.astype(np.float32))
                    item[name+'_video']=torch.zeros(c.frames,3,c.image_size,c.image_size)
                else:
                    item[name+'_tokens']=torch.zeros(shape)
                    item[name+'_video']=self._video(row,idx)
                item[name+'_cache_hit']=torch.tensor(hit);item[name+'_cache_key']=torch.tensor(key)
            elif name=='obs':item['obs']=self._video(row,idx)[0]
            elif name=='future':item['future_rgb']=self._video(row,idx)[0]
            else:item[name+'_video']=self._video(row,idx)
        if self.cached_tokens and c.rgb_weight>0:item['future_rgb']=self._video(r,[future])[0]
        return item


class FrameQueryDataset(Dataset):
    """Exhaustive query frames for selected sources; reference pool stays separate.

    Each shuffled epoch visits every selected frame once (including padded final
    action windows), matching the original benchmark's frame-based epoch unit.
    """
    def __init__(self,data,sources=('imitator_sim',)):
        self.data=data
        self.indices=[i for i,r in enumerate(data.rows) if sampling_bucket(r) in sources and r.get('action_space')]
        self.ends=np.cumsum([data.rows[i]['length'] for i in self.indices],dtype=np.int64)
        if not len(self.ends):raise ValueError('No action frames in requested sources')
    def __len__(self):return int(self.ends[-1])
    def request(self,index):
        if not 0<=index<len(self):raise IndexError(index)
        k=int(np.searchsorted(self.ends,index,side='right'))
        start=int(self.ends[k-1]) if k else 0
        return self.indices[k],int(index-start)
    def __getitem__(self,index):return self.data[self.request(index)]
    def __getitems__(self,indices):return self.data.__getitems__([self.request(i) for i in indices])


class SourceGoalSampler(Sampler):
    """V2: source -> task/category -> goal -> episode; v1 retains goal balancing.

    Destination-aware descriptions can create hundreds of pick/place goals. They
    must not overwhelm a cleaning task with only a few goal labels.
    """
    def __init__(self,data,count,seed=42,rank=0,ratios=None,batch_size=1,external_support_exponent=0.):
        self.count=count;self.seed=seed;self.rank=rank;self.epoch=0
        self.batch_size=batch_size
        if batch_size<1:raise ValueError('Positive batch size required')
        if not 0<=external_support_exponent<=1:raise ValueError('Support exponent must be in [0,1]')
        self.external_support_exponent=external_support_exponent
        self.groups=collections.defaultdict(lambda:collections.defaultdict(lambda:collections.defaultdict(list)))
        for i,r in enumerate(data.rows):
            source=sampling_bucket(r)
            p=r.get('provenance',{})
            family=str(p.get('category',p.get('task',p.get('verb',r['goal_key'])))) if data.raw.get('sampling_hierarchy')=='source_task_goal_episode' else r['goal_key']
            self.groups[source][family][r['goal_key']].append(i)
        ratios=ratios or dict(imitator_sim=.6,imitator_human=.1,agibot=.15,egodex_pose=.1,egodex_video=.025,epic=.025)
        self.sources=[k for k in sorted(self.groups) if ratios.get(k,0)>0]
        if not self.sources:raise ValueError('No source with positive sampling weight')
        self.weights=torch.tensor([ratios[k] for k in self.sources],dtype=torch.float)
        self.family_weights={};self.goal_weights={}
        for source in self.sources:
            exponent=0. if source.startswith('imitator') else external_support_exponent
            families=self.groups[source]
            self.family_weights[source]=torch.tensor([sum(map(len,groups.values()))**exponent for groups in families.values()],dtype=torch.float)
            for family,groups in families.items():
                self.goal_weights[(source,family)]=torch.tensor([len(ids)**exponent for ids in groups.values()],dtype=torch.float)
    def set_epoch(self,e):self.epoch=e
    def __len__(self):return self.count
    def __iter__(self):
        g=torch.Generator().manual_seed(self.seed+self.rank*1000003+self.epoch*31)
        for begin in range(0,self.count,self.batch_size):
            n=min(self.batch_size,self.count-begin)
            if n==1:choices=torch.multinomial(self.weights,1,generator=g)
            else:
                # Systematic source sampling gives floor/ceil(B*p) examples,
                # preserving rare video sources over batches without fixed rounding.
                positions=(torch.rand((),generator=g)+torch.arange(n))/n
                cdf=(self.weights/self.weights.sum()).cumsum(0);cdf[-1]=1.
                choices=torch.searchsorted(cdf,positions)[torch.randperm(n,generator=g)]
            for sid in choices:
                source=self.sources[int(sid)]
                families=list(self.groups[source]);family=families[int(torch.multinomial(self.family_weights[source],1,generator=g))]
                goals=list(self.groups[source][family]);goal=goals[int(torch.multinomial(self.goal_weights[(source,family)],1,generator=g))]
                ids=self.groups[source][family][goal];yield ids[int(torch.randint(len(ids),(),generator=g))]
