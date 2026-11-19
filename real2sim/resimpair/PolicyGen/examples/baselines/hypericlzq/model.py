"""Goal-conditioned hypernetwork with native-space action adapters and causal video SSL.

Goal IDs are used ONLY to sample references and construct contrastive positives.
The deployed model never receives a task ID or task-specific learned embedding.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .modules.video_program_encoder import VideoProgramEncoder
from .modules.hyper_compiler import ContextTransformer, HyperDecoder
from .modules.block_hyperopt_refiner import BlockHyperOptRefiner
from .targets.scratch_target import ScratchTarget


@dataclass
class ZQConfig:
    feat_dim: int = 1024
    seq_patches: int = 32
    frames: int = 10
    horizon: int = 16
    image_size: int = 224
    d: int = 256
    d_prog: int = 64
    d_emb: int = 32
    events: int = 4
    enc_layers: int = 4
    context_layers: int = 4
    nhead: int = 4
    dit_dim: int = 256
    dit_depth: int = 6
    rank: int = 8
    ada_rank: int = 16
    n_atoms: int = 16
    refiner_steps: int = 2
    sample_steps: int = 10
    rgb_size: int = 32
    action_latent_dim: int = 32
    self_reference_prob: float = 0.2
    action_weight: float = 1.0
    future_weight: float = 0.05
    rgb_weight: float = 0.0
    order_weight: float = 0.02
    domain_weight: float = 0.0
    task_weight: float = 0.02
    consistency_weight: float = 0.02
    parameter_weight: float = 0.0  # legacy checkpoints; staged training opts in


def masked_action_mse(pred, target, mask):
    # Equal contribution per trajectory, regardless of native action dimensionality
    # or how many real timesteps remain before the episode terminates.
    error = (pred - target).square().masked_fill(~mask, 0.)
    return (error.sum((1, 2)) / mask.sum((1, 2)).clamp_min(1)).mean()


class HyperICLZQ(nn.Module):
    def __init__(self, config: ZQConfig, spaces: dict, tokenizer=None):
        super().__init__()
        c = self.config = config
        if c.d % c.nhead or c.dit_dim % c.nhead or not 1 <= c.frames <= 64:
            raise ValueError("Attention dimensions must divide nhead; 1 <= frames <= 64")
        if not 0 <= c.self_reference_prob <= 1:
            raise ValueError('self_reference_prob must be between zero and one')
        if min(c.horizon, c.dit_depth, c.enc_layers, c.context_layers, c.sample_steps, c.rgb_size) < 1:
            raise ValueError("Model sizes/steps must be positive")
        if min(c.action_weight, c.future_weight, c.rgb_weight, c.order_weight, c.domain_weight, c.task_weight, c.consistency_weight, c.parameter_weight) < 0:
            raise ValueError("Loss weights must be non-negative")
        self.spaces = spaces
        self.space_names = sorted(spaces)
        self.action_dim = max((int(s["action_dim"]) for s in spaces.values()), default=1)
        self.state_dim = max(1, max((int(s.get("state_dim", 0)) for s in spaces.values()), default=0))
        self.tokenizer = tokenizer
        if tokenizer is not None:
            tokenizer.requires_grad_(False)
        self.E = VideoProgramEncoder(c.feat_dim, c.d, c.d_prog, c.d_emb, c.events,
                                     layers=c.enc_layers, nhead=c.nhead)
        # The baseline future head sees a bidirectional full clip; it is not used here.
        for name in ("delta_emb", "fut_mlp", "dom_inv"):
            getattr(self.E, name).requires_grad_(False)
        self.E.dom_emb.requires_grad_(c.domain_weight > 0)
        self.E.emb_pool.requires_grad_(c.domain_weight > 0)
        self.E.emb_proj.requires_grad_(c.domain_weight > 0)
        if c.action_latent_dim < self.action_dim:
            raise ValueError('action_latent_dim must cover the largest native action space')
        self.action_in = nn.ModuleList([nn.Linear(spaces[n]['action_dim'], c.action_latent_dim) for n in self.space_names])
        self.action_out = nn.ModuleList([nn.Linear(c.action_latent_dim, spaces[n]['action_dim']) for n in self.space_names])
        self.target = ScratchTarget(c.action_latent_dim, c.d, c.horizon, c.dit_dim, c.dit_depth,
                                    c.nhead, c.rank, c.ada_rank, c.n_atoms, c.sample_steps)
        specs = self.target.layer_specs()
        self.T = ContextTransformer(c.d, self.target.n_blocks, c.events, c.d_prog,
                                    layers=c.context_layers, nhead=c.nhead)
        self.T.fb_gate.requires_grad_(False)
        self.R = BlockHyperOptRefiner(c.d, self.target.n_blocks, c.nhead, c.refiner_steps)
        self.R.fb_ca.requires_grad_(False)
        self.D = HyperDecoder(specs, c.d, c.events, c.n_atoms)
        self.D.layer_energy.requires_grad_(False)
        # A and B both zero makes BA's derivative zero forever. Random A, zero B
        # still yields an exactly zero initial residual while permitting learning.
        initialized = set()
        with torch.no_grad():
            for spec in specs:
                key = self.D._res_key[spec.name]
                if key not in initialized:
                    head = self.D.res_head[key]
                    n = spec.rank * spec.in_f
                    nn.init.normal_(head.bias[:n], std=1 / math.sqrt(spec.in_f))
                    initialized.add(key)
        self.space_embedding = nn.Embedding(len(spaces) + 1, c.d)  # final ID = unknown/unlabeled
        self.domain_embedding = nn.Embedding(2, c.d)
        self.obs_feature_proj = nn.Sequential(nn.Linear(c.feat_dim,c.d),nn.LayerNorm(c.d))
        self.obs_attention = nn.MultiheadAttention(c.d,c.nhead,batch_first=True)
        self.obs_queries = nn.Parameter(torch.randn(1,4,c.d)*.02)
        self.state_enc = nn.Sequential(nn.Linear(2*self.state_dim, c.d), nn.GELU(), nn.Linear(c.d, c.d))
        self.obs_proj = nn.Sequential(nn.Linear(3*c.d+1, c.d), nn.GELU(), nn.LayerNorm(c.d))
        self.phase_query = nn.Linear(c.d,c.d)
        self.phase_key = nn.Linear(c.d,c.d)
        self.task_projector = nn.Sequential(nn.Linear(c.d,c.d),nn.GELU(),nn.Linear(c.d,128))
        self.gap_enc = nn.Sequential(nn.Linear(1, c.d), nn.GELU(), nn.Linear(c.d, c.d))
        self.future_head = nn.Sequential(nn.LayerNorm(c.d), nn.Linear(c.d, c.d), nn.GELU(), nn.Linear(c.d, c.feat_dim))
        self.rgb_head = (nn.Sequential(nn.LayerNorm(c.d), nn.Linear(c.d, 3*c.rgb_size*c.rgb_size), nn.Sigmoid())
                         if c.rgb_weight > 0 else None)
        self.order_head = nn.Linear(c.d, 2) if c.order_weight > 0 else None
        self.future_head.requires_grad_(c.future_weight > 0)
        self.gap_enc.requires_grad_(c.future_weight > 0 or c.rgb_weight > 0)
        self.training_stage = 'joint'
        self._joint_trainable = {n for n,p in self.named_parameters() if p.requires_grad}

    def set_training_stage(self, stage):
        """Base learns actions without hypernet; hyper freezes that trained base."""
        if stage not in ('base', 'hyper', 'joint'):
            raise ValueError(stage)
        self.training_stage = stage
        for n,p in self.named_parameters():
            p.requires_grad_(stage == 'joint' and n in self._joint_trainable)
        if stage == 'base':
            for name in ('action_in','action_out','space_embedding','obs_feature_proj',
                         'obs_attention','state_enc','obs_proj'):
                getattr(self,name).requires_grad_(True)
            self.obs_queries.requires_grad_(True)
            self.target.set_stage('A', True)
        elif stage == 'hyper':
            for name in ('E','T','R','D','phase_query','phase_key','task_projector',
                         'domain_embedding','future_head','gap_enc','order_head','rgb_head'):
                module=getattr(self,name,None)
                if module is not None:
                    for n,p in module.named_parameters():
                        p.requires_grad_(name+'.'+n in self._joint_trainable)
            self.target.set_stage('D', False)

    def _tokens(self, batch, kind):
        key = kind + "_tokens"
        if kind+'_cache_hit' in batch:
            hit=batch[kind+'_cache_hit'];result=batch[key].clone()
            missing=torch.where(~hit)[0]
            if len(missing):
                if self.tokenizer is None:raise ValueError('Uncached reference without tokenizer')
                with torch.no_grad():tokens=self.tokenizer(batch[kind+'_video'][missing]).detach()
                result[missing]=tokens.to(result.dtype)
                root=getattr(self,'reference_cache_dir',None)
                if root:
                    from .reference_cache import save_tokens
                    for idx,tok in zip(missing.tolist(),tokens):save_tokens(root,batch[kind+'_cache_key'][idx],tok)
            return result.detach()
        if key in batch:
            tokens=batch[key].detach()
            return tokens.float() if tokens.device.type=='cpu' else tokens
        if self.tokenizer is None:
            raise ValueError("Supply cached frozen tokens or a frozen tokenizer")
        video = (batch['obs'][:,None] if kind=='obs' else
                 batch["future_rgb"][:, None] if kind == "future" else batch[kind + "_video"])
        with torch.no_grad():
            return self.tokenizer(video).detach()

    def encode_program(self, tokens, domains):
        if tokens.device.type=='cpu':tokens=tokens.float()
        return self.E(tokens,domain=domains)

    def compile_program(self, enc, domains, space_ids):
        context = self.T.human_tokens(enc)
        # Target action space is separate from the visual reference's domain.
        identity = self.space_embedding(space_ids) + self.domain_embedding(domains)
        context = torch.cat([context[:, :1] + identity[:, None], context[:, 1:]], dim=1)
        S = self.T(None, context)
        return self.R(S, context)

    def encode(self, tokens, domains, space_ids):
        enc=self.encode_program(tokens,domains)
        return enc,self.compile_program(enc,domains,space_ids)

    def observation(self, batch):
        features=self.obs_feature_proj(self._tokens(batch,'obs')[:,0])
        obs=self.obs_attention(self.obs_queries.expand(len(features),-1,-1),features,features,need_weights=False)[0].mean(1)
        state = self.state_enc(torch.cat([batch["state"] * batch["state_mask"], batch["state_mask"].float()], -1))
        sid = self.space_embedding(batch["space_id"])
        # Action duration is explicit: the same displacement at 10 vs 30 Hz has
        # different dynamics. Units and absolute/delta conventions live in spaces.
        dt = torch.log1p(batch["action_dt"])[:, None]
        return self.obs_proj(torch.cat([obs, state, sid, dt], -1))

    def parameters_for(self, enc, S, h, inverse=None):
        score=torch.einsum('bd,bkd->bk',self.phase_query(h),self.phase_key(enc['Z_evt'])) / math.sqrt(self.config.d)
        return self.D(S,score.softmax(-1),inv=inverse)

    @staticmethod
    def unique_first(keys):
        unique,inverse=torch.unique(keys,dim=0,return_inverse=True)
        first=torch.full((len(unique),),len(keys),dtype=torch.long,device=keys.device)
        first.scatter_reduce_(0,inverse,torch.arange(len(keys),device=keys.device),reduce='amin')
        return first,inverse

    def compile_reference(self, enc, batch, kind):
        domain=kind+'_domain';identity=kind+'_row_id'
        if identity not in batch:
            return self.compile_program(enc,batch[domain],batch['space_id']),None
        # A shared visual reference can target DIFFERENT action spaces; keep
        # these compilers separate. State-dependent phase mixing remains per query.
        first,inverse=self.unique_first(torch.stack([batch[identity],batch['space_id'],batch[domain]],dim=1))
        compact={k:v[first] for k,v in enc.items()}
        return self.compile_program(compact,batch[domain][first],batch['space_id'][first]),inverse

    def velocity(self, x, t, h, P, space_ids):
        latent=x.new_zeros(len(x),x.shape[1],self.config.action_latent_dim)
        for sid,name in enumerate(self.space_names):
            idx=torch.where(space_ids==sid)[0]
            if len(idx): latent=latent.index_copy(0,idx,self.action_in[sid](x[idx,:,:self.spaces[name]['action_dim']]).to(latent.dtype))
        v=self.target.dit(latent,t,h,P)
        native=x.new_zeros(x.shape)
        for sid,name in enumerate(self.space_names):
            idx=torch.where(space_ids==sid)[0]; dim=self.spaces[name]['action_dim']
            if len(idx): native=native.index_copy(0,idx,F.pad(self.action_out[sid](v[idx]),(0,self.action_dim-dim)).to(native.dtype))
        return native

    @staticmethod
    def weighted_error(pred,target,mask,weight):
        error=(pred-target).square().masked_fill(~mask,0).sum((1,2))/mask.sum((1,2)).clamp_min(1)
        # Mean per labeled example; weights express confidence, not sample frequency.
        return (error*weight).mean()

    def video_predictions(self, tokens, enc, S, seconds):
        gap = self.gap_enc(torch.log1p(seconds)[:, None])
        summary = S.mean(1) + enc["frame_g"][:, -1] + gap
        last_patches = self.E.in_proj(tokens[:, -1])
        out = {"features": self.future_head(last_patches + summary[:, None])}
        if self.rgb_head is not None:
            out["rgb"] = self.rgb_head(summary).reshape(-1, 3, self.config.rgb_size, self.config.rgb_size)
        return out

    def forward(self, batch, expert_supervisor=None):
        c = self.config
        if self.training_stage == 'base':
            selected=batch['has_actions'].bool()
            if not selected.any():raise ValueError('Base stage requires action labels')
            b={k:v[selected] for k,v in batch.items()}
            mask=b['action_mask'];actions=b['actions'].masked_fill(~mask,0.)
            noise=torch.randn_like(actions).masked_fill(~mask,0.)
            x,target,t=self.target.head.train_pair(actions,noise,torch.rand(len(actions),device=actions.device))
            velocity=self.velocity(x,t,self.observation(b),None,b['space_id'])
            action=self.weighted_error(velocity,target,mask,b['loss_weight']*b.get('action_importance',1.))
            return {'loss':c.action_weight*action,'action':action}
        # Reuse frozen features between action, consistency and contrastive losses.
        # The same full reference used to pass through DINO multiple times here.
        batch=dict(batch)
        kinds=set()
        if c.action_weight>0 and batch['has_actions'].any():
            kinds.update(('reference','obs'))
            if c.consistency_weight>0 and batch['pair_valid'].any():kinds.add('reference2')
        if c.task_weight>0:kinds.update(('reference','reference2'))
        if any(w>0 for w in (c.future_weight,c.rgb_weight,c.order_weight,c.domain_weight)):kinds.add('context')
        if c.future_weight>0:kinds.add('future')
        for kind in sorted(kinds):
            batch[kind+'_tokens']=self._tokens(batch,kind)
            batch.pop(kind+'_cache_hit',None)
        losses = {}
        diagnostics = {}
        expert_loss = None
        programs={}
        def program(kind, domain_key):
            if kind not in programs:
                if kind+'_row_id' in batch:
                    first,inverse=self.unique_first(torch.stack([batch[kind+'_row_id'],batch[domain_key]],dim=1))
                    compact=self.encode_program(batch[kind+'_tokens'][first],batch[domain_key][first])
                    programs[kind]={k:v[inverse] for k,v in compact.items()}
                else:programs[kind]=self.encode_program(batch[kind+'_tokens'],batch[domain_key])
            return programs[kind]
        selected = batch["has_actions"].bool()
        if c.action_weight > 0 and selected.any():
            b = {k: v[selected] for k, v in batch.items()}
            enc={k:v[selected] for k,v in program('reference','reference_domain').items()}
            S,inverse=self.compile_reference(enc,b,'reference')
            h = self.observation(b)
            P = self.parameters_for(enc,S,h,inverse)
            if c.parameter_weight > 0:
                from .stability import parameter_penalty
                first = self.unique_first(inverse[:, None])[0] if inverse is not None else None
                losses['parameter'], diagnostics['delta_max_ratio'] = parameter_penalty(self.target, P, first)
            mask = b["action_mask"]
            actions = b["actions"].masked_fill(~mask, 0.)
            noise = torch.randn_like(actions).masked_fill(~mask, 0.)
            u = torch.rand(actions.shape[0], device=actions.device)
            x, target, t = self.target.head.train_pair(actions, noise, u)
            velocity = self.velocity(x,t,h,P,b['space_id'])
            # Optional offline event weights affect only action supervision. They
            # never enter observation/reference conditioning or consistency loss.
            losses["action"] = self.weighted_error(velocity,target,mask,b['loss_weight']*b.get('reference_match_weight',1.)*b.get('action_importance',1.))
            if expert_supervisor is not None:
                expert_loss, expert_metrics = expert_supervisor.losses(self,b,S,inverse,velocity,x,t)
                diagnostics.update(expert_metrics)
            per_example=(velocity-target).square().masked_fill(~mask,0.).sum((1,2))/mask.sum((1,2)).clamp_min(1)
            for sid,name in enumerate(self.space_names):
                which=b['space_id']==sid
                if which.any():
                    diagnostics['action_raw/'+name]=per_example[which].mean().detach()
                    diagnostics['action_count/'+name]=which.sum().detach()
            if c.consistency_weight>0 and b['pair_valid'].any():
                enc2={k:v[selected] for k,v in program('reference2','reference2_domain').items()}
                S2,inverse2=self.compile_reference(enc2,b,'reference2')
                v2=self.velocity(x,t,h,self.parameters_for(enc2,S2,h,inverse2),b['space_id'])
                losses['consistency']=self.weighted_error(v2,velocity.detach(),mask,b['pair_valid'].float()*b['pair_confidence'])
        if c.task_weight>0:
            # Multi-positive contrastive loss BEFORE target-body conditioning.
            e1=program('reference','reference_domain')
            e2=program('reference2','reference2_domain')
            z1=F.normalize(self.task_projector(e1['z_eff']+e1['Z_evt'].mean(1)),dim=-1)
            z2=F.normalize(self.task_projector(e2['z_eff']+e2['Z_evt'].mean(1)),dim=-1)
            z=torch.cat([z1,z2])
            ids=torch.cat([batch.get('reference_goal_id',batch['goal_id']),batch.get('reference2_goal_id',batch['goal_id'])])
            eye=torch.eye(len(z),device=z.device,dtype=torch.bool)
            positives=(ids[:,None]==ids[None]) & ~eye
            source=torch.cat([batch.get('reference_source_id',batch['source_id']),batch.get('reference2_source_id',batch['source_id'])])
            allowed=((source[:,None]==source[None]) | positives) & ~eye
            logits=(z@z.T/.1).masked_fill(~allowed,-1e4)
            logp=logits-logits.logsumexp(-1,keepdim=True)
            per_view=-(logp*positives).sum(-1).div(positives.sum(-1).clamp_min(1))
            losses['task']=(per_view*batch['pair_confidence'].repeat(2)).mean()
        if any(w > 0 for w in (c.future_weight, c.rgb_weight, c.order_weight, c.domain_weight)):
            tokens = self._tokens(batch, "context")
            enc, S = self.encode(tokens, batch["domain"], batch["space_id"])
            if c.future_weight > 0 or c.rgb_weight > 0:
                pred = self.video_predictions(tokens, enc, S, batch["future_seconds"])
                if c.future_weight > 0:
                    target = self._tokens(batch, "future")[:, 0]
                    # Fixed teacher features, patch-wise cosine distance. Targets
                    # cannot change to follow the student's current prediction.
                    losses["future"] = (1 - F.cosine_similarity(pred["features"], target, dim=-1)).mean()
                    diagnostics['future_copy_baseline']=(1-F.cosine_similarity(tokens[:,-1],target,dim=-1)).mean().detach()
                    diagnostics['future_excess_over_copy']=(losses['future']-diagnostics['future_copy_baseline']).detach()
                if c.rgb_weight > 0:
                    rgb = F.interpolate(batch["future_rgb"], (c.rgb_size, c.rgb_size), mode="area")
                    losses["rgb"] = F.l1_loss(pred["rgb"], rgb)
            if self.order_head is not None:
                # Train on both directions with equal weight; ignore repeated/static
                # token sequences for which reversal has exactly the same input.
                valid = (tokens - tokens.flip(1)).abs().flatten(1).amax(1) > 1e-6
                if valid.any():
                    _, backward = self.encode(tokens[valid].flip(1), batch["domain"][valid], batch["space_id"][valid])
                    logits = torch.cat([self.order_head(S[valid].mean(1)), self.order_head(backward.mean(1))])
                    n = int(valid.sum())
                    labels = torch.cat([torch.zeros(n, device=S.device), torch.ones(n, device=S.device)]).long()
                    losses["order"] = F.cross_entropy(logits, labels)
            if c.domain_weight > 0:
                losses["domain"] = F.cross_entropy(self.E.dom_emb(enc["z_emb"]), batch["domain"])
        if not losses:
            raise ValueError("This batch has no enabled objective; enable video losses or supply labeled data")
        loss = sum(value * getattr(c, key + "_weight") for key, value in losses.items())
        if expert_loss is not None:
            loss = loss + expert_loss
        return {"loss": loss, **losses, **diagnostics}

    @torch.no_grad()
    def sample_actions(self, batch, compiled=None, policy_parameters=None, observation_latent=None):
        """One de-normalized [H,A] tensor in the manifest's action representation.

        A v2 tensor contains relative targets. Convert Panda targets with
        panda_to_native and the raw query state before sending to the controller.

        Requires reference tokens/video, its domain, a known TARGET space_id,
        current obs, normalized state plus mask (or zeros), and action_dt.
        policy_parameters bypasses the entire parameter-generation path, allowing
        an evaluator to hold the SAME generated parameters over many control steps.
        observation_latent may reuse the current observation encoding; it must not
        be a stale encoding from the parameter-refresh step.
        """
        if ((batch["space_id"] < 0) | (batch["space_id"] >= len(self.space_names))).any():
            raise ValueError("Inference requires a known target action_space")
        h = self.observation(batch) if observation_latent is None else observation_latent
        if self.training_stage == 'base':
            P=None
        elif policy_parameters is None:
            if compiled is None:
                enc,S=self.encode(self._tokens(batch,"reference"),batch["reference_domain"],batch["space_id"])
            else:enc,S=compiled
            P=self.parameters_for(enc,S,h)
        else:P=policy_parameters
        dims = torch.tensor([self.spaces[self.space_names[int(s)]]['action_dim'] for s in batch['space_id']], device=h.device)
        mask = torch.arange(self.action_dim, device=h.device)[None, None] < dims[:, None, None]
        # Single-arm EEF data keeps a padded second arm in the 14-D schema.
        # It must remain zero during sampling, just as it does during training;
        # injecting noise into that unused slot changes the shared action input.
        supported=torch.ones((len(dims),self.action_dim),device=h.device,dtype=torch.bool)
        for i,sid in enumerate(batch['space_id'].tolist()):
            spec=self.spaces[self.space_names[sid]]
            if 'action_valid_count' in spec:
                supported[i,:spec['action_dim']]=torch.tensor(spec['action_valid_count'],device=h.device)>0
        mask=mask&supported[:,None]
        x = torch.randn(h.shape[0], self.config.horizon, self.action_dim, device=h.device) * mask
        for i in reversed(range(self.config.sample_steps)):
            t = torch.full((len(x),), (i+1) / self.config.sample_steps * self.target.head.t_scale, device=x.device)
            x = (x - self.velocity(x,t,h,P,batch['space_id']) / self.config.sample_steps) * mask
        result = []
        for i, sid in enumerate(batch["space_id"].tolist()):
            spec = self.spaces[self.space_names[sid]]
            n = spec["action_dim"]
            mean = x.new_tensor(spec.get("action_mean", [0.] * n))
            std = x.new_tensor(spec.get("action_std", [1.] * n))
            result.append((x[i, :, :n] * std + mean).masked_fill(~supported[i,:n],0.))
        return result

    def parameter_report(self):
        report = {}
        for name, module in self.named_children():
            report[name] = {"all": sum(p.numel() for p in module.parameters()),
                            "trainable": sum(p.numel() for p in module.parameters() if p.requires_grad)}
        report["TOTAL"] = {"all": sum(p.numel() for p in self.parameters()),
                           "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad)}
        report["target_split"] = {
            "base": sum(p.numel() for n, p in self.target.named_parameters() if "atom_" not in n),
            "atoms": sum(p.numel() for n, p in self.target.named_parameters() if "atom_" in n)}
        return report
