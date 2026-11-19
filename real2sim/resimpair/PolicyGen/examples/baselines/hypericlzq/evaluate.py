"""Use the existing L0-L3 simulator and success metrics with a ZQ checkpoint.

Evaluation task trajectories are NEVER used for normalization; use training stats.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .model import HyperICLZQ,ZQConfig
from .modules.frame_tokenizer import FrozenFrameTokenizer,format_video


def _content_sha256(value):
    if torch.is_tensor(value):
        value=value.detach().cpu().numpy()
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


class ResetAuditEnv:
    """Transparent environment proxy that records reset state/RGB hashes."""
    def __init__(self,env):
        self.env=env;self.initial_states=[]
    def __getattr__(self,name):
        return getattr(self.env,name)
    def reset(self,*args,**kwargs):
        obs,info=self.env.reset(*args,**kwargs)
        if not isinstance(obs,dict) or 'state' not in obs or 'rgb' not in obs:
            raise ValueError('Evaluation reset audit expects state and rgb observations')
        self.initial_states.append(dict(seed=kwargs.get('seed'),
            state_sha256=_content_sha256(obs['state']),rgb_sha256=_content_sha256(obs['rgb'])))
        return obs,info


class EvalAgent(nn.Module):
    _use_foundation=True
    def __init__(self,model,hyper_update_interval=32):
        super().__init__();self.model=model
        if hyper_update_interval<0:raise ValueError('hyper_update_interval must be >=0; 0 means once per episode')
        self.hyper_update_interval=hyper_update_interval
        self.space_name=next(n for n in ('panda_eef14_v3','panda_delta_joint16_v2','imitator_panda_joint16') if n in model.spaces)
        self.sid=model.space_names.index(self.space_name)
        self.reset_diagnostics();self.clear_cache()
    def reset_diagnostics(self):
        self.ik_targets=0;self.ik_failures=0
        self.policy_queries=0;self.hyper_parameter_updates=0
        self.hyper_update_steps_by_episode=[]
    def clear_cache(self):
        self.reference=None
        self.compiled=None
        self.cached_policy_parameters=None
        self.cached_parameter_batch_size=None
        self.control_step=0
    @torch.no_grad()
    def prepare_for_eval(self,human_video,robot_obs=None,**kwargs):
        # Clear even if a caller starts a new episode without clear_cache().
        self.clear_cache()
        if getattr(self.model,'training_stage','joint')=='base':
            self.reference=torch.empty(len(human_video),0,0,0,device=human_video.device)
            self.hyper_update_steps_by_episode.append([])
            return
        self.reference=self.model.tokenizer(human_video)
        B=len(self.reference);dev=self.reference.device
        self.compiled=self.model.encode(self.reference,torch.zeros(B,device=dev,dtype=torch.long),torch.full((B,),self.sid,device=dev,dtype=torch.long))
        self.hyper_update_steps_by_episode.append([])
    @torch.no_grad()
    def get_action(self,obs):
        if self.reference is None:raise RuntimeError('Call prepare_for_eval at each episode start')
        image=format_video(obs['rgb'])[:,-1]
        image=F.interpolate(image,(self.model.config.image_size,)*2,mode='bilinear',align_corners=False)
        state=obs['state'];state=state[:,-1] if state.ndim==3 else state
        B=len(state);full=state.new_zeros(B,self.model.state_dim);full[:,:18]=state
        mask=torch.zeros_like(full,dtype=torch.bool);mask[:,:18]=True
        reference=self.reference.expand(B,-1,-1,-1) if len(self.reference)==1 else self.reference
        batch=dict(reference_tokens=reference,reference_domain=torch.zeros(B,dtype=torch.long,device=state.device),
                   space_id=torch.full((B,),self.sid,dtype=torch.long,device=state.device),obs=image,
                   state=full,state_mask=mask,action_dt=torch.full((B,),1/30,device=state.device))
        # Native action values. The processor below deliberately does not denormalize twice.
        # The evaluator below queries once per control step (temporal_agg=True).
        # Keep h current for closed-loop control, but refresh the FULL decoded P
        # only at the selected control-step boundary. Caching S alone is not enough.
        h=self.model.observation(batch)
        if self.cached_policy_parameters is not None and self.cached_parameter_batch_size!=B:
            raise ValueError('Episode batch size changed; reset the episode before changing it')
        refresh=(self.cached_policy_parameters is None or
                 (self.hyper_update_interval>0 and self.control_step%self.hyper_update_interval==0))
        if refresh and getattr(self.model,'training_stage','joint')!='base':
            enc,S=self.compiled
            expanded={k:v.expand(B,*v.shape[1:]) for k,v in enc.items()}
            self.cached_policy_parameters=self.model.parameters_for(expanded,S.expand(B,-1,-1),h)
            self.cached_parameter_batch_size=B
            self.hyper_parameter_updates+=1
            self.hyper_update_steps_by_episode[-1].append(self.control_step)
        actions=torch.stack(self.model.sample_actions(batch,policy_parameters=self.cached_policy_parameters,observation_latent=h))
        self.control_step+=1;self.policy_queries+=1
        spec=self.model.spaces[self.space_name]
        if spec.get('representation')=='eef_relative_v3':
            from .eef import panda_eef_to_native
            raw=state[:,:18]*state.new_tensor(spec['state_std'])+state.new_tensor(spec['state_mean'])
            actions,self.last_ik_stats=panda_eef_to_native(actions,raw)
            self.ik_targets=getattr(self,'ik_targets',0)+self.last_ik_stats['targets']
            self.ik_failures=getattr(self,'ik_failures',0)+self.last_ik_stats['failed']
        elif spec.get('representation')=='relative_v2':
            from .action_representation import panda_to_native
            raw=state[:,:18]*state.new_tensor(spec['state_std'])+state.new_tensor(spec['state_mean'])
            actions=panda_to_native(actions,raw)
        return actions


def summarize_eval_metrics(metrics, *, save_episode_values=False):
    """Summarize metrics and optionally retain the raw per-episode values.

    The raw values make paired repeated evaluations auditable; older callers
    keep the same mean/std-only JSON schema by leaving the option disabled.
    Some conditional trajectory metrics (for example success-only TSS) have
    fewer entries than the number of evaluated episodes, so do not fabricate
    values for episodes where those metrics are undefined.
    """
    result = {}
    for key, value in metrics.items():
        array = np.asarray(value)
        item = dict(mean=float(array.mean()), std=float(array.std()))
        if save_episode_values:
            item['per_episode'] = array.tolist()
        result[key] = item
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--suite',choices=['seen','unseen','seen_heldout'],default='unseen')
    p.add_argument('--num-episodes',type=int,default=10);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--reference-mode',choices=['correct','wrong_task','blank'],default='correct')
    p.add_argument('--device',default='cuda');p.add_argument('--max-steps',type=int,default=500)
    p.add_argument('--sim-backend',default='physx_cpu');p.add_argument('--shader',default='rt-fast')
    p.add_argument('--limit-envs',type=int,default=0,help='Smoke tests only; 0 evaluates the entire suite')
    p.add_argument('--gpu-memory-fraction',type=float,default=.18)
    p.add_argument('--save-episode-metrics', action='store_true',
                   help='Include raw metric arrays in each environment result for paired repeated-evaluation analysis')
    p.add_argument('--hyper-update-interval',type=int,default=32,
                   help='Hold generated policy parameters for N control steps; 0=entire episode, 1=legacy per-step ablation')
    args=p.parse_args();torch.manual_seed(args.seed);random.seed(args.seed);np.random.seed(args.seed)
    if args.hyper_update_interval<0:p.error('--hyper-update-interval must be non-negative')
    if args.device.startswith('cuda'):
        gpu=torch.device(args.device).index
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction,torch.cuda.current_device() if gpu is None else gpu)
    # Optional simulator stack is imported only after argument parsing.
    from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
    from examples.baselines.lerobot_dataset.lerobot_paired_dataset import InputMode
    from examples.baselines.lerobot_dataset.evaluate_processor import HumanVideoSimEvaluateProcessor,HumanVideoSimEvaluateProcessorConfig
    from examples.baselines.diffusion_policy.eval_dp_imitator import load_eval_config,extract_base_env_name,extract_level,make_eval_envs_with_level,clear_l_level,evaluate_with_task_encoder
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False);c=ZQConfig(**ck['config'])
    tok=FrozenFrameTokenizer(backend='dinov2_hf',seq_patches=c.seq_patches,feat_dim=c.feat_dim,image_size=c.image_size,
                            hf_model_name=ck['args'].get('tokenizer_name','facebook/dinov2-large'),
                            hf_cache_dir=ck['args'].get('hf_cache_dir','/mnt/sdf/dataset/zhouxunzhe/huggingface'))
    state=ck['model']
    if any(k.startswith('obs_cnn.') for k in state):
        # Reconstruct the isolated CNN observation branch used by the matched
        # full-policy experiments. Human wrist auxiliary heads are training-only.
        from .observation_full_policy import CNNObservationPolicy
        model=CNNObservationPolicy(c,ck['action_spaces'],tok).to(args.device)
    else:
        model=HyperICLZQ(c,ck['action_spaces'],tok).to(args.device)
    inference_state={k:v for k,v in state.items()
                     if not k.startswith(('human_wrist_aux.', 'human_pinch_aux.', 'hot3d_aux.'))}
    result=model.load_state_dict(inference_state,strict=False)
    if result.unexpected_keys or any(not k.startswith('tokenizer.') for k in result.missing_keys):raise ValueError(str(result))
    model.set_training_stage(ck.get('training_stage','joint'))
    agent=EvalAgent(model,args.hyper_update_interval).eval();spec=ck['action_spaces'][agent.space_name]
    class Processor(HumanVideoSimEvaluateProcessor):
        def _load_sim_normalizer(self):
            # Keep task lookup only. Do not load unseen dataset statistics.
            configs=json.loads(Path(self.config.sim_dataset_file).read_text())
            for i,item in enumerate(configs):
                name=item['repo_id'];self.repo_id_to_dataset_idx[name]=i;self.sim_task_to_dataset_idx[name]=i
                self.dataset_idx_to_repo_id[i]=name;self.dataset_idx_to_root[i]=Path(self.config.sim_root)/item['root']
        def normalize_state_rgb(self,state,rgb,sim_task_id=None):
            q=self._extract_sim_qpos(state)
            return (q-q.new_tensor(spec['state_mean']))/q.new_tensor(spec['state_std']),rgb.float()/255
        def denormalize_action(self,action,sim_task_id=None):return action
        def get_video(self,sim_task_id=None,num_envs=1):
            target=sim_task_id
            if args.reference_mode=='wrong_task':
                current=self._resolve_human_task_id(sim_task_id)
                candidates=sorted({self._resolve_human_task_id(k) for k in self.sim_task_to_dataset_idx})
                if len(candidates)<2:raise ValueError('Wrong-task diagnostic needs at least two tasks')
                wrong=candidates[(candidates.index(current)+1)%len(candidates)]
                target=next(k for k in sorted(self.sim_task_to_dataset_idx) if self._resolve_human_task_id(k)==wrong)
            self.last_reference_task=self._resolve_human_task_id(target)
            video=super().get_video(target,num_envs)
            video=torch.zeros_like(video) if args.reference_mode=='blank' else video
            self.reference_sha256s.append(_content_sha256(video))
            return video
    base=Path(__file__).resolve().parents[1]/'lerobot_dataset'
    suite='seen' if args.suite=='seen_heldout' else args.suite
    cfg=base/'config/exp_configs'
    human=cfg/('human_eval_config_seen_heldout.json' if args.suite=='seen_heldout' else f'human_test_config_{suite}.json')
    processor=Processor(HumanVideoSimEvaluateProcessorConfig(
        human_root='/mnt/sdf/engram_sz_DataCollection/imitator_zed2i_224',human_dataset_file=str(human),
        human_split='test' if args.suite=='seen_heldout' else 'train',human_include_depth=False,human_num_frames=c.frames,
        human_image_size=(c.image_size,c.image_size),human_task_description_file=str(base/'task_desc/human_desc.json'),
        sim_root='/mnt/sdf/engram_sz_DataCollection/IG-10K-Dataset/imitator_sim_v1_zed2i',sim_dataset_file=str(cfg/f'sim_test_config_{suite}.json'),
        task_mapping_file=str(base/'task_mapping.json')))
    envs=load_eval_config(str(base/f'eval/exp_list/{suite}_5tasks_env_list.txt'))['environments']
    if args.limit_envs:envs=envs[:args.limit_envs]
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);results=[]
    (out/'eval_config.json').write_text(json.dumps(dict(args=vars(args),checkpoint_step=ck['step'],
        checkpoint_epoch=ck.get('epoch_progress',{}).get('epoch'),
        checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        evaluation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol='held_hyper_parameters_v1',
        per_episode_metrics_saved=args.save_episode_metrics,
        action_query_interval_control_steps=1,temporal_aggregation=True),indent=2))
    for item in envs:
        env_id=item['env_id'];env=None
        agent.reset_diagnostics()
        processor.reference_sha256s=[]
        try:
            env=make_eval_envs_with_level(extract_base_env_name(env_id),extract_level(env_id),1,args.sim_backend,
                dict(control_mode='pd_joint_pos',reward_mode='dense',obs_mode='rgb',render_mode='rgb_array',max_episode_steps=args.max_steps,
                     sensor_configs=dict(shader_pack=args.shader),human_render_camera_configs=dict(shader_pack=args.shader)),
                dict(obs_horizon=1),str(out/'videos'/env_id),[FlattenRGBDObservationWrapper])
            env=ResetAuditEnv(env)
            metrics=evaluate_with_task_encoder(args.num_episodes,agent,env,
                dict(env_id=env_id,delta_control=False,pred_horizon=c.horizon,temporal_agg=True,light_temporal_agg=False,
                     tagg_window=4,max_timesteps=args.max_steps,device=args.device,sim_backend=args.sim_backend),
                processor,InputMode.VIDEO_ONLY,tokenizer=None)
            row=dict(env_id=env_id,status='ok',suite=args.suite,episodes=args.num_episodes,
                     reference_mode=args.reference_mode,reference_task=processor.last_reference_task,
                     reference_sha256s=processor.reference_sha256s,initial_states=env.initial_states,
                     ik_targets=agent.ik_targets,ik_failures=agent.ik_failures,
                     hyper_update_interval=args.hyper_update_interval,policy_queries=agent.policy_queries,
                     hyper_parameter_updates=agent.hyper_parameter_updates,
                     hyper_update_steps_by_episode=agent.hyper_update_steps_by_episode,
                     metrics=summarize_eval_metrics(metrics, save_episode_values=args.save_episode_metrics))
        except Exception as exc:row=dict(env_id=env_id,status='error',error=str(exc))
        finally:
            if env is not None:env.close()
            clear_l_level()
        results.append(row);(out/'results.json').write_text(json.dumps(results,indent=2));print(row,flush=True)
    if any(r['status']=='error' for r in results):raise SystemExit('Some environments failed; do not average failures as zero success')


if __name__=='__main__':main()
