"""Isolated current-observation CNN for a possible foundation comparison.

The human-reference representation is unchanged. No augmentation is performed.
Preparation and CPU verification do not launch an optimizer or rollout job.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import default_collate
from examples.baselines.encoders.obs_encoder import SimpleCNNEncoder, ObsEncoderConfig
from .model import HyperICLZQ, ZQConfig
from .data import CuratedDataset


class CNNObservationFoundation(HyperICLZQ):
    def __init__(self, config, spaces, tokenizer=None, cnn_seed=73133):
        super().__init__(config,spaces,tokenizer)
        # Creation of this branch must not perturb shared training RNG.
        previous_rng=torch.get_rng_state()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(cnn_seed)
            self.obs_cnn=SimpleCNNEncoder(ObsEncoderConfig(
                encoder_type='simple_cnn',image_size=224,output_dim=config.d))
        assert torch.equal(previous_rng,torch.get_rng_state())
        self.obs_cnn.patch_proj.requires_grad_(False)

    def set_training_stage(self, stage):
        if stage!='base':raise ValueError('This controlled comparison is foundation-only')
        super().set_training_stage(stage)
        for name in ['obs_feature_proj','obs_attention']:
            getattr(self,name).requires_grad_(False)
        self.obs_queries.requires_grad_(False)
        self.obs_cnn.requires_grad_(True)
        self.obs_cnn.patch_proj.requires_grad_(False)

    def observation(self,batch):
        if 'obs_rgb' in batch:
            rgb=batch['obs_rgb']
            if rgb.dtype!=torch.uint8 or rgb.ndim!=4 or tuple(rgb.shape[1:])!=(224,224,3):
                raise ValueError('Require original uint8 BHWC RGB at224px')
            images=rgb.permute(0,3,1,2).float()/255
        else:
            # The common EvalAgent has already converted/resized its RGB to
            # float BCHW. Preserve those pixels rather than quantizing again.
            images=batch['obs']
            if not images.is_floating_point() or tuple(images.shape[1:])!=(3,224,224):
                raise ValueError('Require common evaluator float BCHW images')
        features=self.obs_cnn.features(images)
        obs=self.obs_cnn.proj(self.obs_cnn.pool(features).flatten(1))
        state=self.state_enc(torch.cat([batch['state']*batch['state_mask'],batch['state_mask'].float()],-1))
        sid=self.space_embedding(batch['space_id'])
        dt=torch.log1p(batch['action_dt'])[:,None]
        return self.obs_proj(torch.cat([obs,state,sid,dt],-1))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init-from',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--pilot',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify-existing',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'initial.pt').exists() and not args.verify_existing:raise FileExistsError('Use a fresh preparation directory')
    torch.set_num_threads(4)
    initial=torch.load(args.init_from,map_location='cpu',weights_only=False)
    if not initial['args'].get('initialization_only'):raise ValueError('Fresh shared initialization required')
    config=ZQConfig(**initial['config'])
    torch.manual_seed(73131)
    model=CNNObservationFoundation(config,initial['action_spaces'])
    loaded=model.load_state_dict(initial['model'],strict=False)
    if loaded.unexpected_keys or any(not k.startswith('obs_cnn.') for k in loaded.missing_keys):raise ValueError(str(loaded))
    model.set_training_stage('base')
    for key,value in initial['model'].items():
        if not torch.equal(model.state_dict()[key],value):raise AssertionError('Shared initialization changed: '+key)
    checkpoint=dict(model=model.state_dict(),config=asdict(config),action_spaces=model.spaces,training_stage='base',
        args=dict(initialization_only=True,seed=73131,observation_encoder='simple_cnn',augmentation=False))
    if args.verify_existing:
        saved=torch.load(args.output/'initial.pt',map_location='cpu',weights_only=False)
        assert saved.keys()==checkpoint.keys()
        assert saved['config']==checkpoint['config'] and saved['action_spaces']==checkpoint['action_spaces']
        assert saved['model'].keys()==checkpoint['model'].keys()
        assert all(torch.equal(saved['model'][key],value) for key,value in checkpoint['model'].items())
        del saved
    else:torch.save(checkpoint,args.output/'initial.pt')

    # Bounded real-data forward/backward verification, without optimizer.step.
    data=CuratedDataset(str(args.manifest),config,cached_tokens=True,stage='base')
    pilot=json.loads((args.pilot/'result.json').read_text())
    chosen=[r for r in pilot['rows'] if r['id'] in {x['id'] for x in data.rows}][:2]
    assert len(chosen)==2
    by_id={r['id']:i for i,r in enumerate(data.rows)}
    examples=[]
    for row in chosen:
        i=by_id[row['id']];t=row['frames']//2
        item=data[(i,t)]
        rgb=np.load(row['cache'],mmap_mode='r',allow_pickle=False)
        item['obs_rgb']=torch.from_numpy(np.array(rgb[t]))
        examples.append(item)
    batch=default_collate(examples)
    with torch.no_grad():
        images=batch['obs_rgb'].permute(0,3,1,2).float()/255
        direct=model.obs_cnn.proj(model.obs_cnn.pool(model.obs_cnn.features(images)).flatten(1))
        official,_=model.obs_cnn(images)
        assert torch.equal(direct,official)
        encoding=model.observation(batch)
        float_batch={key:value for key,value in batch.items() if key!='obs_rgb'}
        float_batch['obs']=images
        assert torch.equal(encoding,model.observation(float_batch))
        blank=dict(batch,obs_rgb=torch.zeros_like(batch['obs_rgb']))
        sensitivity=float((encoding-model.observation(blank)).abs().max())
        assert sensitivity>1e-6
    torch.manual_seed(73131);loss=model(batch)['loss']
    assert torch.isfinite(loss);loss.backward()
    gradient=float(torch.linalg.vector_norm(model.obs_cnn.features[0].weight.grad))
    assert gradient>0 and np.isfinite(gradient)
    inactive=['obs_feature_proj.','obs_attention.','obs_queries','obs_cnn.patch_proj.']
    assert all(p.grad is None for name,p in model.named_parameters() if any(name.startswith(k) for k in inactive))
    persisted=torch.load(args.output/'initial.pt',map_location='cpu',weights_only=False)
    for key,value in persisted['model'].items():
        if not torch.equal(model.state_dict()[key],value):raise AssertionError('Backward unexpectedly changed weights')
    result=dict(passed=True,verified_at=time.time(),source_sha256=sha(__file__),
        initial_source_sha256=sha(args.init_from),manifest_sha256=sha(args.manifest),
        initial_checkpoint_sha256=sha(args.output/'initial.pt'),
        shared_initial_tensors_bit_identical=len(initial['model']),
        cnn_trainable_parameters=sum(p.numel() for p in model.obs_cnn.parameters() if p.requires_grad),
        total_trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        real_examples=[r['id'] for r in chosen],first_conv_gradient_norm=gradient,
        observation_changes_with_pixels=sensitivity,official_simple_cnn_global_output_bit_identical=True,
        uint8_training_and_float_eval_encoding_bit_identical=True,
        unused_dino_aggregator_and_patch_head_receive_no_gradients=True,optimizer_steps=0,
        augmentation=False,scope='Current policy observation encoder only; action representation/normalization, state fusion, target and reference architecture retained. No full training launched; RGB preprocessing and GPU input/RNG pairing still require train/eval integration checks.')
    (args.output/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
