"""Video-conditioned CNN observation control, isolated from foundation runs.

Only base and hyper are supported. During hyper, the trained observation branch
and base policy stay frozen; the original reference encoder, compiler, decoder,
phase heads and target atoms have exactly the original hyper-stage trainability.
"""
from .model import HyperICLZQ
from .observation_cnn_ablation import CNNObservationFoundation


class CNNObservationPolicy(CNNObservationFoundation):
    def set_training_stage(self, stage):
        if stage not in ('base', 'hyper'):
            raise ValueError('This controlled full-policy experiment supports base/hyper only')
        HyperICLZQ.set_training_stage(self, stage)
        self.obs_feature_proj.requires_grad_(False)
        self.obs_attention.requires_grad_(False)
        self.obs_queries.requires_grad_(False)
        self.obs_cnn.requires_grad_(stage == 'base')
        self.obs_cnn.patch_proj.requires_grad_(False)


def model_class(encoder):
    if encoder == 'cnn':
        return CNNObservationPolicy
    if encoder == 'dino':
        return HyperICLZQ
    raise ValueError(encoder)
