"""Task/body/label-specific LoRA teachers for the native EEF policy.

Teacher IDs are training metadata only. A frozen teacher retains its own base,
observation encoder and action adapters when the student enters joint training.
"""
import hashlib
import json
import math
import torch
from torch import nn
from .stability import delta_factors, factor_norm_squared


def expert_key(row):
    if not row.get('action_space'):
        return -1
    identity = [row['goal_key'], row['action_space'], row['source'], row.get('label_kind', '')]
    return int(hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()[:15], 16)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(part)
    return digest.hexdigest()


class TaskLoRABank(nn.Module):
    def __init__(self, specs, registry):
        super().__init__()
        self.specs, self.registry = specs, registry
        self.A, self.B, self.db = nn.ParameterDict(), nn.ParameterDict(), nn.ParameterDict()
        self.register_buffer('keys', torch.tensor([r['key'] for r in registry], dtype=torch.long))
        for s in specs:
            key = s.name.replace('.', '__')
            a = torch.empty(len(registry), s.rank, s.in_f)
            nn.init.kaiming_uniform_(a.view(-1, s.in_f), a=math.sqrt(5))
            self.A[key] = nn.Parameter(a)
            self.B[key] = nn.Parameter(torch.zeros(len(registry), s.out_f, s.rank))
            if s.gen_bias:
                self.db[key] = nn.Parameter(torch.zeros(len(registry), s.out_f))

    def lookup(self, keys):
        match = keys[:, None] == self.keys[None]
        return match.long().argmax(1), match.any(1)

    def forward(self, ids):
        result = {}
        for s in self.specs:
            key = s.name.replace('.', '__')
            result[s.name] = dict(A=self.A[key][ids], B=self.B[key][ids])
            if key in self.db:
                result[s.name]['db'] = self.db[key][ids]
        return result


def parameter_statistics(target, generated, teacher):
    """Compare complete delta matrices, never their non-identifiable A/B factors.

    Low-rank Gram products are exactly equivalent to dense matrix losses. A
    1%-of-base norm floor avoids exploding relative losses for near-zero experts.
    Direction/magnitude losses ignore such uninformative teacher layers.
    """
    relative, direction, magnitude = [], [], []
    for name, entry in generated.items():
        layer = target.layer(name)
        with torch.autocast(device_type=layer.base.weight.device.type, enabled=False):
            gl, gr = delta_factors(layer, entry)
            tl, tr = delta_factors(layer, teacher[name])
            g2, t2 = factor_norm_squared(gl, gr), factor_norm_squared(tl, tr)
            diff = factor_norm_squared(torch.cat([gl, -tl], -1), torch.cat([gr, tr], 1))
            floor = (layer.base.weight.detach().float().square().sum() * 1e-4).clamp_min(1e-6)
            relative.append((diff / t2.clamp_min(floor)).mean())
            informative = (t2 > floor).float()
            cosine = ((g2 + t2 - diff) / 2 / (g2.clamp_min(floor) * t2.clamp_min(floor)).sqrt()).clamp(-1, 1)
            direction.append(((1 - cosine) * informative).mean())
            magnitude.append((.5 * (g2.clamp_min(floor).log() - t2.clamp_min(floor).log()).abs() * informative).mean())
            if 'db' in entry:
                relative.append((entry['db'].float() - teacher[name]['db'].float()).square().mean())
    return dict(expert_param=torch.stack(relative).mean(), expert_dir=torch.stack(direction).mean(),
                expert_mag=torch.stack(magnitude).mean())


class ExpertSupervisor:
    """Not a student submodule: teacher state never enters the student's optimizer."""
    def __init__(self, model, bank, enabled):
        self.model = model.eval().requires_grad_(False)
        self.bank = bank.eval().requires_grad_(False)
        self.enabled = torch.as_tensor(enabled, device=bank.keys.device, dtype=torch.bool)
        self.scale = 1.

    @classmethod
    def load(cls, path, spaces, device):
        from .model import HyperICLZQ, ZQConfig
        from .train import load_weights
        ck = torch.load(path, map_location='cpu', weights_only=False)
        if ck.get('stability_failure') or not any(ck['enabled']):
            raise ValueError('No validated healthy experts in bank')
        if file_sha256(ck['base_checkpoint']) != ck['base_sha256']:
            raise ValueError('Expert base checkpoint changed')
        base = torch.load(ck['base_checkpoint'], map_location='cpu', weights_only=False)
        c = ZQConfig(**base['config'])
        model = HyperICLZQ(c, spaces).to(device)
        load_weights(model, base, c, spaces)
        bank = TaskLoRABank(model.target.layer_specs(), ck['registry']).to(device)
        bank.load_state_dict(ck['bank'])
        return cls(model, bank, ck['enabled'])

    def losses(self, student, batch, S, inverse, velocity, x, t):
        ids, known = self.bank.lookup(batch['expert_key'])
        selected = known & self.enabled[ids]
        zero = velocity.sum() * 0
        losses = {k: zero for k in ('expert_param', 'expert_dir', 'expert_mag', 'expert_func')}
        if not selected.any():
            return zero, losses
        b = {k: v[selected] for k, v in batch.items()}
        with torch.no_grad():
            expected = self.model.velocity(x[selected], t[selected], self.model.observation(b),
                                           self.bank(ids[selected]), b['space_id'])
        # Normalize over all labeled queries: rare teacher coverage cannot silently
        # increase its weight, and masked/padded action dimensions never contribute.
        coverage = selected.float().mean()
        losses['expert_func'] = student.weighted_error(velocity[selected], expected,
                                                       b['action_mask'], b['loss_weight']) * coverage
        if student.training_stage == 'hyper':
            # Global program only; per-observation phase residuals stay free to
            # learn through action/function losses, as in the original baseline.
            compact_ids = torch.arange(len(batch['space_id']), device=x.device) if inverse is None else inverse
            first, _ = student.unique_first(torch.stack([compact_ids[selected], ids[selected]], 1))
            generated = student.D(S[compact_ids[selected][first]])
            losses.update(parameter_statistics(student.target, generated, self.bank(ids[selected][first])))
            for key in ('expert_param', 'expert_dir', 'expert_mag'):
                losses[key] = losses[key] * coverage
        # In joint, theta0 and adapters move. Comparing old delta weights is no
        # longer meaningful; retain only the function teacher on its frozen base.
        weights = dict(expert_param=.2, expert_dir=.1, expert_mag=.05, expert_func=.2)
        return self.scale * sum(weights[k] * v for k, v in losses.items()), losses
