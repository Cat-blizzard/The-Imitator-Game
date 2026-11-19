"""Generated-weight regularization and fail-fast checks for staged training."""
import math
import torch


def delta_factors(layer, entry):
    """Exact ||delta W||_F^2 via low-rank Gram matrices, without dense deltas.

    Includes the cross term between manifold and residual updates. The factors
    are not penalized separately: their scale ambiguity must not change this
    regularizer.
    """
    prototype = next(iter(entry.values()))
    count = len(prototype)
    left, right = [], []
    with torch.autocast(device_type=prototype.device.type, enabled=False):
        if 'c' in entry:
            coefficients = entry['c'].float()
            p = layer.atom_P.float().permute(2, 0, 1)
            left.append((p[None] * coefficients[:, None, :, None]).flatten(2))
            q = layer.atom_Q.float().permute(0, 2, 1).flatten(0, 1)
            right.append(q[None].expand(count, -1, -1))
        if 'A' in entry and 'B' in entry:
            factor = entry['B'].float() / math.sqrt(layer.spec.rank)
            if 'g' in entry:
                factor = factor * entry['g'].float()[:, :, None]
            left.append(factor)
            right.append(entry['A'].float())
        if not left:
            left = [prototype.new_zeros(count, layer.spec.out_f, 1, dtype=torch.float32)]
            right = [prototype.new_zeros(count, 1, layer.spec.in_f, dtype=torch.float32)]
        return torch.cat(left, -1), torch.cat(right, 1)


def factor_norm_squared(left, right):
    with torch.autocast(device_type=left.device.type, enabled=False):
        ll = left.float().transpose(-1, -2) @ left.float()
        rr = right.float() @ right.float().transpose(-1, -2)
        return (ll * rr.transpose(-1, -2)).sum((-1, -2)).clamp_min(0.)


def delta_norm_squared(layer, entry):
    return factor_norm_squared(*delta_factors(layer, entry))


def parameter_penalty(target, parameters, first=None, allowance=.25):
    """Penalize updates above 25% of each frozen/base weight's Frobenius norm.

    No expert or task-specific target weights are used. This is a soft penalty,
    not clipping. Healthy small updates are unchanged. Sum over layers, average
    over references. Bias deltas use the layer output width as their scale.
    """
    losses, ratios = [], []
    for name, values in parameters.items():
        entry = values if first is None else {k: v[first] for k, v in values.items()}
        layer = target.layer(name)
        norm2 = delta_norm_squared(layer, entry)
        ratio2 = norm2 / layer.base.weight.detach().float().square().sum().clamp_min(1e-6)
        losses.append((ratio2 - allowance**2).clamp_min(0).mean())
        ratios.append(ratio2.max())
        if 'db' in entry:
            bias2 = entry['db'].float().square().mean(-1)
            losses.append((bias2 - allowance**2).clamp_min(0).mean())
    return torch.stack(losses).sum(), torch.stack(ratios).max().sqrt().detach()


class TrainingInstability(RuntimeError):
    pass


class StabilityGuard:
    """Reject a destructive optimizer update, including large FINITE gradients."""
    def __init__(self, max_grad_norm=1000., max_delta_ratio=2., state=None):
        self.max_grad_norm = max_grad_norm
        self.max_delta_ratio = max_delta_ratio
        self.action_ema = None
        self.observations = 0
        self.best_fitting = None
        if state:
            self.action_ema = state['action_ema']
            self.observations = state['observations']
            self.best_fitting = state.get('best_fitting')

    def state_dict(self):
        return dict(action_ema=self.action_ema, observations=self.observations, best_fitting=self.best_fitting)

    def check_validation(self, action):
        if not math.isfinite(action):
            raise TrainingInstability('Non-finite fixed fitting probe')
        if self.best_fitting is not None and action > max(.5, 5 * self.best_fitting):
            raise TrainingInstability(f'Fixed fitting loss {action:g} deteriorated from {self.best_fitting:g}')
        self.best_fitting = action if self.best_fitting is None else min(self.best_fitting, action)

    def check(self, metrics, grad_norm):
        if not all(math.isfinite(float(x)) for x in (*metrics.values(), grad_norm)):
            raise TrainingInstability('Non-finite loss, diagnostic, or gradient')
        if grad_norm > self.max_grad_norm:
            raise TrainingInstability(f'Gradient norm {grad_norm:g} > {self.max_grad_norm:g}')
        ratio = metrics.get('delta_max_ratio', 0.)
        if ratio > self.max_delta_ratio:
            raise TrainingInstability(f'Generated weight ratio {ratio:g} > {self.max_delta_ratio:g}')
        action = metrics.get('action')
        if action is not None:
            if self.observations >= 100 and action > max(2., 20 * self.action_ema):
                raise TrainingInstability(f'Action loss {action:g} exceeds recent EMA {self.action_ema:g}')
            self.action_ema = action if self.action_ema is None else .99 * self.action_ema + .01 * action
            self.observations += 1
