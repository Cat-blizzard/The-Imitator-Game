r"""PretrainedTarget: a frozen pretrained VLA / WAM whose action-expert Linear layers receive the
generated per-sample deltas (manifold coefficients + zero-init residual LoRA + gate).

Why this shape: WIZARD (2606.07217) generates LoRA for a frozen pi0.5 and RoboTTT (2607.15275) puts
fast weights inside GR00T's DiT action head — both keep the VLM prefix frozen and adapt only the
action expert. We follow that: the VLM / vision tower is never touched, only the expert's
{q,v,o,down} projections (rank 8) get generated deltas.

How the injection works (generic over PyTorch VLAs):
  * every nn.Linear whose qualified name matches `lora_regex` is replaced *in place* by
    `LoRALinear.from_linear(...)` (same weights, same forward when no delta is active);
  * the block index of a layer is parsed from `block_regex` ("layers.(\d+)."), so one skill token per
    expert layer;
  * deltas are passed through `lora_context(P)` because we cannot change the policy's forward signature;
  * the predicted velocity is captured with a forward hook on `velocity_module` (e.g. `action_out_proj`)
    for the function-space losses; `noise` / `time` are forwarded when the policy accepts them
    (lerobot pi0 / SmolVLA do), so student / teacher / second-clip calls see the same (x_t, t).

Profiles below encode the module names of lerobot's SmolVLA / pi0 as of the released code
(`policy.model.vlm_with_expert.lm_expert` / `policy.model.paligemma_with_expert.gemma_expert`,
`policy.model.action_out_proj`). They are duck-typed and validated at construction with explicit error
messages, so a renamed attribute fails loudly instead of silently training nothing.

Stage A for this version = "domain LoRA": a single task-agnostic LoRA set (TaskLoRABank with n_tasks=1)
trained on all robot data, then merged into the base weights (`merge_domain_lora`) -> theta_0.
This mirrors WIZARD's "lightly adapted to the target setup" protocol and keeps stage B–E identical
to the from-scratch version.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.lora_dit import LoRALinear, lora_context
from .base import PDict, TargetPolicy


@dataclass
class VLAProfile:
    loader: str
    lora_regex: str
    block_regex: str
    velocity_module: str
    state_key: str = "observation.state"
    action_key: str = "action"
    task_key: str = "task"


PROFILES: Dict[str, VLAProfile] = {
    "lerobot_smolvla": VLAProfile(
        loader="lerobot_smolvla",
        lora_regex=r"^model\.vlm_with_expert\.lm_expert\..*layers\.\d+\.(self_attn\.(q_proj|v_proj|o_proj)|mlp\.down_proj)$",
        block_regex=r"layers\.(\d+)\.",
        velocity_module="model.action_out_proj"),
    "lerobot_pi0": VLAProfile(
        loader="lerobot_pi0",
        lora_regex=r"^model\.paligemma_with_expert\.gemma_expert\..*layers\.\d+\.(self_attn\.(q_proj|v_proj|o_proj)|mlp\.down_proj)$",
        block_regex=r"layers\.(\d+)\.",
        velocity_module="model.action_out_proj"),
    # GR00T N1.5: flow-matching DiT action head (what RoboTTT builds on). Loader not implemented here
    # (needs the Isaac-GR00T package); regex targets its DiT attention/MLP projections.
    "groot_n15": VLAProfile(
        loader="groot",
        lora_regex=r"^action_head\..*(to_q|to_v|to_out\.0|q_proj|v_proj|o_proj|down_proj|fc2)$",
        block_regex=r"(?:blocks|layers)\.(\d+)\.",
        velocity_module="action_head.action_decoder"),
}


def _get_submodule(root: nn.Module, qualname: str) -> nn.Module:
    m = root
    for part in qualname.split("."):
        m = getattr(m, part)
    return m


def _set_submodule(root: nn.Module, qualname: str, new: nn.Module):
    parts = qualname.split(".")
    parent = _get_submodule(root, ".".join(parts[:-1])) if len(parts) > 1 else root
    setattr(parent, parts[-1], new)


class PretrainedTarget(TargetPolicy):
    def __init__(self, args, act_dim: int, horizon: int, device: torch.device):
        super().__init__()
        self.act_dim, self.horizon, self.args = act_dim, horizon, args
        prof = PROFILES.get(args.vla_profile)
        if args.vla_profile == "custom" or prof is None:
            assert args.vla_lora_regex and args.vla_velocity_module, "custom profile needs --vla-lora-regex/--vla-velocity-module"
            prof = VLAProfile("lerobot_smolvla", args.vla_lora_regex, r"layers\.(\d+)\.", args.vla_velocity_module)
        if args.vla_lora_regex:
            prof.lora_regex = args.vla_lora_regex
        if args.vla_velocity_module:
            prof.velocity_module = args.vla_velocity_module
        self.profile = prof
        self.policy = self._load(prof.loader, args.vla_repo_id).to(device)
        self.policy.requires_grad_(False)
        self.policy.eval()
        self.cfg = getattr(self.policy, "config", None)
        self.chunk_size = int(getattr(self.cfg, "chunk_size", horizon) or horizon)
        self.max_action_dim = int(getattr(self.cfg, "max_action_dim", act_dim) or act_dim)
        self.image_keys = self._image_keys(args)
        if args.vla_identity_norm:
            self._bypass_normalizers()
        self._inject(prof, args.rank, args.n_atoms, args.vla_max_layers)
        self._captured: Dict[str, torch.Tensor] = {}
        self._register_velocity_hook(prof.velocity_module)

    # ------------------------------------------------------------------ #
    # loading / wiring
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load(loader: str, repo_id: str) -> nn.Module:
        if loader == "lerobot_smolvla":
            try:
                from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            except ImportError:
                from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy   # older lerobot
            return SmolVLAPolicy.from_pretrained(repo_id)
        if loader == "lerobot_pi0":
            try:
                from lerobot.policies.pi0.modeling_pi0 import PI0Policy
            except ImportError:
                from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
            return PI0Policy.from_pretrained(repo_id)
        if loader == "groot":
            raise NotImplementedError(
                "GR00T profile: load `Gr00tPolicy` from the Isaac-GR00T package, wrap it in an nn.Module that exposes "
                "`forward(batch)->loss` and `predict_action_chunk(batch)`, and pass it via PretrainedTarget.from_policy().")
        raise ValueError(loader)

    @classmethod
    def from_policy(cls, policy: nn.Module, args, act_dim, horizon, device):
        """Use an already-constructed policy object (any PyTorch VLA/WAM with forward(batch)->loss)."""
        obj = cls.__new__(cls)
        TargetPolicy.__init__(obj)
        obj.act_dim, obj.horizon, obj.args = act_dim, horizon, args
        obj.profile = VLAProfile("custom", args.vla_lora_regex, r"(?:blocks|layers)\.(\d+)\.", args.vla_velocity_module)
        obj.policy = policy.to(device); obj.policy.requires_grad_(False); obj.policy.eval()
        obj.cfg = getattr(policy, "config", None)
        obj.chunk_size = int(getattr(obj.cfg, "chunk_size", horizon) or horizon)
        obj.max_action_dim = int(getattr(obj.cfg, "max_action_dim", act_dim) or act_dim)
        obj.image_keys = obj._image_keys(args)
        obj._inject(obj.profile, args.rank, args.n_atoms, args.vla_max_layers)
        obj._captured = {}
        obj._register_velocity_hook(obj.profile.velocity_module)
        return obj

    def _image_keys(self, args) -> List[str]:
        feats = getattr(self.cfg, "image_features", None)
        if feats:
            keys = list(feats.keys()) if hasattr(feats, "keys") else list(feats)
            return keys[: len(args.cameras)] if len(keys) >= len(args.cameras) else keys
        return [f"observation.images.{c}" for c in args.cameras]

    def _bypass_normalizers(self):
        n = 0
        for name in ("normalize_inputs", "normalize_targets", "unnormalize_outputs"):
            if hasattr(self.policy, name):
                setattr(self.policy, name, nn.Identity())
                n += 1
        if n == 0:
            print("[PretrainedTarget] no normalizer modules found on the policy; make sure the pre/post "
                  "processors are set to identity (Imitator data is already normalised with bounds_q99).")

    def _inject(self, prof: VLAProfile, rank: int, n_atoms: int, max_layers: int):
        pat, bpat = re.compile(prof.lora_regex), re.compile(prof.block_regex)
        matched = [(n, m) for n, m in self.policy.named_modules() if isinstance(m, nn.Linear) and pat.search(n)]
        if not matched:
            names = [n for n, m in self.policy.named_modules() if isinstance(m, nn.Linear)][:40]
            raise RuntimeError(f"lora_regex {prof.lora_regex!r} matched no nn.Linear. First Linear names:\n" + "\n".join(names))
        blocks = sorted({int(bpat.search(n).group(1)) for n, _ in matched if bpat.search(n)})
        if max_layers and len(blocks) > max_layers:
            keep = set(blocks[-max_layers:])
            matched = [(n, m) for n, m in matched if bpat.search(n) and int(bpat.search(n).group(1)) in keep]
            blocks = sorted(keep)
        remap = {b: i for i, b in enumerate(blocks)}
        self.n_blocks = len(blocks)
        self._injected: List[str] = []
        for n, m in matched:
            b = remap[int(bpat.search(n).group(1))] if bpat.search(n) else 0
            wrapped = LoRALinear.from_linear(m, n, rank, b, gen_bias=False, n_atoms=n_atoms)
            wrapped.to(m.weight.device)
            _set_submodule(self.policy, n, wrapped)
            self._injected.append(n)
        gen = sum(s.rank * (s.in_f + s.out_f) for s in self.layer_specs())
        print(f"[PretrainedTarget] injected {len(self._injected)} LoRALinear into {self.n_blocks} expert layers "
              f"(~{gen/1e6:.2f}M generated numbers / task)")

    def _register_velocity_hook(self, qualname: str):
        try:
            mod = _get_submodule(self.policy, qualname)
        except AttributeError:
            print(f"[PretrainedTarget] velocity module {qualname!r} not found -> function-space losses disabled")
            return
        mod.register_forward_hook(lambda m, i, o: self._captured.__setitem__("v", o))

    # ------------------------------------------------------------------ #
    # interface
    # ------------------------------------------------------------------ #
    def lora_layers(self) -> List[LoRALinear]:
        return [_get_submodule(self.policy, n) for n in self._injected]

    def _format_images(self, robot_obs: Dict) -> List[torch.Tensor]:
        image = robot_obs.get("rgb", robot_obs.get("view_1"))
        if image.dim() == 5:
            image = image[:, 0]
        if image.shape[-1] in (3, 4):
            image = image.permute(0, 3, 1, 2)
        image = image[:, :3].float()
        if image.max() > 1.5:
            image = image / 255.0
        return [image]                                   # single camera in the Imitator default

    def build_batch(self, robot_obs: Dict, actions: Optional[torch.Tensor] = None) -> Dict:
        images = self._format_images(robot_obs)
        state = robot_obs.get("state", robot_obs.get("states")).float()
        if state.dim() == 3:
            state = state[:, -1]
        B = state.shape[0]
        batch = {self.profile.state_key: state,
                 self.profile.task_key: [self.args.vla_task_text] * B}
        for k, img in zip(self.image_keys, images):
            batch[k] = img
        if actions is not None:
            a = actions.float()
            pad_t = self.chunk_size - a.shape[1]
            is_pad = torch.zeros(B, self.chunk_size, dtype=torch.bool, device=a.device)
            if pad_t > 0:
                a = torch.cat([a, a[:, -1:].expand(-1, pad_t, -1)], dim=1)
                is_pad[:, self.horizon:] = True
            batch[self.profile.action_key] = a
            batch["action_is_pad"] = is_pad
        return batch

    def _padded_noise(self, noise: Optional[torch.Tensor], B: int, device) -> Optional[torch.Tensor]:
        if noise is None:
            return None
        full = torch.randn(B, self.chunk_size, self.max_action_dim, device=device)
        full[:, : noise.shape[1], : noise.shape[2]] = noise
        return full

    def loss_and_velocity(self, P, robot_obs, actions, h, noise=None, u=None):
        B = actions.shape[0]
        noise = torch.randn_like(actions) if noise is None else noise
        u = torch.rand(B, device=actions.device) if u is None else u
        batch = self.build_batch(robot_obs, actions)
        self._captured.pop("v", None)
        with lora_context(P):
            try:
                out = self.policy.forward(batch, noise=self._padded_noise(noise, B, actions.device), time=u)
                same_xt = True
            except TypeError:                            # policy.forward(batch) only
                out = self.policy.forward(batch)
                same_xt = False
        loss = out[0] if isinstance(out, (tuple, list)) else out
        if isinstance(loss, dict):
            loss = loss["loss"]
        v = self._captured.get("v")
        x0_hat = None
        if v is not None:
            v = v[:, : self.horizon, : self.act_dim]
            if same_xt:
                tt = u.view(-1, 1, 1)
                x_t = (1 - tt) * actions + tt * noise    # lerobot: x_t = time*noise + (1-time)*actions
                x0_hat = x_t - tt * v
        return loss, v, x0_hat, noise, u

    @torch.no_grad()
    def sample(self, P, robot_obs, h):
        batch = self.build_batch(robot_obs)
        with lora_context(P):
            if hasattr(self.policy, "predict_action_chunk"):
                a = self.policy.predict_action_chunk(batch)
            elif hasattr(self.policy, "_get_action_chunk"):
                a = self.policy._get_action_chunk(batch)
            else:
                if hasattr(self.policy, "reset"):
                    self.policy.reset()
                a = torch.stack([self.policy.select_action(batch) for _ in range(self.horizon)], dim=1)
        return a[:, : self.horizon, : self.act_dim].float()

    def set_stage(self, stage: str, train_base: bool):
        # slow weights of the VLA are never trained directly (stage A uses a domain LoRA, then merges)
        self.policy.requires_grad_(False)
        for m in self.lora_layers():
            trainable_atoms = stage in ("D", "E")
            m.atom_P.requires_grad_(trainable_atoms)
            m.atom_Q.requires_grad_(trainable_atoms)

    @torch.no_grad()
    def merge_domain_dense(self, dense: Dict[str, torch.Tensor]):
        """Fold the stage-A domain LoRA (dense deltas, name -> [out,in]) into the frozen weights -> theta_0."""
        for m in self.lora_layers():
            if m.spec.name in dense:
                m.merge_delta_into_base(dense[m.spec.name].to(m.base.weight.device))
        print("[PretrainedTarget] merged domain LoRA into the base policy")
