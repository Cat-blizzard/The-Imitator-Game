"""HyperICLAgent — drop-in replacement for TemplateAgent (same 4-method interface):

    compute_loss(batch)                       -> {"loss": ..., component scalars}
    prepare_for_eval(human_video, robot_obs, human_desc, human_vl_ids)   once per episode
    get_action(obs)                           -> (B, pred_horizon, action_dim)
    clear_cache()

Pipeline (DESIGN_H2R_HYPERNET.md §0.3):
    tokens  = FrozenFrameTokenizer(human_video)          [U, N, P, F]      (frozen DINOv2 / benchmark backbone)
    Z_H     = E(tokens)                                   effect / event / progress / embodiment tokens
    S0      = T(None, human_tokens(Z_H))                  one-shot skill state [U, L, d]
    S0'     = LatentTTT(S0, K)                            optional: human-video self-supervised refinement
    P       = D(S, alpha, inv)                            per-sample deltas for the target's LoRALinear layers
    loss    = target.loss_and_velocity(P, obs, actions)   FM loss (+ teacher / alignment / regularisers)
    S_c     = T(S_{c-1}, human_tokens, feedback_{<=c})    closed loop (eval: every k get_action calls;
                                                          train: 2nd pass with the current-obs feedback token)

The same agent is used with `target="scratch"` (FM-DiT from scratch) and `target="vla"` (frozen
pretrained VLA with generated LoRA). Only `self.target` differs.
"""
from __future__ import annotations

import copy
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HyperICLArgs
from .expert_bank import TaskLoRABank
from .modules import losses as LS
from .modules.frame_tokenizer import FrozenFrameTokenizer
from .modules.hyper_compiler import HyperCompiler
from .modules.video_program_encoder import VideoProgramEncoder
from .modules.world_model import EffectHead, LatentDynamics
from .targets.scratch_target import ScratchTarget


class TaskRegistry:
    """repo_id (str) -> task index, persisted in the checkpoint."""

    def __init__(self, ids: Optional[List[str]] = None, capacity: int = 64):
        self.ids: List[str] = list(ids or [])
        self.capacity = capacity

    def encode(self, repo_ids: List[str], device, allow_new: bool = True) -> torch.Tensor:
        out = []
        for r in repo_ids:
            r = str(r)
            if r not in self.ids:
                if not allow_new or len(self.ids) >= self.capacity:
                    out.append(-1)
                    continue
                self.ids.append(r)
            out.append(self.ids.index(r))
        return torch.tensor(out, dtype=torch.long, device=device)


class _DisabledVideoTokenizer(nn.Module):
    """Placeholder used in stages A/B so the heavy frozen video backbone is not loaded.

    The tokenizer is excluded from checkpoints anyway. If a code path accidentally tries
    to use human video in A/B, fail loudly instead of silently doing the wrong thing.
    """

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "video tokenizer is disabled for stage A/B; human video is only available in stages C/D/E"
        )


class HyperICLAgent(nn.Module):
    def __init__(self, args: HyperICLArgs, device: torch.device, policy_override: Optional[nn.Module] = None):
        super().__init__()
        self.args, self.device = args, device
        self.pred_horizon, self.action_dim, self.obs_horizon = args.pred_horizon, args.action_dim, args.obs_horizon
        self.stage = args.stage
        self.registry = TaskRegistry(capacity=args.max_tasks)

        # 1. frozen video tokenizer (per-frame tokens, not a pooled z).
        # Stages A/B never consume human video, so do not even construct the heavy
        # DINO/template backbone there. The tokenizer is excluded from checkpoints.
        self.uses_human_video = args.stage in ("C", "D", "E")
        if self.uses_human_video:
            template_backbone = None
            if args.tokenizer_backend == "template":
                from examples.baselines.encoders.task_encoder.video_backbone import build_video_backbone
                template_backbone = build_video_backbone(
                    backbone_type=args.frozen_backbone_type, latent_dim=args.task_latent_dim,
                    max_seq_patches=args.frozen_backbone_seq_patches, adapter_layers=args.frozen_backbone_adapter_layers,
                    num_sampled_frames=args.frozen_backbone_num_frames, hf_cache_dir=args.hf_cache_dir)
            self.tokenizer = FrozenFrameTokenizer(
                backend=args.tokenizer_backend,
                template_backbone=template_backbone,
                seq_patches=args.frozen_backbone_seq_patches,
                feat_dim=args.feat_dim,
                image_size=args.image_size[0],
                hf_model_name=args.hf_model_name,
                hf_cache_dir=args.hf_cache_dir,
            )
        else:
            self.tokenizer = _DisabledVideoTokenizer()
            print(f"[agent] stage {args.stage}: video tokenizer/backbone is disabled")

        # 2. robot latent h_t (benchmark encoders; identical in both versions, used by the hypernet side
        #    and — for the scratch target only — as the policy's conditioning)
        from examples.baselines.encoders.obs_encoder import ObservationEncoder, ObsEncoderConfig
        from examples.baselines.encoders.state_encoder import StateEncoder, StateEncoderConfig
        self.obs_encoder = ObservationEncoder(ObsEncoderConfig(
            encoder_type=args.obs_encoder_type, image_size=args.image_size[0],
            output_dim=args.obs_latent_dim, hidden_dim=args.obs_latent_dim))
        self.state_encoder = StateEncoder(StateEncoderConfig(
            state_type=args.state_type, state_dim=args.state_dim, num_frames=args.obs_horizon,
            encoder_type=args.state_encoder_type, output_dim=args.state_latent_dim, hidden_dim=args.state_latent_dim))
        self.obs_proj = nn.Sequential(nn.LayerNorm(args.obs_latent_dim + args.state_latent_dim),
                                      nn.Linear(args.obs_latent_dim + args.state_latent_dim, args.h_dim), nn.GELU(),
                                      nn.Linear(args.h_dim, args.h_dim))
        self.obs_proj_ema = copy.deepcopy(self.obs_proj).requires_grad_(False)      # world-model targets
        self.prog_head = nn.Sequential(nn.Linear(args.h_dim, args.h_dim), nn.GELU(), nn.Linear(args.h_dim, args.d_prog))

        # 3. video program encoder E
        self.E = VideoProgramEncoder(args.feat_dim, args.d, args.d_prog, args.d_emb, args.K, args.n_pool,
                                     args.enc_layers, args.nhead)
        self.null_token = nn.Parameter(torch.zeros(1, 1, args.d))                   # demo-CFG null support

        # 4. target policy
        if args.target == "scratch":
            self.target = ScratchTarget(args.action_dim, args.h_dim, args.pred_horizon, args.dit_dim, args.dit_depth,
                                        args.dit_nhead, args.rank, args.ada_rank, args.n_atoms, args.sample_steps)
        elif args.target == "vla":
            from .targets.pretrained_target import PretrainedTarget
            self.target = (PretrainedTarget.from_policy(policy_override, args, args.action_dim, args.pred_horizon, device)
                           if policy_override is not None else
                           PretrainedTarget(args, args.action_dim, args.pred_horizon, device))
        else:
            raise ValueError(args.target)
        specs = self.target.layer_specs()

        # 5. compiler (T + D + latent TTT + feedback tokenizer), world model, expert bank
        self.compiler = HyperCompiler(
            specs, self.target.n_blocks, args.d, args.K, args.d_prog, args.h_dim,
            args.action_dim, args.pred_horizon, args.n_atoms, args.T_layers, args.nhead,
            args.M_fb, args.use_manifold, args.use_residual, args.use_prev_action,
            args.share_res_heads, args.trunk_dim,
            use_block_refiner=args.use_block_refiner,
            refiner_steps=args.refiner_steps,
            refiner_layers=args.refiner_layers,
            refiner_lr_max=args.refiner_lr_max,
            refiner_on_feedback=args.refiner_on_feedback,
            phase_manifold=args.phase_manifold,
            feedback_gate_max=args.feedback_gate_max,
        )
        self.wm = LatentDynamics(args.h_dim, args.action_dim, args.pred_horizon)
        self.effect_head = EffectHead(args.h_dim, args.d)
        self.bank = TaskLoRABank(specs, 1 if (args.target == "vla" and args.stage == "A") else args.max_tasks)
        self.bank_ready = False
        self._domain_dense = None                       # VLA: merged stage-A domain LoRA, carried through ckpts
        self._spec_blocks = [s.block for s in specs]

        # Frozen DINO patch-token cache is attached by the training script
        # after one-time precomputation. It is intentionally not part of state_dict.
        self.dino_feature_cache = None

        # eval caches
        self._cache = None
        self.set_stage(args.stage)
        self.to(device)

    def set_dino_feature_cache(self, cache) -> None:
        """Attach a multi-clip frozen DINO feature cache."""
        self.dino_feature_cache = cache
        if cache is not None:
            print(
                f"[agent] DINO feature cache attached: "
                f"{cache.num_clips} clips / {cache.num_repos} repos"
            )

    # ------------------------------------------------------------------ #
    # stage handling
    # ------------------------------------------------------------------ #
    def set_stage(self, stage: str):
        self.stage = stage
        a = self.args
        train_base = stage == "A"
        for p in self.parameters():
            p.requires_grad_(False)
        self.target.set_stage(stage, train_base and a.target == "scratch")
        if stage == "A":
            for m in (self.obs_encoder, self.state_encoder, self.obs_proj, self.wm):
                m.requires_grad_(True)
            if a.target == "vla":
                self.bank.requires_grad_(True)                     # domain LoRA (n_tasks=1)
        elif stage == "B":
            self.bank.requires_grad_(True)
        elif stage == "C":
            self.E.requires_grad_(True); self.prog_head.requires_grad_(True)
            if a.target == "vla":                              # obs encoders only serve the hypernet side here
                for m in (self.obs_encoder, self.state_encoder, self.obs_proj):
                    m.requires_grad_(True)
        elif stage in ("D", "E"):
            for m in (self.E, self.compiler, self.prog_head, self.effect_head, self.wm):
                m.requires_grad_(True)
            self.null_token.requires_grad_(True)
            if a.train_obs_in_D or a.target == "vla":
                for m in (self.obs_encoder, self.state_encoder, self.obs_proj):
                    m.requires_grad_(True)
            if stage == "D":
                self.compiler.ttt.requires_grad_(False)
        else:
            raise ValueError(stage)
        self.tokenizer.requires_grad_(False)

    def trainable_report(self) -> Dict[str, float]:
        rep = {}
        for name, m in (("tokenizer", self.tokenizer), ("obs_enc", self.obs_encoder), ("state_enc", self.state_encoder),
                        ("obs_proj", self.obs_proj), ("E", self.E), ("T", self.compiler.T), ("R", self.compiler.R), ("D", self.compiler.D),
                        ("ttt", self.compiler.ttt), ("fb", self.compiler.fb), ("wm", self.wm),
                        ("effect", self.effect_head), ("bank", self.bank), ("target", self.target)):
            rep[name] = sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6
        rep["TOTAL_trainable_M"] = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        rep["TOTAL_all_M"] = sum(p.numel() for p in self.parameters()) / 1e6
        return rep

    def state_dict_for_ckpt(self) -> Dict[str, torch.Tensor]:
        """Everything except the frozen tokenizer backbone and the frozen VLA slow weights (both are
        re-instantiated from their pretrained sources); injected atoms are kept."""
        sd = {}
        for k, v in self.state_dict().items():
            if k.startswith("tokenizer."):
                continue
            if k.startswith("target.policy.") and ".atom_" not in k:
                continue
            sd[k] = v
        return sd

    @torch.no_grad()
    def update_ema(self):
        d = self.args.ema_decay
        for pe, p in zip(self.obs_proj_ema.parameters(), self.obs_proj.parameters()):
            pe.mul_(d).add_(p.detach(), alpha=1 - d)

    # ------------------------------------------------------------------ #
    # formatting (identical to the template)
    # ------------------------------------------------------------------ #
    def _format_image(self, image):
        if image.dim() == 5:
            image = image[:, : self.obs_horizon]
            if image.shape[-1] in [3, 4]:
                image = image.permute(0, 1, 4, 2, 3)
            if image.shape[1] == 1:
                image = image[:, 0]
        elif image.dim() == 4 and image.shape[-1] in [3, 4]:
            image = image.permute(0, 3, 1, 2)
        return image.float()

    def _format_state(self, state):
        if state.dim() == 2:
            state = state.unsqueeze(1)
        return state[:, : self.obs_horizon].float()

    def encode_robot(self, robot_obs: Dict, ema: bool = False) -> torch.Tensor:
        image = robot_obs.get("rgb", robot_obs.get("view_1"))
        state = robot_obs.get("state", robot_obs.get("states"))
        obs_z, _ = self.obs_encoder(self._format_image(image).to(self.device))
        state_z, _ = self.state_encoder(self._format_state(state).to(self.device))
        z = torch.cat([obs_z, state_z], dim=-1)
        return (self.obs_proj_ema if ema else self.obs_proj)(z)

    def robot_progress(self, h: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.prog_head(h), dim=-1)

    @staticmethod
    def phase_assign(p_R: torch.Tensor, rho: torch.Tensor, tau: float = 0.1,
                     prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        """p_R [B,d_p], rho [B,K,d_p] -> alpha [B,K]; optional monotone transition prior."""
        d = (p_R[:, None] - rho).pow(2).sum(-1)
        alpha = torch.softmax(-d / tau, dim=-1)
        if prev is not None:                                  # stay / advance-one / advance-two
            kernel = torch.tensor([0.6, 0.3, 0.1], device=alpha.device)
            K = alpha.shape[1]
            prior = torch.zeros_like(alpha)
            for j, w in enumerate(kernel):
                prior[:, j:] += w * prev[:, : K - j]
            alpha = alpha * (prior + 1e-3)
            alpha = alpha / alpha.sum(-1, keepdim=True)
        return alpha

    # ------------------------------------------------------------------ #
    # batch helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _progress_from_batch(batch: Dict, device) -> Optional[torch.Tensor]:
        """Normalised episode progress in [0,1] of the query chunk, if the dataset provides it.
        Expected keys (add to your paired dataset __getitem__): 'robot_progress' OR
        ('robot_frame_index' and 'robot_episode_length')."""
        if "robot_progress" in batch:
            return batch["robot_progress"].float().to(device).clamp(0, 1)
        if "robot_frame_index" in batch and "robot_episode_length" in batch:
            return (batch["robot_frame_index"].float() / batch["robot_episode_length"].float().clamp_min(1)).to(device).clamp(0, 1)
        return None

    def _select_support(self, inv: torch.Tensor, U: int, n_s: int) -> torch.Tensor:
        """[U, n_s] sample indices used as human support for each task in the batch (never the same
        robot chunk twice as both support and query is a non-issue: support is a human clip)."""
        idx = torch.empty(U, n_s, dtype=torch.long)
        for u in range(U):
            members = (inv == u).nonzero(as_tuple=True)[0].tolist()
            pick = random.sample(members, n_s) if len(members) >= n_s else [random.choice(members) for _ in range(n_s)]
            idx[u] = torch.tensor(pick)
        return idx

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #
    def compute_loss(self, batch: Dict) -> Dict[str, torch.Tensor]:
        a = self.args
        dev = self.device
        robot_obs = {k: v.to(dev, non_blocking=True) for k, v in batch["robot_obs"].items()}
        actions = batch["robot_actions"].to(dev, non_blocking=True).float()
        B = actions.shape[0]
        out: Dict[str, torch.Tensor] = {}

        h = self.encode_robot(robot_obs)                                             # [B, h_dim]

        # ---------------- stage A: base policy (scratch) / domain LoRA (vla) --------
        # No human task/video information is needed at all in stage A.
        if self.stage == "A":
            P = None if a.target == "scratch" else self.bank(torch.zeros(B, dtype=torch.long, device=dev))
            l_fm, _, _, _, _ = self.target.loss_and_velocity(P, robot_obs, actions, h)
            total = a.w_fm * l_fm
            out["fm"] = l_fm.detach()
            l_world = self._world_loss(batch, robot_obs, h, actions)
            if l_world is not None:
                total = total + a.w_world * l_world; out["world"] = l_world.detach()
            out["loss"] = total
            return out

        # Stage B+ needs task identity. Stage C/D/E additionally consume human video.
        task_idx = self.registry.encode(list(batch.get("human_repo_id", ["_"] * B)), dev, allow_new=True)
        task_idx = task_idx.clamp_min(0)

        # ---------------- stage B: per-task expert bank ---------------------------
        if self.stage == "B":
            l_fm, _, _, _, _ = self.target.loss_and_velocity(self.bank(task_idx), robot_obs, actions, h)
            out["fm"] = l_fm.detach(); out["loss"] = a.w_fm * l_fm
            return out

        # ---------------- human support: cached DINO tokens -> E -----------------
        uniq, inv = torch.unique(task_idx, return_inverse=True)
        U = uniq.shape[0]
        n_s = max(1, a.n_support)

        if self.dino_feature_cache is not None:
            # Registry indices map back to the exact human_repo_id used as cache key.
            unique_repo_ids = [
                self.registry.ids[int(u)]
                for u in uniq.detach().cpu().tolist()
            ]
            with torch.no_grad():
                tok = self.dino_feature_cache.get_support(
                    unique_repo_ids,
                    n_s,
                    dev,
                    distinct=True,
                )  # [U, n_s, N, P, F]
            sup = None
        else:
            # Uncached fallback: preserve the original behavior.
            sup = self._select_support(inv.cpu(), U, n_s).to(dev)                  # [U, n_s]
            video = batch["human_video"]
            ids = batch.get("human_repo_id")
            with torch.no_grad():
                flat = sup.flatten()
                vid = video[flat.cpu()].to(dev, non_blocking=True)
                tok = self.tokenizer(
                    vid,
                    [ids[i] for i in flat.tolist()] if ids is not None else None,
                )
                tok = tok.view(U, n_s, *tok.shape[1:])                              # [U,n_s,N,P,F]

        enc1 = self.E(tok[:, 0], domain=0)
        enc2 = self.E(tok[:, 1], domain=0) if n_s >= 2 else None

        # encoder-side losses (stage C and onwards)
        l_future = self.E.future_loss(enc1["frame_g"])
        l_prog = LS.order_loss(enc1["Z_prog"])
        if enc2 is not None:
            l_prog = l_prog + LS.tcc_loss(enc1["Z_prog"], enc2["Z_prog"])
        p_R = self.robot_progress(h)
        progress = self._progress_from_batch(batch, dev)
        if progress is not None:
            l_prog = l_prog + LS.robot_progress_loss(p_R, enc1["Z_prog"][inv], progress)
        out["future"], out["prog"] = l_future.detach(), l_prog.detach()
        total = a.w_future * l_future + a.w_prog * l_prog
        if "robot_video" in batch:                                                  # optional robot-domain pass
            # Robot video is not part of the human DINO cache.
            # Pick one robot sample per unique task even when human support came from cache.
            robot_sup = (
                sup[:, 0]
                if sup is not None
                else self._select_support(inv.cpu(), U, 1).to(dev)[:, 0]
            )
            with torch.no_grad():
                rtok = self.tokenizer(
                    batch["robot_video"][robot_sup.cpu()].to(dev, non_blocking=True)
                )
            encR = self.E(rtok, domain=1)
            lc0, le0, lo0 = self.E.domain_losses(enc1, 0)
            lc1, le1, lo1 = self.E.domain_losses(encR, 1)
            l_dom = 0.5 * (lc0 + lc1) + 0.5 * (le0 + le1) + 0.1 * (lo0 + lo1)
            l_prog_hr = LS.tcc_loss(enc1["Z_prog"], encR["Z_prog"]) + LS.soft_dtw(enc1["Z_prog"], encR["Z_prog"])
            total = total + a.w_domain * l_dom + a.w_prog * l_prog_hr
            out["domain"] = l_dom.detach()
        if self.stage == "C":
            out["loss"] = total
            return out

        # ---------------- stage D/E: compile -> decode -> FM ------------------------
        human_tokens = self.compiler.T.human_tokens(enc1)                            # [U, n_h, d]
        if a.demo_dropout > 0 and self.training:
            drop = (torch.rand(U, device=dev) < a.demo_dropout)[:, None, None]
            human_tokens = torch.where(drop, self.null_token.expand(U, human_tokens.shape[1], -1), human_tokens)
        S0 = self.compiler.compile(human_tokens)                                    # [U, L, d]
        if self.stage == "E" and a.ttt_steps_train > 0:
            S0 = self.compiler.run_ttt(S0, enc1, human_tokens, self.E, a.ttt_steps_train, create_graph=True)
        rho = self.E.phase_anchors(enc1, uniform=a.uniform_phases)                    # [U, K, d_p]
        alpha = self.phase_assign(p_R, rho[inv])                                     # [B, K]
        P_task = self.compiler.decode(S0, None, None)                                # per task (param losses)
        P = self.compiler.decode(S0, alpha, inv)                                     # per sample (forward)

        l_fm, v, x0_hat, noise, u = self.target.loss_and_velocity(P, robot_obs, actions, h)
        total = total + a.w_fm * l_fm
        out["fm"] = l_fm.detach()

        # teacher losses + parameter regularisers, streamed layer by layer (VLA experts are big)
        names = [sp.name for sp in self.target.layer_specs()]
        gen_fn = lambda n: self.compiler.D.dense_delta_layer(P_task, n, self.target)
        tea_fn = (lambda n: self.bank.dense_layer(n, uniq)) if self.bank_ready else None
        st = LS.streaming_param_stats(names, gen_fn, tea_fn, self.target.base_norms())
        if st["rel"] is not None:
            total = total + a.w_param * st["rel"] + a.w_dir * st["cos"] + a.w_mag * st["mag"]
            out["param"], out["dir"], out["mag"] = st["rel"].detach(), st["cos"].detach(), st["mag"].detach()
        if st["e_tea"] is not None:
            blocks_used = [self.target.layer(n).spec.block for n in st["used_tea"]]
            l_layer = LS.layer_energy_kl(self.compiler.D.block_gates(S0), st["e_tea"], blocks_used)
            total = total + a.w_layer * l_layer; out["layer"] = l_layer.detach()
        if self.bank_ready and v is not None and a.w_func > 0:
            with torch.no_grad():
                _, v_t, _, _, _ = self.target.loss_and_velocity(self.bank(task_idx), robot_obs, actions, h, noise, u)
            if v_t is not None:
                l_func = F.mse_loss(v, v_t)
                total = total + a.w_func * l_func; out["func"] = l_func.detach()
        if st["dw_reg"] is not None:
            total = total + a.w_dw * st["dw_reg"]; out["dw"] = st["dw_reg"].detach()
        l_geom = LS.geometry_loss_from_dw2(enc1["z_eff"], st["dw2"])
        if l_geom is not None:
            total = total + a.w_geom * l_geom; out["geom"] = l_geom.detach()
        l_ortho = LS.atom_ortho_reg(self.target.lora_layers())
        l_smooth = LS.skill_smooth_reg(S0)
        total = total + a.w_ortho * l_ortho + a.w_smooth * l_smooth

        # intra-task consistency in function space (second human clip -> same policy function)
        if enc2 is not None and a.w_intra > 0 and v is not None:
            S0b = self.compiler.compile(self.compiler.T.human_tokens(enc2))
            Pb = self.compiler.decode(S0b, alpha, inv)
            _, vb, _, _, _ = self.target.loss_and_velocity(Pb, robot_obs, actions, h, noise, u)
            if vb is not None:
                l_intra = F.mse_loss(v, vb)
                total = total + a.w_intra * l_intra; out["intra"] = l_intra.detach()

        # world / effect losses (need x0_hat and, for L_world, the next observation)
        l_world = self._world_loss(batch, robot_obs, h, actions)
        if l_world is not None:
            total = total + a.w_world * l_world; out["world"] = l_world.detach()
        if x0_hat is not None and a.w_effect > 0:
            h_end = self.wm(h.detach(), x0_hat)
            e_R = self.effect_head(h.detach(), h_end)
            l_eff = (1 - F.cosine_similarity(e_R, enc1["z_eff"][inv].detach(), dim=-1)).mean()
            total = total + a.w_effect * l_eff; out["effect"] = l_eff.detach()

        # ---------------- stage E: feedback-refined second pass ---------------------
        if self.stage == "E" and a.feedback_pass:
            h_fb = h.detach()
            if a.p_img > 0 and x0_hat is not None:                                  # imagined (drifted) feedback
                with torch.no_grad():
                    h_img = self.wm(h_fb, x0_hat.detach())
                use_img = (torch.rand(B, device=dev) < a.p_img)[:, None]
                h_fb = torch.where(use_img, h_img, h_fb)
            chunk_idx = (progress * 32).long() if progress is not None else torch.zeros(B, dtype=torch.long, device=dev)
            a_prev_fb = x0_hat.detach() if (a.use_prev_action and x0_hat is not None) else None
            r = self.compiler.fb(h_fb, self.robot_progress(h_fb), chunk_idx, a_prev=a_prev_fb)[:, None]   # [B,1,d]
            S1 = self.compiler.step(S0[inv], human_tokens[inv], r)                   # [B, L, d]
            alpha1 = self.phase_assign(self.robot_progress(h_fb), rho[inv])
            P1 = self.compiler.decode(S1, alpha1, None)
            l_fb, _, _, _, _ = self.target.loss_and_velocity(P1, robot_obs, actions, h, noise, u)
            total = total + a.w_fb * l_fb; out["fm_fb"] = l_fb.detach()

        out["loss"] = total
        return out

    def _world_loss(self, batch, robot_obs, h, actions) -> Optional[torch.Tensor]:
        """L_world needs the observation one chunk ahead: batch['robot_next_obs'] (same format as robot_obs)."""
        nxt = batch.get("robot_next_obs")
        if nxt is None:
            return None
        nxt = {k: v.to(self.device, non_blocking=True) for k, v in nxt.items()}
        with torch.no_grad():
            h_next = self.encode_robot(nxt, ema=True)
        return F.mse_loss(self.wm(h, actions), h_next)

    # ------------------------------------------------------------------ #
    # evaluation (simulator interface)
    # ------------------------------------------------------------------ #
    def prepare_for_eval(self, human_video: Optional[torch.Tensor], robot_obs: Dict,
                         human_desc: Optional[List[str]] = None, human_vl_ids: Optional[List[str]] = None):
        del human_desc
        self.eval()
        a = self.args
        with torch.no_grad():
            if human_video is None and self.dino_feature_cache is not None and human_vl_ids:
                # Optional cache-only eval path. If an explicit demo video is supplied,
                # always encode that exact video instead of substituting a cached exemplar.
                tok = self.dino_feature_cache.get_support(
                    human_vl_ids, 1, self.device, distinct=False
                )[:, 0]
            else:
                if human_video is None:
                    raise ValueError(
                        "prepare_for_eval needs human_video unless a DINO cache and human_vl_ids are provided"
                    )
                tok = self.tokenizer(human_video.to(self.device), human_vl_ids)
            enc = self.E(tok, domain=0)
            human_tokens = self.compiler.T.human_tokens(enc)
            S = self.compiler.compile(human_tokens)
        if a.ttt_steps_eval > 0:
            S = self.compiler.run_ttt(S, enc, human_tokens, self.E, a.ttt_steps_eval, create_graph=False).detach()
        self._cache = dict(S=S, human_tokens=human_tokens, rho=self.E.phase_anchors(enc, uniform=a.uniform_phases),
                           fb=[], n_calls=0, n_updates=0, alpha_prev=None, prev_action=None)

    @torch.no_grad()
    def get_action(self, obs: Dict) -> torch.Tensor:
        assert self._cache is not None, "call prepare_for_eval() first"
        c, a = self._cache, self.args
        obs = {k: v.to(self.device) for k, v in obs.items()}
        h = self.encode_robot(obs)
        B = h.shape[0]
        S, ht, rho = c["S"], c["human_tokens"], c["rho"]
        if S.shape[0] != B:                                    # one human video shared by all envs
            S, ht, rho = S.expand(B, -1, -1), ht.expand(B, -1, -1), rho.expand(B, -1, -1)
            c["S"], c["human_tokens"], c["rho"] = S, ht, rho
        p_R = self.robot_progress(h)
        k = a.state_update_every
        can_update = (a.max_state_updates <= 0 or c["n_updates"] < a.max_state_updates)
        if k > 0 and c["n_calls"] > 0 and c["n_calls"] % k == 0 and can_update:  # closed-loop recurrence
            r = self.compiler.fb(
                h, p_R,
                torch.full((B,), c["n_calls"] // k, dtype=torch.long, device=self.device),
                a_prev=c.get("prev_action") if a.use_prev_action else None,
            )
            c["fb"].append(r)
            c["S"] = S = self.compiler.step(S, ht, torch.stack(c["fb"][-a.M_fb:], dim=1))
            c["n_updates"] += 1
        alpha = self.phase_assign(p_R, rho, prev=c["alpha_prev"])
        c["alpha_prev"] = alpha
        P = self.compiler.decode(S, alpha, None)
        c["n_calls"] += 1
        action = self.target.sample(P, obs, h)
        c["prev_action"] = action.detach()
        return action

    def clear_cache(self):
        self._cache = None
