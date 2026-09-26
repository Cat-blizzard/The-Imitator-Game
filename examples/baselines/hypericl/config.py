"""All CLI args (tyro) for train_hypericl_imitator.py. Superset of the template's TrainingArgs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class HyperICLArgs:
    # ---- experiment -------------------------------------------------------
    exp_name: Optional[str] = None
    seed: int = 1
    cuda: bool = True
    stage: str = "D"                 # A base | B expert bank | C video encoder | D compiler | E closed-loop+TTT
    init_from: Optional[str] = None  # checkpoint of the previous stage (strict=False load)
    init_dict_from_bank: bool = True # at the start of stage D: SVD-init manifold atoms from the expert bank

    # ---- experiment tracking ------------------------------------------------
    track: bool = True
    wandb_project_name: str = "hypericl"
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_mode: str = "offline"             # online | offline | disabled

    # ---- dataset (The Imitator Game contract; identical to the template) ----
    human_root: str = "/mnt/sdf/engram_sz_DataCollection/imitator_zed2i_224"
    sim_root: str = "/mnt/sdf/engram_sz_DataCollection/IG-10K-Dataset/imitator_sim_v1_zed2i"
    task_mapping_file: str = "examples/baselines/lerobot_dataset/task_mapping.json"
    human_dataset_file: str = "examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_15.json"
    sim_dataset_file: str = "examples/baselines/lerobot_dataset/config/exp_configs/sim_train_config_15.json"
    human_task_description_file: str = "examples/baselines/lerobot_dataset/task_desc/human_desc.json"
    sim_task_description_file: str = "examples/baselines/lerobot_dataset/task_desc/sim_desc.json"
    input_mode: str = "video_only"
    cameras: List[str] = field(default_factory=lambda: ["zed2i"])
    image_size: Tuple[int, int] = (224, 224)
    state_type: str = "qpos"
    include_depth: bool = False
    single_arm: bool = False
    fps: int = 30
    video_backend: str = "torchcodec"

    # ---- training ----------------------------------------------------------
    total_epochs: int = 10
    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 16

    # DataLoader tuning. For video datasets, more workers is not always faster:
    # each worker may create its own decoder and compete for disk/CPU.
    prefetch_factor: int = 2
    persistent_workers: bool = True
    multiprocessing_context: str = "forkserver"   # forkserver | spawn | fork | none
    dataloader_in_order: bool = False              # if supported by this PyTorch version
    worker_torch_threads: int = 1                  # prevent CPU thread oversubscription per worker
    pin_memory: bool = True
    profile_dataloader: bool = True                # log data-wait avg/max to TB/W&B

    # A/B do not use human video. Skip HumanVideoDataset construction + decoding.
    skip_human_video_ab: bool = True

    grad_clip: float = 1.0
    ema_decay: float = 0.995          # EMA of the robot latent projector (world-model targets)
    log_freq: int = 20
    save_epoch_freq: int = 5

    # ---- data shapes -------------------------------------------------------
    action_dim: int = 16
    state_dim: int = 18
    pred_horizon: int = 16
    obs_horizon: int = 1

    # ---- frozen video tokenizer --------------------------------------------
    tokenizer_backend: str = "dinov2_hf"      # template | dinov2_hf | dinov2_hub
    frozen_backbone_type: str = "dinov2_vitl14"
    frozen_backbone_num_frames: int = 10
    frozen_backbone_seq_patches: int = 32
    frozen_backbone_adapter_layers: int = 1
    task_latent_dim: int = 256                # only used by the template backbone constructor
    feat_dim: int = 1024                      # per-token feature dim of the tokenizer (ViT-L = 1024)
    hf_model_name: str = "facebook/dinov2-large"
    hf_cache_dir: Optional[str] = "/mnt/sdf/dataset/zhouxunzhe/huggingface"

    # Frozen DINO patch-token cache for stages C/D/E.
    # Cache stores multiple clips per repo so n_support=2 still sees distinct demos.
    dino_cache_enabled: bool = True
    dino_cache_root: str = "cache/hypericl_dino_cache"
    dino_cache_clips_per_repo: int = 16
    dino_cache_batch_size: int = 8
    dino_cache_num_workers: int = 4
    dino_cache_preload_to_memory: bool = True
    dino_cache_pin_memory: bool = True
    dino_cache_recompute: bool = False
    dino_cache_bf16: bool = True
    dino_cache_skip_human_video: bool = True

    # ---- robot encoders (benchmark modules) --------------------------------
    obs_encoder_type: str = "simple_cnn"
    obs_latent_dim: int = 256
    state_encoder_type: str = "mlp"
    state_latent_dim: int = 256
    h_dim: int = 256                          # robot latent h_t

    # ---- VideoProgramEncoder E --------------------------------------------
    d: int = 256
    d_prog: int = 64
    d_emb: int = 32
    K: int = 4                                # event / phase tokens
    n_pool: int = 4                           # spatial tokens kept per frame
    enc_layers: int = 4
    n_support: int = 2                        # human clips per task per batch (2 -> intra-task losses)
    uniform_phases: bool = True               # rho_k from uniform time segments (v1) vs event attention mass

    # ---- HyperCompiler ---------------------------------------------------------
    T_layers: int = 4
    nhead: int = 4
    M_fb: int = 8                             # feedback tokens kept in the recurrence
    n_atoms: int = 16                         # manifold dictionary size M
    use_manifold: bool = True
    use_residual: bool = True
    use_prev_action: bool = False
    demo_dropout: float = 0.1                 # demo-CFG: probability of a null support per task
    ttt_steps_train: int = 0                  # latent TTT steps unrolled during training (stage E: 1-2)
    ttt_steps_eval: int = 0
    feedback_pass: bool = True                # stage E: second pass with the observation feedback token
    p_img: float = 0.0                        # stage E: prob. of imagined (world-model) feedback
    state_update_every: int = 0               # eval: update S every k get_action calls (0 = one-shot only)

    # ---- target policy: from-scratch FM-DiT  or  pretrained VLA/WAM ----------
    target: str = "scratch"                   # scratch | vla
    share_res_heads: bool = True              # shared residual heads per layer-shape (keeps D small)
    trunk_dim: int = 64

    # ---- block-level learned-optimizer refinement -----------------------------
    # T first predicts one skill token per target block. R then performs a small,
    # topology-aware pseudo-forward/backward refinement in latent space before D.
    use_block_refiner: bool = True
    refiner_steps: int = 2
    refiner_layers: int = 1
    refiner_lr_max: float = 0.10
    refiner_on_feedback: bool = False       # safer until recurrent updates are trained with long unrolls
    phase_manifold: bool = False             # phase can change manifold coefficients, not only AdaLN bias
    feedback_gate_max: float = 0.25         # bound recurrent S updates to prevent long-horizon drift
    max_state_updates: int = 0              # 0 = unlimited; use 8/16 as a diagnostic if eval drifts
    # -- vla target (pretrained; see targets/pretrained_target.py) --
    vla_profile: str = "lerobot_smolvla"      # lerobot_smolvla | lerobot_pi0 | groot_n15 | custom
    vla_repo_id: str = "lerobot/smolvla_base"
    vla_lora_regex: str = ""                  # "" -> profile default; which nn.Linear get generated deltas
    vla_velocity_module: str = ""             # "" -> profile default; module whose output is the predicted velocity
    vla_task_text: str = "do the task shown in the demonstration"
    vla_identity_norm: bool = True            # the Imitator dataset already normalises; bypass VLA normalizers
    vla_max_layers: int = 0                   # 0 = all matched layers; else keep the LAST n expert layers
    # -- scratch target --
    dit_dim: int = 256
    dit_depth: int = 6
    dit_nhead: int = 4
    rank: int = 8
    ada_rank: int = 16
    sample_steps: int = 10
    max_tasks: int = 64                       # capacity of the expert bank / task registry
    train_obs_in_D: bool = False              # keep obs/state encoders frozen after stage A

    # ---- loss weights (DESIGN §5.6) ---------------------------------------
    w_fm: float = 1.0
    w_param: float = 0.2
    w_dir: float = 0.1
    w_mag: float = 0.05
    w_layer: float = 0.05
    w_func: float = 0.2
    w_prog: float = 0.1
    w_future: float = 0.1
    w_world: float = 0.1
    w_effect: float = 0.05
    w_dw: float = 0.01
    w_ortho: float = 1e-3
    w_smooth: float = 1e-3
    w_intra: float = 0.05
    w_domain: float = 0.05
    w_geom: float = 0.02
    w_fb: float = 1.0                         # FM loss of the feedback-refined pass (stage E)
