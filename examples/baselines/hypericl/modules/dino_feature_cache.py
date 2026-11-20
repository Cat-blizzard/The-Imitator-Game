"""Multi-clip DINO patch-token cache for HyperICL.

Why cache here
--------------
HyperICL trains VideoProgramEncoder E, but DINO itself is frozen. Therefore the
expensive DINO output [N, P, F] can be precomputed once and replayed while E
continues to receive gradients normally.

Unlike the older task-level cache, this cache stores multiple clips per human
repo/task. That matters because HyperICL uses n_support >= 2 for TCC and
intra-task consistency; mapping each repo to a single feature would silently
make both supports identical.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


CACHE_VERSION = 1


def _safe_slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))


class DinoClipFeatureCache:
    """Disk + RAM cache.

    store[repo_id][slot] = Tensor[N, P, F] on CPU, usually float16.
    """

    def __init__(
        self,
        cache_root: str,
        *,
        backend: str,
        model_name: str,
        num_frames: int,
        seq_patches: int,
        feat_dim: int,
        image_size: int,
        clips_per_repo: int,
        preload_to_memory: bool = True,
        pin_memory: bool = False,
        verbose: bool = True,
    ):
        self.meta = {
            "version": CACHE_VERSION,
            "backend": str(backend),
            "model_name": str(model_name),
            "num_frames": int(num_frames),
            "seq_patches": int(seq_patches),
            "feat_dim": int(feat_dim),
            "image_size": int(image_size),
            "clips_per_repo": int(clips_per_repo),
        }
        fp = hashlib.sha1(
            json.dumps(self.meta, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]

        tag = _safe_slug(model_name.split("/")[-1])
        self.cache_dir = Path(cache_root) / f"hypericl_dino_{tag}_{fp}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.cache_dir / "patch_tokens.pt"

        self.store: Dict[str, Dict[int, torch.Tensor]] = {}
        self.pin_memory = bool(pin_memory and torch.cuda.is_available())
        self._log = print if verbose else (lambda *a, **k: None)

        if preload_to_memory and self.path.exists():
            self.load()

    @property
    def clips_per_repo(self) -> int:
        return int(self.meta["clips_per_repo"])

    @property
    def num_repos(self) -> int:
        return len(self.store)

    @property
    def num_clips(self) -> int:
        return sum(len(v) for v in self.store.values())

    @property
    def is_populated(self) -> bool:
        return self.num_clips > 0

    def slots(self, repo_id: str) -> List[int]:
        return sorted(self.store.get(str(repo_id), {}).keys())

    def has_slot(self, repo_id: str, slot: int) -> bool:
        return int(slot) in self.store.get(str(repo_id), {})

    def is_complete(
        self,
        repo_ids: Sequence[str],
        clips_per_repo: Optional[int] = None,
    ) -> bool:
        k = self.clips_per_repo if clips_per_repo is None else int(clips_per_repo)
        return all(len(self.slots(rid)) >= k for rid in repo_ids)

    def put(
        self,
        repo_id: str,
        slot: int,
        tokens: torch.Tensor,
    ) -> None:
        rid = str(repo_id)
        slot = int(slot)
        x = tokens.detach().to(device="cpu", dtype=torch.float16).contiguous()
        if self.pin_memory:
            x = x.pin_memory()
        self.store.setdefault(rid, {})[slot] = x

    def get_support(
        self,
        repo_ids: Sequence[str],
        n_support: int,
        device: torch.device,
        *,
        distinct: bool = True,
    ) -> torch.Tensor:
        """Return [U, n_support, N, P, F].

        Each repo samples support clips independently from its cached clip bank.
        """
        n_support = max(1, int(n_support))
        per_repo = []

        for rid in repo_ids:
            rid = str(rid)
            slots = self.slots(rid)
            if not slots:
                raise KeyError(
                    f"DINO cache miss for repo_id={rid!r}. Cache file: {self.path}"
                )

            if distinct and len(slots) >= n_support:
                chosen = random.sample(slots, n_support)
            else:
                chosen = [random.choice(slots) for _ in range(n_support)]

            clips = [self.store[rid][s] for s in chosen]
            per_repo.append(torch.stack(clips, dim=0))

        batch = torch.stack(per_repo, dim=0)  # [U, S, N, P, F]
        return batch.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )

    def flush(self) -> None:
        payload = {
            "meta": self.meta,
            "store": self.store,
        }
        tmp = self.path.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(self.path)
        self._log(
            f"[DinoCache] saved {self.num_clips} clips / {self.num_repos} repos -> {self.path}"
        )

    def load(self) -> None:
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "store" not in payload:
            raise RuntimeError(f"Invalid DINO cache file: {self.path}")

        old_meta = payload.get("meta", {})
        if old_meta != self.meta:
            raise RuntimeError(
                "DINO cache metadata mismatch.\n"
                f"expected={self.meta}\n"
                f"found={old_meta}\n"
                f"path={self.path}"
            )

        self.store = payload["store"]
        if self.pin_memory:
            for rid, slots in self.store.items():
                for s, x in slots.items():
                    if not x.is_pinned():
                        slots[s] = x.pin_memory()

        self._log(
            f"[DinoCache] loaded {self.num_clips} clips / {self.num_repos} repos <- {self.path}"
        )

    def clear(self) -> None:
        self.store = {}
        if self.path.exists():
            self.path.unlink()


def collect_training_repo_ids(paired_dataset) -> List[str]:
    """Collect exactly the human repo ids used by paired_dataset.__getitem__."""
    seen = set()
    mapping = getattr(paired_dataset, "task_to_human_indices", {}) or {}

    for info in mapping.values():
        repos = info.get("repos", []) if isinstance(info, dict) else []
        if repos:
            # HumanSimPairedDataset currently uses repos[0]["repo_id"].
            rid = repos[0].get("repo_id")
            if rid:
                seen.add(str(rid))

    return sorted(seen)


class _ClipPrecomputeDataset(Dataset):
    def __init__(
        self,
        human_dataset,
        jobs: Sequence[Tuple[str, int]],
    ):
        self.human_dataset = human_dataset
        self.jobs = list(jobs)

    def __len__(self):
        return len(self.jobs)

    def __getitem__(self, idx):
        rid, slot = self.jobs[idx]
        item = self.human_dataset._get_target_item(rid)
        return {
            "repo_id": rid,
            "slot": slot,
            "video": item["video"],
            "episode_idx": int(item.get("episode_idx", -1)),
        }


def _worker_init(worker_id: int):
    seed = 9173 + int(worker_id)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    try:
        import cv2
        cv2.setNumThreads(0)
    except Exception:
        pass


def _collate(batch):
    return {
        "repo_id": [b["repo_id"] for b in batch],
        "slot": [int(b["slot"]) for b in batch],
        "episode_idx": [int(b["episode_idx"]) for b in batch],
        "video": torch.stack([b["video"] for b in batch], dim=0),
    }


def precompute_dino_clip_features(
    tokenizer,
    paired_dataset,
    cache: DinoClipFeatureCache,
    device: torch.device,
    *,
    clips_per_repo: int,
    batch_size: int = 8,
    num_workers: int = 4,
    use_bf16: bool = True,
    verbose: bool = True,
) -> DinoClipFeatureCache:
    """Populate missing cache slots.

    DINO/tokenizer is frozen. VideoProgramEncoder E is NOT involved.
    """
    log = print if verbose else (lambda *a, **k: None)

    human_dataset = getattr(paired_dataset, "human_dataset", None)
    if human_dataset is None:
        raise RuntimeError(
            "Cannot precompute DINO cache: paired_dataset.human_dataset is None. "
            "Build the C/D/E dataloader with human video enabled for the first cache build."
        )

    repo_ids = collect_training_repo_ids(paired_dataset)
    if not repo_ids:
        raise RuntimeError("No human repo ids found for DINO cache precompute.")

    jobs = []
    for rid in repo_ids:
        for slot in range(int(clips_per_repo)):
            if not cache.has_slot(rid, slot):
                jobs.append((rid, slot))

    log(
        f"[DinoCache] repos={len(repo_ids)}, clips/repo={clips_per_repo}, "
        f"cached={cache.num_clips}, missing={len(jobs)}"
    )

    if not jobs:
        log("[DinoCache] cache already complete; no DINO precompute needed.")
        return cache

    # Cache canonical features, not a single frozen realization of ColorJitter.
    cfg = getattr(human_dataset, "config", None)
    aug_was_on = bool(getattr(cfg, "enable_augmentation", False))
    if aug_was_on and hasattr(human_dataset, "_setup_transforms_human"):
        cfg.enable_augmentation = False
        human_dataset._setup_transforms_human(cfg)
        log("[DinoCache] temporarily disabled human RGB augmentation during precompute.")

    ds = _ClipPrecomputeDataset(human_dataset, jobs)
    nw = min(max(0, int(num_workers)), len(ds))
    kwargs = dict(
        dataset=ds,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=nw,
        collate_fn=_collate,
        pin_memory=True,
        drop_last=False,
    )
    if nw > 0:
        kwargs["persistent_workers"] = False
        kwargs["prefetch_factor"] = 2
        kwargs["worker_init_fn"] = _worker_init

    loader = DataLoader(**kwargs)

    tokenizer.eval()
    try:
        with torch.no_grad():
            for batch in tqdm(
                loader,
                desc="[DinoCache] precomputing patch tokens",
                disable=not verbose,
            ):
                video = batch["video"].to(device, non_blocking=True)

                amp_enabled = bool(use_bf16 and device.type == "cuda")
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=amp_enabled,
                ):
                    tok = tokenizer(video)
                    # [B, N, P, F]

                tok = tok.detach().cpu()
                for i, (rid, slot) in enumerate(zip(batch["repo_id"], batch["slot"])):
                    cache.put(rid, slot, tok[i])

                # Crash-safe incremental persistence.
                cache.flush()
    finally:
        if aug_was_on and hasattr(human_dataset, "_setup_transforms_human"):
            cfg.enable_augmentation = True
            human_dataset._setup_transforms_human(cfg)

    log(
        f"[DinoCache] complete: {cache.num_clips} clips / "
        f"{cache.num_repos} repos."
    )
    return cache


def setup_dino_feature_cache(
    tokenizer,
    dataloader,
    device: torch.device,
    args,
):
    """Build/load cache and return it.

    This is deliberately HyperICL-specific: cache entries are multi-clip
    [N,P,F] DINO tokens, not pooled task embeddings.
    """
    cache = DinoClipFeatureCache(
        args.dino_cache_root,
        backend=args.tokenizer_backend,
        model_name=args.hf_model_name,
        num_frames=args.frozen_backbone_num_frames,
        seq_patches=args.frozen_backbone_seq_patches,
        feat_dim=args.feat_dim,
        image_size=args.image_size[0],
        clips_per_repo=args.dino_cache_clips_per_repo,
        preload_to_memory=args.dino_cache_preload_to_memory,
        pin_memory=args.dino_cache_pin_memory,
        verbose=True,
    )

    if args.dino_cache_recompute:
        print("[DinoCache] recompute=True: clearing existing cache.")
        cache.clear()

    precompute_dino_clip_features(
        tokenizer,
        dataloader.dataset,
        cache,
        device,
        clips_per_repo=args.dino_cache_clips_per_repo,
        batch_size=args.dino_cache_batch_size,
        num_workers=args.dino_cache_num_workers,
        use_bf16=args.dino_cache_bf16,
        verbose=True,
    )
    return cache
