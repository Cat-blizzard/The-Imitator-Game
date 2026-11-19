"""Cache only the fixed full-video reference frames: ~13 GB, not all frames.

Written lazily by training after frozen feature extraction; no extra GPU process.
The schema must match and source interval identity is part of each cache key.
"""
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def setup_cache(path,config,tokenizer_name):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    spec=dict(version=1,model=tokenizer_name,frames=config.frames,image_size=config.image_size,
        patches=config.seq_patches,features=config.feat_dim,teacher_frozen=True,
        frame_selection='linspace(0,length-1,frames).round',dtype='float16',feature_precision='autocast_bfloat16')
    meta=path/'spec.json'
    if meta.exists() and json.loads(meta.read_text())!=spec:raise ValueError('Reference cache configuration mismatch')
    if not meta.exists():meta.write_text(json.dumps(spec,indent=2))
    return path


def row_key(row):
    identity={k:row.get(k) for k in ('id','video','start_seconds','fps','length')}
    return int(hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:15],16)


def cache_path(root,key):return Path(root)/f'{int(key):015x}.npy'


def save_tokens(root,key,tokens):
    path=cache_path(root,key)
    if path.exists():return
    values=tokens.detach().float().cpu().numpy().astype(np.float16)
    if not np.isfinite(values).all():raise ValueError('Nonfinite frozen reference features')
    temp=Path(str(path)+f'.{os.getpid()}.tmp')
    with temp.open('wb') as f:np.save(f,values,allow_pickle=False)
    temp.replace(path)
