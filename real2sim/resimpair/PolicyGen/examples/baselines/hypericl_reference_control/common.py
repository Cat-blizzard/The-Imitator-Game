import hashlib
import json
import os
from pathlib import Path

DEFAULT_INIT='/mnt/sdf/datasets/hypericl_dense_runs/fullweight_original3750_20261009_v1/dense_aux/joint/epoch_0005.pt'
DEFAULT_ROOT='/mnt/sdf/datasets/hypericl_reference_control_runs/stir_direction_20261014_v1'

def require(value,message):
    if not value:raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

def array_sha(x):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def read(p):return json.loads(Path(p).read_text())

def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()))
    with tmp.open('w') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

def bind_gpu(index):
    from ..hypericlzq.evaluate_isolated_gpu import gpu_identity
    import sys
    require('torch' not in sys.modules and 'sapien' not in sys.modules,'Bind GPU before physical imports')
    gpu=gpu_identity(index)
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu['uuid'],USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='1',
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
    return gpu

def source_pins():
    root=Path(__file__).resolve().parents[3]
    files=list(Path(__file__).parent.glob('*.py'))
    for relative in ['hypericl_dense/model.py','hypericl_dense/dense_decoder.py','hypericl_dense/generated_dit.py',
                     'hypericlzq/model.py','hypericlzq/eef.py','hypericlzq/modules/frame_tokenizer.py',
                     'hypericlzq/modules/video_program_encoder.py']:
        files.append(Path(__file__).parent.parent/relative)
    files.append(root/'mani_skill/envs/tasks/tabletop/dual_tasks/_001_stir_spoon.py')
    return {str(p):sha(p) for p in files}
