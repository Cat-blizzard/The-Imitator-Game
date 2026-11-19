"""Host GPU ownership and interprocess locks, independent of historical datasets."""
from pathlib import Path
import os
import subprocess
import tempfile


def gpu_snapshot(index=0):
    value = subprocess.check_output(['nvidia-smi', '-i', str(index),
        '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
        text=True, timeout=20)
    ident, memory, utilization = [x.strip() for x in value.strip().split(',')]
    owners = subprocess.check_output(['nvidia-smi',
        '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
        '--format=csv,noheader,nounits'], text=True, timeout=20)
    return dict(gpu=index, uuid=ident, memory_MiB=int(memory), utilization=int(utilization),
        compute_owners=[x for x in owners.splitlines() if x.split(',')[0].strip() == ident])


def check_idle(snapshot):
    if snapshot['compute_owners'] or snapshot['memory_MiB'] > 512 or snapshot['utilization'] > 10:
        raise RuntimeError('Selected GPU is occupied; no existing process is stopped')


def lock_paths(index):
    uuid = gpu_snapshot(index)['uuid']
    path = Path(tempfile.gettempdir()) / ('resimpair-' + uuid + '.lock')
    path.touch(exist_ok=True)
    paths = [path]
    for value in os.environ.get('RESIMPAIR_EXTRA_GPU_LOCKS', '').split(os.pathsep):
        if value:
            p = Path(value)
            if not p.is_file():
                raise FileNotFoundError('Explicit GPU coordination lock missing')
            paths.append(p)
    return paths
