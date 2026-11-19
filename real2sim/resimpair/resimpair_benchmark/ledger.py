"""Shared finite job ledger: output relocation never creates new allowance."""
import fcntl
import time
from contextlib import contextmanager
from pathlib import Path

from resimpair_grasp.common import binding, digest, read, resolve, write


@contextmanager
def locked(root):
    root = Path(root)
    with (root / '.ledger.lock').open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def reserve(root, identity, request, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    with locked(root):
        limits = read(root / 'limits.json')
        key = digest(identity)
        path = root / 'jobs' / (key + '.json')
        if path.exists():
            old = read(path)
            if old['request'] != request or old['output'] != str(output):
                raise ValueError('Existing job identity cannot be rebound to another input/output')
            return old, False
        used = len(list((root / 'jobs').glob('*.json')))
        if used >= limits['gpu_job_limit']:
            raise RuntimeError('Shared GPU job budget exhausted')
        record = {'identity': identity, 'request': request, 'output': str(output),
                  'sequence': used, 'reserved_unix': time.time(),
                  'limits': binding(root / 'limits.json')}
        write(path, record)
        return record, True


def check_request(root, identity, request, output):
    p = Path(root) / 'jobs' / (digest(identity) + '.json')
    record = read(p)
    resolve(record['limits'])
    if record['request'] != request or record['output'] != str(Path(output).resolve()):
        raise ValueError('Worker input does not match shared reservation')
    return record
