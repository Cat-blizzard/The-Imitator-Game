"""Run one finite cache stage after the registered GPU comparison finishes."""
import argparse
import fcntl
import os
from pathlib import Path
import time

from resimpair_grasp.common import ROOT, read, write, binding


def run(root):
    from .asset_pool_v1 import caches, report
    root = Path(root).resolve()
    with (root / '.asset_cache_stage.lock').open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        receipt = root / 'cache_stage_started.json'
        if not receipt.exists():
            write(receipt, {'pid': os.getpid(), 'started_unix': time.time(),
                'maximum_seconds': 7200, 'protocol': binding(root / 'protocol.json'),
                'implementation': binding(ROOT / 'resimpair_tasks/asset_pool_v1.py')})
        previous = read(receipt)
        if previous['implementation'] != binding(ROOT / 'resimpair_tasks/asset_pool_v1.py'):
            raise ValueError('Asset worker changed; separate candidate required')
        while not (root / 'controller_comparison.json').exists():
            if time.monotonic() - started > 1800:
                raise TimeoutError('Controller comparison unresolved; no GPU optimization submitted')
            time.sleep(5)
        total = len(read(root / 'asset_selection.json')['rows'])
        while time.monotonic() - started < 7200:
            caches(root)
            terminal = list((root / 'asset_work').glob('*/cache_terminal.json'))
            print({'cache_terminal': len(terminal), 'selected': total}, flush=True)
            if len(terminal) == total:
                result = report(root)
                write(root / 'cache_stage_terminal.json', {'state': 'complete',
                    'elapsed_seconds': time.monotonic() - started, 'ready': result['ready'],
                    'statuses': result['statuses'], 'api_calls': 0})
                return
            time.sleep(5)
        raise TimeoutError('Finite cache stage deadline reached; recover existing jobs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'runs/batch500_20261027')
    run(parser.parse_args().root)
