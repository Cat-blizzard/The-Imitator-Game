"""Resume only unresolved finite stages, yielding to unrelated GPU owners."""
from types import SimpleNamespace
import time

from resimpair_grasp.common import read, write
from .campaign import BASE, gpu, qualify


def finish():
    from resimpair_pickplace import stability_runner
    deadline = time.monotonic() + 3600
    def when_idle(action):
        while True:
            try: return action()
            except RuntimeError as error:
                if 'Selected GPU is occupied' not in str(error): raise
                if time.monotonic() >= deadline: raise RuntimeError('Bounded resource-wait deadline reached')
                print('Unrelated GPU owner: waiting; no resubmission', flush=True)
                time.sleep(5)
    when_idle(lambda: gpu('heldout_asset', ['hybrid']))
    qualify('heldout_asset')
    root = BASE / 'physics_heldout_asset'
    cases = read(root / 'cohort.json')['cases']
    while any(not (root / 'executions' / c['id'] / 'terminal.json').exists() for c in cases):
        when_idle(lambda: stability_runner.run(SimpleNamespace(root=root, case=[], limit=24)))
    result = stability_runner.audit(SimpleNamespace(root=root, case=[], limit=24))
    write(BASE / 'comparison_execution_complete.json', {'finished_unix': time.time(),
        'physics_audit': result, 'paid_api_calls': 0})
    print('Frozen comparison complete', flush=True)


if __name__ == '__main__': finish()
