"""Frozen retained-request comparison before larger task production."""
import argparse
from pathlib import Path
import time
from types import SimpleNamespace

from resimpair_grasp.common import ROOT, binding, read, write, resolve


def run(root):
    from resimpair_pickplace.cache_runtime import initialize
    from resimpair_pickplace.stability_runner import run as execute
    from resimpair_pickplace.requested_goal_audit import audit
    from resimpair_pickplace.runner import code
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import resources
    root = Path(root).resolve()
    previous = ROOT / 'runs/requested_pickplace_20261027'
    protocol = root / 'controller_protocol.json'
    if not protocol.exists():
        write(protocol, {'schema': 'cartesian_transfer_comparison_v1',
            'motion_policy': 'cartesian_transfer_v1', 'release_policy': 'legacy_v1',
            'contact_policies': ['native_default', 'small_tabletop_v1'],
            'inputs': {name: binding(previous / (name + '_requests.json')) for name in ['development', 'validation']},
            'code': code(), 'new_physical_limit': 34, 'api_limit': 0,
            'baseline': {'legacy_native': '13/17', 'legacy_small_tabletop': '15/17'},
            'validation_scope': 'retained known sources, not fresh heldout assets',
            'promotion_requires_fresh_layout_asset_qualification': True,
            'physical_retries': 0})
    else:
        if read(protocol)['code'] != code(): raise ValueError('Frozen controller code changed')
    for value in read(protocol)['inputs'].values(): resolve(value)
    outputs = []
    for split, seed in [('development', 2026102708), ('validation', 2026102780)]:
        for contact in ['native_default', 'small_tabletop_v1']:
            campaign = root / ('controller_' + contact + '_' + split)
            if not campaign.exists():
                args = SimpleNamespace(root=campaign, budget=root / 'compute_budget',
                    requests=previous / (split + '_requests.json'), release_policy='legacy_v1',
                    motion_policy='cartesian_transfer_v1', contact_policy=contact,
                    contact_diagnostic=False, gpu=2, assets=1, layouts=1, physical_limit=1,
                    method='cudex', library=ROOT / 'assets/precomputed_grasps_v1', seed=seed)
                print({'init': str(campaign), 'result': initialize(args)}, flush=True)
            for case in read(campaign / 'cohort.json')['cases']:
                if (campaign / 'executions' / case['id'] / 'terminal.json').exists(): continue
                # Driver utilization can linger briefly after our own process exits.
                # Never stop or bypass a live external owner.
                for attempt in range(16):
                    snapshot = resources.gpu_snapshot(2)
                    try: resources.check_idle(snapshot); break
                    except RuntimeError:
                        if snapshot['compute_owners'] or attempt == 15: raise
                        time.sleep(1)
                execute(SimpleNamespace(root=campaign, case=[case['id']], limit=1))
            result = audit(SimpleNamespace(root=campaign))
            outputs.append({'split': split, 'contact': contact, 'result': result})
            print({'audit': str(campaign), 'passed': result['passed'], 'registered': result['registered']}, flush=True)
    target = root / 'controller_comparison.json'
    if not target.exists(): write(target, {'protocol': binding(protocol), 'rows': outputs, 'default_promoted': False})
    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'runs/batch500_20261027')
    run(parser.parse_args().root)
