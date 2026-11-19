"""Collect the pp2 pilot evidence (assets, scenes, physics, renders, references, cost) into one JSON."""
from __future__ import annotations

import argparse
from collections import Counter
import glob
import json
from pathlib import Path
import statistics

from resimpair_grasp.common import ROOT, binding, read


def cohort_stats(root):
    root = Path(root)
    cases = read(root / 'cohort.json')['cases']
    summary = read(root / 'acceptance/summary.json')
    seconds = [read(p)['elapsed_seconds'] for p in root.glob('executions/*/terminal.json')]
    receipts = [read(p) for p in root.glob('run_receipts/*.json')]
    failures = Counter()
    for p in root.glob('acceptance/cases/*.json'):
        r = read(p)
        if not r['passed']:
            failures[(r['checks'].get('process_completed', {}).get('failure') or ','.join(r['failures'])).split(':')[0]] += 1
    counterfactual = read(root / 'acceptance/counterfactual.json') if (root / 'acceptance/counterfactual.json').exists() else None
    factory = read(resolve_factory(root))
    return {'mode': read(root / 'cohort.json')['mode'], 'scenes_requested': len(factory['scenes']),
            'scenes_accepted': sum(1 for s in factory['scenes'] if s['tasks']),
            'scene_preflight_rejections': [s.get('rejected') for s in factory['scenes'] if not s['tasks']],
            'tasks_executed': summary['executed'], 'tasks_passed': summary['passed'], 'by_family': summary['by_family'],
            'failure_reasons': dict(failures), 'worker_seconds_median': statistics.median(seconds),
            'worker_seconds_p90': sorted(seconds)[int(.9 * (len(seconds) - 1))],
            'wall_seconds': sum(r['wall_seconds'] for r in receipts), 'workers': [r['workers'] for r in receipts],
            'counterfactual_pairs': counterfactual and counterfactual['pairs'],
            'counterfactual_false_passes': counterfactual and len(counterfactual['false_passes'])}


def resolve_factory(cohort):
    ref = read(Path(cohort) / 'cohort.json')['factory']
    path = Path(ref['path']); return path if path.is_absolute() else ROOT / path


def reference_stats(roots):
    rows, usage, spend = [], Counter(), 0
    for root in map(Path, roots):
        for marker in root.glob('campaign_ledger/stages/*/call_started.json'):
            stage = marker.parent; usage[read(marker)['kind']] += 1
            for name in ('credit_terminal_interval.json', 'credit_response_interval.json'):
                if (stage / name).exists():
                    spend += read(stage / name).get('observed_token_quota_delta') or 0; break
        rows += [read(p) for p in root.glob('cases/*/result.json')]
    def rate(key, subset):
        known = [r[key]['correct'] for r in subset if key in r and r[key].get('correct') is not None]
        return {'correct': sum(known), 'known': len(known), 'unknown': sum(1 for r in subset if key in r and r[key].get('correct') is None)}
    unordered = [r for r in rows if not r['program']['ordered']]
    ordered = [r for r in rows if r['program']['ordered']]
    return {'cases': len(rows), 'calls_started': dict(usage), 'observed_quota': spend, 'nominal_usd': spend / 500000,
            'endpoint_identification': {'all': rate('endpoints', rows), 'unordered': rate('endpoints', unordered),
                                        'ordered': rate('endpoints', ordered)},
            'robot_video_identification': {'all': rate('robot', rows), 'ordered': rate('robot', ordered)},
            'rows': [{'case': r['case'], 'instruction': r['instruction'], 'ordered': r['program']['ordered'],
                      'endpoints': r.get('endpoints', {}).get('chosen_instruction'), 'endpoints_correct': r.get('endpoints', {}).get('correct'),
                      'robot': r.get('robot', {}).get('chosen_instruction'), 'robot_correct': r.get('robot', {}).get('correct'),
                      'unknown': [k for k in ('endpoints', 'robot') if r.get(k, {}).get('unknown')]} for r in rows]}


def build(output):
    catalog = read(ROOT / 'runs/pp2_catalog_20261101_v2/catalog.json')
    assets = catalog['assets']
    evidence = {
        'catalog': {'curated_manipulables': len(read(ROOT / 'configs/pp2/canonical_assets_v1.json')['manipulables']),
                    'usable_manipulables': Counter(a['partition'] for a in assets if a['role'] == 'manipulable'),
                    'usable_receptacles': Counter(a['partition'] for a in assets if a['role'] == 'receptacle'),
                    'categories': sorted({a['category'] for a in assets}),
                    'derived_surface_admissions': sum(1 for a in assets if a['role'] == 'manipulable' and a['admission']['derived_descriptor']),
                    'excluded': [{'source_id': e['source_id'], 'reason': e.get('rejected') or 'no_grasp_after_native_checks'} for e in catalog['excluded']]},
        'slice': cohort_stats(ROOT / 'runs/pp2_slice_20261101_r2/cohort'),
        'production_b1_before_fixes': cohort_stats(ROOT / 'runs/pp2_production_20261101_b1/cohort'),
        'production_b2_trivial_and_slot_fixes': cohort_stats(ROOT / 'runs/pp2_production_20261101_b2/cohort'),
        'production_b3_all_fixes': cohort_stats(ROOT / 'runs/pp2_production_20261101_b3/cohort'),
        'evaluation_heldout': cohort_stats(ROOT / 'runs/pp2_eval_20261101_e1/cohort'),
        'renders': {'videos': len(glob.glob(str(ROOT / 'runs/pp2_slice_20261101_r2/renders/*/render_result.json'))),
                    'seconds_median': statistics.median(read(p)['seconds'] for p in (ROOT / 'runs/pp2_slice_20261101_r2/renders').glob('*/render_result.json')),
                    'rooms': sorted({read(p)['room'] for p in (ROOT / 'runs/pp2_slice_20261101_r2/renders').glob('*/render_result.json')})},
        'references': reference_stats([ROOT / f'runs/pp2_reference_pilot_20261101{s}' for s in ('', '_r2', '_r3')]),
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(evidence, indent=1, ensure_ascii=False, default=dict))
    return evidence


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); e = build(a.output)
    print(json.dumps({k: v for k, v in e.items() if k != 'references'} | {'references': {k: v for k, v in e['references'].items() if k != 'rows'}},
                     indent=1, ensure_ascii=False, default=dict)[:6000])
