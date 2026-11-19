"""One command for the whole pp2 chain, resumable.

    .venv/bin/python -B -m resimpair_pp2.pipeline --root runs/<name> [--config config.json]

The run directory holds config.json (written on first start, then authoritative) and one directory per
stage. Each stage writes stages/<name>.json when it finishes and is skipped on rerun, so an interrupted
run resumes where it stopped (an execution that was in flight is recorded as interrupted, never rerun).

  factory   scenes.build (parallel waves): scenes and their task sets    -> factory/
  execute   execute.prepare + run: one physical attempt per task         -> cohort/executions/
  accept    accept.evaluate_cohort + counterfactual matrix               -> cohort/acceptance/
  render    ray-traced videos of accepted executions x room variants     -> renders/<case>/v<k>/
  dataset   dataset.index: meta + trajectory per accepted execution      -> dataset/
  groups.*  reference groups (config groups.count > 0): the same chain under groups/, then
            groups/manifest.json maps each program's human-video source to its robot samples
  groups.human_video  (config human_video.budget_cny > 0) seedance.run: robot-free first/last frames ->
            Seedance video per program, ledger-capped; registered 'generated' until reviewed -> groups/human_videos/
            (rerun `python -m resimpair_pp2.seedance run --manifest ...` to continue past the budget)
  report    yields per stage, family, asset and reference program        -> report.json

Reference groups are the "one human video, hundreds of robot samples" structure: a fully executed
source scene is the human-video source (layout 0); groups.layouts re-laid-out copies (new positions,
yaws, distractors and robot start; same objects and programs) x groups.room_variants renders are robot
samples of the same programs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from resimpair_grasp.common import ROOT, binding, read, write

VERSION = 'pp2_pipeline_v1'
DEFAULTS = {
    'catalog': 'runs/pp2_catalog_20261102_v11/catalog.json', 'partition': 'development', 'scenes': 24,
    'seed': 20262601, 'factors': True, 'multi_quota': 6, 'k': 6, 'factory_workers': 16,
    'mode': 'production', 'workers': 48, 'worker_module': 'resimpair_pp2.worker', 'gpus': [0, 1, 2, 3], 'render_per_gpu': 2, 'render_stride': 1,
    'room_variants': 1, 'min_free_gpu_mib': 12 * 1024,
    'groups': {'count': 0, 'layouts': 20, 'min_accepted_programs': 4, 'room_variants': 2, 'seed': 20262701},
    'human_video': {'model': '2.0', 'resolution': '720p', 'budget_cny': 0, 'gpu': 0},
}


def config(root, override=None):
    path = Path(root) / 'config.json'
    if path.exists():
        cfg = read(path)
        if override and {**cfg, **override} != cfg:
            raise ValueError('config_differs_from_the_runs_recorded_config')
        return cfg
    cfg = json.loads(json.dumps(DEFAULTS)); cfg.update(override or {})
    cfg['groups'] = {**DEFAULTS['groups'], **(override or {}).get('groups', {})}
    cfg['human_video'] = {**DEFAULTS['human_video'], **(override or {}).get('human_video', {})}
    git = lambda *a: subprocess.run(['git', *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    code = {'commit': git('rev-parse', 'HEAD'), 'uncommitted_changes': git('status', '--porcelain', '--', 'resimpair_pp2')}
    write(path, {**cfg, 'schema': VERSION, 'created_unix': time.time(), 'code': code})
    return read(path)


def stage(root, name, function):
    done = Path(root) / 'stages' / f'{name}.json'
    if done.exists():
        return read(done)['result']
    started = time.monotonic(); print(json.dumps({'stage': name, 'status': 'started'}), flush=True)
    result = function()
    write(done, {'stage': name, 'result': result, 'seconds': time.monotonic() - started, 'finished_unix': time.time()})
    print(json.dumps({'stage': name, 'status': 'done', 'seconds': round(time.monotonic() - started, 1)}), flush=True)
    return result


def close_interrupted(cohort):
    """Executions reserved but never finished (the run was killed): record them, never rerun them."""
    rows = []
    for case in read(Path(cohort) / 'cohort.json')['cases']:
        directory = Path(cohort) / 'executions' / case['id']
        if (directory / 'reservation.json').exists() and not (directory / 'terminal.json').exists():
            write(directory / 'terminal.json', {'case': case['id'], 'physical_attempts': 1, 'returncode': 'interrupted',
                  'admission': {'stop_admission': True, 'reasons': ['run_interrupted']}, 'schema': VERSION + '_interrupted'})
            rows.append(case['id'])
    return rows


def execute_stage(sub, cfg):
    from . import execute
    cohort = sub / 'cohort'
    if not (cohort / 'cohort.json').exists():
        factory = read(sub / 'factory' / 'factory.json')
        tasks = sum(r['tasks'] for r in factory['scenes'])
        execute.prepare(sub / 'factory', cohort, cfg['mode'], tasks, worker_module=cfg['worker_module'])
    interrupted = close_interrupted(cohort)
    result = execute.run(cohort, cfg['workers'])
    return {**result, 'interrupted_recorded': interrupted}


def accept_stage(sub):
    from . import accept
    summary = accept.evaluate_cohort(sub / 'cohort')
    matrix = accept.counterfactual_matrix(sub / 'cohort')
    write(sub / 'cohort' / 'acceptance' / 'counterfactual.json', matrix)
    return {**{k: summary[k] for k in ('executed', 'passed', 'by_family', 'failure_counts')},
            'counterfactual_pairs': matrix['pairs'], 'counterfactual_false_passes': len(matrix['false_passes'])}


def accepted(cohort):
    return [c['id'] for c in read(Path(cohort) / 'cohort.json')['cases']
            if (Path(cohort) / 'acceptance/cases' / (c['id'] + '.json')).exists()
            and read(Path(cohort) / 'acceptance/cases' / (c['id'] + '.json'))['passed']]


def render_stage(sub, cfg, variants, passes=3):
    """Render every accepted execution x variant; jobs that fail or time out are retried in the next pass
    (finished renders are skipped), and whatever is still missing after the last pass is recorded."""
    from . import render
    for attempt in range(passes):
        rows = render.run(sub / 'cohort', sub / 'renders', accepted(sub / 'cohort'), cfg['gpus'], cfg['render_stride'],
                          cfg['render_per_gpu'], variants=variants, min_free_mib=cfg['min_free_gpu_mib'])
        missing = [r for r in rows if 'skipped' in r or 'failed' in r]
        if not missing:
            break
    return {'renders': len(rows) - len(missing), 'missing': missing, 'passes': attempt + 1,
            'render_seconds': sum(r.get('seconds', 0) for r in rows)}


def chain(root, sub, cfg, variants, prefix=''):
    stage(root, prefix + 'execute', lambda: execute_stage(sub, cfg))
    stage(root, prefix + 'accept', lambda: accept_stage(sub))
    stage(root, prefix + 'render', lambda: render_stage(sub, cfg, variants))
    from . import dataset
    return stage(root, prefix + 'dataset', lambda: dataset.index(sub / 'cohort', sub / 'renders', sub / 'dataset'))


def group_sources(root, cfg):
    """Source scenes for reference groups: most accepted programs first (at least groups.min_accepted_programs)."""
    cohort = Path(root) / 'cohort'; per = Counter(c.rsplit('__', 1)[0] for c in accepted(cohort))
    factory = {Path(r['directory']).name: Path(r['directory']) for r in read(Path(root) / 'factory/factory.json')['scenes'] if r['tasks']}
    ranked = sorted((s for s, n in per.items() if n >= cfg['groups']['min_accepted_programs']), key=lambda s: (-per[s], s))
    return [str(factory[s]) for s in ranked[:cfg['groups']['count']]]


def report(root, cfg):
    root = Path(root); out = {'schema': VERSION + '_report', 'config': cfg, 'stages': {}}
    for p in sorted((root / 'stages').glob('*.json')):
        r = read(p); result = r['result'] if isinstance(r['result'], dict) else {'items': len(r['result'])}
        out['stages'][p.stem] = {'seconds': round(r['seconds'], 1),
                                 **{k: v for k, v in result.items() if not isinstance(v, (list, dict)) or k == 'by_family'}}
    for sub, name in ((root, 'main'), (root / 'groups', 'groups')):
        if not (sub / 'factory/factory.json').exists():
            continue
        factory = read(sub / 'factory/factory.json')['scenes']
        rows = {'scenes': len(factory), 'scenes_with_tasks': sum(bool(r['tasks']) for r in factory),
                'tasks': sum(r['tasks'] for r in factory),
                'rejections': dict(Counter((r.get('rejected') or '')[:48] for r in factory if not r['tasks']))}
        if (sub / 'cohort/acceptance/cases').exists():
            verdicts = [read(p) for p in (sub / 'cohort/acceptance/cases').glob('*.json')]
            fam = defaultdict(lambda: [0, 0])
            for v in verdicts:
                fam[v['family']][0] += v['passed']; fam[v['family']][1] += 1
            rows['accepted'] = sum(v['passed'] for v in verdicts); rows['executed'] = len(verdicts)
            rows['by_family'] = {f: f'{a}/{b}' for f, (a, b) in sorted(fam.items())}
            rows['failures'] = dict(Counter(f for v in verdicts for f in v['failures']).most_common(12))
        out[name] = rows
    if (root / 'groups/manifest.json').exists():
        m = read(root / 'groups/manifest.json')
        per = [len(p['robot_samples']) for v in m['groups'].values() for p in v]
        out['reference'] = {'groups': len(m['groups']), 'programs': m['programs'], 'robot_samples_per_program': per,
                            'robot_sample_renders': m['robot_sample_renders']}
    write(root / 'report.json', out)
    return {k: out[k] for k in ('main', 'groups', 'reference') if k in out}


def run(root, override=None):
    from . import scenes, groups, dataset
    root = Path(root).resolve(); cfg = config(root, override)
    stage(root, 'factory', lambda: scenes.build(root / 'factory', cfg['scenes'], cfg['partition'], cfg['seed'], k=cfg['k'],
                                                 catalog_path=ROOT / cfg['catalog'], factors=cfg['factors'],
                                                 multi_quota=cfg['multi_quota'], workers=cfg['factory_workers']))
    chain(root, root, cfg, cfg['room_variants'])
    g = cfg['groups']
    if g['count']:
        sources = stage(root, 'groups.select', lambda: group_sources(root, cfg))
        stage(root, 'groups.factory', lambda: groups.build(sources, g['layouts'], g['seed'], root / 'groups/factory',
                                                           ROOT / cfg['catalog'], workers=cfg['factory_workers']))
        chain(root, root / 'groups', cfg, g['room_variants'], prefix='groups.')
        stage(root, 'groups.manifest', lambda: dataset.group_manifest(root / 'dataset', root / 'groups/dataset',
                                                                      root / 'groups/manifest.json'))
        hv = {**DEFAULTS['human_video'], **cfg.get('human_video', {})}
        if hv['budget_cny'] > 0:
            from . import seedance
            stage(root, 'groups.human_video', lambda: seedance.run(root / 'groups/manifest.json', hv['budget_cny'], hv['model'],
                                                                   hv['resolution'], gpu=hv['gpu']))
    return report(root, cfg)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--config', type=Path)
    a = p.parse_args()
    print(json.dumps(run(a.root, read(a.config) if a.config else None), indent=1, ensure_ascii=False))
