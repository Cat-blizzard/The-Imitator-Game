"""Local HTML gallery for a pp2 cohort: per scene, every task with its robot videos and acceptance."""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path

from resimpair_grasp.common import read


def gallery(cohort, renders, output, references=()):
    cohort, renders, output = Path(cohort).resolve(), Path(renders).resolve(), Path(output).resolve()
    rel = lambda p: html.escape(os.path.relpath(p, output.parent), quote=True)
    accepted = {p.stem: read(p) for p in (cohort / 'acceptance/cases').glob('*.json')}
    human = {}
    for root in references:
        for result in Path(root).glob('cases/*/result.json'):
            human[result.parent.name] = (result.parent, read(result))
    scenes = {}
    for case in read(cohort / 'cohort.json')['cases']:
        scenes.setdefault(case['scene_id'], []).append(case)
    parts = ['<!doctype html><meta charset="utf-8"><title>pp2 cohort</title><style>'
             'body{font-family:sans-serif;margin:16px;background:#fafafa}h2{margin-top:32px}'
             '.row{display:flex;gap:12px;flex-wrap:wrap}.card{background:#fff;border:1px solid #ddd;padding:8px;width:540px}'
             'video{width:260px}.ok{color:#176c2f}.bad{color:#a11}</style>',
             f'<h1>{html.escape(cohort.parent.name)}</h1>']
    for scene_id, cases in scenes.items():
        parts.append(f'<h2>{html.escape(scene_id)}</h2><div class="row">')
        for case in cases:
            verdict = accepted.get(case['id'], {})
            passed = verdict.get('passed')
            parts.append(f'<div class="card"><b>{html.escape(case["instruction"])}</b><br>'
                         f'<span>{html.escape(case["pp2_family"])}</span> · '
                         f'<span class="{"ok" if passed else "bad"}">{"accepted" if passed else "failed: " + html.escape(",".join(verdict.get("failures", ["not run"])))}</span><br>')
            for camera in ('robot_side', 'opposite_high'):
                video = renders / case['id'] / f'{camera}.mp4'
                if video.exists():
                    parts.append(f'<video src="{rel(video)}" controls muted loop></video>')
            if case['id'] in human:
                folder, result = human[case['id']]
                if (folder / 'human.mp4').exists():
                    h = result.get('human', {})
                    parts.append(f'<br>human reference: <video src="{rel(folder / "human.mp4")}" controls muted loop></video>'
                                 f'<br>blind id: {html.escape(str(h.get("chosen_instruction")))} '
                                 f'<span class="{"ok" if h.get("correct") else "bad"}">({"correct" if h.get("correct") else "wrong"})</span>')
            parts.append('</div>')
        parts.append('</div>')
    output.write_text('\n'.join(parts))
    return str(output)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort', type=Path, required=True); p.add_argument('--renders', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--references', type=Path, nargs='*', default=[])
    a = p.parse_args(); print(gallery(a.cohort, a.renders, a.output, a.references))
