"""6x6 contact sheet of accepted pp2 samples: initial | final frame per cell, cropped to the workspace."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np

from resimpair_grasp.common import ROOT, read, resolve
from .render import SHADER, room_seed


def select(cohorts, count, seed=0, multi_share=0.):
    """Round-robin over task families, at most one sample per scene, accepted executions only."""
    accepted = []
    for cohort in cohorts:
        for p in (Path(cohort) / 'acceptance/cases').glob('*.json'):
            r = read(p)
            if r['passed']:
                accepted.append({**r, 'cohort': str(Path(cohort).resolve())})
    by_family = defaultdict(list)
    rng = np.random.default_rng(seed)
    for r in sorted(accepted, key=lambda r: r['case']):
        by_family[r['family']].append(r)
    for rows in by_family.values():
        rng.shuffle(rows)
    picked, scenes = [], set()
    multi = [f for f in by_family if f.startswith('multi')]
    while multi and sum(r['family'].startswith('multi') for r in picked) < int(round(multi_share * count)) \
            and any(by_family[f] for f in multi):
        for family in multi:
            rows = by_family[family]
            while rows and rows[0]['case'].split('__')[0] in scenes:
                rows.pop(0)
            if rows and sum(r['family'].startswith('multi') for r in picked) < int(round(multi_share * count)):
                r = rows.pop(0); picked.append(r); scenes.add(r['case'].split('__')[0])
    while len(picked) < count and any(by_family.values()):
        for family in sorted(by_family, key=lambda f: len(by_family[f])):
            rows = by_family[family]
            while rows and rows[0]['case'].split('__')[0] in scenes:
                rows.pop(0)
            if rows and len(picked) < count:
                r = rows.pop(0); picked.append(r); scenes.add(r['case'].split('__')[0])
    return picked


def render_keyframes(execution, target, seed, gpu, camera_mode, scale=2, shader=SHADER):
    if not (Path(target) / 'keyframes.json').exists():
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(ROOT), OMP_NUM_THREADS='2')
        subprocess.run([sys.executable, '-m', 'resimpair_pp2.render', 'keyframes', '--execution', str(execution),
                        '--output', str(target), '--seed', str(seed), '--camera-mode', camera_mode, '--scale', str(scale),
                        '--shader', shader],
                       cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return read(Path(target) / 'keyframes.json')


def contrast(info, which):
    """Minimum luminance contrast (0-255) between each visible object and a ring around it."""
    from PIL import Image
    from scipy.ndimage import binary_dilation
    image = np.asarray(Image.open(resolve(info['images'][which])).convert('L'), float)
    masks = np.load(resolve(info['images']['object_masks']))[which]
    values = []
    for mask in masks:
        if mask.sum() < 20:
            continue
        ring = binary_dilation(mask, iterations=6) & ~mask
        values.append(abs(image[mask].mean() - image[ring].mean()))
    return min(values) if values else 0.


def choose_room(execution, output, gpu, camera_mode, tries=5, threshold=15., shader=SHADER):
    """Visual-only: try room seeds for one scene and keep the first whose table makes every object stand out."""
    scene_id = Path(execution).name.rsplit('__', 1)[0]
    record = Path(output) / 'room_choice' / f'{scene_id}.json'
    if record.exists():
        return read(record)['seed']
    rows = []
    for k in range(tries):
        seed = room_seed(scene_id) + 1000 * k
        info = render_keyframes(execution, Path(output) / 'room_choice' / scene_id / str(k), seed, gpu, camera_mode, 1, shader)
        rows.append({'seed': seed, 'room': info['room'], 'initial_min_contrast': contrast(info, 'initial')})
        if rows[-1]['initial_min_contrast'] >= threshold:
            break
    best = max(rows, key=lambda r: r['initial_min_contrast'])
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({'scene_id': scene_id, 'tries': rows, 'seed': best['seed'], 'threshold': threshold}, indent=1))
    return best['seed']


def keyframes(cohort, output, cases, gpus, camera_mode='bounded', choose=True, shader=SHADER):
    cohort, output = Path(cohort).resolve(), Path(output).resolve()
    def one(item):
        i, case = item; gpu = gpus[i % len(gpus)]; execution = cohort / 'executions' / case
        seed = choose_room(execution, output, gpu, camera_mode, shader=shader) if choose else room_seed(case.rsplit('__', 1)[0])
        return render_keyframes(execution, output / 'keyframes' / case, seed, gpu, camera_mode, shader=shader)
    with ThreadPoolExecutor(len(gpus) * 2) as pool:
        return dict(zip(cases, pool.map(one, enumerate(cases))))


def crop_box(masks, shape, margin=150, aspect=4 / 3):
    union = masks['initial'].any(0) | masks['final'].any(0)
    ys, xs = np.nonzero(union)
    y0, y1, x0, x1 = ys.min() - margin, ys.max() + margin, xs.min() - margin, xs.max() + margin
    h, w = y1 - y0, x1 - x0
    if w / h < aspect:
        grow = (h * aspect - w) / 2; x0, x1 = x0 - grow, x1 + grow
    else:
        grow = (w / aspect - h) / 2; y0, y1 = y0 - grow, y1 + grow
    H, W = shape
    dx = max(0, -x0) - max(0, x1 - W); dy = max(0, -y0) - max(0, y1 - H)
    return [int(max(0, x0 + dx)), int(max(0, y0 + dy)), int(min(W, x1 + dx)), int(min(H, y1 + dy))]


def sheet(frames, picked, output, cols=6, tile=(360, 270)):
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 15); bold = ImageFont.truetype('DejaVuSans-Bold.ttf', 15)
    except OSError:
        font = bold = ImageFont.load_default()
    caption_h = 62; cell_w, cell_h = 2 * tile[0] + 6, tile[1] + caption_h
    rows = (len(picked) + cols - 1) // cols
    canvas = Image.new('RGB', (cols * cell_w + (cols + 1) * 10, rows * cell_h + (rows + 1) * 10), (250, 250, 248))
    draw = ImageDraw.Draw(canvas); meta = []
    for k, r in enumerate(picked):
        info = frames[r['case']]; images = info['images']
        a = Image.open(resolve(images['initial'])).convert('RGB'); b = Image.open(resolve(images['final'])).convert('RGB')
        box = crop_box(np.load(resolve(images['object_masks'])), (a.height, a.width))
        x = 10 + (k % cols) * (cell_w + 10); y = 10 + (k // cols) * (cell_h + 10)
        canvas.paste(a.crop(box).resize(tile, Image.LANCZOS), (x, y))
        canvas.paste(b.crop(box).resize(tile, Image.LANCZOS), (x + tile[0] + 6, y))
        task = read(Path(r['cohort']) / 'executions' / r['case'] / 'effective_task.json')
        draw.text((x + 4, y + 4), 'start', fill=(255, 255, 255), font=bold)
        draw.text((x + tile[0] + 10, y + 4), 'end', fill=(255, 255, 255), font=bold)
        lines = textwrap.wrap(f"{k + 1}. {task['instruction']}", 88)[:2]
        draw.text((x, y + tile[1] + 4), '\n'.join(lines), fill=(20, 20, 20), font=font)
        draw.text((x, y + tile[1] + 42), f"{r['family']} | {info['room']}", fill=(110, 110, 110), font=font)
        meta.append({'cell': k + 1, 'cohort': r['cohort'], 'case': r['case'], 'instruction': task['instruction'],
                     'family': r['family'], 'room': info['room'], 'crop': box,
                     'contrast_initial': contrast(info, 'initial'), 'contrast_final': contrast(info, 'final')})
    canvas.save(output)
    Path(output).with_suffix('.json').write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    return meta


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohorts', type=Path, nargs='+', required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpus', type=int, nargs='+', default=[2, 3]); p.add_argument('--count', type=int, default=36)
    p.add_argument('--multi-share', type=float, default=0.); p.add_argument('--shader', default=SHADER)
    a = p.parse_args()
    picked = select(a.cohorts, a.count, multi_share=a.multi_share)
    frames = {}
    for cohort in {r['cohort'] for r in picked}:
        frames.update(keyframes(cohort, a.output.parent, [r['case'] for r in picked if r['cohort'] == cohort], a.gpus,
                                shader=a.shader))
    meta = sheet(frames, picked, a.output)
    print(json.dumps({'cells': len(meta), 'families': Counter(m['family'] for m in meta),
                      'rooms': len({m['room'] for m in meta})}, indent=1))
