"""Ray-tracing audit of catalog visuals: which assets turn dark under ray tracing, and does flipping fix them.

Geometry alone cannot decide a repair: some inside-out parts (lids, liners) already ray-trace
correctly and break when flipped. So every asset is rendered alone, framed by its bounds:
rasterized reference, ray traced, and ray traced after each `lighting.repair_entity` mode (flip
inside-out closed parts; flip everything, for open single-layer meshes). Object brightness is
normalized by the floor in the same image. An asset is repaired at render time only when ray
tracing darkens it and a repair restores it; assets that stay dark are dropped from catalog v10.

Workers append one JSON line per asset and exit after BATCH assets; `run` relaunches them until
their shard is done, and kills a worker that makes no progress for STALL_S (the 3,020-asset v12
audit deadlocked all four workers after ~15 min with nothing written). An asset that stalls
twice is recorded as an error row.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np

from resimpair_grasp.common import ROOT, binding, read, resolve, write

VERSION = 'pp2_rt_audit_v1'
DARK = .5        # ray traced / rasterized relative brightness below this is "turned dark"
RESTORED = .75   # ... and the repair must bring it back above this
BATCH = 80       # assets per worker process (a fresh renderer per batch)
STALL_S = 180    # no new row for this long: kill and relaunch the worker


def visuals(catalog_path):
    """source_id -> visual GLB for every catalog asset, distractor-only rows included."""
    catalog = read(catalog_path); rows = {}
    for asset in catalog['assets'] + catalog.get('distractor_only', []):
        visual = read(resolve(asset['descriptor']))['visual']
        rows[asset['source_id']] = str(resolve(visual) if isinstance(visual, dict) else ROOT / visual)
    return rows


def shot(glb, shader, repair=None):
    import sapien
    from scipy.spatial.transform import Rotation
    from .lighting import repair_entity
    sapien.render.set_camera_shader_dir('rt' if shader == 'rt' else 'default')
    s = sapien.Scene(); s.set_ambient_light([.3, .3, .3]); s.add_directional_light([.3, .2, -1], [1.5, 1.5, 1.5], shadow=True)
    s.add_area_light_for_ray_tracing(sapien.Pose([0, 0, 1.2], [0.7071068, 0, 0.7071068, 0]), [6, 6, 6], .6, .6)
    s.add_area_light_for_ray_tracing(sapien.Pose([-1.2, 0, .6], Rotation.from_euler('y', -20, degrees=True).as_quat(scalar_first=True)),
                                     [2, 2, 2], .6, .6)
    b = s.create_actor_builder()
    b.add_box_visual(sapien.Pose([0, 0, -.01]), [2, 2, .01], sapien.render.RenderMaterial(base_color=[.6, .6, .6, 1])); floor = b.build_static()
    b = s.create_actor_builder(); b.add_visual_from_file(glb); a = b.build_kinematic()
    share = repair_entity(a, repair) if repair else 0.
    low, high = np.asarray(a.find_component_by_type(sapien.render.RenderBodyComponent).compute_global_aabb_tight())
    centre, size = (low + high) / 2, float(np.max(high - low))
    c = s.add_camera('c', 256, 256, .8, .01, 10)
    c.set_entity_pose(sapien.Pose(centre + np.array([-1.6, 0, 1.1]) * size,
                                  Rotation.from_euler('y', np.arctan2(1.1, 1.6)).as_quat(scalar_first=True)))
    if shader == 'rt':
        c.set_property('toneMapper', 2); c.set_property('exposure', 1.)
    s.update_render(); c.take_picture()
    rgb = np.clip(c.get_picture('Color')[..., :3], 0, 1); ids = c.get_picture('Segmentation')[..., 1]
    luminance = rgb.mean(-1); obj = ids == a.per_scene_id; ground = ids == floor.per_scene_id
    if obj.sum() < 30 or ground.sum() < 30:
        return {'pixels': int(obj.sum()), 'relative': None, 'repaired_share': share}
    return {'pixels': int(obj.sum()), 'relative': float(luminance[obj].mean() / luminance[ground].mean()), 'repaired_share': share}


def done_rows(path):
    rows = []
    for line in (Path(path).read_text().splitlines() if Path(path).exists() else []):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:  # a worker killed mid-write; that asset is simply redone
            pass
    return rows


def worker(items, output, limit=BATCH):
    from resimpair_runtime import configure
    configure()
    import sapien
    sapien.render.set_ray_tracing_samples_per_pixel(4); sapien.render.set_ray_tracing_path_depth(3)
    sapien.render.set_ray_tracing_denoiser('optix')
    seen = {r['source_id'] for r in done_rows(output)}
    todo = [(source_id, glb) for source_id, glb in read(items) if source_id not in seen][:limit]
    with open(output, 'a') as out:
        for source_id, glb in todo:
            try:
                raster, traced = shot(glb, 'default'), shot(glb, 'rt')
                row = {'source_id': source_id, 'visual': glb, 'raster': raster, 'rt': traced}
                if raster['relative'] and traced['relative'] is not None and traced['relative'] / raster['relative'] < DARK:
                    for mode in ('closed', 'all'):
                        fixed = shot(glb, 'rt', mode)
                        row['rt_' + mode] = fixed if fixed['repaired_share'] else None
            except Exception as error:
                row = {'source_id': source_id, 'visual': glb, 'error': repr(error)[:200]}
            out.write(json.dumps(row) + '\n'); out.flush()


def shard(chunk, rows, log, gpu):
    """Relaunch BATCH-sized workers on one GPU until every item of the shard has a row."""
    items = read(chunk); stalls = {}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(ROOT))
    while True:
        seen = {r['source_id'] for r in done_rows(rows)}
        todo = [i for i in items if i[0] not in seen]
        if not todo:
            return
        if stalls.get(todo[0][0], 0) >= 2:
            with open(rows, 'a') as out:
                out.write(json.dumps({'source_id': todo[0][0], 'visual': todo[0][1], 'error': 'stalled_twice'}) + '\n')
            continue
        proc = subprocess.Popen([sys.executable, '-m', 'resimpair_pp2.rt_audit', 'worker', '--items', str(chunk),
                                 '--output', str(rows)], cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=open(log, 'a'))
        count, last = len(seen), time.time()
        while proc.poll() is None:
            time.sleep(5)
            now = len(done_rows(rows))
            if now != count:
                count, last = now, time.time()
            elif time.time() - last > STALL_S:
                proc.kill(); proc.wait()
                head = next(i for i in items if i[0] not in {r['source_id'] for r in done_rows(rows)})
                stalls[head[0]] = stalls.get(head[0], 0) + 1


def decide(row):
    if 'error' in row or not row['raster']['relative'] or row['rt']['relative'] is None:
        return 'unmeasured'
    if row['rt']['relative'] / row['raster']['relative'] >= DARK:
        return 'ok'
    restored = {mode: row[f'rt_{mode}']['relative'] / row['raster']['relative'] for mode in ('closed', 'all') if row.get(f'rt_{mode}')}
    best = max(restored, key=restored.get, default=None)
    return 'repair_' + best if best and restored[best] >= RESTORED else 'dark'


def run(catalog_path, output, gpus):
    output = Path(output).resolve(); (output / 'shards').mkdir(parents=True, exist_ok=True)
    items = sorted(visuals(catalog_path).items()); threads = []
    for k, gpu in enumerate(gpus):
        chunk = output / 'shards' / f'items_{k}.json'
        if not chunk.exists():
            write(chunk, items[k::len(gpus)])
        threads.append(threading.Thread(target=shard, args=(chunk, output / 'shards' / f'rows_{k}.jsonl',
                                                             output / 'shards' / f'log_{k}.txt', gpu)))
        threads[-1].start()
    for t in threads:
        t.join()
    rows = [r for k in range(len(gpus)) for r in done_rows(output / 'shards' / f'rows_{k}.jsonl')]
    for r in rows:
        r['decision'] = decide(r)
    result = {'schema': VERSION, 'catalog': binding(Path(catalog_path)), 'thresholds': {'dark': DARK, 'restored': RESTORED},
              'counts': {d: sum(r['decision'] == d for r in rows) for d in ('ok', 'repair_closed', 'repair_all', 'dark', 'unmeasured')},
              'repair': sorted(r['source_id'] + ' ' + r['decision'] for r in rows if r['decision'].startswith('repair_')),
              'dark': sorted(r['source_id'] for r in rows if r['decision'] == 'dark'), 'rows': rows}
    write(output / 'audit.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    w = sub.add_parser('worker'); w.add_argument('--items', type=Path, required=True); w.add_argument('--output', type=Path, required=True)
    r = sub.add_parser('run'); r.add_argument('--catalog', type=Path, required=True); r.add_argument('--output', type=Path, required=True)
    r.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3])
    a = p.parse_args()
    if a.command == 'worker':
        worker(a.items, a.output)
    else:
        result = run(a.catalog, a.output, a.gpus)
        print(json.dumps({k: result[k] for k in ('counts', 'repair', 'dark')}, indent=1))
