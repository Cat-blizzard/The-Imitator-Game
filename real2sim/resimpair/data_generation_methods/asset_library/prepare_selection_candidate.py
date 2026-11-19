"""Prepare bounded local assets for the candidate selector; never admits training.

Reuses the established visual/collision baking implementation. Design scales are
explicit choices, not recovered measurements. Only GLB inputs are accepted here;
GSO OBJ materials need their own dependency resolver and are not silently lost.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image, ImageDraw

from ..human_to_robot_pipeline.household_assets import bind, prepare_asset
from ..r2h_pipeline.common import file_sha256, write_json

ROOT = Path('/mnt/sdg/datasets/open_assets/household_scaleup_20261020_v1')
# These category assignments are explicit developer choices. Each is accompanied
# by its original record and preview, rather than inferred from a substring match.
SELECTION = [
    ('calculator', 'objaverse', '8ba3cf9cf9d44e39a64f5b6f6a0b6a03', .12, 'flat_shortest_axis'),
    ('scissors', 'objaverse', '9b4ef190a04942c399bf8df394b59d76', .16, 'flat_shortest_axis'),
    ('hammer', 'objaverse', '22448dd0295e4591b0277fc6a770b254', .17, 'flat_shortest_axis'),
    ('flashlight', 'objaverse', '98b5663abe0c46ef855e3c3f64a0d7f9', .14, 'flat_shortest_axis'),
    ('paintbrush', 'objaverse', 'b0ca5dbd4b834662a50e1e475a643d75', .16, 'flat_shortest_axis'),
    ('hairbrush', 'objaverse', '77dbf9d035cc45f09fe771a72462f89a', .15, 'flat_shortest_axis'),
    ('headphones', 'objaverse', '10150b1eef39498cbcfc9185a40eb76a', .16, 'flat_shortest_axis'),
    ('candle_holder', 'abo', 'B07B8JXP8G', .14, 'upright_y'),
    ('picture_frame', 'abo', 'B075YLXSLP', .16, 'flat_shortest_axis'),
    ('clock', 'abo', 'B07B8PXMRT', .15, 'upright_y'),
]


def prepare(output: Path, root=ROOT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    wanted = {(s, uid) for _, s, uid, _, _ in SELECTION}
    records = {}
    for source in ('objaverse', 'abo'):
        for line in (root / source / 'catalog.jsonl').open():
            row = json.loads(line)
            uid = row.get('uid', row.get('source_id'))
            if (source, uid) in wanted:
                records[source, uid] = row
    assets, failures = [], []
    for category, source, uid, size, orientation in SELECTION:
        try:
            raw = dict(records[source, uid])
            # The old preparation API expects these attribution field names.
            raw['uid'] = uid
            raw['author'] = raw.get('author', raw.get('attribution'))
            if Path(raw['geometry_path']).suffix != '.glb':
                raise ValueError('Only self-contained GLB supported by this preparation route')
            a = prepare_asset(raw, category, size, orientation, output)
            a['source_asset_id'] = f'{source}:{uid}'
            a['candidate_version'] = 'task_asset_selection_candidate_v1'
            a['qualification'] = {
                'static_geometry_preflight': 'passed',
                'static_scene_execution': 'unknown_not_executed',
                'active_manipulation': 'unsupported',
                'rendered_semantics': 'unknown_not_rendered',
            }
            a['capabilities'] = {'static_untouched_obstacle': 'preflight_only'}
            a['semantic_categories'] = [category]
            a['category_assignment'] = 'explicit_developer_selection_from_source_name; CPU preview pending'
            write_json(Path(a['visual']).parent / 'qualification_asset.json', a)
            a['asset_spec'] = bind(Path(a['visual']).parent / 'qualification_asset.json')
            assets.append(a)
        except Exception as exc:
            failures.append({'source_asset_id': f'{source}:{uid}', 'category': category,
                             'error_type': type(exc).__name__, 'error': str(exc)})
    gallery = Image.new('RGB', (1200, max(1, (len(assets)+2)//3)*230), 'white')
    draw = ImageDraw.Draw(gallery)
    for i, a in enumerate(assets):
        preview = Image.open(a['preview']['path']).crop((0, 0, 320, 360))
        preview.thumbnail((200, 210))
        x, y = (i % 3)*400, (i//3)*230
        gallery.paste(preview, (x, y))
        draw.text((x+195, y+40), a['category'], fill='black')
        draw.text((x+195, y+65), a['source_asset_id'].split(':')[0], fill='black')
        draw.text((x+195, y+90), 'static geometry only', fill='black')
        draw.text((x+195, y+115), 'cm: '+','.join(f'{v*100:.1f}' for v in a['extents_m']), fill='black')
    gallery.save(output/'gallery.jpg')
    result = {'schema':'household_static_distractor_catalog_v1', 'assets':assets,
              'failed_candidates':failures, 'requested_count':len(SELECTION),
              'simulation_executed':False, 'training_admitted':False,
              'new_downloads':0, 'elapsed_seconds':time.monotonic()-started,
              'source_index':bind(root/'index/catalog.jsonl'),
              'selection':[list(x) for x in SELECTION],
              'limitations':['Static convex collision may fill cavities/handles.',
                             'Explicit design scale is not measured original physical scale.',
                             'Mesh loading and CPU previews do not establish simulator quality.']}
    write_json(output/'catalog.json', result)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args()
    r=prepare(args.output)
    print(json.dumps({'prepared':len(r['assets']), 'failures':r['failed_candidates'],
                      'elapsed_seconds':r['elapsed_seconds']}))
