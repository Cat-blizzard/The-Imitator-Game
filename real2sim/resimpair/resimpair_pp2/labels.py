"""Batched semantic labelling of asset thumbnails: everyday noun, color, size plausibility.

The reviewer sees each thumbnail with its metric dimensions. Labels are
candidate metadata for task descriptors; a malformed batch stays unknown and
its assets are simply not admitted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from resimpair_grasp.common import ROOT, read, resolve
from corl2026_workshop.experiments.autonomous_pipeline_v4.core import bind, write
from .reference import reserve_budget

BATCH = 24
SCHEMA = {'type': 'object', 'properties': {'items': {'type': 'array', 'items': {'type': 'object', 'properties': {
    'index': {'type': 'integer'},
    'noun': {'type': 'string'},
    'color': {'type': 'string', 'enum': ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown',
                                          'white', 'gray', 'black', 'beige', 'silver', 'gold', 'multicolor']},
    'recognizable': {'type': 'boolean'},
    'plausible_real_size': {'type': 'boolean'},
    'everyday_tabletop_item': {'type': 'boolean'},
    'single_clean_object': {'type': 'boolean'},
    'open_container': {'type': 'boolean'}},
    'required': ['index', 'noun', 'color', 'recognizable', 'plausible_real_size', 'everyday_tabletop_item',
                 'single_clean_object', 'open_container']}}}, 'required': ['items']}

PROMPT = ('Each image shows one 3D asset on a plain gray floor; its true size is given in the label. For every image '
          'return: noun = the everyday English name a person would use for it (1-3 words, lowercase, e.g. "mug", '
          '"banana", "screwdriver", "toy car", "bowl"); color = its dominant color; recognizable = a person would '
          'confidently name it; plausible_real_size = a real object of that kind could have this size; '
          'everyday_tabletop_item = it is a common household or office object that could sit on a table and be '
          'carried in one hand (or, if open_container, hold small objects); single_clean_object = one object, not a '
          'scene, base plate, character figure, scan fragment or several glued parts; open_container = bowl, basket, '
          'tray, plate, box or pan that is open on top. Answer for every index.')


def label(root, thumbs, rows, output):
    root, output = Path(root).resolve(), Path(output); budget = reserve_budget(root)
    from corl2026_workshop.experiments.autonomous_pipeline_v4.models import reviewed_json
    results = {}
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]; name = f'{Path(output).stem}_{start:04d}'
        request = root / 'requests' / f'{name}.json'
        images = []
        for i, (key, ref) in enumerate(chunk):
            d = read(resolve(ref)); dims = ' x '.join(f'{100 * v:.0f}' for v in d['dimensions_m'])
            images.append({'label': f'Image index {i}: size {dims} cm', 'image': bind(Path(thumbs) / f'{key}.png')})
        write(request, {'schema': 'pp2_asset_labels_v1', 'model_config': bind(root / 'config.json'), 'prompt': PROMPT,
                        'images': images, 'response_schema': SCHEMA, 'keys': [k for k, _ in chunk]})
        try:
            review, _ = reviewed_json(root, budget, request)
            for item in review['items']:
                if 0 <= item['index'] < len(chunk):
                    results[chunk[item['index']][0]] = item
        except Exception as error:
            for key, _ in chunk:
                results.setdefault(key, {'unknown': f'{type(error).__name__}:{getattr(error, "code", error)}'[:160]})
        print(json.dumps({'batch': start, 'labelled': len(results)}), flush=True)
    write(output, {'schema': 'pp2_asset_labels_v1', 'labels': results, 'usage': budget.usage()})
    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--list', type=Path, required=True)
    p.add_argument('--thumbs', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); print(len(label(a.root, a.thumbs, read(a.list), a.output)))
