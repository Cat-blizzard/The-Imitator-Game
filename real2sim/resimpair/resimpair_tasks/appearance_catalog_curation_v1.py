"""Correct metadata role assignment without rewriting the downloaded baseline.

The first download selection matched 'tile' inside 'textile'/'tileable'. This
read-only-in/new-directory-out curation fixes that real failure, preserving all
original selection, transfer failures, sources, split groups and byte receipts.
"""
import argparse
from collections import Counter
from pathlib import Path
import re

from resimpair_grasp.common import binding, read, resolve, write
from .appearance_catalog_v2 import classify as original_classify

VERSION = 'appearance_semantic_curation_v1'
FAMILIES = ['kitchen', 'dining', 'office', 'workshop', 'storage']


def material_roles(asset_id, info):
    cats = {c.lower() for c in info.get('categories', [])}
    words = set(re.findall(r'[a-z]+', ' '.join([asset_id, info.get('name', ''),
        *info.get('categories', []), *info.get('tags', [])]).lower()))
    if cats & {'fabric', 'leather', 'textile', 'textiles'} or words & {'textile', 'textiles', 'leather', 'upholstery'}:
        return None, 'textile_or_leather_is_not_a_qualified_rigid_surface'
    if cats & {'terrain', 'aerial', 'asphalt', 'road', 'roofing', 'bark', 'sand', 'rock', 'snow', 'ice'}:
        return None, 'outdoor_ground_or_natural_surface'
    dirty = bool(words & {'damaged', 'rust', 'rusty', 'broken', 'dirty', 'rotting', 'peeling'})
    if 'wood' in cats or words & {'wood', 'veneer', 'wooden'}:
        family = 'wood'; roles = ['floor', 'wall', 'cabinet']; families = ['workshop', 'storage'] if dirty else FAMILIES
        if not dirty: roles.append('tabletop')
    elif words & {'marble', 'granite', 'terrazzo'}:
        family = 'stone'; roles = ['tabletop', 'floor', 'wall']; families = ['kitchen', 'dining', 'office']
    elif cats & {'tiles', 'tile'} or words & {'tile', 'tiles', 'tiling'}:
        family = 'tile'; roles = ['floor', 'wall']; families = ['kitchen', 'dining', 'storage']
    elif 'metal' in cats or words & {'metal', 'steel', 'aluminum', 'aluminium', 'iron', 'copper', 'brass'}:
        family = 'metal'; roles = ['metal', 'cabinet', 'wall']; families = ['workshop', 'storage'] if dirty else list(FAMILIES)
        if not dirty and not words & {'corrugated', 'sheet', 'mesh', 'grate'}: roles.append('tabletop')
    elif 'brick' in cats or words & {'brick', 'bricks'}:
        family = 'brick'; roles = ['wall']; families = ['office', 'workshop', 'storage']
    elif 'plaster' in cats or words & {'plaster', 'paint', 'painted'}:
        family = 'plaster'; roles = ['wall']; families = ['workshop', 'storage'] if dirty else FAMILIES
    elif 'concrete' in cats or words & {'concrete', 'cement'}:
        family = 'concrete'; roles = ['floor', 'wall']; families = ['workshop', 'storage']
    else:
        return None, 'no_supported_rigid_surface_family_in_metadata'
    return {'roles': roles, 'categories': list(families), 'source_material_family': family,
            'default_metallic': 1.0 if family == 'metal' else 0.0}, None


def curate(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists(): raise ValueError('curation_output_must_be_new')
    value = read(source)
    if value['schema'] != 'resimpair_appearance_catalog_v2': raise ValueError('downloaded_catalog_v2_required')
    entries = []; excluded = []; changed = []
    for entry in value['entries']:
        receipt = read(resolve(entry['receipt']))
        for ref in entry['resources'].values(): resolve(ref)
        if entry['kind'] == 'hdri': entries.append(entry); continue
        actual, reason = material_roles(entry['source_asset_id'], receipt['source_metadata'])
        if actual is None:
            excluded.append({'id': entry['id'], 'reason': reason, 'original_entry': entry}); continue
        differences = {k: {'before': entry.get(k), 'after': v} for k, v in actual.items() if entry.get(k) != v}
        updated = {**entry, **actual, 'semantic_tag_basis': 'publisher_metadata_word_boundary_rules_v1_not_manual_scene_validation'}
        if differences: changed.append({'id': entry['id'], 'fields': differences})
        entries.append(updated)
    result = {**value, 'entries': entries, 'uncurated_catalog': binding(source),
        'curation': {'schema': VERSION, 'implementation': binding(__file__),
            'failure_diagnosis': "substring 'tile' matched textile/tileable; original records retained",
            'excluded': excluded, 'relabeled': changed, 'download_attempts_added': 0,
            'original_failures_preserved': True, 'new_asset_sources_added': 0,
            'source_partitions_unchanged': True},
        'counts': {**value['counts'], 'unique_active_assets': len(entries),
            'by_kind': dict(Counter(e['kind'] for e in entries)),
            'by_role': dict(Counter(r for e in entries for r in e['roles'])),
            'by_scene_category': dict(Counter(c for e in entries for c in e['categories'])),
            'by_partition': dict(Counter(e['asset_partition'] for e in entries)),
            'curation_excluded': len(excluded), 'curation_relabeled': len(changed)},
        'curated_from_same_completed_download_campaign': True}
    write(output / 'catalog.json', result)
    return {'catalog': binding(output / 'catalog.json'), 'counts': result['counts']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args(); print(curate(a.source, a.output))
