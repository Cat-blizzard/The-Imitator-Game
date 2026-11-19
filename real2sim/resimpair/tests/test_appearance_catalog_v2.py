import hashlib
from pathlib import Path

import pytest

from resimpair_tasks.appearance_catalog_v2 import (
    classify, select, choose_resources, summarize, family_group, transfer,
)


def test_cloth_and_terrain_do_not_become_tabletops():
    assert classify('crepe_georgette', {'categories': ['fabric']}, 'material') is None
    assert classify('forest_ground', {'categories': ['terrain', 'wood']}, 'material') is None
    value = classify('brick_wall', {'categories': ['brick', 'wall']}, 'material')
    assert 'tabletop' not in value['roles']


def test_hdris_require_indoor_and_reject_ruined_buildings():
    assert classify('office_01', {'categories': ['outdoor']}, 'hdri') is None
    assert classify('abandoned_kitchen', {'categories': ['indoor']}, 'hdri') is None
    value = classify('empty_studio', {'categories': ['indoor', 'studio']}, 'hdri')
    assert value['categories'] == ['generic_indoor']


def test_selection_balances_and_keeps_source_variants_together():
    values = {f'wood_table_{i:02d}': {'categories': ['wood']} for i in range(6)}
    values.update({'brick_wall_01': {'categories': ['brick']}, 'brick_wall_02': {'categories': ['brick']}})
    rows, counts = select(values, 'material', 8, 1)
    assert counts['selected_groups'] == {'brick': 2, 'wood': 6}
    assert len({r['asset_partition'] for r in rows if r['source_material_family'] == 'wood'}) == 1
    assert select(values, 'material', 8, 1)[0] == rows
    assert family_group('office_01_2') == 'office'


def test_missing_candidates_do_not_silently_duplicate():
    with pytest.raises(ValueError, match='insufficient_metadata_candidates'):
        select({'marble': {'categories': ['marble']}}, 'material', 2, 1)


def test_publisher_hash_host_and_gl_normal_required():
    spec = {'url': 'https://dl.polyhaven.org/file/test.jpg', 'size': 100, 'md5': 'a'*32}
    data = {k: {'1k': {'jpg': spec}} for k in ('Diffuse', 'Rough', 'nor_gl')}
    assert set(choose_resources(data, 'material')) == {'base_color', 'roughness', 'normal'}
    data['nor_dx'] = data.pop('nor_gl')
    with pytest.raises(ValueError, match='missing_required_1k_channel'):
        choose_resources(data, 'material')
    data['nor_gl'] = {'1k': {'jpg': {**spec, 'url': 'https://unrelated.invalid/a'}}}
    with pytest.raises(ValueError, match='unexpected_asset_host'):
        choose_resources(data, 'material')


def test_corrupt_existing_bytes_are_not_admitted(tmp_path):
    path = tmp_path / 'file.jpg'; path.write_bytes(b'bad')
    guard = transfer.Guard(tmp_path, 1000, reserve_bytes=0)
    spec = {'size': 3, 'md5': hashlib.md5(b'yes').hexdigest(), 'url': 'https://dl.polyhaven.org/file/a.jpg'}
    with pytest.raises(ValueError, match='wrong publisher MD5'):
        transfer.download(spec, path, guard)
    assert path.read_bytes() == b'bad'


def test_identical_content_not_double_counted_across_partitions():
    rows = [{'status': 'completed', 'entry': {'id': name, 'kind': 'material', 'roles': ['tabletop'],
             'categories': ['office'], 'asset_partition': part,
             'resources': {'base_color': {'sha256': 'same'}}}}
            for name, part in [('a', 'development'), ('b', 'heldout_appearance')]]
    result = summarize(rows, {})
    assert result['counts']['unique_active_assets'] == 1
    assert result['duplicates'][0]['same_bytes_as'] == 'a'
    assert len(result['entries']) == 1
