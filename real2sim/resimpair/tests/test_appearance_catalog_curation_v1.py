from resimpair_tasks.appearance_catalog_curation_v1 import material_roles
from resimpair_tasks.appearance_catalog_v2 import classify


def test_real_textile_false_positive_is_rejected():
    info = {'name': 'Leather Red 03', 'categories': ['fabric', 'leather'],
            'tags': ['leather', 'red', 'even', 'smooth', 'new', 'lustrous', 'textile ', 'upholstery']}
    assert classify('leather_red_03', info, 'material')['source_material_family'] == 'tile'
    actual, reason = material_roles('leather_red_03', info)
    assert actual is None and reason == 'textile_or_leather_is_not_a_qualified_rigid_surface'


def test_tileable_does_not_mean_tile():
    info = {'categories': ['plaster', 'wall'], 'tags': ['smooth', 'tileable']}
    assert classify('plain_plaster', info, 'material')['source_material_family'] == 'tile'
    assert material_roles('plain_plaster', info)[0]['source_material_family'] == 'plaster'
    assert material_roles('plain_plaster', info)[0]['roles'] == ['wall']


def test_clean_tiles_and_wood_still_usable():
    assert material_roles('square_tiles_02', {'categories': ['floor', 'tiles']})[0]['source_material_family'] == 'tile'
    assert 'tabletop' in material_roles('clean_wood_01', {'categories': ['wood']})[0]['roles']
    assert 'tabletop' not in material_roles('rusty_metal_01', {'categories': ['metal']})[0]['roles']
    assert 'office' in material_roles('clean_brass_01', {'categories': ['metal']})[0]['categories']
    assert 'office' not in material_roles('rusty_metal_01', {'categories': ['metal']})[0]['categories']
