import pytest
from resimpair_grasp.common import ROOT,read
from resimpair_assets.metric_semantics_v1 import profile
from resimpair_assets.metric_admission_v1 import inspect

CONFIG=read(ROOT/'configs/metric_population_v7.json')


@pytest.mark.parametrize('name,categories,expected',[
    ('Fridge',['appliances'],'background'),
    ('Sewing Machine','appliances','background'),
    ('Old Microwave',['appliances'],'background'),
    ('Escritorio de 3 cajones',['HOME_FURNITURE_AND_DECOR'],'background'),
    ('Rolling bar cart',['HOME_FURNITURE_AND_DECOR'],'background'),
    ('Renova Shopping Bag',['bags_cases'],'background'),
    ('Toaster',['appliances'],'manipulation'),
    ('Unnamed utensil',['utensil'],'manipulation')])
def test_original_category_hierarchy_used_when_old_classification_is_null(name,categories,expected):
    p=profile({'name':name,'source_categories':categories,'category':None},CONFIG)
    assert p['role']==expected
    assert p['classification_evidence']['source_categories']


def test_unknown_object_does_not_default_to_gripper_scaled_dynamic():
    p=profile({'name':'ambiguous mesh','source_categories':['unknown']},CONFIG)
    assert p['name']=='role_scale_pending'


def test_real_miniature_furniture_caches_are_rejected_by_latest_source_plan():
    path=ROOT/'runs/metric_assets_20261030_v7/bundle_003/admission_audit.json'
    if not path.exists():pytest.skip('saved snapshot unavailable')
    rows=read(path)['cache_entries']
    bad=['Fridge','Sewing Machine','Old Microwave','Renova Shopping Bag','Movian','Rolling Bar Cart']
    for token in bad:
        row=next(r for r in rows if token in r['name'])
        assert not row['admission']['passed']
        assert 'current_source_role_not_dynamic' in row['admission']['reasons']


def test_real_single_source_size_variation_still_admitted():
    root=ROOT/'runs/metric_assets_20261030_v7/bundle_003'
    if not root.exists():pytest.skip('saved snapshot unavailable')
    bundle=read(root/'BUNDLE.json')
    assert bundle['source_semantics_checked_against_current_policy']['all_cache_entries_passed']
    entries=read(root/'native_cache_index.json')['entries']
    e=next(e for e in entries if 'Crayola_Bonus_64' in e['source_id'])
    assert e['semantic_scale_admission']['ratio_to_current_base']==pytest.approx(.9,rel=1e-5)
    assert all(e['semantic_scale_admission']['passed'] for e in entries)
    containers=read(root/'containers.json')
    assert len(containers['assets'])==107
