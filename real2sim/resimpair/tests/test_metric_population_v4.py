import json
from pathlib import Path
import numpy as np
import pytest
import trimesh
from resimpair_assets.metric_inventory_v6 import semantic_profile, metric_match, DEFAULT
from resimpair_assets.metric_native_v3 import support_evidence
from resimpair_grasp.common import ROOT, read, resolve
from resimpair_grasp.grasps import load


@pytest.mark.parametrize('name,role',[
    ('Стол со скатертью (Table with tablecloth)','background'),
    ('Upright Dyson Vacuum Cleaner','background'),('Curtain in the wind','background'),
    ('Toy miniature chair','manipulation'),('Tabletop toaster','manipulation'),
    ('Room Essentials Bowl','container'),('Room_Essentials_Bowl','container'),('Kitchen_Table','background'),('Unknown rigid mesh','manipulation')])
def test_role_errors_are_retrieval_specific_not_gripper_scaling(name,role):
    assert semantic_profile({'name':name},read(DEFAULT))['role']==role


def test_ABO_conflicting_dimensions_do_not_pick_a_convenient_measurement():
    records=[{'dimensions_m':{'height':s,'length':2*s,'width':3*s}} for s in [1.,2.]]
    assert metric_match([1,2,3],records)['status']=='conflicting_metric_records'
    assert metric_match([1,1,1],[records[0]])['status']=='no_metadata_uniform_shape_match'


def test_support_uses_native_COM_and_rejects_outside_projection():
    box=trimesh.creation.box(extents=[.1,.1,.1])
    assert support_evidence([box],[0,0,0])['status']=='pass'
    assert support_evidence([box],[.2,0,0])['status']=='COM_outside_support'


def test_real_inventory_preserves_all_raw_sources_and_source_partitions():
    path=ROOT/'runs/metric_assets_20261030_v4/inventory_v6/source_state_index.json'
    if not path.exists():pytest.skip('saved full inventory unavailable')
    value=read(path)
    assert len(value['entries'])==58117
    assert len({e['source_id'] for e in value['entries']})==58117
    assert sum(value['summary']['source_counts'].values())==58117
    assert all(e['reason'] and e['status'] for e in value['entries'])


def test_real_new_metric_pose_binding_and_retained_original_cache():
    new=ROOT/'runs/metric_assets_20261030_v3/cudex_shared/native_cache_index.json'
    if not new.exists():pytest.skip('saved real GPU qualification unavailable')
    entry=read(new)['entries'][0]
    d=read(resolve(entry['asset']));support=read(resolve(d['native_support']))
    settle=read(resolve(support['cpu_settle']))
    assert settle['status']=='pass' and settle['frames']==360
    assert all(settle['checks'].values())
    assert len(load(entry['cache'],entry['asset'])['candidates'])==21
    original=read(ROOT/'runs/cudex_bulk_5524_20261028/asset_skip_v5/native_cache_index.json')
    old=next(e for e in original['entries'] if e['source_id']==entry['source_id'])
    assert len(load(old['cache'],old['asset'])['candidates'])==47
    with pytest.raises(ValueError):load(old['cache'],entry['asset'])
