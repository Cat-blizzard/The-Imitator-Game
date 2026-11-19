from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import trimesh
from resimpair_grasp.common import ROOT,read
from resimpair_tabletop.geometry import propose,prepare_mesh,Clearance
from resimpair_tabletop.validation import check_pose

PROFILE=read(ROOT/'configs/tabletop_bulk_v1.json')


def test_native_gripper_table_and_swept_object_checks_reject_bad_pose():
    pytest.importorskip('embreex')
    mesh=trimesh.creation.box(extents=[.05,.05,.08])
    candidates,_=propose(mesh,PROFILE,0)
    assert len(candidates)>0
    good=candidates[0];assert check_pose(good,PROFILE,mesh.bounds)
    guard=Clearance(mesh);T=np.asarray(good['tcp_in_asset'])
    assert guard.check(T,mesh.bounds[0,2],.002)[0]
    buried=T.copy();buried[2,3]=mesh.bounds[0,2]
    assert not guard.check(buried,mesh.bounds[0,2],.002)[0]
    # Add an overhang in the vertical approach corridor, without changing the
    # originally sampled contacts. The continuous swept hand must reject it.
    roof=trimesh.creation.box(extents=[.14,.14,.015]);roof.apply_translation([0,0,.15])
    assert not Clearance(trimesh.util.concatenate([mesh,roof])).check(T,mesh.bounds[0,2],.002)[0]
    for field,value in [('width_m',.2),('physical_status','passed'),('table_clearance_m',0)]:
        bad=deepcopy(good);bad[field]=value
        with pytest.raises(ValueError):check_pose(bad,PROFILE,mesh.bounds)


def test_bulk_scale_is_chosen_before_grasp_search_and_flat_objects_are_skipped():
    cube=trimesh.creation.box(extents=[3,3,3]);m,S,poses,info=prepare_mesh(cube,PROFILE)
    assert np.isclose(m.extents.max(),PROFILE['min_longest_m'])
    assert poses and not info['real_metric_size_inferred']
    with pytest.raises(ValueError,match='wide_flat'):
        prepare_mesh(trimesh.creation.box(extents=[.2,.2,.005]),PROFILE)


def test_unknown_semantic_category_is_not_a_grasp_eligibility_gate(tmp_path):
    from resimpair_assets.catalog import build
    from resimpair_tabletop.bulk import plan
    import json
    mesh=tmp_path/'shape.glb';trimesh.creation.icosphere(subdivisions=1).export(mesh)
    source=tmp_path/'source.jsonl';source.write_text(json.dumps({'asset_id':'custom:padlock','source':'custom','name':'Padlock',
        'kind':'model','categories':['lids_handles_hardware'],'primary_path':str(mesh)})+'\n')
    index=tmp_path/'raw.sqlite';build(source,ROOT/'configs/asset_roles_v4.json',index)
    options=SimpleNamespace(root=tmp_path/'bulk',index=index,profile=ROOT/'configs/tabletop_bulk_v1.json',limit=10,
        workers=2,timeout=60,disk_gib=1,wall_seconds=120,seed=1,partition=None)
    result=plan(options);assert result['scheduled']==1 and result['physics_executions']==0
