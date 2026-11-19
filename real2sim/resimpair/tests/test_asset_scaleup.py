from pathlib import Path
import copy
import json
import numpy as np
import pytest
import trimesh

from resimpair_assets.catalog import categories,classify,balanced,build,search
from resimpair_assets.acceptance import evaluate
from resimpair_grasp.common import ROOT,binding,write

POLICY=json.loads((ROOT/'configs/asset_roles_v1.json').read_text())


def test_metadata_types_and_unsupported_affordances():
    assert categories('Bottles and Cans and Cups')==['Bottles and Cans and Cups']
    assert categories(None)==[]
    assert classify({'kind':'MJCF_definition','name':'bottle','categories':[]},POLICY)[1]=='not_a_mesh_model'
    assert classify({'kind':'model','name':'kitchen scene with cup','categories':['drinkware']},POLICY)[0] is None
    assert classify({'kind':'model','name':'bottle shelf','categories':['furniture_storage']},POLICY)[0] is None
    d,reason=classify({'kind':'model','name':'blue water bottle','categories':'Consumer Goods'},POLICY)
    assert reason is None and d['category']=='bottle' and d['semantic_status']=='metadata_candidate_not_visual_confirmation'


def test_fruit_words_and_multipacks_do_not_grant_manipulation_semantics():
    policy=json.loads((ROOT/'configs/asset_roles_v2.json').read_text())
    for name in ['Focus Lime Squeezer','Apple iPhone Cover','5 paint bottles','Pepsi 12 12 fl oz cans']:
        d,_=classify({'kind':'model','name':name,'categories':[]},policy)
        assert d and d['semantic_ambiguities'] and not d['manipulation_candidate']
    d,_=classify({'kind':'model','name':'lime squeezer','categories':[]},policy)
    assert d['category']=='utensil'


def test_source_ontology_and_object_noun_override_color_words():
    policy=json.loads((ROOT/'configs/asset_roles_v3.json').read_text())
    assert classify({'kind':'model','name':'ASICS Safety Orange','categories':'Shoes'},policy)[0] is None
    d,_=classify({'kind':'model','name':'Orchid Pot Ceramic Lime','categories':'Consumer Goods'},policy)
    assert d['category']=='container'
    d,_=classify({'kind':'model','name':'low poly banana','categories':['food_props']},policy)
    assert d['category']=='fruit'
    assert classify({'kind':'model','name':'orange scented candle','categories':[]},policy)[0] is None


def test_balanced_sampling_is_reproducible_and_deduplicates_content():
    rows=[{'id':f'{c}:{i}','category':c,'source':s,'source_sha':f'{c}:{i}'} for c in ['cup','bottle'] for s in ['a','b'] for i in range(10)]
    a=balanced(rows,12,5);b=balanced(list(reversed(rows)),12,5)
    assert a==b and len(a)==12 and len({r['source_sha'] for r in a})==12
    assert sum(x['category']=='cup' for x in a)==6
    assert not any(x['id']=='cup:3' for x in balanced(rows,30,5,exclude=['cup:3']))


def test_raw_search_does_not_confuse_filename_with_qualification(tmp_path):
    mesh=tmp_path/'bottle.glb';trimesh.creation.box().export(mesh)
    source=tmp_path/'raw.jsonl';source.write_text(json.dumps({'asset_id':'x:1','source':'x','kind':'model','name':'green bottle','categories':'Consumer Goods','primary_path':str(mesh)})+'\n')
    output=tmp_path/'index.sqlite';result=build(source,ROOT/'configs/asset_roles_v1.json',output)
    assert result['qualified_for_simulation']==0
    rows=search(output,query='bottle',role='manipulation')
    assert len(rows)==1 and rows[0]['classification']['category']=='bottle'
    assert not search(output,query='cup')
    with pytest.raises(ValueError):build(source,ROOT/'configs/asset_roles_v1.json',output)


def test_static_proxy_has_narrow_role_and_cannot_hide_size_mismatch(tmp_path):
    mesh=trimesh.creation.box(extents=[.05,.06,.10])
    visual=tmp_path/'visual.glb';mesh.export(visual)
    collision=tmp_path/'box.obj';mesh.export(collision)
    d={'collision_policy':'static_only_conservative_AABB','roles':{'rigid_manipulation_candidate':False,'container_support':False},
       'visual':binding(visual),'collisions':[binding(collision)],'bounds_m':mesh.bounds.tolist(),
       'dimensions_m':mesh.extents.tolist(),'half_size':(mesh.extents/2).tolist(),'design_longest_m':.10}
    path=tmp_path/'descriptor.json';write(path,d)
    assert evaluate(path)['passed']
    for n,mutate in [('role',lambda x:x['roles'].update(rigid_manipulation_candidate=True)),
                     ('size',lambda x:x.update(half_size=[.05,.06,.10])),
                     ('scale',lambda x:x.update(design_longest_m=.20))]:
        bad=copy.deepcopy(d);mutate(bad);p=tmp_path/(n+'.json');write(p,bad)
        with pytest.raises(ValueError):evaluate(p)
