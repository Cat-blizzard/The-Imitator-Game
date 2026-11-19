import json
from pathlib import Path
import numpy as np
import pytest
import trimesh
from scipy.spatial import ConvexHull
from resimpair_grasp.common import read, write, binding
from resimpair_tabletop.sample_report import native_scalars


def test_sample_report_serializes_numpy_without_changing_decisions():
    report={'pass':np.bool_(False),'counts':[np.int64(3)],'value':np.float64(.25)}
    converted=json.loads(json.dumps(native_scalars(report),allow_nan=False))
    assert converted=={'pass':False,'counts':[3],'value':.25}


def test_native_proxy_encloses_real_vertices_after_cooking(tmp_path):
    pytest.importorskip('sapien')
    from resimpair_tabletop.proxy_v2 import cook_enclosing, PROFILE
    mesh=trimesh.creation.icosphere(subdivisions=2,radius=.04)
    envelope=mesh.copy();envelope.apply_scale(1.004)
    _,history,cooked,violation=cook_enclosing(envelope,mesh,tmp_path)
    equations=ConvexHull(cooked.vertices).equations
    observed=(mesh.vertices@equations[:,:3].T+equations[:,3]).max()
    assert observed<=PROFILE['max_enclosure_violation_m']
    assert violation==pytest.approx(observed)
    assert history[-1]['vertex_error_m']<1e-6
    assert 'container_interior' in PROFILE['forbidden_roles']


def test_query_keeps_size_split_and_unmeasured_status(tmp_path):
    from resimpair_tabletop.query import select
    rows=[]
    for i,(split,size) in enumerate([('development',.08),('heldout_asset',.08),('development',.3)]):
        path=tmp_path/f'{i}.json'
        write(path,{'selected_variant':'p00','variants':[{'id':'p00','dimensions_m':[size]*3}]})
        rows.append({'source_id':str(i),'status':'offline_tabletop_feasible','partition':split,
                     'category':'unknown','asset':binding(path)})
    catalog=tmp_path/'catalog.json';write(catalog,{'rows':rows})
    select(catalog,tmp_path/'selection',1,'development',[],.15,4)
    plan=read(tmp_path/'selection/plan.json')
    assert plan['sample'][0]['source_id']=='0'
    assert plan['physical_status']=='unmeasured'
    with pytest.raises(ValueError,match='only 1'):
        select(catalog,tmp_path/'too_many',2,'development',[],.15,4)
