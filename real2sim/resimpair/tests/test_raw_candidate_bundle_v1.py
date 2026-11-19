import pytest
from resimpair_assets.raw_candidate_bundle_v1 import receiving_geometry


@pytest.mark.parametrize('height,floor,depth,expected,admit',[
    (.021,.0035,.0175,'shallow_tray_like_cavity',True),
    (.454,.438,.016,'elevated_shallow_recess_or_support_surface',False),
    (.128,.004,.124,'deeper_open_cavity_geometry_candidate',True),
    (.485,.451,.034,'intermediate_cavity_geometry_unknown_semantics',False),
])
def test_receiving_relations_are_not_semantic_passes(height,floor,depth,expected,admit):
    d={'bounds_m':[[-.2,-.2,0],[.2,.2,height]],
       'cavity':{'floor_z_m':floor,'depth_m':depth,'flat_floor_radius_m':.15}}
    r=receiving_geometry(d)
    assert r['classification']==expected and r['automatic_receiving_use']==admit
    assert r['assertable_by_R2H'] is False
    assert r['asserted_semantic_relation']=='unknown'


def test_nonfinite_container_cannot_enter_catalog():
    with pytest.raises(ValueError):receiving_geometry({'bounds_m':[[-1,-1,0],[1,1,1]],
       'cavity':{'floor_z_m':0,'depth_m':float('nan'),'flat_floor_radius_m':.15}})
