import pytest
import trimesh
from resimpair_grasp.common import binding
from resimpair_tabletop.native_mass import calibrated_density,native_mass


def test_native_compound_density_reaches_requested_mass_without_dynamics(tmp_path):
    pytest.importorskip('sapien')
    bindings=[]
    for i in range(3):
        mesh=trimesh.creation.box(extents=[.02,.015,.025]);mesh.apply_translation([i*.022,0,0])
        path=tmp_path/f'part{i}.obj';mesh.export(path);bindings.append(binding(path))
    density,history=calibrated_density(bindings,2000.,.08)
    assert len(history)>1
    assert abs(native_mass(bindings,density)-.08)<1e-6
    assert history[0]['native_mass_kg']!=pytest.approx(.08)


def test_native_mass_preflight_rejects_invalid_target():
    with pytest.raises(ValueError,match='Positive'):calibrated_density([],1.,float('nan'))
