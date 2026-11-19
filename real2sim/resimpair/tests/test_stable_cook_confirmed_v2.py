from unittest.mock import patch

import numpy as np
import pytest
import trimesh

from resimpair_assets import stable_cook_confirmed_v2 as new
from resimpair_grasp.common import binding, read, write


def test_distance_retains_original_symmetric_vertex_rule():
    rng=np.random.default_rng(35);a=rng.normal(size=(11,3));b=a[:10].copy()
    pair=np.linalg.norm(a[:,None]-b[None,:],axis=-1)
    assert new.pointset_error(a,b)==max(pair.min(0).max(),pair.min(1).max())
    assert new.pointset_error(a,a[::-1])==0
    with pytest.raises(ValueError):new.pointset_error(a,np.full((1,3),np.nan))


def fake_probe(error):
    def probe(path,output):
        output.mkdir(parents=True)
        mesh=trimesh.load(path,force='mesh',process=False)
        np.savez(output/'native.npz',vertices=mesh.vertices+[.002,0,0],faces=mesh.faces)
        result={'vertex_error_m':error,'convergence_passed':error<new.CONVERGENCE_M,
                'runtime_replay_passed':error<new.RUNTIME_REPLAY_M,
                'native_arrays':binding(output/'native.npz')}
        write(output/'result.json',result);return result
    return probe


def test_nonconverged_file_is_never_returned_or_retried_beyond_three_conversions(tmp_path):
    with patch.object(new,'probe_file',side_effect=fake_probe(.002))as probe:
        with pytest.raises(ValueError,match='within 3 conversions'):
            new.stable_cook(trimesh.creation.box(extents=[.04,.03,.02]),tmp_path,0)
    assert probe.call_count==4 and len(list(tmp_path.glob('hull_0_cooked_*.obj')))==3
    result=read(tmp_path/'hull_0_confirmation_v2/RESULT.json')
    assert result['returned_file']is None and result['conversions']==3


def test_confirmed_file_is_returned_without_an_unchecked_export(tmp_path):
    with patch.object(new,'probe_file',side_effect=fake_probe(0.)):
        ref,passes=new.stable_cook(trimesh.creation.box(extents=[.04,.03,.02]),tmp_path,0)
    assert ref==binding(tmp_path/'hull_0_source.obj')==passes[-1]['input']==passes[-1]['output']
    assert not list(tmp_path.glob('hull_0_cooked_*.obj'))
    assert passes[-1]['vertex_error_m']==0 and passes[-1]['returned_file_exported_after_check']is False
    with pytest.raises(ValueError,match='no_overwrite'):
        new.stable_cook(trimesh.creation.box(),tmp_path,0)


@pytest.mark.parametrize('error', [1e-6, 5e-6])
def test_runtime_tolerance_never_replaces_stricter_convergence(tmp_path, error):
    with patch.object(new, 'probe_file', side_effect=fake_probe(error)) as probe:
        with pytest.raises(ValueError, match='within 3 conversions'):
            new.stable_cook(trimesh.creation.box(), tmp_path, 0)
    assert probe.call_count == 4
    assert read(tmp_path / 'hull_0_confirmation_v2/RESULT.json')['returned_file'] is None
