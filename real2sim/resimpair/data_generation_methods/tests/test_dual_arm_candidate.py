import numpy as np
import pytest
from data_generation_methods.dual_arm_candidate.synchronized_pick import synchronize_paths,paired_dense,measured_hold_command


def test_shorter_path_stretched_without_start_or_end_shift():
    a=np.linspace(np.zeros(7),np.ones(7),5)
    b=np.linspace(np.ones(7)*2,np.ones(7)*3,3)
    p=synchronize_paths([a,b])
    assert p.shape==(5,2,7)
    np.testing.assert_array_equal(p[:,0],a)
    np.testing.assert_array_equal(p[0,1],b[0]);np.testing.assert_array_equal(p[-1,1],b[-1])
    assert abs(np.diff(p[:,1],axis=0)).max()<=abs(np.diff(b,axis=0)).max()
    assert (abs(np.diff(p,axis=0)).sum(axis=2)>0).all()


def test_fixed_other_arm_stays_fixed_for_baseline():
    a=np.linspace(np.zeros(7),np.ones(7),8);fixed=np.ones((1,7))*2
    p=synchronize_paths([a,fixed]);np.testing.assert_array_equal(p[:,1],np.ones((8,7))*2)


def test_paired_densification_uses_common_time_not_cartesian_product():
    p=np.zeros((2,2,9));p[1,0]=.09;p[1,1]=-.03
    dense=paired_dense(p)
    assert len(dense)>=7
    np.testing.assert_allclose(dense[:,0],-3*dense[:,1])
    assert abs(np.diff(dense,axis=0)).max()<=.01500001


@pytest.mark.parametrize('paths',[[np.zeros((3,6)),np.zeros((2,7))],[np.zeros((0,7)),np.zeros((2,7))],[np.ones((2,7))*np.nan,np.zeros((2,7))]])
def test_bad_path_rejected(paths):
    with pytest.raises(ValueError):synchronize_paths(paths)


def test_measured_hold_target_maps_panda_normalized_action():
    command,target=measured_hold_command([.024,.026])
    assert target==pytest.approx(.0235)
    assert -.01+(command+1)*.025==pytest.approx(target)
    assert measured_hold_command([.0004,.0006])[1]==0.


def test_measured_hold_target_rejects_unknown_aperture():
    with pytest.raises(ValueError):measured_hold_command([.02,float('nan')])
