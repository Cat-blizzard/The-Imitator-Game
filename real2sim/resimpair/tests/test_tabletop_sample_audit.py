import numpy as np
from resimpair_tabletop.sample_audit import lift_measurements,wilson


def test_lift_needs_contact_duration_and_no_teleport():
    state=np.zeros((40,1,13));state[:12,0,2]=np.linspace(0,.20,12);state[12:,0,2]=.20
    arrays={'timestamps_seconds':np.arange(40)/60.,'object_states':state,'grasp':np.ones((40,2,1),bool)}
    assert lift_measurements(arrays,0,0)['passed']
    arrays['grasp'][:]=False
    assert not lift_measurements(arrays,0,0)['passed']
    arrays['grasp'][:]=True;state[:12,0,2]=0
    assert not lift_measurements(arrays,0,0)['passed']


def test_small_samples_do_not_prove_universal_success():
    low,high=wilson(6,6)
    assert low<.7 and high<=1
