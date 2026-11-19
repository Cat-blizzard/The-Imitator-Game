"""Integration regression over retained real media; never starts a simulator."""
from pathlib import Path

import numpy as np
import pytest

from resimpair_grasp.common import ROOT, read, resolve
from resimpair_pickplace.render_lineage_validator_v2 import validate


NEGATIVES = ROOT/'validation/pickplace_automation_20261029/independent/lineage_negative_controls_v2'
RENDERED = ROOT/'runs/deferred_render_20261029_v1'
pytestmark = pytest.mark.skipif(not (NEGATIVES/'cases.json').exists(),
                                reason='requires retained real-render regression artifacts')


@pytest.mark.parametrize('name', ['original', 'appearance', 'original_bound', 'appearance_bound'])
def test_actual_render_preserves_original_state_clock_and_calibration(name):
    result = validate(RENDERED/name)
    assert result['passed'], result['errors']
    assert result['frames_per_camera'] == 398
    assert len(result['cameras']) == 2
    assert result['new_physical_executions'] == 0


@pytest.mark.parametrize('name', ['foreign_saved_states', 'mapping_binding_mismatch',
    'wrong_camera_K', 'missing_readback_dimension', 'changed_partition',
    'swapped_small_object_endpoints', 'absolute_clock_drift'])
def test_retained_corruption_is_rejected(name):
    cases = read(NEGATIVES/'cases.json')['cases']
    case = next(row for row in cases if row['id'] == name)
    resolve(case['bundle']); resolve(case['frame_mapping'])
    result = validate(case['directory'])
    assert not result['passed']
    assert any(case['expected_error'] in error['reason'] or case['expected_error'] == error['type']
               for error in result['errors']), result['errors']


def test_real_clock_drift_exposes_relative_tolerance_false_acceptance():
    import av
    bundle = read(NEGATIVES/'absolute_clock_drift/execution_bundle.json')
    with np.load(resolve(bundle['measurements']), allow_pickle=False) as arrays:
        expected = arrays['timestamps_seconds']
    with av.open(str(resolve(bundle['cameras']['opposite_high']['video']))) as clip:
        actual = np.array([float(frame.pts*frame.time_base) for frame in clip.decode(video=0)])
    assert np.allclose(actual, expected, atol=1e-5)  # prior compiler's implicit rtol
    assert not np.allclose(actual, expected, atol=1e-5, rtol=0)
