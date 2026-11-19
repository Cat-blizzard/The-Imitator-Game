import numpy as np
import pytest

from data_generation_methods.r2h_pipeline.contact_diagnostics import (
    corrected_displacement, forward_backward_valid, validate_roi,
)


def test_translation_of_entire_camera_is_not_reported_as_handle_motion():
    h0 = np.array([[10, 10], [20, 10], [20, 20]], dtype=float)
    b0 = np.array([[50, 50], [60, 50], [60, 60], [55, 50], [54, 54], [70, 70]], dtype=float)
    shift = np.array([7, -3])
    assert corrected_displacement(h0+shift, h0, b0+shift, b0) == [0.0, 0.0]
    assert corrected_displacement(h0+shift+[2, 4], h0, b0+shift, b0) == [2.0, 4.0]


def test_loss_of_point_support_is_unknown_not_zero_displacement():
    for hnum, bnum in [(2, 6), (3, 5), (0, 0)]:
        h0, b0 = np.zeros((hnum, 2)), np.zeros((bnum, 2))
        assert corrected_displacement(h0, h0, b0, b0) is None


def test_forward_backward_and_image_bounds_reject_unreliable_tracks():
    previous = np.array([[5, 5], [10, 10], [20, 20], [30, 30]], dtype=float)
    forward = previous+1
    forward[2] = [101, 20]
    backward = previous.copy()
    backward[1] += [3, 0]
    valid, _ = forward_backward_valid(
        previous, forward, backward, [1, 1, 1, 1], [1, 1, 1, 0],
        width=100, height=100, max_error_px=1.5,
    )
    assert valid.tolist() == [True, False, False, False]


@pytest.mark.parametrize("roi", [[0.1, 0.2, 0.1, 0.5], [0, 0, 1.1, 1], [0, 0, float("nan"), 1]])
def test_invalid_manual_roi_fails_before_inference(roi):
    with pytest.raises(ValueError):
        validate_roi(roi)
