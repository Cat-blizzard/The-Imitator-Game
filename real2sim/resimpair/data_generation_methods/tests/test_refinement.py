import pytest

from data_generation_methods.r2h_pipeline.refinement import plan_refinement


def req(lo, hi, name="q"):
    return dict(query_id=name, frame_interval=[lo, hi],
                interval="start_inclusive_end_inclusive")


def test_short_event_fully_sampled_whole_video_deferred():
    p = plan_refinement(193, [0, 68, 79, 192],
                        [req(0, 192, "absence"), req(68, 79, "grasp")], max_frames=20)
    assert set(range(68, 80)) <= set(p["selected_frame_indices"])
    assert p["requests"][0]["selection"] == "deferred_budget"
    assert p["requests"][1]["selection"] == "continuous_interval_selected"
    assert p["added_frame_count"] == 10
    assert not p["requests"][1]["semantic_uncertainty_resolved"]


def test_overlapping_requests_share_frames():
    p = plan_refinement(20, [0, 19], [req(4, 8), req(6, 10)], max_frames=9)
    assert len(p["selected_frame_indices"]) == 9
    assert all(r["selection"] == "continuous_interval_selected" for r in p["requests"])


def test_interval_that_cannot_fit_is_not_partially_called_dense():
    p = plan_refinement(20, [0, 19], [req(5, 14)], max_frames=9)
    assert p["selected_frame_indices"] == [0, 19]
    assert p["requests"][0]["selection"] == "deferred_budget"


@pytest.mark.parametrize("bounds", [(-1, 3), (5, 20), (5, 4), (True, 5)])
def test_out_of_media_bounds_rejected(bounds):
    with pytest.raises(ValueError):
        plan_refinement(20, [0, 19], [req(*bounds)])
