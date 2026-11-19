import pytest

from data_generation_methods.r2h_pipeline.budgeted_review_frames import select_review_frames


def fixture(count=100, events=()):
    rows = [{"frame_index": i, "timestamp_seconds": i / 24,
             "trajectory_px": [i, 10], "tracking_state": "appearance_candidate"}
            for i in range(count)]
    return {"source": {"decoded_frames": count}, "events": list(events)}, rows


def event(start, end, kind="appearance_candidate_displacement", priority=1):
    return {"kind": kind, "priority": priority, "start_frame": start, "end_frame": end}


def test_primary_event_uses_recorded_rules_and_unknown_without_hardcoded_indices():
    report, rows = fixture(events=[event(31, 35), event(33, 33, "tracking_unknown_interval", 2)])
    rows[33]["trajectory_px"] = None
    packet = select_review_frames(report, rows, 8)
    assert len(packet["selected_frame_indices"]) == 8
    assert {0, 31, 33, 35, 99} <= set(packet["selected_frame_indices"])
    assert len(packet["global_coverage_selection_steps"]) == 3
    assert packet["full_video_verified"] is False
    assert packet["training_admitted"] is False
    assert packet["event_coverage"][0]["dense_interval_fully_sampled"] is False


@pytest.mark.parametrize("budget", [0, 1, 2, 3, 4, 8, 12, 16, 120])
def test_hard_budget_boundaries_and_explicit_infeasibility(budget):
    report, rows = fixture(events=[event(5, 9)])
    rows[7]["trajectory_px"] = None
    packet = select_review_frames(report, rows, budget)
    assert len(packet["selected_frame_indices"]) == min(budget, len(rows))
    if budget >= 2:
        assert {0, 99} <= set(packet["selected_frame_indices"])
    assert packet["budget_insufficient_for_required_frames"] == (budget < 5)
    assert bool(packet["missing_required_frame_indices"]) == (budget < 5)


def test_multiple_events_report_remaining_uncovered_events_and_unknown_intervals():
    report, rows = fixture(events=[
        event(10, 12), event(40, 42), event(70, 80, "tracking_unknown_interval", 2)])
    rows[11]["trajectory_px"] = None
    packet = select_review_frames(report, rows, 5)
    assert packet["selected_frame_indices"] == [0, 10, 11, 12, 99]
    assert packet["unrepresented_event_ids"] == ["event_002", "event_003"]
    assert packet["event_representative_coverage_incomplete"] is True
    assert packet["unknown_intervals_with_missing_representatives"][0]["start_frame"] == 70
    assert packet["budget_status"] == "required_frames_selected_other_event_representatives_uncovered"


def test_priority_takes_precedence_over_earlier_frame():
    report, rows = fixture(events=[event(10, 11, priority=2), event(50, 51, priority=1)])
    packet = select_review_frames(report, rows, 4)
    assert packet["selected_frame_indices"] == [0, 50, 51, 99]
    assert packet["primary_displacement_event_id"] == "event_002"


def test_no_event_global_coverage_and_single_frame_are_well_defined():
    report, rows = fixture()
    packet = select_review_frames(report, rows, 3)
    assert packet["selected_frame_indices"] == [0, 49, 99]
    assert packet["primary_displacement_event_id"] is None
    single_report, single_rows = fixture(count=1)
    assert select_review_frames(single_report, single_rows, 8)["selected_frame_indices"] == [0]


def test_timestamp_or_count_mismatch_is_rejected():
    report, rows = fixture(count=3)
    rows[2]["timestamp_seconds"] = rows[1]["timestamp_seconds"]
    with pytest.raises(ValueError, match="timestamps"):
        select_review_frames(report, rows, 8)
    with pytest.raises(ValueError, match="count mismatch"):
        select_review_frames(report, rows[:2], 8)
