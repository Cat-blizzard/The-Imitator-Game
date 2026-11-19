import numpy as np
import pytest
from data_generation_methods.human_to_robot_pipeline.plan_geometry_validator import validate_geometry


def path(points):
    result = np.repeat(np.eye(4)[None], len(points), axis=0)
    result[:,:3,3] = points
    return result


def test_exact_endpoint_is_checked_when_provided():
    poses = path([[0,0,.3], [0,0,.2]])
    report = validate_geometry(poses, workspace_min=[-1,-1,0], workspace_max=[1,1,1],
                               desired_end_pose=poses[-1])
    assert report["endpoint"]["status"] == "pass"
    assert not report["live_execution_guard_installed"]


def test_missing_desired_endpoint_remains_unknown():
    report = validate_geometry(path([[0,0,.2]]), workspace_min=[-1,-1,0], workspace_max=[1,1,1])
    assert report["endpoint"]["status"] == "unknown"


def test_task_departure_is_caught_before_camera_boundary():
    report = validate_geometry(path([[0,0,.3], [0,-.4,.5], [0,-.9,.8]]),
                               workspace_min=[-.35,-.55,0], workspace_max=[.3,.55,.65],
                               xy_center=[0,0], xy_radius_m=.12, maximum_z=.34)
    assert report["failures"]["outside_local_xy_corridor"] == [1,2]
    assert report["failures"]["outside_workspace"] == [2]


def test_valid_shape_wrong_goal_cannot_pass():
    poses = path([[0,0,.3], [.05,0,.2]])
    goal = path([[0,0,.2]])[0]
    report = validate_geometry(poses, workspace_min=[-1,-1,0], workspace_max=[1,1,1],
                               desired_end_pose=goal)
    assert report["endpoint"]["status"] == "fail"


def test_unqualified_experimental_adapter_stops_before_any_new_output(tmp_path):
    from data_generation_methods.human_to_robot_pipeline.experimental_apple_adapter import run
    output = tmp_path/"must_not_exist"
    with pytest.raises(RuntimeError, match="strict per-path"):
        run({}, {}, output, 2)
    assert not output.exists()
