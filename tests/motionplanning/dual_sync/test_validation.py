"""Preserve caller-specific conversion, ownership and error contracts."""
from __future__ import annotations

import numpy as np
import pytest

from mani_skill.examples.motionplanning.dual.sync import combined_collision
from mani_skill.examples.motionplanning.dual.sync import curobo_backend
from mani_skill.examples.motionplanning.dual.sync import curobo_scene
from mani_skill.examples.motionplanning.dual.sync import curobo_spheres
from mani_skill.examples.motionplanning.dual.sync import pill_task
from mani_skill.examples.motionplanning.dual.sync import planning_endpoint
from mani_skill.examples.motionplanning.dual.sync._arrays import as_float_array


def test_shared_conversion_preserves_tensor_protocol_and_numpy_views():
    source = np.arange(6, dtype=np.float64).reshape(2, 3)
    calls = []

    class Tensor:
        def detach(self):
            calls.append("detach")
            return self

        def cpu(self):
            calls.append("cpu")
            return self

        def numpy(self):
            calls.append("numpy")
            return source

    for convert in (as_float_array, pill_task._array, curobo_scene._array, curobo_spheres._array):
        calls.clear()
        actual = convert(Tensor())
        assert calls == ["detach", "cpu", "numpy"]
        assert actual.dtype == np.float64 and actual.shape == (2, 3)
        assert np.shares_memory(actual, source)
        assert np.shares_memory(convert(source), source)
        assert convert(np.array([1], dtype=np.float32)).dtype == np.float64
        assert np.isnan(convert([np.nan])).all()


def test_geometry_single_view_and_task_owned_vector_remain_distinct():
    source = np.arange(3, dtype=np.float64)[None]
    scene_vector = curobo_scene._single(source, 3, "XYZ")
    task_vector = pill_task._single(source, 3, "XYZ")
    collision_vector = combined_collision._single(source, 3)
    assert np.shares_memory(scene_vector, source)
    assert not np.shares_memory(task_vector, source)
    assert not np.shares_memory(collision_vector, source)
    task_vector[:] = 10.
    collision_vector[:] = 20.
    assert np.array_equal(source, [[0., 1., 2.]])


@pytest.mark.parametrize("value", ([1., 2.], [1., 2., np.inf]))
def test_finite_validation_keeps_error_labels_and_shape_details(value):
    with pytest.raises(ValueError) as error:
        pill_task._finite(value, (3,), "target XYZ")
    assert str(error.value) == f"target XYZ must be finite with shape (3,), got {np.shape(value)}"
    for finite in (curobo_scene._finite, curobo_backend._finite):
        with pytest.raises(ValueError) as error:
            finite(value, (3,), "target XYZ")
        assert str(error.value) == "target XYZ must be finite with shape (3,)"


def test_collision_and_endpoint_keep_owned_arrays_and_finite_errors():
    source = np.arange(3, dtype=np.float64)
    for module, message in (
        (combined_collision, "Joint states, poses and times must be finite"),
        (planning_endpoint, "Endpoint inputs must be finite"),
    ):
        result = module._array(source)
        assert not np.shares_memory(result, source)
        with pytest.raises(ValueError) as error:
            module._array([np.nan])
        assert str(error.value) == message


def test_endpoint_independent_tensor_protocol_and_quaternion_no_mutation():
    class CpuArray:
        def cpu(self):
            return self

        def numpy(self):
            return np.ones(3)

    assert np.array_equal(planning_endpoint._array(CpuArray()), np.ones(3))
    pose = np.array([.1, .2, .3, 1. + 1e-7, 0., 0., 0.])
    before = pose.copy()
    normalized = planning_endpoint._pose(pose)
    assert np.array_equal(pose, before)
    assert not np.shares_memory(normalized, pose)
    assert np.linalg.norm(normalized[3:]) == pytest.approx(1.)
    quaternion = pose[3:]
    pill_task.quaternion_matrix_wxyz(quaternion)
    assert np.array_equal(pose, before)
