from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from data_generation_methods.r2h_pipeline.local_hands import inspect_image


def test_detector_cap_cannot_hide_the_extra_hands_being_checked():
    with pytest.raises(ValueError, match="num_hands >= 4"):
        inspect_image("missing.png", "missing.task", num_hands=2)
    with pytest.raises(ValueError, match="exceed"):
        inspect_image("missing.png", "missing.task", num_hands=4,
                      max_expected_hands=4)


@pytest.mark.parametrize("detected", [0, 2, 3])
def test_detected_count_never_becomes_quality_approval(tmp_path, monkeypatch, detected):
    # Exercise release semantics without depending on optional MediaPipe or GPU.
    image, model = tmp_path / "frame.png", tmp_path / "model.task"
    image.write_bytes(b"fixture image")
    model.write_bytes(b"fixture model")
    landmarks = [
        [SimpleNamespace(x=i/40, y=i/50, z=0.0) for i in range(21)]
        for _ in range(detected)
    ]
    result = SimpleNamespace(
        hand_landmarks=landmarks,
        hand_world_landmarks=landmarks,
        handedness=[[SimpleNamespace(category_name="Left", score=0.9)]
                    for _ in range(detected)],
    )

    class Detector:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def detect(self, image):
            return result

    class BaseOptions:
        Delegate = SimpleNamespace(CPU="CPU")

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    mp = ModuleType("mediapipe")
    mp.__version__ = "fixture"
    mp.Image = SimpleNamespace(create_from_file=lambda _: SimpleNamespace(width=512, height=512))
    tasks, python, vision = [ModuleType(name) for name in (
        "mediapipe.tasks", "mediapipe.tasks.python", "mediapipe.tasks.python.vision")]
    python.BaseOptions = BaseOptions
    vision.HandLandmarkerOptions = lambda **kwargs: SimpleNamespace(**kwargs)
    vision.RunningMode = SimpleNamespace(IMAGE="IMAGE")
    vision.HandLandmarker = SimpleNamespace(create_from_options=lambda _: Detector())
    tasks.python, python.vision, mp.tasks = python, vision, tasks
    for module in (mp, tasks, python, vision):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    out = inspect_image(image, model)
    assert out["detected_hand_count_lower_bound"] == detected
    assert out["count_is_complete"] is False
    assert out["quality_release"] is False
    assert out["training_admitted"] is False
    assert out["extra_hand_candidate"] is (detected > 2)
    assert out["finding"] != "pass"
