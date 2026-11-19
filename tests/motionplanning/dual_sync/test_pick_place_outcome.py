"""Task success, early termination and recording failures remain separate."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from mani_skill.examples.motionplanning.dual.sync import pick_place


@pytest.mark.parametrize("success,fail,truncated,continued", [
    (True, False, False, True), (False, False, False, False),
    (True, True, False, False), (True, False, True, False),
])
def test_observer_only_continues_success_for_finishing(success, fail, truncated, continued):
    raw = NS(elapsed_steps=np.array([0]), agent=NS(agents=[]), targets=[],
             region_contains=lambda _: np.array([False]))

    def step(_):
        raw.elapsed_steps += 1
        return None, 0., True, truncated, {"success": success, "fail": fail}

    source = NS(unwrapped=raw, step=step)
    observer = pick_place.ObservedEpisode(source)
    output = observer.step({})
    assert bool(output[2]) is not continued
    assert observer.rows[-1]["terminated"] is True
    assert observer.rows[-1]["success"] is success
    assert bool(output[3]) is truncated


def test_recording_failure_preserves_real_task_success(tmp_path, monkeypatch):
    import json

    output = tmp_path/"episode"

    def disk_error():
        raise OSError("injected recording error")

    env = NS(reset=lambda **_: None, close=lambda: None,
             _json_data={}, flush_trajectory=disk_error, flush_video=lambda: None)
    monkeypatch.setattr(pick_place.gym, "make", lambda *_, **__: env)
    monkeypatch.setattr(pick_place, "RecordEpisode", lambda *_, **__: env)
    monkeypatch.setattr(pick_place, "solve", lambda *_, **__: {
        "execution": {"status": "passed"}, "task_success": True,
        "released": [True, True], "steps": 261})
    with pytest.raises(SystemExit) as raised:
        pick_place.main(["--robot", "piper", "--output-dir", str(output)])
    assert raised.value.code == 1
    result = json.loads((output/"result.json").read_text())
    assert result["task_success"] is True
    assert result["execution"]["status"] == "passed"
    assert result["recording"]["status"] == "failed"
    assert "injected recording error" in result["recording"]["error"]
