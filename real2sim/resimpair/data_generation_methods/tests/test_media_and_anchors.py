import base64
import io
from pathlib import Path
from types import SimpleNamespace
import subprocess

from PIL import Image
import pytest

from data_generation_methods.r2h_pipeline import anchors, media, providers
from data_generation_methods.r2h_pipeline.common import file_sha256
from data_generation_methods.r2h_pipeline.cli import DEFAULT_CONFIG


def test_provider_derivative_keeps_time_and_original(tmp_path):
    original = tmp_path / "native.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=blue:s=320x240:r=20:d=4",
                    "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", str(original)], check=True)
    source_hash = file_sha256(original)
    result = media.prepare_seedance_video(original, tmp_path / "api.mp4")
    assert file_sha256(original) == source_hash
    assert result["derived"]["fps"] == 24
    assert result["derived"]["width"] * result["derived"]["height"] >= 407696
    assert abs(result["source"]["duration_seconds"] - result["derived"]["duration_seconds"]) <= 1 / 24
    assert media.prepare_seedance_video(original, tmp_path / "api.mp4") == result


def test_long_edit_source_is_not_silently_cropped(tmp_path, monkeypatch):
    original = tmp_path / "native.mp4"; original.write_bytes(b"fixture")
    monkeypatch.setattr(media, "probe_video", lambda _: {"duration_seconds": 100})
    with pytest.raises(ValueError, match="No automatic prefix crop"):
        media.prepare_seedance_video(original, tmp_path / "api.mp4")
    assert not (tmp_path / "api.mp4").exists()


@pytest.fixture
def anchor_store(tmp_path):
    image = tmp_path / "initial.png"
    Image.new("RGB", (8, 8), "blue").save(image)
    frame = {"path": str(image), "sha256": file_sha256(image), "frame_index": 0}
    bound = []
    return SimpleNamespace(path=tmp_path / "ledger.sqlite3",
        rows=lambda: [{"group_id": "group-a", "job_id": "a"}, {"group_id": "group-a", "job_id": "b"}],
        load_job=lambda _: {"media": {"event_frames": [frame]}},
        bind_anchor=lambda *args: bound.append(args) or {"path": str(args[1])}, bound=bound)


def test_anchor_is_shared_and_api_result_resumes_without_repeat(anchor_store, monkeypatch):
    anchors.plan_anchor(anchor_store, "group-a", DEFAULT_CONFIG)
    output = io.BytesIO(); Image.new("RGB", (8, 8), "red").save(output, format="PNG")
    calls = []
    class FakeClient:
        def __init__(self, _): pass
        def generate_image(self, image, prompt, model):
            calls.append((image, prompt, model))
            return {"images": [{"mime_type": "image/png", "data": base64.b64encode(output.getvalue()).decode()}]}
    monkeypatch.setattr(providers, "GeminiClient", FakeClient)
    monkeypatch.setenv("GEMINI_API_KEY", "test-only")
    first = anchors.generate_anchor(anchor_store, "group-a", DEFAULT_CONFIG)
    second = anchors.generate_anchor(anchor_store, "group-a", DEFAULT_CONFIG)
    assert first == second and len(calls) == 1
    assert "retrieve_middle" not in calls[0][1]


def test_uncertain_anchor_call_is_not_repeated(anchor_store, monkeypatch):
    anchors.plan_anchor(anchor_store, "group-a", DEFAULT_CONFIG)
    calls = []
    class FakeClient:
        def __init__(self, _): pass
        def generate_image(self, *args, **kwargs):
            calls.append(1)
            raise providers.ProviderError("submission_unknown", submission_unknown=True)
    monkeypatch.setattr(providers, "GeminiClient", FakeClient)
    monkeypatch.setenv("GEMINI_API_KEY", "test-only")
    with pytest.raises(providers.ProviderError):
        anchors.generate_anchor(anchor_store, "group-a", DEFAULT_CONFIG)
    with pytest.raises(ValueError, match="do not automatically resubmit"):
        anchors.generate_anchor(anchor_store, "group-a", DEFAULT_CONFIG)
    assert len(calls) == 1
