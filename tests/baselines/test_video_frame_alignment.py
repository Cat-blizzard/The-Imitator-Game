"""Frame-index accuracy and decoder-resource recovery without GPU dependencies."""
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import torch
import av
import numpy as np

from imitator.dataset import video_utils as video


@pytest.fixture
def decoder(monkeypatch):
    instances = []

    class IndexedDecoder:
        """Model the observed approximate-seek one-frame offset, including PTS."""
        def __init__(self, source, seek_mode):
            self.metadata = SimpleNamespace(average_fps=20.)
            self.seek_mode = seek_mode
            self.source = source
            instances.append(self)

        def get_frames_at(self, indices):
            actual = torch.tensor(indices) + (1 if self.seek_mode == "approximate" else 0)
            return SimpleNamespace(
                data=actual[:, None, None, None].expand(-1, 3, 4, 5).to(torch.uint8),
                pts_seconds=actual.double() / 20.)

    module = ModuleType("torchcodec.decoders")
    module.VideoDecoder = IndexedDecoder
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", module)
    return IndexedDecoder, instances


def test_exact_indexed_read_returns_requested_frame_not_the_adjacent_frame(decoder, tmp_path):
    constructor, instances = decoder
    cache = video.VideoDecoderCache()
    path = tmp_path / "frames.mp4"
    # Demonstrate the failure being modeled, then require the public reader to
    # return the intended frame values under the existing sub-frame tolerance.
    inaccurate = constructor(str(path), seek_mode="approximate").get_frames_at([4, 7])
    torch.testing.assert_close(inaccurate.pts_seconds, torch.tensor([.25, .4], dtype=torch.float64))
    try:
        actual = video.decode_video_frames_torchcodec(path, [.2, .35], .001, decoder_cache=cache)
        expected = torch.tensor([4., 7.])[:, None, None, None].expand(2, 3, 4, 5) / 255.
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    finally:
        cache.clear()


def test_stale_cached_response_is_rechecked_and_recovered_without_accepting_wrong_frame(decoder, tmp_path):
    constructor, _ = decoder
    cache = video.VideoDecoderCache()
    path = str(tmp_path / "frames.mp4")
    resource = SimpleNamespace(closed=False)
    resource.close = lambda: setattr(resource, "closed", True)
    cache._cache[path] = (constructor(path, seek_mode="approximate"), resource)
    actual = video.decode_video_frames_torchcodec(path, [.2], .001, decoder_cache=cache)
    torch.testing.assert_close(actual, torch.full((1, 3, 4, 5), 4. / 255.), atol=0, rtol=0)
    assert resource.closed
    cache.clear()


def test_legacy_file_handle_source_retains_exact_frame_alignment(decoder, monkeypatch, tmp_path):
    constructor, instances = decoder

    def require_handle(source, seek_mode):
        if isinstance(source, str):
            raise TypeError("unsupported source type")
        return constructor(source, seek_mode)

    monkeypatch.setattr(sys.modules["torchcodec.decoders"], "VideoDecoder", require_handle)
    path = tmp_path / "frames.mp4"
    path.write_bytes(b"synthetic frame decoder fixture")
    cache = video.VideoDecoderCache()
    try:
        actual = video.decode_video_frames_torchcodec(path, [.3], .001, decoder_cache=cache)
        torch.testing.assert_close(actual, torch.full((1, 3, 4, 5), 6. / 255.), atol=0, rtol=0)
        handle = instances[-1].source
        assert not handle.closed
    finally:
        cache.clear()
    assert handle.closed


def test_real_lossless_video_returns_requested_frame_pixels(tmp_path):
    """Read actual encoded bytes through torchcodec, including both end frames."""
    path = tmp_path / "indexed.mp4"
    frames = np.stack([np.full((32, 48, 3), value, dtype=np.uint8)
                       for value in (0, 32, 64, 96, 128, 160, 192, 224)])
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264rgb", rate=20)
        stream.width, stream.height = 48, 32
        stream.pix_fmt = "rgb24"
        stream.options = {"crf": "0", "preset": "ultrafast", "g": "4"}
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    indices = [0, 3, 4, 6, 7]
    cache = video.VideoDecoderCache()
    try:
        actual = video.decode_video_frames_torchcodec(
            path, [index / 20. for index in indices], .001, decoder_cache=cache)
    finally:
        cache.clear()
    expected = torch.from_numpy(frames[indices].transpose(0, 3, 1, 2)).float() / 255.
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
