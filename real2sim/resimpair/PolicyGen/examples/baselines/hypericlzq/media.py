"""Read original videos without unpacking whole datasets. Archive offsets are read-only."""
from __future__ import annotations
import io
import zipfile
from functools import lru_cache
import av
import numpy as np


@lru_cache(maxsize=4)
def _zip(path):
    return zipfile.ZipFile(path)


def video_bytes(source):
    if source['kind'] == 'zip':
        return _zip(source['path']).read(source['member'])
    if source['kind'] == 'tar':
        with open(source['path'], 'rb') as f:
            f.seek(source['offset'])
            data = f.read(source['size'])
        if len(data) != source['size']:
            raise IOError('Incomplete archive member')
        return data
    return source['path']


def frames_at(source, times, size=224):
    """Seek independently at exact presentation timestamps; no future frame substitution."""
    data = video_bytes(source)
    result = []
    with av.open(io.BytesIO(data) if isinstance(data, bytes) else data) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'SLICE'
        stream.codec_context.thread_count = 1
        fps = float(stream.average_rate or 30)
        for seconds in times:
            seconds = max(0., float(seconds))
            # MP4 seeking uses decoding timestamps: with B-frames an exact PTS
            # seek can start one frame AFTER the requested observation. Decode
            # from an earlier keyframe and select by presentation time instead.
            container.seek(int(max(0.,seconds-1.) / stream.time_base), stream=stream, backward=True)
            chosen = None
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                stamp = float(frame.pts * stream.time_base)
                if stamp >= seconds - .51 / fps:
                    chosen = frame
                    break
            if chosen is None:
                raise ValueError(f'No frame at {seconds:.3f}s in {source}')
            result.append(chosen.reformat(width=size, height=size).to_ndarray(format='rgb24'))
    return np.stack(result)


def inspect_video(source, start, length, fps):
    data = video_bytes(source)
    with av.open(io.BytesIO(data) if isinstance(data, bytes) else data) as c:
        s = c.streams.video[0]
        duration = float(s.duration * s.time_base) if s.duration is not None else c.duration / 1e6
        meta = dict(width=s.width, height=s.height, fps=float(s.average_rate or fps), duration=duration)
    if start + (length-1)/fps > duration + 2/fps:
        raise ValueError('Video shorter than trajectory')
    x = frames_at(source, start + np.array([0, (length-1)//2, length-1])/fps, 96)
    meta.update(sampled_frames=3, pixel_std=float(x.std()), mean_brightness=float(x.mean()),
                temporal_change=float(np.abs(np.diff(x.astype(float), axis=0)).mean()))
    if meta['pixel_std'] < 5 or meta['mean_brightness'] < 4:
        raise ValueError('Nearly blank video')
    return meta, x
