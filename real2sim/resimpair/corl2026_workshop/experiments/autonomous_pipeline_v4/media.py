"""Native-clock frame packets; full decode is distinct from visual sampling."""
from __future__ import annotations

from pathlib import Path
import subprocess
import json

import cv2
import numpy as np

from .core import bind, checked, read, require, write
from .timing import traced


@traced("frame_decode_and_sampling", lambda video, output, frame_budget=49: output, cache_file="packet.json")
def packet(video, output, frame_budget=49):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "packet.json").exists():
        result = read(output / "packet.json")
        require(result["video"] == bind(video), "Frame packet media changed")
        for row in result["frames"]:
            checked(row["image"])
        return result
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "frame=best_effort_timestamp_time", "-of", "json", str(video)],
                           check=True, capture_output=True, text=True, timeout=60)
    times = [float(f["best_effort_timestamp_time"]) for f in json.loads(probe.stdout)["frames"]]
    require(len(times) > 1 and all(b > a for a, b in zip(times, times[1:])), "Missing or nonmonotone native PTS")
    indices = set(np.rint(np.linspace(0, len(times) - 1, min(frame_budget, len(times)))).astype(int).tolist())
    capture = cv2.VideoCapture(str(video))
    require(capture.isOpened(), "Video cannot be opened")
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = []
    count = 0
    shape = None
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if shape is None:
                shape = image.shape
            require(image.shape == shape and image.size > 0, "Invalid decoded image")
            if count in indices:
                scale = min(1., 768 / max(image.shape[:2]))
                resized = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
                path = output / f"frame_{count:06d}.jpg"
                require(not path.exists(), "Partial frame packet exists; recover or use a new output")
                require(cv2.imwrite(str(path), resized, [cv2.IMWRITE_JPEG_QUALITY, 92]), "Frame write failed")
                frames.append({"frame_index": count, "pts_seconds": times[count], "image": bind(path)})
            count += 1
    finally:
        capture.release()
    require(count == len(times), "Decode length differs from native timestamps")
    result = {"schema": "resimpair_frame_packet_v4", "video": bind(video), "frames": frames,
              "decoded_frames": count, "fps": fps, "width": shape[1], "height": shape[0],
              "duration_seconds": times[-1] + 1 / fps, "decode_complete": True,
              "visual_observation_scope": "only the supplied sampled frames", "all_native_pts_seconds": times}
    write(output / "packet.json", result)
    return result
