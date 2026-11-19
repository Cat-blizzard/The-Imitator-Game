"""Small filesystem helpers; never evaluate shell text or print credentials."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value, *, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    if path.exists() and not overwrite:
        if path.read_bytes() == data:
            return
        raise ValueError("Artifact already exists with different contents: " + str(path))
    fd, name = tempfile.mkstemp(prefix=".r2h-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(name, 0o600)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def probe_video(path, *, decode=True):
    """Validate actual media, without treating it as task-quality approval."""
    path = Path(path).resolve(strict=True)
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames:format=duration", "-of", "json", str(path)
    ], capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise ValueError("Media probe failed")
    parsed = json.loads(result.stdout)
    if not parsed.get("streams"):
        raise ValueError("No video stream")
    video = parsed["streams"][0]
    duration = float(parsed["format"]["duration"])
    num, den = video["avg_frame_rate"].split("/")
    fps = float(num) / float(den)
    if not all(math.isfinite(v) and v > 0 for v in (duration, fps, video["width"], video["height"])):
        raise ValueError("Invalid media dimensions or timebase")
    if decode:
        result = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(path),
                                 "-map", "0:v:0", "-f", "null", "-"],
                                capture_output=True, timeout=300)
        if result.returncode:
            raise ValueError("Full media decode failed")
    return {"path": str(path), "sha256": file_sha256(path), "duration_seconds": duration,
            "fps": fps, "width": video["width"], "height": video["height"],
            "frames": int(video["nb_frames"]) if str(video.get("nb_frames", "")).isdigit() else None,
            "decode_passed": bool(decode), "full_decode_passed": bool(decode)}


def load_env_file(path):
    """Load literal NAME=value entries, without interpolation or eval."""
    if path is None:
        return
    import re
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError("Invalid credentials-file entry")
        name, value = line.split("=", 1)
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name.strip()):
            raise ValueError("Invalid credentials-file variable name")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not os.environ.get(name.strip()):
            os.environ[name.strip()] = value
