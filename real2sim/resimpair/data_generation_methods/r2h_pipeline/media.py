"""Provider input derivatives retain original time; they never replace labels."""
from pathlib import Path
import subprocess

from .common import file_sha256, probe_video, read_json, write_json


def prepare_seedance_video(source, output):
    source, output = Path(source).resolve(strict=True), Path(output).resolve()
    if source == output:
        raise ValueError("Never overwrite original media")
    before = probe_video(source)
    if not 4 <= before["duration_seconds"] <= 30:
        raise ValueError("Seedance edit requires a complete 4–30s source. No automatic prefix crop or speed-up.")
    metadata_path = output.with_suffix(output.suffix + ".json")
    if output.exists():
        metadata = read_json(metadata_path)
        if (metadata["source"]["sha256"] != before["sha256"] or
                metadata["derived"]["sha256"] != file_sha256(output)):
            raise ValueError("Existing derivative/source hash mismatch")
        return metadata
    output.parent.mkdir(parents=True, exist_ok=True)
    # Smallest side >=768 means square 512^2 source meets the documented pixel
    # floor after resampling. fps duplicates/drops frames on the same timeline.
    result = subprocess.run([
        "ffmpeg", "-v", "error", "-nostdin", "-n", "-i", str(source),
        "-map", "0:v:0", "-an", "-vf",
        r"scale=if(gte(iw\,ih)\,-2\,768):if(gte(iw\,ih)\,768\,-2),fps=24",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output)
    ], capture_output=True, timeout=300)
    if result.returncode:
        output.unlink(missing_ok=True)
        raise ValueError("Provider video derivative failed")
    after = probe_video(output)
    if (abs(after["duration_seconds"] - before["duration_seconds"]) > 1 / 24 + .001
            or after["width"] * after["height"] < 407696
            or output.stat().st_size > 200 * 1024**2):
        output.unlink(missing_ok=True)
        raise ValueError("Provider media limits/time preservation failed")
    metadata = {"schema": "r2h_provider_media_v1", "source": before, "derived": after,
                "timeline_transform": "identity_seconds; resize and 24fps resampling only",
                "source_actions_modified": False, "human_action_ground_truth": False,
                "training_admitted": False}
    write_json(metadata_path, metadata)
    return metadata
