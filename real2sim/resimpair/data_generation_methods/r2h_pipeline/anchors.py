"""One goal-independent human initial image for every shared initial-state group."""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path

from .common import file_sha256, json_sha256, read_json, write_json


ANCHOR_PROMPT = (
    "Edit this initial scene into the INITIAL frame of a human tabletop demonstration. "
    "Replace robot arms/grippers with natural human hands/forearms resting away from the objects. "
    "Do not start any action or imply which object will be selected. Keep all three blocks, their "
    "initial vertical order and identities, supporting platform, brown and blue parking areas, "
    "camera view and spatial relationships unchanged. Keep the objects visible. "
    "No text, arrows, task labels, extra objects, or target-specific hints. "
    "This same frame will be used for all possible task goals."
)


def directory_for(store, group_id):
    return store.path.resolve().parent / "anchors" / json_sha256(group_id)[:20]


def plan_anchor(store, group_id, config):
    rows = [r for r in store.rows() if r["group_id"] == group_id]
    if len(rows) < 2:
        raise ValueError("Shared anchor requires at least two goals in one initial group")
    frames = []
    for row in rows:
        job = store.load_job(row["job_id"])
        first = [f for f in job["media"]["event_frames"] if f["frame_index"] == 0]
        if len(first) != 1 or file_sha256(first[0]["path"]) != first[0]["sha256"]:
            raise ValueError("Missing or changed initial frame")
        frames.append(first[0])
    if len({f["sha256"] for f in frames}) != 1:
        raise ValueError("Goal branches do not share the same decoded initial frame")
    directory = directory_for(store, group_id)
    request = {"schema": "r2h_anchor_request_v1", "group_id": group_id,
               "source_image": {"path": frames[0]["path"], "sha256": frames[0]["sha256"]},
               "prompt": ANCHOR_PROMPT, "model": config["anchor_model"],
               "goal_conditioned": False, "training_admitted": False}
    write_json(directory / "request.json", request)
    return {"group_id": group_id, "request_path": str(directory / "request.json"),
            "goal_conditioned": False, "network_calls_made": False}


def generate_anchor(store, group_id, config):
    from PIL import Image
    from .providers import GeminiClient
    directory = directory_for(store, group_id)
    request = read_json(directory / "request.json")
    if file_sha256(request["source_image"]["path"]) != request["source_image"]["sha256"]:
        raise ValueError("Anchor source changed")
    if request["model"] != config["anchor_model"] or request["prompt"] != ANCHOR_PROMPT:
        raise ValueError("Anchor model or goal-independent prompt changed")
    marker = directory / "call_started.json"
    response_path = directory / "response.json"
    if response_path.exists():
        result = read_json(response_path)
    else:
        key_env = config["gemini"]["api_key_env"]
        if not os.environ.get(key_env):
            raise ValueError("Missing environment variable: " + key_env)
        # An exclusive marker prevents two workers from making the same paid call.
        try:
            with marker.open("x") as stream:
                json.dump({"request_sha256": file_sha256(directory / "request.json"),
                           "state": "started_response_unknown_until_response_saved"}, stream)
        except FileExistsError:
            raise ValueError("Anchor call already started without saved result; do not automatically resubmit") from None
        result = GeminiClient(config["gemini"]).generate_image(
            Path(request["source_image"]["path"]), request["prompt"], model=request["model"])
        write_json(response_path, result)
    images = result.get("images", [])
    if not images:
        raise ValueError("No generated anchor image")
    selected = images[0]
    suffix = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}.get(selected["mime_type"])
    if suffix is None:
        raise ValueError("Unsupported generated image format")
    data = base64.b64decode(selected["data"], validate=True)
    with Image.open(io.BytesIO(data)) as img:
        img.verify()
    image_path = directory / ("human_initial" + suffix)
    if image_path.exists() and image_path.read_bytes() != data:
        raise ValueError("Generated anchor changed")
    if not image_path.exists():
        with image_path.open("xb") as stream:
            stream.write(data)
    provenance = {"schema": "r2h_anchor_provenance_v1", "origin": "actual_image_api",
                  "goal_conditioned": False, "model": request["model"],
                  "request_sha256": file_sha256(directory / "request.json"),
                  "response_sha256": file_sha256(response_path), "image_sha256": file_sha256(image_path),
                  "source_image_sha256": request["source_image"]["sha256"],
                  "quality_status": "requires_initial_scene_and_full_video_review", "training_admitted": False}
    provenance_path = directory / "provenance.json"
    write_json(provenance_path, provenance)
    return store.bind_anchor(group_id, image_path, provenance_path)
