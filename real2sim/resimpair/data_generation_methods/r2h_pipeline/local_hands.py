"""Optional CPU hand inventory; never a proof of complete hand count or contact.

Run this module with the isolated MediaPipe environment. Importing the module
does not load MediaPipe, download a model, read credentials, or make API calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time


MODEL_SOURCE = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_GUIDE = (
    "https://developers.google.com/edge/mediapipe/solutions/"
    "vision/hand_landmarker"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_image(image_path: str | Path, model_path: str | Path, *,
                  num_hands: int = 6, max_expected_hands: int = 2,
                  min_detection_confidence: float = 0.4,
                  min_presence_confidence: float = 0.4) -> dict:
    """Return detected instances and evidence bounds, with no quality pass.

    ``num_hands`` must exceed the expected count and be at least four. Otherwise
    detector configuration could hide the extra hands this check looks for.
    Landmark extent boxes are *not* an anatomical hand segmentation.
    """
    if type(num_hands) is not int or num_hands < 4:
        raise ValueError("extra-hand inspection requires num_hands >= 4")
    if (type(max_expected_hands) is not int or max_expected_hands < 0
            or num_hands <= max_expected_hands):
        raise ValueError("num_hands must exceed the nonnegative expected maximum")
    for value in (min_detection_confidence, min_presence_confidence):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1:
            raise ValueError("confidence thresholds must be in (0, 1]")
    image_path, model_path = Path(image_path).resolve(), Path(model_path).resolve()
    if not image_path.is_file() or not model_path.is_file():
        raise ValueError("image and model must be existing files")

    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    started = time.perf_counter()
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(model_path),
            delegate=python.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=num_hands,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_presence_confidence,
    )
    image = mp.Image.create_from_file(str(image_path))
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        initialized = time.perf_counter()
        result = landmarker.detect(image)
        finished = time.perf_counter()

    hands = []
    for index, landmarks in enumerate(result.hand_landmarks):
        points = [{"x": float(p.x), "y": float(p.y), "z": float(p.z)}
                  for p in landmarks]
        world_points = [
            {"x": float(p.x), "y": float(p.y), "z": float(p.z)}
            for p in result.hand_world_landmarks[index]
        ]
        xs, ys = [p["x"] for p in points], [p["y"] for p in points]
        hands.append({
            "instance_id": index,
            "landmark_extent_box_normalized_xyxy": [min(xs), min(ys), max(xs), max(ys)],
            "landmarks_image_normalized": points,
            "world_landmarks_hand_centered_meters": world_points,
            "handedness": [
                {"category": c.category_name, "score": float(c.score)}
                for c in result.handedness[index]
            ],
        })
    count = len(hands)
    return {
        "schema": "local_hand_inventory_v1",
        "tool": "mediapipe_hand_landmarker",
        "mediapipe_version": mp.__version__,
        "model": {"path": str(model_path), "sha256": _sha256(model_path),
                  "official_download_url": MODEL_SOURCE, "guide": MODEL_GUIDE},
        "image": {"path": str(image_path), "sha256": _sha256(image_path),
                  "width": image.width, "height": image.height},
        "configuration": {"device": "CPU", "running_mode": "IMAGE",
                          "num_hands": num_hands,
                          "max_expected_hands": max_expected_hands,
                          "min_detection_confidence": min_detection_confidence,
                          "min_presence_confidence": min_presence_confidence},
        "detected_hand_count_lower_bound": count,
        "lower_bound_is_conditional_on_detections_being_true_positive": True,
        "count_is_complete": False,
        "detector_capacity_reached": count == num_hands,
        "extra_hand_candidate": count > max_expected_hands,
        "finding": "possible_extra_hand_requires_confirmation" if count > max_expected_hands else "inconclusive",
        "quality_release": False,
        "training_admitted": False,
        "timing_seconds": {"load_and_initialize": initialized - started,
                           "cpu_detection_wall": finished - initialized,
                           "total_including_close": time.perf_counter() - started},
        "hands": hands,
        "limitations": [
            "Detected instances are candidates, not a complete inventory; false positives remain possible.",
            "No detections does not mean no hands. Occlusion, generated anatomy and cropping can cause misses.",
            "Left/right handedness does not establish a persistent actor identity.",
            "Hand-centered world landmarks are estimates, not camera-frame wrist motion ground truth.",
            "2D landmark proximity does not establish physical hand-object contact.",
            "Combine with actor-specific VLM review, crop inspection and temporal evidence before release.",
        ],
    }


def annotate_inventory(result: dict, output_path: str | Path) -> None:
    """Save a diagnostic overlay while leaving the source image untouched."""
    import cv2
    source = Path(result["image"]["path"])
    if _sha256(source) != result["image"]["sha256"]:
        raise ValueError("source image changed after inspection")
    image = cv2.imread(str(source))
    if image is None:
        raise ValueError("diagnostic source cannot be decoded")
    h, w = image.shape[:2]
    for hand in result["hands"]:
        x0, y0, x1, y1 = hand["landmark_extent_box_normalized_xyxy"]
        p0, p1 = (round(x0*w), round(y0*h)), (round(x1*w), round(y1*h))
        cv2.rectangle(image, p0, p1, (0, 160, 255), 2)
        for point in hand["landmarks_image_normalized"]:
            cv2.circle(image, (round(point["x"]*w), round(point["y"]*h)),
                       2, (255, 200, 0), -1)
        cv2.putText(image, f"candidate {hand['instance_id']}", p0,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 160, 255), 1)
    cv2.putText(image, f"detected={len(result['hands'])}; incomplete count",
                (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    if not cv2.imwrite(str(output_path), image):
        raise ValueError("could not save diagnostic overlay")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-hands", type=int, default=6)
    parser.add_argument("--max-expected-hands", type=int, default=2)
    parser.add_argument("--annotated", type=Path)
    args = parser.parse_args()
    if args.output.exists() or (args.annotated and args.annotated.exists()):
        raise SystemExit("refusing to overwrite existing evidence")
    result = inspect_image(args.image, args.model, num_hands=args.num_hands,
                           max_expected_hands=args.max_expected_hands)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    if args.annotated:
        args.annotated.parent.mkdir(parents=True, exist_ok=True)
        annotate_inventory(result, args.annotated)
    print(json.dumps({k: result[k] for k in [
        "detected_hand_count_lower_bound", "extra_hand_candidate", "finding",
        "timing_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
