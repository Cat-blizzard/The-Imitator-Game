"""Reviewed 2D hidden-content extent diagnostic; never an identity pass.

The automatic layer reuses scene registration. Semantic-slot correspondence
and target object/context ROIs are explicit reviewed inputs. An enlarged
visible extent can contradict preservation; a similar extent cannot prove it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .common import file_sha256, write_json
from .scene_proposals import binding, checked_path, register_images, validate_box


SCHEMA = "hidden_identity_extent_diagnostic_v1"


def _image(side):
    evidence = side["image"]
    if not isinstance(evidence, dict) or len(evidence.get("sha256", "")) != 64:
        raise ValueError("Image must be hash bound")
    path = checked_path(evidence)
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError("Cannot decode bound image")
    return image


def _pixel_box(box, width, height):
    a = np.asarray(box, float)
    if a.shape != (4,):
        raise ValueError("Pixel box must contain four coordinates")
    validate_box(a / [width, height, width, height])
    return a


def dimension_intervals(box, error_px):
    if not np.isfinite(error_px) or error_px < 0:
        raise ValueError("Invalid ROI boundary error")
    w, h = np.asarray(box)[2:] - np.asarray(box)[:2]
    low = np.array([w, h]) - 2 * error_px
    high = np.array([w, h]) + 2 * error_px
    if np.any(low <= 0):
        raise ValueError("ROI unresolved at its declared boundary uncertainty")
    return {"width": [float(low[0]), float(high[0])],
            "height": [float(low[1]), float(high[1])],
            "area": [float(np.prod(low)), float(np.prod(high))],
            "aspect": [float(low[0] / high[1]), float(high[0] / low[1])]}


def _divide(numerator, denominator):
    return [float(numerator[0] / denominator[1]), float(numerator[1] / denominator[0])]


def relative_measurements(obj, context, error_px):
    obj, context = np.asarray(obj), np.asarray(context)
    if np.any(obj[:2] < context[:2]) or np.any(obj[2:] > context[2:]):
        raise ValueError("Object ROI is outside its reviewed context; possible ROI exchange")
    a, b = dimension_intervals(obj, error_px), dimension_intervals(context, error_px)
    result = {k: _divide(a[k], b[k]) for k in ("width", "height", "area")}
    result["object_aspect_interval"] = a["aspect"]
    center = (obj[:2] + obj[2:]) / 2
    result["center_in_context_xy"] = ((center - context[:2]) / (context[2:] - context[:2])).tolist()
    result["center_in_context_intervals_xy"] = [
        [(float(center[i] - context[i]) - 2 * error_px) / b[k][1],
         (float(center[i] - context[i]) + 2 * error_px) / b[k][0]]
        for i, k in enumerate(("width", "height"))]
    return result


def reviewed_correspondence_status(review, source, target):
    if review.get("status") != "reviewed_semantic_slot" or not review.get("reviewer") or not review.get("evidence"):
        return "Semantic-slot correspondence has not been explicitly reviewed"
    if (review.get("source_image_sha256") != source["image"]["sha256"]
            or review.get("target_image_sha256") != target["image"]["sha256"]):
        return "Correspondence review belongs to different images"
    if not review.get("source_slot") or review.get("source_slot") != review.get("target_slot"):
        return "Source and target semantic slots disagree"
    if review.get("same_context_instance") is not True or review.get("comparable_view") is not True:
        return "Context identity or comparable viewpoint is unresolved"
    if source.get("object_visibility") != "fully_visible":
        return "Source complete object extent is unresolved"
    if target.get("object_visibility") not in {"fully_visible", "visible_extent_lower_bound"}:
        return "Target object is occluded or extent is unresolved"
    return None


def _source_geometry_check(source, box):
    evidence = source.get("geometry_evidence")
    if evidence is None:
        return
    path = checked_path(evidence["visibility"])
    with np.load(path, allow_pickle=False) as qa:
        index, obj = evidence["frame_index"], evidence["object_index"]
        if qa["object_pixels_512"][index, obj] <= 0:
            raise ValueError("Source projected object is not actually visible")
        expected = qa["projected_bounds_224"][index, obj] / 224 * source["image"]["width"]
    np.testing.assert_allclose(box, expected, atol=1e-6, rtol=0,
                               err_msg="Source object ROI differs from bound simulator geometry")


def diagnose_pair(annotation, protocol, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": SCHEMA, "sample_id": annotation.get("sample_id"),
        "status": "unknown", "identity_preserved": "unknown", "identity_pass": False,
        "automatic_identity_detection": False, "training_admitted": False,
        "new_api_calls": 0, "input_annotation": annotation,
        "protocol": protocol, "source_code": binding(__file__),
        "manual_interventions_required": ["target object/context ROI localization", "semantic-slot/view correspondence review"],
    }
    try:
        source, target = annotation["source"], annotation["target"]
        source_image, target_image = _image(source), _image(target)
        registration = register_images(source_image, target_image)
        # This can succeed on the cabinet while failing to establish the cube.
        registration["semantic_object_correspondence"] = "unknown"
        registration["identity_pass"] = False
        write_json(output / "automatic_registration.json", registration)
        result["automatic_registration"] = binding(output / "automatic_registration.json")
        reason = reviewed_correspondence_status(annotation["correspondence"], source, target)
        if reason:
            result["reason"] = reason
        else:
            boxes = {}
            for label, side, image in (("source", source, source_image), ("target", target, target_image)):
                height, width = image.shape[:2]
                if (side["image"]["width"], side["image"]["height"]) != (width, height):
                    raise ValueError("ROI dimensions differ from bound image")
                obj = _pixel_box(side["object_box_xyxy_px"], width, height)
                if label == "source":
                    _source_geometry_check(source, obj)
                contexts = {k: _pixel_box(v, width, height) for k, v in side["contexts_xyxy_px"].items()}
                if not contexts:
                    raise ValueError("A reviewed reference context is required")
                boxes[label] = (obj, contexts)
            if set(boxes["source"][1]) != set(boxes["target"][1]):
                raise ValueError("Source and target reference contexts differ")
            metrics, flags = {}, []
            thresholds = protocol["severe_flags"]
            for context in boxes["source"][1]:
                measured = {}
                for label in ("source", "target"):
                    obj, contexts = boxes[label]
                    measured[label] = relative_measurements(
                        obj, contexts[context], protocol["roi_boundary_uncertainty_px"][label])
                factors = {k: _divide(measured["target"][k], measured["source"][k])
                           for k in ("width", "height", "area")}
                measured["target_to_source_relative_extent_factor_intervals"] = factors
                measured["aspect_factor_interval"] = _divide(
                    measured["target"]["object_aspect_interval"], measured["source"]["object_aspect_interval"])
                for dimension in ("width", "height", "area"):
                    threshold = thresholds[f"relative_{dimension}_factor_lower_bound_gt"]
                    if factors[dimension][0] > threshold:
                        flags.append({"context": context, "dimension": dimension,
                                      "factor_lower_bound": factors[dimension][0], "threshold": threshold})
                metrics[context] = measured
            result.update(
                status="strong_extent_inconsistency" if flags else "unknown_no_severe_extent_flag",
                metrics=metrics, strong_flags=flags,
                reason=("Reviewed visible extent exceeds the fixed diagnostic bound even after ROI uncertainty"
                        if flags else "Coarse size diagnostic did not establish a severe mismatch; identity remains unknown"),
                measurement_scope="2D visible extent relative to reviewed image context; not 3D scale or semantic identity",
                target_extent_scope=target["object_visibility"],
                limitations=[
                    "Semantic-slot correspondence and target ROIs were manually reviewed, not automatically detected.",
                    "A similar extent, matching colors or successful background registration never awards identity pass.",
                    "Context deformation, viewpoint/depth change and occlusion can invalidate ratios; review them separately.",
                    "Fixed coarse thresholds are engineering diagnostics, not calibrated acceptance standards.",
                ])
            for label, image in (("source", source_image), ("target", target_image)):
                overlay = image.copy()
                obj, contexts = boxes[label]
                for name, box in [("object", obj), *contexts.items()]:
                    a, b, c, d = [int(round(x)) for x in box]
                    color = (0, 0, 255) if name == "object" else (255, 220, 0)
                    cv2.rectangle(overlay, (a, b), (c, d), color, 1)
                    cv2.putText(overlay, name, (a, max(12, b - 4)), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
                cv2.imwrite(str(output / f"{label}_reviewed_rois.png"), overlay)
    except (ValueError, KeyError, AssertionError, OSError) as error:
        result.update(status="unknown_invalid_evidence", reason=str(error), error_type=type(error).__name__)
    write_json(output / "diagnostic.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = diagnose_pair(json.loads(args.annotation.read_text()), json.loads(args.protocol.read_text()), args.output)
    print(json.dumps({k: result[k] for k in ("schema", "sample_id", "status", "identity_pass")}))


if __name__ == "__main__":
    main()
