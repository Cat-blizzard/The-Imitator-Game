"""Independent byte/ROI arithmetic audit of the frozen cabinet diagnostic.

Does not import the candidate, alter its thresholds, or approve object identity.
Direct review of four final-frame overlays is separately identified as review.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import av
import cv2
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(binding):
    path = Path(binding["path"])
    assert sha(path) == binding["sha256"], str(path)
    return path


def extent(box, error):
    width, height = box[2] - box[0], box[3] - box[1]
    assert width > 2 * error and height > 2 * error
    return {"width": (width - 2 * error, width + 2 * error),
            "height": (height - 2 * error, height + 2 * error),
            "area": ((width - 2 * error) * (height - 2 * error),
                     (width + 2 * error) * (height + 2 * error))}


def run(study, output):
    output = Path(output)
    assert not output.exists(), "Use a new independent report"
    study = Path(study)
    report = json.loads(study.read_text())
    protocol = json.loads(verify(report["protocol"]).read_text())
    for key in ("candidate_code", "study_code", "source_proposal"):
        verify(report[key])
    frame_groups, rows = defaultdict(dict), []
    for item in report["cases"]:
        diagnostic = json.loads(verify(item["result"]).read_text())
        annotation = diagnostic["input_annotation"]
        assert diagnostic["identity_pass"] is False
        assert diagnostic["protocol"] == protocol
        metrics = {}
        for context in protocol["contexts"]:
            ranges = {}
            for label in ("source", "target"):
                side = annotation[label]
                frame = side["image"]
                verify(frame)
                verify(frame["video"])
                frame_groups[frame["video"]["path"]][frame["frame_index_0based"]] = frame
                error = protocol["roi_boundary_uncertainty_px"][label]
                obj = extent(side["object_box_xyxy_px"], error)
                ref = extent(side["contexts_xyxy_px"][context], error)
                ranges[label] = {k: [obj[k][0] / ref[k][1], obj[k][1] / ref[k][0]] for k in obj}
            factors = {k: [ranges["target"][k][0] / ranges["source"][k][1],
                           ranges["target"][k][1] / ranges["source"][k][0]] for k in ranges["source"]}
            for dimension, interval in factors.items():
                recorded = diagnostic["metrics"][context]["target_to_source_relative_extent_factor_intervals"][dimension]
                np.testing.assert_allclose(interval, recorded, rtol=1e-14, atol=0)
            metrics[context] = factors
        flag = any(bounds[0] > protocol["severe_flags"][f"relative_{dimension}_factor_lower_bound_gt"]
                   for factors in metrics.values() for dimension, bounds in factors.items())
        expected = "strong_extent_inconsistency" if flag else "unknown_no_severe_extent_flag"
        assert item["status"] == diagnostic["status"] == expected
        rows.append({"case": item["case"], "frame_index": item["frame_index"], "status": expected,
                     "independent_interval_match": True, "identity_pass": False})
    decoded = []
    for path, frames in frame_groups.items():
        matched = []
        with av.open(path) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index not in frames:
                    continue
                binding = frames[index]
                image = frame.to_ndarray(format="rgb24")
                assert hashlib.sha256(image.tobytes()).hexdigest() == binding["decoded_RGB_sha256"]
                stored = cv2.cvtColor(cv2.imread(binding["path"]), cv2.COLOR_BGR2RGB)
                np.testing.assert_array_equal(image, stored)
                assert abs(float(frame.pts * frame.time_base) - binding["timestamp_seconds"]) < 1e-8
                matched.append(index)
        assert set(matched) == set(frames)
        decoded.append({"video": path, "verified_frame_indices": matched})
    result = {"schema": "independent_hidden_extent_arithmetic_audit_v1", "study": str(study),
              "study_sha256": sha(study), "auditor_sha256": sha(__file__), "rows": rows,
              "independent_decode": decoded, "unique_images_redecoded": sum(len(x) for x in frame_groups.values()),
              "manual_root_overlay_review": {
                  "frames": ["source_498", "cabinet_A_original_192", "cabinet_A_redraw_192", "cabinet_B_192"],
                  "observation": "Both A outputs show a tall grid in the upper cavity; B shows small cube-like contents. This is a known-case visual review, not automatic identity acceptance."},
              "retained_miss": "A_redraw is visually wrong but the fixed conservative extent bound returns unknown.",
              "normal_case_false_positive_rate": None, "identity_passes": 0, "api_calls": 0, "gpu_runs": 0}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.study, args.output)
    print(json.dumps({"verified_observations": len(result["rows"]),
                      "unique_images_redecoded": result["unique_images_redecoded"], "api_calls": 0}))
