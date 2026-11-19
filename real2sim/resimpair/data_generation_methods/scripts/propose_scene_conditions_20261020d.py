"""Prepare offline scene candidates; explicit --review enables one cached VLM call."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_generation_methods.r2h_pipeline.scene_proposals import (
    apply_correction, binding, build_proposals, verify_proposal_bindings)
from data_generation_methods.r2h_pipeline.common import load_env_file, write_json, json_sha256

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "proposal_sha256": {"type": "string"},
        "target_frame_sha256": {"type": "string"},
        "decisions": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "roi_id": {"type": "string"},
                "status": {"type": "string", "enum": ["keep", "correct", "unknown"]},
                "box_xyxy_normalized": {"type": "array", "items": {"type": "number"},
                                       "minItems": 4, "maxItems": 4},
                "reason": {"type": "string"},
            }, "required": ["roi_id", "status", "box_xyxy_normalized", "reason"]}},
    }, "required": ["proposal_sha256", "target_frame_sha256", "decisions"],
}


def review_candidates(output, config, *, repair=False):
    from data_generation_methods.r2h_pipeline.batch_pipeline import paid_stage
    from data_generation_methods.r2h_pipeline.providers import GeminiClient
    output = Path(output)
    proposal = json.loads((output/"proposal.json").read_text())
    verify_proposal_bindings(proposal)
    request = json.loads((output/"vlm_correction_request.json").read_text())
    if not request["required"]:
        return None
    prompt = json.dumps(request, ensure_ascii=False) + (
        "\nReturn JSON matching this schema. Echo the two binding hashes exactly. "
        "For unknown include [0,0,1,1] only as an unused placeholder. All returned "
        "boxes refer to TARGET_GENERATED_INITIAL, never source coordinates.\n") + json.dumps(SCHEMA)
    images = request["images"]
    stage_name = "vlm_correction_v1"
    repair_inputs = {}
    if repair:
        original = output/"vlm_correction_v1/result.json"
        error_path = output/"vlm_correction_v1/validation_error.json"
        if not original.is_file() or not error_path.is_file():
            raise ValueError("Explicit repair needs a preserved response and validation error")
        old = json.loads(original.read_text())
        error = json.loads(error_path.read_text())
        if error["response_sha256"] != json_sha256(old):
            raise ValueError("Validation error does not bind current original response")
        repair_inputs = {"original_response": binding(original),
                         "validation_error": binding(error_path)}
        prompt += (
            "\nThis is one explicit schema/decision repair, using the SAME images. "
            "The original response below failed deterministic validation. Never change raw "
            "evidence. A keep decision is legal ONLY if that target ROI has status candidate "
            "and an existing target box. For an unknown target proposal, choose correct with "
            "a visibly justified target box, or unknown. A source-image box cannot be kept "
            "as though it were an established target-image box. Return the complete corrected "
            "response, with the SAME two binding hashes. Do not infer task success.\n"
            +json.dumps({"original_response":old["review"], "validation_error":error["error"]}))
        stage_name = "vlm_correction_repair_v2"
    def call(transport):
        client = GeminiClient(config["gemini"], transport=transport)
        parts = [{"text": prompt}]
        for item in images:
            parts += [{"text": item["label"]}, client._inline(Path(item["path"]), "image")]
        return client._review_result(client._generate(
            client.model, parts, client._review_config(prompt, SCHEMA, True)), SCHEMA, True)
    result = paid_stage(output/stage_name, {
        "kind": "geometry_roi_proposal_correction",
        "request": binding(output/"vlm_correction_request.json"),
        "images": images, "response_schema": SCHEMA, "prompt": prompt,
        "model": config["gemini"]["model"], **repair_inputs,
    }, call)
    patch = result["review"]
    try:
        corrected = apply_correction(proposal, patch)
    except ValueError as exc:
        write_json(output/stage_name/"validation_error.json", {
            "schema":"r2h_roi_correction_validation_error_v1",
            "error":str(exc), "response_sha256":json_sha256(result),
            "response":binding(output/stage_name/"result.json"),
            "proposal":binding(output/"proposal.json"),
            "automatic_retry":False, "training_admitted":False})
        raise
    suffix = "_repair_v2" if repair else ""
    write_json(output/f"correction_patch{suffix}.json", patch)
    write_json(output/f"corrected_rois{suffix}.json", corrected)
    return corrected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--target-video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review", action="store_true",
                        help="One paid, cached Gemini call; no generation or training")
    parser.add_argument("--repair-review", action="store_true",
                        help="One explicit versioned correction repair after preserved validation failure")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[1]/"config.echoflow.example.json")
    parser.add_argument("--credentials", type=Path,
                        default=Path("/home/xzzhou/.config/policygen/credentials/echoflow.env"))
    args = parser.parse_args()
    if args.input:
        proposal = build_proposals(args.input, args.output, target_video=args.target_video)
        print(json.dumps({
            "output": str(args.output),
            "hidden_objects": sum(x["hidden_then_revealed"] for x in proposal["hidden_state_proposals"]),
            "keyframes": len(proposal["source_keyframes"]),
            "source_rois": len(proposal["source_roi_candidates"]),
            "registration_status": proposal["target"]["registration"]["status"] if proposal["target"] else None,
        }))
    if args.review and args.repair_review:
        parser.error("Choose --review OR --repair-review")
    if args.review or args.repair_review:
        load_env_file(args.credentials)
        result = review_candidates(args.output, json.loads(args.config.read_text()), repair=args.repair_review)
        print(json.dumps({"review_finished": result is not None, "output": str(args.output)}))
    if not args.input and not args.review and not args.repair_review:
        parser.error("Provide --input and/or --review")


if __name__ == "__main__":
    main()
