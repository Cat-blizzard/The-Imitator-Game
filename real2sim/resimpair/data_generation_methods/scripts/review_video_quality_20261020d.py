"""Prepare/run one bounded, independent ordinal-quality Gemini request."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_generation_methods.r2h_pipeline.quality_model_review import prepare_review, review_video

CABINET_CONTRACT = {
    "initial": "The image-right cabinet door is closed.",
    "necessary_steps": ["Approach and manipulate the right-door handle",
                        "Open the same right door", "Release the handle and withdraw"],
    "goal": "The image-right door remains open after the hand withdraws.",
    "invariants": ["The image-left door retains its initial state",
                   "Preserve visible task objects and scene identity"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text()) if args.contract else CABINET_CONTRACT
    result = (review_video(args.video, args.output, contract) if args.execute
              else prepare_review(args.video, args.output, contract))
    print(json.dumps({k: result[k] for k in
                      ("frame_count", "ordinal_vector", "status", "hard_visible_violations",
                       "training_admitted") if k in result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
