"""Additive visible-mechanism checks; existing actor/identity/hand gates stay intact."""
from pathlib import Path

from . import hand_review, review
from .core import bind, implementation, read, require, write
from .hand_correspondence import review_requirements
from .media import packet
from .models import reviewed_json
from .prompts import review_prompt

VERSION = "native_mechanism_visual_review_v1"


def checks(contract):
    result = review.video_checks(contract)
    info = contract["generation_information"]
    result["functional_geometry"] = (
        "All native functional shapes and relative sizes survive: tool heads, handles, housing, gaps, "
        "container wall pieces and loose solid beads. Do not substitute a familiar bowl, spoon, switch "
        "or liquid just because the task has that name. Judge candidate pixels against inventory.")
    if info.get("required_cyclic_motion"):
        result["cyclic_motion"] = (
            "Follow the functional tool tip through the ordered required_spatial_targets. Verify the "
            "declared direction_in_image and exactly the declared number of full turns. Cite intermediate "
            "positions around the loop and start/end; identical endpoints alone cannot establish a turn. "
            "If sampled frames do not distinguish the direction or count, mark unknown.")
    if any(e.get("joint_state_evidence") for e in info["required_events"]):
        result["joint_motion_between_endpoints"] = (
            "Visible intermediate joint motion agrees with the declared displacement/return events. "
            "Touching or pointing at a spring control without discernible depression and rebound is "
            "insufficient; unchanged start/end images alone do not prove actuation. Unknown if too small or hidden.")
    return result


def video(root, budget, directory, contract, *, wire_schema_adapter=None):
    from .review_transport import WIRE_SCHEMA_ADAPTER
    require(wire_schema_adapter in {None, WIRE_SCHEMA_ADAPTER}, 'unsupported_wire_schema_adapter')
    root, directory = Path(root), Path(directory)
    output, path = directory / "review.json", directory / "human.mp4"
    if output.exists():
        result = read(output)
        require(result["candidate"] == bind(path) and result["rules"] == VERSION, "candidate_review_binding_changed")
        return result
    info = contract["generation_information"]
    hands = review_requirements(info)
    frames = packet(path, directory / "review_frames", read(root / "experiment.json")["review_frame_budget"])
    supplied = [{k: f[k] for k in ["frame_index", "pts_seconds"]} for f in frames["frames"]]
    targets = [e["id"] for e in info["inventory"] if e["role"] == "target"]
    required = checks(contract)
    request = {"schema": "native_mechanism_visual_request_v1", "implementation": implementation(),
               "model_config": bind(root / "config.json"), "candidate": bind(path),
               "prompt": review_prompt(required, info, supplied, "generated_human_video")
                   + review.census_instruction(targets) + hand_review.instruction(hands) + review.WIRE_INSTRUCTION,
               "response_schema": review.wire_schema(census=True, hands=bool(hands)),
               "acceptance_schema": review.schema(required, [f["frame_index"] for f in supplied], targets, hands),
               "frames": bind(directory / "review_frames/packet.json"), "rules": VERSION,
               "images": [{"label": f"CANDIDATE native frame {f['frame_index']} PTS {f['pts_seconds']:.9f} seconds", "image": f["image"]}
                          for f in frames["frames"]]}
    if wire_schema_adapter is not None:
        request['wire_schema_adapter'] = wire_schema_adapter
    write(directory / "review_request.json", request)
    observations, evidence = reviewed_json(root, budget, directory / "review_request.json")
    observations = review.normalize_wire(observations, required, targets, hands)
    gate = review.validate(observations, required, [f["frame_index"] for f in supplied],
                           contract["measured"]["required_event_precedence"], targets, hands)
    result = {"rules": VERSION, "candidate": bind(path), "packet": bind(directory / "review_frames/packet.json"),
              "observations": observations, "model_evidence": evidence, "gate": gate,
              "automatic_visual_inference": True, "manual_observation_count": 0,
              "training_admitted": False, "all_frame_guarantee": False}
    write(output, result)
    return result
