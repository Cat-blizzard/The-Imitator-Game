"""Blind human-video observations -> an evidence-bound draft, never approval.

No task captions, historical contracts, simulator states or filenames enter
the extraction prompt. The video window itself is still an externally supplied
candidate; this module does not claim to have retrieved/discovered that window.
"""
from __future__ import annotations
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from .common import file_sha256, json_sha256, probe_video, write_json
from .scene_proposals import binding, checked_path, frame_at


def obj(properties):
    return {"type": "object", "additionalProperties": False,
            "properties": properties, "required": list(properties)}


TEXT = {"type": "string"}
EVIDENCE = {"type":"array", "items":obj({
    "frame_index": {"type":"integer", "minimum":0},
    "pts_seconds": {"type":"number", "minimum":0},
    "visible_detail": TEXT})}
RELATION = obj({"subject":TEXT, "predicate":TEXT, "object":TEXT,
                "observation":{"type":"string","enum":["observed","unknown"]},
                "description":TEXT, "evidence":EVIDENCE})
SCHEMA = obj({
    "entities":{"type":"array", "items":obj({
        "id":TEXT, "kind":{"type":"string","enum":["hand","person","object","container",
                                                   "mechanism","support","unknown"]},
        "visual_description":TEXT, "identity_evidence":EVIDENCE})},
    "initial_relations":{"type":"array","items":RELATION},
    "final_relations":{"type":"array","items":RELATION},
    "events":{"type":"array","items":obj({
        "id":TEXT, "verb":TEXT, "entity_ids":{"type":"array","items":TEXT},
        "observation":{"type":"string","enum":["observed","unknown"]},
        "temporal_scope":{"type":"string","enum":["visible_milestone","sampled_transition","unknown"]},
        "description":TEXT, "evidence":EVIDENCE})},
    "observed_order":{"type":"array","items":obj({
        "before_event":TEXT, "after_event":TEXT, "reason":TEXT})},
    "window_assessment":obj({
        "scope":{"type":"string","enum":["complete_atomic_candidate","partial_action",
                                         "multiple_actions","unknown"]},
        "begins_before_relevant_action":{"type":"string","enum":["yes","no","unknown"]},
        "ends_after_visible_release_or_settle":{"type":"string","enum":["yes","no","unknown"]},
        "boundary_evidence":EVIDENCE,
        "possible_missing_prefix_or_suffix":TEXT}),
    "hand_roles":{"type":"array","items":obj({
        "entity_id":TEXT,
        "role":{"type":"string","enum":["active_manipulator","support_or_stabilize",
                                        "resting","uncertain"]},
        "description":TEXT, "evidence":EVIDENCE})},
    "hidden_state_observations":{"type":"array","items":obj({
        "entity_id":TEXT,
        "initial_visibility":{"type":"string","enum":["visible","occluded_candidate","unknown"]},
        "later_visibility":{"type":"string","enum":["visible","unknown"]},
        "description":TEXT, "evidence":EVIDENCE})},
    "dense_review_requests":{"type":"array","items":obj({
        "first_supplied_frame":{"type":"integer","minimum":0},
        "last_supplied_frame":{"type":"integer","minimum":0},
        "reason":TEXT})},
    "uncertainties":{"type":"array","items":TEXT},
})


def extraction_prompt(packet):
    public_frames = [{"frame_index": f["frame_index_0based"], "pts_seconds": f["pts_seconds"]}
                     for f in packet["frames"]]
    return (
        "Describe ONLY the visible human manipulation in the supplied chronological frames. "
        "No desired task label or outcome is provided. This is a draft observation proposal, "
        "not task-success assessment. Ignore filenames, latent prior datasets, and what a "
        "typical activity normally includes. Give persistent entity IDs from their first "
        "visible appearance; do not rename moving objects by their later position. "
        "Describe initial/final visible relations, witnessed actions and observed order. "
        "A supplied video window may be an incomplete subtask, may begin mid-action or end "
        "before release; identify these cases. Do not silently call an atomic subaction a "
        "complete compound task. Do not infer that a sequence seen once is causally necessary. "
        "Inventory both active and static support/resting hands; static does not mean useless. "
        "A hand near a handle does not prove grasp or force. An apparently empty/closed region "
        "can conceal objects; initial occlusion stays unknown unless the images show evidence. "
        "If later contents become visible, describe that observation with the actual later "
        "frame, without inventing an initial position, shape, material or physics. "
        "Cite ONLY supplied native frame_index and exact pts_seconds for every observed claim. "
        "Initial relations require evidence from the first 20% of this candidate window; final "
        "relations require the last 20%. Unseen contact or motion between sparse frames remains "
        "unknown. Request denser frames between supplied endpoints if needed. "
        "A sampled_transition means a visible state difference across observed frames, not "
        "a certified continuous trajectory. In relation.object use an entity ID, or an empty "
        "string for unary relations. observed_order edges must be acyclic and supported by "
        "the cited event timing; do not order overlapping ambiguous events. "
        "An empty array/unknown is preferred to guessing. No human 3D trajectory, contact force, "
        "necessary causal precedence, sim equivalence, or training suitability is certified. "
        "Return ONLY JSON matching the schema.\n"
        + json.dumps({"candidate_window_frames":packet["window_frames"],
                      "interval":"start_inclusive_end_exclusive",
                      "source_native_frame_count":packet["video"]["frames"],
                      "supplied_frames":public_frames,
                      "response_schema":SCHEMA}, ensure_ascii=False)
    )


def prepare_packet(video, output, *, start=0, end=None, frame_budget=24):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    media = probe_video(video)
    if not 8 <= frame_budget <= 48:
        raise ValueError("frame budget must be 8..48")
    raw = subprocess.run(["ffprobe","-v","error","-select_streams","v:0",
                          "-show_entries","frame=best_effort_timestamp_time","-of","json",str(video)],
                         check=True,capture_output=True,text=True)
    timestamps = [float(x["best_effort_timestamp_time"]) for x in json.loads(raw.stdout)["frames"]]
    if len(timestamps) != media["frames"] or any(b <= a for a,b in zip(timestamps,timestamps[1:])):
        raise ValueError("Unsupported missing/nonmonotonic native timestamps")
    end = len(timestamps) if end is None else end
    if not 0 <= start < end <= len(timestamps):
        raise ValueError("Invalid native candidate window")
    indices = sorted(set(np.rint(np.linspace(start,end-1,min(frame_budget,end-start))).astype(int).tolist()))
    frames = []
    for index in indices:
        image, frame = frame_at(video,index,output/f"frame{index:06d}.png")
        frame["pts_seconds"] = timestamps[index]
        scale=min(1.,1024/max(image.shape[:2]))
        view=cv2.resize(image,(round(image.shape[1]*scale),round(image.shape[0]*scale)))
        view_path=output/f"review_frame{index:06d}.jpg"
        cv2.imwrite(str(view_path),view,[cv2.IMWRITE_JPEG_QUALITY,92])
        frame["review_image"]={**binding(view_path),"width":view.shape[1],"height":view.shape[0],
                               "transform":"aspect-preserving resize max_dimension1024, JPEG92"}
        frames.append(frame)
    packet = {
        "schema":"r2h_blind_human_task_packet_v1", "video":media,
        "window_frames":[start,end], "interval":"start_inclusive_end_exclusive",
        "frames":frames, "frame_budget":frame_budget,
        "input_scope":"externally_selected_real_human_candidate_window",
        "window_was_automatically_discovered":False,
        "historical_task_text_supplied":False, "simulator_evidence_supplied":False,
        "source_type":"real_human", "training_admitted":False,
    }
    write_json(output/"packet.json",packet)
    write_json(output/"request.json",{
        "schema":"r2h_blind_human_task_request_v1","packet":binding(output/"packet.json"),
        "prompt":extraction_prompt(packet),"response_schema":SCHEMA,
        "images":[{"label":f"native frame {f['frame_index_0based']} | PTS {f['pts_seconds']:.9f} s",
                   "path":f["review_image"]["path"],"sha256":f["review_image"]["sha256"]} for f in frames],
        "no_network_called":True,
    })
    return packet


def validate_observations(response, packet):
    """Validate provenance/timing/identity, not the truth of VLM observations."""
    from jsonschema import validate
    validate(response,SCHEMA)
    supplied = {f["frame_index_0based"]: f["pts_seconds"] for f in packet["frames"]}
    start,end = packet["window_frames"]
    def evidence(items, *, required=False, phase=None):
        if required and not items:
            raise ValueError("Observed claim lacks frame evidence")
        if len({e["frame_index"] for e in items}) != len(items):
            raise ValueError("Duplicate evidence frames")
        for item in items:
            idx = item["frame_index"]
            if idx not in supplied or abs(item["pts_seconds"]-supplied[idx]) > 1e-6:
                raise ValueError("Citation not among supplied native frames/PTS")
            if not item["visible_detail"].strip():
                raise ValueError("Evidence needs visible detail")
            if phase == "initial" and idx > start + .2*(end-start-1):
                raise ValueError("Initial relation cites outside initial window")
            if phase == "final" and idx < end-1 - .2*(end-start-1):
                raise ValueError("Final relation cites outside final window")
    ids=[x["id"] for x in response["entities"]]
    if len(ids) != len(set(ids)) or any(not x.strip() for x in ids):
        raise ValueError("Entity IDs must be unique nonempty strings")
    entities={e["id"]:e for e in response["entities"]}
    for item in response["entities"]:
        evidence(item["identity_evidence"], required=True)
    for phase in ["initial","final"]:
        for item in response[phase+"_relations"]:
            if item["subject"] not in entities or (item["object"] and item["object"] not in entities):
                raise ValueError("Relation references unknown entity")
            evidence(item["evidence"], required=item["observation"]=="observed", phase=phase)
    events={e["id"]:e for e in response["events"]}
    if len(events) != len(response["events"]) or any(not x.strip() for x in events):
        raise ValueError("Event IDs must be unique nonempty strings")
    for item in events.values():
        if not item["entity_ids"] or not set(item["entity_ids"]) <= set(entities):
            raise ValueError("Event references unknown entity")
        evidence(item["evidence"],required=item["observation"]=="observed")
        if item["observation"]=="observed" and item["temporal_scope"]=="unknown":
            raise ValueError("Observed event needs a visible temporal scope")
    graph={k:set() for k in events}
    for edge in response["observed_order"]:
        a,b=edge["before_event"],edge["after_event"]
        if a not in events or b not in events or a==b:
            raise ValueError("Invalid observed-order edge")
        ea,eb=events[a],events[b]
        if any(e["observation"]!="observed" for e in [ea,eb]):
            raise ValueError("Cannot establish order of unknown events")
        if max(x["frame_index"] for x in ea["evidence"]) > min(x["frame_index"] for x in eb["evidence"]):
            raise ValueError("Observed event intervals overlap or reverse")
        graph[a].add(b)
    def visit(node,active,finished):
        if node in active: raise ValueError("Observed-order cycle")
        if node in finished:return
        for other in graph[node]:visit(other,active|{node},finished)
        finished.add(node)
    finished=set()
    for node in graph:visit(node,set(),finished)
    evidence(response["window_assessment"]["boundary_evidence"])
    for item in response["hand_roles"]:
        if item["entity_id"] not in entities or entities[item["entity_id"]]["kind"]!="hand":
            raise ValueError("Hand role references a non-hand entity")
        evidence(item["evidence"],required=item["role"]!="uncertain")
    for item in response["hidden_state_observations"]:
        if item["entity_id"] not in entities:
            raise ValueError("Hidden-state observation references unknown entity")
        evidence(item["evidence"],required=item["later_visibility"]=="visible")
    for item in response["dense_review_requests"]:
        a,b=item["first_supplied_frame"],item["last_supplied_frame"]
        if a not in supplied or b not in supplied or a>b:
            raise ValueError("Dense review request not bounded by supplied frames")
    return True


def compile_draft(response, packet, *, raw_response=None):
    validate_observations(response,packet)
    return {
        "schema":"r2h_human_task_draft_v1",
        "source_video":packet["video"],
        "candidate_window_frames":packet["window_frames"],
        "packet_sha256":json_sha256(packet),
        "raw_response":raw_response,
        "observations":response,
        "necessary_precedence":{"status":"unknown","edges":[],
            "reason":"A single observed execution does not establish causal necessity."},
        "human_sim_equivalence":"unverified",
        "contract_status":"vlm_draft_requires_independent_review",
        "manual_semantic_reviewed":False, "training_admitted":False,
        "human_3d_ground_truth_created":False,
        "physical_contact_verified":False,
        "validation_scope":"schema, entity references, cited supplied frames/PTS and temporal consistency only",
        "next_steps":["Review requested dense intervals and ambiguous boundaries.",
                      "Resolve roles and allowable scene/embodiment changes before compiling a simulator specification.",
                      "Check both human and simulator evidence independently; do not propagate draft to approval flags."],
    }


def extract_draft(output,config):
    from .batch_pipeline import paid_stage
    from .providers import GeminiClient
    output=Path(output)
    packet=json.loads((output/"packet.json").read_text())
    request=json.loads((output/"request.json").read_text())
    checked_path(packet["video"])
    checked_path(request["packet"])
    for image in request["images"]:checked_path(image)
    def call(transport):
        client=GeminiClient(config["gemini"],transport=transport)
        parts=[{"text":request["prompt"]}]
        for image in request["images"]:
            parts += [{"text":image["label"]},client._inline(Path(image["path"]),"image")]
        return client._review_result(client._generate(
            client.model,parts,client._review_config(request["prompt"],SCHEMA,True)),SCHEMA,True)
    result=paid_stage(output/"blind_extraction_v1",{
        "kind":"blind_real_human_task_draft","model":config["gemini"]["model"],
        "request":binding(output/"request.json"),"packet":binding(output/"packet.json"),
        "images":request["images"],"schema":SCHEMA,
    },call)
    try:
        draft=compile_draft(result["review"],packet,
                            raw_response=binding(output/"blind_extraction_v1/result.json"))
    except Exception as exc:
        write_json(output/"blind_extraction_v1/validation_error.json",{
            "error":str(exc),"response":binding(output/"blind_extraction_v1/result.json"),
            "automatic_retry":False,"training_admitted":False})
        raise
    write_json(output/"draft_task_contract.json",draft)
    return draft
