import copy
from pathlib import Path

import pytest

from data_generation_methods.r2h_pipeline import posthoc_controlled_repair as mod
from data_generation_methods.r2h_pipeline.batch_pipeline import binding
from data_generation_methods.r2h_pipeline.common import read_json, write_json
from data_generation_methods.r2h_pipeline.controlled_experiment import (
    create_experiment, common_video_prompt,
)


@pytest.fixture(params=[("apple","B"),("cabinet","A")])
def prepared(tmp_path, request):
    from PIL import Image
    case_id,arm=request.param
    image=tmp_path/"source.png"
    Image.new("RGB",(32,32),"white").save(image)
    case={key:binding(image) for key in (
        "human_source","robot_video","robot_trajectory","sim_initial","sim_terminal",
        "human_initial","source_record")}
    case.update(id=case_id,maximum_visible_hands=2,
                task={"initial":"initial","necessary_steps":["move"],"goal":"final","invariants":["same"]})
    config={"seedance":{"model":"doubao-seedance-2-5-260628","base_url":"https://example.invalid"},
            "gemini":{"model":"gemini-3.1-pro-preview","base_url":"https://example.invalid"},
            "anchor_model":"gemini-nano-banana-2.1"}
    root=tmp_path/"experiment"
    create_experiment(root,[case],config)
    parent=root/case_id/arm
    video=parent/"attempt_01/candidate.mp4"
    video.parent.mkdir(parents=True,exist_ok=True)
    video.write_bytes(b"original-video")
    failure=root/"audit_initial"/mod.ALLOWED[(case_id,arm)]
    write_json(failure,{"video":binding(video),"judgement":"locally observed failure"})
    kinds=("image","video","review") if arm=="B" else ("video","review")
    for kind in kinds:
        stage=parent/"stages"/f"{kind}_01"
        write_json(stage/"request_metadata.json",{
            "experiment_call_kind":kind,"prompt":common_video_prompt(case) if kind=="video" else "blind task review"})
        write_json(stage/"call_started.json",{})
        if kind!="review":
            write_json(stage/"result.json",{"completed":True})
    write_json(parent/"stages/review_01/failure.json",{
        "code":"submission_unknown" if arm=="B" else "review_schema_violation"})
    old=read_json(root/"experiment.json")
    result=mod.prepare_branch(root,case_id,arm)
    assert read_json(root/"experiment.json")==old
    return root,parent,Path(result["branch"]),case


def test_preparation_preserves_original_unknown_and_counts_it(prepared):
    root,parent,branch,case=prepared
    m=read_json(branch/"branch.json")
    assert m["original_review_response_not_reissued"]
    assert not (parent/"stages/review_01/result.json").exists()
    assert mod.aggregate_counts(parent)["review"]==1
    assert mod.aggregate_counts(branch)=={"image":0,"video":0,"review":0}
    assert m["video_prompt"]==common_video_prompt(case)
    assert m["posthoc_not_preregistered_initial_result"]


def test_remaining_budget_counts_every_descendant_unknown(prepared,monkeypatch):
    _,parent,branch,_=prepared
    stage=parent/"another_branch/stages/review_02"
    write_json(stage/"request_metadata.json",{"experiment_call_kind":"review"})
    write_json(stage/"call_started.json",{})
    write_json(stage/"result.json",{"completed":True})
    m=read_json(branch/"branch.json")
    def never(*args,**kwargs):
        raise AssertionError("Must not reach provider")
    monkeypatch.setattr(mod,"paid_stage",never)
    with pytest.raises(RuntimeError,match="Combined"):
        mod._new_paid(branch,m,"review","review_02",{},never)


def test_new_branch_unknown_blocks_further_paid_operation(prepared,monkeypatch):
    _,_,branch,_=prepared
    stage=branch/"stages/video_02"
    write_json(stage/"request_metadata.json",{"experiment_call_kind":"video"})
    write_json(stage/"call_started.json",{})
    m=read_json(branch/"branch.json")
    with pytest.raises(RuntimeError,match="Unresolved repair"):
        mod._new_paid(branch,m,"review","review_02",{},lambda _:None)


def test_unknown_other_new_branch_cannot_be_escaped(prepared):
    _,parent,branch,_=prepared
    stage=parent/"another_branch/stages/video_02"
    write_json(stage/"request_metadata.json",{"experiment_call_kind":"video"})
    write_json(stage/"call_started.json",{})
    with pytest.raises(RuntimeError,match="Unresolved repair"):
        mod._new_paid(branch,read_json(branch/"branch.json"),"review","review_02",{},lambda _:None)


def test_a_is_exact_prompt_redraw_b_requires_static_acceptance(prepared):
    _,_,branch,case=prepared
    m=read_json(branch/"branch.json")
    if m["arm"]=="A":
        payload=mod._video_payload(branch,m)
        assert payload["content"][0]["text"]==common_video_prompt(case)
        assert [p.get("role") for p in payload["content"][1:]]==["first_frame"]
        assert m["this_branch_new_call_caps"]["image"]==0
    else:
        with pytest.raises(FileNotFoundError):
            mod._video_payload(branch,m)
        (branch/"human_terminal.png").write_bytes(Path(case["human_initial"]["path"]).read_bytes())
        mod.accept_terminal(branch,reviewer="test-only",reason="Synthetic test fixture; no real approval")
        payload=mod._video_payload(branch,m)
        assert [p.get("role") for p in payload["content"][1:]]==["first_frame","last_frame"]
        (branch/"human_terminal.png").write_bytes(b"tampered")
        with pytest.raises(ValueError):
            mod._video_payload(branch,m)


def test_apple_edit_uses_one_target_and_explicit_state_delta(prepared):
    _,_,branch,_=prepared
    p=read_json(branch/"image_request_plan.json")
    if p["new_call_required"]:
        assert p["supporting_images"]==[]
        assert "LEFT bowl must be EMPTY" in p["prompt"]
        assert "RIGHT bowl must contain exactly" in p["prompt"]
