import json
import pytest
from data_generation_methods.r2h_pipeline.common import file_sha256, json_sha256, write_json
from data_generation_methods.scripts.propose_scene_conditions_20261020d import review_candidates


def test_failed_unknown_keep_is_preserved_and_explicit_repair_has_new_stage(tmp_path,monkeypatch):
    media=tmp_path/"image.png"
    media.write_bytes(b"test binding, provider is mocked")
    bound={"path":str(media),"sha256":file_sha256(media)}
    proposal={"bindings":{},"source_keyframes":[],
              "target":{"video":bound,"frame":bound,
                        "roi_candidates":[{"id":"background","role":"background_reference","status":"unknown"}]}}
    request={"required":True,"images":[],"proposal_sha256":json_sha256(proposal),
             "target_frame_sha256":bound["sha256"]}
    write_json(tmp_path/"proposal.json",proposal)
    write_json(tmp_path/"vlm_correction_request.json",request)
    bad={"proposal_sha256":request["proposal_sha256"],"target_frame_sha256":bound["sha256"],
         "decisions":[{"roi_id":"background","status":"keep","box_xyxy_normalized":[.1,.1,.3,.3],
                       "reason":"visible patch"}]}
    seen=[]
    def fake_stage(directory,metadata,operation):
        seen.append((directory.name,metadata))
        review=json.loads(json.dumps(bad))
        if directory.name=="vlm_correction_repair_v2":
            review["decisions"][0]["status"]="correct"
        result={"review":review}
        write_json(directory/"result.json",result)
        return result
    monkeypatch.setattr("data_generation_methods.r2h_pipeline.batch_pipeline.paid_stage",fake_stage)
    with pytest.raises(ValueError,match="Cannot keep"):
        review_candidates(tmp_path,{"gemini":{"model":"test"}})
    original=(tmp_path/"vlm_correction_v1/result.json").read_bytes()
    assert (tmp_path/"vlm_correction_v1/validation_error.json").is_file()
    assert not (tmp_path/"corrected_rois.json").exists()
    result=review_candidates(tmp_path,{"gemini":{"model":"test"}},repair=True)
    assert (tmp_path/"vlm_correction_v1/result.json").read_bytes()==original
    assert [x[0] for x in seen]==["vlm_correction_v1","vlm_correction_repair_v2"]
    assert "original_response" in seen[1][1]
    assert not result["training_admitted"]
    assert (tmp_path/"corrected_rois_repair_v2.json").is_file()
