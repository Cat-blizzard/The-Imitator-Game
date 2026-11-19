"""Local role/binding/submission regressions, not generated-video quality tests."""
import base64
from copy import deepcopy
import json

from PIL import Image
import pytest

from corl2026_workshop.experiments.autonomous_pipeline_v4.core import bind, read, write
from corl2026_workshop.experiments.autonomous_pipeline_v4.motion_reference import reference_payload
from resimpair_tasks import endpoint_conditioning_v1 as candidate


@pytest.fixture
def inputs(tmp_path):
    config = {"seedance": {"model": "doubao-seedance-2-5-260628", "base_url": "https://example.com",
                            "api_key_env": "RESIMPAIR_ENDPOINT_TEST_KEY", "poll_interval_seconds": 1}}
    write(tmp_path / "config.json", config)
    paths = {}
    for name, color in [("initial", "white"), ("terminal", "black"), ("sim_initial", "blue"), ("sim_terminal", "green")]:
        path = tmp_path / (name + ".png")
        Image.new("RGB", (300, 300), color).save(path)
        paths[name] = bind(path)
    write(tmp_path / "contract.json", {"task": "one_apple_left_to_right_bowl", "hand": "right"})
    write(tmp_path / "verdict.json", {"schema": "test_independent_anchor_verdict", "actor_hand": "right",
                                     "note": "binding test fixture; not a real quality verdict"})
    (tmp_path / "reference.mp4").write_bytes(b"opaque binding fixture, not simulation evidence")
    video = bind(tmp_path / "reference.mp4")
    write(tmp_path / "locator.json", {"url": "https://example.com/reference.mp4", "source": video})
    write(tmp_path / "upload.json", {"source": video, "remote_sha256_matches_local": True,
                                      "private_locator": bind(tmp_path / "locator.json")})
    return {"root": tmp_path, "config": config, **paths, "contract": bind(tmp_path / "contract.json"),
            "condition_evidence": {"anchor_verdict": bind(tmp_path / "verdict.json")},
            "reference_video": video, "remote_reference": bind(tmp_path / "upload.json")}


def build(i, method="endpoint_reference", **overrides):
    kwargs = {"contract": i["contract"], "task_prompt": "One person uses their anatomical RIGHT hand to transfer the apple.",
              "initial_image": i["initial"], "condition_evidence": i["condition_evidence"]}
    if method != "motion_reference":
        kwargs["terminal_image"] = i["terminal"]
    if method != "endpoint_frames":
        kwargs.update(reference_video=i["reference_video"], remote_reference=i["remote_reference"])
    kwargs.update(overrides)
    return candidate.build_request(i["root"], method, **kwargs)


@pytest.mark.parametrize("method,expected", [
    ("motion_reference", ["reference_image", "reference_video"]),
    ("endpoint_reference", ["reference_image", "reference_image", "reference_video"]),
    ("endpoint_frames", ["first_frame", "last_frame"]),
])
def test_real_provider_role_validation_and_exact_image_bytes(inputs, method, expected):
    from data_generation_methods.r2h_pipeline.providers import SeedanceClient
    request = build(inputs, method)
    body = candidate.payload(inputs["config"], request)
    assert [part["role"] for part in body["content"][1:]] == expected
    assert SeedanceClient(inputs["config"]["seedance"]).prepare_submit(body)["json"] == body
    assert base64.b64decode(body["content"][1]["image_url"]["url"].split(",", 1)[1]) == (inputs["root"] / "initial.png").read_bytes()
    if method != "motion_reference":
        assert base64.b64decode(body["content"][2]["image_url"]["url"].split(",", 1)[1]) == (inputs["root"] / "terminal.png").read_bytes()
    if method == "endpoint_frames":
        assert "omni_reference_task_type" not in body
        assert "reference.mp4" not in json.dumps(body)


def test_baseline_payload_matches_existing_reference_route(inputs):
    request = build(inputs, "motion_reference")
    assert candidate.payload(inputs["config"], request) == reference_payload(inputs["config"], request)
    assert "terminal_image" not in request and "@image2" not in request["prompt"]


def test_same_task_text_and_separate_routing_are_preserved(inputs):
    a, b, c = [build(inputs, name) for name in ["motion_reference", "endpoint_reference", "endpoint_frames"]]
    assert a["task_prompt"] == b["task_prompt"] == c["task_prompt"]
    assert b["prompt_components"]["routing"] != a["prompt_components"]["routing"]
    assert b["endpoint_semantics"] == {"human_image": "task_initial_state", "terminal_image": "task_terminal_state"}


def test_ab_change_is_only_terminal_image_and_minimal_role_explanation(inputs):
    a = build(inputs, "motion_reference")
    b = build(inputs, "endpoint_reference")
    assert b["prompt_components"]["routing"] == (
        a["prompt_components"]["routing"] + candidate.ENDPOINT_REFERENCE_ADDITION
    )
    assert b["prompt"].replace(candidate.ENDPOINT_REFERENCE_ADDITION, "", 1) == a["prompt"]
    original = candidate.payload(inputs["config"], a)
    with_endpoint = candidate.payload(inputs["config"], b)
    assert with_endpoint["content"][2]["role"] == "reference_image"
    del with_endpoint["content"][2]
    with_endpoint["content"][0]["text"] = with_endpoint["content"][0]["text"].replace(
        candidate.ENDPOINT_REFERENCE_ADDITION, "", 1
    )
    assert with_endpoint == original


@pytest.mark.parametrize("method,overrides,error", [
    ("endpoint_frames", {"reference_video": "video"}, "cannot_include_reference_video"),
    ("motion_reference", {"terminal_image": "terminal"}, "terminal_presence_mismatch"),
    ("endpoint_reference", {"terminal_image": None}, "terminal_presence_mismatch"),
    ("endpoint_reference", {"reference_video": None}, "requires_motion_receipt"),
])
def test_invalid_mode_combinations_fail_before_network(inputs, method, overrides, error):
    kwargs = {k: inputs["reference_video"] if v == "video" else inputs["terminal"] if v == "terminal" else v
              for k, v in overrides.items()}
    with pytest.raises(ValueError, match=error):
        build(inputs, method, **kwargs)


def test_terminal_tampering_is_detected(inputs):
    request = build(inputs)
    Image.new("RGB", (300, 300), "red").save(inputs["root"] / "terminal.png")
    with pytest.raises(ValueError, match="Artifact bytes"):
        candidate.payload(inputs["config"], request)


def test_independent_evidence_tampering_is_detected(inputs):
    request = build(inputs)
    (inputs["root"] / "verdict.json").write_text('{}')
    with pytest.raises(ValueError, match="Artifact bytes"):
        candidate.payload(inputs["config"], request)


def test_missing_independent_anchor_binding_is_rejected(inputs):
    with pytest.raises(ValueError, match="requires_anchor_verdict"):
        build(inputs, condition_evidence={})


def test_unrelated_upload_is_rejected(inputs):
    request = build(inputs)
    request["reference_video"] = inputs["initial"]
    with pytest.raises(ValueError, match="upload_binding_mismatch"):
        candidate.payload(inputs["config"], request)


def test_settings_cannot_inject_another_payload(inputs):
    with pytest.raises(ValueError, match="override_roles"):
        build(inputs, settings={"content": []})
    with pytest.raises(ValueError, match="adaptive_ratio"):
        build(inputs, "endpoint_frames", settings={"ratio": "16:9"})


def test_png_mime_not_assumed_from_filename(inputs):
    target = inputs["root"] / "really_jpeg.png"
    Image.new("RGB", (300, 300), "red").save(target, format="JPEG")
    with pytest.raises(ValueError, match="requires_png"):
        build(inputs, terminal_image=bind(target))


def test_terminal_generation_uses_same_person_and_three_ordered_references(inputs):
    request = candidate.build_terminal_image_request(inputs["root"], human_initial=inputs["initial"],
        simulation_initial=inputs["sim_initial"], simulation_terminal=inputs["sim_terminal"],
        task_delta_prompt="Move only the apple from the left bowl into the right bowl; both hands released.",
        condition_evidence=inputs["condition_evidence"])
    assert request["images"] == [inputs["initial"], inputs["sim_initial"], inputs["sim_terminal"]]
    assert request["image_meanings"] == ["human_initial", "simulation_initial", "simulation_terminal"]
    assert request["requires_independent_terminal_review_before_video"] is True
    assert "same single human operator" in request["prompt"]


def actual_budget(i, monkeypatch):
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import budget
    monkeypatch.setattr(budget, "REGISTRY", i["root"] / "registry")
    specification = {"schema": "resimpair_cumulative_budget_v4", "campaign_id": "endpoint_local_test",
        "ledger_path": str(i["root"] / "ledger"), "new_call_limits": {"image": 0, "video": 1, "review": 0},
        "ancestor_started_calls": {"image": 0, "video": 0, "review": 0},
        "lineage_cumulative_limits": {"image": 0, "video": 1, "review": 0},
        "ancestor_evidence": [], "max_concurrent": 1, "timeout_seconds": 60}
    write(i["root"] / "budget.json", specification)
    return budget.Budget(i["root"] / "budget.json")


def submit_fixture(i, monkeypatch, transport):
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import models
    from corl2026_workshop.experiments.task_contract_v3 import direct_transport
    monkeypatch.setenv("RESIMPAIR_ENDPOINT_TEST_KEY", "local-offline-test")
    monkeypatch.setattr(models, "configured", lambda root: deepcopy(i["config"]))
    monkeypatch.setattr(direct_transport, "call", transport)
    request = build(i)
    path = i["root"] / "request.json"
    write(path, request)
    return actual_budget(i, monkeypatch), path


def test_real_budget_submission_records_roles_and_reuses_saved_id(inputs, monkeypatch):
    sent = []
    def transport(method, url, **kwargs):
        sent.append(kwargs["json"])
        return {"id": "cgt-offline-test"}
    budget, path = submit_fixture(inputs, monkeypatch, transport)
    a = candidate.submit(inputs["root"], budget, path)
    b = candidate.submit(inputs["root"], budget, path)
    assert a == b and len(sent) == 1
    assert budget.usage()["new_started"]["video"] == 1
    stage = budget.directory("video", path)
    assert read(stage / "provider_response.json")["id"] == "cgt-offline-test"
    audit = read(stage / "actual_input_audit.json")
    assert audit["roles"] == ["reference_image", "reference_image", "reference_video"]
    assert audit["terminal_image"] == inputs["terminal"]
    assert audit["visual_quality_inferred_by_payload_guard"] is False


def test_unknown_submission_cannot_post_again(inputs, monkeypatch):
    from data_generation_methods.r2h_pipeline.providers import ProviderError
    sent = []
    def transport(method, url, **kwargs):
        sent.append(1)
        raise TimeoutError("local offline uncertainty fixture")
    budget, path = submit_fixture(inputs, monkeypatch, transport)
    with pytest.raises(ProviderError, match="submission_unknown"):
        candidate.submit(inputs["root"], budget, path)
    with pytest.raises(RuntimeError, match="no_resubmission"):
        candidate.submit(inputs["root"], budget, path)
    assert len(sent) == 1 and budget.usage()["new_started"]["video"] == 1
    assert read(budget.directory("video", path) / "failure.json")["provider_outcome"] == "unknown"


def test_config_changed_before_submission_does_not_consume_budget(inputs, monkeypatch):
    budget, path = submit_fixture(inputs, monkeypatch, lambda *a, **k: pytest.fail("must not submit"))
    (inputs["root"] / "config.json").write_text('{}')
    with pytest.raises(ValueError, match="model_config_changed"):
        candidate.submit(inputs["root"], budget, path)
    assert budget.usage()["new_started"]["video"] == 0
