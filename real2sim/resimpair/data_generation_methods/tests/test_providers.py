"""Offline protocol and failure tests; no provider credentials or network are used."""
import base64
import copy
import json
from pathlib import Path
import sys
from urllib.error import HTTPError

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r2h_pipeline.providers import GeminiClient, ProviderError, SeedanceClient


class Transport:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, url, *, headers, json, timeout):
        self.calls.append({"method": method, "url": url, "headers": headers,
                           "json": copy.deepcopy(json), "timeout": timeout})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture(autouse=True)
def fake_credentials(monkeypatch):
    # These fixture variables are invented test credentials, not the user's keys.
    monkeypatch.setenv("R2H_TEST_ARK_KEY", "fake-ark-secret")
    monkeypatch.setenv("R2H_TEST_GEMINI_KEY", "fake-gemini-secret")


def seedance(transport, **config):
    return SeedanceClient({"model": "doubao-seedance-2-5-260628",
                           "api_key_env": "R2H_TEST_ARK_KEY", **config}, transport=transport)


def gemini(transport, **config):
    return GeminiClient({"model": "gemini-3.1-pro-preview",
                         "api_key_env": "R2H_TEST_GEMINI_KEY", **config}, transport=transport)


def edit_payload():
    return {"content": [{"type": "text", "text": "Replace the robot with a person."},
                        {"type": "video_url", "video_url": {"url": "https://media.example/input.mp4?sig=private"},
                         "role": "reference_video"}], "omni_reference_task_type": "edit"}


def candidate(*parts, finish="STOP"):
    return {"candidates": [{"finishReason": finish, "content": {"parts": list(parts)}}],
            "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 50}}


def test_prepare_without_credentials_or_network(monkeypatch):
    monkeypatch.delenv("R2H_TEST_ARK_KEY")
    transport = Transport()
    client = seedance(transport)
    payload = edit_payload()
    prepared = client.prepare_submit(payload)
    assert not transport.calls
    assert "headers" not in prepared
    assert prepared["json"]["model"] == "doubao-seedance-2-5-260628"
    assert prepared["json"]["duration"] == -1
    assert prepared["json"]["ratio"] == "adaptive"
    assert "model" not in payload  # No mutation of reviewed user input.


def test_seedance_submit_and_single_poll():
    transport = Transport({"id": "cgt-test-1"},
                          {"id": "cgt-test-1", "status": "running"},
                          {"id": "cgt-test-1", "status": "succeeded",
                           "content": {"video_url": "https://output.example/video.mp4?sig=private"},
                           "usage": {"completion_tokens": 1536}})
    client = seedance(transport)
    result = client.submit(edit_payload())
    assert result == {"status": "submitted", "task_id": "cgt-test-1", "output_url": None,
                      "usage": {}, "model": "doubao-seedance-2-5-260628"}
    assert transport.calls[0]["headers"]["Authorization"] == "Bearer fake-ark-secret"
    assert client.poll("cgt-test-1")["status"] == "running"
    result = client.poll("cgt-test-1")
    assert result["status"] == "succeeded" and result["usage"]["completion_tokens"] == 1536
    assert [c["method"] for c in transport.calls] == ["POST", "GET", "GET"]


@pytest.mark.parametrize("uri", ["data:video/mp4;base64,YQ==", "/tmp/video.mp4", "file:///tmp/video.mp4"])
def test_seedance_video_cannot_inline_or_local(uri):
    transport = Transport()
    payload = edit_payload()
    payload["content"][1]["video_url"]["url"] = uri
    with pytest.raises(ProviderError) as caught:
        seedance(transport).submit(payload)
    assert caught.value.code == "video_requires_http_or_asset"
    assert not transport.calls


def test_seedance_asset_and_inline_image_modes():
    payload = edit_payload()
    payload["content"][1]["video_url"]["url"] = "asset://test-video"
    payload["content"].append({"type": "image_url", "role": "reference_image",
                               "image_url": {"url": "data:image/png;base64,YQ=="}})
    prepared = seedance(Transport()).prepare_submit(payload)
    assert prepared["json"]["content"][-1]["role"] == "reference_image"


@pytest.mark.parametrize("change,code", [
    ({"duration": 5}, "edit_requires_original_duration"),
    ({"ratio": "16:9"}, "edit_extend_requires_adaptive_ratio"),
    ({"frames": 100}, "edit_requires_original_duration"),
    ({"model": "guessed-model"}, "model_mismatch"),
])
def test_seedance_edit_constraints(change, code):
    payload = {**edit_payload(), **change}
    with pytest.raises(ProviderError) as caught:
        seedance(Transport()).prepare_submit(payload)
    assert caught.value.code == code


def test_mixed_reference_and_first_frame_rejected():
    payload = edit_payload()
    payload["content"].append({"type": "image_url", "role": "first_frame",
                               "image_url": {"url": "https://media.example/anchor.png"}})
    with pytest.raises(ProviderError, match="mixed_frame_and_reference_modes"):
        seedance(Transport()).prepare_submit(payload)


def test_first_frame_sets_adaptive_ratio():
    payload = {"content": [{"type": "image_url", "role": "first_frame",
                             "image_url": {"url": "https://media.example/anchor.png"}}]}
    assert seedance(Transport()).prepare_submit(payload)["json"]["ratio"] == "adaptive"


def test_post_timeout_is_unknown_once_and_sanitized():
    secret = "fake-ark-secret https://private.example/?token=never-log"
    transport = Transport(TimeoutError(secret))
    with pytest.raises(ProviderError) as caught:
        seedance(transport).submit(edit_payload())
    error = caught.value
    assert error.code == "submission_unknown" and error.submission_unknown and not error.retryable
    assert len(transport.calls) == 1
    assert "secret" not in str(error) and "private.example" not in str(error)
    assert error.__suppress_context__


def test_get_timeout_retryable_without_automatic_retry():
    transport = Transport(TimeoutError("private details"))
    with pytest.raises(ProviderError) as caught:
        seedance(transport).poll("cgt-test")
    assert caught.value.retryable and not caught.value.submission_unknown
    assert len(transport.calls) == 1


@pytest.mark.parametrize("status,ambiguous,retryable", [(400, False, False), (429, False, True), (503, True, False)])
def test_http_errors_never_echo_provider_body(status, ambiguous, retryable):
    failure = HTTPError("https://private.example/?key=secret", status, "secret raw failure", {}, None)
    with pytest.raises(ProviderError) as caught:
        seedance(Transport(failure)).submit(edit_payload())
    assert caught.value.http_status == status
    assert caught.value.submission_unknown == ambiguous
    assert caught.value.retryable == retryable
    assert "private.example" not in str(caught.value) and "secret" not in str(caught.value)


def test_missing_submission_id_ambiguous():
    with pytest.raises(ProviderError) as caught:
        seedance(Transport({"unexpected": "response"})).submit(edit_payload())
    assert caught.value.submission_unknown


def test_failed_poll_sanitized_and_invalid_task_path_rejected():
    transport = Transport({"status": "failed", "error": {"message": "private source URL"}})
    result = seedance(transport).poll("cgt-test")
    assert result["error_code"] == "provider_task_failed"
    assert "private" not in json.dumps(result)
    with pytest.raises(ProviderError, match="invalid_task_id"):
        seedance(transport).poll("../private?key=secret")
    assert len(transport.calls) == 1


@pytest.mark.parametrize("model", [None, "", "a/b?key=secret"])
def test_explicit_valid_model_required(model):
    with pytest.raises(ProviderError, match="explicit_model_required"):
        SeedanceClient({"model": model}, transport=Transport())


def test_missing_key_fails_locally(monkeypatch):
    monkeypatch.delenv("R2H_TEST_ARK_KEY")
    transport = Transport()
    with pytest.raises(ProviderError, match="credentials_missing"):
        seedance(transport).submit(edit_payload())
    assert not transport.calls


def test_gemini_video_json_schema_and_anonymous_payload(tmp_path):
    video = tmp_path / "expected_goal_secret_task.mp4"
    video.write_bytes(b"offline toy media")
    response = candidate({"text": "hidden reasoning", "thought": True}, {"text": '{"observed_goal":"choice_b"}'})
    transport = Transport(response)
    schema = {"type": "object", "properties": {"observed_goal": {"type": "string"}},
              "required": ["observed_goal"], "additionalProperties": False}
    result = gemini(transport).review(video, "Observe the anonymous choices.", schema)
    assert result["review"] == {"observed_goal": "choice_b"}
    assert result["usage"]["promptTokenCount"] == 100
    request = transport.calls[0]
    assert request["headers"]["x-goog-api-key"] == "fake-gemini-secret"
    assert "?key=" not in request["url"]
    assert request["json"]["generationConfig"]["responseJsonSchema"] == schema
    assert video.name not in json.dumps(request["json"])
    assert str(tmp_path) not in json.dumps(request["json"])
    assert base64.b64decode(request["json"]["contents"][0]["parts"][0]["inlineData"]["data"]) == video.read_bytes()


def test_gemini_oversize_does_not_clip_upload_or_call(tmp_path):
    video = tmp_path / "large.mp4"
    with video.open("wb") as output:
        output.truncate(15 * 1024 * 1024 + 1)
    transport = Transport()
    with pytest.raises(ProviderError, match="inline_limit_exceeded"):
        gemini(transport).review(video, "Review")
    assert not transport.calls
    assert video.stat().st_size == 15 * 1024 * 1024 + 1


def test_gemini_total_json_limit_includes_prompt(tmp_path):
    video = tmp_path / "small.mp4"
    video.write_bytes(b"tiny")
    transport = Transport()
    with pytest.raises(ProviderError, match="inline_limit_exceeded"):
        gemini(transport).review(video, "x" * (20 * 1024 * 1024))
    assert not transport.calls


@pytest.mark.parametrize("response,code", [
    ({"promptFeedback": {"blockReason": "SAFETY"}}, "response_blocked"),
    ({"candidates": []}, "missing_candidate"),
    (candidate({"text": '{"x":1}'}, finish="MAX_TOKENS"), "incomplete_or_blocked_response"),
    (candidate({"text": "```json\n{}\n```"}), "invalid_review_json"),
    (candidate({"text": "[]"}), "review_must_be_object"),
])
def test_gemini_rejects_incomplete_non_json_and_blocked(tmp_path, response, code):
    video = tmp_path / "toy.mp4"
    video.write_bytes(b"toy")
    with pytest.raises(ProviderError) as caught:
        gemini(Transport(response)).review(video, "Review")
    assert caught.value.code == code


def test_gemini_image_anchor_uses_separate_image_model_and_no_json_schema(tmp_path):
    image = tmp_path / "source_private_goal.png"
    image.write_bytes(b"toy image input")
    output_b64 = base64.b64encode(b"toy generated image").decode("ascii")
    response = candidate({"text": "Done"}, {"inlineData": {"mimeType": "image/png", "data": output_b64}})
    transport = Transport(response)
    result = gemini(transport).generate_image(image, "Replace the robot with a person.")
    assert result["images"] == [{"mime_type": "image/png", "data": output_b64}]
    assert result["model"] == "gemini-nano-banana-2.1"
    assert transport.calls[0]["url"].endswith("/models/gemini-nano-banana-2.1:generateContent")
    assert transport.calls[0]["json"]["generationConfig"] == {"responseModalities": ["TEXT", "IMAGE"]}
    assert image.name not in json.dumps(transport.calls[0]["json"])


@pytest.mark.parametrize("parts,code", [
    ([{"text": "No image"}], "missing_image_response"),
    ([{"inlineData": {"mimeType": "image/png", "data": "!bad!"}}], "invalid_image_response"),
    ([{"inlineData": {"mimeType": "video/mp4", "data": "YQ=="}}], "invalid_image_response"),
])
def test_gemini_image_rejects_bad_output(tmp_path, parts, code):
    image = tmp_path / "toy.png"
    image.write_bytes(b"toy")
    with pytest.raises(ProviderError) as caught:
        gemini(Transport(candidate(*parts))).generate_image(image, "Edit")
    assert caught.value.code == code
