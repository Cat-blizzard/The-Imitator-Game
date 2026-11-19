"""Offline tests for explicit frames and strict, locally validated review JSON."""
import base64
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r2h_pipeline.providers import GeminiClient, ProviderError, parse_review_json


SCHEMA = {
    "type": "object",
    "required": ["verdict", "events"],
    "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "fail", "unknown"]},
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["frame_index", "timestamp_seconds"],
                "additionalProperties": False,
                "properties": {
                    "frame_index": {"type": "integer", "minimum": 0},
                    "timestamp_seconds": {"type": "number", "minimum": 0},
                },
            },
        },
    },
}
REVIEW = {"verdict": "fail", "events": [{"frame_index": 3, "timestamp_seconds": .125}]}


def candidate(*parts):
    return {"candidates": [{"finishReason": "STOP", "content": {"parts": list(parts)}}],
            "usageMetadata": {"promptTokenCount": 50}}


class Transport:
    def __init__(self, response=None):
        self.response = response
        self.calls = []

    def __call__(self, method, url, *, headers, json, timeout):
        self.calls.append(copy.deepcopy(json))
        return self.response


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("R2H_FRAME_TEST_KEY", "fake-frame-test-key")
    transport = Transport(candidate({"text": json.dumps(REVIEW)}))
    return GeminiClient({"model": "offline-test-model", "api_key_env": "R2H_FRAME_TEST_KEY"},
                        transport=transport)


@pytest.fixture
def frames(tmp_path):
    result = []
    for index, timestamp in [(0, 0.0), (3, .125), (6, .25)]:
        path = tmp_path / f"private_expected_goal_frame_{index}.png"
        path.write_bytes(f"offline-image-{index}".encode())
        result.append({"path": path, "timestamp_seconds": timestamp, "frame_index": index})
    return result


@pytest.mark.parametrize("wrapper,normalization", [
    ("{}", "none"),
    (" \n{}\t", "none"),
    ("```json\n{}\n```", "outer_json_fence"),
    ("\n```json\r\n{}\r\n```\t", "outer_json_fence"),
])
def test_parser_records_raw_text_and_only_outer_fence_normalization(wrapper, normalization):
    raw = wrapper.format(json.dumps(REVIEW))
    schema_before = copy.deepcopy(SCHEMA)
    result = parse_review_json(raw, SCHEMA, allow_json_fence=True)
    assert result == {"review": REVIEW, "raw_text": raw, "normalization": normalization,
                      "parse_status": "valid", "schema_validated": True}
    assert SCHEMA == schema_before


def test_parser_is_strict_by_default_and_optional_schema_is_explicit():
    with pytest.raises(ProviderError, match="invalid_review_json"):
        parse_review_json("```json\n{}\n```")
    result = parse_review_json('{"comment":"a ```json marker inside a string"}')
    assert result["schema_validated"] is False
    assert result["normalization"] == "none"


@pytest.mark.parametrize("text", [
    'Here is the result:\n```json\n{}\n```',
    '```json\n{}\n```\nDone.',
    '```json\n{}\n```\n```json\n{}\n```',
    '```json\n{} {}\n```',
    '{} {}',
    '```JSON\n{}\n```',
    '```\n{}\n```',
    '```json {}\n```',
    '```json\n```json\n{}\n```\n```',
    '```json\n{"verdict": "fail",}\n```',
    '{"verdict":"pass","verdict":"fail"}',
    '{"nested":{"frame_index":0,"frame_index":3}}',
    '{"timestamp":NaN}',
    '{"timestamp":Infinity}',
    '{"timestamp":-Infinity}',
    '{"timestamp":1e999}',
    '\u00a0{}\u00a0',
])
def test_parser_does_not_repair_or_extract_ambiguous_json(text):
    with pytest.raises(ProviderError) as caught:
        parse_review_json(text, allow_json_fence=True)
    assert caught.value.code == "invalid_review_json"
    assert not caught.value.retryable


@pytest.mark.parametrize("value", [
    {"verdict": "fail", "events": [], "unrecognized": True},
    {"verdict": "fail"},
    {"verdict": "looks good", "events": []},
    {"verdict": "fail", "events": [{"frame_index": 3, "timestamp_seconds": .125, "extra": 1}]},
    {"verdict": "fail", "events": [{"frame_index": 3}]},
    {"verdict": "fail", "events": [{"frame_index": True, "timestamp_seconds": .125}]},
    {"verdict": "fail", "events": [{"frame_index": 3, "timestamp_seconds": -.125}]},
])
def test_local_jsonschema_rejects_unknown_missing_enum_and_nested_type_violations(value):
    raw = "```json\n" + json.dumps(value) + "\n```"
    with pytest.raises(ProviderError) as caught:
        parse_review_json(raw, SCHEMA, allow_json_fence=True)
    assert caught.value.code == "review_schema_violation"
    assert "verdict" not in str(caught.value)


def test_jsonschema_supports_local_references_and_does_not_fetch_remote_references():
    schema = {"$defs": {"review": SCHEMA}, "$ref": "#/$defs/review"}
    assert parse_review_json(json.dumps(REVIEW), schema)["review"] == REVIEW
    with pytest.raises(ProviderError, match="review_schema_violation"):
        parse_review_json("{}", {"$ref": "https://not-a-real-schema.invalid/schema.json"})


def test_frames_interleave_exact_timestamps_and_images_without_private_paths(client, frames):
    raw = "```json\n" + json.dumps(REVIEW) + "\n```"
    client.transport.response = candidate(
        {"thought": True, "text": "ignore this non-JSON reasoning"},
        {"text": raw[:20]}, {"text": raw[20:]},
    )
    before = copy.deepcopy(frames)
    result = client.review_frames(frames, "Inspect every frame.", SCHEMA)
    assert result["review"] == REVIEW
    assert result["raw_text"] == raw
    assert result["normalization"] == "outer_json_fence"
    assert result["schema_validated"] is True
    assert result["parse_status"] == "valid"
    assert result["usage"] == {"promptTokenCount": 50}
    assert len(client.transport.calls) == 1
    payload = client.transport.calls[0]
    assert payload["generationConfig"]["responseJsonSchema"] == SCHEMA
    parts = payload["contents"][0]["parts"]
    assert parts[0] == {"text": "Inspect every frame."}
    assert len(parts) == 1 + 2 * len(frames)
    for position, frame in enumerate(frames):
        assert json.loads(parts[1 + position * 2]["text"]) == {
            "frame_index": frame["frame_index"], "timestamp_seconds": frame["timestamp_seconds"],
        }
        image = parts[2 + position * 2]["inlineData"]
        assert image["mimeType"] == "image/png"
        assert base64.b64decode(image["data"]) == frame["path"].read_bytes()
        assert str(frame["path"]) not in json.dumps(payload)
        assert frame["path"].name not in json.dumps(payload)
    assert frames == before


def test_video_still_rejects_fences_and_frames_can_disable_normalization(client, frames, tmp_path):
    raw = "```json\n" + json.dumps(REVIEW) + "\n```"
    client.transport.response = candidate({"text": raw})
    video = tmp_path / "video.mp4"
    video.write_bytes(b"offline video")
    with pytest.raises(ProviderError, match="invalid_review_json"):
        client.review(video, "Review", SCHEMA)
    with pytest.raises(ProviderError, match="invalid_review_json"):
        client.review_frames(frames, "Review", SCHEMA, allow_json_fence=False)
    result = client.review(video, "Review", SCHEMA, allow_json_fence=True)
    assert result["normalization"] == "outer_json_fence"
    assert len(client.transport.calls) == 3  # Exactly one call per explicit request.


def test_schema_violations_fail_after_one_response_without_retry(client, frames):
    client.transport.response = candidate({"text": '{"verdict":"pass","events":[],"extra":1}'})
    with pytest.raises(ProviderError, match="review_schema_violation"):
        client.review_frames(frames, "Review", SCHEMA)
    assert len(client.transport.calls) == 1


@pytest.mark.parametrize("schema", [
    [], {"type": "not-a-jsonschema-type"}, {"required": "verdict"}, {"minimum": float("nan")},
])
def test_invalid_schema_fails_before_transport(client, frames, schema):
    with pytest.raises(ProviderError, match="invalid_response_schema"):
        client.review_frames(frames, "Review", schema)
    assert not client.transport.calls


@pytest.mark.parametrize("key,value", [
    ("timestamp_seconds", -.125),
    ("timestamp_seconds", True),
    ("timestamp_seconds", "0.125"),
    ("timestamp_seconds", float("nan")),
    ("timestamp_seconds", float("inf")),
    ("timestamp_seconds", 10 ** 400),
    ("frame_index", True),
    ("frame_index", -1),
    ("frame_index", 3.0),
    ("path", None),
])
def test_invalid_frame_metadata_fails_before_transport(client, frames, key, value):
    frames[1][key] = value
    with pytest.raises(ProviderError, match="invalid_frames"):
        client.review_frames(frames, "Review", SCHEMA)
    assert not client.transport.calls


@pytest.mark.parametrize("mutate", [
    lambda frames: frames.reverse(),
    lambda frames: frames[1].update(timestamp_seconds=0.0),
    lambda frames: frames[1].update(frame_index=0),
    lambda frames: frames[1].update(frame_index=7),
    lambda frames: frames[1].update(timestamp_seconds=.5),
])
def test_frame_times_and_indices_must_increase_without_sorting(client, frames, mutate):
    mutate(frames)
    before = copy.deepcopy(frames)
    with pytest.raises(ProviderError, match="invalid_frame_sequence"):
        client.review_frames(frames, "Review", SCHEMA)
    assert not client.transport.calls
    assert frames == before


@pytest.mark.parametrize("frames", [[], {}, [None], [{"path": "image.png"}], [
    {"path": "image.png", "timestamp_seconds": 0, "frame_index": 0, "expected_goal": "private"},
]])
def test_invalid_frame_collection_is_rejected_locally(client, frames):
    with pytest.raises(ProviderError, match="invalid_frames"):
        client.review_frames(frames, "Review", SCHEMA)
    assert not client.transport.calls


def test_combined_frame_budget_includes_all_images(client, tmp_path):
    frames = []
    for index in range(2):
        path = tmp_path / f"large-{index}.png"
        with path.open("wb") as output:
            output.truncate(8 * 1024 * 1024)
        frames.append({"path": path, "timestamp_seconds": index * .125, "frame_index": index})
    with pytest.raises(ProviderError, match="inline_limit_exceeded"):
        client.review_frames(frames, "Review", SCHEMA)
    assert not client.transport.calls
    assert all(frame["path"].stat().st_size == 8 * 1024 * 1024 for frame in frames)
