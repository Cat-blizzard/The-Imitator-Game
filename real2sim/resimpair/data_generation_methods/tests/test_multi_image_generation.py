import base64
import copy

import pytest

from data_generation_methods.r2h_pipeline.providers import GeminiClient, ProviderError


@pytest.fixture
def image_client(tmp_path, monkeypatch):
    monkeypatch.setenv("R2H_IMAGE_TEST_KEY", "test-not-real")
    images = []
    for name, content in (("edit.png", b"edit"), ("reference.png", b"reference")):
        path = tmp_path / name
        path.write_bytes(content)
        images.append(path)
    calls = []
    def transport(method, url, *, headers, json, timeout):
        calls.append(copy.deepcopy(json))
        return {"candidates": [{"finishReason": "STOP", "content": {"parts": [{
            "inlineData": {"mimeType": "image/png",
                           "data": base64.b64encode(b"generated").decode()}}]}}]}
    client = GeminiClient({"model": "gemini-3.1-pro-preview",
                           "api_key_env": "R2H_IMAGE_TEST_KEY"}, transport=transport)
    return client, images, calls


def test_edit_and_reference_are_distinct_parts(image_client):
    client, images, calls = image_client
    result = client.generate_image_with_references(images[0], [images[1]], "Keep the visible state.")
    parts = calls[0]["contents"][0]["parts"]
    assert base64.b64decode(parts[0]["inlineData"]["data"]) == b"edit"
    assert parts[1]["text"] == "SUPPORTING REFERENCE IMAGE 1:"
    assert base64.b64decode(parts[2]["inlineData"]["data"]) == b"reference"
    assert parts[-1]["text"] == "Keep the visible state."
    assert len(result["images"]) == 1


def test_original_single_image_interface_keeps_payload(image_client):
    client, images, calls = image_client
    client.generate_image(images[0], "Edit only the arm.")
    parts = calls[0]["contents"][0]["parts"]
    assert len(parts) == 2
    assert parts[1] == {"text": "Edit only the arm."}


def test_reference_overflow_fails_before_call(image_client):
    client, images, calls = image_client
    with pytest.raises(ProviderError) as error:
        client.generate_image_with_references(images[0], [images[1]] * 9, "Edit.")
    assert error.value.code == "invalid_image_references"
    assert not calls
