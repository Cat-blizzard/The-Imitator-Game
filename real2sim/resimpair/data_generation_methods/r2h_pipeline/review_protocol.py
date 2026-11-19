"""Auditable container-only compatibility for a saved provider review response.

This adapter never calls a provider and never changes the saved response. Its
only supported shape difference is a fully specified event object where the
original strict schema requires an array of such objects. It may wrap that
object in a singleton list. It cannot supply fields, translate names, coerce
types, change truth values, reconstruct times or add evidence.

The returned sidecar is named ``protocol_shape_adapter_v1``. Raw provider schema
compliance remains false even when the derived review passes the original
schema. Passing that schema does not establish task correctness or data release.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

from .providers import GeminiClient, ProviderError, parse_review_json
from .task_evidence import canonical_sha256
from .task_review import build_review_schema


class ReviewProtocolError(ValueError):
    """An unsupported or ambiguous protocol difference; no review is produced."""


def adapt_provider_review_shape(provider_response_path, contract):
    """Return an auditable sidecar for the one supported wire-format mismatch.

    ``provider_response_path`` is the original saved provider response JSON,
    including its candidates/content/parts envelope. Save the returned object
    separately as ``protocol_shape_adapter_v1.json``. Its ``review`` value can
    then be passed through the normal evidence normalizer; all frame, media,
    contract and temporal checks remain necessary.

    No sidecar is returned for an already valid response (use the ordinary
    parser), a truncated/blocked response, multiple candidates, or any other
    schema difference. The original schema is applied before and after wrapping,
    and every wrapped object's complete contents are validated independently.
    """
    from jsonschema import validators

    path = Path(provider_response_path).resolve(strict=True)
    raw_bytes = path.read_bytes()
    schema = build_review_schema(contract)
    validator_class = validators.validator_for(schema)
    validator_class.check_schema(schema)
    validator = validator_class(schema)
    try:
        envelope = parse_review_json(raw_bytes.decode("utf-8"))["review"]
        candidates = envelope.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 1:
            raise ReviewProtocolError("Require exactly one saved response candidate")
        parts = GeminiClient._parts(envelope)
        raw_text = "".join(part.get("text", "") for part in parts
                           if not part.get("thought") and isinstance(part.get("text"), str))
        parsed = parse_review_json(raw_text, allow_json_fence=True)
    except (ProviderError, UnicodeError) as error:
        raise ReviewProtocolError("Saved response is not one complete review JSON object") from error
    original = parsed["review"]
    if set(original) != set(schema["properties"]):
        raise ReviewProtocolError("Top-level review keys must already match the strict schema")
    if not isinstance(original["events"], dict):
        raise ReviewProtocolError("The events container must already be an object")
    event_schema = schema["properties"]["events"]["properties"]
    if set(original["events"]) != set(event_schema):
        raise ReviewProtocolError("Event IDs must already match the strict schema")

    errors = list(validator.iter_errors(original))
    if not errors:
        raise ReviewProtocolError("Response already satisfies the schema; no adapter is needed")
    for error in errors:
        location = list(error.absolute_path)
        if not (len(location) == 2 and location[0] == "events"
                and location[1] in event_schema and error.validator == "type"
                and error.validator_value == "array" and isinstance(error.instance, dict)):
            raise ReviewProtocolError("Unsupported schema difference; only event object-to-array is allowed")

    review, changes = deepcopy(original), []
    for key, value in original["events"].items():
        if not isinstance(value, dict):
            continue
        item_validator = validator_class(event_schema[key]["items"])
        if not item_validator.is_valid(value):
            raise ReviewProtocolError("Singleton event is incomplete or invalid; fields cannot be repaired")
        review["events"][key] = [deepcopy(value)]
        # Equal hashes certify that the wrapped item's contents are unchanged.
        item_digest = canonical_sha256(value)
        changes.append({
            "path": "/events/" + key,
            "operation": "wrap_object_in_singleton_array",
            "before_type": "object", "after_type": "array", "after_length": 1,
            "original_item_sha256": item_digest,
            "wrapped_item_sha256": canonical_sha256(review["events"][key][0]),
            "item_contents_changed": False,
        })
    if not changes or not validator.is_valid(review):
        raise ReviewProtocolError("Derived review does not satisfy the original strict schema")
    return {
        "schema": "r2h_review_protocol_shape_adapter_v1",
        "adapter": "protocol_shape_adapter_v1",
        "review": review,
        "changes": changes,
        "raw_artifact": {"path": str(path), "sha256": hashlib.sha256(raw_bytes).hexdigest()},
        "raw_text_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        "contract_sha256": canonical_sha256(contract),
        "response_schema_sha256": canonical_sha256(schema),
        "original_parsed_review_sha256": canonical_sha256(original),
        "canonical_review_sha256": canonical_sha256(review),
        "raw_text_normalization": parsed["normalization"],
        "raw_response_modified": False,
        "provider_schema_compliant": False,
        "schema_validated": True,
        "schema_validation_scope": "derived_review_after_recorded_container_wrapping_only",
        "values_fields_timestamps_evidence_modified": False,
        "task_correctness": "not_evaluated",
        "training_admitted": False,
        "quality_release_approved": False,
        "provider_calls": 0,
    }
