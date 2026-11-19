"""Wire-only Gemini schema compatibility; original local validators stay intact.

This file is copied into an immutable retained package. Its sibling
``review_transport`` performs the already-qualified v1 transformation first.
Only ``additionalProperties`` is omitted from the outgoing schema, never from
the request's original response_schema or its locally enforced acceptance.
"""
from copy import deepcopy
import importlib

VERSION = "gemini_additional_properties_wire_omission_v2"


def omit_additional_properties(schema):
    result = deepcopy(schema)
    edits = []

    def visit(node, path):
        if not isinstance(node, dict):
            return
        if "additionalProperties" in node:
            value = node.pop("additionalProperties")
            edits.append({"path": path + ["additionalProperties"],
                          "action": "omit_wire_only", "original_value": value})
        for name in ("properties", "$defs", "definitions", "patternProperties", "dependentSchemas"):
            for key, child in node.get(name, {}).items():
                visit(child, path + [name, key])
        for name in ("items", "contains", "not", "if", "then", "else", "propertyNames", "unevaluatedProperties"):
            visit(node.get(name), path + [name])
        for name in ("anyOf", "oneOf", "allOf", "prefixItems"):
            for i, child in enumerate(node.get(name, [])):
                visit(child, path + [name, i])

    visit(result, [])
    return result, edits


def _retained():
    return importlib.import_module(__package__ + ".review_transport")


def wire_schema(schema, adapter=None):
    from .core import digest
    prior, receipt = _retained().wire_schema(schema, adapter)
    encoded, edits = omit_additional_properties(prior)
    return encoded, {**receipt, "adapter": VERSION,
        "retained_adapter": receipt["adapter"],
        "retained_wire_schema_sha256": receipt["wire_schema_sha256"],
        "wire_schema_sha256": digest(encoded),
        "transformations": receipt["transformations"] + edits,
        "original_schema_enforced_locally": True, "acceptance_rules_changed": False}


def generation_config(config, schema, *, wire_schema_adapter=None):
    # Retained function verifies the historical route/model qualification and
    # requested v1 dialect. No new route, protocol, or transport fallback.
    prior = _retained().generation_config(config, schema, wire_schema_adapter=wire_schema_adapter)
    parameter = config["gemini"].get("review_schema_parameter", "responseJsonSchema")
    if parameter != "responseSchema":
        raise ValueError("wire_v2_requires_explicit_qualified_responseSchema_route")
    encoded, _ = wire_schema(schema, wire_schema_adapter)
    return {**prior, parameter: encoded}
