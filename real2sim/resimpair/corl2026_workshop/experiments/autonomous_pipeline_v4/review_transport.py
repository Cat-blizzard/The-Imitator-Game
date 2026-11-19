"""Opt-in schema parameter compatibility without changing visual predicates."""
from copy import deepcopy
from pathlib import Path

from jsonschema import validate

from .core import checked, digest, read, require

WIRE_SCHEMA_ADAPTER = "gemini_response_schema_subset_v1"


def wire_schema(schema, adapter=None):
    """Encode the legacy provider dialect without changing local acceptance.

    Only two representational differences are allowed by this opt-in version.
    The original schema must still validate every returned response locally.
    """
    require(adapter in {None, WIRE_SCHEMA_ADAPTER}, "unsupported_wire_schema_adapter")
    result = deepcopy(schema); edits = []
    if adapter is not None:
        def visit(node, path):
            if "uniqueItems" in node:
                edits.append({"path": path+["uniqueItems"], "action": "omit_wire_only",
                              "original_value": node.pop("uniqueItems")})
            if "enum" in node and "type" not in node:
                require(node["enum"] and all(isinstance(v, str) for v in node["enum"]),
                        "wire_adapter_requires_pure_string_enum")
                node["type"] = "string"
                edits.append({"path": path+["type"], "action": "explicit_existing_string_enum_type"})
            for key in ["properties", "$defs", "definitions"]:
                for name, value in node.get(key, {}).items(): visit(value, path+[key, name])
            if isinstance(node.get("items"), dict): visit(node["items"], path+["items"])
            for key in ["anyOf", "oneOf", "allOf", "prefixItems"]:
                for i, value in enumerate(node.get(key, [])): visit(value, path+[key, i])
        visit(result, [])
    return result, {"adapter": adapter, "original_schema_sha256": digest(schema),
                    "wire_schema_sha256": digest(result), "transformations": edits,
                    "original_schema_enforced_locally": True, "acceptance_rules_changed": False}


def generation_config(config, schema, *, wire_schema_adapter=None):
    settings = config["gemini"]
    parameter = settings.get("review_schema_parameter", "responseJsonSchema")
    require(parameter in {"responseJsonSchema", "responseSchema"}, "unsupported_review_schema_parameter")
    if parameter != "responseJsonSchema":
        require("review_schema_qualification" in settings, "alternate_schema_parameter_requires_actual_probe")
        proof = read(checked(settings["review_schema_qualification"]))
        require(proof["parameter"] == parameter, "schema_probe_parameter_mismatch")
        prior_config = read(checked(proof["model_config"]))["gemini"]
        require(all(settings[k] == prior_config[k] for k in ["model", "base_url"]),
                "schema_probe_model_or_route_changed")
        results = read(checked(proof["results"]))
        for prefix in ["flat", "nested"]:
            rows = [r for r in results["rows"] if r["id"] == prefix + "_" + parameter]
            require(len(rows) == 1 and rows[0]["status"] == "schema_conformant", "required_schema_probe_failed")
            row = rows[0]
            request = read(checked(row["request"]))
            result_path = Path(checked(row["result"]))
            validate(read(result_path)["review"], request["response_schema"])
            audit = read(result_path.parent / "actual_input_audit.json")
            require(audit["verified"] is True and audit["request"] == row["request"]
                    and audit["generation_config"] == {"responseMimeType": "application/json", parameter: request["response_schema"]},
                    "schema_probe_wire_evidence_mismatch")
    require(wire_schema_adapter is None or parameter == "responseSchema", "wire_adapter_requires_responseSchema")
    encoded, _ = wire_schema(schema, wire_schema_adapter)
    return {"responseMimeType": "application/json", parameter: encoded}
