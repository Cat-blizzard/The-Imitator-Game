"""One structured, bounded API for CLI and agent tool integrations.

This uses the retained SAPIEN backend. The JSON-lines transport is explicitly
not MCP; clients can call these functions without installing another runtime.
"""
import inspect
import json

MAX_RESPONSE_BYTES = 2 * 2**20
MAX_REQUEST_BYTES = 2**20
MUTATING_TOOLS = {"repair_propose", "scene_apply_revision", "scene_apply_release", "execution_prepare", "execution_run"}


def _tools():
    from .observe import scene_measure, run_status, run_observe, run_diagnose
    from .execution import describe_profiles, campaign_status, prepare, run
    from .repair import propose_repairs
    from .revisions import apply_proposal
    from .release_variants import apply_release_variant
    return {
        "scene_measure": scene_measure, "run_status": run_status,
        "run_observe": run_observe, "run_diagnose": run_diagnose,
        "repair_propose": propose_repairs, "scene_apply_revision": apply_proposal,
        "scene_apply_release": apply_release_variant,
        "execution_profiles": describe_profiles, "campaign_status": campaign_status,
        "execution_prepare": prepare, "execution_run": run,
    }


def capabilities():
    read_only = {"scene_measure", "run_status", "run_observe", "run_diagnose",
                 "execution_profiles", "campaign_status"}
    return {"schema": "resimpair_scene_tools_v1", "backend": "retained_sapien_maniskill",
        "transport": "python_or_local_json_lines; not_MCP",
        "model_API_calls": 0,
        "tools": [{"name": name, "signature": str(inspect.signature(function)),
                   "read_only": name in read_only,
                   "physical_execution": name == "execution_run"}
                  for name, function in _tools().items()]}


def call(method, params=None):
    methods = _tools()
    if method not in methods:
        raise ValueError("unknown_scene_tool")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError("tool_params_must_be_an_object")
    if len(json.dumps(params, ensure_ascii=False, allow_nan=False).encode()) > MAX_REQUEST_BYTES:
        raise ValueError("request_exceeds_one_MiB")
    function = methods[method]
    inspect.signature(function).bind(**params)
    result = function(**params)
    result_size = len(json.dumps(result, ensure_ascii=False, allow_nan=False).encode())
    if result_size > MAX_RESPONSE_BYTES:
        if method not in MUTATING_TOOLS:
            raise ValueError("response_exceeds_two_MiB; narrow_the_query_or_read_saved_artifact")
        # A completed mutation must not masquerade as a failed submission.
        return {"status": "operation_returned_result_omitted", "method": method,
                "result_bytes": result_size, "do_not_resubmit": True,
                "physical_success": "unknown; consult independent audit",
                "recover_from": {k: params[k] for k in ("output", "cohort", "campaign", "name") if k in params},
                "reason": "result_exceeds_two_MiB; query_saved_artifacts_or_campaign_status"}
    return result
