"""Read-only provider credit measurements, separate from token-price estimates.

Only allowlisted numeric fields leave the response. A token-wide counter delta
is an observed debit interval, not a request-level invoice or an RMB payment.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit, quote
from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler

from .budget import Budget
from .core import bind, read, require, write


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _get(origin, endpoint, key=None):
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    try:
        with build_opener(ProxyHandler({}), NoRedirect()).open(
                Request(origin + endpoint, headers=headers), timeout=25) as response:
            raw = response.read(4 * 1024 * 1024 + 1)
            require(len(raw) <= 4 * 1024 * 1024, "billing_response_too_large")
            return {"http_status": response.status, "data": json.loads(raw)}
    except Exception as exc:
        return {"http_status": getattr(exc, "code", None), "error_type": type(exc).__name__}


def _numeric(data, names):
    return {k: data[k] for k in names if isinstance(data, dict) and type(data.get(k)) in (int, float, bool)}


def connection(root):
    from .models import configured
    config = configured(root)
    url = urlsplit(config["gemini"]["base_url"])
    video_url = urlsplit(config["seedance"]["base_url"])
    require(url.scheme == "https" and url.hostname == "api.echoflowai.cc"
            and video_url.hostname == url.hostname, "billing_origin_not_predeclared")
    env = config["gemini"]["api_key_env"]
    require(config["seedance"]["api_key_env"] == env, "billing_requires_same_token_for_models")
    key = os.environ.get(env)
    require(bool(key), "configured_credential_missing")
    return config, "https://" + url.netloc, key


def provider_metadata(root):
    root = Path(root)
    dest = root / "billing/provider_metadata.json"
    if dest.exists():
        return read(dest)
    config, origin, _ = connection(root)
    status, pricing = _get(origin, "/api/status"), _get(origin, "/api/pricing")
    body = status.get("data", {}).get("data", {})
    price = pricing.get("data", {})
    models = {config["gemini"]["model"], config["seedance"]["model"], config["anchor_model"]}
    fields = {"model_name", "quota_type", "model_ratio", "model_price", "completion_ratio", "cache_ratio", "enable_groups"}
    result = {"schema": "provider_public_billing_metadata_v1", "queried_unix": time.time(),
              "origin": origin, "status_http": status.get("http_status"), "pricing_http": pricing.get("http_status"),
              "public_units": _numeric(body, ["quota_per_unit", "usd_exchange_rate", "display_in_currency"]),
              "model_rows": [{k: v for k, v in row.items() if k in fields}
                             for row in price.get("data", []) if row.get("model_name") in models],
              "scope": "public quote/configuration; not a request invoice or cash exchange transaction",
              "secrets_saved": False}
    groups = {g for row in result["model_rows"] for g in row.get("enable_groups", [])}
    result["group_ratios"] = {g: v for g, v in price.get("group_ratio", {}).items() if g in groups}
    write(dest, result)
    return result


def snapshot(root, path):
    path = Path(path)
    if path.exists():
        return read(path)
    _, origin, key = connection(root)
    start = time.time()
    usage = _get(origin, "/api/usage/token/", key)
    billing = _get(origin, "/v1/dashboard/billing/usage", key)
    body = usage.get("data", {})
    result = {"schema": "provider_credit_snapshot_v1", "started_unix": start, "finished_unix": time.time(),
              "token_usage_http": usage.get("http_status"), "token_usage_success": body.get("success") is True,
              "token_quota": _numeric(body.get("data", {}), ["total_used", "total_available", "total_granted", "unlimited_quota"]),
              "compatible_billing_http": billing.get("http_status"),
              "compatible_billing": _numeric(billing.get("data", {}), ["total_usage"]),
              "scope": "whole configured API token; concurrent external usage and settlement lag are not excluded",
              "credential_handling": "private environment file in memory; Bearer header only; redirects disabled"}
    write(path, result)
    return result


def difference(before, after, metadata):
    a, b = before.get("token_quota", {}).get("total_used"), after.get("token_quota", {}).get("total_used")
    delta = b - a if type(a) in (int, float) and type(b) in (int, float) else None
    unit = metadata.get("public_units", {}).get("quota_per_unit")
    nominal = delta / unit if delta is not None and type(unit) in (int, float) and unit > 0 else None
    return {"observed_token_quota_delta": delta, "nominal_usd_using_public_quota_unit": nominal,
            "actual_rmb_charge": None, "request_invoice_verified": False,
            "zero_delta_proves_free_call": False,
            "attribution": "serial local stage interval; external token use and delayed billing remain unverified"}


def token_logs(root, path, start, end=None):
    """Legacy provider endpoint: credential stays in memory, never in saved URLs.

    The provider's Bearer variant was tested and denied. Its documented legacy
    query-key route returns paginated items. Read only; redirects are forbidden.
    """
    path = Path(path)
    if path.exists():
        return read(path)
    config, origin, key = connection(root)
    end = int(end or time.time())
    models = {config["gemini"]["model"], config["seedance"]["model"], config["anchor_model"]}
    response = _get(origin, "/api/log/token?key=" + quote(key, safe="")
                    + f"&start_timestamp={int(start)}&end_timestamp={end}&p=1&page_size=1000")
    body = response.get("data", {})
    data = body.get("data", {})
    sequence = data if isinstance(data, list) else data.get("items", []) if isinstance(data, dict) else []
    rows = []
    for row in sequence:
        if not isinstance(row, dict) or not start <= row.get("created_at", 0) <= end or row.get("model_name") not in models:
            continue
        safe = _numeric(row, ["id", "created_at", "type", "quota", "prompt_tokens", "completion_tokens", "use_time", "is_stream"])
        safe["model_name"] = row["model_name"]
        try:
            other = json.loads(row.get("other", "{}"))
        except (ValueError, TypeError):
            other = {}
        safe["billing_details"] = _numeric(other, ["model_ratio", "group_ratio", "completion_ratio", "model_price", "cache_ratio"])
        rows.append(safe)
    result = {"schema": "provider_token_log_receipt_v1", "queried_unix": time.time(),
              "endpoint_path": "/api/log/token", "success": body.get("success") is True,
              "http_status": response.get("http_status"), "query_window": [start, end],
              "returned_rows": len(sequence), "server_total": data.get("total") if isinstance(data, dict) else len(sequence),
              "rows": rows, "raw_account_content_saved": False,
              "credential_handling": "legacy documented query in memory only; full URL and key never logged; no redirects",
              "scope": "provider consume/refund records; exact request association requires independent local interval matching"}
    write(path, result)
    return result


def validate_retired_billed_review(stage):
    """Permit different calls after a uniquely billed, irrecoverable response loss.

    This NEVER fabricates a result or permits resubmitting the lost request.
    The original unknown failure and all costs remain immutable evidence.
    """
    from .core import checked
    stage = Path(stage)
    receipt_path = stage / "terminal_resolution.json"
    if not receipt_path.exists():
        return False
    receipt = read(receipt_path)
    require(receipt["resolution"] == "billed_terminal_output_unavailable_no_resubmission", "invalid_billing_resolution")
    require(receipt["call_started"] == bind(stage / "call_started.json")
            and receipt["original_failure"] == bind(stage / "failure.json"), "resolution_call_binding_changed")
    started, failure = read(stage / "call_started.json"), read(stage / "failure.json")
    require(started["kind"] == "review" and failure["provider_outcome"] == "unknown"
            and not (stage / "result.json").exists(), "only_lost_review_response_can_be_retired_here")
    logs = read(checked(receipt["provider_logs"]))
    before = read(checked(receipt["credit_before"]))
    after = read(checked(receipt["credit_after"]))
    delta = after["token_quota"]["total_used"] - before["token_quota"]["total_used"]
    matches = [row for row in logs["rows"] if row.get("type") == 2 and row.get("model_name") == receipt["model"]
               and started["started_unix"] <= row["created_at"] <= after["finished_unix"]
               and row.get("quota") == delta and delta > 0]
    require(logs["success"] and len(matches) == 1 and matches[0]["id"] == receipt["provider_log_id"],
            "billed_review_resolution_requires_unique_model_time_quota_match")
    require(receipt["quality_result"] == "unknown" and receipt["resubmit_allowed"] is False,
            "billing_must_not_invent_quality_or_authorize_duplicate_post")
    return True


class MeteredBudget(Budget):
    """Existing immutable cumulative admission plus read-only before/after data."""
    def __init__(self, root):
        self.root = Path(root)
        super().__init__(self.root / "budget.json")
        self.metadata = provider_metadata(self.root)

    def call(self, kind, request, operation):
        def metered(stage):
            before = snapshot(self.root, stage / "credit_before.json")
            try:
                return operation(stage)
            finally:
                after = snapshot(self.root, stage / "credit_after_response.json")
                write(stage / "credit_response_interval.json", difference(before, after, self.metadata))
        return super().call(kind, request, metered)

    def settle_video(self, stage):
        stage = Path(stage)
        require((stage / "video_terminal.json").exists(), "cannot_settle_inflight_video")
        after = snapshot(self.root, stage / "credit_after_terminal.json")
        before = read(stage / "credit_before.json")
        write(stage / "credit_terminal_interval.json", difference(before, after, self.metadata))


def report(root, output):
    root = Path(root)
    budget = Budget(root / "budget.json")
    metadata = read(root / "billing/provider_metadata.json")
    rows = []
    for marker in sorted(budget.ledger.glob("stages/*/call_started.json"), key=lambda p: read(p)["started_unix"]):
        stage, started = marker.parent, read(marker)
        intervals = [stage / "credit_terminal_interval.json", stage / "credit_response_interval.json"]
        interval = next((p for p in intervals if p.exists()), None)
        response = read(stage / "provider_response.json") if (stage / "provider_response.json").exists() else {}
        terminal = read(stage / "video_terminal.json") if (stage / "video_terminal.json").exists() else {}
        usage = response.get("usageMetadata", response.get("usage", terminal.get("usage", {})))
        usage = _numeric(usage, list(usage)) if isinstance(usage, dict) else {}
        rows.append({"kind": started["kind"], "started": bind(marker), "started_unix": started["started_unix"],
                     "provider_usage": usage, "credit_interval": read(interval) if interval else None,
                     "video_status": terminal.get("status"), "failure": bind(stage / "failure.json") if (stage / "failure.json").exists() else None,
                     "result_recorded": (stage / "result.json").exists()})
    intervals = [r["credit_interval"] for r in rows]
    known = all(i is not None and i["observed_token_quota_delta"] is not None for i in intervals)
    result = {"schema": "paid_campaign_usage_v1", "budget": bind(root / "budget.json"), **budget.usage(),
              "rows": rows, "all_started_including_failures_counted": True,
              "sum_observed_stage_quota_deltas": sum(i["observed_token_quota_delta"] for i in intervals) if known else None,
              "sum_nominal_usd_stage_deltas": sum(i["nominal_usd_using_public_quota_unit"] for i in intervals)
                  if known and all(i["nominal_usd_using_public_quota_unit"] is not None for i in intervals) else None,
              "actual_rmb_charge": None, "request_level_bill_verified": False,
              "provider_metadata": bind(root / "billing/provider_metadata.json")}
    if (root / "billing/campaign_after.json").exists():
        before = read(root / "billing/campaign_before.json")
        after = read(root / "billing/campaign_after.json")
        logs = token_logs(root, root / "billing/provider_final_logs.json", before["started_unix"], after["finished_unix"] + 1)
        consumes = [r for r in logs["rows"] if r.get("type") == 2]
        logged = sum(r.get("quota", 0) for r in consumes) if logs["success"] else None
        unit = metadata.get("public_units", {}).get("quota_per_unit")
        result["provider_consume_log_records"] = consumes
        result["provider_logs"] = bind(root / "billing/provider_final_logs.json")
        result["provider_logged_quota"] = logged
        result["provider_logged_nominal_usd"] = logged / unit if logged is not None and unit else None
        result["provider_log_quota_matches_observed_stage_deltas"] = logged == result["sum_observed_stage_quota_deltas"] if logged is not None else None
        result["request_matching_limit"] = "Model/time/quota interval association; no recovered provider response for retired review and no RMB cash invoice"
    write(output, result)
    return result
