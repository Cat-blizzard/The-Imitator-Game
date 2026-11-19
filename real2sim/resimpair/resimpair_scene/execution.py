"""Frozen execution profiles over the retained, finite campaign supervisors.

This module does not register allocations, sample scenes, retry requests, or
evaluate physics. Preparation consumes *allocated* capacity, even before a
worker is submitted. Execution delegates locks, timeouts and reservation
handling to the original driver selected at preparation time.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import fcntl
import importlib
from pathlib import Path
import re
from types import MappingProxyType

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write


VERSION = "scene_execution_v1"
DEFAULT_PROFILE = "checked_withdrawal"
ALLOCATION_REGISTRY = ROOT / "runtime/production_integration_v1"


@dataclass(frozen=True)
class ExecutionProfile:
    module: str
    status: str
    description: str
    prefix_reuse: bool = True
    full_open: bool = True


# Configuration, not a second controller implementation. Candidate selection
# is always explicit; a successful run cannot change DEFAULT_PROFILE.
PROFILE_REGISTRY = MappingProxyType({
    "checked_withdrawal": ExecutionProfile(
        "resimpair_tasks.production_integration_v4", "default",
        "Native geometry, checked opening and withdrawal; retained baseline."),
    "contact_latch": ExecutionProfile(
        "resimpair_tasks.production_integration_v6", "candidate",
        "Adds cached-width contact-latched closure; explicit opt-in."),
    "conditional_descent": ExecutionProfile(
        "resimpair_tasks.production_integration_v8", "candidate",
        "Adds conditional payload-centre descent and contact latch; explicit opt-in."),
    "task_release": ExecutionProfile(
        "resimpair_tasks.production_integration_v2", "candidate",
        "Retained task-selected supported or above-rim release; measured held geometry and operation-local full opening."),
})


def _profile(name):
    if name not in PROFILE_REGISTRY:
        raise ValueError("unknown_execution_profile")
    config = PROFILE_REGISTRY[name]
    driver = importlib.import_module(config.module)
    return driver, {"name": name, "status": config.status,
        "description": config.description, "module": config.module,
        "driver": binding(driver.__file__),
        "prepare_options": {"prefix_reuse": config.prefix_reuse, "full_open": config.full_open}}


def describe_profiles():
    """Return JSON-ready profile configuration and the unchanged default."""
    return {"schema": VERSION, "default_profile": DEFAULT_PROFILE,
        "profiles": {name: _profile(name)[1] for name in PROFILE_REGISTRY}}


def _component(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", value):
        raise ValueError("single_safe_path_component_required")
    return value


def _bindings(value):
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            yield value
        else:
            for child in value.values():
                yield from _bindings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _bindings(child)


def _cases(cases, *, verify_files=False):
    if not isinstance(cases, (list, tuple)) or not cases:
        raise ValueError("finite_nonempty_cases_required")
    result = deepcopy(list(cases))
    ids = []
    for case in result:
        ids.append(_component(case["id"]))
        if case.get("execution_id", case["id"]) != case["id"]:
            raise ValueError("case_and_execution_id_must_match")
        for key in ("scene", "task"):
            if not isinstance(case.get(key), dict) or not {"path", "sha256"} <= case[key].keys():
                raise ValueError("bound_scene_and_task_required")
        if verify_files:
            for ref in _bindings(case):
                resolve(ref)
    if len(set(ids)) != len(ids):
        raise ValueError("unique_case_ids_required")
    digest(result)  # Reject non-JSON / non-finite configuration before writing.
    return result


def _campaign(campaign):
    root = Path(campaign).resolve()
    budget_path, allocation_path = root / "budget/limits.json", root / "allocation.json"
    if not budget_path.is_file() or not allocation_path.is_file():
        raise ValueError("existing_registered_campaign_required")
    budget, allocation = read(budget_path), read(allocation_path)
    receipt_path = resolve(allocation["registration"]).resolve()
    receipt = read(receipt_path)
    allocation_id = _component(budget["allocation_id"])
    if (receipt_path != (ALLOCATION_REGISTRY / (allocation_id + ".json")).resolve()
            or receipt.get("root") != str(root)
            or receipt.get("budget") != binding(budget_path)
            or receipt.get("allocation_id") != allocation_id
            or type(receipt.get("limit")) is not int
            or receipt["limit"] != budget.get("physical_attempt_limit")):
        raise ValueError("registered_allocation_root_budget_or_limit_mismatch")
    limit = budget["physical_attempt_limit"]
    if type(limit) is not int or limit < 0:
        raise ValueError("finite_physical_attempt_limit_required")
    if (budget.get("parallel_workers") != 1 or budget.get("retries_per_request") != 0
            or budget.get("API_calls") != 0 or budget.get("GPU_jobs") != 0
            or type(budget.get("timeout_seconds")) not in (int, float)
            or not 0 < budget["timeout_seconds"] < float("inf")):
        raise ValueError("retained_serial_CPU_no_retry_budget_required")
    if ("prior_scope_limit" in budget and "cumulative_scope_limit" in budget
            and budget["prior_scope_limit"] + limit != budget["cumulative_scope_limit"]):
        raise ValueError("cumulative_allocation_limit_mismatch")
    return root, budget, {"budget": binding(budget_path), "allocation": binding(allocation_path),
                          "registration": binding(receipt_path)}


def _cohort(folder, campaign, refs):
    if folder.is_symlink() or folder.resolve().parent != campaign:
        raise ValueError("cohort_must_belong_to_registered_campaign")
    frozen = read(folder / "frozen_inputs.json")
    for name in ("experiment", "cohort", "rules", "protocol"):
        if frozen.get(name) != binding(folder / (name + ".json")):
            raise ValueError("cohort_input_binding_mismatch")
    for ref in frozen.values():
        resolve(ref)
    protocol, spec = read(folder / "protocol.json"), read(folder / "experiment.json")
    if (protocol.get("campaign_root") != str(campaign)
            or protocol.get("allocation") != refs["allocation"]
            or protocol.get("budget") != refs["budget"]
            or Path(spec["budget"]).resolve() != campaign / "budget"):
        raise ValueError("cohort_campaign_binding_mismatch")
    resolve(protocol["driver"])
    cases = _cases(read(folder / "cohort.json")["cases"])
    metadata = folder / "scene_execution.json"
    if metadata.exists():
        document = read(metadata)
        request = read(resolve(document["request"]))
        if (request["campaign"] != str(campaign) or request["name"] != folder.name
                or request["cases"] != cases or request["campaign_bindings"] != refs
                or request["profile"] != _profile(request["profile"]["name"])[1]
                or request["profile"]["driver"] != protocol["driver"]
                or document["frozen_inputs"] != binding(folder / "frozen_inputs.json")):
            raise ValueError("frozen_execution_profile_or_request_mismatch")
    return {"root": folder, "cases": cases, "protocol": protocol, "experiment": spec}


def _snapshot(campaign):
    root, budget, refs = _campaign(campaign)
    cohorts = {p.parent.name: _cohort(p.parent, root, refs) for p in sorted(root.glob("*/cohort.json"))}
    intents = {}
    for path in sorted((root / ".scene_execution/requests").glob("*.json")):
        request = read(path)
        name = _component(request["name"])
        if (path.stem != name or request["campaign"] != str(root)
                or request["campaign_bindings"] != refs):
            raise ValueError("preparation_intent_campaign_binding_mismatch")
        request["cases"] = _cases(request["cases"])
        if name in cohorts and request["cases"] != cohorts[name]["cases"]:
            raise ValueError("preparation_intent_cohort_conflict")
        intents[name] = request
    incomplete = []
    for path in root.iterdir():
        if path.is_dir() and path.name not in cohorts and (path / "experiment.json").exists():
            incomplete.append(path.name)
    incomplete += [name for name in intents if not (root / name / "scene_execution.json").exists()]
    incomplete = sorted(set(incomplete))
    reservations = sorted((root / "budget/reservations").glob("*.json"))
    if [p.name for p in reservations] != [f"{i:04d}.json" for i in range(len(reservations))]:
        raise ValueError("reservation_sequence_gap_or_renamed_receipt")
    submitted, terminal = set(), set()
    for path in reservations:
        record = read(path)
        name, case_id = record["cohort"], record["case"]["id"]
        key = (name, case_id)
        if name not in cohorts or key in submitted:
            raise ValueError("duplicate_or_orphan_physical_reservation")
        cohort = cohorts[name]
        if (record["case"] not in cohort["cases"] or record.get("campaign") != str(cohort["root"])
                or record["protocol"] != binding(cohort["root"] / "protocol.json")):
            raise ValueError("reservation_case_or_protocol_mismatch")
        folder = cohort["root"] / "executions" / case_id
        if not (folder / "reservation.json").is_file() or read(folder / "reservation.json") != record:
            raise ValueError("physical_reservation_mirror_mismatch")
        submitted.add(key)
        if (folder / "terminal.json").is_file():
            result = read(folder / "terminal.json")
            if result.get("case") != case_id or result.get("physical_attempts") != 1:
                raise ValueError("terminal_attempt_identity_mismatch")
            terminal.add(key)
    for name, cohort in cohorts.items():
        for path in (cohort["root"] / "executions").glob("*/reservation.json"):
            if (name, path.parent.name) not in submitted:
                raise ValueError("local_reservation_missing_from_shared_budget")
        for path in (cohort["root"] / "executions").glob("*/terminal.json"):
            if (name, path.parent.name) not in terminal:
                raise ValueError("terminal_missing_from_shared_budget")
    allocated = sum(len(c["cases"]) for c in cohorts.values())
    allocated += sum(len(c["cases"]) for name, c in intents.items() if name not in cohorts)
    if max(allocated, len(submitted)) > budget["physical_attempt_limit"]:
        raise ValueError("campaign_allocation_exceeded")
    known_drivers = {digest(_profile(name)[1]["driver"]): name for name in PROFILE_REGISTRY}
    rows = []
    for name, cohort in cohorts.items():
        used = sum(n == name for n, _ in submitted)
        done = sum(n == name for n, _ in terminal)
        rows.append({"name": name, "root": str(cohort["root"]), "allocated_requests": len(cohort["cases"]),
            "submitted_attempts": used, "terminal_attempts": done, "unknown_or_active_attempts": used - done,
            "prepared_unsubmitted": len(cohort["cases"]) - used,
            "profile": known_drivers.get(digest(cohort["protocol"]["driver"]), "legacy_read_only"),
            "driver": cohort["protocol"]["driver"]})
    status = {"schema": VERSION, "campaign": str(root), "allocation_id": budget["allocation_id"],
        "bindings": refs, "budget": budget, "physical_attempt_limit": budget["physical_attempt_limit"],
        "allocated_requests": allocated, "submitted_attempts": len(submitted),
        "terminal_attempts": len(terminal), "unknown_or_active_attempts": len(submitted - terminal),
        "prepared_unsubmitted": allocated - len(submitted),
        "remaining_submission_allowance": budget["physical_attempt_limit"] - len(submitted),
        "remaining_unallocated": budget["physical_attempt_limit"] - allocated,
        "incomplete_preparations": incomplete, "admission_stopped": (root / "admission_stop.json").exists(),
        "cohorts": rows, "physical_success_not_assessed": True}
    return status, cohorts, intents


def campaign_status(campaign):
    """Read actual ledgers; prepared requests reserve capacity, successes are not inferred."""
    return _snapshot(campaign)[0]


@contextmanager
def _locked(campaign, *, preparing=False):
    # Same campaign supervisor lock during prepare prevents a preparation from
    # racing the retained launcher. During run the old driver owns that lock.
    paths = [campaign / ".scene_execution.lock"]
    if preparing:
        paths.append(campaign / ".supervisor.lock")
    handles = []
    try:
        for path in paths:
            handle = path.open("a+")
            handles.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        for handle in reversed(handles):
            handle.close()


def _request_keys(case):
    keys = {("id", case["id"]), ("input", case["scene"]["sha256"], case["task"]["sha256"])}
    if case.get("source_request") and case.get("requested_task_id"):
        keys.add(("source_request", case["source_request"]["sha256"], case["requested_task_id"]))
    return keys


def _matches_profile(cohort, profile):
    options = profile["prepare_options"]
    return (cohort["protocol"]["driver"] == profile["driver"]
        and cohort["protocol"].get("prefix_reuse") == options["prefix_reuse"]
        and cohort["experiment"].get("integration_prefix_reuse") == options["prefix_reuse"]
        and cohort["experiment"].get("integration_full_open") == options["full_open"])


def validate_release_profile(cases, profile):
    """Reject unsupported release composition before allocating an attempt.

    The retained v4/v6/v8 checked-withdrawal adapters accept supported release
    only. The original v2 routes each explicit low-drop operation through its
    measured-attachment waypoint and full-aperture opening implementation.
    """
    from resimpair_pickplace.above_rim_release_v1 import policy
    from resimpair_pickplace.drop_event_protocol_v1 import validate_contract
    for case in cases:
        task = read(resolve(case["task"]))
        operations = task.get("operations", [])
        for operation in operations:
            drop = policy(operation)
            if drop is None:
                continue
            if profile != "task_release":
                raise ValueError("explicit_drop_requires_task_release_profile; supported_withdrawal_is_incompatible")
            if validate_contract(task) is None:
                raise ValueError("drop_requires_frozen_supported_or_low_drop_contract")


def prepare(campaign, name, cases, profile=DEFAULT_PROFILE):
    """Freeze a finite cohort in an existing allocation; never sample or register budget.

    Repeating identical arguments for the same cohort is read-only/idempotent.
    Reusing its request under another name in the same profile, changing
    inputs, or an interrupted preparation requires explicit reconciliation.
    An explicitly selected different profile may form a paired comparison;
    both attempts consume allocation and their prior cohorts are recorded.
    """
    root, _, refs = _campaign(campaign)
    _component(name)
    if name in {"budget", "prepared_inputs", "executions", "runtime"}:
        raise ValueError("reserved_cohort_name")
    cases = _cases(cases, verify_files=True)
    driver, selected = _profile(profile)
    validate_release_profile(cases, profile)
    output = root / name
    with _locked(root, preparing=True):
        status, cohorts, intents = _snapshot(root)
        if name in cohorts:
            if (cohorts[name]["cases"] != cases or not _matches_profile(cohorts[name], selected)):
                raise ValueError("existing_cohort_name_input_or_profile_conflict")
            if name in status["incomplete_preparations"]:
                raise ValueError("incomplete_preparation_reconcile_without_retry")
            return {"root": str(output), "requests": len(cases), "profile": profile,
                    "reused": True, "campaign_status": status}
        if output.exists() or name in intents or status["incomplete_preparations"]:
            raise ValueError("incomplete_or_conflicting_preparation_reconcile_without_retry")
        if status["admission_stopped"] or status["unknown_or_active_attempts"]:
            raise ValueError("campaign_stopped_or_unknown_attempt_recover_no_retry")
        seen, cross_profile = set(), []
        requested_keys = set().union(*(_request_keys(case) for case in cases))
        for old_name, record in cohorts.items():
            same_profile = _matches_profile(record, selected)
            matched = []
            for case in record["cases"]:
                keys = _request_keys(case)
                if same_profile:
                    seen.update(keys)
                elif keys & requested_keys:
                    matched.append(case["id"])
            if matched:
                cross_profile.append({"cohort": old_name, "case_ids": matched,
                    "protocol": binding(record["root"] / "protocol.json"),
                    "driver": record["protocol"]["driver"]})
        for record in intents.values():
            if (record["profile"]["driver"] == selected["driver"]
                    and record["profile"]["prepare_options"] == selected["prepare_options"]):
                for case in record["cases"]:
                    seen.update(_request_keys(case))
        for case in cases:
            keys = _request_keys(case)
            if keys & seen:
                raise ValueError("request_already_allocated_recover_original_cohort")
            seen.update(keys)
        if len(cases) > status["remaining_unallocated"]:
            raise ValueError("prepared_and_submitted_requests_exceed_allocation")
        request_path = root / ".scene_execution/requests" / (name + ".json")
        write(request_path, {"schema": VERSION, "campaign": str(root), "name": name,
            "campaign_bindings": refs, "profile": selected, "cases": cases,
            "cross_profile_predecessors": cross_profile})
        # A crash after this intent still consumes its allocation. Never delete
        # it or choose another name in order to replace a failed preparation.
        driver.prepare_cohort(root, name, cases, **selected["prepare_options"])
        prepared = _cohort(output, root, refs)
        if prepared["cases"] != cases or not _matches_profile(prepared, selected):
            raise ValueError("driver_changed_frozen_request_or_profile")
        write(output / "scene_execution.json", {"schema": VERSION, "request": binding(request_path),
            "frozen_inputs": binding(output / "frozen_inputs.json")})
        status = campaign_status(root)
    return {"root": str(output), "requests": len(cases), "profile": profile,
            "reused": False, "campaign_status": status}


def run(cohort, limit):
    """Dispatch only the frozen profile to its retained supervisor.

    ``limit`` is a finite maximum of new submissions for this invocation. It
    cannot exceed this cohort's remaining requests or the registered budget.
    All-terminal repeats return zero without calling a worker.
    """
    output = Path(cohort).resolve()
    if type(limit) is not int or limit <= 0:
        raise ValueError("positive_bounded_launch_limit_required")
    protocol = read(output / "protocol.json")
    campaign, _, _ = _campaign(protocol["campaign_root"])
    if output.parent != campaign:
        raise ValueError("cohort_must_belong_to_registered_campaign")
    with _locked(campaign):
        status, cohorts, _ = _snapshot(campaign)
        current = cohorts[output.name]
        available = next(row for row in status["cohorts"] if row["name"] == output.name)
        if limit > min(status["physical_attempt_limit"], available["allocated_requests"]):
            raise ValueError("launch_limit_exceeds_frozen_cohort_or_allocation")
        if status["admission_stopped"] or status["unknown_or_active_attempts"]:
            raise ValueError("campaign_stopped_or_unknown_attempt_recover_no_retry")
        if status["incomplete_preparations"]:
            raise ValueError("incomplete_preparation_reconcile_without_retry")
        profile = available["profile"]
        if profile not in PROFILE_REGISTRY:
            raise ValueError("historical_driver_read_only_use_original_frozen_entrypoint")
        driver, selected = _profile(profile)
        if not _matches_profile(current, selected):
            raise ValueError("frozen_profile_configuration_mismatch")
        for case in current["cases"]:
            _cases([case], verify_files=True)
        validate_release_profile(current["cases"], profile)
        from resimpair_pickplace.runner import verify
        verify(output)
        if available["prepared_unsubmitted"] == 0:
            return {"root": str(output), "profile": profile, "new_attempts": 0, "all_attempts_terminal": True}
        if limit > min(available["prepared_unsubmitted"], status["remaining_submission_allowance"]):
            raise ValueError("launch_limit_exceeds_remaining_requests_or_budget")
        result = driver.run(output, limit=limit)
    return {**result, "root": str(output), "profile": profile}
