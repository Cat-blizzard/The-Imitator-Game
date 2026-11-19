"""Offline CLI/ledger integration fixtures; none certify real media or call APIs.

Every provider, video probe and download is replaced below. All accepted fixture
decisions stay in pytest's temporary directory and never enter a real catalog.
"""
from argparse import Namespace
from copy import deepcopy
import io
from pathlib import Path
import socket

import pytest

from data_generation_methods.r2h_pipeline import cli, providers
from data_generation_methods.r2h_pipeline.common import file_sha256, read_json, write_json
from data_generation_methods.r2h_pipeline.review import blind_review_prompt
from data_generation_methods.r2h_pipeline.store import Store


@pytest.fixture
def flow(tmp_path, monkeypatch):
    calls = {"submit": [], "poll": [], "review": [], "download": []}
    state = {"submit_error": None, "review_error": None, "poll_status": "succeeded"}

    def no_network(*args, **kwargs):
        raise AssertionError("Network access is forbidden in this fixture")

    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(cli.urllib.request, "urlopen", no_network)
    monkeypatch.setattr(providers, "_urllib_transport", no_network)
    monkeypatch.setenv("ARK_API_KEY", "fixture-only-not-a-real-key")
    monkeypatch.setenv("GEMINI_API_KEY", "fixture-only-not-a-real-key")

    def probe(path):
        path = Path(path)
        return {"path": str(path.resolve()), "sha256": file_sha256(path),
                "duration_seconds": 8., "fps": 24., "frames": 192,
                "width": 512, "height": 512, "full_decode_passed": True,
                "decode_passed": True}

    original_download = cli.download_video

    def download(url, path, *args, **kwargs):
        calls["download"].append(url)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"FAKE_VIDEO_FOR_UNIT_TEST_ONLY:" + url.encode())
        return probe(path)

    monkeypatch.setattr(cli, "probe_video", probe)
    monkeypatch.setattr(cli, "download_video", download)

    aliases = {"choice_a": "factory:middle_blue", "choice_b": "factory:bottom_brown",
               "choice_c": "factory:reverse_stack"}
    required = [
        {"id": "e1", "role": "block2", "verb": "pick_and_place", "destination_role": "brown_pad"},
        {"id": "e2", "role": "block1", "verb": "pick_and_place", "destination_role": "blue_pad"},
        {"id": "e3", "role": "block2", "verb": "pick_and_place", "destination_role": "block0"}]
    observed_events = []
    for i, event in enumerate(required):
        start = .5 + 2 * i
        observed_events.append({k: event[k] for k in ("role", "verb", "destination_role")})
        observed_events[-1].update(t_start=start, t_end=start + 1, confidence=.95,
                                  evidence=[{"timestamp_seconds": start + .5,
                                             "description": "Fixture-only grasp, placement and release."}])
    state["review"] = {"observed_goal": "choice_a", "goal_confidence": .95,
                       "window_scope": "full_task", "initial_state_visible": True,
                       "final_state_visible": True, "events": observed_events,
                       "issues": [], "uncertainty": .05}

    class FakeSeedance:
        def __init__(self, config):
            self.config = config

        def prepare_submit(self, payload):
            return {"json": deepcopy(payload)}

        def submit(self, payload):
            calls["submit"].append(deepcopy(payload))
            if state["submit_error"]:
                raise state["submit_error"]
            return {"status": "submitted", "task_id": "fixture-task-" + str(len(calls["submit"])),
                    "output_url": None, "usage": {}}

        def poll(self, task_id):
            calls["poll"].append(task_id)
            return {"status": state["poll_status"], "task_id": task_id,
                    "output_url": "https://fixture.invalid/" + task_id + ".mp4", "usage": {}}

    class FakeGemini:
        def __init__(self, config):
            pass

        def review(self, path, prompt, response_schema=None):
            calls["review"].append({"path": str(path), "prompt": prompt, "schema": response_schema})
            if state["review_error"]:
                raise state["review_error"]
            return {"status": "succeeded", "review": deepcopy(state["review"]), "usage": {},
                    "model": "fixture-not-a-real-model"}

    monkeypatch.setattr(providers, "SeedanceClient", FakeSeedance)
    monkeypatch.setattr(providers, "GeminiClient", FakeGemini)

    entries = []
    for job_id, group, goal, split in [
            ("fixture-a", "layout-a", aliases["choice_a"], "train"),
            ("fixture-b", "layout-b", aliases["choice_a"], "train"),
            ("fixture-other-goal", "layout-b", aliases["choice_b"], "train"),
            ("fixture-other-split", "layout-c", aliases["choice_a"], "validation")]:
        directory = tmp_path / job_id
        directory.mkdir()
        native = directory / "native.mp4"
        native.write_bytes(b"SOURCE_FIXTURE_ONLY")
        spec = {"schema": "r2h_review_spec_v1", "blind_goal_aliases": aliases,
                "allowed_roles": ["block0", "block1", "block2", "platform", "blue_pad", "brown_pad"],
                "allowed_verbs": ["pick_and_place"], "required_events": required,
                "event_order": [["e1", "e2"], ["e2", "e3"]], "expected_window_scope": "full_task"}
        write_json(directory / "contract.json", {"goal_id": goal, "window_scope": "full_task",
                                                  "review_spec": spec})
        write_json(directory / "evidence.json", {"fixture_only": True})
        write_json(directory / "frames/frames.json", {"fixture_only": True, "frames": []})
        (directory / "prompt.txt").write_text("Synthetic unit fixture generation prompt.")
        (directory / "blind_review_prompt.txt").write_text(blind_review_prompt())
        job = {"schema": "r2h_job_v1", "job_id": job_id, "group_id": group,
               "goal_id": goal, "split": split, "training_admitted": False,
               "source": {"goal_id": goal, "group_id": group, "split": split,
                          "episode_id": job_id, "fixture_only": True},
               "media": {"native_video": probe(native)}}
        for key, name in (("contract_path", "contract.json"), ("evidence_path", "evidence.json"),
                          ("prompt_path", "prompt.txt"), ("blind_review_prompt_path", "blind_review_prompt.txt")):
            job[key] = str(directory / name)
        write_json(directory / "job.json", job)
        artifact_names = ("job.json", "contract.json", "evidence.json", "prompt.txt",
                          "blind_review_prompt.txt", "frames/frames.json")
        entries.append({"job_id": job_id, "job_path": str(directory / "job.json"), "group_id": group,
                        "goal_id": goal, "split": split,
                        "artifact_sha256": {name: file_sha256(directory / name) for name in artifact_names}})
    catalog = {"schema": "r2h_catalog_v1", "jobs": entries, "training_admitted": False}
    store = Store(tmp_path / "ledger.sqlite3")
    store.register(catalog)
    image = tmp_path / "anchor.png"
    image.write_bytes(b"PNG_FIXTURE_ONLY")
    for group in ("layout-a", "layout-b", "layout-c"):
        store.bind_anchor(group, image)

    def args(job="fixture-a", **kwargs):
        return Namespace(job=job, mode="semantic", duration=8, video_uri=None, derived_media=None, **kwargs)

    def generate(job="fixture-a"):
        cli.plan(args(job), store, cli.DEFAULT_CONFIG)
        cli.submit(args(job), store, cli.DEFAULT_CONFIG)
        cli.poll(args(job), store, cli.DEFAULT_CONFIG)

    def accept(job="fixture-a"):
        generate(job)
        result = cli.review(args(job), store, cli.DEFAULT_CONFIG)
        assert result["status"] == "accepted_for_pairing", result

    yield Namespace(tmp=tmp_path, store=store, calls=calls, state=state, catalog=catalog,
                    args=args, generate=generate, accept=accept, cfg=cli.DEFAULT_CONFIG,
                    original_download=original_download)
    store.close()


def test_offline_lifecycle_exports_only_same_goal_split_other_layout(flow):
    flow.accept("fixture-a")
    flow.accept("fixture-b")
    output = flow.tmp / "pairs.json"
    result = cli.export_pairs(Namespace(output=str(output)), flow.store)
    assert result["pair_candidates"] == 2
    pairs = read_json(output)["pairs"]
    assert {(p["reference_job_id"], p["query_job_id"]) for p in pairs} == {
        ("fixture-a", "fixture-b"), ("fixture-b", "fixture-a")}
    assert all(p["training_admitted"] is False and p["human_wrist_gt_available"] is False for p in pairs)
    first = output.read_bytes()
    assert cli.export_pairs(Namespace(output=str(output)), flow.store) == result
    assert output.read_bytes() == first
    assert len(flow.calls["submit"]) == len(flow.calls["review"]) == 2
    # The fake reviewer receives the common blind prompt, not the private job.
    assert all("factory:" not in c["prompt"] and "fixture-a" not in c["prompt"] for c in flow.calls["review"])


def test_duplicate_submit_never_posts_twice(flow):
    cli.plan(flow.args(), flow.store, flow.cfg)
    cli.submit(flow.args(), flow.store, flow.cfg)
    with pytest.raises(ValueError):
        cli.submit(flow.args(), flow.store, flow.cfg)
    assert len(flow.calls["submit"]) == 1
    assert flow.store.get("fixture-a")["attempts"] == 1


def test_unknown_submission_cannot_be_reposted_or_replanned(flow):
    cli.plan(flow.args(), flow.store, flow.cfg)
    flow.state["submit_error"] = providers.ProviderError("fixture_timeout", submission_unknown=True)
    assert cli.submit(flow.args(), flow.store, flow.cfg)["status"] == "submission_unknown"
    for function in (cli.submit, cli.plan):
        with pytest.raises(ValueError):
            function(flow.args(), flow.store, flow.cfg)
    assert len(flow.calls["submit"]) == 1
    assert flow.store.get("fixture-a")["attempts"] == 1


def test_cli_worker_skips_unknown_and_manual_task_attach_does_not_post(flow, capsys):
    cli.plan(flow.args(), flow.store, flow.cfg)
    flow.state["submit_error"] = providers.ProviderError("fixture_timeout", submission_unknown=True)
    cli.submit(flow.args(), flow.store, flow.cfg)
    assert cli.main(["--db", str(flow.store.path), "work", "--max-submissions", "9",
                     "--max-reviews", "0", "--max-polls", "0"]) == 0
    assert '"processed": []' in capsys.readouterr().out
    assert len(flow.calls["submit"]) == 1
    assert cli.main(["--db", str(flow.store.path), "attach-task", "--job", "fixture-a",
                     "--task-id", "fixture-recovered-task"]) == 0
    assert "fixture-only-not-a-real-key" not in capsys.readouterr().out
    assert cli.poll(flow.args(), flow.store, flow.cfg)["status"] == "generated"
    assert flow.calls["poll"] == ["fixture-recovered-task"]
    assert len(flow.calls["submit"]) == 1


def test_cli_register_is_idempotent_for_unchanged_catalog(flow, capsys):
    path = flow.tmp / "catalog.json"
    write_json(path, flow.catalog)
    command = ["--db", str(flow.store.path), "register", "--catalog", str(path)]
    assert cli.main(command) == 0
    first = capsys.readouterr().out
    assert cli.main(command) == 0
    assert capsys.readouterr().out == first
    assert flow.calls["submit"] == []
    assert flow.store.db.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 4


def test_post_persistence_failure_stays_unknown_not_retryable(flow, monkeypatch):
    cli.plan(flow.args(), flow.store, flow.cfg)
    original = cli.write_json

    def fail_submission_record(path, value, **kwargs):
        if Path(path).name == "submission.json":
            raise OSError("Fixture local disk failure after provider POST")
        return original(path, value, **kwargs)

    monkeypatch.setattr(cli, "write_json", fail_submission_record)
    with pytest.raises(OSError):
        cli.submit(flow.args(), flow.store, flow.cfg)
    assert flow.store.get("fixture-a")["status"] == "submission_unknown"
    with pytest.raises(ValueError):
        cli.submit(flow.args(), flow.store, flow.cfg)
    assert len(flow.calls["submit"]) == 1


def test_explicit_failure_has_bounded_new_attempts(flow):
    flow.state["submit_error"] = providers.ProviderError("fixture_rejected_request")
    for expected in (1, 2):
        cli.plan(flow.args(), flow.store, flow.cfg)
        assert cli.submit(flow.args(), flow.store, flow.cfg)["status"] == "generation_failed"
        assert flow.store.get("fixture-a")["attempts"] == expected
    with pytest.raises(ValueError, match="budget exhausted"):
        cli.plan(flow.args(), flow.store, flow.cfg)
    assert len(flow.calls["submit"]) == 2


def test_tampered_request_rejected_before_post(flow):
    cli.plan(flow.args(), flow.store, flow.cfg)
    path = Path(flow.store.get("fixture-a")["request_path"])
    doc = read_json(path)
    doc["payload"]["content"][0]["text"] += " Changed after plan."
    write_json(path, doc, overwrite=True)
    with pytest.raises(ValueError, match="request changed"):
        cli.submit(flow.args(), flow.store, flow.cfg)
    assert flow.calls["submit"] == []


@pytest.mark.parametrize("name", ["contract.json", "evidence.json", "prompt.txt", "blind_review_prompt.txt"])
def test_registered_bundle_mutation_blocks_execution(flow, name):
    path = flow.tmp / "fixture-a" / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed after registration"):
        flow.store.load_job("fixture-a")
    assert flow.calls["submit"] == []


def test_poll_reuses_task_without_resubmitting(flow):
    cli.plan(flow.args(), flow.store, flow.cfg)
    cli.submit(flow.args(), flow.store, flow.cfg)
    flow.state["poll_status"] = "running"
    assert cli.poll(flow.args(), flow.store, flow.cfg)["status"] == "generating"
    flow.state["poll_status"] = "succeeded"
    assert cli.poll(flow.args(), flow.store, flow.cfg)["status"] == "generated"
    with pytest.raises(ValueError):
        cli.poll(flow.args(), flow.store, flow.cfg)
    assert len(flow.calls["submit"]) == 1
    assert flow.calls["poll"] == ["fixture-task-1", "fixture-task-1"]


@pytest.mark.parametrize("filename", ["response.json", "decision.json", "invocation.json", "generated.mp4"])
def test_accepted_review_artifact_tampering_blocks_export(flow, filename):
    flow.accept()
    row = flow.store.get("fixture-a")
    path = Path(row["output_path"]) if filename == "generated.mp4" else Path(row["decision_path"]).parent / filename
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="binding changed"):
        cli.export_pairs(Namespace(output=str(flow.tmp / "pairs.json")), flow.store)


def test_forging_decision_bindings_does_not_change_trusted_ledger_hash(flow):
    flow.accept()
    row = flow.store.get("fixture-a")
    response_path = Path(row["review_path"])
    response = read_json(response_path)
    response["review"]["observed_goal"] = "choice_b"
    write_json(response_path, response, overwrite=True)
    decision_path = Path(row["decision_path"])
    decision = read_json(decision_path)
    decision["bindings"]["response_sha256"] = file_sha256(response_path)
    write_json(decision_path, decision, overwrite=True)
    with pytest.raises(ValueError, match="binding changed"):
        cli.export_pairs(Namespace(output=str(flow.tmp / "pairs.json")), flow.store)


def test_wrong_goal_rejected_and_not_exported(flow):
    flow.generate()
    flow.state["review"]["observed_goal"] = "choice_b"
    result = cli.review(flow.args(), flow.store, flow.cfg)
    assert result["status"] == "rejected" and "GOAL_MISMATCH" in result["error_codes"]
    output = flow.tmp / "pairs.json"
    assert cli.export_pairs(Namespace(output=str(output)), flow.store)["pair_candidates"] == 0


def test_review_failure_does_not_automatically_retry(flow):
    flow.generate()
    flow.state["review_error"] = providers.ProviderError("fixture_review_timeout")
    result = cli.review(flow.args(), flow.store, flow.cfg)
    assert result["status"] == "review_failed" and result["automatic_retry"] is False
    assert len(flow.calls["review"]) == 1


def test_first_registration_rejects_declared_artifact_hash_mismatch(flow):
    path = flow.tmp / "fixture-a/blind_review_prompt.txt"
    path.write_text(path.read_text() + " Altered before first registration.")
    fresh = Store(flow.tmp / "fresh.sqlite3")
    try:
        with pytest.raises(ValueError, match="[Hh]ash|artifact|changed"):
            fresh.register(flow.catalog)
        assert fresh.rows() == []
    finally:
        fresh.close()


def test_arbitrary_preexisting_generated_file_is_not_a_provider_download(flow):
    path = flow.tmp / "unrelated-generated.mp4"
    path.write_bytes(b"UNRELATED_PREEXISTING_VIDEO_FIXTURE")
    # probe_video is deliberately mocked, but the real download function must
    # refuse this unbound preexisting file without making any network request.
    with pytest.raises(ValueError):
        flow.original_download("https://fixture.invalid/new-task.mp4", path)


def test_bound_download_can_resume_with_renewed_url_without_second_fetch(flow, monkeypatch):
    fetches = []

    def local_response(url, **kwargs):
        fetches.append(url)
        return io.BytesIO(b"IN_MEMORY_DOWNLOAD_FIXTURE_ONLY")

    monkeypatch.setattr(cli.urllib.request, "urlopen", local_response)
    path = flow.tmp / "receipt-bound.mp4"
    first = flow.original_download("https://fixture.invalid/first-url", path,
                                   task_id="fixture-task", request_sha256="a" * 64)
    second = flow.original_download("https://fixture.invalid/renewed-signed-url", path,
                                    task_id="fixture-task", request_sha256="a" * 64)
    assert first == second
    assert fetches == ["https://fixture.invalid/first-url"]
    receipt = read_json(path.with_suffix(".mp4.download.json"))
    assert receipt["origin"]["task_id"] == "fixture-task"
    assert receipt["media"]["sha256"] == file_sha256(path)


@pytest.mark.parametrize("mutation", ["different_task", "changed_bytes"])
def test_download_receipt_cannot_authorize_other_task_or_changed_bytes(flow, monkeypatch, mutation):
    monkeypatch.setattr(cli.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"IN_MEMORY_FIXTURE_ONLY"))
    path = flow.tmp / "receipt-bound.mp4"
    flow.original_download("https://fixture.invalid/task", path,
                           task_id="fixture-task", request_sha256="a" * 64)
    if mutation == "changed_bytes":
        path.write_bytes(b"CHANGED_AFTER_DOWNLOAD_FIXTURE")
    task_id = "another-fixture-task" if mutation == "different_task" else "fixture-task"
    with pytest.raises(ValueError, match="origin or hash mismatch"):
        flow.original_download("https://fixture.invalid/task", path,
                               task_id=task_id, request_sha256="a" * 64)


def test_unexplained_manipulation_is_not_exported_as_a_full_pair(flow):
    flow.generate()
    flow.state["review"]["events"].append({
        "role": "block1", "verb": "pick_and_place", "destination_role": "brown_pad",
        "t_start": 6., "t_end": 7., "confidence": .95,
        "evidence": [{"timestamp_seconds": 6.5, "description": "Fixture target moved again."}]})
    result = cli.review(flow.args(), flow.store, flow.cfg)
    assert result["status"] == "needs_review"
    assert "UNEXPECTED_MANIPULATION" in result["error_codes"]
    assert cli.export_pairs(Namespace(output=str(flow.tmp / "pairs.json")), flow.store)["pair_candidates"] == 0
