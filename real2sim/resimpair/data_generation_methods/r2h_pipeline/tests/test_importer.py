import copy
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from data_generation_methods.r2h_pipeline import importer
from data_generation_methods.r2h_pipeline.contracts import INITIAL_HASH_KEYS, PROGRAMS
from data_generation_methods.r2h_pipeline.tests.test_contracts import fixture_row_receipt


def write_json(path, doc):
    path.write_text(json.dumps(doc), encoding="utf-8")


@pytest.fixture
def source_manifest(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    media = {}
    for size in (224, 512):
        path = root / f"rgb{size}.mp4"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                        f"color=black:s={size}x{size}:r=20", "-frames:v", "24", "-c:v", "libx264",
                        "-threads", "1", "-bf", "0", str(path)], check=True, capture_output=True)
        media[size] = path
    episodes = []
    for pi, program in enumerate(PROGRAMS):
        directory = root / program
        directory.mkdir()
        row, receipt = fixture_row_receipt(program)
        initial = np.array([0., 1., 2.], np.float32)
        np.savez_compressed(directory / "arrays.npz", native_commands=np.full((24, 16), pi, np.float32),
                            timestamps_seconds=np.arange(24) / 20, initial_physics_state=initial,
                            stage=np.full(24, "fixture_stage"), object_states=np.zeros((24, 3, 13)))
        receipt.update(production_approved=True, qualification_only=False,
                       action_space="panda_eef14_v3", label_kind="FK_of_executed_native_PD_joint_targets",
                       arrays=str(directory / "arrays.npz"), video=str(media[224]),
                       reference_video={"path": str(media[512])},
                       reference_video_sha256=importer._sha(media[512]),
                       sha256={"arrays.npz": importer._sha(directory / "arrays.npz"), "video.mp4": importer._sha(media[224])})
        receipt["initial_physics_sha256"] = hashlib.sha256(initial.tobytes()).hexdigest()
        write_json(directory / "receipt.json", receipt)
        row.update(id="fixture_" + program, task="long_block_access_restore", length=24, fps=20,
                   action_space=receipt["action_space"], label_kind=receipt["label_kind"],
                   production_approved=True, formal_release_approved=True, qualification_only=False,
                   collection_receipt=str(directory / "receipt.json"), receipt_sha256=importer._sha(directory / "receipt.json"),
                   arrays=str(directory / "arrays.npz"), video={"path": str(media[224])},
                   reference_video={"path": str(media[512])}, reference_video_sha256=receipt["reference_video_sha256"],
                   source_video_sha256=importer._sha(media[224]))
        episodes.append(row)
    manifest = root / "manifest.json"
    write_json(manifest, {"formal_release_approved": True, "episodes": episodes})
    return manifest


def test_real_cpu_prepare_full_task_and_idempotence(source_manifest, tmp_path, monkeypatch):
    monkeypatch.setattr(importer, "blind_review_prompt", lambda: "Observe only the supplied anonymous video.\n")
    output = tmp_path / "prepared"
    first = importer.prepare_catalog(source_manifest, output, groups=1)
    second = importer.prepare_catalog(source_manifest, output, groups=1)
    assert first == second
    assert first["prepared_jobs"] == 3 and first["generated_jobs"] == 0
    assert first["training_admitted"] is False
    assert all(x["split"] == "train" for x in first["jobs"])
    assert len(first["groups"][0]["command_divergences"]) == 3
    for entry in first["jobs"]:
        job = json.loads(Path(entry["job_path"]).read_text())
        assert not Path(job["outputs"]["generated_video"]).exists()
        assert job["media"]["native_video"]["frames"] == 24
        assert job["durations"]["full_task_preserved"]
        frames = job["media"]["event_frames"]
        assert frames[0]["frame_index"] == 0 and frames[-1]["frame_index"] == 23
        assert all(Path(x["path"]).is_file() for x in frames)
        assert all(Path(x["path"]).name.startswith("frame_") for x in frames)


def test_payload_corruption_rejected_before_catalog(source_manifest, tmp_path):
    doc = json.loads(source_manifest.read_text())
    with Path(doc["episodes"][0]["arrays"]).open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="hash mismatch"):
        importer.prepare_catalog(source_manifest, tmp_path / "prepared", groups=1)
    assert not (tmp_path / "prepared/catalog.json").exists()


def test_incomplete_or_cross_split_groups_rejected(source_manifest):
    doc = json.loads(source_manifest.read_text())
    missing = copy.deepcopy(doc)
    missing["episodes"].pop()
    with pytest.raises(ValueError, match="complete train groups"):
        importer._select_rows(missing, 1, "train")
    cross = copy.deepcopy(doc)
    cross["episodes"][0]["split"] = "test"
    with pytest.raises(ValueError, match="crosses splits"):
        importer._select_rows(cross, 1, "train")


def test_four_hash_group_difference_rejected(source_manifest):
    doc = json.loads(source_manifest.read_text())
    items = [importer._source_row(row) for row in doc["episodes"]]
    items[1]["receipt"][INITIAL_HASH_KEYS[1]] = "b" * 64
    with pytest.raises(ValueError, match="Goal-dependent"):
        importer._group_evidence(items)


def test_tampered_cached_artifact_rejected(source_manifest, tmp_path, monkeypatch):
    monkeypatch.setattr(importer, "blind_review_prompt", lambda: "Anonymous review\n")
    output = tmp_path / "prepared"
    catalog = importer.prepare_catalog(source_manifest, output, groups=1)
    path = Path(catalog["jobs"][0]["job_path"]).parent / "prompt.txt"
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        importer.prepare_catalog(source_manifest, output, groups=1)


def test_existing_output_conflicting_request_fails(source_manifest, tmp_path, monkeypatch):
    monkeypatch.setattr(importer, "blind_review_prompt", lambda: "Anonymous review\n")
    output = tmp_path / "prepared"
    importer.prepare_catalog(source_manifest, output, groups=1)
    doc = json.loads(source_manifest.read_text())
    doc["unrelated_metadata_change"] = True
    write_json(source_manifest, doc)
    with pytest.raises(ValueError, match="Catalog configuration/source conflict"):
        importer.prepare_catalog(source_manifest, output, groups=1)
