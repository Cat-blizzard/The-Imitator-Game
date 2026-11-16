"""Prepare a byte-preserving TIG TODO transfer package without native execution.

Contract: source paths remain unchanged; only a new/empty output directory is
written. Three archives contain explicitly inventoried regular files, with no
links, special files, Python caches or Git metadata. Inventory rows record a
relative path, logical bytes and filesystem identity/timestamps; large source
files are read once by tar rather than hashed separately. Compressed payloads
receive SHA256 checksums. No dependencies are installed and no GPU is queried.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Callable, Iterable


SCHEMA = "tig-todo-transfer-v1"
REFERENCE = "d6d16ec511bc389e0a207692730c137bc022ef14"
RUNTIME = "27164a18deee3d012106704f4daed413a036318e"
SNAPSHOT_REF = "refs/heads/snapshot/tig-dual-arm-curobo-20261117"
WORKERS = (
    "curobo_tig_worker.py", "curobo_tig_session_worker.py",
    "curobo_tig_payload_worker.py", "curobo_tig_fast_payload_worker.py",
    "curobo_tig_task_payload_worker.py", "curobo_tig_seed_policy_worker.py")
SUPPORT_REQUIRED = tuple(f"infrastructure/{name}" for name in (
    *WORKERS, "TIG-IMPLEMENTATION-PLAN.md", "requirements-tig-official.in",
    "requirements-tig-official.lock.txt", "constraints-tig-official-overrides.txt",
    "install-tig-official.sh", "prepare-tig-runtime.sh", "validate-tig-official.py",
    "validate-runtime.py", "run-minimal-validation.sh", "runtime-validation.md",
    "tig-original-audit-20261115.json", "tig-official-installed.txt",
    "runtime/tig-video-seek.patch", "runtime/tig-video-seek.provenance.json",
    "runtime/tig-runtime-preparation.json", "validation/tig-official",
    "validation/tig-runtime"))
SUPPORT_OPTIONAL = tuple(f"infrastructure/{name}" for name in (
    "prepare-assets.sh", "asset-preparation.json", "complete-tig-dataset.py",
    "download-resumable.py", "validate-tig-dataset-coverage.py",
    "validate-tig-real-batch.py", "download-plan.json"))
SUPPORT_GLOBS = (
    "manifest-tig-assets*.json", "manifest-hf-ig10k*.json",
    "manifest-hf-imitator_*.json", "manifest-ms-imitator_*.json")
EXCLUDED = (
    "Official TIG human/simulation/real-robot demonstration datasets",
    "Model weights and unrelated policy checkpoints",
    "Virtual environments and package/compilation/download caches",
    "Independent BPP, RoboTwin and ZeroWAM projects and experiments",
    "Duplicate downloaded asset archives",
    "Old generated URDFs with host-specific absolute mesh paths",
    "Credentials, account configuration and infrastructure/hf-transfer-auth",
    "__pycache__ directories and *.pyc files")


class TransferError(RuntimeError):
    """An explicit source selection or transfer invariant was not satisfied."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TransferError(message)


def progress(stage: str, **values: Any) -> None:
    print(json.dumps({"stage": stage, **values}, ensure_ascii=False), flush=True)


def relative_path(value: str) -> PurePosixPath:
    require(isinstance(value, str) and bool(value), "A nonempty relative path is required")
    require("\\" not in value and "\x00" not in value, f"Unsafe path: {value!r}")
    path = PurePosixPath(value)
    require(bool(path.parts) and not path.is_absolute() and ".." not in path.parts,
            f"Path must remain relative: {value!r}")
    require(not any(":" in part for part in path.parts) and ".git" not in path.parts,
            f"Nonportable or Git metadata path: {value!r}")
    require(path.as_posix() == value, f"Noncanonical path: {value!r}")
    return path


def plain_source(root: Path, name: str) -> tuple[Path, os.stat_result]:
    """Validate every parent before accepting a regular file or directory."""
    relative = relative_path(name)
    current = root
    for index, part in enumerate(relative.parts):
        current = current / part
        info = current.lstat()
        require(not stat.S_ISLNK(info.st_mode), f"Symbolic link is not allowed: {name}")
        if index < len(relative.parts) - 1:
            require(stat.S_ISDIR(info.st_mode), f"Source parent is not a directory: {name}")
    require(stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode),
            f"Special source file is not allowed: {name}")
    return current, info


def command(arguments: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(arguments, cwd=cwd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, check=False)
    if result.returncode:
        raise TransferError(f"Command failed ({result.returncode}): {shlex.join(arguments)}\n"
                            f"{result.stderr.strip()}")
    return result.stdout


def git(arguments: list[str], *, cwd: Path) -> str:
    return command(["git", "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false",
                    *arguments], cwd=cwd).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data: Any) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(data, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")


def inventory(root: Path, selections: Iterable[str], output: Path,
              kind: str) -> dict[str, Any]:
    """Write JSONL metadata and NUL-delimited exact archive members, never hashes."""
    metadata = output / "inventory" / f"{kind}.files.jsonl"
    file_list = output / "inventory" / f"{kind}.files0"
    metadata.parent.mkdir(exist_ok=True)
    count = total_bytes = skipped = 0
    seen: set[str] = set()

    def walk(name: str) -> Iterable[tuple[str, os.stat_result]]:
        nonlocal skipped
        path, info = plain_source(root, name)
        if path.name == "__pycache__" or (stat.S_ISREG(info.st_mode) and path.suffix == ".pyc"):
            skipped += 1
            return
        if stat.S_ISREG(info.st_mode):
            yield name, info
            return
        with os.scandir(path) as entries:
            children = sorted(entry.name for entry in entries)
        for child in children:
            yield from walk(f"{name}/{child}")

    with metadata.open("x", encoding="utf-8") as records, file_list.open("xb") as members:
        for selection in sorted(set(selections)):
            relative_path(selection)
            for name, info in walk(selection):
                require(name not in seen, f"Overlapping archive selection: {name}")
                seen.add(name)
                row = {"path": name, "bytes": info.st_size, "mtime_ns": info.st_mtime_ns,
                       "ctime_ns": info.st_ctime_ns, "mode": stat.S_IMODE(info.st_mode),
                       "device": info.st_dev, "inode": info.st_ino}
                records.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                members.write(os.fsencode(name) + b"\x00")
                count += 1
                total_bytes += info.st_size
                if count % 50000 == 0:
                    progress("inventory", kind=kind, files=count, logical_bytes=total_bytes)
    require(count > 0, f"Archive selection has no regular files: {kind}")
    summary = {"kind": kind, "file_count": count, "logical_bytes": total_bytes,
               "excluded_cache_entries": skipped,
               "metadata": metadata.relative_to(output).as_posix(),
               "file_list": file_list.relative_to(output).as_posix(),
               "source_file_hashing": "NOT_PERFORMED; compressed archives receive SHA256"}
    progress("inventory-complete", **summary)
    return summary


def verify_source_metadata(root: Path, metadata: Path) -> int:
    """Reject files changed or replaced after their inventory was captured."""
    count = 0
    with metadata.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            _, info = plain_source(root, row["path"])
            require(stat.S_ISREG(info.st_mode), f"Source is no longer regular: {row['path']}")
            actual = (info.st_size, info.st_mtime_ns, info.st_ctime_ns,
                      stat.S_IMODE(info.st_mode), info.st_dev, info.st_ino)
            expected = tuple(row[key] for key in ("bytes", "mtime_ns", "ctime_ns", "mode", "device", "inode"))
            require(actual == expected, f"Source changed during preparation: {row['path']}")
            count += 1
    return count


def make_archive(root: Path, output: Path, summary: dict[str, Any],
                 archive_name: str, threads: int) -> dict[str, Any]:
    require(isinstance(threads, int) and 1 <= threads <= 64, "threads must be in [1, 64]")
    relative_path(archive_name)
    target = output / archive_name
    partial = target.with_name(target.name + ".partial")
    require(not target.exists() and not partial.exists(), f"Archive already exists: {target}")
    progress("archive-start", kind=summary["kind"], files=summary["file_count"],
             logical_bytes=summary["logical_bytes"])
    with (output / "inventory" / f"{summary['kind']}.tar.log").open("xb") as log:
        arguments = ["tar", "--create", "--format=pax", "--null", "--no-recursion",
                     "--verbatim-files-from", "--directory", str(root),
                     "--files-from", str(output / summary["file_list"]),
                     "--use-compress-program", f"zstd -T{threads} -1", "--file", str(partial)]
        result = subprocess.run(arguments, stdout=log, stderr=subprocess.STDOUT, check=False)
    require(result.returncode == 0, f"tar failed for {archive_name}; inspect its inventory/*.tar.log")
    checked = verify_source_metadata(root, output / summary["metadata"])
    require(checked == summary["file_count"], f"Inventory count changed: {summary['kind']}")
    os.replace(partial, target)
    progress("archive-complete", kind=summary["kind"], compressed_bytes=target.stat().st_size,
             source_files_verified=checked)
    return {"path": archive_name, "kind": summary["kind"], **{
        key: summary[key] for key in ("file_count", "logical_bytes", "metadata")}}


def create_bundle(repository: Path, destination: Path, references: list[str]) -> None:
    partial = destination.with_name(destination.name + ".partial")
    require(not destination.exists() and not partial.exists(), f"Bundle already exists: {destination}")
    git(["bundle", "create", str(partial), *references], cwd=repository)
    git(["bundle", "verify", str(partial)], cwd=repository)
    os.replace(partial, destination)
    progress("bundle-complete", path=destination.name, bytes=destination.stat().st_size)


def portable_freeze(raw: str) -> tuple[str, list[str]]:
    """Remove only the cuRobo checkout reference; keep the official wheel URL."""
    kept: list[str] = []
    removed: list[str] = []
    for line in raw.splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+)\s*@\s*(.+)$", line.strip())
        if match and match[2].startswith("file:"):
            name = re.sub(r"[-_.]+", "-", match[1]).lower()
            require(name in {"curobo", "nvidia-curobo"},
                    f"Unexpected local package reference requires review: {match[1]}")
            removed.append(line)
            continue
        require(not line.lstrip().startswith(("-e ", "--editable ")),
                "Unexpected editable package reference requires review")
        require(re.search(r"https?://[^/\s]*@", line) is None,
                "Requirement URL contains embedded account information")
        kept.append(line)
    header = ("# Installed-package inventory; rebuild and validate on the destination host.\n"
              "# Install the bundled pinned cuRobo source separately in curobo-tig.\n")
    return header + "\n".join(kept) + "\n", removed


def export_environment(root: Path, output: Path, profile: str) -> dict[str, Any]:
    python = root / "infrastructure/envs" / profile / "bin/python"
    require(python.is_file(), f"Environment interpreter does not exist: {python}")
    with tempfile.TemporaryDirectory(prefix="tig-env-inventory-") as temporary:
        raw = command(["uv", "pip", "freeze", "--python", str(python)], cwd=Path(temporary))
        metadata_text = command([str(python), "-I", "-c",
            "import importlib.metadata as m,json,platform; "
            "print(json.dumps({'python_version':platform.python_version(),"
            "'packages':sorted([{'name':d.metadata['Name'],'version':d.version} "
            "for d in m.distributions()],key=lambda x:((x['name'] or '').lower(),x['version']))}))"],
            cwd=Path(temporary))
    portable, removed = portable_freeze(raw)
    directory = output / "environment"
    (directory / f"{profile}.raw-freeze.txt").write_text(raw, encoding="utf-8")
    (directory / f"{profile}.portable-requirements.txt").write_text(portable, encoding="utf-8")
    metadata = json.loads(metadata_text)
    require(isinstance(metadata.get("packages"), list) and bool(metadata["packages"]),
            f"Environment has no installed-package metadata: {profile}")
    metadata.update({"profile": profile, "source_python": str(python),
                     "local_curobo_references_removed_from_portable": removed,
                     "dependency_installation": "NOT_RUN", "compatibility_validation": "NOT_RUN"})
    write_json(directory / f"{profile}.metadata.json", metadata)
    progress("environment-exported", profile=profile, packages=len(metadata["packages"]),
             python_version=metadata["python_version"])
    return {"profile": profile, "python_version": metadata["python_version"],
            "package_count": len(metadata["packages"]), "removed_local_references": len(removed)}


def empty_output(output: Path) -> None:
    require(not output.is_symlink(), "Output may not be a symbolic link")
    if output.exists():
        require(output.is_dir() and not any(output.iterdir()),
                "Output must be new or empty; refusing to overwrite")


def support_selections(root: Path) -> tuple[list[str], list[str]]:
    selections = list(SUPPORT_REQUIRED)
    absent: list[str] = []
    for name in SUPPORT_REQUIRED:
        plain_source(root, name)
    for name in SUPPORT_OPTIONAL:
        path = root / name
        if path.exists() or path.is_symlink():
            plain_source(root, name)
            selections.append(name)
        else:
            absent.append(name)
    for pattern in SUPPORT_GLOBS:
        for path in sorted((root / "infrastructure").glob(pattern)):
            name = path.relative_to(root).as_posix()
            plain_source(root, name)
            selections.append(name)
    return sorted(set(selections)), absent


def package_payloads(output: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(output.rglob("*")):
        require(not path.is_symlink(), f"Package payload is a symbolic link: {path}")
        if path.is_dir():
            continue
        require(stat.S_ISREG(path.stat().st_mode), f"Package payload is not regular: {path}")
        name = path.relative_to(output).as_posix()
        relative_path(name)
        require(not name.endswith(".partial"), f"Incomplete payload remains: {name}")
        if name in {"manifest.json", "SHA256SUMS"}:
            continue
        progress("payload-hash", path=name, bytes=path.stat().st_size)
        rows.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    require(bool(rows), "Transfer package contains no payload files")
    return rows


def prepare(source_root: Path, repository: Path, output: Path, threads: int = 2) -> dict[str, Any]:
    empty_output(output)
    require(1 <= threads <= 64, "threads must be in [1, 64]")
    for program in ("git", "tar", "zstd", "uv"):
        require(shutil.which(program) is not None, f"Required command is missing: {program}")
    require(source_root.is_dir() and not source_root.is_symlink(), "Source root must be a real directory")
    require(repository.is_dir() and not repository.is_symlink(), "Repository must be a real directory")
    root, repo, destination = source_root.resolve(), repository.resolve(), output.absolute()
    require(not destination.resolve().is_relative_to(root / "infrastructure/validation/dual-arm")
            and not destination.resolve().is_relative_to(root / "infrastructure/maniskill"),
            "Output may not be inside an archived evidence/assets tree")
    selections, absent = support_selections(root)
    for name in ("infrastructure/validation/dual-arm", "infrastructure/maniskill"):
        _, info = plain_source(root, name)
        require(stat.S_ISDIR(info.st_mode), f"Required source directory is missing: {name}")
    reference = root / "The-Imitator-Game"
    curobo = root / "infrastructure/backends/curobo"
    commits = {"reference": REFERENCE, "development": git(["rev-parse", "HEAD"], cwd=repo),
               "runtime": RUNTIME, "curobo": git(["rev-parse", "HEAD"], cwd=curobo)}
    for name, commit in commits.items():
        require(re.fullmatch(r"[0-9a-f]{40}", commit) is not None, f"Invalid {name} commit")
    curobo_tag = {"name": "v0.9.0", "object": git(["rev-parse", "refs/tags/v0.9.0"], cwd=curobo),
                  "commit": git(["rev-parse", "refs/tags/v0.9.0^{commit}"], cwd=curobo),
                  "describe": git(["describe", "--tags", "--always", "HEAD"], cwd=curobo)}
    require(curobo_tag["commit"] == commits["curobo"],
            "cuRobo HEAD must identify the validated v0.9.0 source tag")
    require(git(["rev-parse", "HEAD"], cwd=reference) == REFERENCE,
            "Historical clean reference must remain pinned at d6d16ec")
    require(git(["rev-parse", "refs/heads/main"], cwd=repo) == REFERENCE,
            "TIG main ref must identify the preserved reference commit")
    require(git(["rev-parse", SNAPSHOT_REF], cwd=repo) == commits["development"],
            "Snapshot branch must identify the prepared repository HEAD")
    git(["merge-base", "--is-ancestor", RUNTIME, commits["development"]], cwd=repo)
    for name, checkout in (("reference", reference), ("development", repo), ("curobo", curobo)):
        require(git(["status", "--porcelain", "--untracked-files=all"], cwd=checkout) == "",
                f"{name} repository has unsaved modifications; commit or audit before packaging")
    for name in ("common.env", "tig.env", "tig-reference.env"):
        plain_source(root, f"infrastructure/{name}")
    for name in ("infrastructure/migration_tools/restore_workspace.py", "docs/tig-dual-arm/MIGRATION.md",
                 "docs/tig-dual-arm/CONTINUE-HERE.md"):
        plain_source(repo, name)
    for profile in ("tig-official", "curobo-tig"):
        require((root / "infrastructure/envs" / profile / "bin/python").is_file(),
                f"Missing installed environment: {profile}")
    empty_output(destination)
    destination.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve()
    (destination / "code").mkdir()
    (destination / "environment/legacy-env").mkdir(parents=True)
    progress("prepare-start", source_root=str(root), output=str(destination), commits=commits)
    create_bundle(repo, destination / "code/tig.bundle", ["refs/heads/main", SNAPSHOT_REF])
    create_bundle(curobo, destination / "code/curobo.bundle", ["HEAD", "refs/tags/v0.9.0"])
    shutil.copyfile(repo / "infrastructure/migration_tools/restore_workspace.py", destination / "restore_workspace.py")
    shutil.copyfile(repo / "docs/tig-dual-arm/MIGRATION.md", destination / "README.md")
    shutil.copyfile(repo / "docs/tig-dual-arm/CONTINUE-HERE.md", destination / "CONTINUE-HERE.md")
    for name in ("common.env", "tig.env", "tig-reference.env"):
        shutil.copyfile(root / "infrastructure" / name, destination / "environment/legacy-env" / name)
    environments = [export_environment(root, destination, profile)
                    for profile in ("tig-official", "curobo-tig")]
    inventories: list[dict[str, Any]] = []
    archives: list[dict[str, Any]] = []
    for kind, selected, name in (
        ("support", selections, "workspace-support.tar.zst"),
        ("evidence", ["infrastructure/validation/dual-arm"], "dual-arm-evidence.tar.zst"),
        ("assets", ["infrastructure/maniskill"], "simulation-assets.tar.zst")):
        summary = inventory(root, selected, destination, kind)
        inventories.append(summary)
        archives.append(make_archive(root, destination, summary, name, threads))
    for summary in inventories:
        checked = verify_source_metadata(root, destination / summary["metadata"])
        require(checked == summary["file_count"], f"Final source inventory differs: {summary['kind']}")
    manifest: dict[str, Any] = {
        "schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root), "commits": commits,
        "curobo_version_source": curobo_tag,
        "bundles": {"tig": "code/tig.bundle", "curobo": "code/curobo.bundle"},
        "archives": archives, "inventories": inventories, "environments": environments,
        "optional_support_absent": absent, "excluded": list(EXCLUDED),
        "source_metadata_verified_after_archiving": True,
        "source_content_hashing": "Archive SHA256 only; no extra whole-source read",
        "legacy_env_policy": "Original snapshots only; do not source on the new server",
        "derived_urdf_policy": "Regenerate from pinned assets on the new host",
        "observed_preparation_system": {"platform": platform.system(), "kernel": platform.release(),
                                        "machine": platform.machine(), "python_version": platform.python_version()},
        "dependency_installation": "NOT_RUN", "gpu_validation": "NOT_RUN",
        "payload_files": package_payloads(destination)}
    write_json(destination / "manifest.json", manifest)
    with (destination / "SHA256SUMS").open("x", encoding="utf-8") as sums:
        for row in manifest["payload_files"]:
            sums.write(f"{row['sha256']}  {row['path']}\n")
        sums.write(f"{sha256(destination / 'manifest.json')}  manifest.json\n")
    summary = {"status": "prepared", "output": str(destination),
               "source_files": sum(item["file_count"] for item in inventories),
               "logical_source_bytes": sum(item["logical_bytes"] for item in inventories),
               "payload_bytes": sum(item["bytes"] for item in manifest["payload_files"]),
               "payload_files": len(manifest["payload_files"]), "commits": commits,
               "gpu_execution": "NOT_RUN", "upload": "NOT_RUN"}
    progress("prepare-complete", **summary)
    return summary


def self_test() -> dict[str, Any]:
    """Exercise real compressed inventory, byte extraction and rejection paths."""
    for program in ("git", "tar", "zstd"):
        require(shutil.which(program) is not None, f"Self-test requires {program}")
    rejected_cases: list[str] = []

    def rejected(action: Callable[[], Any], label: str) -> None:
        try:
            action()
        except (TransferError, OSError):
            rejected_cases.append(label)
            return
        raise AssertionError(f"Negative case was accepted: {label}")

    with tempfile.TemporaryDirectory(prefix="tig-prepare-self-test-") as temporary:
        base = Path(temporary)
        source, output = base / "source", base / "package"
        source.mkdir()
        output.mkdir()
        folder = source / "infrastructure/validation/dual-arm/test case"
        folder.mkdir(parents=True)
        payloads = {"report.json": b'{"status":"failed","keep":true}\n',
                    "trajectory.h5": b"synthetic\x00trajectory\xff",
                    "line\nbreak.txt": b"NUL-delimited archive member names\n"}
        for name, data in payloads.items():
            (folder / name).write_bytes(data)
        (folder / "__pycache__").mkdir()
        (folder / "__pycache__/unused.pyc").write_bytes(b"cache")
        (folder / "unused.pyc").write_bytes(b"cache")
        summary = inventory(source, ["infrastructure/validation/dual-arm"], output, "evidence")
        assert summary["file_count"] == 3 and summary["logical_bytes"] == sum(map(len, payloads.values()))
        assert summary["excluded_cache_entries"] == 2
        archive_row = make_archive(source, output, summary, "tiny.tar.zst", 1)
        assert archive_row["file_count"] == 3 and (output / "tiny.tar.zst").is_file()
        archive = base / "tiny.tar"
        command(["zstd", "--quiet", "--decompress", str(output / "tiny.tar.zst"), "-o", str(archive)])
        with tarfile.open(archive) as stored:
            members = stored.getmembers()
            assert len(members) == 3 and all(item.isreg() for item in members)
            for member in members:
                relative_path(member.name)
                content = stored.extractfile(member)
                assert content is not None
                assert content.read() == payloads[PurePosixPath(member.name).name]
        assert verify_source_metadata(source, output / summary["metadata"]) == 3
        for name in ("../escape", "/absolute", "a/../b", "a\\b", ".git/config", "a:b", "a//b"):
            rejected(lambda name=name: relative_path(name), f"path:{name}")
        (folder / "link").symlink_to(folder / "report.json")
        rejected(lambda: plain_source(source, "infrastructure/validation/dual-arm/test case/link"), "symbolic_link")
        rejected(lambda: plain_source(source, "infrastructure/validation/dual-arm/test case/link/child"), "symlink_parent")
        os.mkfifo(folder / "fifo")
        rejected(lambda: plain_source(source, "infrastructure/validation/dual-arm/test case/fifo"), "special_file")
        rejected(lambda: empty_output(output), "nonempty_output")
        (folder / "report.json").write_bytes(b"changed")
        rejected(lambda: verify_source_metadata(source, output / summary["metadata"]), "source_changed")
        raw = ("numpy==1.26.4\nnvidia-curobo @ file:///local/curobo\n"
               "torchcodec @ https://download.pytorch.org/whl/torchcodec.whl\n")
        portable, removed = portable_freeze(raw)
        assert len(removed) == 1 and "file:///local" not in portable
        assert "torchcodec @ https://download.pytorch.org/whl/torchcodec.whl" in portable
        rejected(lambda: portable_freeze("unknown @ file:///secret\n"), "unknown_local_requirement")
        rejected(lambda: portable_freeze("demo @ https://user:token@example.invalid/demo.whl\n"), "url_credentials")
        repo = base / "repo"
        git(["init", "-b", "main", str(repo)], cwd=base)
        git(["config", "user.name", "Transfer Self Test"], cwd=repo)
        git(["config", "user.email", "transfer-self-test@example.invalid"], cwd=repo)
        (repo / "README.md").write_text("tiny source identity\n", encoding="utf-8")
        git(["add", "README.md"], cwd=repo)
        git(["commit", "-m", "synthetic source"], cwd=repo)
        git(["tag", "v0.9.0"], cwd=repo)
        create_bundle(repo, output / "tiny.bundle", ["refs/heads/main", "refs/tags/v0.9.0"])
        bundle_refs = git(["bundle", "list-heads", str(output / "tiny.bundle")], cwd=repo)
        assert "refs/tags/v0.9.0" in bundle_refs
        restored = base / "restored-source"
        git(["clone", "--branch", "main", str(output / "tiny.bundle"), str(restored)], cwd=base)
        assert git(["describe", "--tags", "HEAD"], cwd=restored) == "v0.9.0"
        rows = package_payloads(output)
        assert all(len(row["sha256"]) == 64 and row["bytes"] >= 0 for row in rows)
        assert {row["path"] for row in rows} >= {"tiny.tar.zst", "tiny.bundle", summary["metadata"]}
        return {"self_test": "PASS", "actual_archive_files": 3,
                "actual_git_bundles_verified": 1, "payload_inventory_rows": len(rows),
                "negative_cases": rejected_cases, "source_file_hashes_generated": 0,
                "dependency_installation": "NOT_RUN", "gpu_execution": "NOT_RUN"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    commands = parser.add_subparsers(dest="command")
    prepare_parser = commands.add_parser("prepare", help="Prepare into a new or empty local package directory")
    prepare_parser.add_argument("--source-root", required=True, type=Path)
    prepare_parser.add_argument("--repository", required=True, type=Path)
    prepare_parser.add_argument("--output", required=True, type=Path)
    prepare_parser.add_argument("--threads", type=int, default=2)
    arguments = parser.parse_args()
    try:
        if arguments.self_test:
            require(arguments.command is None, "--self-test cannot be combined with prepare")
            result = self_test()
        elif arguments.command == "prepare":
            result = prepare(arguments.source_root.expanduser(), arguments.repository.expanduser(),
                             arguments.output.expanduser(), arguments.threads)
        else:
            parser.error("Provide prepare or --self-test")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    except (TransferError, OSError, ValueError, tarfile.TarError) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
