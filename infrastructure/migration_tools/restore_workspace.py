"""Verify and restore an immutable TIG TODO transfer package without GPU work.

Contract: manifest dictionaries use schema ``tig-todo-transfer-v1``; commits
are full 40-character Git object IDs; payload rows contain one relative POSIX
path, a nonnegative byte count, and a SHA256. Archives contain regular files
and directories only. Restore writes only into a new or empty destination.
No Python dependency installation, environment activation, or simulation runs
are performed. Git and zstd are external command prerequisites; Python imports
use only the standard library.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, BinaryIO, Iterator


SCHEMA = "tig-todo-transfer-v1"
COMMIT_KEYS = ("reference", "development", "runtime", "curobo")
BRANCH = "feat/dual-arm-continuation"
TIG_REMOTE = "https://github.com/Cat-blizzard/The-Imitator-Game.git"
CUROBO_REMOTE = "https://github.com/NVlabs/curobo.git"
CUROBO_TAG = "v0.9.0"


class TransferError(RuntimeError):
    """A package or destination failed validation before it was trusted."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TransferError(message)


def relative_path(value: Any, *, allow_root: bool = False) -> PurePosixPath:
    """Return an archive/package path, rejecting traversal and platform aliases."""
    require(isinstance(value, str) and bool(value), "A nonempty POSIX path is required")
    require("\\" not in value and "\x00" not in value, f"Unsafe path: {value!r}")
    path = PurePosixPath(value)
    require(not path.is_absolute() and ".." not in path.parts,
            f"Path must remain relative: {value!r}")
    require(not any(":" in part for part in path.parts), f"Nonportable path: {value!r}")
    require(".git" not in path.parts, f"Archive/payload may not inject Git metadata: {value!r}")
    require(allow_root or bool(path.parts), f"Root file path is invalid: {value!r}")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plain_payload(package: Path, name: str) -> Path:
    relative = relative_path(name)
    current = package
    for index, part in enumerate(relative.parts):
        current = current / part
        require(current.exists() and not current.is_symlink(),
                f"Payload is missing or uses a symbolic link: {name}")
        if index < len(relative.parts) - 1:
            require(current.is_dir(), f"Payload parent is not a directory: {name}")
    require(current.is_file(), f"Payload is not a regular file: {name}")
    require(stat.S_ISREG(current.stat().st_mode), f"Payload is not regular: {name}")
    return current


def load_and_verify(package: Path) -> dict[str, Any]:
    """Validate structure and every declared payload byte before any restore."""
    require(package.is_dir(), f"Package directory does not exist: {package}")
    manifest_path = package / "manifest.json"
    require(manifest_path.is_file() and not manifest_path.is_symlink(),
            "Package needs a regular manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(isinstance(manifest, dict) and manifest.get("schema") == SCHEMA,
            f"Expected schema {SCHEMA}")
    require(isinstance(manifest.get("source_root"), str), "Missing source_root provenance")
    commits = manifest.get("commits")
    require(isinstance(commits, dict), "commits must be an object")
    for key in COMMIT_KEYS:
        require(isinstance(commits.get(key), str)
                and re.fullmatch(r"[0-9a-f]{40}", commits[key]) is not None,
                f"Invalid full Git commit: {key}")
    rows = manifest.get("payload_files")
    require(isinstance(rows, list) and bool(rows), "payload_files must be a nonempty list")
    declared: dict[str, dict[str, Any]] = {}
    for row in rows:
        require(isinstance(row, dict), "Each payload row must be an object")
        name = relative_path(row.get("path")).as_posix()
        require(name == row["path"] and name != "manifest.json" and name not in declared,
                f"Duplicate or noncanonical payload path: {name}")
        size, digest = row.get("bytes"), row.get("sha256")
        require(isinstance(size, int) and not isinstance(size, bool) and size >= 0,
                f"Invalid payload byte count: {name}")
        require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
                f"Invalid payload SHA256: {name}")
        path = plain_payload(package, name)
        require(path.stat().st_size == size, f"Payload size mismatch: {name}")
        require(sha256(path) == digest, f"Payload SHA256 mismatch: {name}")
        declared[name] = row
    bundles = manifest.get("bundles")
    require(isinstance(bundles, dict), "bundles must be an object")
    for key in ("tig", "curobo"):
        require(isinstance(bundles.get(key), str) and bundles[key] in declared,
                f"Bundle is absent from payload inventory: {key}")
    archives = manifest.get("archives")
    require(isinstance(archives, list) and len(archives) == 3,
            "Exactly three support/evidence/assets archives are required")
    kinds: set[str] = set()
    archive_names: set[str] = set()
    for row in archives:
        require(isinstance(row, dict) and row.get("kind") in {"support", "evidence", "assets"},
                "Unknown archive kind")
        require(row["kind"] not in kinds and isinstance(row.get("path"), str)
                and row["path"] in declared and row["path"] not in archive_names,
                "Duplicate archive or archive missing from payload inventory")
        kinds.add(row["kind"])
        archive_names.add(row["path"])
    for profile in ("tig-official", "curobo-tig"):
        name = f"environment/{profile}.portable-requirements.txt"
        require(name in declared, f"Missing portable environment inventory: {name}")
    return manifest


def command(arguments: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(arguments, cwd=cwd, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode:
        raise TransferError(f"Command failed ({completed.returncode}): {shlex.join(arguments)}\n"
                            f"{completed.stderr.strip()}")
    return completed.stdout.strip()


def git(arguments: list[str], *, cwd: Path | None = None) -> str:
    return command(["git", "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false",
                    "-c", "protocol.file.allow=always", *arguments], cwd=cwd)


@contextmanager
def archive_stream(path: Path) -> Iterator[tarfile.TarFile]:
    """Stream zstd into tarfile without requiring a second uncompressed copy."""
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(["zstd", "--decompress", "--stdout", "--", str(path)],
                                   stdout=subprocess.PIPE, stderr=errors)
        assert process.stdout is not None
        archive: tarfile.TarFile | None = None
        try:
            archive = tarfile.open(fileobj=process.stdout, mode="r|")
            yield archive
            archive.close()
            archive = None
            # Consume padding and force the decompressor to validate its trailer.
            while process.stdout.read(1024 * 1024):
                pass
            process.stdout.close()
            status = process.wait()
            if status:
                errors.seek(0)
                raise TransferError(f"zstd rejected {path.name}: "
                                    f"{errors.read().decode('utf-8', errors='replace')}")
        except BaseException:
            if archive is not None:
                archive.close()
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
            process.wait()
            raise


def member_path(member: tarfile.TarInfo) -> PurePosixPath:
    require(member.isdir() or member.isreg(),
            f"Archive contains a link or special file: {member.name}")
    require(not member.issparse(), f"Sparse archive member is not supported: {member.name}")
    path = relative_path(member.name, allow_root=member.isdir())
    require(member.size >= 0 and (member.isreg() or member.size == 0),
            f"Invalid archive member size: {member.name}")
    return path


def preflight_archives(package: Path, manifest: dict[str, Any]) -> dict[str, int]:
    """Reject traversal, links and cross-archive file collisions before writing."""
    members: dict[str, str] = {}
    directories: set[str] = set()
    counts: dict[str, int] = {}
    for row in manifest["archives"]:
        count = 0
        with archive_stream(plain_payload(package, row["path"])) as archive:
            for member in archive:
                path = member_path(member)
                if not path.parts:
                    continue
                name = path.as_posix()
                kind = "directory" if member.isdir() else "file"
                require(name not in members or (members[name] == kind == "directory"),
                        f"Archive member would overwrite another member: {name}")
                for parent in path.parents:
                    require(members.get(parent.as_posix()) != "file",
                            f"Archive parent is a file: {name}")
                    directories.add(parent.as_posix())
                if kind == "file":
                    require(name not in directories,
                            f"Archive file replaces a directory: {name}")
                else:
                    directories.add(name)
                members[name] = kind
                count += 1
        counts[row["kind"]] = count
    return counts


def safe_directory(root: Path, relative: PurePosixPath) -> Path:
    current = root
    for part in relative.parts:
        current = current / part
        require(not current.is_symlink(), f"Refusing symbolic-link extraction parent: {current}")
        if current.exists():
            require(current.is_dir(), f"Extraction parent is not a directory: {current}")
        else:
            current.mkdir(mode=0o755)
    return current


def copy_exclusive(source: BinaryIO, target: Path, expected_bytes: int) -> None:
    require(not target.exists() and not target.is_symlink(),
            f"Refusing to overwrite a restored file: {target}")
    with target.open("xb") as output:
        shutil.copyfileobj(source, output, length=1024 * 1024)
    require(target.stat().st_size == expected_bytes, f"Extracted byte count differs: {target}")
    target.chmod(0o644)


def extract_archive(package: Path, row: dict[str, Any], root: Path) -> None:
    with archive_stream(plain_payload(package, row["path"])) as archive:
        for member in archive:
            relative = member_path(member)
            if member.isdir():
                safe_directory(root, relative)
                continue
            parent = safe_directory(root, relative.parent)
            source = archive.extractfile(member)
            require(source is not None, f"Regular member has no content: {member.name}")
            assert source is not None
            with source:
                copy_exclusive(source, parent / relative.name, member.size)


def empty_destination(destination: Path) -> None:
    require(not destination.is_symlink(), "Destination itself may not be a symbolic link")
    if destination.exists():
        require(destination.is_dir(), "Destination must be a directory")
        require(not any(destination.iterdir()), "Destination is not empty; refusing to overwrite")


def check_identity(repository: Path, expected: str) -> dict[str, Any]:
    actual = git(["rev-parse", "HEAD"], cwd=repository)
    require(actual == expected, f"Restored Git identity differs: {repository}")
    status = git(["status", "--porcelain", "--untracked-files=all"], cwd=repository)
    require(status == "", f"Restored checkout is not clean: {repository}\n{status}")
    return {"commit": actual, "clean": True}


def generate_environment(root: Path) -> None:
    infrastructure = safe_directory(root, PurePosixPath("infrastructure"))
    for name in ("tmp", "locks", "hf-cache", "xet-cache", "torch-cache", "uv-cache"):
        safe_directory(infrastructure, PurePosixPath(name))
    common = infrastructure / "common.env"
    development = infrastructure / "tig-development.env"
    for path in (common, development):
        require(not path.exists() and not path.is_symlink(),
                f"Package must not contain generated machine environment: {path}")
    lines = ["# Generated for this restored workspace; environment packages are not installed.",
             f"export ZERO_WAM_TIG_ROOT={shlex.quote(str(root))}"]
    for variable, suffix in (("HF_HOME", "infrastructure/hf-cache"),
                             ("HF_XET_CACHE", "infrastructure/xet-cache"),
                             ("TORCH_HOME", "infrastructure/torch-cache"),
                             ("UV_CACHE_DIR", "infrastructure/uv-cache"),
                             ("TMPDIR", "infrastructure/tmp"),
                             ("MS_ASSET_DIR", "infrastructure/maniskill"),
                             ("TIG_HUMAN_ROOT", "The-Imitator-Game/demos/demo_data"),
                             ("TIG_SIM_ROOT", "The-Imitator-Game/demos/imitator_data")):
        lines.append(f'export {variable}="$ZERO_WAM_TIG_ROOT/{suffix}"')
    common.write_text("\n".join(lines) + "\n", encoding="utf-8")
    development.write_text(
        "# Source this after rebuilding infrastructure/envs/tig-official.\n"
        f"source {shlex.quote(str(common))}\n"
        'if [[ ! -f "$ZERO_WAM_TIG_ROOT/infrastructure/envs/tig-official/bin/activate" ]]; then\n'
        "  printf '%s\\n' 'Rebuild tig-official from the exported requirements first.' >&2\n"
        "  return 1 2>/dev/null || exit 1\n"
        "fi\n"
        'source "$ZERO_WAM_TIG_ROOT/infrastructure/envs/tig-official/bin/activate"\n'
        'export TIG_SOURCE_ROOT="$ZERO_WAM_TIG_ROOT/infrastructure/development/tig-dual-arm"\n'
        'export PYTHONPATH="$TIG_SOURCE_ROOT:$ZERO_WAM_TIG_ROOT/infrastructure"\n',
        encoding="utf-8")
    common.chmod(0o644)
    development.chmod(0o644)


def restore(package: Path, destination: Path) -> dict[str, Any]:
    """Restore byte-preserved evidence plus fresh Git checkouts and machine env."""
    empty_destination(destination)
    package = package.resolve()
    destination = destination.absolute()
    manifest = load_and_verify(package)
    require(shutil.which("git") is not None and shutil.which("zstd") is not None,
            "Restoration requires git and zstd on PATH")
    counts = preflight_archives(package, manifest)
    # No destination mutation occurs until all payload and archive checks pass.
    empty_destination(destination)
    destination.mkdir(parents=True, exist_ok=True, mode=0o755)
    root = destination.resolve()
    commits = manifest["commits"]
    reference = root / "The-Imitator-Game"
    git(["clone", "--no-hardlinks", "--branch", "main",
         str(plain_payload(package, manifest["bundles"]["tig"])), str(reference)])
    git(["remote", "set-url", "origin", TIG_REMOTE], cwd=reference)
    check_identity(reference, commits["reference"])
    runtime = root / "infrastructure/runtime/The-Imitator-Game"
    development = root / "infrastructure/development/tig-dual-arm"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    development.parent.mkdir(parents=True, exist_ok=True)
    git(["worktree", "add", "--detach", str(runtime), commits["runtime"]], cwd=reference)
    git(["worktree", "add", "-b", BRANCH, str(development), commits["development"]], cwd=reference)
    curobo = root / "infrastructure/backends/curobo"
    curobo.parent.mkdir(parents=True, exist_ok=True)
    git(["clone", "--no-hardlinks", "--no-checkout",
         str(plain_payload(package, manifest["bundles"]["curobo"])), str(curobo)])
    git(["remote", "set-url", "origin", CUROBO_REMOTE], cwd=curobo)
    git(["checkout", "--detach", commits["curobo"]], cwd=curobo)
    curobo_tag = git(["describe", "--tags", "--exact-match", "HEAD"], cwd=curobo)
    require(curobo_tag == CUROBO_TAG,
            f"Restored cuRobo tag must be {CUROBO_TAG}; found {curobo_tag}")
    for row in manifest["archives"]:
        extract_archive(package, row, root)
    for row in manifest["payload_files"]:
        relative = PurePosixPath(row["path"])
        if relative.parts[0] == "environment":
            copied = PurePosixPath("infrastructure/migration-environment", *relative.parts[1:])
            parent = safe_directory(root, copied.parent)
            with plain_payload(package, row["path"]).open("rb") as source:
                copy_exclusive(source, parent / copied.name, row["bytes"])
    generate_environment(root)
    identities = {key: check_identity(path, commits[key]) for key, path in (
        ("reference", reference), ("runtime", runtime),
        ("development", development), ("curobo", curobo))}
    report = {"schema": "tig-todo-restoration-v1", "package": str(package),
              "destination": str(root), "source_root": manifest["source_root"],
              "manifest_sha256": sha256(package / "manifest.json"),
              "identities": identities, "archive_members": counts,
              "remotes": {"tig": git(["remote", "get-url", "origin"], cwd=reference),
                          "curobo": git(["remote", "get-url", "origin"], cwd=curobo)},
              "curobo_source_tag": curobo_tag,
              "dependency_installation": "NOT_RUN", "gpu_validation": "NOT_RUN",
              "historical_evidence_paths_rewritten": False,
              "derived_urdf_regeneration": "REQUIRED_ON_NEW_HOST",
              "remaining_path_adaptation": [
                  "Historical records retain original absolute paths.",
                  "Replay CPU self-test contains an original-root assertion; compatibility mount or a separately recorded source adaptation is required."]}
    (root / "infrastructure/migration-restoration.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def self_test() -> dict[str, Any]:
    """Exercise real tiny Git bundles and zstd archives without real environments."""
    require(shutil.which("git") is not None and shutil.which("zstd") is not None,
            "Self-test requires git and zstd")
    with tempfile.TemporaryDirectory(prefix="tig-restore-self-test-") as temporary:
        base = Path(temporary)
        package = base / "package"
        (package / "code").mkdir(parents=True)
        (package / "environment").mkdir()

        def repository(name: str) -> Path:
            path = base / name
            git(["init", "-b", "main", str(path)])
            git(["config", "user.name", "Transfer Self Test"], cwd=path)
            git(["config", "user.email", "transfer-self-test@example.invalid"], cwd=path)
            return path

        def commit(path: Path, name: str) -> str:
            (path / "README.md").write_text(name + "\n", encoding="utf-8")
            git(["add", "README.md"], cwd=path)
            git(["commit", "-m", name], cwd=path)
            return git(["rev-parse", "HEAD"], cwd=path)

        source = repository("tig-source")
        reference = commit(source, "reference")
        git(["checkout", "-b", "runtime-test"], cwd=source)
        runtime = commit(source, "runtime")
        git(["checkout", "-b", "snapshot/tig-dual-arm-curobo-20261117"], cwd=source)
        development = commit(source, "development")
        git(["bundle", "create", str(package / "code/tig.bundle"),
             "main", "snapshot/tig-dual-arm-curobo-20261117"], cwd=source)
        cu_source = repository("curobo-source")
        curobo = commit(cu_source, "official-curobo")
        git(["tag", CUROBO_TAG], cwd=cu_source)
        git(["bundle", "create", str(package / "code/curobo.bundle"),
             "HEAD", f"refs/tags/{CUROBO_TAG}"], cwd=cu_source)
        for profile in ("tig-official", "curobo-tig"):
            (package / "environment" / f"{profile}.portable-requirements.txt").write_text(
                "# synthetic inventory; do not install\n", encoding="utf-8")

        def make_archive(name: str, member_name: str, *, symbolic: bool = False) -> None:
            tar_path = base / "scratch.tar"
            with tarfile.open(tar_path, "w") as archive:
                member = tarfile.TarInfo(member_name)
                if symbolic:
                    member.type, member.linkname = tarfile.SYMTYPE, "../../escape"
                    archive.addfile(member)
                else:
                    import io
                    payload = b"byte-preserved synthetic evidence\n"
                    member.size = len(payload)
                    member.mode = 0o6777
                    archive.addfile(member, io.BytesIO(payload))
            command(["zstd", "--quiet", "--force", str(tar_path), "-o", str(package / name)])

        archives = [
            {"path": "workspace-support.tar.zst", "kind": "support"},
            {"path": "dual-arm-evidence.tar.zst", "kind": "evidence"},
            {"path": "simulation-assets.tar.zst", "kind": "assets"}]
        make_archive(archives[0]["path"], "infrastructure/curobo_tig_worker.py")
        make_archive(archives[1]["path"], "infrastructure/validation/dual-arm/test/report.json")
        make_archive(archives[2]["path"], "infrastructure/maniskill/data/test/object.txt")
        manifest: dict[str, Any] = {"schema": SCHEMA, "source_root": "/original/tig-workspace",
            "commits": {"reference": reference, "runtime": runtime,
                        "development": development, "curobo": curobo},
            "bundles": {"tig": "code/tig.bundle", "curobo": "code/curobo.bundle"},
            "archives": archives}

        def inventory() -> None:
            manifest["payload_files"] = [{"path": path.relative_to(package).as_posix(),
                "bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in sorted(package.rglob("*")) if path.is_file() and path.name != "manifest.json"]
            (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        def rejected(action: Any, label: str) -> None:
            try:
                action()
            except TransferError:
                return
            raise AssertionError(f"Negative case was accepted: {label}")

        inventory()
        load_and_verify(package)
        tampered = package / "environment/tig-official.portable-requirements.txt"
        original = tampered.read_bytes()
        tampered.write_bytes(b"X" * len(original))
        rejected(lambda: load_and_verify(package), "SHA256 corruption")
        tampered.write_bytes(original)
        nonempty = base / "occupied"
        nonempty.mkdir()
        (nonempty / "keep.txt").write_text("keep", encoding="utf-8")
        rejected(lambda: restore(package, nonempty), "nonempty destination")
        assert (nonempty / "keep.txt").read_text(encoding="utf-8") == "keep"
        make_archive(archives[2]["path"], "../escape.txt")
        inventory()
        traversal_destination = base / "traversal-destination"
        rejected(lambda: restore(package, traversal_destination), "archive traversal")
        assert not traversal_destination.exists() and not (base / "escape.txt").exists()
        make_archive(archives[2]["path"], "infrastructure/link", symbolic=True)
        inventory()
        rejected(lambda: restore(package, base / "link-destination"), "archive symbolic link")
        assert not (base / "link-destination").exists()
        make_archive(archives[2]["path"], "infrastructure/maniskill/data/test/object.txt")
        inventory()
        destination = base / "restored workspace"
        report = restore(package, destination)
        assert all(row["clean"] for row in report["identities"].values())
        assert report["identities"]["development"]["commit"] == development
        assert report["remotes"] == {"tig": TIG_REMOTE, "curobo": CUROBO_REMOTE}
        assert report["curobo_source_tag"] == CUROBO_TAG
        for relative in ("The-Imitator-Game", "infrastructure/runtime/The-Imitator-Game",
                         "infrastructure/development/tig-dual-arm"):
            assert git(["remote", "get-url", "origin"], cwd=destination / relative) == TIG_REMOTE
        assert git(["remote", "get-url", "origin"],
                   cwd=destination / "infrastructure/backends/curobo") == CUROBO_REMOTE
        assert git(["describe", "--tags", "--exact-match", "HEAD"],
                   cwd=destination / "infrastructure/backends/curobo") == CUROBO_TAG
        assert git(["branch", "--show-current"], cwd=destination / "infrastructure/development/tig-dual-arm") == BRANCH
        assert stat.S_IMODE((destination / "infrastructure/maniskill/data/test/object.txt").stat().st_mode) == 0o644
        common = (destination / "infrastructure/common.env").read_text(encoding="utf-8")
        assert str(destination) in common and "/original/tig-workspace" not in common
        assert not (destination / "infrastructure/envs").exists()
        assert not (destination / "infrastructure/backends/tig-derived-models").exists()
        rejected(lambda: restore(package, destination), "restoring twice")
        assert report["gpu_validation"] == "NOT_RUN"
        return {"self_test": "PASS", "actual_git_repositories": 2,
                "restored_clean_checkouts": 4, "archive_kinds": 3,
                "negative_cases": ["sha256_corruption", "nonempty_destination", "path_traversal",
                                   "symbolic_link", "second_restore"],
                "dependency_installation": "NOT_RUN", "gpu_execution": "NOT_RUN"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    commands = parser.add_subparsers(dest="command")
    verify_parser = commands.add_parser("verify", help="Verify every listed payload checksum and size")
    verify_parser.add_argument("--package", required=True, type=Path)
    restore_parser = commands.add_parser("restore", help="Restore into a new/empty workspace")
    restore_parser.add_argument("--package", required=True, type=Path)
    restore_parser.add_argument("--destination", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        if arguments.self_test:
            require(arguments.command is None, "--self-test cannot be combined with a subcommand")
            result = self_test()
        elif arguments.command == "verify":
            manifest = load_and_verify(arguments.package.resolve())
            result = {"status": "verified", "schema": manifest["schema"],
                      "payload_files": len(manifest["payload_files"]),
                      "payload_bytes": sum(row["bytes"] for row in manifest["payload_files"]),
                      "gpu_execution": "NOT_RUN"}
        elif arguments.command == "restore":
            result = restore(arguments.package.expanduser(), arguments.destination.expanduser())
        else:
            parser.error("Provide verify, restore, or --self-test")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (TransferError, OSError, ValueError, tarfile.TarError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
