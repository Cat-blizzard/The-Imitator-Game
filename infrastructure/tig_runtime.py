"""Resolve TIG code separately from optional external cuRobo dependencies.

Clients import ``infrastructure.tig_runtime``; sibling worker scripts import
``tig_runtime``. Default worker code always belongs to this module's checkout.
TIG_WORKSPACE selects the dependency workspace, TIG_CUROBO_PYTHON selects its
interpreter, TIG_CUROBO_SOURCE selects the official source root used for identity
checks, and TIG_CACHE_DIR selects the cache *parent* directory. MS_ASSET_DIR is
honoured for assets. Relative overrides are relative to code_root, not cwd.

With no overrides, an existing ancestor dependency layout is reused; otherwise
dependencies default to this checkout. This never searches for another TIG
development checkout. Path resolution/environment construction create nothing,
import no simulator/planner, and launch no process. Cache creation is explicit.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import re
from typing import Any


def _absolute(value: str | os.PathLike[str], base: Path) -> Path:
    """Normalize a lexical path without resolving an interpreter's symlink.

    Resolving venv/bin/python to the base interpreter would select the wrong
    site-packages when passed to Popen. Keep that venv invocation path intact.
    """
    if not isinstance(value, (str, os.PathLike)) or not os.fspath(value):
        raise ValueError("A nonempty filesystem path is required")
    path = Path(value).expanduser()
    return Path(os.path.abspath(path if path.is_absolute() else base / path))


def _dependency_workspace(code_root: Path) -> Path:
    for candidate in (code_root, *code_root.parents):
        infrastructure = candidate / "infrastructure"
        if ((infrastructure / "envs/curobo-tig/bin/python").is_file()
                or (infrastructure / "backends/curobo/curobo/__init__.py").is_file()):
            return candidate
    return code_root


@dataclass(frozen=True)
class RuntimePaths:
    code_root: Path
    workspace_root: Path
    curobo_python: Path
    curobo_source: Path
    cache_root: Path
    asset_dir: Path

    @property
    def worker_dir(self) -> Path:
        return self.code_root / "infrastructure"

    @property
    def runtime_cache(self) -> Path:
        return self.cache_root / "curobo-v2-runtime-cache"

    @property
    def warp_cache(self) -> Path:
        # Native wp.config.kernel_cache_dir used this separate directory.
        return self.cache_root / "curobo-v2-warp-cache"

    @property
    def cuda_cache(self) -> Path:
        return self.runtime_cache / "cuda"

    @property
    def inductor_cache(self) -> Path:
        return self.runtime_cache / "inductor"

    @property
    def warp_environment_cache(self) -> Path:
        # Preserve the existing launcher WARP_CACHE_PATH independently of the
        # native worker's wp.config.kernel_cache_dir above.
        return self.runtime_cache / "warp"

    def worker_path(self, worker: str | os.PathLike[str]) -> Path:
        path = Path(worker).expanduser()
        if not os.fspath(worker) or path.suffix != ".py":
            raise ValueError("An explicit worker Python file is required")
        if path.is_absolute():
            return _absolute(path, self.code_root)
        return _absolute(path, self.worker_dir if path.parent == Path(".") else self.code_root)

    def as_dict(self) -> dict[str, str]:
        return {name: str(getattr(self, name)) for name in (
            "code_root", "worker_dir", "workspace_root", "curobo_python",
            "curobo_source", "cache_root", "runtime_cache", "warp_cache", "asset_dir")}


def resolve_runtime_paths(*, code_root: str | os.PathLike[str] | None = None,
                          environ: Mapping[str, str] | None = None) -> RuntimePaths:
    """Return paths without importing dependencies, creating dirs or executing.

    code_root is normally inferred from this file in checkout/infrastructure.
    The optional argument supports embedding and isolated CPU tests. An explicit
    dependency override wins even if it is not installed yet; launch validation
    reports missing files rather than silently choosing another code checkout.
    """
    environment = os.environ if environ is None else environ
    code = Path(__file__).resolve().parent.parent if code_root is None else _absolute(code_root, Path.cwd())
    workspace = (_absolute(environment["TIG_WORKSPACE"], code)
        if "TIG_WORKSPACE" in environment else _dependency_workspace(code))
    def override(name: str, fallback: Path) -> Path:
        return _absolute(environment[name], code) if name in environment else fallback
    return RuntimePaths(code, workspace,
        override("TIG_CUROBO_PYTHON", workspace / "infrastructure/envs/curobo-tig/bin/python"),
        override("TIG_CUROBO_SOURCE", workspace / "infrastructure/backends/curobo"),
        override("TIG_CACHE_DIR", workspace / "infrastructure/tmp"),
        override("MS_ASSET_DIR", workspace / "infrastructure/maniskill"))


def worker_environment(gpu_uuid: str, *, paths: RuntimePaths | None = None,
                       environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build the existing isolated worker environment without modifying os.environ.

    PYTHONPATH is removed, preserving the independent cuRobo interpreter's
    package environment. TIG_CUROBO_SOURCE is for identity verification; it is
    deliberately not inserted into Python's import path.
    """
    if not isinstance(gpu_uuid, str) or re.fullmatch(r"GPU-[0-9a-fA-F-]+", gpu_uuid) is None:
        raise ValueError("One explicit GPU UUID is required")
    environment = dict(os.environ if environ is None else environ)
    paths = paths or resolve_runtime_paths(environ=environment)
    environment.pop("PYTHONPATH", None)
    environment.update(CUDA_VISIBLE_DEVICES=gpu_uuid, PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
        TIG_WORKSPACE=str(paths.workspace_root), TIG_CUROBO_PYTHON=str(paths.curobo_python),
        TIG_CUROBO_SOURCE=str(paths.curobo_source), TIG_CACHE_DIR=str(paths.cache_root),
        MS_ASSET_DIR=str(paths.asset_dir), CUDA_CACHE_PATH=str(paths.cuda_cache),
        TORCHINDUCTOR_CACHE_DIR=str(paths.inductor_cache),
        WARP_CACHE_PATH=str(paths.warp_environment_cache))
    return environment


def ensure_cache_directories(paths: RuntimePaths) -> None:
    """Explicit opt-in IO for launchers/workers immediately before native use."""
    for path in (paths.runtime_cache, paths.warp_cache, paths.cuda_cache,
                 paths.inductor_cache, paths.warp_environment_cache):
        path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class WorkerLaunch:
    argv: tuple[str, ...]
    env: dict[str, str]
    cwd: Path
    paths: RuntimePaths

    def popen_kwargs(self) -> dict[str, Any]:
        """Return only shared options; the transport keeps log/protocol ownership."""
        return {"env": self.env.copy(), "cwd": self.cwd, "text": True,
            "encoding": "utf-8", "bufsize": 1}


def worker_launch_spec(worker: str | os.PathLike[str], *, gpu_uuid: str,
                       extra_args: Sequence[str] = ("--serve",),
                       python: str | os.PathLike[str] | None = None,
                       paths: RuntimePaths | None = None,
                       environ: Mapping[str, str] | None = None,
                       check_files: bool = True) -> WorkerLaunch:
    """Prepare Popen arguments without spawning or opening a log file.

    A worker basename resolves only inside the current checkout's infrastructure
    directory. An absolute selected-worker override remains explicit, matching
    the existing --worker-script interface; no missing default falls back to an
    external workspace's worker. The transport owns timeout/log/RPC lifecycle.
    """
    if (not isinstance(extra_args, Sequence) or isinstance(extra_args, (str, bytes))
            or not all(isinstance(arg, str) for arg in extra_args)):
        raise ValueError("Worker arguments must be a sequence of strings")
    if type(check_files) is not bool:
        raise ValueError("check_files must be an explicit boolean")
    paths = paths or resolve_runtime_paths(environ=environ)
    interpreter = paths.curobo_python if python is None else _absolute(python, paths.code_root)
    paths = replace(paths, curobo_python=interpreter)
    script = paths.worker_path(worker)
    if check_files:
        if not interpreter.is_file():
            raise FileNotFoundError(f"cuRobo interpreter missing: {interpreter}; set TIG_CUROBO_PYTHON")
        if not script.is_file():
            raise FileNotFoundError(f"Worker code missing in selected checkout: {script}")
    environment = worker_environment(gpu_uuid, paths=paths, environ=environ)
    return WorkerLaunch((str(interpreter), "-B", "-u", str(script), *extra_args),
        environment, paths.code_root, paths)


def self_test() -> dict[str, Any]:
    """Temporary filesystem fixtures only; no process, planner or GPU exists."""
    import tempfile

    def reject(call: Any) -> None:
        try:
            call()
        except (ValueError, FileNotFoundError, TypeError):
            return
        raise AssertionError("Invalid runtime input accepted")

    with tempfile.TemporaryDirectory(prefix="tig-runtime-cpu-") as temporary:
        root = Path(temporary)
        workspace = root / "dependencies"
        python = workspace / "infrastructure/envs/curobo-tig/bin/python"
        python.parent.mkdir(parents=True)
        python.write_text("not executed\n")
        code = root / "checkout with spaces"
        (code / "infrastructure").mkdir(parents=True)
        script = code / "infrastructure/curobo_tig_task_payload_worker.py"
        script.write_text("# not executed\n")
        base = {"TIG_WORKSPACE": str(workspace), "PYTHONPATH": "/old/client/env",
            "UNRELATED": "preserved"}
        original = base.copy()
        paths = resolve_runtime_paths(code_root=code, environ=base)
        assert paths.code_root == code and paths.workspace_root == workspace
        assert paths.curobo_python == python
        assert paths.worker_path(script.name) == script
        assert paths.worker_path("infrastructure/"+script.name) == script
        assert paths.runtime_cache == workspace / "infrastructure/tmp/curobo-v2-runtime-cache"
        assert paths.warp_cache == workspace / "infrastructure/tmp/curobo-v2-warp-cache"
        assert not paths.cache_root.exists()
        launch = worker_launch_spec(script.name, gpu_uuid="GPU-abcd-1234", paths=paths,
            environ=base, extra_args=("--serve", "--task-id", "fixture"))
        assert launch.argv == (str(python), "-B", "-u", str(script), "--serve", "--task-id", "fixture")
        assert launch.cwd == code and "PYTHONPATH" not in launch.env
        assert launch.env["UNRELATED"] == "preserved" and base == original
        assert launch.env["CUDA_CACHE_PATH"] == str(paths.runtime_cache / "cuda")
        assert launch.env["WARP_CACHE_PATH"] == str(paths.runtime_cache / "warp")
        assert resolve_runtime_paths(code_root=code, environ=launch.env) == paths
        assert not paths.cache_root.exists()
        ensure_cache_directories(paths)
        assert all(path.is_dir() for path in (paths.runtime_cache, paths.warp_cache,
            paths.cuda_cache, paths.inductor_cache, paths.warp_environment_cache))
        # No external/legacy worker is substituted when current code is absent.
        external_worker = workspace / "infrastructure/missing_worker.py"
        external_worker.write_text("# never selected automatically\n")
        reject(lambda: worker_launch_spec(external_worker.name, gpu_uuid="GPU-abcd", paths=paths, environ={}))
        explicit = worker_launch_spec(external_worker, gpu_uuid="GPU-abcd", paths=paths, environ={})
        assert explicit.argv[3] == str(external_worker)
        # A fresh checkout can locate dependencies in an ancestor, without any
        # old DEV tree or another code checkout being installed there.
        nested = workspace / "sandboxes/new-checkout"
        nested.mkdir(parents=True)
        legacy = resolve_runtime_paths(code_root=nested, environ={})
        assert legacy.workspace_root == workspace and legacy.code_root == nested
        standalone = root / "standalone"
        standalone.mkdir()
        assert resolve_runtime_paths(code_root=standalone, environ={}).workspace_root == standalone
        # Explicit env/source/cache/assets override the shared workspace layout.
        link = root / "selected venv/bin/python"
        link.parent.mkdir(parents=True)
        link.symlink_to(python)
        overridden = resolve_runtime_paths(code_root=code, environ={**base,
            "TIG_CUROBO_PYTHON":str(link), "TIG_CUROBO_SOURCE":"../official source",
            "TIG_CACHE_DIR":"../cache elsewhere", "MS_ASSET_DIR":"../assets elsewhere"})
        assert overridden.curobo_python == link and overridden.curobo_python != link.resolve()
        assert overridden.curobo_source == root / "official source"
        assert overridden.cache_root == root / "cache elsewhere"
        assert overridden.asset_dir == root / "assets elsewhere"
        assert worker_launch_spec(script.name, gpu_uuid="GPU-abcd", paths=overridden, environ={}).argv[0] == str(link)
        selected = worker_launch_spec(script.name, gpu_uuid="GPU-abcd", paths=paths,
            python=link, environ={})
        assert selected.paths.curobo_python == link and selected.env["TIG_CUROBO_PYTHON"] == str(link)
        for invalid in ("0", "GPU-abcd,GPU-1234", "GPU-", None):
            reject(lambda: worker_environment(invalid, paths=paths, environ={}))
        reject(lambda: resolve_runtime_paths(code_root=code, environ={"TIG_WORKSPACE":""}))
        reject(lambda: worker_launch_spec(script.name,gpu_uuid="GPU-abcd",paths=paths,environ={},extra_args="--serve"))
    return {"status":"passed", "environment_steps":0, "processes_started":0,
        "gpu_used":False, "checks":["current code and external dependencies stay separate",
            "all four TIG path overrides and asset override", "legacy ancestor fallback without old DEV",
            "standalone checkout default", "no fallback to another workspace worker",
            "venv interpreter symlink preserved", "isolated deterministic env and unchanged caller mapping",
            "legacy cache layout plus explicit opt-in directory creation", "spaces in argv paths",
            "selected worker and task arguments", "bad GPU/path/argument rejection"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(self_test(), indent=2, allow_nan=False))
