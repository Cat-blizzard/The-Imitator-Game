"""Imitator 2.0: tests that need ReSimPair data which is not part of this repository (asset packs, run outputs,
validation evidence, reports) are reported as skipped with the missing path, instead of failing.

Only FileNotFoundError for a path under one of the data folders below is converted; every other failure is reported
unchanged. Provide the data (see real2sim/README.md) to run those tests.
"""
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIRS = tuple(os.path.join(ROOT, d) + os.sep for d in ("assets", "runs", "runtime", "validation", "reports", "provenance"))


def _missing_data(exc) -> str | None:
    while exc is not None:
        if isinstance(exc, FileNotFoundError) and exc.filename:
            p = os.path.abspath(str(exc.filename))
            if p.startswith(DATA_DIRS):
                return os.path.relpath(p, ROOT)
        exc = exc.__cause__ or exc.__context__
    return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.failed and call.excinfo is not None:
        missing = _missing_data(call.excinfo.value)
        if missing:
            rep.outcome = "skipped"
            rep.longrepr = (str(item.fspath), item.location[1] or 0, f"needs ReSimPair data not in this repository: {missing}")


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    outcome = yield
    rep = outcome.get_result()
    if rep.failed and getattr(rep, "longrepr", None) is not None:
        text = str(rep.longrepr)
        if "FileNotFoundError" in text:
            for d in DATA_DIRS:
                rel = os.path.basename(d.rstrip(os.sep))
                if (os.sep + rel + os.sep) in text or f"'{rel}/" in text:
                    rep.outcome = "skipped"
                    rep.longrepr = (str(collector.fspath), 0, f"needs ReSimPair data not in this repository ({rel}/)")
                    break
