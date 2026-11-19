"""Shared test paths: repository root, checked-in test data, and a scratch directory for outputs.

IMITATOR_TEST_TMP overrides the scratch directory (default: <system tmp>/imitator_tests).
"""
import os
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))          # tests/
REPO = os.path.dirname(HERE)
DATA = os.path.join(HERE, "imitator", "data")
TMP = os.environ.get("IMITATOR_TEST_TMP") or os.path.join(tempfile.gettempdir(), "imitator_tests")
os.makedirs(TMP, exist_ok=True)


def repo_path(p: str) -> str:
    """repo-relative paths (as stored in the freeze manifests) -> absolute"""
    return p if os.path.isabs(p) else os.path.join(REPO, p)


def stock_tasks_available() -> bool:
    """ManiSkill's stock tabletop tasks (PickCube, PushCube, ...) exist only in ManiSkill 3.0.1, not in the Imitator
    fork; the imitator Panda wrappers of those tasks are registered only when they exist"""
    import importlib.util
    return importlib.util.find_spec("mani_skill.envs.tasks.tabletop.pick_cube") is not None


def needs_stock_tasks():
    import pytest
    return pytest.mark.skipif(not stock_tasks_available(),
                              reason="needs ManiSkill's stock tabletop tasks (ManiSkill 3.0.1; not in the Imitator fork)")
