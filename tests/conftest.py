import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
# The tests/imitator package must not shadow the platform's imitator package.
# Keep tests/ available for _paths, after the production source root.
sys.path[:] = [_REPO, _HERE, *(p for p in sys.path if p not in (_REPO, _HERE))]
