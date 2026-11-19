import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (_HERE, os.path.dirname(_HERE)):          # tests/ (for _paths) and the repository root
    if p not in sys.path:
        sys.path.insert(0, p)
