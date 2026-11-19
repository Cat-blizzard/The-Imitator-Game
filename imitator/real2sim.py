"""Optional real -> sim -> human pipeline (ReSimPair), vendored in real2sim/resimpair.

    python -m imitator.real2sim doctor
    python -m imitator.real2sim pickplace compose --root runs/pnp_pair_demo \
        --index assets/general_pickplace_demo_v1/cache_index.json \
        --containers assets/general_pickplace_demo_v1/container_catalog.json --seed 2026102690

Arguments are passed unchanged to ReSimPair's CLI (real2sim/resimpair/resimpair.py) with the working directory set to
real2sim/resimpair, where its data folders live (assets/, runs/, runtime/, private/; all git-ignored). ManiSkill is
this repository's fork (real2sim/resimpair/PolicyGen/mani_skill is a link to ../../../mani_skill). Paid stages read
their keys from the environment (ECHOFLOW_API_KEY, ARK_API_KEY). See real2sim/README.md.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "real2sim" / "resimpair"


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not (ROOT / "resimpair.py").is_file():
        print(f"real2sim is not available: {ROOT} is missing", file=sys.stderr)
        return 2
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        argv = ["--help"]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "PolicyGen"), str(ROOT.parent.parent)]
                                        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return subprocess.run([sys.executable, "-B", str(ROOT / "resimpair.py"), *argv], cwd=ROOT, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
