#!/usr/bin/env python3
"""Run the standalone pipeline from any working directory."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_generation_methods.r2h_pipeline.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
