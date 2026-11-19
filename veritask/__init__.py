"""veritask: credible physical task verdicts for robot manipulation (working name).

Core idea: a task is specified by a physical outcome contract (spec/contract.py); every verdict carries margins,
'unknown' when measurements are missing, and the verifier's own credibility is measured (audit/credibility.py).
Submodules are imported lazily by callers so that `import veritask` does not pull in simulators or ML libraries.
"""
__version__ = "0.1.0"
