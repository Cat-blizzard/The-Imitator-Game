"""Compatibility imports for ACT; native observation/action IO lives in public evaluation."""
from imitator.eval.native_io import live_observation, masked_l1, prepare_rgb, split_action

__all__ = ["live_observation", "masked_l1", "prepare_rgb", "split_action"]

if __name__ == "__main__":
    import runpy
    runpy.run_module("imitator.eval.native_io", run_name="__main__")
