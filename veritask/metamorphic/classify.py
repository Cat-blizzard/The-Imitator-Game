"""Task classification for the metamorphic tests (card B1 step 2): from the instruction.

  place  the instruction asks for a persistent end state of the WORLD (put / place / stack / insert / push or pull
         to a position / open / close / turn / hang / move onto ...): after the robot lets go and leaves, a genuine
         completion persists.  M2, M3, M5 apply.
  hold   the instruction asks the ROBOT to keep something (lift / hold / pick up without a place target / shake /
         hand over / reach a point): releasing is a failure by the task's own definition.  M2/M3/M5 do not apply.
  event  the instruction asks for a momentary act (press / click / ring / hit / beat): a persistent end state is
         not part of the task.  Treated like 'hold' (M2/M3/M5 do not apply); reported separately.

`classify(instruction)` is the keyword rule used when a benchmark script gives no explicit table; the benchmark
scripts pass explicit, pre-registered per-task tables (with the instruction text and a rationale), which win.
"""
from __future__ import annotations

import re

EVENT_WORDS = ("press", "click", "ring", "beat", "hit", "tap", "hammer the", "knock")
HOLD_WORDS = ("lift", "hold", "raise", "pick up", "pick", "grab", "shake", "hand over", "handover", "reach",
              "take out", "unplug")
PLACE_WORDS = ("put", "place", "stack", "insert", "plug", "push", "pull", "slide", "open", "close", "turn",
               "hang", "move", "sweep", "drop", "stand", "upright", "into", "onto", "on top", "in the", "on the",
               "to the goal", "to the target", "rotate", "flip", "fold", "cover", "store", "dump", "pour")


def _has(text: str, words) -> list:
    t = " " + re.sub(r"[^a-z0-9 ]+", " ", text.lower()) + " "
    return [w for w in words if f" {w} " in t or (" " in w and w in t)]


def classify(instruction: str) -> tuple[str, str]:
    """-> (class, rationale) by keyword rule."""
    ev, hd, pl = _has(instruction, EVENT_WORDS), _has(instruction, HOLD_WORDS), _has(instruction, PLACE_WORDS)
    if ev and not pl:
        return "event", f"event words {ev}"
    if hd and not pl:
        return "hold", f"hold words {hd}, no place target"
    if pl:
        return "place", f"place words {pl}"
    if ev:
        return "event", f"event words {ev}"
    return "place", "default (no keyword)"
