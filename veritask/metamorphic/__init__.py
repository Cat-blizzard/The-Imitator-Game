"""veritask.metamorphic: benchmark-agnostic metamorphic tests of success predicates (card B1).

  from veritask.metamorphic import Adapter, Protocol, run_episode
  row = run_episode(adapter, episode, Protocol(nudge_dv_mps=0.2), task=..., episode_id=..., task_class="place")

CLI: python -m veritask metamorphic {run,summarize,classify,items}
"""
from .adapter import Adapter, PhaseLog, Rollout
from .classify import classify
from .protocol import ITEMS, PROTOCOL_VERSION, Protocol, applicable
from .runner import ROW_SCHEMA, append_row, done_episodes, nudge_direction, run_episode
from .stats import count, summarize, summarize_group, wilson

__all__ = ["Adapter", "PhaseLog", "Rollout", "Protocol", "ITEMS", "PROTOCOL_VERSION", "applicable", "classify",
           "run_episode", "append_row", "done_episodes", "nudge_direction", "ROW_SCHEMA", "count", "summarize",
           "summarize_group", "wilson"]
