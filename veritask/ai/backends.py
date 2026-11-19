"""Provider-neutral backend registry. Callers ask for a backend by name; heavy SDKs are imported lazily.

  local      -> ai.models.LocalVLM            (Hugging Face weights on the local GPU)
  openai     -> ai.models.OpenAICompatible    (vLLM, or any OpenAI-protocol server)
  claude     -> ai.claude_backend.ClaudeBackend (official Anthropic SDK)
  agent      -> ai.agent_queue.AgentQueue     (agent-in-the-loop queue; current stand-in for frontier APIs)
  gemini     -> not implemented yet (slot reserved; add ai/gemini_backend.py with the same .chat interface)
"""
from __future__ import annotations


def make(name: str, **kw):
    if name == "local":
        from .models import LocalVLM
        return LocalVLM(**kw)
    if name == "openai":
        from .models import OpenAICompatible
        return OpenAICompatible(**kw)
    if name == "claude":
        from .claude_backend import ClaudeBackend
        return ClaudeBackend(**kw)
    if name == "agent":
        from .agent_queue import AgentQueue
        return AgentQueue(**kw)
    raise NotImplementedError(f"backend '{name}' not implemented (reserved: gemini)")
