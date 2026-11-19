"""Claude backend (official Anthropic Python SDK) for veritask AI features: judges, predicate review, contract drafting.

Not exercised against the live API in this repo yet (no credentials configured); kept behind the same
.chat(text, images) interface as ai/models.py. Credentials resolve the SDK's usual way (ANTHROPIC_API_KEY,
ANTHROPIC_AUTH_TOKEN, or an `ant auth login` profile) - never put keys in task files.

Defaults: model claude-opus-5-5; adaptive thinking is always on for this model (control depth via effort);
server-side refusal fallbacks enabled ("default" routing) so a policy decline is retried on a fallback model.
"""
from __future__ import annotations

import base64, hashlib, json, mimetypes, time

import anthropic


class ClaudeBackend:
    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium", max_tokens: int = 4096,
                 log_path: str | None = None, timeout: float = 120.0):
        self.client = anthropic.Anthropic(timeout=timeout)
        self.model, self.effort, self.max_tokens, self.log_path = model, effort, max_tokens, log_path

    @staticmethod
    def _image_block(path: str) -> dict:
        media = mimetypes.guess_type(path)[0] or "image/png"
        with open(path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode("utf-8")
        return {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}

    def chat(self, text: str, images: list[str] = ()) -> str:
        content = [self._image_block(p) for p in images] + [{"type": "text", "text": text}]
        t0 = time.time()
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                output_config={"effort": self.effort},
                messages=[{"role": "user", "content": content}],
            )
        except anthropic.BadRequestError as e:
            raise RuntimeError(f"Claude bad request: {e.message}") from e
        except anthropic.RateLimitError as e:
            raise RuntimeError(f"Claude rate limited, retry after {e.response.headers.get('retry-after', '?')}s") from e
        except anthropic.APIStatusError as e:
            raise RuntimeError(f"Claude API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise RuntimeError("Claude network error") from e
        if response.stop_reason == "refusal":
            cat = response.stop_details.category if response.stop_details else None
            out = f"__REFUSAL__:{cat}"
        else:
            out = "".join(b.text for b in response.content if b.type == "text")
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(dict(model=response.model, prompt_sha1=hashlib.sha1(text.encode()).hexdigest()[:12],
                                        inputs=list(images), output=out, stop_reason=response.stop_reason,
                                        seconds=round(time.time() - t0, 3)), ensure_ascii=False) + "\n")
        return out
