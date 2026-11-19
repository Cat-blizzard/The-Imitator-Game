"""Model adapters for AI-native features. One interface, several backends:

  - LocalVLM: a Hugging Face / ModelScope vision-language model run in-process on the local GPU (no data leaves
    the machine, no per-call cost; used for judges and for drafting/reviewing contracts)
  - OpenAICompatible: any server speaking the OpenAI chat-completions protocol (vLLM, commercial APIs). Keys come
    from the environment, never from task files.

Every call is recorded (model id + revision, prompt hash, inputs, output, latency) so AI-produced artifacts are
reproducible and auditable. The platform never treats a model output as final when a physical check exists:
model output -> proposal -> deterministic verification (see ai/loop.py).
"""
from __future__ import annotations

import hashlib, json, os, time
from dataclasses import dataclass


@dataclass
class CallRecord:
    model: str
    prompt_sha1: str
    inputs: list[str]
    output: str
    seconds: float

    def to_json(self):
        return json.dumps(self.__dict__, ensure_ascii=False)


class LocalVLM:
    """Qwen-VL style chat model via transformers (AutoModelForImageTextToText + AutoProcessor)."""

    def __init__(self, path: str, max_new_tokens: int = 16, dtype: str = "bfloat16", log_path: str | None = None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.path = path; self.max_new_tokens = max_new_tokens; self.log_path = log_path
        self.processor = AutoProcessor.from_pretrained(path)
        self.model = AutoModelForImageTextToText.from_pretrained(path, dtype=getattr(torch, dtype), device_map="cuda").eval()

    def chat(self, text: str, images: list[str] = ()) -> str:
        import torch
        from PIL import Image
        content = [{"type": "image", "image": Image.open(p).convert("RGB")} for p in images] + [{"type": "text", "text": text}]
        msgs = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                                    return_tensors="pt").to(self.model.device)
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        txt = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        self._log(text, list(images), txt, time.time() - t0)
        return txt

    def yes_probability(self, text: str, images: list[str] = ()) -> float:
        """P('YES') vs P('NO') for the first generated token — a calibrated-ish score for AUROC."""
        import torch
        from PIL import Image
        content = [{"type": "image", "image": Image.open(p).convert("RGB")} for p in images] + [{"type": "text", "text": text}]
        inputs = self.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
                                                    add_generation_prompt=True, return_dict=True, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1]
        tok = self.processor.tokenizer
        ids = {w: tok.encode(w, add_special_tokens=False)[0] for w in ("YES", "NO", "Yes", "No")}
        l = logits.float()
        y = torch.logsumexp(torch.stack([l[ids["YES"]], l[ids["Yes"]]]), 0); n = torch.logsumexp(torch.stack([l[ids["NO"]], l[ids["No"]]]), 0)
        return float(torch.sigmoid(y - n))

    def _log(self, text, images, out, dt):
        if not self.log_path:
            return
        rec = CallRecord(os.path.basename(self.path.rstrip("/")), hashlib.sha1(text.encode()).hexdigest()[:12], images, out, round(dt, 3))
        with open(self.log_path, "a") as f:
            f.write(rec.to_json() + "\n")


class OpenAICompatible:
    """Minimal chat client for OpenAI-compatible endpoints (text only here; images as data URLs if needed)."""

    def __init__(self, base_url: str, model: str, api_key_env: str = "OPENAI_API_KEY", log_path: str | None = None):
        self.base_url = base_url.rstrip("/"); self.model = model; self.key = os.environ.get(api_key_env, ""); self.log_path = log_path

    def chat(self, text: str, images: list[str] = ()) -> str:
        import urllib.request
        body = json.dumps({"model": self.model, "messages": [{"role": "user", "content": text}], "temperature": 0}).encode()
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body,
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        t0 = time.time()
        out = json.load(urllib.request.urlopen(req, timeout=120))["choices"][0]["message"]["content"]
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(CallRecord(self.model, hashlib.sha1(text.encode()).hexdigest()[:12], list(images), out, round(time.time() - t0, 3)).to_json() + "\n")
        return out
