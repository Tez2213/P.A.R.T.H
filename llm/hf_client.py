from typing import List, Dict, Generator
from .base import BaseLLM, Message


# Lazy import to keep startup fast if HF not used


def _load_pipeline(model: str, device: str, dtype: str, trust_remote_code: bool):
    from transformers import pipeline
    return pipeline(
        task="text-generation",
        model=model,
        device=device if device != "auto" else None,  # transformers auto-handles if None
        torch_dtype=None if dtype == "auto" else dtype,
        trust_remote_code=trust_remote_code,
    )


class HFClient(BaseLLM):
    def __init__(self, model: str, params=None, device="auto", dtype="auto", trust_remote_code=False):
        super().__init__(model, params)
        self.pipe = _load_pipeline(model, device, dtype, trust_remote_code)


    def _build_prompt(self, messages: List[Message]) -> str:
        # Simple fallback prompt formatting; for chat-tuned models you may prefer a tokenizer chat template
        parts = []
        for m in messages:
            role = m.get("role", "user")
            parts.append(f"[{role.upper()}]: {m.get('content','')}")
        parts.append("[ASSISTANT]:")
        return "\n\n".join(parts)


    def chat(self, messages: List[Message]) -> str:
        p = self.params or {}
        prompt = self._build_prompt(messages)
        out = self.pipe(
            prompt,
            max_new_tokens=p.get("max_tokens", 1024),
            temperature=p.get("temperature", 0.2),
            top_p=p.get("top_p", 0.9),
            do_sample=True,
        )[0]["generated_text"]
        # Return text after the last [ASSISTANT]: tag
        return out.split("[ASSISTANT]:", 1)[-1].strip()


    def stream(self, messages: List[Message]):
        # Basic non-streaming fallback
        yield self.chat(messages)