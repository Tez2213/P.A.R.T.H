import json
import requests
from typing import List, Dict, Generator
from .base import BaseLLM, Message

def embed(self, text: str) -> List[float]:
    """
    Generate embeddings for text using Ollama's embeddings API.
    Default model: nomic-embed-text
    """
    url = f"{self.host}/api/embeddings"
    resp = requests.post(url, json={"model": "nomic-embed-text", "prompt": text}, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    return data.get("embedding", [])

class OllamaClient(BaseLLM):
    def __init__(self, model: str, host: str = "http://127.0.0.1:11434", params=None):
        super().__init__(model, params)
        self.host = host.rstrip("/")


    def _payload(self, messages: List[Message], stream: bool) -> dict:
        # Map common params to Ollama keys when possible
        p = self.params or {}
        options = {}
        if "temperature" in p: options["temperature"] = p["temperature"]
        if "top_p" in p: options["top_p"] = p["top_p"]
        if "seed" in p: options["seed"] = p["seed"]
        if "max_tokens" in p: options["num_predict"] = p["max_tokens"]
        return {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": options,
        }


    def chat(self, messages: List[Message]) -> str:
        url = f"{self.host}/api/chat"
        resp = requests.post(url, json=self._payload(messages, stream=False), timeout=600)
        resp.raise_for_status()
        data = resp.json()
        # Ollama returns {"message": {"role": "assistant", "content": "..."}, ...}
        return data.get("message", {}).get("content", "")


    def stream(self, messages: List[Message]) -> Generator[str, None, None]:
        url = f"{self.host}/api/chat"
        with requests.post(url, json=self._payload(messages, stream=True), stream=True, timeout=600) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line.decode("utf-8"))
                if chunk.get("done"):
                    break
                delta = chunk.get("message", {}).get("content", "")
                if delta:
                    yield delta