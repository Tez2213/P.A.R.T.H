import yaml
from typing import List
from .prompt_manager import PromptManager
from .memory import PostgresMemory  
from llm.base import Message
from llm.ollama_client import OllamaClient
from llm.hf_client import HFClient

_BACKENDS = {
    "ollama": OllamaClient,
    "huggingface": HFClient,
}


class ChatRunner:
    def __init__(self, cfg_path: str = "config.yaml"):
        self.cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
        self.pm = PromptManager(
            system_path=self.cfg["prompts"]["system"],
            tools_path=self.cfg["prompts"].get("tools"),
        )

        # Postgres memory (embedding only)
        mem_cfg = self.cfg["memory"]
        self.mem = PostgresMemory(
            dbname=mem_cfg["dbname"],
            user=mem_cfg["user"],
            password=mem_cfg["password"],
            host=mem_cfg.get("host", "localhost"),
            session_id=mem_cfg.get("session_id", "default"),
        )

        # Choose backend
        backend = self.cfg.get("backend", "ollama")
        params = self.cfg.get("params", {})
        if backend == "ollama":
            self.llm = _BACKENDS[backend](
                model=self.cfg["models"]["ollama"],
                host=self.cfg["ollama"]["host"],
                params=params,
            )
        elif backend == "huggingface":
            hf = self.cfg["huggingface"]
            self.llm = _BACKENDS[backend](
                model=self.cfg["models"]["huggingface"],
                params=params,
                device=hf.get("device", "auto"),
                dtype=hf.get("dtype", "auto"),
                trust_remote_code=bool(hf.get("trust_remote_code", False)),
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _build_messages(self, user_input: str) -> List[Message]:
        """
        Build context: system prompt + most relevant messages (embedding search) + current input.
        """
        system = {"role": "system", "content": self.pm.system_prompt()}

        # Generate embedding for current input
        query_embedding = self.llm.embed(user_input)

        # Retrieve relevant history from Postgres (by embedding similarity)
        history = self.mem.relevant_history(query_embedding, limit=10)

        current = {"role": "user", "content": user_input}
        return [system, *history, current]

    def ask(self, user_input: str, stream: bool = True) -> str:
        messages = self._build_messages(user_input)

        if stream:
            chunks = []
            for token in self.llm.stream(messages):
                print(token, end="", flush=True)
                chunks.append(token)
            print()
            answer = "".join(chunks)
        else:
            answer = self.llm.complete(messages)
            print(answer)

        # Save user + assistant messages with embeddings
        user_embedding = self.llm.embed(user_input)
        self.mem.add("user", user_input, user_embedding)

        answer_embedding = self.llm.embed(answer)
        self.mem.add("assistant", answer, answer_embedding)

        return answer


if __name__ == "__main__":
    runner = ChatRunner("config.yaml")
    print("Hello! My name is PARTH. How can I help you?\n")

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            print("\n👋 Goodbye!")
            break
        print("Bot: ", end="")
        runner.ask(user_input, stream=True)
        print()
