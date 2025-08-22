import yaml
from typing import List
from .prompt_manager import PromptManager
from .memory import PostgresMemory  
from llm.base import Message
from llm.ollama_client import OllamaClient
from sentence_transformers import SentenceTransformer


class ChatRunner:
    def __init__(self, cfg_path: str = "config.yaml"):
        self.cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
        self.pm = PromptManager(
            system_path=self.cfg["prompts"]["system"],
            tools_path=self.cfg["prompts"].get("tools"),
        )

        # Postgres memory (embedding_dim = 1024 for BGE-M3)
        mem_cfg = self.cfg["memory"]
        self.mem = PostgresMemory(
            dbname=mem_cfg["dbname"],
            user=mem_cfg["user"],
            password=mem_cfg["password"],
            host=mem_cfg.get("host", "localhost"),
            session_id=mem_cfg.get("session_id", "default"),
            embedding_dim=1024
        )

        # Load Ollama chat model (phi4)
        self.llm = OllamaClient(
            model=self.cfg["models"]["ollama"],   # phi4
            host=self.cfg["ollama"]["host"],
            params=self.cfg.get("params", {}),
        )

        # Load BGE-M3 embedding model
        self.embedder = SentenceTransformer("BAAI/bge-m3")

    def _embed(self, text: str) -> List[float]:
        """Generate embeddings using BGE-M3"""
        return self.embedder.encode(text).tolist()

    def _build_messages(self, user_input: str) -> List[Message]:
        """
        Build context: system prompt + most relevant messages (embedding search) + current input.
        """
        system = {"role": "system", "content": self.pm.system_prompt()}

        # Retrieve relevant history using embeddings
        history = self.mem.relevant_history(user_input, limit=10)

        current = {"role": "user", "content": user_input}
        return [system, *history, current]

    def ask(self, user_input: str, stream: bool = True) -> str:
        messages = self._build_messages(user_input)

        # Chat via phi4
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

        # Save messages with embeddings (BGE-M3)
        user_embedding = self._embed(user_input)
        self.mem.add("user", user_input, user_embedding)

        answer_embedding = self._embed(answer)
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
