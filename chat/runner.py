import yaml
import gc
import logging
import os
from typing import List, Optional
from .prompt_manager import PromptManager
from .memory import PostgresMemory  
from llm.base import Message
from llm.ollama_client import OllamaClient


class ChatRunner:
    def __init__(self, cfg_path: str = "config.yaml"):
        """
        Initialize ChatRunner with complete config-driven approach.
        Zero hardcoding - everything configurable.
        """
        # Load configuration
        self.cfg = self._load_config(cfg_path)
        
        # Setup logging
        self._setup_logging()
        
        # Setup performance optimizations
        self._setup_performance()
        
        # Initialize components
        self._init_prompt_manager()
        self._init_memory()
        self._init_llm()
        
        logging.info("ChatRunner initialized successfully")

    def _load_config(self, cfg_path: str) -> dict:
        """Load and validate configuration."""
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            
            # Validate required sections
            required_sections = ["models", "prompts", "database", "memory", "ollama"]
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Missing required config section: {section}")
            
            return config
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {cfg_path}: {e}")

    def _setup_logging(self):
        """Setup logging from config."""
        log_config = self.cfg.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO").upper())
        
        # Create logs directory if specified
        log_file = log_config.get("file")
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        logging.basicConfig(
            level=level,
            filename=log_file,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def _setup_performance(self):
        """Setup performance optimizations from config."""
        perf_config = self.cfg.get("performance", {})
        
        if perf_config.get("enable_garbage_collection", True):
            gc.enable()
        
        # Set memory limits if specified
        memory_limit = perf_config.get("memory_limit_mb")
        if memory_limit:
            # Set environment variables for memory optimization
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{memory_limit//4}"
            os.environ["TOKENIZERS_PARALLELISM"] = str(perf_config.get("parallel_processing", False)).lower()

    def _init_prompt_manager(self):
        """Initialize prompt manager from config."""
        prompt_config = self.cfg["prompts"]
        self.pm = PromptManager(
            system_path=prompt_config["system"],
            tools_path=prompt_config.get("tools")
        )

    def _init_memory(self):
        """Initialize memory with shared embedding model."""
        logging.info("Initializing memory system...")
        self.mem = PostgresMemory(self.cfg)
        
        # Share the embedding model to prevent duplicate loading
        self.embedder = self.mem.model
        logging.info(f"Memory initialized with {self.cfg['models']['embedding']}")

    def _init_llm(self):
        """Initialize LLM based on backend configuration."""
        backend = self.cfg.get("backend", "ollama")
        
        if backend == "ollama":
            self.llm = OllamaClient(
                model=self.cfg["models"]["ollama"],
                host=self.cfg["ollama"]["host"],
                params=self.cfg.get("params", {})
            )
            logging.info(f"Ollama client initialized with model: {self.cfg['models']['ollama']}")
        elif backend == "huggingface":
            # Future: implement HuggingFace client
            raise NotImplementedError("HuggingFace backend not yet implemented")
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def _embed(self, text: str) -> List[float]:
        """Generate embeddings using shared model."""
        return self.embedder.encode(text).tolist()

    def _build_messages(self, user_input: str) -> List[Message]:
        """Build context messages from config and history."""
        # Validate input length
        security_config = self.cfg.get("security", {})
        max_length = security_config.get("max_input_length", 4096)
        if len(user_input) > max_length:
            raise ValueError(f"Input too long: {len(user_input)} > {max_length}")
        
        # System prompt
        system = {"role": "system", "content": self.pm.system_prompt()}

        # Retrieve relevant history using configured parameters
        history = self.mem.relevant_history(user_input)
        
        # Current user input
        current = {"role": "user", "content": user_input}
        
        return [system, *history, current]

    def ask(self, user_input: str, stream: Optional[bool] = None) -> str:
        """
        Process user input and generate response.
        Stream setting from config if not explicitly provided.
        """
        # Use stream setting from config if not provided
        if stream is None:
            stream = self.cfg.get("ollama", {}).get("stream", True)
        
        # Build messages
        messages = self._build_messages(user_input)

        # Generate response
        if stream:
            chunks = []
            for token in self.llm.stream(messages):
                print(token, end="", flush=True)
                chunks.append(token)
            print()
            answer = "".join(chunks)
        else:
            answer = self.llm.chat(messages)
            print(answer)

        # Save conversation with embeddings
        user_embedding = self._embed(user_input)
        self.mem.add("user", user_input, user_embedding)

        answer_embedding = self._embed(answer)
        self.mem.add("assistant", answer, answer_embedding)

        # Performance cleanup
        if self.cfg.get("performance", {}).get("enable_garbage_collection", True):
            gc.collect()

        return answer

    def get_stats(self) -> dict:
        """Get session statistics."""
        return {
            "config": {
                "backend": self.cfg.get("backend"),
                "model": self.cfg["models"][self.cfg.get("backend", "ollama")],
                "embedding_model": self.cfg["models"]["embedding"]
            },
            "memory": self.mem.get_session_stats()
        }

    def clear_memory(self):
        """Clear current session memory."""
        self.mem.clear_session()
        logging.info("Session memory cleared")


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
