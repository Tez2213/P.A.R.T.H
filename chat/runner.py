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
        # Store config path for vision system
        self.cfg_path = cfg_path
        
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
        self._init_vision()
        
        logging.debug("ChatRunner initialized successfully")

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
        logging.debug("Initializing memory system...")
        self.mem = PostgresMemory(self.cfg)
        
        # Share the embedding model to prevent duplicate loading
        self.embedder = self.mem.model
        logging.debug(f"Memory initialized with {self.cfg['models']['embedding']}")

    def _init_llm(self):
        """Initialize LLM based on backend configuration."""
        backend = self.cfg.get("backend", "ollama")
        
        if backend == "ollama":
            self.llm = OllamaClient(
                model=self.cfg["models"]["ollama"],
                host=self.cfg["ollama"]["host"],
                params=self.cfg.get("params", {})
            )
            logging.debug(f"Ollama client initialized with model: {self.cfg['models']['ollama']}")
        elif backend == "huggingface":
            # Future: implement HuggingFace client
            raise NotImplementedError("HuggingFace backend not yet implemented")
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def _init_vision(self):
        """Initialize vision system if enabled."""
        vision_config = self.cfg.get("vision", {})
        if vision_config.get("enabled", False):
            try:
                # Configure ultralytics logging based on vision config
                vision_logging = vision_config.get("logging", {})
                if vision_logging.get("suppress_ultralytics", True):
                    import os
                    import warnings
                    os.environ['YOLO_VERBOSE'] = 'False'
                    os.environ['ULTRALYTICS_VERBOSE'] = 'False'
                    warnings.filterwarnings('ignore', category=UserWarning, module='ultralytics')
                
                # Import vision system
                import sys
                sys.path.append(os.path.dirname(os.path.dirname(__file__)))
                from vision.vision import FastVision
                
                # Set ultralytics logging level
                import logging as ultralytics_logging
                ultralytics_log_level = vision_logging.get("log_level", "ERROR").upper()
                ultralytics_logging.getLogger('ultralytics').setLevel(getattr(ultralytics_logging, ultralytics_log_level))
                
                self.vision = FastVision(self.cfg_path if hasattr(self, 'cfg_path') else "config.yaml")
                
                # Set up vision callback to update memory with visual observations
                self.vision.set_vision_callback(self._on_vision_update)
                
                # Start background vision processing
                self.vision.start_background_vision()
                
                logging.debug("Vision system initialized and started")
            except Exception as e:
                logging.error(f"Failed to initialize vision system: {e}")
                self.vision = None
        else:
            self.vision = None
            logging.debug("Vision system disabled in config")

    def _on_vision_update(self, description: str, detections: List):
        """Callback for vision updates - store in memory for context."""
        try:
            # Create a vision context message
            vision_context = f"[VISION UPDATE] {description}"
            
            # Store in memory with embedding for retrieval
            vision_embedding = self._embed(vision_context)
            self.mem.add("system", vision_context, vision_embedding)
            
            logging.debug(f"Vision update stored: {description}")
        except Exception as e:
            logging.error(f"Error storing vision update: {e}")

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
        system_content = self.pm.system_prompt()
        
        # Add current vision context if available and user asks about vision
        vision_keywords = ["see", "look", "vision", "camera", "what", "show", "view", "observe", "visual"]
        if self.vision and any(keyword in user_input.lower() for keyword in vision_keywords):
            current_vision = self.vision.get_current_vision()
            if current_vision["enabled"] and current_vision["description"]:
                vision_context = f"\n\nCURRENT VISION: {current_vision['description']}"
                system_content += vision_context
        
        system = {"role": "system", "content": system_content}

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
        stats = {
            "config": {
                "backend": self.cfg.get("backend"),
                "model": self.cfg["models"][self.cfg.get("backend", "ollama")],
                "embedding_model": self.cfg["models"]["embedding"]
            },
            "memory": self.mem.get_session_stats()
        }
        
        # Add vision stats if available
        if hasattr(self, 'vision') and self.vision:
            vision_state = self.vision.get_current_vision()
            stats["vision"] = {
                "enabled": vision_state["enabled"],
                "model_type": vision_state["model_type"],
                "current_detections": len(vision_state["detections"]),
                "description": vision_state["description"]
            }
        
        return stats

    def clear_memory(self):
        """Clear current session memory."""
        self.mem.clear_session()
        logging.info("Session memory cleared")

    def stop_vision(self):
        """Stop vision system."""
        if hasattr(self, 'vision') and self.vision:
            self.vision.stop()
            logging.info("Vision system stopped")

    def get_vision_description(self) -> str:
        """Get current vision description for manual queries."""
        if hasattr(self, 'vision') and self.vision:
            return self.vision.capture_frame_description()
        return "Vision system is not available."


if __name__ == "__main__":
    runner = ChatRunner("config.yaml")
    print("Hello! My name is PARTH. How can I help you?")
    print("I can see through my camera! Ask me 'what do you see?' to know what I'm looking at.")
    print("Type 'exit' to quit, 'stats' for system info, or 'vision' to check my vision status.\n")

    try:
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in {"exit", "quit"}:
                print("\n👋 Goodbye!")
                runner.stop_vision()  # Clean shutdown of vision
                break
            elif user_input.lower() == "stats":
                stats = runner.get_stats()
                print(f"\n📊 System Stats:")
                print(f"Backend: {stats['config']['backend']}")
                print(f"Model: {stats['config']['model']}")
                print(f"Memory: {stats['memory']['message_count']} messages")
                if 'vision' in stats:
                    print(f"Vision: {stats['vision']['model_type']} ({'enabled' if stats['vision']['enabled'] else 'disabled'})")
                    if stats['vision']['enabled'] and stats['vision']['description']:
                        print(f"Current view: {stats['vision']['description']}")
                print()
                continue
            elif user_input.lower() == "vision":
                vision_desc = runner.get_vision_description()
                print(f"👁️  Vision Status: {vision_desc}\n")
                continue
            elif not user_input:
                continue
                
            print("PARTH: ", end="")
            runner.ask(user_input, stream=True)
            print()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        runner.stop_vision()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        runner.stop_vision()
