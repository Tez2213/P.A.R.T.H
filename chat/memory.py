import psycopg2
from pgvector.psycopg2 import register_vector
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import logging


class PostgresMemory:
    def __init__(self, config: dict):
        """
        Initialize PostgreSQL memory with complete config-driven approach.
        No hardcoded values - everything from config.
        """
        # Extract database config
        db_config = config["database"]["connection"]
        memory_config = config["memory"]
        
        # Connect to database
        self.conn = psycopg2.connect(
            dbname=db_config["dbname"],
            user=db_config["user"],
            password=db_config["password"],
            host=db_config["host"],
            port=db_config.get("port", 5432),
            connect_timeout=db_config.get("timeout", 30)
        )
        
        # Memory configuration
        self.session_id = memory_config["session_id"]
        self.embedding_dim = memory_config["embedding_dim"]
        self.similarity_threshold = memory_config.get("similarity_threshold", 0.7)
        self.max_history_retrieval = memory_config.get("max_history_retrieval", 10)
        self.enable_similarity_search = memory_config.get("enable_similarity_search", True)

        # Register pgvector type
        register_vector(self.conn)

        # Load embedding model from config
        embedding_model = config["models"]["embedding"]
        logging.info(f"Loading embedding model: {embedding_model}")
        self.model = SentenceTransformer(embedding_model)
        
        # Initialize database table
        self._init_table()

    def _init_table(self):
        """Create memory table with configurable embedding dimension."""
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS memory (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    embedding vector({self.embedding_dim}),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            # Create index for better performance
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_session_embedding 
                ON memory USING ivfflat (embedding vector_cosine_ops)
                """
            )
            self.conn.commit()

    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings using configured model."""
        if not isinstance(text, str):
            raise ValueError(f"embed_text expects a string, got: {type(text)}")
        return self.model.encode(text).tolist()

    def add(self, role: str, content: str, embedding: Optional[List[float]] = None):
        """Insert message with optional pre-computed embedding."""
        if embedding is None:
            embedding = self.embed_text(content)

        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO memory (session_id, role, content, embedding) VALUES (%s, %s, %s, %s)",
                (self.session_id, role, content, embedding)
            )
            self.conn.commit()

    def relevant_history(self, query: str, limit: Optional[int] = None) -> List[Dict]:
        """
        Retrieve relevant history using configurable parameters.
        """
        if not self.enable_similarity_search:
            return []
            
        if not isinstance(query, str):
            raise ValueError("relevant_history expects query as string")

        # Use configured limit or default
        limit = limit or self.max_history_retrieval
        
        query_embedding = self.embed_text(query)

        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT role, content, (embedding <-> %s::vector) as distance
                FROM memory
                WHERE session_id = %s
                AND embedding IS NOT NULL
                AND (embedding <-> %s::vector) < %s
                ORDER BY embedding <-> %s::vector
                LIMIT %s
                """,
                (query_embedding, self.session_id, query_embedding, 
                 self.similarity_threshold, query_embedding, limit)
            )
            rows = cur.fetchall()

        return [{"role": r[0], "content": r[1], "similarity": 1 - r[2]} for r in rows]

    def clear_session(self):
        """Clear current session data."""
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM memory WHERE session_id = %s", (self.session_id,))
            self.conn.commit()

    def get_session_stats(self) -> Dict:
        """Get statistics for current session."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM memory WHERE session_id = %s",
                (self.session_id,)
            )
            count = cur.fetchone()[0]
            
        return {
            "message_count": count,
            "session_id": self.session_id
        }
