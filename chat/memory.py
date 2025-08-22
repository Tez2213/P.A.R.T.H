import psycopg2
from pgvector.psycopg2 import register_vector
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer


class PostgresMemory:
    def __init__(
        self,
        dbname: str,
        user: str,
        password: str,
        host: str = "localhost",
        session_id: str = "default",
        embedding_dim: int = 1024,  # bge-m3 dimension
    ):
        self.conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host)
        self.session_id = session_id
        self.embedding_dim = embedding_dim

        # Register pgvector type with psycopg2
        register_vector(self.conn)

        # Load embedding model (BGE-M3)
        self.model = SentenceTransformer("BAAI/bge-m3")

        # Ensure memory table exists
        self._init_table()

    def _init_table(self):
        """Ensure the memory table exists with the correct embedding dimension."""
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS memory (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    embedding vector({self.embedding_dim})
                )
                """
            )
            self.conn.commit()

    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for given text using BGE-M3."""
        if not isinstance(text, str):
            raise ValueError("embed_text expects a string, got: {}".format(type(text)))
        return self.model.encode(text).tolist()

    def add(self, role: str, content: str, embedding: Optional[List[float]] = None):
        """Insert a new message with optional embedding."""
        if embedding is None:
            embedding = self.embed_text(content)

        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO memory (session_id, role, content, embedding) VALUES (%s, %s, %s, %s)",
                (self.session_id, role, content, embedding)
            )
            self.conn.commit()

    def relevant_history(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Retrieve the most relevant past messages using cosine distance
        via pgvector's <-> operator.
        """
        # Ensure query is a string
        if not isinstance(query, str):
            raise ValueError("relevant_history expects query as string")

        # Generate embedding for the query
        query_embedding = self.embed_text(query)

        with self.conn.cursor() as cur:
            # Explicitly cast to vector to avoid operator error
            cur.execute(
                f"""
                SELECT role, content
                FROM memory
                WHERE session_id = %s
                AND embedding IS NOT NULL
                ORDER BY embedding <-> %s::vector
                LIMIT %s
                """,
                (self.session_id, query_embedding, limit)
            )
            rows = cur.fetchall()

        # Convert results to list of dicts
        return [{"role": r[0], "content": r[1]} for r in rows]
