import psycopg2
from pgvector.psycopg2 import register_vector
from typing import List, Dict, Optional


class PostgresMemory:
    def __init__(self, dbname: str, user: str, password: str, host: str = "localhost",
                 session_id: str = "default", embedding_dim: int = 768):
        self.conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host)
        self.session_id = session_id
        self.embedding_dim = embedding_dim

        # Register pgvector type with psycopg2
        register_vector(self.conn)

        # Ensure memory table exists
        self._init_table()

    def _init_table(self):
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

    def add(self, role: str, content: str, embedding: Optional[List[float]] = None):
        with self.conn.cursor() as cur:
            if embedding:
                cur.execute(
                    "INSERT INTO memory (session_id, role, content, embedding) VALUES (%s, %s, %s, %s)",
                    (self.session_id, role, content, embedding)
                )
            else:
                cur.execute(
                    "INSERT INTO memory (session_id, role, content) VALUES (%s, %s, %s)",
                    (self.session_id, role, content)
                )
            self.conn.commit()

    def relevant_history(self, query_embedding: List[float], limit: int = 10) -> List[Dict]:
        """
        Retrieve most relevant past messages using cosine distance.
        Requires pgvector (<-> operator).
        """
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT role, content
                FROM memory
                WHERE session_id = %s
                AND embedding IS NOT NULL
                ORDER BY embedding <-> %s
                LIMIT %s
                """,
                (self.session_id, query_embedding, limit)
            )
            rows = cur.fetchall()
        return [{"role": r[0], "content": r[1]} for r in rows]
