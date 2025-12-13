# feedback_store.py
import sqlite3
import json
import time
from typing import Optional, Dict, Any, List

DB_PATH = "feedback.db"

class FeedbackStore:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL,
                user_id TEXT,
                question TEXT,
                answer TEXT,
                context_json TEXT,
                sources_json TEXT,
                score REAL,
                feedback INTEGER -- 1 полезно, 0 бесполезно
            )
            """)
            conn.commit()

    def log_feedback(self, user_id: str, question: str, answer: str, context: Dict[str,Any], sources: List[Dict[str,Any]], score: Optional[float], feedback: int):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO feedback (ts, user_id, question, answer, context_json, sources_json, score, feedback) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (time.time(), user_id, question, answer, json.dumps(context, ensure_ascii=False), json.dumps(sources, ensure_ascii=False), score, feedback)
            )
            conn.commit()

    def fetch_all(self, limit: int = 10000):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, ts, user_id, question, answer, context_json, sources_json, score, feedback FROM feedback ORDER BY ts DESC LIMIT ?", (limit,))
            rows = cur.fetchall()
        result = []
        for r in rows:
            result.append({
                "id": r[0],
                "ts": r[1],
                "user_id": r[2],
                "question": r[3],
                "answer": r[4],
                "context": json.loads(r[5]) if r[5] else {},
                "sources": json.loads(r[6]) if r[6] else [],
                "score": r[7],
                "feedback": r[8],
            })
        return result

#