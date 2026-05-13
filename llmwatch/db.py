import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS query_metrics (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project     TEXT    NOT NULL,
    model       TEXT    NOT NULL,
    timestamp   TEXT    NOT NULL,
    latency_ms  REAL    NOT NULL,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens  INTEGER GENERATED ALWAYS AS (input_tokens + output_tokens) VIRTUAL,
    cost_usd    REAL    NOT NULL DEFAULT 0.0,
    ragas_score REAL,
    error       TEXT
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_project_ts ON query_metrics (project, timestamp);
"""


class MetricsDB:
    def __init__(self, db_path: str = "llmwatch.db"):
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        if not getattr(self._local, "conn", None):
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self) -> None:
        conn = self._conn()
        conn.execute(_CREATE_TABLE)
        conn.execute(_CREATE_INDEX)
        conn.commit()

    def insert_metric(
        self,
        project: str,
        model: str,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        ragas_score: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        conn = self._conn()
        conn.execute(
            """
            INSERT INTO query_metrics
                (project, model, timestamp, latency_ms, input_tokens,
                 output_tokens, cost_usd, ragas_score, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project,
                model,
                datetime.now(timezone.utc).isoformat(),
                latency_ms,
                input_tokens,
                output_tokens,
                cost_usd,
                ragas_score,
                error,
            ),
        )
        conn.commit()

    def get_metrics(
        self, project: str, limit: int = 500
    ) -> List[Dict[str, Any]]:
        conn = self._conn()
        rows = conn.execute(
            """
            SELECT id, model, timestamp, latency_ms, input_tokens,
                   output_tokens, (input_tokens + output_tokens) as total_tokens,
                   cost_usd, ragas_score, error
            FROM query_metrics
            WHERE project = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (project, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_percentiles(self, project: str) -> Dict[str, float]:
        """Return P50 and P95 latency in ms for the project."""
        conn = self._conn()
        rows = conn.execute(
            "SELECT latency_ms FROM query_metrics WHERE project = ? AND error IS NULL ORDER BY latency_ms",
            (project,),
        ).fetchall()
        if not rows:
            return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
        values = [r[0] for r in rows]
        n = len(values)
        return {
            "p50_ms": round(values[int(n * 0.50)], 2),
            "p95_ms": round(values[min(int(n * 0.95), n - 1)], 2),
            "p99_ms": round(values[min(int(n * 0.99), n - 1)], 2),
        }

    def get_cumulative_cost(self, project: str) -> float:
        conn = self._conn()
        row = conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0.0) FROM query_metrics WHERE project = ?",
            (project,),
        ).fetchone()
        return round(row[0], 6)

    def get_summary(self, project: str) -> Dict[str, Any]:
        conn = self._conn()
        row = conn.execute(
            """
            SELECT
                COUNT(*)                             AS total_queries,
                COALESCE(AVG(latency_ms), 0)         AS avg_latency_ms,
                COALESCE(SUM(cost_usd), 0)           AS total_cost_usd,
                COALESCE(AVG(cost_usd), 0)           AS avg_cost_usd,
                COALESCE(SUM(input_tokens), 0)       AS total_input_tokens,
                COALESCE(SUM(output_tokens), 0)      AS total_output_tokens,
                COALESCE(AVG(ragas_score), 0)        AS avg_ragas_score,
                SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS error_count
            FROM query_metrics
            WHERE project = ?
            """,
            (project,),
        ).fetchone()
        summary = dict(row)
        summary.update(self.get_percentiles(project))
        summary["error_rate"] = (
            round(summary["error_count"] / summary["total_queries"] * 100, 2)
            if summary["total_queries"] > 0
            else 0.0
        )
        return summary

    def update_last_ragas_score(self, project: str, score: float) -> None:
        """Update the most recent metric row with a RAGAS score."""
        conn = self._conn()
        conn.execute(
            """
            UPDATE query_metrics SET ragas_score = ?
            WHERE id = (
                SELECT id FROM query_metrics
                WHERE project = ?
                ORDER BY id DESC LIMIT 1
            )
            """,
            (round(score, 4), project),
        )
        conn.commit()

    def get_cost_by_model(self, project: str) -> List[Dict[str, Any]]:
        conn = self._conn()
        rows = conn.execute(
            """
            SELECT model,
                   COUNT(*)              AS queries,
                   SUM(cost_usd)         AS total_cost_usd,
                   AVG(cost_usd)         AS avg_cost_usd,
                   SUM(input_tokens)     AS total_input_tokens,
                   SUM(output_tokens)    AS total_output_tokens,
                   AVG(latency_ms)       AS avg_latency_ms
            FROM query_metrics
            WHERE project = ?
            GROUP BY model
            ORDER BY total_cost_usd DESC
            """,
            (project,),
        ).fetchall()
        return [dict(r) for r in rows]
