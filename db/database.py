"""
EdgeForge database abstraction layer.

Currently backed by SQLite (stdlib sqlite3). Kept deliberately thin and
dict-based (not a heavy ORM) so migrating to PostgreSQL later means:
  - swapping the connection factory
  - replacing '?' placeholders with '%s' (or using a query-builder shim)
  - the SQL in schema.sql is already ANSI-ish and avoids SQLite-only
    features (no unusual pragmas relied on at query time).

All access goes through Database, which returns rows as dicts.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Optional

SCHEMA_PATH = Path(__file__).parent / "schema.sql"
DEFAULT_DB_PATH = os.environ.get("EDGEFORGE_DB_PATH", "edgeforge.db")


def _dict_factory(cursor, row):
    fields = [c[0] for c in cursor.description]
    return dict(zip(fields, row))


class Database:
    """Thread-safe-ish wrapper around a single SQLite file.

    SQLite connections aren't safe to share across threads by default;
    we open one connection per thread via threading.local(), which is
    fine for Flask's default dev server and for gunicorn with threaded
    workers. Under heavy concurrency this should move to a real
    connection pool against PostgreSQL.
    """

    def __init__(self, path: str = DEFAULT_DB_PATH):
        self.path = path
        self._local = threading.local()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(self.path, timeout=30)
            conn.row_factory = _dict_factory
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            self._local.conn = conn
        return self._local.conn

    def _init_schema(self):
        conn = self._connect()
        with open(SCHEMA_PATH, "r") as f:
            conn.executescript(f.read())
        conn.commit()

    @contextmanager
    def cursor(self):
        conn = self._connect()
        cur = conn.cursor()
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()

    def execute(self, sql: str, params: Iterable[Any] = ()) -> int:
        """Run an INSERT/UPDATE/DELETE. Returns lastrowid."""
        with self.cursor() as cur:
            cur.execute(sql, params)
            return cur.lastrowid

    def executemany(self, sql: str, seq_of_params: Iterable[Iterable[Any]]) -> None:
        with self.cursor() as cur:
            cur.executemany(sql, seq_of_params)

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[dict]:
        with self.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def query_one(self, sql: str, params: Iterable[Any] = ()) -> Optional[dict]:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    # ------------------------------------------------------------------
    # Convenience helpers used across modules
    # ------------------------------------------------------------------

    def get_or_create_security(self, symbol: str, **kwargs) -> int:
        row = self.query_one("SELECT id FROM securities WHERE symbol = ?", (symbol,))
        if row:
            return row["id"]
        cols = ["symbol"] + list(kwargs.keys())
        placeholders = ",".join(["?"] * len(cols))
        values = [symbol] + list(kwargs.values())
        return self.execute(
            f"INSERT INTO securities ({','.join(cols)}) VALUES ({placeholders})",
            values,
        )

    def get_or_create_feature_definition(
        self, name: str, category: str, description: str = "",
        formula_ref: str = "", params: Optional[dict] = None, origin: str = "builtin"
    ) -> int:
        row = self.query_one("SELECT id FROM feature_definitions WHERE name = ?", (name,))
        if row:
            return row["id"]
        return self.execute(
            """INSERT INTO feature_definitions
               (name, category, description, formula_ref, params_json, origin)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (name, category, description, formula_ref, json.dumps(params or {}), origin),
        )

    def log_research_event(self, hypothesis_id: Optional[int], experiment_id: Optional[int],
                            event: str, detail: str = ""):
        self.execute(
            """INSERT INTO research_log (hypothesis_id, experiment_id, event, detail)
               VALUES (?, ?, ?, ?)""",
            (hypothesis_id, experiment_id, event, detail),
        )

    def research_stats(self) -> dict:
        """Powers the dashboard's transparency numbers."""
        total = self.query_one("SELECT COUNT(*) c FROM hypotheses")["c"]
        survived_initial = self.query_one(
            "SELECT COUNT(*) c FROM hypotheses WHERE status IN "
            "('survived_initial','survived_oos','survived_adversarial')"
        )["c"]
        survived_oos = self.query_one(
            "SELECT COUNT(*) c FROM hypotheses WHERE status IN "
            "('survived_oos','survived_adversarial')"
        )["c"]
        survived_adversarial = self.query_one(
            "SELECT COUNT(*) c FROM hypotheses WHERE status = 'survived_adversarial'"
        )["c"]
        rejected = self.query_one(
            "SELECT COUNT(*) c FROM hypotheses WHERE status = 'rejected'"
        )["c"]
        return {
            "hypotheses_tested": total,
            "survived_initial_testing": survived_initial,
            "survived_oos_testing": survived_oos,
            "survived_adversarial_validation": survived_adversarial,
            "rejected": rejected,
        }


_db_instance: Optional[Database] = None


def get_db() -> Database:
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance
