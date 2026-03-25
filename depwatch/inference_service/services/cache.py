"""SQLite-backed snapshot cache for computed feature vectors.

Caches feature snapshots keyed by (owner, repo, month) so repeat queries
for the same package are instant. Cache survives server restarts.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from depwatch.common.features import FeatureVector

logger = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = "data/cache/snapshots.db"


class SnapshotCache:
    """On-disk SQLite cache for computed feature snapshots.

    Thread-safe via SQLite's built-in locking. Intended as a simple
    key-value store — no complex queries.
    """

    def __init__(self, db_path: str = DEFAULT_CACHE_PATH) -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS snapshots (
                owner TEXT NOT NULL,
                repo TEXT NOT NULL,
                month TEXT NOT NULL,
                features TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (owner, repo, month)
            )
            """
        )
        self._conn.commit()

    def get(self, owner: str, repo: str, month: str) -> FeatureVector | None:
        """Look up a cached feature snapshot.

        Args:
            owner: GitHub owner (e.g. 'pallets').
            repo: GitHub repo name (e.g. 'flask').
            month: Year-month string (e.g. '2025-03').

        Returns:
            Cached FeatureVector or None if not found.
        """
        cursor = self._conn.execute(
            "SELECT features FROM snapshots WHERE owner = ? AND repo = ? AND month = ?",
            (owner.lower(), repo.lower(), month),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        data: dict[str, float] = json.loads(row[0])
        return FeatureVector(**data)

    def put(self, owner: str, repo: str, month: str, features: FeatureVector) -> None:
        """Store a feature snapshot in the cache.

        Overwrites any existing entry for the same key.
        """
        features_json = json.dumps(
            {name: getattr(features, name) for name in FeatureVector.model_fields},
        )
        self._conn.execute(
            """
            INSERT OR REPLACE INTO snapshots (owner, repo, month, features)
            VALUES (?, ?, ?, ?)
            """,
            (owner.lower(), repo.lower(), month, features_json),
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
