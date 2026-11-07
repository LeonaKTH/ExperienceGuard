"""Append-only SQLite source of truth with a recoverable JSONL outbox."""

from __future__ import annotations
import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from filelock import FileLock
from ..types import Experience
from ..utils.core import canonical, digest


class ExperienceStore:
    def __init__(self, path: str | Path, record_type=Experience):
        self.record_type = record_type
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.journal = self.path.with_suffix(".jsonl")
        with self.connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS experiences(id TEXT PRIMARY KEY, payload TEXT NOT NULL, sha TEXT NOT NULL);
            CREATE TRIGGER IF NOT EXISTS immutable_update BEFORE UPDATE ON experiences
            BEGIN SELECT RAISE(ABORT, 'experiences are immutable'); END;
            CREATE TRIGGER IF NOT EXISTS immutable_delete BEFORE DELETE ON experiences
            BEGIN SELECT RAISE(ABORT, 'experiences are immutable'); END;
            CREATE TABLE IF NOT EXISTS outbox(id TEXT PRIMARY KEY REFERENCES experiences(id), exported INTEGER NOT NULL DEFAULT 0);
            """)
        self.flush()

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=20)
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def add(self, experience: Experience):
        payload, sha = canonical(experience.to_dict()), experience.content_hash
        with self.connect() as db:
            old = db.execute(
                "SELECT sha FROM experiences WHERE id=?", (experience.experience_id,)
            ).fetchone()
            if old:
                if old[0] != sha:
                    raise ValueError("refusing to overwrite an existing experience")
                return sha
            db.execute(
                "INSERT INTO experiences VALUES(?,?,?)",
                (experience.experience_id, payload, sha),
            )
            db.execute("INSERT INTO outbox(id) VALUES(?)", (experience.experience_id,))
        self.flush()
        return sha

    def flush(self):
        with FileLock(str(self.journal) + ".lock"):
            seen = set()
            if self.journal.exists():
                # Detect interrupted/corrupted writes rather than silently discarding evidence.
                for line in self.journal.read_text(encoding="utf-8").splitlines():
                    obj = json.loads(line)
                    if digest(obj["experience"]) != obj["sha256"]:
                        raise ValueError("journal hash mismatch")
                    seen.add((obj["experience"]["experience_id"], obj["sha256"]))
            with self.connect() as db:
                rows = db.execute(
                    "SELECT e.id,e.payload,e.sha FROM experiences e JOIN outbox o ON e.id=o.id WHERE o.exported=0 ORDER BY e.id"
                ).fetchall()
                with self.journal.open("a", encoding="utf-8", newline="\n") as stream:
                    for eid, payload, sha in rows:
                        if (eid, sha) not in seen:
                            stream.write(
                                canonical(
                                    {"experience": json.loads(payload), "sha256": sha}
                                )
                                + "\n"
                            )
                            stream.flush()
                            os.fsync(stream.fileno())
                        db.execute("UPDATE outbox SET exported=1 WHERE id=?", (eid,))

    def get(self, eid):
        with self.connect() as db:
            row = db.execute(
                "SELECT payload,sha FROM experiences WHERE id=?", (eid,)
            ).fetchone()
        if not row:
            raise KeyError(eid)
        value = json.loads(row[0])
        if digest(value) != row[1]:
            raise ValueError("stored experience hash mismatch")
        return self.record_type(**value)

    def all(self):
        with self.connect() as db:
            ids = [r[0] for r in db.execute("SELECT id FROM experiences ORDER BY id")]
        return [self.get(eid) for eid in ids]
