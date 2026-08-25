"""Cross-chat relay: person registry and structured fact store.

LOCAL-ONLY — NOT FOR UPSTREAM. This feature is not intended as a PR to
NousResearch/hermes-agent; it stays in this fork/install only. Its
no-privacy-filter default (anything tagged as relevant gets relayed, no
topic gate) is a deliberate personal policy choice, not a general-audience
default. See the spec (``hermes-agent-cross-chat-relay``) for the full
decision log.

Lets Alfred connect something said in one person's chat to a relevant thing
said in another person's chat and proactively relay it — e.g. two family
members independently mentioning the same errand. See the spec for the full
design: ``hermes-agent-cross-chat-relay`` in the project's spec tree.

Deliberately isolated from the main ``state.db`` in its own ``relay.db``,
mirroring how ``kanban_db.py`` keeps kanban state separate — smaller blast
radius, no interaction with the heavily-churned core session/message schema.

Architecture (see the spec for the full rationale):
- ``known_people`` / ``person_channels``: a general person registry, mapping
  each person to their chat identity on one or more platforms.
- ``relay_facts`` / ``relay_fact_targets``: one row per noteworthy fact, one
  row per person it's tagged for — modeled directly on
  ``kanban_notify_subs``'s subscription-row shape (one row per interested
  party, not a JSON blob of targets).
- Delivery is a separate background watcher (``gateway/relay_watchers.py``),
  not this module — this module only owns the data. The agent never gets a
  direct send-to-third-party tool; see ``toolsets.py``'s documented
  agents-do-not-get-send-message design constraint, which this feature
  deliberately preserves rather than reverses.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import NamedTuple, Optional

from hermes_constants import get_hermes_home

_SCHEMA = """
CREATE TABLE IF NOT EXISTS known_people (
    person_id     TEXT PRIMARY KEY,
    display_name  TEXT NOT NULL,
    created_at    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS person_channels (
    platform      TEXT NOT NULL,
    chat_id       TEXT NOT NULL,
    person_id     TEXT NOT NULL REFERENCES known_people(person_id),
    PRIMARY KEY (platform, chat_id)
);

CREATE TABLE IF NOT EXISTS relay_facts (
    fact_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source_person_id  TEXT REFERENCES known_people(person_id),
    source_platform   TEXT NOT NULL,
    source_chat_id    TEXT NOT NULL,
    created_at        INTEGER NOT NULL,
    summary           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS relay_fact_targets (
    fact_id           INTEGER NOT NULL REFERENCES relay_facts(fact_id),
    target_person_id  TEXT NOT NULL REFERENCES known_people(person_id),
    delivered_at      INTEGER,
    PRIMARY KEY (fact_id, target_person_id)
);

CREATE INDEX IF NOT EXISTS idx_relay_fact_targets_pending
    ON relay_fact_targets(delivered_at)
    WHERE delivered_at IS NULL;
"""


def relay_db_path() -> Path:
    """Return the path to ``relay.db``. No board-scoping — one relay store
    per Hermes home, unlike kanban's per-board databases, since a person
    registry is inherently a single cross-cutting concern, not something a
    user would want partitioned."""
    return get_hermes_home() / "relay.db"


def connect(path: Optional[Path] = None) -> sqlite3.Connection:
    """Open a relay-store connection, creating the schema if needed.

    Uses ``connect_tracked`` so the live-connection registry knows this file
    is open, same rationale as ``kanban_db.py``'s ``_sqlite_connect``: an
    untracked ``open()``/``close()`` elsewhere could cancel this process's
    POSIX advisory locks on the database.
    """
    from hermes_cli.sqlite_safe_read import connect_tracked

    db_path = path or relay_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect_tracked(db_path, connect_fn=sqlite3.connect, isolation_level=None)
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    return conn


def register_person(conn: sqlite3.Connection, person_id: str, display_name: str) -> None:
    """Register a person, or update their display name if already known.

    ``person_id`` is a caller-chosen stable slug (e.g. "roshani"), not
    auto-generated — the registry is small and human-curated, so a
    predictable id is more useful than a surrogate one.
    """
    conn.execute(
        """INSERT INTO known_people (person_id, display_name, created_at)
           VALUES (?, ?, ?)
           ON CONFLICT(person_id) DO UPDATE SET display_name = excluded.display_name""",
        (person_id, display_name, int(time.time())),
    )


def link_channel(conn: sqlite3.Connection, platform: str, chat_id: str, person_id: str) -> None:
    """Associate a (platform, chat_id) with a registered person.

    A chat_id belongs to exactly one person (primary key on platform+chat_id);
    a person may have channels on multiple platforms.
    """
    conn.execute(
        """INSERT INTO person_channels (platform, chat_id, person_id)
           VALUES (?, ?, ?)
           ON CONFLICT(platform, chat_id) DO UPDATE SET person_id = excluded.person_id""",
        (platform, chat_id, person_id),
    )


def resolve_person_by_name(conn: sqlite3.Connection, name: str) -> Optional[str]:
    """Resolve a display name to a person_id, case-insensitively.

    Exact match only, deliberately — no fuzzy matching. A wrong fuzzy match
    here means relaying a private fact to the wrong person, which is a much
    worse failure than the agent simply not finding a match (see the spec's
    "unresolved-name handling" open question).
    """
    row = conn.execute(
        "SELECT person_id FROM known_people WHERE lower(display_name) = lower(?)",
        (name,),
    ).fetchone()
    return row[0] if row else None


def resolve_channel(conn: sqlite3.Connection, person_id: str, platform: str) -> Optional[str]:
    """Return the chat_id for a person on a given platform, if known."""
    row = conn.execute(
        "SELECT chat_id FROM person_channels WHERE person_id = ? AND platform = ?",
        (person_id, platform),
    ).fetchone()
    return row[0] if row else None


def record_relay_fact(
    conn: sqlite3.Connection,
    *,
    source_person_id: Optional[str],
    source_platform: str,
    source_chat_id: str,
    summary: str,
    target_person_ids: list[str],
) -> int:
    """Record a relay-worthy fact and its tagged targets in one transaction.

    ``target_person_ids`` should already be resolved (via
    ``resolve_person_by_name``) by the caller — this function does not
    resolve names, so an unresolvable name never silently becomes a
    dangling/garbage target row.
    """
    conn.execute("BEGIN")
    try:
        cursor = conn.execute(
            """INSERT INTO relay_facts
               (source_person_id, source_platform, source_chat_id, created_at, summary)
               VALUES (?, ?, ?, ?, ?)""",
            (source_person_id, source_platform, source_chat_id, int(time.time()), summary),
        )
        fact_id = cursor.lastrowid
        conn.executemany(
            """INSERT INTO relay_fact_targets (fact_id, target_person_id)
               VALUES (?, ?)""",
            [(fact_id, target_id) for target_id in target_person_ids],
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return fact_id


class PendingRelay(NamedTuple):
    fact_id: int
    target_person_id: str
    summary: str
    source_person_id: Optional[str]


def pending_relay_targets(conn: sqlite3.Connection, limit: int = 100) -> list[PendingRelay]:
    """Return undelivered (fact, target) pairs, oldest first.

    Used by the delivery watcher (``gateway/relay_watchers.py``) each poll
    tick — deliberately does not resolve the target's chat_id here, since
    that's a per-platform concern the watcher handles (a target could in
    principle have channels on multiple platforms; picking which one is the
    watcher's job, not this query's).
    """
    rows = conn.execute(
        """SELECT t.fact_id, t.target_person_id, f.summary, f.source_person_id
           FROM relay_fact_targets t
           JOIN relay_facts f ON f.fact_id = t.fact_id
           WHERE t.delivered_at IS NULL
           ORDER BY f.created_at ASC
           LIMIT ?""",
        (limit,),
    ).fetchall()
    return [PendingRelay(*row) for row in rows]


def mark_delivered(conn: sqlite3.Connection, fact_id: int, target_person_id: str) -> None:
    conn.execute(
        """UPDATE relay_fact_targets SET delivered_at = ?
           WHERE fact_id = ? AND target_person_id = ?""",
        (int(time.time()), fact_id, target_person_id),
    )
