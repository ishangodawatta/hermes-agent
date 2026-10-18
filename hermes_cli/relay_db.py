"""Cross-chat relay: structured fact store, backed by family-facts for identity.

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

Architecture (see the ``hermes-agent-person-registry-consolidation`` spec for
the full rationale and decision log):
- **No local person registry.** This module used to keep its own
  ``known_people``/``person_channels`` tables, duplicating identity data
  that ``family-facts`` (``~/.hermes/skills/family/family-facts``) already
  owns canonically. That duplication meant registering a new family member
  required two separate manual steps and could silently drift out of sync
  (confirmed 2026-08-26: Senara had to be found and registered by hand in
  relay.db despite family-facts already knowing her). Removed 2026-08-26 —
  every person lookup now goes live, read-only, to family-facts' own
  ``family.db`` on every call. relay.db stores family-facts' own
  ``person_id`` values directly (e.g. ``"ammi"``, ``"nangi"``) on its own
  rows — not a second, locally-invented id space.
- ``relay_facts`` / ``relay_fact_targets``: one row per noteworthy fact, one
  row per person it's tagged for — modeled directly on
  ``kanban_notify_subs``'s subscription-row shape (one row per interested
  party, not a JSON blob of targets).
- ``relay_activities``: activities logged with no known target yet, for the
  background scanner's cross-person matching.
- Delivery is a separate background watcher (``gateway/relay_watchers.py``),
  not this module — this module only owns the data. The agent never gets a
  direct send-to-third-party tool; see ``toolsets.py``'s documented
  agents-do-not-get-send-message design constraint, which this feature
  deliberately preserves rather than reverses.
"""

from __future__ import annotations

import os
import re
import sqlite3
import time
from pathlib import Path
from typing import NamedTuple, Optional

from hermes_constants import get_hermes_home

_SCHEMA = """
CREATE TABLE IF NOT EXISTS relay_facts (
    fact_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source_person_id  TEXT,
    source_platform   TEXT NOT NULL,
    source_chat_id    TEXT NOT NULL,
    created_at        INTEGER NOT NULL,
    summary           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS relay_fact_targets (
    fact_id           INTEGER NOT NULL REFERENCES relay_facts(fact_id),
    target_person_id  TEXT NOT NULL,
    delivered_at      INTEGER,
    PRIMARY KEY (fact_id, target_person_id)
);

CREATE INDEX IF NOT EXISTS idx_relay_fact_targets_pending
    ON relay_fact_targets(delivered_at)
    WHERE delivered_at IS NULL;

-- Logged activities feed the background scanner's cross-person matching
-- (gateway/relay_watchers.py's _relay_scanner_watcher). Distinct from
-- relay_facts: an activity is logged with no known target yet ("heading to
-- Tesco" — nobody obviously needs telling right now); relay_facts/targets
-- are only created once the scanner finds an actual match against another
-- person's activity. Deliberately NOT the calendar — see the spec's
-- "same-person calendar-conflict reminders" section for why calendar
-- integration is a separate, prompt-level concern, not this table.
--
-- person_id here is family-facts' own person_id (see module docstring) —
-- no local FK, since there's no local people table to reference anymore.
CREATE TABLE IF NOT EXISTS relay_activities (
    activity_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id        TEXT NOT NULL,
    platform         TEXT NOT NULL,
    chat_id          TEXT NOT NULL,
    description      TEXT NOT NULL,
    occurred_at      INTEGER NOT NULL,
    compared_at      INTEGER,
    matched_fact_id  INTEGER REFERENCES relay_facts(fact_id)
);

CREATE INDEX IF NOT EXISTS idx_relay_activities_uncompared
    ON relay_activities(compared_at)
    WHERE compared_at IS NULL;
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

    Drops the retired ``known_people``/``person_channels`` tables on every
    connect (a no-op once already dropped) — cheap, and keeps an existing
    live database tidy rather than leaving dead tables around indefinitely.
    Safe unconditionally: this module never enabled FK enforcement
    (``PRAGMA foreign_keys`` is never set), so nothing else in this schema
    was ever actually constrained by their presence.
    """
    from hermes_cli.sqlite_safe_read import connect_tracked

    db_path = path or relay_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect_tracked(db_path, connect_fn=sqlite3.connect, isolation_level=None)
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    conn.execute("DROP TABLE IF EXISTS person_channels")
    conn.execute("DROP TABLE IF EXISTS known_people")
    return conn


# ---------------------------------------------------------------------------
# family-facts adapter — the sole identity source. Read-only, cross-file:
# family-facts is a skill script (~/.hermes/skills/family/family-facts), not
# part of the hermes-agent package, so it can't be imported directly — these
# functions read its known schema (people/identifiers tables) via a
# read-only connection instead. Every function here degrades to None/a
# fallback rather than raising when family-facts is unreachable (e.g. its
# db doesn't exist yet) — relay features simply treat that person as
# unregistered, the same as before this module ever had its own tables.
# ---------------------------------------------------------------------------


def _family_facts_db_path() -> Optional[Path]:
    """Resolve family-facts' family.db, mirroring its own db_path() logic."""
    explicit = os.environ.get("FAMILY_FACTS_DB")
    path = Path(explicit).expanduser() if explicit else get_hermes_home() / "family" / "family.db"
    return path if path.is_file() else None


def _family_facts_readonly_connect() -> Optional[sqlite3.Connection]:
    db_path = _family_facts_db_path()
    if not db_path:
        return None
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def resolve_person_from_chat_id(chat_id: str) -> Optional[tuple[str, str]]:
    """Resolve a WhatsApp chat_id to (person_id, display_name) via
    family-facts — the sole identity store; relay.db keeps no cache of its
    own, so every lookup reflects family-facts' current state immediately.
    """
    digits = re.sub(r"\D", "", chat_id)
    if not digits:
        return None
    conn = _family_facts_readonly_connect()
    if not conn:
        return None
    try:
        row = conn.execute(
            "SELECT p.person_id, p.display_name FROM identifiers i "
            "JOIN people p ON p.person_id = i.person_id WHERE i.identifier = ?",
            (digits,),
        ).fetchone()
        return (row[0], row[1]) if row else None
    finally:
        conn.close()


def resolve_person_by_name(name: str) -> Optional[str]:
    """Resolve a display name to family-facts' person_id, case-insensitively.

    Exact match only, deliberately — no fuzzy matching. A wrong fuzzy match
    here means relaying a private fact to the wrong person, which is a much
    worse failure than the agent simply not finding a match (see the spec's
    "unresolved-name handling" open question).
    """
    conn = _family_facts_readonly_connect()
    if not conn:
        return None
    try:
        row = conn.execute(
            "SELECT person_id FROM people WHERE lower(display_name) = lower(?)",
            (name,),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def display_name_for_person(person_id: str) -> str:
    """Best-effort display name for a family-facts person_id.

    Falls back to the raw id if family-facts is unreachable or doesn't know
    it — callers always get a printable string back rather than handling
    None, matching the old known_people-backed behaviour's fallback shape.
    """
    conn = _family_facts_readonly_connect()
    if not conn:
        return person_id
    try:
        row = conn.execute(
            "SELECT display_name FROM people WHERE person_id = ?", (person_id,)
        ).fetchone()
        return row[0] if row else person_id
    finally:
        conn.close()


def resolve_chat_id_for_person(person_id: str, *, platform: str = "whatsapp") -> Optional[str]:
    """Resolve family-facts' person_id back to a deliverable chat_id.

    Prefers the longest identifier on file for that person+kind: every
    WhatsApp identifier this family currently has is the longer @lid-style
    numeric id, not the older, shorter phone number (confirmed 2026-08-26,
    see the person-registry-consolidation spec) — a pragmatic heuristic for
    a single-family install, not a general rule for other id formats.
    """
    conn = _family_facts_readonly_connect()
    if not conn:
        return None
    try:
        row = conn.execute(
            "SELECT identifier FROM identifiers WHERE person_id = ? AND kind = ? "
            "ORDER BY length(identifier) DESC LIMIT 1",
            (person_id, platform),
        ).fetchone()
        return f"{row[0]}@lid" if row else None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# relay_facts / relay_fact_targets
# ---------------------------------------------------------------------------


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


class RecentFact(NamedTuple):
    fact_id: int
    summary: str
    created_at: int


def most_recent_fact_between(
    conn: sqlite3.Connection, person_a: str, person_b: str, *, since: int
) -> Optional[RecentFact]:
    """Most recent relay_fact already exchanged between this pair of people
    (either direction — a fact sourced from A targeting B, or from B
    targeting A), within the given window.

    Feeds the scanner's duplicate-relay dedup check (see the
    cross-chat-relay spec's "Duplicate-relay dedup" section): before
    relaying a new match, check whether this pair was already told about
    something recently, and if so, judge whether the new match is the same
    underlying topic (suppress) or genuinely different (relay as normal).
    """
    row = conn.execute(
        """SELECT f.fact_id, f.summary, f.created_at
           FROM relay_facts f
           JOIN relay_fact_targets t ON t.fact_id = f.fact_id
           WHERE f.created_at >= ?
             AND (
               (f.source_person_id = ? AND t.target_person_id = ?)
               OR (f.source_person_id = ? AND t.target_person_id = ?)
             )
           ORDER BY f.created_at DESC
           LIMIT 1""",
        (since, person_a, person_b, person_b, person_a),
    ).fetchone()
    return RecentFact(*row) if row else None


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


# ---------------------------------------------------------------------------
# relay_activities
# ---------------------------------------------------------------------------


def log_activity(
    conn: sqlite3.Connection,
    *,
    person_id: str,
    platform: str,
    chat_id: str,
    description: str,
    occurred_at: Optional[int] = None,
) -> int:
    """Log an activity with no known target yet — the scanner watcher
    (gateway/relay_watchers.py) matches it against other people's activities
    later. Distinct from record_relay_fact, which is for the rare case the
    agent already knows who to tell.

    ``person_id`` must already be resolved (via resolve_person_from_chat_id)
    by the caller — this function does not do identity resolution itself.
    """
    cursor = conn.execute(
        """INSERT INTO relay_activities
           (person_id, platform, chat_id, description, occurred_at)
           VALUES (?, ?, ?, ?, ?)""",
        (person_id, platform, chat_id, description, occurred_at or int(time.time())),
    )
    return cursor.lastrowid


class LoggedActivity(NamedTuple):
    activity_id: int
    person_id: str
    description: str
    occurred_at: int
    platform: str
    chat_id: str


def uncompared_activities(conn: sqlite3.Connection, limit: int = 20) -> list[LoggedActivity]:
    """Activities the scanner hasn't evaluated yet, oldest first."""
    rows = conn.execute(
        """SELECT activity_id, person_id, description, occurred_at, platform, chat_id
           FROM relay_activities
           WHERE compared_at IS NULL
           ORDER BY occurred_at ASC
           LIMIT ?""",
        (limit,),
    ).fetchall()
    return [LoggedActivity(*row) for row in rows]


def candidate_activities(
    conn: sqlite3.Connection,
    *,
    exclude_person_id: str,
    since: int,
) -> list[LoggedActivity]:
    """Other people's activities since ``since`` (a unix timestamp) — the
    pool a pivot activity gets compared against. Includes already-compared
    activities: a no-match activity can still become matched later when a
    new related one arrives, so being "compared" doesn't exclude it from
    being a candidate for someone else's comparison."""
    rows = conn.execute(
        """SELECT activity_id, person_id, description, occurred_at, platform, chat_id
           FROM relay_activities
           WHERE person_id != ? AND occurred_at >= ?
           ORDER BY occurred_at ASC""",
        (exclude_person_id, since),
    ).fetchall()
    return [LoggedActivity(*row) for row in rows]


def mark_activity_compared(
    conn: sqlite3.Connection, activity_id: int, *, matched_fact_id: Optional[int] = None
) -> None:
    conn.execute(
        """UPDATE relay_activities SET compared_at = ?, matched_fact_id = ?
           WHERE activity_id = ?""",
        (int(time.time()), matched_fact_id, activity_id),
    )


def mark_activity_matched(conn: sqlite3.Connection, activity_id: int, fact_id: int) -> None:
    """Retroactively mark an already-compared candidate activity as matched,
    once a later pivot activity's comparison connects it to a new fact."""
    conn.execute(
        "UPDATE relay_activities SET matched_fact_id = ? WHERE activity_id = ?",
        (fact_id, activity_id),
    )
