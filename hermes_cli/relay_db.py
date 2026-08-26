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

import os
import re
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

-- Logged activities feed the background scanner's cross-person matching
-- (gateway/relay_watchers.py's _relay_scanner_watcher). Distinct from
-- relay_facts: an activity is logged with no known target yet ("heading to
-- Tesco" — nobody obviously needs telling right now); relay_facts/targets
-- are only created once the scanner finds an actual match against another
-- person's activity. Deliberately NOT the calendar — see the spec's
-- "same-person calendar-conflict reminders" section for why calendar
-- integration is a separate, prompt-level concern, not this table.
CREATE TABLE IF NOT EXISTS relay_activities (
    activity_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id        TEXT NOT NULL REFERENCES known_people(person_id),
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

    Falls back to family-facts (see sync_person_by_name_from_family_facts)
    when not already known locally — see the person-registry-consolidation
    spec: family-facts is the canonical registry, relay.db should not need a
    second manual registration for someone it already knows.
    """
    row = conn.execute(
        "SELECT person_id FROM known_people WHERE lower(display_name) = lower(?)",
        (name,),
    ).fetchone()
    if row:
        return row[0]
    return sync_person_by_name_from_family_facts(conn, name)


def _family_facts_db_path() -> Optional[Path]:
    """Resolve family-facts' family.db, mirroring its own db_path() logic.

    family-facts is a skill script (~/.hermes/skills/family/family-facts),
    not part of the hermes-agent package, so it can't be imported directly —
    this reads its known schema (people/identifiers tables) via a read-only
    connection instead.
    """
    explicit = os.environ.get("FAMILY_FACTS_DB")
    path = Path(explicit).expanduser() if explicit else get_hermes_home() / "family" / "family.db"
    return path if path.is_file() else None


def _family_facts_display_name_for_chat_id(chat_id: str) -> Optional[str]:
    """Resolve a chat_id's digits against family-facts' identifiers table."""
    digits = re.sub(r"\D", "", chat_id)
    if not digits:
        return None
    db_path = _family_facts_db_path()
    if not db_path:
        return None
    ff_conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = ff_conn.execute(
            "SELECT p.display_name FROM identifiers i "
            "JOIN people p ON p.person_id = i.person_id WHERE i.identifier = ?",
            (digits,),
        ).fetchone()
        return row[0] if row else None
    finally:
        ff_conn.close()


def sync_person_from_family_facts(
    conn: sqlite3.Connection, platform: str, chat_id: str
) -> Optional[str]:
    """Auto-register a person in relay.db from family-facts' canonical
    registry, keyed off a chat_id already seen live — avoids a second,
    manual registration step for anyone family-facts already knows (see the
    person-registry-consolidation spec: this was previously a fully manual,
    easy-to-forget step — Senara had to be found and registered by hand
    2026-08-26 despite family-facts already knowing her).

    relay.db's own person_id stays a lowercased-display-name slug (not
    family-facts' own person_id, which uses kinship-term slugs like "ammi")
    to avoid migrating existing relay_activities/relay_facts rows, which
    already use this convention.

    Returns the resolved person_id, or None if family-facts doesn't know
    this chat_id either (a genuinely new contact — needs a manual
    `person add` in family-facts first, not something relay.db can invent).
    """
    display_name = _family_facts_display_name_for_chat_id(chat_id)
    if not display_name:
        return None
    person_id = display_name.strip().lower()
    register_person(conn, person_id, display_name)
    link_channel(conn, platform, chat_id, person_id)
    return person_id


def sync_person_by_name_from_family_facts(conn: sqlite3.Connection, name: str) -> Optional[str]:
    """Fallback for resolve_person_by_name: check family-facts by display
    name and auto-register into relay.db (including a best-effort WhatsApp
    identifier) if found there but not yet locally known.

    Identifier reconstruction is best-effort: every WhatsApp identifier this
    family currently has on file is @lid-suffixed (confirmed 2026-08-26,
    see the person-registry-consolidation spec) — fine for a single-family
    install, not a general rule for other WhatsApp id formats.
    """
    db_path = _family_facts_db_path()
    if not db_path:
        return None
    ff_conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = ff_conn.execute(
            "SELECT p.display_name, "
            "(SELECT identifier FROM identifiers WHERE person_id = p.person_id "
            " AND kind = 'whatsapp' ORDER BY length(identifier) DESC LIMIT 1) "
            "FROM people p WHERE lower(p.display_name) = lower(?)",
            (name,),
        ).fetchone()
    finally:
        ff_conn.close()
    if not row:
        return None
    display_name, best_identifier = row
    person_id = display_name.strip().lower()
    register_person(conn, person_id, display_name)
    if best_identifier:
        link_channel(conn, "whatsapp", f"{best_identifier}@lid", person_id)
    return person_id


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
    agent already knows who to tell."""
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
