"""Tests for relay_db's family-facts sync fallback (person-registry-consolidation spec).

LOCAL-ONLY feature. family-facts is the canonical person registry; relay.db
should auto-register anyone family-facts already knows rather than requiring
a second, manual registration step.
"""

from __future__ import annotations

import sqlite3
import time

import pytest

from hermes_cli import relay_db

_FAMILY_FACTS_SCHEMA = """
CREATE TABLE people (
    person_id    TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    calendar_id  TEXT,
    timezone     TEXT NOT NULL DEFAULT 'Europe/London',
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE TABLE identifiers (
    identifier TEXT PRIMARY KEY,
    person_id  TEXT NOT NULL REFERENCES people(person_id),
    kind       TEXT NOT NULL DEFAULT 'whatsapp',
    created_at TEXT NOT NULL
);
"""


@pytest.fixture
def family_facts_db(tmp_path, monkeypatch):
    db_path = tmp_path / "family.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(_FAMILY_FACTS_SCHEMA)
    now = "2026-08-26T00:00:00+00:00"
    conn.execute(
        "INSERT INTO people VALUES ('ammi', 'Roshani', NULL, 'Europe/London', ?, ?)",
        (now, now),
    )
    conn.execute(
        "INSERT INTO identifiers VALUES ('107339940163598', 'ammi', 'whatsapp', ?)", (now,)
    )
    conn.execute(
        "INSERT INTO identifiers VALUES ('94777421980', 'ammi', 'whatsapp', ?)", (now,)
    )
    conn.commit()
    conn.close()
    monkeypatch.setenv("FAMILY_FACTS_DB", str(db_path))
    return db_path


@pytest.fixture
def relay_conn(tmp_path, monkeypatch):
    db_path = tmp_path / "relay.db"
    monkeypatch.setattr(relay_db, "relay_db_path", lambda: db_path)
    return db_path


class TestFamilyFactsDbPath:
    def test_returns_none_when_no_env_and_no_default_file(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FAMILY_FACTS_DB", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "nonexistent_home"))
        assert relay_db._family_facts_db_path() is None

    def test_returns_path_when_env_points_at_real_file(self, family_facts_db):
        assert relay_db._family_facts_db_path() == family_facts_db


class TestSyncPersonFromFamilyFacts:
    def test_registers_and_returns_person_id_for_known_lid(self, family_facts_db, relay_conn):
        conn = relay_db.connect()
        try:
            person_id = relay_db.sync_person_from_family_facts(
                conn, "whatsapp", "107339940163598@lid"
            )
            assert person_id == "roshani"
            row = conn.execute(
                "SELECT person_id FROM person_channels WHERE platform='whatsapp' AND chat_id=?",
                ("107339940163598@lid",),
            ).fetchone()
            assert row == ("roshani",)
            name_row = conn.execute(
                "SELECT display_name FROM known_people WHERE person_id='roshani'"
            ).fetchone()
            assert name_row == ("Roshani",)
        finally:
            conn.close()

    def test_matches_old_phone_identifier_too(self, family_facts_db, relay_conn):
        conn = relay_db.connect()
        try:
            person_id = relay_db.sync_person_from_family_facts(conn, "whatsapp", "94777421980@lid")
            assert person_id == "roshani"
        finally:
            conn.close()

    def test_returns_none_for_unknown_chat_id(self, family_facts_db, relay_conn):
        conn = relay_db.connect()
        try:
            person_id = relay_db.sync_person_from_family_facts(conn, "whatsapp", "999999@lid")
            assert person_id is None
        finally:
            conn.close()

    def test_returns_none_when_family_facts_db_missing(self, relay_conn, monkeypatch):
        monkeypatch.delenv("FAMILY_FACTS_DB", raising=False)
        monkeypatch.setenv("HERMES_HOME", "/nonexistent")
        conn = relay_db.connect()
        try:
            person_id = relay_db.sync_person_from_family_facts(conn, "whatsapp", "107339940163598@lid")
            assert person_id is None
        finally:
            conn.close()


class TestResolvePersonByNameFallback:
    def test_falls_back_to_family_facts_by_display_name(self, family_facts_db, relay_conn):
        conn = relay_db.connect()
        try:
            person_id = relay_db.resolve_person_by_name(conn, "Roshani")
            assert person_id == "roshani"
            # Best-effort identifier reconstruction should have registered a channel too.
            row = conn.execute(
                "SELECT chat_id FROM person_channels WHERE person_id='roshani'"
            ).fetchone()
            assert row == ("107339940163598@lid",)
        finally:
            conn.close()

    def test_short_circuits_before_family_facts_when_already_known_locally(
        self, family_facts_db, relay_conn, monkeypatch
    ):
        called = []
        monkeypatch.setattr(
            relay_db, "sync_person_by_name_from_family_facts",
            lambda *a, **kw: called.append(True) or None,
        )
        conn = relay_db.connect()
        try:
            relay_db.register_person(conn, "roshani", "Roshani")
            person_id = relay_db.resolve_person_by_name(conn, "Roshani")
            assert person_id == "roshani"
            assert called == []  # fallback never invoked -- local match won
        finally:
            conn.close()

    def test_returns_none_for_unknown_name(self, family_facts_db, relay_conn):
        conn = relay_db.connect()
        try:
            assert relay_db.resolve_person_by_name(conn, "Nobody") is None
        finally:
            conn.close()
