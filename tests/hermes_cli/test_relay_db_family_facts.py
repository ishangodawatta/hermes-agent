"""Tests for relay_db's family-facts identity adapter (person-registry-consolidation spec).

LOCAL-ONLY feature. family-facts is the sole identity store for relay
features — relay.db keeps no person/channel tables of its own; every lookup
goes live, read-only, to family-facts' family.db.
"""

from __future__ import annotations

import sqlite3

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
        "INSERT INTO people VALUES ('ammi', 'Roshani', NULL, 'Europe/London', ?, ?)", (now, now)
    )
    conn.execute("INSERT INTO identifiers VALUES ('107339940163598', 'ammi', 'whatsapp', ?)", (now,))
    conn.execute("INSERT INTO identifiers VALUES ('94777421980', 'ammi', 'whatsapp', ?)", (now,))
    conn.commit()
    conn.close()
    monkeypatch.setenv("FAMILY_FACTS_DB", str(db_path))
    return db_path


@pytest.fixture
def no_family_facts_db(monkeypatch):
    monkeypatch.delenv("FAMILY_FACTS_DB", raising=False)
    monkeypatch.setenv("HERMES_HOME", "/nonexistent")


class TestResolvePersonFromChatId:
    def test_resolves_current_lid(self, family_facts_db):
        assert relay_db.resolve_person_from_chat_id("107339940163598@lid") == ("ammi", "Roshani")

    def test_resolves_old_phone_identifier_too(self, family_facts_db):
        assert relay_db.resolve_person_from_chat_id("94777421980@lid") == ("ammi", "Roshani")

    def test_unknown_chat_id_returns_none(self, family_facts_db):
        assert relay_db.resolve_person_from_chat_id("999999@lid") is None

    def test_returns_none_when_family_facts_missing(self, no_family_facts_db):
        assert relay_db.resolve_person_from_chat_id("107339940163598@lid") is None

    def test_non_numeric_chat_id_returns_none(self, family_facts_db):
        assert relay_db.resolve_person_from_chat_id("") is None


class TestResolvePersonByName:
    def test_case_insensitive_exact_match(self, family_facts_db):
        assert relay_db.resolve_person_by_name("roshani") == "ammi"
        assert relay_db.resolve_person_by_name("Roshani") == "ammi"

    def test_unknown_name_returns_none(self, family_facts_db):
        assert relay_db.resolve_person_by_name("Nobody") is None

    def test_returns_none_when_family_facts_missing(self, no_family_facts_db):
        assert relay_db.resolve_person_by_name("Roshani") is None


class TestDisplayNameForPerson:
    def test_known_person(self, family_facts_db):
        assert relay_db.display_name_for_person("ammi") == "Roshani"

    def test_unknown_person_falls_back_to_raw_id(self, family_facts_db):
        assert relay_db.display_name_for_person("nobody") == "nobody"

    def test_missing_family_facts_falls_back_to_raw_id(self, no_family_facts_db):
        assert relay_db.display_name_for_person("ammi") == "ammi"


class TestResolveChatIdForPerson:
    def test_prefers_longest_identifier(self, family_facts_db):
        # 107339940163598 (15 digits) vs 94777421980 (11 digits) on file for ammi.
        assert relay_db.resolve_chat_id_for_person("ammi") == "107339940163598@lid"

    def test_unknown_person_returns_none(self, family_facts_db):
        assert relay_db.resolve_chat_id_for_person("nobody") is None

    def test_wrong_kind_returns_none(self, family_facts_db):
        assert relay_db.resolve_chat_id_for_person("ammi", platform="sms") is None


class TestConnectDropsRetiredTables:
    def test_known_people_and_person_channels_are_gone(self, tmp_path, monkeypatch):
        db_path = tmp_path / "relay.db"
        monkeypatch.setattr(relay_db, "relay_db_path", lambda: db_path)
        # Simulate a pre-migration database that still has the old tables.
        conn = relay_db.connect()
        conn.executescript(
            "CREATE TABLE IF NOT EXISTS known_people (person_id TEXT PRIMARY KEY);"
            "CREATE TABLE IF NOT EXISTS person_channels (platform TEXT, chat_id TEXT);"
        )
        conn.close()

        conn = relay_db.connect()
        try:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        finally:
            conn.close()
        assert "known_people" not in tables
        assert "person_channels" not in tables
        assert "relay_activities" in tables
