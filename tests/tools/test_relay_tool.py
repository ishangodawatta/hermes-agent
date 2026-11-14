"""Tests for tools/relay_tool.py's relay_note (person-registry-consolidation spec).

LOCAL-ONLY feature. Identity resolution goes through family-facts (mocked
here) rather than a local person registry.
"""

from __future__ import annotations

import pytest

from hermes_cli import relay_db
from tools import relay_tool

_SOURCE_CHAT_ID = "221139326480632@lid"
_SOURCE_PERSON = ("ishan", "Ishan")


@pytest.fixture
def relay_conn(tmp_path, monkeypatch):
    db_path = tmp_path / "relay.db"
    monkeypatch.setattr(relay_db, "relay_db_path", lambda: db_path)
    return db_path


@pytest.fixture
def fake_session(monkeypatch):
    class _FakeSessionDB:
        def __init__(self, read_only=True):
            pass

        def get_session(self, session_id):
            if session_id != "sess-1":
                return None
            return {"source": "whatsapp", "chat_id": _SOURCE_CHAT_ID}

    monkeypatch.setattr("hermes_state.SessionDB", _FakeSessionDB)


@pytest.fixture
def family_facts_stub(monkeypatch):
    """Ishan (source) resolves; Ananya resolves by name; anyone else doesn't."""
    def _resolve_from_chat_id(chat_id):
        return _SOURCE_PERSON if chat_id == _SOURCE_CHAT_ID else None

    def _resolve_by_name(name):
        return {"ishan": "ishan", "ananya": "ananya"}.get(name.strip().lower())

    monkeypatch.setattr(relay_db, "resolve_person_from_chat_id", _resolve_from_chat_id)
    monkeypatch.setattr(relay_db, "resolve_person_by_name", _resolve_by_name)


class TestRelayNote:
    def test_queues_fact_for_resolved_target(self, relay_conn, fake_session, family_facts_stub):
        result = relay_tool.relay_note(
            {"summary": "Ishan is heading to Tesco", "interested": ["Ananya"]},
            session_id="sess-1",
        )
        assert "Queued for relay to 1 recipient(s)" in result

        conn = relay_db.connect()
        try:
            fact = conn.execute(
                "SELECT source_person_id, summary FROM relay_facts"
            ).fetchone()
            target = conn.execute(
                "SELECT target_person_id FROM relay_fact_targets"
            ).fetchone()
        finally:
            conn.close()
        assert fact == ("ishan", "Ishan is heading to Tesco")
        assert target == ("ananya",)

    def test_unresolved_target_name_not_queued(self, relay_conn, fake_session, family_facts_stub):
        result = relay_tool.relay_note(
            {"summary": "test", "interested": ["Nobody"]},
            session_id="sess-1",
        )
        assert "Not queued" in result
        assert "Nobody" in result

    def test_missing_summary_errors(self, relay_conn, fake_session, family_facts_stub):
        result = relay_tool.relay_note({"summary": "", "interested": ["Ananya"]}, session_id="sess-1")
        assert result.startswith("Error:")

    def test_missing_interested_errors(self, relay_conn, fake_session, family_facts_stub):
        result = relay_tool.relay_note({"summary": "x", "interested": []}, session_id="sess-1")
        assert result.startswith("Error:")

    def test_unresolved_source_still_queues_with_null_source_person(
        self, relay_conn, fake_session, monkeypatch
    ):
        monkeypatch.setattr(relay_db, "resolve_person_from_chat_id", lambda chat_id: None)
        monkeypatch.setattr(relay_db, "resolve_person_by_name", lambda name: "ananya" if name.lower() == "ananya" else None)
        result = relay_tool.relay_note(
            {"summary": "test", "interested": ["Ananya"]}, session_id="sess-1",
        )
        assert "Queued for relay to 1 recipient(s)" in result
        conn = relay_db.connect()
        try:
            fact = conn.execute("SELECT source_person_id FROM relay_facts").fetchone()
        finally:
            conn.close()
        assert fact == (None,)
