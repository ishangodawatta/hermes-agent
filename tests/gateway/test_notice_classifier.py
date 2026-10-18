"""Tests for the notice/classifier pass (gateway/notice_classifier.py).

LOCAL-ONLY feature — see the hermes-agent-alfred-manager-architecture and
hermes-agent-person-registry-consolidation specs.
"""

from __future__ import annotations

import json

import pytest

from gateway import notice_classifier
from hermes_cli import relay_db

_KNOWN_CHAT_ID = "221139326480632@lid"
_KNOWN_PERSON = ("ishan", "Ishan")


@pytest.fixture
def relay_conn(tmp_path, monkeypatch):
    """Isolated relay.db for each test (activities/facts only — no local
    person registry to seed, identity comes from family-facts)."""
    db_path = tmp_path / "relay.db"
    monkeypatch.setattr(relay_db, "relay_db_path", lambda: db_path)
    return db_path


@pytest.fixture
def known_person(monkeypatch):
    """Stub family-facts resolution: _KNOWN_CHAT_ID -> _KNOWN_PERSON, nothing
    else known. Notice-classifier routing tests care about what happens
    given a resolved identity, not about family-facts' own lookup logic
    (covered separately in tests/hermes_cli/test_relay_db_family_facts.py)."""
    def _resolve(chat_id):
        return _KNOWN_PERSON if chat_id == _KNOWN_CHAT_ID else None

    monkeypatch.setattr(relay_db, "resolve_person_from_chat_id", _resolve)


def _fake_llm(payload: dict):
    def _run(prompt: str):
        return {"final": json.dumps(payload), "error": None}

    return _run


class TestParseNoticeJudgment:
    def test_valid_json(self):
        text = '{"is_errand": true, "errand_description": "x"}'
        assert notice_classifier._parse_notice_judgment(text) == {
            "is_errand": True,
            "errand_description": "x",
        }

    def test_json_with_surrounding_prose(self):
        text = 'Sure, here it is:\n{"is_errand": false}\nThanks.'
        assert notice_classifier._parse_notice_judgment(text) == {"is_errand": False}

    def test_garbage_returns_none(self):
        assert notice_classifier._parse_notice_judgment("not json at all") is None

    def test_empty_returns_none(self):
        assert notice_classifier._parse_notice_judgment("") is None


class TestRunNoticeClassifierTick:
    def test_empty_message_short_circuits(self, relay_conn, known_person, monkeypatch):
        called = []
        monkeypatch.setattr(notice_classifier, "_run_notice_classifier_llm", lambda p: called.append(p) or {"final": "", "error": None})
        notice_classifier.run_notice_classifier_tick(
            platform="whatsapp", chat_id=_KNOWN_CHAT_ID, message_text="   ",
        )
        assert called == []

    def test_reaction_marker_short_circuits(self, relay_conn, known_person, monkeypatch):
        # Regression test: a plain reaction was hallucinated into a
        # fabricated errand ("Ishan going to Tesco") before this guard --
        # see the alfred-manager-architecture spec's "Reaction handling".
        called = []
        monkeypatch.setattr(notice_classifier, "_run_notice_classifier_llm", lambda p: called.append(p) or {"final": "", "error": None})
        notice_classifier.run_notice_classifier_tick(
            platform="whatsapp", chat_id=_KNOWN_CHAT_ID,
            message_text="[Reaction: \U0001F44D to 3EB04F396F3F9B43BCD123]",
        )
        assert called == []

    def test_unlinked_chat_short_circuits(self, relay_conn, known_person, monkeypatch):
        called = []
        monkeypatch.setattr(notice_classifier, "_run_notice_classifier_llm", lambda p: called.append(p) or {"final": "", "error": None})
        notice_classifier.run_notice_classifier_tick(
            platform="whatsapp", chat_id="not-a-registered-chat", message_text="heading to Tesco",
        )
        assert called == []

    def test_llm_error_writes_nothing(self, relay_conn, known_person, monkeypatch):
        monkeypatch.setattr(
            notice_classifier, "_run_notice_classifier_llm",
            lambda p: {"final": "", "error": "boom"},
        )
        todoist_calls = []
        monkeypatch.setattr(notice_classifier, "_todoist_add", lambda *a, **kw: todoist_calls.append((a, kw)))
        notice_classifier.run_notice_classifier_tick(
            platform="whatsapp", chat_id=_KNOWN_CHAT_ID, message_text="heading to Tesco",
        )
        conn = relay_db.connect()
        try:
            assert relay_db.uncompared_activities(conn) == []
        finally:
            conn.close()
        assert todoist_calls == []

    def test_unparseable_judgment_writes_nothing(self, relay_conn, known_person, monkeypatch):
        monkeypatch.setattr(
            notice_classifier, "_run_notice_classifier_llm",
            lambda p: {"final": "not json", "error": None},
        )
        notice_classifier.run_notice_classifier_tick(
            platform="whatsapp", chat_id=_KNOWN_CHAT_ID, message_text="heading to Tesco",
        )
        conn = relay_db.connect()
        try:
            assert relay_db.uncompared_activities(conn) == []
        finally:
            conn.close()

    def test_errand_signal_logs_activity(self, relay_conn, known_person, monkeypatch):
        monkeypatch.setattr(
            notice_classifier, "_run_notice_classifier_llm",
            _fake_llm({
                "is_errand": True, "errand_description": "Ishan is heading to Tesco",
                "is_item_request": False, "item": "",
                "is_commitment": False, "commitment_description": "",
            }),
        )
        notice_classifier.run_notice_classifier_tick(
            platform="whatsapp", chat_id=_KNOWN_CHAT_ID, message_text="heading to Tesco",
        )
        conn = relay_db.connect()
        try:
            activities = relay_db.uncompared_activities(conn)
        finally:
            conn.close()
        assert len(activities) == 1
        assert activities[0].person_id == "ishan"
        assert activities[0].description == "Ishan is heading to Tesco"
        assert activities[0].platform == "whatsapp"
        assert activities[0].chat_id == _KNOWN_CHAT_ID

    def test_item_request_signal_calls_todoist_with_shopping_and_person_label(self, relay_conn, known_person, monkeypatch):
        monkeypatch.setattr(
            notice_classifier, "_run_notice_classifier_llm",
            _fake_llm({
                "is_errand": False, "errand_description": "",
                "is_item_request": True, "item": "toothpaste",
                "is_commitment": False, "commitment_description": "",
            }),
        )
        todoist_calls = []
        monkeypatch.setattr(notice_classifier, "_todoist_add", lambda content, labels: todoist_calls.append((content, labels)))
        notice_classifier.run_notice_classifier_tick(
            platform="whatsapp", chat_id=_KNOWN_CHAT_ID, message_text="can someone grab toothpaste",
        )
        assert todoist_calls == [("toothpaste", ["shopping", "task", "ishan"])]

    def test_commitment_signal_calls_todoist_with_person_label_only(self, relay_conn, known_person, monkeypatch):
        monkeypatch.setattr(
            notice_classifier, "_run_notice_classifier_llm",
            _fake_llm({
                "is_errand": False, "errand_description": "",
                "is_item_request": False, "item": "",
                "is_commitment": True, "commitment_description": "Ishan will call the dentist tomorrow",
            }),
        )
        todoist_calls = []
        monkeypatch.setattr(notice_classifier, "_todoist_add", lambda content, labels: todoist_calls.append((content, labels)))
        notice_classifier.run_notice_classifier_tick(
            platform="whatsapp", chat_id=_KNOWN_CHAT_ID, message_text="I'll call the dentist tomorrow",
        )
        assert todoist_calls == [("Ishan will call the dentist tomorrow", ["task", "ishan"])]

    def test_multiple_signals_all_route(self, relay_conn, known_person, monkeypatch):
        monkeypatch.setattr(
            notice_classifier, "_run_notice_classifier_llm",
            _fake_llm({
                "is_errand": True, "errand_description": "Ishan is heading to Tesco",
                "is_item_request": True, "item": "bread",
                "is_commitment": False, "commitment_description": "",
            }),
        )
        todoist_calls = []
        monkeypatch.setattr(notice_classifier, "_todoist_add", lambda content, labels: todoist_calls.append((content, labels)))
        notice_classifier.run_notice_classifier_tick(
            platform="whatsapp", chat_id=_KNOWN_CHAT_ID,
            message_text="heading to Tesco, grab bread while I'm there",
        )
        conn = relay_db.connect()
        try:
            activities = relay_db.uncompared_activities(conn)
        finally:
            conn.close()
        assert len(activities) == 1
        assert todoist_calls == [("bread", ["shopping", "task", "ishan"])]

    def test_no_signals_writes_nothing(self, relay_conn, known_person, monkeypatch):
        monkeypatch.setattr(
            notice_classifier, "_run_notice_classifier_llm",
            _fake_llm({
                "is_errand": False, "errand_description": "",
                "is_item_request": False, "item": "",
                "is_commitment": False, "commitment_description": "",
            }),
        )
        todoist_calls = []
        monkeypatch.setattr(notice_classifier, "_todoist_add", lambda content, labels: todoist_calls.append((content, labels)))
        notice_classifier.run_notice_classifier_tick(
            platform="whatsapp", chat_id=_KNOWN_CHAT_ID, message_text="haha nice one",
        )
        conn = relay_db.connect()
        try:
            assert relay_db.uncompared_activities(conn) == []
        finally:
            conn.close()
        assert todoist_calls == []
