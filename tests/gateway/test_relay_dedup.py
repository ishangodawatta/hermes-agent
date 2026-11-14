"""Tests for the duplicate-relay dedup fix (cross-chat-relay spec's
"Duplicate-relay dedup" section).

LOCAL-ONLY feature. Covers relay_db.most_recent_fact_between() directly,
the dedup window config resolver, and the scanner's suppress-vs-relay
branch in isolation.
"""

from __future__ import annotations

import json
import time

import pytest

from gateway import relay_watchers
from hermes_cli import relay_db


@pytest.fixture
def relay_conn(tmp_path, monkeypatch):
    db_path = tmp_path / "relay.db"
    monkeypatch.setattr(relay_db, "relay_db_path", lambda: db_path)
    return db_path


class TestMostRecentFactBetween:
    def test_finds_fact_either_direction(self, relay_conn):
        conn = relay_db.connect()
        try:
            relay_db.record_relay_fact(
                conn, source_person_id="ishan", source_platform="whatsapp",
                source_chat_id="c1", summary="Ananya is heading to Tesco",
                target_person_ids=["ananya"],
            )
            found = relay_db.most_recent_fact_between(
                conn, "ananya", "ishan", since=int(time.time()) - 3600
            )
            assert found is not None
            assert found.summary == "Ananya is heading to Tesco"

            found_reverse = relay_db.most_recent_fact_between(
                conn, "ishan", "ananya", since=int(time.time()) - 3600
            )
            assert found_reverse is not None
        finally:
            conn.close()

    def test_returns_none_outside_window(self, relay_conn):
        conn = relay_db.connect()
        try:
            conn.execute(
                "INSERT INTO relay_facts (source_person_id, source_platform, "
                "source_chat_id, created_at, summary) VALUES (?, ?, ?, ?, ?)",
                ("ishan", "whatsapp", "c1", int(time.time()) - 10 * 3600, "old fact"),
            )
            fact_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.execute(
                "INSERT INTO relay_fact_targets (fact_id, target_person_id) VALUES (?, ?)",
                (fact_id, "ananya"),
            )
            found = relay_db.most_recent_fact_between(
                conn, "ishan", "ananya", since=int(time.time()) - 4 * 3600
            )
            assert found is None
        finally:
            conn.close()

    def test_returns_none_for_unrelated_pair(self, relay_conn):
        conn = relay_db.connect()
        try:
            relay_db.record_relay_fact(
                conn, source_person_id="ishan", source_platform="whatsapp",
                source_chat_id="c1", summary="x", target_person_ids=["ananya"],
            )
            found = relay_db.most_recent_fact_between(
                conn, "ishan", "roshani", since=int(time.time()) - 3600
            )
            assert found is None
        finally:
            conn.close()


class TestDedupWindowConfig:
    def test_default_is_four_hours(self):
        assert relay_watchers._get_dedup_window_seconds({}) == 4 * 3600

    def test_respects_override(self):
        cfg = {"relay": {"dedup_window_hours": 2}}
        assert relay_watchers._get_dedup_window_seconds(cfg) == 2 * 3600

    def test_invalid_value_falls_back_to_default(self):
        cfg = {"relay": {"dedup_window_hours": "not-a-number"}}
        assert relay_watchers._get_dedup_window_seconds(cfg) == 4 * 3600


class TestScanTickDedup:
    """Exercises _relay_scan_tick's suppress-vs-relay branch directly,
    stubbing out family-facts resolution and the LLM judgment calls."""

    def _setup_pair(self, conn, *, prior_summary: str, prior_age_seconds: int = 300):
        """Seed one already-relayed fact between ishan/ananya, then log a
        fresh uncompared activity for ishan that will match ananya's
        existing (already-compared) activity as its candidate."""
        now = int(time.time())
        relay_db.log_activity(
            conn, person_id="ananya", platform="whatsapp", chat_id="ananya_chat",
            description="Ananya is heading to Tesco", occurred_at=now - 600,
        )
        ananya_activity_id = conn.execute(
            "SELECT activity_id FROM relay_activities WHERE person_id='ananya'"
        ).fetchone()[0]
        conn.execute(
            "UPDATE relay_activities SET compared_at = ? WHERE activity_id = ?",
            (now - 590, ananya_activity_id),
        )
        relay_db.record_relay_fact(
            conn, source_person_id="ananya", source_platform="whatsapp",
            source_chat_id="ananya_chat", summary=prior_summary,
            target_person_ids=["ishan"],
        )
        relay_db.log_activity(
            conn, person_id="ishan", platform="whatsapp", chat_id="ishan_chat",
            description="Ishan is also going to Tesco", occurred_at=now,
        )

    @pytest.mark.asyncio
    async def test_suppresses_when_judged_same_topic(self, relay_conn, monkeypatch):
        conn = relay_db.connect()
        try:
            self._setup_pair(conn, prior_summary="Ananya headed to Tesco a bit ago.")
        finally:
            conn.close()

        monkeypatch.setattr(relay_db, "display_name_for_person", lambda pid: pid.capitalize())
        monkeypatch.setattr(
            relay_watchers, "_run_relay_llm_review",
            lambda prompt: (
                {"final": json.dumps({"same_topic": True}), "error": None}
                if "same_topic" in prompt
                else {
                    "final": json.dumps({
                        "match": True, "candidate_index": 0,
                        "summary_for_Ishan": "Ananya is also at Tesco.",
                        "summary_for_other": "Ishan is also at Tesco.",
                    }),
                    "error": None,
                }
            ),
        )
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly", lambda: {}, raising=False
        )

        runner = relay_watchers.GatewayRelayWatchersMixin()
        await runner._relay_scan_tick()

        conn = relay_db.connect()
        try:
            fact_count = conn.execute("SELECT COUNT(*) FROM relay_facts").fetchone()[0]
            assert fact_count == 1  # no new fact created -- suppressed
            ishan_activity = conn.execute(
                "SELECT compared_at, matched_fact_id FROM relay_activities WHERE person_id='ishan'"
            ).fetchone()
            assert ishan_activity[0] is not None
            assert ishan_activity[1] is not None  # marked against the existing fact
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_relays_when_judged_different_topic(self, relay_conn, monkeypatch):
        conn = relay_db.connect()
        try:
            self._setup_pair(conn, prior_summary="Ananya is picking up a birthday cake.")
        finally:
            conn.close()

        monkeypatch.setattr(relay_db, "display_name_for_person", lambda pid: pid.capitalize())
        monkeypatch.setattr(
            relay_watchers, "_run_relay_llm_review",
            lambda prompt: (
                {"final": json.dumps({"same_topic": False}), "error": None}
                if "same_topic" in prompt
                else {
                    "final": json.dumps({
                        "match": True, "candidate_index": 0,
                        "summary_for_Ishan": "Ananya is also at Tesco.",
                        "summary_for_other": "Ishan is also at Tesco.",
                    }),
                    "error": None,
                }
            ),
        )
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly", lambda: {}, raising=False
        )

        runner = relay_watchers.GatewayRelayWatchersMixin()
        await runner._relay_scan_tick()

        conn = relay_db.connect()
        try:
            fact_count = conn.execute("SELECT COUNT(*) FROM relay_facts").fetchone()[0]
            assert fact_count == 3  # the seeded prior fact + 2 new (pivot + other direction)
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_relays_when_dedup_judgment_errors(self, relay_conn, monkeypatch):
        """Fail-safe direction: an uncertain dedup call must not silently
        drop a distinct coincidence -- proceed with a normal relay."""
        conn = relay_db.connect()
        try:
            self._setup_pair(conn, prior_summary="Ananya headed to Tesco a bit ago.")
        finally:
            conn.close()

        monkeypatch.setattr(relay_db, "display_name_for_person", lambda pid: pid.capitalize())
        monkeypatch.setattr(
            relay_watchers, "_run_relay_llm_review",
            lambda prompt: (
                {"final": "", "error": "boom"}
                if "same_topic" in prompt
                else {
                    "final": json.dumps({
                        "match": True, "candidate_index": 0,
                        "summary_for_Ishan": "Ananya is also at Tesco.",
                        "summary_for_other": "Ishan is also at Tesco.",
                    }),
                    "error": None,
                }
            ),
        )
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly", lambda: {}, raising=False
        )

        runner = relay_watchers.GatewayRelayWatchersMixin()
        await runner._relay_scan_tick()

        conn = relay_db.connect()
        try:
            fact_count = conn.execute("SELECT COUNT(*) FROM relay_facts").fetchone()[0]
            assert fact_count == 3  # proceeded normally despite the dedup-check error
        finally:
            conn.close()
