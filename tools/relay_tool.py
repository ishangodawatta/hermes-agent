"""Cross-chat relay: the agent-facing write side.

LOCAL-ONLY — NOT FOR UPSTREAM. See hermes_cli/relay_db.py and the
``hermes-agent-cross-chat-relay`` spec for the full design and the decision
to keep this out of NousResearch/hermes-agent.

There is no agent-callable send tool here — the agent can record that
something is worth relaying, or log something that might turn out to matter
later, but delivery and cross-person matching both happen out-of-band via
``gateway/relay_watchers.py``'s background watchers, preserving hermes-agent's
existing "agent can't self-send to a third party" design constraint
(``toolsets.py``'s documented agents-do-not-get-send-message rule).

Two tools, for two different situations:
- ``relay_note``: the agent already knows who should hear about something
  right now (rare — needs cross-person context the agent usually doesn't
  have).
- ``log_activity``: the common case — something worth remembering (an
  errand, a plan) with no one obvious to tell yet. The background scanner
  (``_relay_scanner_watcher``) compares it against other people's logged
  activities later and creates a relay_note-equivalent fact automatically if
  it finds a same-day match.
"""

from __future__ import annotations

from typing import Any, Optional

from hermes_cli import relay_db

RELAY_NOTE_SCHEMA = {
    "name": "relay_note",
    "description": (
        "Record something worth proactively relaying to another registered family "
        "member, e.g. two people independently mentioning the same errand or plan. "
        "This does NOT send a message yourself — it queues the note for background "
        "delivery to each named person's own chat, shortly after this call. Use when "
        "someone tells you something that a specific other registered person would "
        "plausibly want to know, and only for ordinary family-coordination information "
        "(errands, plans, schedules) — not for sensitive personal disclosures unless "
        "the speaker clearly intends them to be shared.\n\n"
        "If a name you name isn't a registered contact, the note is not queued and "
        "you're told so in the response — don't assume it went through."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": (
                    "A short, third-person summary of the fact to relay, written as "
                    "the message the target person will actually receive (e.g. "
                    "'Ishan mentioned he's heading to Tesco'). Not raw message text."
                ),
            },
            "interested": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Display names of registered people who should be notified, e.g. "
                    "['Ananya']. Must match a known contact's name."
                ),
            },
        },
        "required": ["summary", "interested"],
    },
}


def relay_note(
    args: dict[str, Any],
    *,
    session_id: Optional[str] = None,
    **_kw: Any,
) -> str:
    summary = (args.get("summary") or "").strip()
    interested = args.get("interested") or []
    if not summary:
        return "Error: summary is required."
    if not interested:
        return "Error: interested must name at least one person."
    if not session_id:
        return "Error: no session context available; cannot determine the source chat."

    from hermes_state import SessionDB

    session_db = SessionDB(read_only=True)
    session = session_db.get_session(session_id)
    if not session:
        return "Error: could not resolve the current session's chat."
    source_platform = session.get("source") or ""
    source_chat_id = session.get("chat_id") or ""
    if not source_platform or not source_chat_id:
        return "Error: current session has no platform/chat_id — cannot relay from here."

    conn = relay_db.connect()
    try:
        source_person_id = None
        row = conn.execute(
            "SELECT person_id FROM person_channels WHERE platform = ? AND chat_id = ?",
            (source_platform, source_chat_id),
        ).fetchone()
        if row:
            source_person_id = row[0]

        resolved: list[str] = []
        unresolved: list[str] = []
        for name in interested:
            person_id = relay_db.resolve_person_by_name(conn, name)
            if person_id:
                resolved.append(person_id)
            else:
                unresolved.append(name)

        if not resolved:
            return (
                "Not queued: none of the named people are registered contacts "
                f"({', '.join(interested)}). Nothing was relayed."
            )

        fact_id = relay_db.record_relay_fact(
            conn,
            source_person_id=source_person_id,
            source_platform=source_platform,
            source_chat_id=source_chat_id,
            summary=summary,
            target_person_ids=resolved,
        )
    finally:
        conn.close()

    parts = [f"Queued for relay to {len(resolved)} recipient(s) (fact #{fact_id})."]
    if unresolved:
        parts.append(
            f"Not relayed to unregistered name(s): {', '.join(unresolved)} — "
            "no matching contact, nothing was sent to them."
        )
    return " ".join(parts)


LOG_ACTIVITY_SCHEMA = {
    "name": "log_activity",
    "description": (
        "Log something worth remembering in case it turns out to matter later — an "
        "errand, a plan, something the person is about to do — even with no one "
        "specific to tell right now. A background check compares this against other "
        "registered family members' logged activities later that day and relays "
        "automatically if a real coincidence turns up (e.g. someone else independently "
        "mentions the same errand). Use this instead of relay_note whenever you don't "
        "already know a specific person who needs telling — that's the common case.\n\n"
        "If the activity means the person will be away from home, also consider "
        "checking the family calendar (the google-workspace skill's calendar script) "
        "for anything in the next couple of hours they'd need to be back for, and "
        "mention it naturally in your reply if relevant — a human secretary would say "
        "so in the same breath, not as a separate message."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": (
                    "A short, third-person description of the activity, e.g. "
                    "'Ishan is heading to Tesco'. Not raw message text."
                ),
            },
        },
        "required": ["description"],
    },
}


def log_activity(
    args: dict[str, Any],
    *,
    session_id: Optional[str] = None,
    **_kw: Any,
) -> str:
    description = (args.get("description") or "").strip()
    if not description:
        return "Error: description is required."
    if not session_id:
        return "Error: no session context available; cannot determine the source chat."

    from hermes_state import SessionDB

    session_db = SessionDB(read_only=True)
    session = session_db.get_session(session_id)
    if not session:
        return "Error: could not resolve the current session's chat."
    platform = session.get("source") or ""
    chat_id = session.get("chat_id") or ""
    if not platform or not chat_id:
        return "Error: current session has no platform/chat_id — cannot log from here."

    conn = relay_db.connect()
    try:
        row = conn.execute(
            "SELECT person_id FROM person_channels WHERE platform = ? AND chat_id = ?",
            (platform, chat_id),
        ).fetchone()
        if not row:
            return "Not logged: this chat isn't linked to a registered person."
        person_id = row[0]
        activity_id = relay_db.log_activity(
            conn,
            person_id=person_id,
            platform=platform,
            chat_id=chat_id,
            description=description,
        )
    finally:
        conn.close()

    return f"Logged (activity #{activity_id}). No action needed now."


from tools.registry import registry  # noqa: E402

registry.register(
    name="relay_note",
    toolset="relay",
    schema=RELAY_NOTE_SCHEMA,
    handler=lambda args, **kw: relay_note(args, session_id=kw.get("session_id")),
    emoji="📨",
)

registry.register(
    name="log_activity",
    toolset="relay",
    schema=LOG_ACTIVITY_SCHEMA,
    handler=lambda args, **kw: log_activity(args, session_id=kw.get("session_id")),
    emoji="📝",
)
