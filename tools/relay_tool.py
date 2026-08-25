"""Cross-chat relay: the agent-facing write side.

LOCAL-ONLY — NOT FOR UPSTREAM. See hermes_cli/relay_db.py and the
``hermes-agent-cross-chat-relay`` spec for the full design and the decision
to keep this out of NousResearch/hermes-agent.

This is deliberately the ONLY relay-related tool the agent gets. There is no
agent-callable send tool here — the agent can record that something is worth
relaying and to whom, but delivery happens out-of-band via the
``gateway/relay_watchers.py`` background watcher, preserving hermes-agent's
existing "agent can't self-send to a third party" design constraint
(``toolsets.py``'s documented agents-do-not-get-send-message rule).
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


from tools.registry import registry  # noqa: E402

registry.register(
    name="relay_note",
    toolset="relay",
    schema=RELAY_NOTE_SCHEMA,
    handler=lambda args, **kw: relay_note(args, session_id=kw.get("session_id")),
    emoji="📨",
)
