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

One agent-callable tool:
- ``relay_note``: the agent already knows who should hear about something
  right now (rare — needs cross-person context the agent usually doesn't
  have).

The common case — an ordinary errand or plan with no one obvious to tell yet
— is no longer an agent tool. It's handled by a silent, always-run background
classifier (``gateway/notice_classifier.py``) that runs on every inbound
message independently of the conversational agent, because relying on the
live model to remember to call a tool for this proved unreliable in practice
(confirmed via live instrumentation 2026-08-26 — the instruction reached the
model, the tool was available, and it still wasn't called). See the
``hermes-agent-alfred-manager-architecture`` spec.
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

    source_resolved = relay_db.resolve_person_from_chat_id(source_chat_id)
    source_person_id = source_resolved[0] if source_resolved else None

    conn = relay_db.connect()
    try:
        resolved: list[str] = []
        unresolved: list[str] = []
        for name in interested:
            person_id = relay_db.resolve_person_by_name(name)
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
