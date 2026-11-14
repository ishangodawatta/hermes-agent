"""Cross-chat relay: delivery watcher.

LOCAL-ONLY — NOT FOR UPSTREAM. See hermes_cli/relay_db.py, tools/relay_tool.py,
and the ``hermes-agent-cross-chat-relay`` spec for the full design.

Mirrors gateway/kanban_watchers.py's ``_kanban_notifier_watcher`` shape
(durable row store + poll + deliver via the adapter's own ``send()``),
simplified for relay's single-store, one-shot-delivery model — no
multi-board, no multi-profile, no subscription lifecycle (a
``relay_fact_targets`` row is delivered once, then done; nothing re-arms it).

Deliberately does NOT give the agent a send-to-third-party tool — this
watcher is the only thing that ever calls ``adapter.send()`` for a relay,
same outside-the-loop delivery path the kanban notifier already uses. See
``toolsets.py``'s documented agents-do-not-get-send-message design
constraint, which this preserves rather than reverses.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger("gateway.run")


def _resolve_relay_scanner_runtime(cfg: Dict[str, Any]):
    """Resolve provider/model for the relay-scanner judgment fork.

    Mirrors agent/curator.py's _resolve_review_runtime() exactly, but reads
    auxiliary.relay_scanner.{provider,model} instead of auxiliary.curator —
    a separate aux-task slot so the scanner can run on a different (e.g.
    cheaper) model than curator without them fighting over one config key.
    Not imported from curator.py directly: that function is hardcoded to
    the "curator" slot name, and duplicating this ~15-line resolver is
    simpler than threading a task-name parameter through upstream-tracked
    code for a local-only feature's sake.
    """
    _main = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    _main_provider = _main.get("provider") or "auto"
    _main_model = _main.get("default") or _main.get("model") or ""

    _aux = cfg.get("auxiliary", {}) if isinstance(cfg.get("auxiliary"), dict) else {}
    _task = _aux.get("relay_scanner", {}) if isinstance(_aux.get("relay_scanner"), dict) else {}
    _task_provider = (_task.get("provider") or "").strip() or None
    _task_model = (_task.get("model") or "").strip() or None
    if _task_provider and _task_provider != "auto" and _task_model:
        return _task_provider, _task_model, _task.get("api_key"), _task.get("base_url")

    return _main_provider, _main_model, None, None


def _run_relay_llm_review(prompt: str) -> Dict[str, Any]:
    """Spawn an isolated AIAgent fork to judge a candidate relay match.

    Mirrors agent/curator.py's _run_llm_review() shape (see that function's
    docstring for the full rationale): no tools, redirected stdout/stderr,
    never raises — callers get a structured failure instead. Returns a dict
    with "final" (raw response text) and "error" (set on failure).
    """
    result: Dict[str, Any] = {"final": "", "error": None}
    try:
        from run_agent import AIAgent
    except Exception as e:
        result["error"] = f"AIAgent import failed: {e}"
        return result

    _model_name = ""
    _resolved_provider = None
    _api_key = None
    _base_url = None
    _api_mode = None
    _credential_pool = None
    _request_overrides: Dict[str, Any] = {}
    _max_tokens = None
    try:
        from hermes_cli.config import load_config_readonly
        from hermes_cli.runtime_provider import resolve_runtime_provider

        _cfg = load_config_readonly()
        _provider, _model_name, _explicit_key, _explicit_base = _resolve_relay_scanner_runtime(_cfg)
        _rp = resolve_runtime_provider(
            requested=_provider,
            target_model=_model_name,
            explicit_api_key=_explicit_key,
            explicit_base_url=_explicit_base,
        )
        _api_key = _rp.get("api_key")
        _base_url = _rp.get("base_url")
        _api_mode = _rp.get("api_mode")
        _resolved_provider = _rp.get("provider") or _provider
        _credential_pool = _rp.get("credential_pool")
        _request_overrides = _rp.get("request_overrides") or {}
        _max_tokens = _rp.get("max_output_tokens")
        if isinstance(_rp.get("model"), str) and _rp["model"].strip():
            _model_name = _rp["model"].strip()
    except Exception as e:
        logger.debug("relay scanner: provider resolution failed: %s", e, exc_info=True)

    try:
        _agent_kwargs: Dict[str, Any] = {}
        if isinstance(_max_tokens, int):
            _agent_kwargs["max_tokens"] = _max_tokens
        review_agent = AIAgent(
            model=_model_name,
            provider=_resolved_provider,
            api_key=_api_key,
            base_url=_base_url,
            api_mode=_api_mode,
            credential_pool=_credential_pool,
            request_overrides=_request_overrides,
            **_agent_kwargs,
            enabled_toolsets=[],
            max_iterations=3,
            quiet_mode=True,
            platform="relay_scanner",
            skip_context_files=True,
            skip_memory=True,
        )
        review_agent._memory_nudge_interval = 0
        review_agent._skill_nudge_interval = 0

        with open(os.devnull, "w", encoding="utf-8") as _devnull, \
             contextlib.redirect_stdout(_devnull), \
             contextlib.redirect_stderr(_devnull):
            conv_result = review_agent.run_conversation(user_message=prompt)

        final = ""
        if isinstance(conv_result, dict):
            final = str(conv_result.get("final_response") or "").strip()
        result["final"] = final
    except Exception as e:
        logger.exception("relay scanner: judgment fork failed")
        result["error"] = str(e)
    return result


_MATCH_PROMPT_TEMPLATE = """You are checking whether a newly-logged family activity relates to any \
of a set of other family members' recently-logged activities, to decide whether Alfred should \
proactively let both people know about the connection.

New activity, from {pivot_name}, said {pivot_when}:
"{pivot_description}"

Other family members' recent activities today:
{candidates_block}

Does the new activity genuinely relate to any of these — the same errand, place, plan, or \
otherwise something both people would plausibly want to know about each other? A shared \
keyword alone is not enough; it must be a real, meaningful coincidence.

Respond with ONLY a JSON object, no other text:
{{"match": true/false, "candidate_index": <index of the matching candidate, or null>, \
"summary_for_{pivot_name}": "<short third-person message telling {pivot_name} about the other \
person's activity, naming when it happened if not just now>", \
"summary_for_other": "<short third-person message telling the other person about {pivot_name}'s \
activity>"}}

If there is no real match, respond {{"match": false, "candidate_index": null, \
"summary_for_{pivot_name}": "", "summary_for_other": ""}}."""


_DEDUP_PROMPT_TEMPLATE = """You are checking whether a new family coincidence match is substantially the \
same topic as one already relayed to the same two people recently, to avoid telling them the same thing \
twice.

Already relayed, {prior_when}:
"{prior_summary}"

New match about to be relayed:
"{new_summary}"

Is the new match about the SAME underlying topic or event as the one already relayed (telling them again \
would be redundant), or a genuinely DIFFERENT topic or coincidence between the same two people (it should \
still be sent)? A shared topic phrased differently is still the same topic — judge substance, not wording.

Respond with ONLY a JSON object, no other text:
{{"same_topic": true/false}}"""


def _get_dedup_window_seconds(cfg: Dict[str, Any]) -> int:
    """Hours a prior relay between the same pair counts as "recent" for
    duplicate-topic suppression. Config: relay.dedup_window_hours (default
    4) — see the cross-chat-relay spec's "Duplicate-relay dedup" section for
    why 4h and why this stays a knob rather than a hardcoded constant (no
    empirical tuning yet).
    """
    relay_cfg = cfg.get("relay", {}) if isinstance(cfg.get("relay"), dict) else {}
    hours = relay_cfg.get("dedup_window_hours", 4)
    try:
        return max(0, int(hours)) * 3600
    except (TypeError, ValueError):
        return 4 * 3600


def _format_when(occurred_at: int, now: int) -> str:
    delta = max(0, now - occurred_at)
    if delta < 300:
        return "just now"
    if delta < 3600:
        return f"{delta // 60} minutes ago"
    if delta < 6 * 3600:
        return f"{delta // 3600} hours ago"
    return "earlier today" if delta < 20 * 3600 else "yesterday"


class GatewayRelayWatchersMixin:
    """Background relay-delivery loop methods for GatewayRunner."""

    async def _relay_scanner_watcher(self, interval: float = 4.0) -> None:
        """Poll ``relay_activities`` for uncompared entries and cross-match
        each against other people's same-day activities.

        Short interval by design (see the spec's "Detection layer" section):
        a true synchronous hook fired from inside a live conversation turn
        would either block that turn on an LLM call or need real async task
        scheduling wired into the conversation loop — a short poll achieves
        functionally the same "near-immediate" feel with far less risk.

        Lowered from 15.0 to 4.0 (2026-08-26): a real live test measured
        ~34s from activity-logged to compared_at, of which up to 15s was
        pure "waiting for the next tick" rather than the judgment LLM call
        itself. An empty tick is a cheap SELECT that returns immediately
        (see _relay_scan_tick's early exit when there are no pivots), so a
        tighter interval costs nothing when idle and only shortens the
        worst-case wait when there's real work to pick up.
        """
        await asyncio.sleep(5)
        while self._running:
            try:
                await self._relay_scan_tick()
            except Exception:
                logger.exception("relay scanner: tick failed")
            await asyncio.sleep(interval)

    async def _relay_scan_tick(self) -> None:
        from hermes_cli import relay_db

        def _collect_pivots():
            conn = relay_db.connect()
            try:
                return relay_db.uncompared_activities(conn)
            finally:
                conn.close()

        pivots = await asyncio.to_thread(_collect_pivots)
        if not pivots:
            return

        import time as _time

        now = int(_time.time())
        day_start = now - 24 * 3600

        for pivot in pivots:
            def _already_handled_this_tick(pivot=pivot):
                conn = relay_db.connect()
                try:
                    row = conn.execute(
                        "SELECT compared_at, matched_fact_id FROM relay_activities "
                        "WHERE activity_id = ?",
                        (pivot.activity_id,),
                    ).fetchone()
                    if row is None:
                        return True
                    compared_at, matched_fact_id = row
                    return compared_at is not None or matched_fact_id is not None
                finally:
                    conn.close()

            # Re-check against the DB, not just the batch fetched at tick
            # start: if an earlier pivot in THIS tick already matched this
            # activity as its candidate, mark_activity_matched() sets
            # matched_fact_id but deliberately NOT compared_at (a matched
            # activity should still get a normal pivot turn in a LATER
            # tick, in case it separately relates to something else too —
            # matching isn't meant to be exclusive across ticks). But
            # without checking matched_fact_id here too, this SAME tick
            # would re-evaluate it as its own pivot and could match a
            # second time, relaying the same coincidence twice. Confirmed
            # to happen this way during testing (Ishan's and Ananya's
            # Tesco activities both matched independently in one tick
            # before this fix — 4 facts instead of 2).
            if await asyncio.to_thread(_already_handled_this_tick):
                continue

            def _load_context(pivot=pivot):
                pivot_name = relay_db.display_name_for_person(pivot.person_id)
                conn = relay_db.connect()
                try:
                    candidates = relay_db.candidate_activities(
                        conn, exclude_person_id=pivot.person_id, since=day_start
                    )
                    return pivot_name, candidates
                finally:
                    conn.close()

            pivot_name, candidates = await asyncio.to_thread(_load_context)

            if not candidates:
                await asyncio.to_thread(self._relay_mark_compared_sync, pivot.activity_id, None)
                continue

            candidates_block = "\n".join(
                f'{i}. From {self._relay_person_display_name(c.person_id)}, '
                f'{_format_when(c.occurred_at, now)}: "{c.description}"'
                for i, c in enumerate(candidates)
            )
            prompt = _MATCH_PROMPT_TEMPLATE.format(
                pivot_name=pivot_name,
                pivot_when=_format_when(pivot.occurred_at, now),
                pivot_description=pivot.description,
                candidates_block=candidates_block,
            )

            review = await asyncio.to_thread(_run_relay_llm_review, prompt)
            if review.get("error"):
                logger.warning(
                    "relay scanner: judgment failed for activity #%s: %s",
                    pivot.activity_id, review["error"],
                )
                # Leave compared_at unset — retried next tick rather than
                # silently dropped, since this is a real failure, not "no match".
                continue

            parsed = self._relay_parse_judgment(review.get("final") or "")
            if parsed is None:
                # The fork ran without raising, but didn't return parseable
                # JSON (e.g. an API error surfaced as response text rather
                # than an exception — confirmed to happen this way during
                # testing). Treat this as a retriable failure, not "no
                # match": leave compared_at unset so the next tick retries,
                # rather than silently losing a genuine coincidence to a
                # transient error.
                logger.warning(
                    "relay scanner: unparseable judgment for activity #%s, will retry: %r",
                    pivot.activity_id, (review.get("final") or "")[:200],
                )
                continue
            if not parsed.get("match"):
                await asyncio.to_thread(self._relay_mark_compared_sync, pivot.activity_id, None)
                continue

            idx = parsed.get("candidate_index")
            if not isinstance(idx, int) or not (0 <= idx < len(candidates)):
                logger.warning(
                    "relay scanner: judgment claimed a match but candidate_index %r "
                    "is invalid for activity #%s; treating as no match",
                    idx, pivot.activity_id,
                )
                await asyncio.to_thread(self._relay_mark_compared_sync, pivot.activity_id, None)
                continue

            matched = candidates[idx]
            summary_for_pivot = str(parsed.get(f"summary_for_{pivot_name}") or "").strip()
            summary_for_other = str(parsed.get("summary_for_other") or "").strip()
            if not summary_for_pivot or not summary_for_other:
                logger.warning(
                    "relay scanner: judgment claimed a match but omitted a summary "
                    "for activity #%s; treating as no match rather than relaying blank text",
                    pivot.activity_id,
                )
                await asyncio.to_thread(self._relay_mark_compared_sync, pivot.activity_id, None)
                continue

            def _find_recent_fact(pivot=pivot, matched=matched):
                conn = relay_db.connect()
                try:
                    from hermes_cli.config import load_config_readonly

                    window = _get_dedup_window_seconds(load_config_readonly())
                    return relay_db.most_recent_fact_between(
                        conn, pivot.person_id, matched.person_id, since=now - window
                    )
                finally:
                    conn.close()

            recent_fact = await asyncio.to_thread(_find_recent_fact)
            if recent_fact is not None:
                dedup_prompt = _DEDUP_PROMPT_TEMPLATE.format(
                    prior_when=_format_when(recent_fact.created_at, now),
                    prior_summary=recent_fact.summary,
                    new_summary=summary_for_pivot,
                )
                dedup_review = await asyncio.to_thread(_run_relay_llm_review, dedup_prompt)
                dedup_parsed = (
                    None
                    if dedup_review.get("error")
                    else self._relay_parse_judgment(dedup_review.get("final") or "")
                )
                if dedup_parsed is not None and dedup_parsed.get("same_topic"):
                    # Same topic already relayed recently -- suppress a
                    # redundant second relay, but still mark both activities
                    # against the EXISTING fact so neither is retried forever.
                    def _suppress_duplicate(
                        pivot=pivot, matched=matched, fact_id=recent_fact.fact_id,
                    ):
                        conn = relay_db.connect()
                        try:
                            relay_db.mark_activity_compared(
                                conn, pivot.activity_id, matched_fact_id=fact_id
                            )
                            relay_db.mark_activity_matched(conn, matched.activity_id, fact_id)
                        finally:
                            conn.close()

                    await asyncio.to_thread(_suppress_duplicate)
                    logger.debug(
                        "relay scanner: suppressed duplicate relay for activity #%s (%s) "
                        "with #%s (%s) -- same topic as fact #%s",
                        pivot.activity_id, pivot_name, matched.activity_id,
                        matched.person_id, recent_fact.fact_id,
                    )
                    continue
                # On judgment failure/error, or a genuinely different topic,
                # fall through and relay as normal -- an occasional duplicate
                # is a much smaller cost than silently losing a distinct,
                # real coincidence to an uncertain dedup call.

            def _write_match(
                pivot=pivot, matched=matched,
                summary_for_pivot=summary_for_pivot, summary_for_other=summary_for_other,
            ):
                conn = relay_db.connect()
                try:
                    # Two facts, each sourced from the activity it's actually
                    # telling the *other* person about — matched.platform/
                    # chat_id for the fact delivered to pivot, and vice versa.
                    fact_id = relay_db.record_relay_fact(
                        conn,
                        source_person_id=matched.person_id,
                        source_platform=matched.platform,
                        source_chat_id=matched.chat_id,
                        summary=summary_for_pivot,
                        target_person_ids=[pivot.person_id],
                    )
                    relay_db.record_relay_fact(
                        conn,
                        source_person_id=pivot.person_id,
                        source_platform=pivot.platform,
                        source_chat_id=pivot.chat_id,
                        summary=summary_for_other,
                        target_person_ids=[matched.person_id],
                    )
                    relay_db.mark_activity_compared(conn, pivot.activity_id, matched_fact_id=fact_id)
                    relay_db.mark_activity_matched(conn, matched.activity_id, fact_id)
                finally:
                    conn.close()

            await asyncio.to_thread(_write_match)
            logger.debug(
                "relay scanner: matched activity #%s (%s) with #%s (%s)",
                pivot.activity_id, pivot_name, matched.activity_id, matched.person_id,
            )

    def _relay_person_display_name(self, person_id: str) -> str:
        from hermes_cli import relay_db

        return relay_db.display_name_for_person(person_id)

    def _relay_mark_compared_sync(self, activity_id: int, matched_fact_id: Optional[int]) -> None:
        from hermes_cli import relay_db

        conn = relay_db.connect()
        try:
            relay_db.mark_activity_compared(conn, activity_id, matched_fact_id=matched_fact_id)
        finally:
            conn.close()

    def _relay_parse_judgment(self, text: str) -> Optional[Dict[str, Any]]:
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except (json.JSONDecodeError, ValueError):
            return None

    async def _relay_notifier_watcher(self, interval: float = 3.0) -> None:
        """Poll ``relay_fact_targets`` and deliver pending relays.

        Runs in the gateway event loop; all SQLite work is pushed to a
        thread via ``asyncio.to_thread`` so the loop never blocks on the WAL
        lock, same rationale as the kanban notifier. Failures in one tick
        don't stop subsequent ticks — a delivery that fails this tick is
        simply retried next tick, since it stays ``delivered_at IS NULL``.

        Lowered from 10.0 to 3.0 (2026-08-26), same rationale as the
        scanner's interval reduction — an empty tick is a cheap SELECT with
        an early exit, so a tighter interval only shortens real delivery
        latency, not idle cost.
        """
        await asyncio.sleep(5)  # let adapters finish wiring up before the first tick
        while self._running:
            try:
                await self._relay_deliver_pending_tick()
            except Exception:
                logger.exception("relay notifier: tick failed")
            await asyncio.sleep(interval)

    async def _relay_deliver_pending_tick(self) -> None:
        from hermes_cli import relay_db

        def _collect():
            conn = relay_db.connect()
            try:
                pending = relay_db.pending_relay_targets(conn)
            finally:
                conn.close()
            out = []
            for item in pending:
                # Only WhatsApp is wired for delivery today (see
                # resolve_chat_id_for_person's docstring) -- this widens
                # naturally once family-facts carries other platform kinds.
                chat_id = relay_db.resolve_chat_id_for_person(
                    item.target_person_id, platform="whatsapp"
                )
                platform = "whatsapp" if chat_id else None
                out.append((item, platform, chat_id))
            return out

        deliveries = await asyncio.to_thread(_collect)
        if not deliveries:
            return

        from gateway.config import Platform

        active_platforms = {
            getattr(platform, "value", str(platform)).lower()
            for platform in self.adapters.keys()
        }

        for item, platform_str, chat_id in deliveries:
            if not platform_str or not chat_id:
                # Target person has no registered channel yet — leave the
                # fact pending rather than dropping it; it'll deliver once
                # someone links a channel for them.
                logger.debug(
                    "relay notifier: target %s has no registered channel; "
                    "leaving fact #%s pending",
                    item.target_person_id, item.fact_id,
                )
                continue
            if platform_str.lower() not in active_platforms:
                continue
            try:
                platform = Platform(platform_str)
            except ValueError:
                logger.warning(
                    "relay notifier: unknown platform %r for fact #%s",
                    platform_str, item.fact_id,
                )
                continue
            adapter = self._authorization_adapter(platform, None)
            if not adapter:
                continue
            try:
                send_res = await adapter.send(chat_id, item.summary)
                if getattr(send_res, "success", True) is False:
                    raise RuntimeError(
                        "adapter send() reported failure: "
                        f"{getattr(send_res, 'error', None) or 'unknown error'}"
                    )
            except Exception:
                logger.exception(
                    "relay notifier: delivery failed for fact #%s to %s/%s; "
                    "will retry next tick",
                    item.fact_id, platform_str, chat_id,
                )
                continue

            def _mark(fact_id=item.fact_id, target=item.target_person_id):
                conn = relay_db.connect()
                try:
                    relay_db.mark_delivered(conn, fact_id, target)
                finally:
                    conn.close()

            await asyncio.to_thread(_mark)
            logger.debug(
                "relay notifier: delivered fact #%s to %s/%s",
                item.fact_id, platform_str, chat_id,
            )
