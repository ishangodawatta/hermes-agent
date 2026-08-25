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
import logging

logger = logging.getLogger("gateway.run")


class GatewayRelayWatchersMixin:
    """Background relay-delivery loop methods for GatewayRunner."""

    async def _relay_notifier_watcher(self, interval: float = 10.0) -> None:
        """Poll ``relay_fact_targets`` and deliver pending relays.

        Runs in the gateway event loop; all SQLite work is pushed to a
        thread via ``asyncio.to_thread`` so the loop never blocks on the WAL
        lock, same rationale as the kanban notifier. Failures in one tick
        don't stop subsequent ticks — a delivery that fails this tick is
        simply retried next tick, since it stays ``delivered_at IS NULL``.
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
                out = []
                for item in pending:
                    row = conn.execute(
                        "SELECT platform, chat_id FROM person_channels WHERE person_id = ?",
                        (item.target_person_id,),
                    ).fetchone()
                    platform, chat_id = row if row else (None, None)
                    out.append((item, platform, chat_id))
                return out
            finally:
                conn.close()

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
