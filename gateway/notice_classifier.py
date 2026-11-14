"""Cross-chat relay: silent per-message notice/classifier pass.

LOCAL-ONLY — NOT FOR UPSTREAM. See hermes_cli/relay_db.py, gateway/relay_watchers.py,
and the ``hermes-agent-alfred-manager-architecture`` spec for the full design.

Runs once per inbound WhatsApp message, in parallel with Alfred's own reply
(never blocking it, never talking to the user itself). Replaces the
``log_activity`` agent tool, which depended on the live conversational model
remembering to call it — confirmed via live instrumentation (2026-08-26) to be
a real, reproducible model-compliance gap, not a prompt-delivery bug: the
instruction reached the model, the tool was available, and it still wasn't
called. This module removes that dependency entirely by making the "should
this be logged" judgment a separate, always-run, isolated fork — the same
pattern as ``relay_watchers.py``'s ``_run_relay_llm_review`` / ``agent/curator.py``'s
``_run_llm_review``.

Routes each message's independent signals to whichever existing store already
fits, rather than one new catch-all table (see the spec's "relay_activities
stays narrow" decision):
- an errand/plan with coincidence potential -> ``relay_activities`` (feeds
  ``_relay_scanner_watcher``)
- an explicit item request -> a Todoist task, ``shopping`` label + the
  requester's own person-label
- a commitment/follow-up with no coincidence angle -> a Todoist task, the
  requester's own person-label
A single message can hit more than one of these, or none.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import subprocess
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("gateway.run")

_TODOIST_SCRIPT_ENV = "HERMES_HOME"
_TODOIST_SCRIPT_REL = "skills/productivity/todoist/scripts/todoist.py"
# Discovered once via `todoist.py projects` (2026-08-26) — this is where
# Alfred's own agent-created tasks already land (dog-check nudges, etc.).
_AGENT_TASKS_PROJECT_ID = "6hCWJ6ffp369M3R9"


def _resolve_notice_classifier_runtime(cfg: Dict[str, Any]):
    """Resolve provider/model for the notice-classifier judgment fork.

    Mirrors relay_watchers.py's _resolve_relay_scanner_runtime() exactly, but
    reads auxiliary.notice_classifier.{provider,model} — its own aux-task
    slot so it can run on a different (e.g. cheaper) model than the relay
    scanner or curator without them fighting over one config key.
    """
    _main = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    _main_provider = _main.get("provider") or "auto"
    _main_model = _main.get("default") or _main.get("model") or ""

    _aux = cfg.get("auxiliary", {}) if isinstance(cfg.get("auxiliary"), dict) else {}
    _task = _aux.get("notice_classifier", {}) if isinstance(_aux.get("notice_classifier"), dict) else {}
    _task_provider = (_task.get("provider") or "").strip() or None
    _task_model = (_task.get("model") or "").strip() or None
    if _task_provider and _task_provider != "auto" and _task_model:
        return _task_provider, _task_model, _task.get("api_key"), _task.get("base_url")

    return _main_provider, _main_model, None, None


def _run_notice_classifier_llm(prompt: str) -> Dict[str, Any]:
    """Spawn an isolated AIAgent fork to classify one message.

    Same shape as relay_watchers.py's _run_relay_llm_review: no tools,
    redirected stdout/stderr, never raises — callers get a structured
    failure instead.
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
        _provider, _model_name, _explicit_key, _explicit_base = _resolve_notice_classifier_runtime(_cfg)
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
        logger.debug("notice classifier: provider resolution failed: %s", e, exc_info=True)

    try:
        _agent_kwargs: Dict[str, Any] = {}
        if isinstance(_max_tokens, int):
            _agent_kwargs["max_tokens"] = _max_tokens
        classifier_agent = AIAgent(
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
            platform="notice_classifier",
            skip_context_files=True,
            skip_memory=True,
        )
        classifier_agent._memory_nudge_interval = 0
        classifier_agent._skill_nudge_interval = 0

        with open(os.devnull, "w", encoding="utf-8") as _devnull, \
             contextlib.redirect_stdout(_devnull), \
             contextlib.redirect_stderr(_devnull):
            conv_result = classifier_agent.run_conversation(user_message=prompt)

        final = ""
        if isinstance(conv_result, dict):
            final = str(conv_result.get("final_response") or "").strip()
        result["final"] = final
    except Exception as e:
        logger.exception("notice classifier: judgment fork failed")
        result["error"] = str(e)
    return result


_NOTICE_PROMPT_TEMPLATE = """You are a silent background classifier for a family WhatsApp assistant. \
You see one message and must decide, independently of however the assistant replies, whether \
anything in it is worth persisting for later. You never reply to anyone — you only classify.

Message from {person_name}:
"{message_text}"

Decide three independent things:
1. Is this an ordinary errand, plan, or "heading out" statement worth logging in case it \
coincides with another family member's separately-mentioned plan (e.g. "heading to Tesco", \
"popping out for a bit")? Only things describing the person doing or going somewhere qualify.
2. Does the message explicitly ask for an item to be picked up or bought (e.g. "can someone \
grab toothpaste next time they're at Tesco")?
3. Does the message describe a commitment or something that should be followed up on later \
(e.g. "I'll call the dentist tomorrow", "remind me to chase the plumber")?

A message can be none of these (ordinary chit-chat, a question, a reply) — in that case all \
three are false.

Respond with ONLY a JSON object, no other text:
{{"is_errand": true/false, "errand_description": "<short third-person description if is_errand, else empty>", \
"is_item_request": true/false, "item": "<the plain item name if is_item_request, else empty>", \
"is_commitment": true/false, "commitment_description": "<short third-person description if is_commitment, else empty>"}}"""


def _parse_notice_judgment(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None


def _todoist_script_path() -> Optional[str]:
    hermes_home = os.environ.get(_TODOIST_SCRIPT_ENV)
    if not hermes_home:
        try:
            from hermes_constants import get_hermes_home

            hermes_home = str(get_hermes_home())
        except Exception:
            return None
    path = os.path.join(hermes_home, _TODOIST_SCRIPT_REL)
    return path if os.path.isfile(path) else None


def _todoist_add(content: str, *, labels: list[str]) -> None:
    """Best-effort Todoist task creation. Never raises — a failed write here
    should not surface to the user or crash the classifier tick."""
    script = _todoist_script_path()
    if not script:
        logger.warning("notice classifier: todoist.py not found, skipping task creation")
        return
    cmd = [
        "python3", script, "add", content,
        "--project", _AGENT_TASKS_PROJECT_ID,
        "--labels", ",".join(labels),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if proc.returncode != 0:
            logger.warning("notice classifier: todoist add failed: %s", proc.stderr[:500])
    except Exception:
        logger.exception("notice classifier: todoist add raised")


_REACTION_MARKER_RE = re.compile(r"^\[Reaction:\s")


def run_notice_classifier_tick(*, platform: str, chat_id: str, message_text: str) -> None:
    """Classify one inbound message and route any signals. Synchronous —
    callers dispatch this via asyncio.to_thread so it never blocks the
    conversational turn. Fails silently (logged, not raised): a missed
    classification is no worse than today's status quo, and this must never
    be able to disrupt or delay the user-facing reply.

    A WhatsApp reaction (``[Reaction: <emoji> to <msgid>]``, see
    bridge_helpers.js's formatReactionText()) is skipped before the LLM is
    even invoked — confirmed live 2026-08-26 that a plain thumbs-up got
    hallucinated into a fabricated errand ("Ishan going to Tesco") since
    a fresh, context-free classifier has nothing to ground a reaction in.
    A reaction can never itself be an errand/request/commitment, so there's
    nothing for the classifier to legitimately find here. Treating a 👍/✅
    as a genuine acknowledgement (e.g. closing a related Todoist task) is
    real follow-on work blocked on resolving the reacted-to message's own
    content, which isn't reliably available today — see the
    alfred-manager-architecture spec's "Reaction handling" section.
    """
    if not message_text or not message_text.strip():
        return
    if _REACTION_MARKER_RE.match(message_text.strip()):
        return

    from hermes_cli import relay_db

    resolved = relay_db.resolve_person_from_chat_id(chat_id)
    if not resolved:
        return  # chat isn't linked to a family-facts person; nothing to classify against
    person_id, person_name = resolved

    prompt = _NOTICE_PROMPT_TEMPLATE.format(person_name=person_name, message_text=message_text)
    review = _run_notice_classifier_llm(prompt)
    if review.get("error"):
        logger.warning("notice classifier: judgment failed: %s", review["error"])
        return

    parsed = _parse_notice_judgment(review.get("final") or "")
    if parsed is None:
        logger.warning(
            "notice classifier: unparseable judgment, dropping this turn: %r",
            (review.get("final") or "")[:200],
        )
        return

    label = person_name.strip().lower()

    if parsed.get("is_errand"):
        description = str(parsed.get("errand_description") or "").strip()
        if description:
            conn = relay_db.connect()
            try:
                activity_id = relay_db.log_activity(
                    conn,
                    person_id=person_id,
                    platform=platform,
                    chat_id=chat_id,
                    description=description,
                )
                logger.debug("notice classifier: logged activity #%s for %s", activity_id, person_name)
            finally:
                conn.close()

    if parsed.get("is_item_request"):
        item = str(parsed.get("item") or "").strip()
        if item:
            # "shopping" is a topic label (subject matter), not a type --
            # every agent_tasks item needs a real type too, or the
            # taxonomy-conformance checker flags it (see
            # hermes-agent-task-followthrough-consolidation spec).
            _todoist_add(item, labels=["shopping", "task", label])
            logger.debug("notice classifier: logged shopping item %r for %s", item, person_name)

    if parsed.get("is_commitment"):
        description = str(parsed.get("commitment_description") or "").strip()
        if description:
            # "task" type: it's the speaker's own commitment to do
            # something, not a chase-someone-else waiting-reply.
            _todoist_add(description, labels=["task", label])
            logger.debug("notice classifier: logged commitment for %s", person_name)
