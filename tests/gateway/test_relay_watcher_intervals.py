"""Guards the relay scanner/notifier poll-interval defaults.

LOCAL-ONLY feature. Lowered 2026-08-26 after a live test showed most of the
end-to-end latency was "waiting for the next tick," not the judgment LLM
call itself — see gateway/relay_watchers.py's docstrings for the measured
numbers. This test exists purely to catch an accidental regression back
toward the old, slower defaults.
"""

import inspect

from gateway.relay_watchers import GatewayRelayWatchersMixin


def _default_interval(method_name: str) -> float:
    sig = inspect.signature(getattr(GatewayRelayWatchersMixin, method_name))
    return sig.parameters["interval"].default


def test_scanner_interval_is_tightened():
    assert _default_interval("_relay_scanner_watcher") == 4.0


def test_notifier_interval_is_tightened():
    assert _default_interval("_relay_notifier_watcher") == 3.0
