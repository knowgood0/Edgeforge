"""
Execution provider abstraction — PAPER TRADING ONLY.

EdgeForge never places live orders. ExecutionProvider implementations
only ever (a) send a webhook payload describing a signal, or (b) submit
to a broker's explicitly-paper/sandbox endpoint. There is no live-order
code path in this module, intentionally.

This lets EdgeForge stay decoupled from your existing Flask/Webull bot:
EdgeForge exports a strategy as a machine-readable definition (see
export_strategy_definition below) and/or fires a webhook; your bot
decides what to do with it. Nothing here assumes your bot's internals.
"""

from __future__ import annotations

import abc
import json
import os
from dataclasses import dataclass
from typing import Optional


class ExecutionProviderError(Exception):
    pass


@dataclass
class SignalPayload:
    symbol: str
    direction: str          # 'long' or 'short'
    action: str             # 'entry' or 'exit'
    strategy_name: str
    strategy_version_id: int
    quantity_pct_equity: Optional[float] = None
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    metadata: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "action": self.action,
            "strategy_name": self.strategy_name,
            "strategy_version_id": self.strategy_version_id,
            "quantity_pct_equity": self.quantity_pct_equity,
            "stop_price": self.stop_price,
            "target_price": self.target_price,
            "metadata": self.metadata or {},
        }


class ExecutionProvider(abc.ABC):
    name: str = "abstract"

    @abc.abstractmethod
    def send_signal(self, payload: SignalPayload) -> dict:
        """Send a paper-trading signal. Must return a dict describing
        what happened (e.g. {"status": "sent", "response": ...}) and
        must never place a live order."""
        raise NotImplementedError


class CustomWebhookProvider(ExecutionProvider):
    """POSTs the signal as JSON to a user-configured webhook URL — e.g.
    your existing Flask/Render bot's ingestion endpoint. This is the
    generic connector; it doesn't assume anything about what's on the
    other end."""

    name = "custom_webhook"

    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or os.environ.get("EDGEFORGE_WEBHOOK_URL")
        if not self.webhook_url:
            raise ExecutionProviderError(
                "No webhook URL configured. Set EDGEFORGE_WEBHOOK_URL or pass one explicitly."
            )

    def send_signal(self, payload: SignalPayload) -> dict:
        try:
            import requests
        except ImportError as e:
            raise ExecutionProviderError("requests package not installed.") from e
        try:
            resp = requests.post(self.webhook_url, json=payload.to_dict(), timeout=10)
            resp.raise_for_status()
            return {"status": "sent", "http_status": resp.status_code}
        except Exception as e:
            raise ExecutionProviderError(f"Webhook send failed: {e}") from e


class TradingViewWebhookProvider(CustomWebhookProvider):
    """Same mechanics as CustomWebhookProvider — named separately because
    TradingView alert-webhook payload conventions (plain string bodies
    from Pine `alert()` calls) sometimes differ from JSON POSTs. Kept as
    a distinct class so that difference has an obvious place to live
    once you wire it to a real TradingView alert."""
    name = "tradingview_webhook"


class WebullPaperProvider(ExecutionProvider):
    """Placeholder for direct Webull sandbox integration. NOT implemented
    here — you already have working Webull sandbox order-construction
    code in your separate trading-bot project, and per your own note
    that SDK's options-order schema needs to be resolved by inspecting
    the SDK source directly, not guessed at. Wiring this class to reuse
    that bot (e.g. via CustomWebhookProvider pointed at your bot's
    endpoint) is the recommended near-term path rather than duplicating
    that integration inside EdgeForge."""

    name = "webull_paper"

    def send_signal(self, payload: SignalPayload) -> dict:
        raise NotImplementedError(
            "Direct Webull integration is not implemented in EdgeForge. "
            "Use CustomWebhookProvider pointed at your existing trading bot "
            "instead of duplicating that integration here."
        )


_PROVIDERS = {
    "custom_webhook": CustomWebhookProvider,
    "tradingview_webhook": TradingViewWebhookProvider,
    "webull_paper": WebullPaperProvider,
}


def get_execution_provider(name: str, **kwargs) -> ExecutionProvider:
    if name not in _PROVIDERS:
        raise ValueError(f"Unknown execution provider: {name}")
    return _PROVIDERS[name](**kwargs)


def export_strategy_definition(strategy_name: str, strategy_version_id: int,
                                definition: dict, symbol: str) -> dict:
    """The decoupling point: a plain JSON strategy definition your
    existing Python bot (or anything else) could consume — no EdgeForge
    runtime dependency required on the consuming side."""
    return {
        "strategy_name": strategy_name,
        "strategy_version_id": strategy_version_id,
        "symbol": symbol,
        "definition": definition,
        "source": "EdgeForge",
        "schema_version": 1,
    }
