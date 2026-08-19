"""
Event-driven backtest engine.

Design goal: it should be structurally difficult to accidentally use
future information, not just conventionally avoided.

Pipeline per bar i:
  1. Signal is evaluated using data available AS OF bar i's CLOSE
     (i.e. using columns/features at index i, which by construction —
     see features/engineering.py — only use data up to and including i).
  2. If a signal fires on bar i, the resulting order is not filled on
     bar i. It is queued and filled on bar i+1, using either i+1's open
     (default) or i+1's close, per `execution_mode`.
  3. Stops/targets are checked using bar i+1's high/low — i.e. a trade
     entered on i+1's open can still be stopped out intraday on i+1
     itself, which is realistic (not lookahead, since the stop is a
     price level fixed at entry, not information from the future).

This engine does NOT support same-bar signal->fill. That restriction is
intentional; it's the single biggest source of accidental lookahead bias
in amateur backtesters (e.g. "buy when RSI < 30" filled at that same
bar's close, when RSI < 30 could only be confirmed once the close was
already known).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd


@dataclass
class Trade:
    symbol: str
    direction: str  # 'long' or 'short'
    entry_date: str
    entry_price: float
    quantity: float
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    max_holding_days: Optional[int] = None
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    holding_days: Optional[int] = None

    def close(self, exit_date: str, exit_price: float, reason: str):
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.exit_reason = reason
        sign = 1 if self.direction == "long" else -1
        self.pnl = sign * (exit_price - self.entry_price) * self.quantity
        self.pnl_pct = sign * (exit_price - self.entry_price) / self.entry_price
        self.holding_days = (pd.Timestamp(exit_date) - pd.Timestamp(self.entry_date)).days


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    execution_mode: str = "next_bar_open"     # 'next_bar_open' | 'next_bar_close'
    slippage_bps: float = 5.0                 # basis points of adverse slippage on fills
    commission_per_trade: float = 0.0         # flat $ per round trip (entry+exit charged separately)
    position_size_pct: float = 1.0            # fraction of capital risked per trade (simple fixed-fractional)
    max_concurrent_positions: int = 1         # this engine is single-symbol; kept for future portfolio mode
    direction: str = "long"                   # 'long' or 'short' — strategy-level, not per-signal
    stop_loss_pct: Optional[float] = None      # e.g. 0.05 = 5% stop from entry
    take_profit_pct: Optional[float] = None
    max_holding_days: Optional[int] = None
    allow_reentry_next_bar: bool = True


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity_curve: pd.Series
    stats: dict = field(default_factory=dict)


def _apply_slippage(price: float, direction: str, is_entry: bool, bps: float) -> float:
    """Slippage always works against the trader."""
    factor = bps / 10_000.0
    if (direction == "long" and is_entry) or (direction == "short" and not is_entry):
        return price * (1 + factor)
    return price * (1 - factor)


def run_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    symbol: str,
    config: BacktestConfig,
) -> BacktestResult:
    """
    df: OHLCV DataFrame, DatetimeIndex ascending.
    signal: boolean Series aligned to df.index. True on bar i means
            "entry condition confirmed as of bar i's close" — the order
            fills on bar i+1 per config.execution_mode.
    """
    if not df.index.is_monotonic_increasing:
        raise ValueError("df must be sorted ascending by date — required to prevent lookahead.")
    if not signal.index.equals(df.index):
        raise ValueError("signal index must exactly match df index.")

    dates = df.index
    n = len(df)
    trades: list[Trade] = []
    open_trade: Optional[Trade] = None
    equity = config.initial_capital
    equity_curve = pd.Series(index=dates, dtype=float)
    pending_entry = False  # set True on bar i if signal[i], filled on i+1

    for i in range(n):
        date_i = dates[i]
        row = df.iloc[i]

        # --- 1. Fill a pending entry from the PREVIOUS bar's signal ---
        if pending_entry and open_trade is None:
            fill_price = row["open"] if config.execution_mode == "next_bar_open" else row["close"]
            fill_price = _apply_slippage(fill_price, config.direction, is_entry=True, bps=config.slippage_bps)
            capital_at_risk = equity * config.position_size_pct
            quantity = capital_at_risk / fill_price if fill_price > 0 else 0
            stop_price = None
            target_price = None
            if config.stop_loss_pct:
                stop_price = (fill_price * (1 - config.stop_loss_pct) if config.direction == "long"
                              else fill_price * (1 + config.stop_loss_pct))
            if config.take_profit_pct:
                target_price = (fill_price * (1 + config.take_profit_pct) if config.direction == "long"
                                else fill_price * (1 - config.take_profit_pct))
            open_trade = Trade(
                symbol=symbol, direction=config.direction, entry_date=str(date_i.date()),
                entry_price=fill_price, quantity=quantity,
                stop_price=stop_price, target_price=target_price,
                max_holding_days=config.max_holding_days,
            )
            equity -= config.commission_per_trade
            pending_entry = False

        # --- 2. Manage an open position using THIS bar's intraday range ---
        if open_trade is not None:
            exit_reason = None
            exit_price = None

            if open_trade.stop_price is not None:
                hit_stop = (row["low"] <= open_trade.stop_price if config.direction == "long"
                            else row["high"] >= open_trade.stop_price)
                if hit_stop:
                    exit_reason, exit_price = "stop", open_trade.stop_price

            if exit_reason is None and open_trade.target_price is not None:
                hit_target = (row["high"] >= open_trade.target_price if config.direction == "long"
                              else row["low"] <= open_trade.target_price)
                if hit_target:
                    exit_reason, exit_price = "target", open_trade.target_price

            if exit_reason is None and config.max_holding_days:
                held = (date_i - pd.Timestamp(open_trade.entry_date)).days
                if held >= config.max_holding_days:
                    exit_reason, exit_price = "time_exit", row["close"]

            if exit_reason is None and i == n - 1:
                exit_reason, exit_price = "end_of_data", row["close"]

            if exit_reason:
                exit_price = _apply_slippage(exit_price, config.direction, is_entry=False, bps=config.slippage_bps)
                open_trade.close(str(date_i.date()), exit_price, exit_reason)
                sign = 1 if config.direction == "long" else -1
                equity += open_trade.pnl - config.commission_per_trade
                trades.append(open_trade)
                open_trade = None

        # --- 3. Evaluate signal for a fill on the NEXT bar ---
        if open_trade is None and not pending_entry and config.allow_reentry_next_bar:
            if i < n - 1 and bool(signal.iloc[i]):
                pending_entry = True

        # mark-to-market equity (unrealized on open position valued at close)
        unrealized = 0.0
        if open_trade is not None:
            sign = 1 if config.direction == "long" else -1
            unrealized = sign * (row["close"] - open_trade.entry_price) * open_trade.quantity
        equity_curve.iloc[i] = equity + unrealized

    stats = compute_stats(trades, equity_curve, config.initial_capital)
    return BacktestResult(trades=trades, equity_curve=equity_curve, stats=stats)


def compute_stats(trades: list[Trade], equity_curve: pd.Series, initial_capital: float) -> dict:
    if not trades:
        return {
            "num_trades": 0, "total_return": 0.0, "cagr": 0.0, "sharpe": None,
            "sortino": None, "max_drawdown": 0.0, "win_rate": None,
            "profit_factor": None, "avg_trade_return": None, "payoff_ratio": None,
        }

    returns = np.array([t.pnl_pct for t in trades])
    wins = returns[returns > 0]
    losses = returns[returns <= 0]

    total_return = (equity_curve.iloc[-1] / initial_capital) - 1 if len(equity_curve) else 0.0
    n_years = max((equity_curve.index[-1] - equity_curve.index[0]).days / 365.25, 1e-6) if len(equity_curve) > 1 else 1e-6
    cagr = (1 + total_return) ** (1 / n_years) - 1 if total_return > -1 else -1.0

    daily_ret = equity_curve.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * math.sqrt(252)) if daily_ret.std() > 0 else None
    downside = daily_ret[daily_ret < 0]
    sortino = (daily_ret.mean() / downside.std() * math.sqrt(252)) if len(downside) > 1 and downside.std() > 0 else None

    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min() if len(drawdown) else 0.0

    win_rate = len(wins) / len(returns) if len(returns) else None
    gross_profit = wins.sum() if len(wins) else 0.0
    gross_loss = -losses.sum() if len(losses) else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else None)
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = -losses.mean() if len(losses) else 0.0
    payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else None

    return {
        "num_trades": len(trades),
        "total_return": float(total_return),
        "cagr": float(cagr),
        "sharpe": float(sharpe) if sharpe is not None else None,
        "sortino": float(sortino) if sortino is not None else None,
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate) if win_rate is not None else None,
        "profit_factor": float(profit_factor) if profit_factor not in (None, float("inf")) else profit_factor,
        "avg_trade_return": float(returns.mean()),
        "payoff_ratio": float(payoff_ratio) if payoff_ratio is not None else None,
    }
