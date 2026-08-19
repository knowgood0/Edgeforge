"""
Tests targeting the most dangerous failure mode: a backtester that
quietly uses information from the future. Also covers execution timing
and basic stat correctness.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestConfig, run_backtest


def make_df(prices: list[float], start="2020-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(prices), freq="B")
    df = pd.DataFrame({
        "open": prices, "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices], "close": prices,
        "adj_close": prices, "volume": [1_000_000] * len(prices),
    }, index=dates)
    return df


def test_signal_fills_next_bar_not_same_bar():
    """A signal true on bar i must NOT be filled using bar i's own price."""
    prices = [100, 100, 100, 100, 200, 100, 100]  # spike on bar index 4
    df = make_df(prices)
    signal = pd.Series([False, False, False, True, False, False, False], index=df.index)
    config = BacktestConfig(execution_mode="next_bar_open", slippage_bps=0, commission_per_trade=0,
                             position_size_pct=1.0, stop_loss_pct=None, take_profit_pct=None)
    result = run_backtest(df, signal, "TEST", config)
    assert len(result.trades) == 1
    trade = result.trades[0]
    # Signal fired on bar index 3 (price 100) -> must fill on bar index 4's open (200),
    # never on bar 3's own price.
    assert trade.entry_price == pytest.approx(200.0)
    assert trade.entry_date == str(df.index[4].date())


def test_last_bar_signal_never_fills():
    """A signal on the FINAL bar has no future bar to fill on and must
    be dropped, not filled using data that doesn't exist yet."""
    prices = [100, 101, 102, 103, 999]
    df = make_df(prices)
    signal = pd.Series([False, False, False, False, True], index=df.index)
    config = BacktestConfig()
    result = run_backtest(df, signal, "TEST", config)
    assert len(result.trades) == 0


def test_mismatched_index_raises():
    df = make_df([100, 101, 102])
    bad_signal = pd.Series([True, False], index=df.index[:2])
    with pytest.raises(ValueError):
        run_backtest(df, bad_signal, "TEST", BacktestConfig())


def test_unsorted_index_raises():
    df = make_df([100, 101, 102])
    df = df.iloc[::-1]  # reverse
    signal = pd.Series([False] * len(df), index=df.index)
    with pytest.raises(ValueError):
        run_backtest(df, signal, "TEST", BacktestConfig())


def test_stop_loss_triggers_on_intraday_low_next_bar():
    prices = [100, 100, 100, 100, 100, 100]
    df = make_df(prices)
    # widen the range on the bar right after entry so the stop is hit
    df.loc[df.index[3], "low"] = 90  # entry bar's next bar will show a big low
    signal = pd.Series([False, True, False, False, False, False], index=df.index)
    config = BacktestConfig(execution_mode="next_bar_open", slippage_bps=0, stop_loss_pct=0.05)
    result = run_backtest(df, signal, "TEST", config)
    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "stop"


def test_no_trade_when_no_signal():
    df = make_df([100] * 10)
    signal = pd.Series([False] * 10, index=df.index)
    result = run_backtest(df, signal, "TEST", BacktestConfig())
    assert len(result.trades) == 0
    assert result.stats["num_trades"] == 0


def test_commission_reduces_equity():
    prices = [100, 101, 102, 103, 104, 105]
    df = make_df(prices)
    signal = pd.Series([False, True, False, False, False, False], index=df.index)
    cfg_no_comm = BacktestConfig(commission_per_trade=0, slippage_bps=0)
    cfg_comm = BacktestConfig(commission_per_trade=50, slippage_bps=0)
    r1 = run_backtest(df, signal, "TEST", cfg_no_comm)
    r2 = run_backtest(df, signal, "TEST", cfg_comm)
    assert r2.equity_curve.iloc[-1] < r1.equity_curve.iloc[-1]


def test_slippage_is_adverse_to_trader():
    prices = [100, 100, 100, 100, 105, 100]
    df = make_df(prices)
    signal = pd.Series([False, True, False, False, False, False], index=df.index)
    cfg_no_slip = BacktestConfig(slippage_bps=0)
    cfg_slip = BacktestConfig(slippage_bps=100)  # 1%
    r1 = run_backtest(df, signal, "TEST", cfg_no_slip)
    r2 = run_backtest(df, signal, "TEST", cfg_slip)
    # entry price with slippage must be higher (worse) for a long entry
    assert r2.trades[0].entry_price > r1.trades[0].entry_price
