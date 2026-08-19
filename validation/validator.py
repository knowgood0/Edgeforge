"""
Validation suite.

These functions turn a single promising backtest into evidence about
whether the effect is likely to be real. Nothing here "passes" or
"fails" a strategy on its own — they compute metrics; the Edge Score
(research/scoring.py) and the AI Skeptic role interpret them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from backtest.engine import BacktestConfig, run_backtest, compute_stats


def train_test_split(df: pd.DataFrame, train_frac: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simple chronological split. Never shuffle time series data —
    that alone would introduce lookahead bias."""
    split_idx = int(len(df) * train_frac)
    return df.iloc[:split_idx], df.iloc[split_idx:]


def walk_forward_windows(df: pd.DataFrame, train_days: int, test_days: int,
                          step_days: Optional[int] = None) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Rolling-origin walk-forward windows. Each (train, test) pair is
    strictly chronological — test always follows train in time."""
    step = step_days or test_days
    windows = []
    start = 0
    while start + train_days + test_days <= len(df):
        train = df.iloc[start:start + train_days]
        test = df.iloc[start + train_days:start + train_days + test_days]
        windows.append((train, test))
        start += step
    return windows


def run_walk_forward(
    df: pd.DataFrame,
    signal_fn: Callable[[pd.DataFrame], pd.Series],
    symbol: str,
    config: BacktestConfig,
    train_days: int = 252,
    test_days: int = 63,
) -> dict:
    """signal_fn takes a df slice and returns the boolean signal Series
    for it (so parameters can, in principle, be re-fit per training
    window — for a fixed-rule strategy, signal_fn just recomputes the
    same rule on each window's data)."""
    windows = walk_forward_windows(df, train_days, test_days)
    if not windows:
        return {"windows": [], "summary": None, "note": "Not enough data for requested window sizes."}

    results = []
    for train_df, test_df in windows:
        sig = signal_fn(test_df)
        res = run_backtest(test_df, sig, symbol, config)
        results.append({
            "test_start": str(test_df.index[0].date()),
            "test_end": str(test_df.index[-1].date()),
            "stats": res.stats,
        })

    sharpes = [r["stats"]["sharpe"] for r in results if r["stats"]["sharpe"] is not None]
    returns = [r["stats"]["total_return"] for r in results]
    summary = {
        "num_windows": len(results),
        "pct_windows_profitable": float(np.mean([r > 0 for r in returns])) if returns else None,
        "avg_sharpe": float(np.mean(sharpes)) if sharpes else None,
        "sharpe_std": float(np.std(sharpes)) if len(sharpes) > 1 else None,
        "avg_return": float(np.mean(returns)) if returns else None,
    }
    return {"windows": results, "summary": summary}


def parameter_sensitivity(
    df: pd.DataFrame,
    build_signal_fn: Callable[[pd.DataFrame, float], pd.Series],
    symbol: str,
    config: BacktestConfig,
    param_values: list[float],
) -> dict:
    """Sweeps ONE parameter across nearby values and reports how much
    performance degrades. A cliff (only the exact optimized value works)
    is the classic overfitting signature; a plateau is reassuring."""
    rows = []
    for pv in param_values:
        sig = build_signal_fn(df, pv)
        res = run_backtest(df, sig, symbol, config)
        rows.append({"param_value": pv, **res.stats})

    sharpes = [r["sharpe"] for r in rows if r["sharpe"] is not None]
    stability = None
    if len(sharpes) > 1 and np.mean(np.abs(sharpes)) > 1e-9:
        stability = 1 - (np.std(sharpes) / (np.mean(np.abs(sharpes)) + 1e-9))
    return {"sweep": rows, "stability_score": stability}


def cross_security_test(
    price_data: dict[str, pd.DataFrame],
    signal_fn: Callable[[pd.DataFrame], pd.Series],
    config: BacktestConfig,
) -> dict:
    """Applies the IDENTICAL rule (no per-symbol re-optimization) across
    multiple securities to see whether an effect generalizes or is
    idiosyncratic to the symbol it was discovered on."""
    results = {}
    for symbol, df in price_data.items():
        sig = signal_fn(df)
        res = run_backtest(df, sig, symbol, config)
        results[symbol] = res.stats

    sharpes = [r["sharpe"] for r in results.values() if r["sharpe"] is not None]
    positive = [r for r in results.values() if r["total_return"] and r["total_return"] > 0]
    return {
        "per_symbol": results,
        "num_symbols_tested": len(results),
        "num_symbols_positive": len(positive),
        "pct_symbols_positive": len(positive) / len(results) if results else None,
        "avg_sharpe_across_symbols": float(np.mean(sharpes)) if sharpes else None,
        "sharpe_std_across_symbols": float(np.std(sharpes)) if len(sharpes) > 1 else None,
    }


def monte_carlo_trade_shuffle(trade_returns: list[float], n_simulations: int = 2000,
                               seed: int = 42) -> dict:
    """Resamples the trade sequence (bootstrap with replacement) to
    estimate a distribution of plausible outcomes, rather than trusting
    the single realized equity curve's path dependency."""
    if len(trade_returns) < 5:
        return {"note": "Too few trades for meaningful Monte Carlo analysis (need >=5).", "n_trades": len(trade_returns)}

    rng = np.random.default_rng(seed)
    arr = np.array(trade_returns)
    final_returns = []
    max_drawdowns = []
    for _ in range(n_simulations):
        sample = rng.choice(arr, size=len(arr), replace=True)
        equity = np.cumprod(1 + sample)
        final_returns.append(equity[-1] - 1)
        running_max = np.maximum.accumulate(equity)
        dd = (equity - running_max) / running_max
        max_drawdowns.append(dd.min())

    final_returns = np.array(final_returns)
    return {
        "n_simulations": n_simulations,
        "median_return": float(np.median(final_returns)),
        "return_5th_pctile": float(np.percentile(final_returns, 5)),
        "return_95th_pctile": float(np.percentile(final_returns, 95)),
        "pct_simulations_profitable": float(np.mean(final_returns > 0)),
        "median_max_drawdown": float(np.median(max_drawdowns)),
        "worst_5pct_drawdown": float(np.percentile(max_drawdowns, 5)),
    }


def transaction_cost_sensitivity(
    df: pd.DataFrame, signal: pd.Series, symbol: str, base_config: BacktestConfig,
    slippage_multipliers: Optional[list[float]] = None,
) -> dict:
    """Re-runs the backtest at escalating cost assumptions. If the edge
    only survives at unrealistically low costs, that's disqualifying,
    not a footnote."""
    multipliers = slippage_multipliers or [0.0, 1.0, 2.0, 4.0]
    rows = []
    for m in multipliers:
        cfg = BacktestConfig(**{**base_config.__dict__, "slippage_bps": base_config.slippage_bps * m})
        res = run_backtest(df, signal, symbol, cfg)
        rows.append({"slippage_multiplier": m, "slippage_bps": cfg.slippage_bps, **res.stats})
    return {"sweep": rows}


def deflated_sharpe_ratio(observed_sharpe: float, n_trials: int, n_returns: int,
                           skew: float = 0.0, kurtosis: float = 3.0) -> Optional[float]:
    """Bailey & Lopez de Prado's deflated Sharpe ratio: the probability
    that the observed Sharpe is genuinely positive after accounting for
    how many strategy variants were searched over (n_trials) and the
    non-normality of returns. This is the core defense against
    multiple-testing false discoveries — a spectacular Sharpe found
    among 100,000 tested variants means much less than the same Sharpe
    found in a single, pre-registered test.
    """
    if observed_sharpe is None or n_returns < 10 or n_trials < 1:
        return None
    try:
        from scipy.stats import norm
    except ImportError:
        return None

    # Expected max Sharpe under the null across n_trials independent trials
    # (approximation from Bailey & Lopez de Prado 2014).
    euler_mascheroni = 0.5772156649
    if n_trials > 1:
        expected_max_sr = (1 - euler_mascheroni) * norm.ppf(1 - 1 / n_trials) + \
                           euler_mascheroni * norm.ppf(1 - 1 / (n_trials * math.e))
        expected_max_sr /= math.sqrt(n_returns)
    else:
        expected_max_sr = 0.0

    sr_std = math.sqrt(max(
        (1 - skew * observed_sharpe + ((kurtosis - 1) / 4) * observed_sharpe ** 2) / (n_returns - 1),
        1e-12,
    ))
    z = (observed_sharpe - expected_max_sr) / sr_std
    return float(norm.cdf(z))


def probability_of_backtest_overfitting_note() -> str:
    return (
        "Full Combinatorially Symmetric Cross-Validation (CSCV / PBO per "
        "Bailey, Borwein, Lopez de Prado & Zhu 2015) requires many "
        "train/test partitions of the SAME strategy-selection process and "
        "is intentionally not faked here with a placeholder number. It "
        "should be run whenever a strategy has survived a parameter search "
        "(research/optimizer.py), using the actual set of candidate "
        "parameterizations that were tried — not implemented yet in this "
        "MVP. Use deflated_sharpe_ratio() with an honest n_trials count "
        "as the interim multiple-testing safeguard."
    )
