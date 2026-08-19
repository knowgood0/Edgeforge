"""
Feature engineering library.

Every function takes a price DataFrame (columns: open, high, low, close,
adj_close, volume; DatetimeIndex) and returns a pandas Series aligned to
that index. Functions must only use data available *as of* each row's
date (no centered rolling windows, no future leakage) — this is what
makes them safe to feed into the backtester later without accidentally
handing the strategy tomorrow's information.

FEATURE_REGISTRY maps feature name -> (category, callable, default_params)
so the research layer can enumerate what's available and the AI can
propose new entries at runtime (persisted via
Database.get_or_create_feature_definition with origin='ai_proposed').
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

FEATURE_REGISTRY: dict[str, dict] = {}


def register(name: str, category: str, params: dict | None = None):
    def deco(fn: Callable):
        FEATURE_REGISTRY[name] = {"category": category, "fn": fn, "params": params or {}}
        return fn
    return deco


# ------------------------------------------------------------------
# PRICE
# ------------------------------------------------------------------

@register("returns_1d", "price")
def returns_1d(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change()


@register("log_returns_1d", "price")
def log_returns_1d(df: pd.DataFrame) -> pd.Series:
    return np.log(df["close"] / df["close"].shift(1))


@register("gap_pct", "price")
def gap_pct(df: pd.DataFrame) -> pd.Series:
    """Overnight gap: today's open vs. yesterday's close."""
    return (df["open"] - df["close"].shift(1)) / df["close"].shift(1)


@register("range_pct", "price")
def range_pct(df: pd.DataFrame) -> pd.Series:
    """Intraday range as % of close."""
    return (df["high"] - df["low"]) / df["close"]


def multi_day_return(df: pd.DataFrame, n: int) -> pd.Series:
    return df["close"].pct_change(n)


# ------------------------------------------------------------------
# TREND
# ------------------------------------------------------------------

def sma(df: pd.DataFrame, period: int) -> pd.Series:
    return df["close"].rolling(period).mean()


def ema(df: pd.DataFrame, period: int) -> pd.Series:
    return df["close"].ewm(span=period, adjust=False).mean()


def wma(df: pd.DataFrame, period: int) -> pd.Series:
    weights = np.arange(1, period + 1)
    return df["close"].rolling(period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def price_vs_ma_pct(df: pd.DataFrame, period: int, ma_type: str = "sma") -> pd.Series:
    ma = sma(df, period) if ma_type == "sma" else ema(df, period)
    return (df["close"] - ma) / ma


def ma_crossover(df: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    """+1 if fast MA above slow MA, -1 otherwise."""
    f, s = sma(df, fast), sma(df, slow)
    return np.sign(f - s)


def ma_slope(df: pd.DataFrame, period: int, lookback: int = 5) -> pd.Series:
    """Rate of change of the moving average itself (trend steepness)."""
    m = sma(df, period)
    return m.pct_change(lookback)


# ------------------------------------------------------------------
# MOMENTUM
# ------------------------------------------------------------------

def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.where(avg_loss != 0, 100.0)


def roc(df: pd.DataFrame, period: int = 10) -> pd.Series:
    return df["close"].pct_change(period) * 100


def stochastic_k(df: pd.DataFrame, period: int = 14) -> pd.Series:
    low_min = df["low"].rolling(period).min()
    high_max = df["high"].rolling(period).max()
    return 100 * (df["close"] - low_min) / (high_max - low_min)


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_max = df["high"].rolling(period).max()
    low_min = df["low"].rolling(period).min()
    return -100 * (high_max - df["close"]) / (high_max - low_min)


def rsi_rate_of_change(df: pd.DataFrame, rsi_period: int = 14, roc_period: int = 3) -> pd.Series:
    """Derived: how fast RSI itself is moving. Example of a second-order
    feature the AI research layer is encouraged to generate more of."""
    r = rsi(df, rsi_period)
    return r.diff(roc_period)


# ------------------------------------------------------------------
# VOLATILITY
# ------------------------------------------------------------------

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def bollinger_pct_b(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.Series:
    mid = sma(df, period)
    std = df["close"].rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return (df["close"] - lower) / (upper - lower)


def rolling_volatility(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Annualized close-to-close volatility."""
    return returns_1d(df).rolling(period).std() * np.sqrt(252)


def volatility_percentile(df: pd.DataFrame, period: int = 20, lookback: int = 252) -> pd.Series:
    """Where current rolling vol sits within its own trailing history —
    lets a strategy condition on 'volatility is unusually high/low for
    THIS security', not just an absolute threshold."""
    vol = rolling_volatility(df, period)
    return vol.rolling(lookback).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x.dropna()) > 1 else np.nan,
        raw=False,
    )


def distance_from_ma_in_atr(df: pd.DataFrame, ma_period: int = 20, atr_period: int = 14) -> pd.Series:
    """Derived feature: distance from moving average expressed in ATR
    units, rather than raw % — normalizes across securities with very
    different volatility regimes."""
    m = sma(df, ma_period)
    a = atr(df, atr_period)
    return (df["close"] - m) / a.replace(0, np.nan)


# ------------------------------------------------------------------
# VOLUME
# ------------------------------------------------------------------

def relative_volume(df: pd.DataFrame, period: int = 20) -> pd.Series:
    avg_vol = df["volume"].rolling(period).mean()
    return df["volume"] / avg_vol


def volume_percentile(df: pd.DataFrame, lookback: int = 252) -> pd.Series:
    return df["volume"].rolling(lookback).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x.dropna()) > 1 else np.nan,
        raw=False,
    )


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum()


def volume_relative_to_volatility(df: pd.DataFrame, vol_period: int = 20, atr_period: int = 14) -> pd.Series:
    """Derived: relative volume normalized by relative range —
    distinguishes 'high volume, high range' (trend day) from
    'high volume, low range' (accumulation/absorption)."""
    rv = relative_volume(df, vol_period)
    rng = true_range(df) / atr(df, atr_period)
    return rv / rng.replace(0, np.nan)


# ------------------------------------------------------------------
# MARKET CONTEXT (requires a second DataFrame, e.g. SPY)
# ------------------------------------------------------------------

def excess_return_vs_market(df: pd.DataFrame, market_df: pd.DataFrame, period: int = 1) -> pd.Series:
    """Security's return minus the market's return over the same window —
    isolates idiosyncratic behavior from broad market moves."""
    sec_ret = df["close"].pct_change(period)
    mkt_ret = market_df["close"].reindex(df.index).pct_change(period)
    return sec_ret - mkt_ret


def relative_strength_vs_market(df: pd.DataFrame, market_df: pd.DataFrame, period: int = 20) -> pd.Series:
    sec_cum = df["close"] / df["close"].shift(period)
    mkt_cum = market_df["close"].reindex(df.index) / market_df["close"].reindex(df.index).shift(period)
    return sec_cum / mkt_cum


# ------------------------------------------------------------------
# CALENDAR
# ------------------------------------------------------------------

def day_of_week(df: pd.DataFrame) -> pd.Series:
    return pd.Series(df.index.dayofweek, index=df.index)


def month_of_year(df: pd.DataFrame) -> pd.Series:
    return pd.Series(df.index.month, index=df.index)


def is_month_end_window(df: pd.DataFrame, window: int = 3) -> pd.Series:
    """1 if within `window` trading days of month end (rebalancing flows)."""
    is_end = df.index.to_series().groupby([df.index.year, df.index.month]).transform(
        lambda x: x.rank(ascending=False) <= window
    )
    return is_end.astype(int)


# ------------------------------------------------------------------
# ORCHESTRATION
# ------------------------------------------------------------------

def compute_feature(name: str, df: pd.DataFrame, market_df: pd.DataFrame | None = None,
                     **params) -> pd.Series:
    """Dispatch by name for features not in the simple no-arg registry
    (most take a `period` or similar). Kept explicit rather than fully
    dynamic so parameter mistakes fail loudly instead of silently."""
    dispatch = {
        "sma": lambda: sma(df, params.get("period", 20)),
        "ema": lambda: ema(df, params.get("period", 20)),
        "wma": lambda: wma(df, params.get("period", 20)),
        "price_vs_ma_pct": lambda: price_vs_ma_pct(df, params.get("period", 20), params.get("ma_type", "sma")),
        "ma_crossover": lambda: ma_crossover(df, params.get("fast", 10), params.get("slow", 50)),
        "ma_slope": lambda: ma_slope(df, params.get("period", 20), params.get("lookback", 5)),
        "rsi": lambda: rsi(df, params.get("period", 14)),
        "roc": lambda: roc(df, params.get("period", 10)),
        "stochastic_k": lambda: stochastic_k(df, params.get("period", 14)),
        "williams_r": lambda: williams_r(df, params.get("period", 14)),
        "rsi_rate_of_change": lambda: rsi_rate_of_change(df, params.get("rsi_period", 14), params.get("roc_period", 3)),
        "atr": lambda: atr(df, params.get("period", 14)),
        "bollinger_pct_b": lambda: bollinger_pct_b(df, params.get("period", 20), params.get("num_std", 2.0)),
        "rolling_volatility": lambda: rolling_volatility(df, params.get("period", 20)),
        "volatility_percentile": lambda: volatility_percentile(df, params.get("period", 20), params.get("lookback", 252)),
        "distance_from_ma_in_atr": lambda: distance_from_ma_in_atr(df, params.get("ma_period", 20), params.get("atr_period", 14)),
        "relative_volume": lambda: relative_volume(df, params.get("period", 20)),
        "volume_percentile": lambda: volume_percentile(df, params.get("lookback", 252)),
        "obv": lambda: obv(df),
        "volume_relative_to_volatility": lambda: volume_relative_to_volatility(df, params.get("vol_period", 20), params.get("atr_period", 14)),
        "multi_day_return": lambda: multi_day_return(df, params.get("n", 5)),
    }
    if name in FEATURE_REGISTRY:
        return FEATURE_REGISTRY[name]["fn"](df)
    if name in dispatch:
        return dispatch[name]()
    if market_df is not None:
        market_dispatch = {
            "excess_return_vs_market": lambda: excess_return_vs_market(df, market_df, params.get("period", 1)),
            "relative_strength_vs_market": lambda: relative_strength_vs_market(df, market_df, params.get("period", 20)),
        }
        if name in market_dispatch:
            return market_dispatch[name]()
    raise ValueError(f"Unknown feature: {name}")


def compute_feature_matrix(df: pd.DataFrame, feature_specs: list[dict],
                            market_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """feature_specs: list of {"name": str, "params": dict, "output_name": str}
    Returns a DataFrame with one column per requested feature, aligned to df.index.
    """
    out = {}
    for spec in feature_specs:
        col = spec.get("output_name", spec["name"])
        out[col] = compute_feature(spec["name"], df, market_df=market_df, **spec.get("params", {}))
    return pd.DataFrame(out, index=df.index)
