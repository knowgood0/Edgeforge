import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from features import engineering as feat


def make_df(prices, highs=None, lows=None, volumes=None):
    dates = pd.date_range("2020-01-01", periods=len(prices), freq="B")
    return pd.DataFrame({
        "open": prices,
        "high": highs or [p * 1.02 for p in prices],
        "low": lows or [p * 0.98 for p in prices],
        "close": prices,
        "adj_close": prices,
        "volume": volumes or [1_000_000] * len(prices),
    }, index=dates)


def test_sma_matches_manual_calc():
    prices = [10, 11, 12, 13, 14, 15]
    df = make_df(prices)
    result = feat.sma(df, 3)
    # SMA(3) at index 2 = mean(10,11,12) = 11
    assert result.iloc[2] == pytest.approx(11.0)
    assert pd.isna(result.iloc[1])  # not enough data yet


def test_rsi_bounds():
    np.random.seed(0)
    prices = 100 + np.cumsum(np.random.randn(200))
    df = make_df(list(prices))
    r = feat.rsi(df, 14)
    valid = r.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_rsi_all_gains_approaches_100():
    prices = list(range(100, 150))  # strictly increasing
    df = make_df(prices)
    r = feat.rsi(df, 14)
    assert r.iloc[-1] > 95


def test_returns_1d_no_future_leakage():
    """Feature at index i must only use data through index i."""
    prices = [100, 105, 95, 120, 80]
    df = make_df(prices)
    r = feat.returns_1d(df)
    # Truncating the df to i+1 rows must give the identical value at i
    for i in range(1, len(df)):
        truncated = df.iloc[: i + 1]
        r_trunc = feat.returns_1d(truncated)
        assert r_trunc.iloc[-1] == pytest.approx(r.iloc[i])


def test_atr_no_future_leakage():
    np.random.seed(1)
    prices = list(100 + np.cumsum(np.random.randn(60)))
    df = make_df(prices)
    full = feat.atr(df, 14)
    truncated = feat.atr(df.iloc[:40], 14)
    # value at index 39 should be identical whether or not later rows exist
    assert full.iloc[39] == pytest.approx(truncated.iloc[-1], rel=1e-9)


def test_bollinger_pct_b_no_future_leakage():
    np.random.seed(2)
    prices = list(100 + np.cumsum(np.random.randn(60)))
    df = make_df(prices)
    full = feat.bollinger_pct_b(df, 20)
    truncated = feat.bollinger_pct_b(df.iloc[:45], 20)
    assert full.iloc[44] == pytest.approx(truncated.iloc[-1], rel=1e-9)


def test_compute_feature_matrix_shapes():
    df = make_df(list(100 + np.cumsum(np.random.randn(60))))
    specs = [
        {"name": "rsi", "params": {"period": 14}, "output_name": "rsi_14"},
        {"name": "sma", "params": {"period": 20}, "output_name": "sma_20"},
    ]
    matrix = feat.compute_feature_matrix(df, specs)
    assert list(matrix.columns) == ["rsi_14", "sma_20"]
    assert len(matrix) == len(df)


def test_unknown_feature_raises():
    df = make_df([1, 2, 3])
    with pytest.raises(ValueError):
        feat.compute_feature("not_a_real_feature", df)
