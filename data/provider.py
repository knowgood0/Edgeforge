"""
Data provider abstraction.

DataProvider defines the interface every market-data source must implement.
YFinanceProvider is the initial concrete implementation (free, no API key,
good enough for daily-bar research). Swapping in a paid vendor later
(Polygon, IEX, Norgate, etc.) means writing a new class here — nothing
upstream should need to change.

This module deliberately does NOT silently fill gaps or pretend data is
clean. Every provider must run data-quality checks and log them via
Database.execute against data_quality_log, per EdgeForge's "no fake
results" rule: if data is missing or suspect, say so.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd


@dataclass
class Bar:
    date: str
    open: float
    high: float
    low: float
    close: float
    adj_close: Optional[float]
    volume: Optional[int]


class DataProvider(abc.ABC):
    """Abstract base for any historical-price data source."""

    name: str = "abstract"

    @abc.abstractmethod
    def get_daily_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Return a DataFrame indexed by date with columns:
        open, high, low, close, adj_close, volume.
        Must raise DataUnavailableError (not return fake/empty-but-silent
        data) if the symbol/range can't be fetched.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_corporate_actions(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Return splits/dividends with columns: date, action_type, detail."""
        raise NotImplementedError

    def check_data_quality(self, df: pd.DataFrame, symbol: str) -> list[dict]:
        """Generic sanity checks any provider's data should pass through.
        Returns a list of {check_type, date, detail, severity} dicts —
        callers are responsible for persisting these to data_quality_log.
        """
        issues = []
        if df.empty:
            issues.append({
                "check_type": "no_data", "date": None,
                "detail": f"No bars returned for {symbol}", "severity": "critical",
            })
            return issues

        # Missing trading days (rough check: gaps > 4 calendar days on weekdays)
        idx = pd.to_datetime(df.index)
        gaps = idx.to_series().diff().dt.days
        for d, gap in zip(idx[1:], gaps[1:]):
            if gap and gap > 4:
                issues.append({
                    "check_type": "missing_bar", "date": str(d.date()),
                    "detail": f"{int(gap)}-day gap before this date", "severity": "warning",
                })

        # Extreme single-day moves that might indicate a bad print or
        # unadjusted split
        ret = df["close"].pct_change()
        spikes = ret[(ret.abs() > 0.5)]
        for d, r in spikes.items():
            issues.append({
                "check_type": "price_spike", "date": str(pd.Timestamp(d).date()),
                "detail": f"{r:.1%} single-day move — check for unadjusted split/data error",
                "severity": "warning",
            })

        # Zero or negative prices/volume
        bad_price = df[(df[["open", "high", "low", "close"]] <= 0).any(axis=1)]
        for d in bad_price.index:
            issues.append({
                "check_type": "invalid_price", "date": str(pd.Timestamp(d).date()),
                "detail": "Non-positive OHLC value", "severity": "critical",
            })

        return issues


class DataUnavailableError(Exception):
    pass


class YFinanceProvider(DataProvider):
    """Default provider. Requires the `yfinance` package and outbound
    network access at runtime (not available inside this dev sandbox —
    this class is written and unit-testable, but live fetches must be
    exercised on your actual deployment, e.g. Render, where network
    access exists)."""

    name = "yfinance"

    def get_daily_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as e:
            raise DataUnavailableError(
                "yfinance is not installed. Run: pip install yfinance"
            ) from e

        try:
            raw = yf.download(
                symbol, start=start, end=end, auto_adjust=False,
                progress=False, threads=False,
            )
        except Exception as e:
            raise DataUnavailableError(f"yfinance fetch failed for {symbol}: {e}") from e

        if raw is None or raw.empty:
            raise DataUnavailableError(f"No data returned for {symbol} in [{start}, {end}]")

        # yfinance sometimes returns MultiIndex columns for single-symbol
        # requests depending on version; normalize.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] for c in raw.columns]

        df = raw.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
        })[["open", "high", "low", "close", "adj_close", "volume"]]
        df.index.name = "date"
        return df

    def get_corporate_actions(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as e:
            raise DataUnavailableError(
                "yfinance is not installed. Run: pip install yfinance"
            ) from e

        ticker = yf.Ticker(symbol)
        actions = ticker.actions  # DataFrame with Dividends, Stock Splits columns
        if actions is None or actions.empty:
            return pd.DataFrame(columns=["date", "action_type", "detail"])

        actions = actions.loc[(actions.index >= start) & (actions.index <= end)]
        rows = []
        for dt, row in actions.iterrows():
            if row.get("Dividends", 0):
                rows.append({"date": str(dt.date()), "action_type": "dividend",
                              "detail": str(row["Dividends"])})
            if row.get("Stock Splits", 0):
                rows.append({"date": str(dt.date()), "action_type": "split",
                              "detail": str(row["Stock Splits"])})
        return pd.DataFrame(rows)


def get_provider(name: str = "yfinance") -> DataProvider:
    if name == "yfinance":
        return YFinanceProvider()
    raise ValueError(f"Unknown data provider: {name}")
