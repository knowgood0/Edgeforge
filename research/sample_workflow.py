"""
Sample research workflow — the one explicitly requested in the spec:

  "Search for short-term mean-reversion edges in liquid large-cap
   stocks after unusually large daily declines."

This wires together every real module (data -> features -> hypothesis
-> backtest -> validation -> scoring -> DB persistence) end to end on
whatever price data is available. It does NOT call any AI provider (so
it runs with zero API keys configured) and does NOT fabricate results:
if no price data can be loaded (e.g. no network access, or yfinance
isn't installed), it reports that plainly and stops rather than
inventing numbers.

Run with:  python -m research.sample_workflow --symbol AAPL
"""

from __future__ import annotations

import argparse
import json
import sys

from backtest.engine import BacktestConfig, run_backtest
from data.provider import get_provider, DataUnavailableError
from db.database import get_db
from research.hypothesis import build_signal_from_conditions, register_hypothesis, advance_hypothesis_status
from research.scoring import EdgeScoreInputs, compute_edge_score, status_label
from validation.validator import (
    train_test_split, run_walk_forward, parameter_sensitivity,
    transaction_cost_sensitivity, monte_carlo_trade_shuffle, deflated_sharpe_ratio,
)


HYPOTHESIS_STATEMENT = (
    "Large-cap stocks that decline >3% in a single day while trading above "
    "their 200-day SMA tend to exhibit short-term mean reversion over the "
    "following 5 trading days."
)


def make_conditions(decline_pct: float = -0.03):
    return [
        {"feature": "returns_1d", "params": {}, "op": "<", "value": decline_pct},
        {"feature": "price_vs_ma_pct", "params": {"period": 200, "ma_type": "sma"}, "op": ">", "value": 0},
    ]


def run(symbol: str, start: str = "2015-01-01", end: str = "2025-01-01") -> dict:
    db = get_db()
    provider = get_provider("yfinance")

    print(f"[1/6] Fetching data for {symbol}...")
    try:
        df = provider.get_daily_bars(symbol, start, end)
    except DataUnavailableError as e:
        return {
            "status": "stopped",
            "reason": (
                f"Could not fetch price data: {e}. This sandbox has no outbound "
                "network access, so this step must be run on your actual "
                "deployment (Render, local machine, etc.) where yfinance can "
                "reach the network. No results were fabricated."
            ),
        }

    issues = provider.check_data_quality(df, symbol)
    for issue in issues:
        db.execute(
            """INSERT INTO data_quality_log (security_id, check_type, date, detail, severity)
               VALUES ((SELECT id FROM securities WHERE symbol = ?), ?, ?, ?, ?)""",
            (symbol, issue["check_type"], issue["date"], issue["detail"], issue["severity"]),
        ) if db.query_one("SELECT id FROM securities WHERE symbol = ?", (symbol,)) else None

    sec_id = db.get_or_create_security(symbol)

    print("[2/6] Registering hypothesis (checking research memory for duplicates)...")
    reg = register_hypothesis(db, HYPOTHESIS_STATEMENT, origin="user")
    hyp_id = reg["hypothesis_id"]
    if reg["similar_to"]:
        print(f"  -> Similar to hypothesis #{reg['similar_to']['id']} "
              f"(similarity={reg['similar_to']['similarity']}, status={reg['similar_to']['status']})")

    conditions = make_conditions()
    signal_fn = build_signal_from_conditions(conditions, logic="and")
    signal = signal_fn(df)
    n_signals = int(signal.sum())
    print(f"[3/6] Signal fires on {n_signals} of {len(df)} bars.")

    if n_signals < 5:
        advance_hypothesis_status(db, hyp_id, "rejected", detail="insufficient sample size")
        return {"status": "rejected", "reason": f"Only {n_signals} signal occurrences — insufficient sample size."}

    config = BacktestConfig(
        initial_capital=100_000, execution_mode="next_bar_open", slippage_bps=5,
        commission_per_trade=1.0, position_size_pct=0.1, direction="long",
        max_holding_days=5, stop_loss_pct=0.08,
    )

    train_df, test_df = train_test_split(df, train_frac=0.7)
    train_signal = signal.loc[train_df.index]
    test_signal = signal.loc[test_df.index]

    print("[4/6] Running in-sample and out-of-sample backtests...")
    train_result = run_backtest(train_df, train_signal, symbol, config)
    test_result = run_backtest(test_df, test_signal, symbol, config)

    print("[5/6] Running validation suite (walk-forward, param sensitivity, "
          "transaction-cost sensitivity, Monte Carlo)...")

    wf = run_walk_forward(df, signal_fn, symbol, config, train_days=252, test_days=63)

    def build_sig_at_decline(d, decline_val):
        return build_signal_from_conditions(make_conditions(decline_val), logic="and")(d)

    param_sweep = parameter_sensitivity(
        df, build_sig_at_decline, symbol, config,
        param_values=[-0.02, -0.025, -0.03, -0.035, -0.04],
    )

    cost_sens = transaction_cost_sensitivity(df, signal, symbol, config)
    cost_returns = [r["total_return"] for r in cost_sens["sweep"]]

    trade_returns = [t.pnl_pct for t in test_result.trades]
    mc = monte_carlo_trade_shuffle(trade_returns)

    # Honest multiple-testing accounting: this sample workflow tested 1
    # hypothesis with 1 parameter sweep of 5 values = 5 effective trials.
    n_trials_tested = 1 + len(param_sweep["sweep"])
    dsr = deflated_sharpe_ratio(
        observed_sharpe=test_result.stats.get("sharpe"),
        n_trials=n_trials_tested,
        n_returns=len(trade_returns),
    )

    print("[6/6] Computing Edge Score...")
    inputs = EdgeScoreInputs(
        oos_sharpe=test_result.stats.get("sharpe"),
        walk_forward_pct_profitable=wf["summary"]["pct_windows_profitable"] if wf["summary"] else None,
        walk_forward_sharpe_std=wf["summary"]["sharpe_std"] if wf["summary"] else None,
        walk_forward_avg_sharpe=wf["summary"]["avg_sharpe"] if wf["summary"] else None,
        num_trades=test_result.stats.get("num_trades", 0),
        parameter_stability_score=param_sweep["stability_score"],
        cross_security_pct_positive=None,  # single-symbol run; cross-security test is a separate step
        cross_security_sharpe_std=None,
        regime_results=None,  # TODO: not implemented in this sample workflow yet
        cost_sweep_returns=cost_returns,
        deflated_sharpe_prob=dsr,
    )
    edge = compute_edge_score(inputs)

    status = "survived_oos" if test_result.stats.get("total_return", 0) > 0 else "rejected"
    advance_hypothesis_status(db, hyp_id, status, detail=json.dumps(edge))

    db.execute(
        """INSERT INTO edge_scores (hypothesis_id, score, components_json, deflated_sharpe,
           probability_backtest_overfitting, false_discovery_rate_est)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (hyp_id, edge["score"], json.dumps(edge["components"]), dsr, None, None),
    )

    result = {
        "status": "complete",
        "hypothesis_id": hyp_id,
        "symbol": symbol,
        "signal_occurrences": n_signals,
        "in_sample_stats": train_result.stats,
        "out_of_sample_stats": test_result.stats,
        "walk_forward_summary": wf["summary"],
        "parameter_sensitivity": param_sweep,
        "transaction_cost_sensitivity": cost_sens,
        "monte_carlo": mc,
        "deflated_sharpe_ratio_prob_positive": dsr,
        "n_effective_trials_for_dsr": n_trials_tested,
        "edge_score": edge,
        "edge_score_label": status_label(edge["score"]),
        "note": "Cross-security and market-regime tests are not run in this "
                "single-symbol sample workflow — run them separately via "
                "validation.validator.cross_security_test before treating "
                "this as a broadly validated edge.",
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-01-01")
    args = parser.parse_args()

    output = run(args.symbol, args.start, args.end)
    print(json.dumps(output, indent=2, default=str))
    if output.get("status") == "stopped":
        sys.exit(1)
