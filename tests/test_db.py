import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from db.database import Database
from research.hypothesis import register_hypothesis, find_similar_hypothesis, reject_hypothesis, advance_hypothesis_status
from research.scoring import EdgeScoreInputs, compute_edge_score, status_label
from pinescript.generator import generate_pine_script, UnsupportedStrategyError


@pytest.fixture
def db():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        yield Database(path=f.name)


def test_schema_creates_all_core_tables(db):
    tables = {r["name"] for r in db.query(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    for expected in ["securities", "price_bars", "hypotheses", "experiments",
                      "backtest_runs", "validation_results", "edge_scores",
                      "strategies", "strategy_versions", "pine_scripts",
                      "paper_trades", "research_log", "data_quality_log"]:
        assert expected in tables


def test_get_or_create_security_is_idempotent(db):
    id1 = db.get_or_create_security("AAPL", name="Apple Inc.")
    id2 = db.get_or_create_security("AAPL")
    assert id1 == id2


def test_register_hypothesis_and_duplicate_detection(db):
    r1 = register_hypothesis(db, "RSI below 30 predicts mean reversion in large caps")
    assert r1["similar_to"] is None

    reject_hypothesis(db, r1["hypothesis_id"], "failed OOS test, driven entirely by 2020-2021")

    r2 = register_hypothesis(db, "RSI below 30 predicts mean reversion in large cap stocks")
    assert r2["similar_to"] is not None
    assert r2["similar_to"]["id"] == r1["hypothesis_id"]
    assert r2["similar_to"]["status"] == "rejected"


def test_advance_hypothesis_status_rejects_invalid(db):
    r = register_hypothesis(db, "test hypothesis")
    with pytest.raises(ValueError):
        advance_hypothesis_status(db, r["hypothesis_id"], "not_a_real_status")


def test_research_stats_counts_correctly(db):
    h1 = register_hypothesis(db, "hypothesis one")["hypothesis_id"]
    h2 = register_hypothesis(db, "hypothesis two")["hypothesis_id"]
    advance_hypothesis_status(db, h1, "survived_oos")
    reject_hypothesis(db, h2, "no edge found")

    stats = db.research_stats()
    assert stats["hypotheses_tested"] == 2
    assert stats["survived_oos_testing"] == 1
    assert stats["rejected"] == 1


def test_edge_score_bounds():
    inputs = EdgeScoreInputs(
        oos_sharpe=1.8, walk_forward_pct_profitable=0.9, walk_forward_sharpe_std=0.1,
        walk_forward_avg_sharpe=1.5, num_trades=150, parameter_stability_score=0.9,
        cross_security_pct_positive=0.85, cross_security_sharpe_std=0.2,
        regime_results={"bull": 1.2, "bear": 0.3}, cost_sweep_returns=[0.2, 0.18, 0.14, 0.05],
        deflated_sharpe_prob=0.95,
    )
    result = compute_edge_score(inputs)
    assert 0 <= result["score"] <= 100
    assert result["score"] > 70  # strong inputs should score well
    assert status_label(result["score"]) == "Promising"


def test_edge_score_weak_inputs_score_low():
    inputs = EdgeScoreInputs(oos_sharpe=-0.5, num_trades=8)
    result = compute_edge_score(inputs)
    assert result["score"] < 35
    assert status_label(result["score"]) == "Not Supported"


def test_pinescript_generation_basic():
    definition = {
        "entry": {
            "direction": "long",
            "logic": "and",
            "conditions": [
                {"indicator": "rsi", "params": {"period": 14}, "op": "<", "value": 30},
                {"indicator": "price_vs_sma", "params": {"period": 200}, "op": ">", "value": 0},
            ],
        },
        "exit": {"stop_loss_pct": 0.05, "take_profit_pct": 0.10, "max_holding_days": 10},
        "position_sizing": {"pct_of_equity": 10},
    }
    script = generate_pine_script("Test Strategy", 1, definition)
    assert "//@version=5" in script
    assert "ta.rsi(close, 14)" in script
    assert "strategy.entry" in script
    assert "Strategy Version 1" in script


def test_pinescript_rejects_unsupported_indicator():
    definition = {
        "entry": {
            "direction": "long", "logic": "and",
            "conditions": [{"indicator": "made_up_indicator", "params": {}, "op": "<", "value": 1}],
        },
    }
    with pytest.raises(UnsupportedStrategyError):
        generate_pine_script("Bad Strategy", 1, definition)
