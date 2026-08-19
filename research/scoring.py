"""
Edge Score: a composite research-confidence/robustness score (0-100).

Explicitly NOT a "probability of profit." It answers: "how much
statistical and structural evidence supports this being a real,
persistent relationship, as opposed to a residual of overfitting or
regime luck?" Every component is stored so the UI can show the
breakdown rather than a black-box number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


COMPONENT_WEIGHTS = {
    "oos_performance": 0.20,
    "walk_forward_consistency": 0.15,
    "sample_size": 0.10,
    "parameter_stability": 0.15,
    "cross_security_consistency": 0.15,
    "regime_consistency": 0.10,
    "cost_sensitivity": 0.10,
    "overfitting_penalty": 0.05,   # subtracted, see below
}


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def score_oos_performance(oos_sharpe: Optional[float]) -> float:
    if oos_sharpe is None:
        return 0.0
    # Sharpe 0 -> 0, Sharpe 1.5+ -> 1.0, linear between, negative -> 0
    return _clip01(oos_sharpe / 1.5)


def score_walk_forward_consistency(pct_windows_profitable: Optional[float],
                                    sharpe_std: Optional[float], avg_sharpe: Optional[float]) -> float:
    if pct_windows_profitable is None:
        return 0.0
    base = pct_windows_profitable
    if sharpe_std is not None and avg_sharpe and abs(avg_sharpe) > 1e-9:
        cv_penalty = _clip01(sharpe_std / abs(avg_sharpe))
        base = base * (1 - 0.5 * cv_penalty)
    return _clip01(base)


def score_sample_size(num_trades: int) -> float:
    # Rough heuristic: <20 trades is very weak evidence, >=100 is solid.
    if num_trades <= 0:
        return 0.0
    return _clip01((num_trades - 20) / 80) if num_trades > 20 else _clip01(num_trades / 20) * 0.3


def score_parameter_stability(stability_score: Optional[float]) -> float:
    if stability_score is None:
        return 0.0
    return _clip01(stability_score)


def score_cross_security_consistency(pct_symbols_positive: Optional[float],
                                      sharpe_std_across: Optional[float]) -> float:
    if pct_symbols_positive is None:
        return 0.0
    base = pct_symbols_positive
    if sharpe_std_across is not None:
        base *= _clip01(1 - sharpe_std_across / 2.0)
    return _clip01(base)


def score_regime_consistency(regime_results: Optional[dict]) -> float:
    """regime_results: {"bull": sharpe, "bear": sharpe, "high_vol": sharpe, ...}
    Rewards positive performance across regimes, not just the best one."""
    if not regime_results:
        return 0.0
    vals = [v for v in regime_results.values() if v is not None]
    if not vals:
        return 0.0
    positive_frac = sum(1 for v in vals if v > 0) / len(vals)
    return _clip01(positive_frac)


def score_cost_sensitivity(cost_sweep_returns: Optional[list[float]]) -> float:
    """cost_sweep_returns: total_return at increasing cost multipliers
    [0x, 1x, 2x, 4x]. Penalizes edges that evaporate quickly under
    realistic-to-stressed costs."""
    if not cost_sweep_returns or len(cost_sweep_returns) < 2:
        return 0.0
    base_case = cost_sweep_returns[1] if len(cost_sweep_returns) > 1 else cost_sweep_returns[0]
    stressed_case = cost_sweep_returns[-1]
    if base_case <= 0:
        return 0.0
    retained = _clip01(stressed_case / base_case) if base_case > 0 else 0.0
    return retained


def score_overfitting_penalty(deflated_sharpe_prob: Optional[float]) -> float:
    """Returns a PENALTY (0 = no penalty, 1 = max penalty), based on the
    deflated Sharpe's implied probability the true Sharpe is <= 0."""
    if deflated_sharpe_prob is None:
        return 0.5  # unknown multiple-testing exposure -> moderate default penalty
    return _clip01(1 - deflated_sharpe_prob)


@dataclass
class EdgeScoreInputs:
    oos_sharpe: Optional[float] = None
    walk_forward_pct_profitable: Optional[float] = None
    walk_forward_sharpe_std: Optional[float] = None
    walk_forward_avg_sharpe: Optional[float] = None
    num_trades: int = 0
    parameter_stability_score: Optional[float] = None
    cross_security_pct_positive: Optional[float] = None
    cross_security_sharpe_std: Optional[float] = None
    regime_results: Optional[dict] = None
    cost_sweep_returns: Optional[list[float]] = None
    deflated_sharpe_prob: Optional[float] = None


def compute_edge_score(inputs: EdgeScoreInputs) -> dict:
    components = {
        "oos_performance": score_oos_performance(inputs.oos_sharpe),
        "walk_forward_consistency": score_walk_forward_consistency(
            inputs.walk_forward_pct_profitable, inputs.walk_forward_sharpe_std, inputs.walk_forward_avg_sharpe),
        "sample_size": score_sample_size(inputs.num_trades),
        "parameter_stability": score_parameter_stability(inputs.parameter_stability_score),
        "cross_security_consistency": score_cross_security_consistency(
            inputs.cross_security_pct_positive, inputs.cross_security_sharpe_std),
        "regime_consistency": score_regime_consistency(inputs.regime_results),
        "cost_sensitivity": score_cost_sensitivity(inputs.cost_sweep_returns),
    }
    overfitting_penalty = score_overfitting_penalty(inputs.deflated_sharpe_prob)

    weighted_sum = sum(components[k] * COMPONENT_WEIGHTS[k] for k in components)
    penalty = overfitting_penalty * COMPONENT_WEIGHTS["overfitting_penalty"]
    raw_score = weighted_sum - penalty
    # normalize by the weight mass actually used (all components present here)
    total_weight = sum(COMPONENT_WEIGHTS[k] for k in components) 
    score_0_100 = _clip01(raw_score / total_weight) * 100

    return {
        "score": round(score_0_100, 1),
        "components": {k: round(v, 3) for k, v in components.items()},
        "overfitting_penalty": round(overfitting_penalty, 3),
    }


def status_label(score: float) -> str:
    if score >= 75:
        return "Promising"
    if score >= 55:
        return "Interesting"
    if score >= 35:
        return "Weak"
    return "Not Supported"
