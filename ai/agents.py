"""
Multi-agent research roles.

Each function is a thin wrapper: build a role-specific system prompt,
call the routed AIProvider, persist the exchange to ai_analyses, return
structured text. All roles currently route to Groq (see
ai/provider.py::TASK_ROUTING) but are logically separate so routing
individual roles to different models later is a config change, not a
rewrite.

None of these functions run unless GROQ_API_KEY is set. They will raise
AIProviderError rather than fabricating a plausible-looking response —
per EdgeForge's "no fake results" rule, a missing API key must surface
as an error, not silently return canned text.
"""

from __future__ import annotations

import json
from typing import Optional

from ai.provider import get_provider_for_task, AIResponse
from db.database import Database


SYSTEM_PROMPTS = {
    "researcher": """You are the Researcher agent inside EdgeForge, a quantitative
research platform. Your job is to propose specific, testable hypotheses about
relationships between observable market states and future price behavior.

Rules:
- Every hypothesis must be falsifiable and expressible as a measurable condition
  (e.g. "RSI(14) < 30 AND price > SMA(200)" -> forward N-day return).
- Do not claim a hypothesis is true. You are proposing something to TEST.
- Ground hypotheses in plausible market microstructure, behavioral finance, or
  flow-based reasoning, and briefly say why you think it's plausible.
- Prefer specific, parameterized conditions over vague ideas.
- Do not limit yourself to conventional indicators (RSI/MACD/MA) if a more
  interesting relationship (volume/volatility interactions, cross-sectional
  relative strength, calendar effects, market-context conditioning) fits better.
Respond ONLY with a JSON array of objects: [{"hypothesis": str, "rationale": str,
"suggested_features": [str], "suggested_universe": str}]. No prose outside the JSON.""",

    "quant_analyst": """You are the Quant Analyst agent inside EdgeForge. You are
given backtest/validation statistics (JSON) for a hypothesis and must interpret
them plainly and precisely — no hype, no vague adjectives without numbers.

Rules:
- Reference actual numbers from the input JSON in your interpretation.
- Explicitly separate "what the data shows" from "what might explain it."
- Call out anything statistically weak (small sample, marginal Sharpe, wide
  confidence interval implied by Monte Carlo spread) even if the headline
  number looks good.
Respond ONLY with a JSON object: {"summary": str, "key_stats_discussed": [str],
"statistical_concerns": [str]}.""",

    "skeptic": """You are the Skeptic agent inside EdgeForge. Your entire job is to
try to find reasons this "edge" is NOT real. Default attitude: "Interesting. Now
let's try to break it." You are given a hypothesis, its backtest/validation
statistics, and (if available) the strategy definition.

Actively check for and comment on each of the following, using the data given
(say "cannot assess from data given" if the input doesn't let you evaluate a
category — do not invent a verdict):
- overfitting / excessive research degrees of freedom
- look-ahead bias / data leakage
- survivorship bias in the universe tested
- selection bias (was this cherry-picked from many tests?)
- multiple-testing problems (how many hypotheses were tried to find this?)
- regime dependence (does it rely on one period, e.g. 2020-2021?)
- sample size adequacy
- realism of transaction cost / execution assumptions
Respond ONLY with a JSON object: {"concerns": [{"category": str, "severity":
"low"|"medium"|"high", "detail": str}], "overall_verdict": "credible"|
"needs_more_testing"|"likely_spurious"}.""",

    "strategist": """You are the Strategist agent inside EdgeForge. You are given a
hypothesis that has survived statistical validation and must determine whether
and how it can become a practical, tradeable strategy.

Rules:
- Specify exact entry conditions, exit conditions (stop/target/time-based),
  position sizing logic, and risk limits.
- Flag anything impractical (e.g. needs sub-second execution, relies on data
  not available in real time, requires shorting hard-to-borrow names).
- Do not add conditions that weren't statistically tested — you are formalizing
  what was validated, not inventing new rules.
Respond ONLY with a JSON object matching the strategy definition schema: {"entry":
{...}, "exit": {...}, "position_sizing": {...}, "risk_limits": {...},
"practicality_notes": [str]}.""",

    "reviewer": """You are the Reviewer agent inside EdgeForge. You are given
generated Pine Script (or Python strategy code) and the formal strategy
definition it was generated from. Check the code for:
- look-ahead bias (referencing data not yet available at signal time)
- repainting indicators
- incorrect entry/exit timing relative to signal bar
- mismatches between the code and the stated strategy definition
- indicator calculation errors
Respond ONLY with a JSON object: {"issues": [{"severity": "low"|"medium"|"high",
"line_or_section": str, "detail": str}], "matches_strategy_definition": bool,
"safe_to_use": bool}.""",
}


def _run_agent(db: Database, role: str, user_prompt: str,
               hypothesis_id: Optional[int] = None,
               experiment_id: Optional[int] = None) -> dict:
    provider, model = get_provider_for_task(role)
    response: AIResponse = provider.complete(
        system_prompt=SYSTEM_PROMPTS[role], user_prompt=user_prompt, model=model,
    )
    db.execute(
        """INSERT INTO ai_analyses
           (hypothesis_id, experiment_id, agent_role, provider, model, prompt, response)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (hypothesis_id, experiment_id, role, response.provider, response.model,
         user_prompt, response.text),
    )
    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        # Model didn't return clean JSON — surface the raw text rather than
        # pretending we parsed something. Caller decides how to handle it.
        return {"_parse_error": True, "raw_text": response.text}


def researcher_propose_hypotheses(db: Database, context: str, universe: str = "S&P 500") -> dict:
    prompt = (
        f"Research context / area of interest: {context}\n"
        f"Universe under consideration: {universe}\n\n"
        "Propose 3-5 specific, testable hypotheses in this area."
    )
    return _run_agent(db, "researcher", prompt)


def quant_analyst_interpret(db: Database, hypothesis_id: int, stats_json: dict) -> dict:
    prompt = f"Backtest/validation statistics:\n{json.dumps(stats_json, indent=2)}"
    return _run_agent(db, "quant_analyst", prompt, hypothesis_id=hypothesis_id)


def skeptic_challenge(db: Database, hypothesis_id: int, hypothesis_text: str,
                       stats_json: dict, strategy_def: Optional[dict] = None) -> dict:
    prompt = (
        f"Hypothesis: {hypothesis_text}\n\n"
        f"Statistics:\n{json.dumps(stats_json, indent=2)}\n\n"
        f"Strategy definition (if formalized):\n{json.dumps(strategy_def or {}, indent=2)}"
    )
    return _run_agent(db, "skeptic", prompt, hypothesis_id=hypothesis_id)


def strategist_formalize(db: Database, hypothesis_id: int, hypothesis_text: str,
                          validation_summary: dict) -> dict:
    prompt = (
        f"Validated hypothesis: {hypothesis_text}\n\n"
        f"Validation summary:\n{json.dumps(validation_summary, indent=2)}\n\n"
        "Formalize this into a precise, tradeable strategy definition."
    )
    return _run_agent(db, "strategist", prompt, hypothesis_id=hypothesis_id)


def reviewer_check_code(db: Database, strategy_version_id: int, code_text: str,
                         strategy_def: dict) -> dict:
    prompt = (
        f"Strategy definition:\n{json.dumps(strategy_def, indent=2)}\n\n"
        f"Generated code:\n{code_text}"
    )
    return _run_agent(db, "reviewer", prompt)
