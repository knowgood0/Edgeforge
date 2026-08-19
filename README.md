# EdgeForge

*Forge hypotheses. Break assumptions. Find edges.*

An AI-assisted quantitative research laboratory for discovering, testing,
and challenging potential trading edges — not a signal service, not a
screener. The system's default posture toward any profitable backtest is
"interesting, now let's try to break it."

## What's real vs. what's scaffolded (read this first)

Per the project's own "no fake results" rule, here's an honest inventory.

**Fully implemented, real logic:**
- SQLite schema covering the entire planned scope (data → strategies → paper trades)
- `DataProvider` abstraction with a `yfinance` implementation + data-quality checks
- Feature library (price/trend/momentum/volatility/volume/market-context/calendar/derived), all leak-safe (see `tests/test_features.py`)
- Event-driven backtest engine with strict next-bar-fill execution, slippage, commissions, stops/targets, time exits (see `tests/test_backtest.py` for the lookahead-prevention tests)
- Validation suite: train/test split, walk-forward, parameter sensitivity, cross-security testing, transaction-cost sensitivity, Monte Carlo trade-shuffle bootstrap, deflated Sharpe ratio
- Edge Score composite scoring with visible component breakdown
- `AIProvider` abstraction with a working Groq implementation and five distinct agent roles (Researcher, Quant Analyst, Skeptic, Strategist, Reviewer) with real, separate system prompts
- Pine Script generator (V1: mechanically translates a constrained but real condition schema — raises an error rather than guessing on anything outside that schema)
- `ExecutionProvider` abstraction with a working generic webhook implementation (paper-trading signal payloads only — no live-order code path exists anywhere in this codebase)
- Mobile-first PWA (installable, dark, offline app-shell) wired to a real Flask API — no mock data in the frontend
- Automated tests for the highest-risk failure points: lookahead prevention, execution timing, indicator correctness, DB persistence, Edge Score bounds, Pine Script generation

**Explicitly NOT implemented yet (Phase 2):**
- Autonomous research loop (the code to *run* multi-stage AI research repeatedly and pick its own next direction — the individual agent roles it would call already exist)
- "Try to Kill It" automated adversarial test suite (the individual techniques — bootstrap, parameter shifts, regime tests — exist in `validation/validator.py`; the one-button orchestration across all of them doesn't yet)
- Market-regime classification/testing
- Bayesian optimization / genetic search over parameter spaces (current param search is a manual sweep, by design — see `validation.validator.parameter_sensitivity`)
- Direct Webull integration inside EdgeForge (intentionally — reuse your existing bot via the webhook provider instead of duplicating that SDK work)
- Full Probability of Backtest Overfitting (CSCV) — deflated Sharpe is implemented as the interim multiple-testing safeguard; see `validation.validator.probability_of_backtest_overfitting_note()`

## Why this scope for a first delivery

The full spec is a multi-week system. Shipping a "complete" version of
all twelve modules in one pass would have meant either very shallow
stub code everywhere, or code I couldn't actually verify runs correctly
in this sandbox (no outbound network access here — Groq and yfinance
calls have to be exercised on your real deployment). Instead, Phase 1
is the part that has to be *correct*, because every later stage trusts
it: if the backtester can leak future information or the validation
math is wrong, nothing downstream — however polished — is trustworthy.

## Setup

```bash
cd edgeforge
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and add your GROQ_API_KEY
python app.py
```

Open `http://localhost:5000` (or your Render URL) — on Android, use
Chrome's "Add to Home Screen" to install it as a PWA.

Run tests:
```bash
pytest tests/ -v
```

## Sample research workflow

The exact example from the spec — "search for short-term mean-reversion
edges in liquid large-cap stocks after unusually large daily declines" —
is wired end to end in `research/sample_workflow.py` and exposed via the
Research tab / `POST /api/research/run-sample`. It requires real network
access to fetch price data, so it won't produce results inside this dev
sandbox — that's the tool honestly reporting "data unavailable," not a
bug.

```bash
python -m research.sample_workflow --symbol AAPL
```

## Project layout

```
edgeforge/
  app.py                    Flask entry point + API routes
  db/schema.sql              Full database schema
  db/database.py             DB abstraction (SQLite now, Postgres-portable)
  data/provider.py            DataProvider abstraction + yfinance impl
  features/engineering.py     Feature library
  backtest/engine.py          Event-driven backtester
  validation/validator.py     Validation & robustness suite
  research/scoring.py         Edge Score
  research/hypothesis.py      Hypothesis -> signal translation, research memory
  research/sample_workflow.py End-to-end example pipeline
  ai/provider.py               AIProvider abstraction + Groq
  ai/agents.py                 Researcher/Quant Analyst/Skeptic/Strategist/Reviewer roles
  pinescript/generator.py      Strategy definition -> Pine Script v5
  execution/provider.py        ExecutionProvider abstraction (paper trading only)
  static/                      PWA frontend (vanilla HTML/CSS/JS, no build step)
  tests/                       Automated tests
```

## Deploying

Written to deploy the same way as your other Flask projects (Render):
`gunicorn app:app`, with `GROQ_API_KEY` and (once you wire it)
`EDGEFORGE_WEBHOOK_URL` set as environment variables in the Render
dashboard — never in code.

## Next steps (Phase 2, when you're ready)

1. Wire `research/sample_workflow.py`'s pattern into a real "Guided" mode
   endpoint that takes a user's plain-English research prompt and calls
   the Researcher agent to generate the hypothesis conditions.
2. Build the autonomous research loop (resource-limited: max experiments,
   max runtime, max API calls, per the spec) using the existing agent
   roles + research memory.
3. Build the "Try to Kill It" one-button orchestration over the existing
   adversarial-technique building blocks.
4. Regime classification (bull/bear/high-vol/low-vol) as a feature +
   validation dimension.
5. Strategy versioning UI + comparison view in the Strategies tab.
