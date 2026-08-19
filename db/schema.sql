-- EdgeForge database schema
-- SQLite now; written to be portable to PostgreSQL later (see db/database.py notes).
-- All timestamps stored as ISO8601 UTC strings.

PRAGMA foreign_keys = ON;

-- ============================================================
-- CORE MARKET DATA
-- ============================================================

CREATE TABLE IF NOT EXISTS securities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL UNIQUE,
    name TEXT,
    asset_class TEXT DEFAULT 'equity',      -- equity, etf, index, future, option, crypto
    sector TEXT,
    industry TEXT,
    is_active INTEGER DEFAULT 1,            -- 0 if delisted (survivorship tracking)
    delisted_date TEXT,
    first_data_date TEXT,
    last_data_date TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS price_bars (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    security_id INTEGER NOT NULL REFERENCES securities(id),
    date TEXT NOT NULL,                     -- YYYY-MM-DD
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    adj_close REAL,
    volume INTEGER,
    source TEXT,                            -- e.g. 'yfinance'
    is_suspect INTEGER DEFAULT 0,           -- data-quality flag
    suspect_reason TEXT,
    UNIQUE(security_id, date)
);
CREATE INDEX IF NOT EXISTS idx_price_bars_sec_date ON price_bars(security_id, date);

CREATE TABLE IF NOT EXISTS corporate_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    security_id INTEGER NOT NULL REFERENCES securities(id),
    date TEXT NOT NULL,
    action_type TEXT NOT NULL,              -- split, dividend, symbol_change, delisting
    detail TEXT,                            -- e.g. '2:1', '0.24', 'renamed to XYZ'
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS data_quality_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    security_id INTEGER REFERENCES securities(id),
    check_type TEXT NOT NULL,               -- missing_bar, price_spike, volume_anomaly, etc.
    date TEXT,
    detail TEXT,
    severity TEXT DEFAULT 'info',           -- info, warning, critical
    created_at TEXT DEFAULT (datetime('now'))
);

-- ============================================================
-- FEATURE ENGINEERING
-- ============================================================

CREATE TABLE IF NOT EXISTS feature_definitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,              -- e.g. 'rsi_14', 'ret_spy_excess_5d'
    category TEXT NOT NULL,                 -- price, trend, momentum, volatility, volume, market_context, calendar, derived
    description TEXT,
    formula_ref TEXT,                       -- python function name / module path that computes it
    params_json TEXT,                       -- JSON of parameters used, e.g. {"period":14}
    origin TEXT DEFAULT 'builtin',          -- builtin, ai_proposed, user_proposed
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS feature_values (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    security_id INTEGER NOT NULL REFERENCES securities(id),
    feature_id INTEGER NOT NULL REFERENCES feature_definitions(id),
    date TEXT NOT NULL,
    value REAL,
    UNIQUE(security_id, feature_id, date)
);
CREATE INDEX IF NOT EXISTS idx_feature_values_lookup ON feature_values(feature_id, date);

-- ============================================================
-- RESEARCH: HYPOTHESES & EXPERIMENTS
-- ============================================================

CREATE TABLE IF NOT EXISTS hypotheses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    statement TEXT NOT NULL,                -- human-readable hypothesis
    origin TEXT DEFAULT 'user',             -- user, ai_researcher, autonomous
    status TEXT DEFAULT 'proposed',         -- proposed, testing, survived_initial, survived_oos,
                                             -- survived_adversarial, rejected, duplicate
    rejected_reason TEXT,
    similar_to_hypothesis_id INTEGER REFERENCES hypotheses(id),  -- research-memory linkage
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id INTEGER REFERENCES hypotheses(id),
    experiment_type TEXT NOT NULL,          -- backtest, param_search, cross_security, walk_forward, monte_carlo, adversarial
    universe_json TEXT,                     -- JSON list of symbols tested
    date_range_start TEXT,
    date_range_end TEXT,
    params_json TEXT,                       -- full parameter set used
    status TEXT DEFAULT 'queued',           -- queued, running, complete, failed
    started_at TEXT,
    completed_at TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS research_log (
    -- append-only ledger of every hypothesis tested, for multiple-testing accounting
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id INTEGER REFERENCES hypotheses(id),
    experiment_id INTEGER REFERENCES experiments(id),
    event TEXT NOT NULL,                    -- 'tested', 'rejected', 'survived_stage:X'
    detail TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- ============================================================
-- BACKTESTING
-- ============================================================

CREATE TABLE IF NOT EXISTS backtest_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER REFERENCES experiments(id),
    strategy_version_id INTEGER,            -- FK added below after strategy_versions defined
    symbol TEXT NOT NULL,
    date_range_start TEXT,
    date_range_end TEXT,
    execution_mode TEXT DEFAULT 'next_bar_open',  -- next_bar_open, next_bar_close, etc.
    slippage_bps REAL DEFAULT 5,
    commission_per_trade REAL DEFAULT 0,
    initial_capital REAL DEFAULT 100000,
    position_sizing_json TEXT,
    is_train INTEGER DEFAULT 1,             -- 1 = in-sample, 0 = out-of-sample
    total_return REAL,
    cagr REAL,
    sharpe REAL,
    sortino REAL,
    max_drawdown REAL,
    win_rate REAL,
    profit_factor REAL,
    num_trades INTEGER,
    avg_trade_return REAL,
    payoff_ratio REAL,
    stats_json TEXT,                        -- full stats blob
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS backtest_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backtest_run_id INTEGER NOT NULL REFERENCES backtest_runs(id),
    symbol TEXT NOT NULL,
    entry_date TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_date TEXT,
    exit_price REAL,
    direction TEXT DEFAULT 'long',          -- long, short
    quantity REAL,
    exit_reason TEXT,                       -- signal, stop, target, trailing_stop, time_exit, end_of_data
    pnl REAL,
    pnl_pct REAL,
    holding_days INTEGER
);
CREATE INDEX IF NOT EXISTS idx_backtest_trades_run ON backtest_trades(backtest_run_id);

-- ============================================================
-- VALIDATION & ROBUSTNESS
-- ============================================================

CREATE TABLE IF NOT EXISTS validation_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id INTEGER REFERENCES hypotheses(id),
    strategy_version_id INTEGER,
    validation_type TEXT NOT NULL,          -- train_test, oos, walk_forward, param_sensitivity,
                                             -- cross_security, market_regime, tx_cost_sensitivity, monte_carlo
    passed INTEGER,                         -- 0/1, per pre-registered threshold
    metrics_json TEXT,
    notes TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS robustness_tests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id INTEGER REFERENCES hypotheses(id),
    parameter_name TEXT NOT NULL,
    parameter_value REAL NOT NULL,
    metric_name TEXT NOT NULL,              -- e.g. sharpe, total_return
    metric_value REAL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS adversarial_tests (
    -- "Try to Kill It" results
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_version_id INTEGER,
    test_type TEXT NOT NULL,                -- threshold_shift, holding_period_shift, filter_removal,
                                             -- tx_cost_shock, cross_security, cross_period, regime,
                                             -- trade_shuffle, bootstrap, placebo, drop_best_period
    result TEXT,                            -- survived, weakened, failed
    detail_json TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- ============================================================
-- STATISTICAL ANALYSIS / EDGE SCORE
-- ============================================================

CREATE TABLE IF NOT EXISTS edge_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id INTEGER REFERENCES hypotheses(id),
    strategy_version_id INTEGER,
    score REAL NOT NULL,                    -- 0-100 composite
    components_json TEXT NOT NULL,          -- breakdown of each component
    deflated_sharpe REAL,
    probability_backtest_overfitting REAL,
    false_discovery_rate_est REAL,
    computed_at TEXT DEFAULT (datetime('now'))
);

-- ============================================================
-- AI RESEARCH
-- ============================================================

CREATE TABLE IF NOT EXISTS ai_analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id INTEGER REFERENCES hypotheses(id),
    experiment_id INTEGER REFERENCES experiments(id),
    agent_role TEXT NOT NULL,               -- researcher, quant_analyst, skeptic, strategist, code_engineer, reviewer
    provider TEXT,                          -- groq, openai, anthropic, local
    model TEXT,
    prompt TEXT,
    response TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS research_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id INTEGER REFERENCES hypotheses(id),
    strategy_version_id INTEGER,
    what_was_discovered TEXT,
    why_it_might_exist TEXT,
    historical_evidence TEXT,
    oos_evidence TEXT,
    robustness TEXT,
    weaknesses TEXT,
    possible_explanation TEXT,
    alternative_explanations TEXT,
    what_could_invalidate_it TEXT,
    recommended_next_experiments TEXT,
    overall_assessment TEXT,
    generated_at TEXT DEFAULT (datetime('now'))
);

-- ============================================================
-- STRATEGIES
-- ============================================================

CREATE TABLE IF NOT EXISTS strategies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    hypothesis_id INTEGER REFERENCES hypotheses(id),
    universe_json TEXT,                     -- symbols/universe this applies to
    description TEXT,
    is_favorite INTEGER DEFAULT 0,
    is_archived INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS strategy_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id INTEGER NOT NULL REFERENCES strategies(id),
    version_number INTEGER NOT NULL,
    definition_json TEXT NOT NULL,          -- entry/exit conditions, indicators, params, sizing, risk limits
    edge_score REAL,
    validation_status TEXT DEFAULT 'unvalidated',
    notes TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(strategy_id, version_number)
);

CREATE TABLE IF NOT EXISTS pine_scripts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_version_id INTEGER NOT NULL REFERENCES strategy_versions(id),
    script_text TEXT NOT NULL,
    reviewed INTEGER DEFAULT 0,
    review_notes TEXT,                      -- lookahead/repainting/timing review results
    generated_at TEXT DEFAULT (datetime('now'))
);

-- ============================================================
-- PAPER TRADING / EXECUTION
-- ============================================================

CREATE TABLE IF NOT EXISTS execution_providers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,                     -- tradingview_webhook, webull_paper, custom_webhook
    config_json TEXT,                       -- non-secret config only
    is_active INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS paper_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_version_id INTEGER REFERENCES strategy_versions(id),
    execution_provider_id INTEGER REFERENCES execution_providers(id),
    symbol TEXT NOT NULL,
    direction TEXT,
    entry_time TEXT,
    entry_price REAL,
    exit_time TEXT,
    exit_price REAL,
    quantity REAL,
    status TEXT DEFAULT 'open',             -- open, closed, cancelled
    pnl REAL,
    raw_payload_json TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- ============================================================
-- MISC
-- ============================================================

CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value TEXT
);
