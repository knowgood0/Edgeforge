"""
Pine Script generator — V1.

Per spec: this must NOT invent rules. It mechanically translates a
formal strategy definition (the same JSON schema stored in
strategy_versions.definition_json) into Pine Script v5. If a strategy
definition uses a condition type this generator doesn't yet support, it
raises UnsupportedStrategyError rather than silently dropping or
guessing at the rule — a wrong-but-plausible-looking Pine script is far
more dangerous than a clear error.

Supported condition schema (V1):
{
  "entry": {
    "direction": "long" | "short",
    "conditions": [
      {"indicator": "rsi", "params": {"period": 14}, "op": "<", "value": 30},
      {"indicator": "price_vs_sma", "params": {"period": 200}, "op": ">", "value": 0}
    ],
    "logic": "and"  # or "or"
  },
  "exit": {
    "stop_loss_pct": 0.05,
    "take_profit_pct": 0.10,
    "max_holding_days": 10
  },
  "position_sizing": {"pct_of_equity": 100}
}

After generation, run ai.agents.reviewer_check_code against the output
before treating it as trustworthy — this module only guarantees
mechanical faithfulness to the definition, not that the definition
itself is sound (that's the Skeptic/Reviewer's job).
"""

from __future__ import annotations

import json


class UnsupportedStrategyError(Exception):
    pass


_INDICATOR_TEMPLATES = {
    "rsi": lambda p: f"ta.rsi(close, {p.get('period', 14)})",
    "sma": lambda p: f"ta.sma(close, {p.get('period', 20)})",
    "ema": lambda p: f"ta.ema(close, {p.get('period', 20)})",
    "price_vs_sma": lambda p: f"(close - ta.sma(close, {p.get('period', 200)})) / ta.sma(close, {p.get('period', 200)}) * 100",
    "atr": lambda p: f"ta.atr({p.get('period', 14)})",
    "relative_volume": lambda p: f"volume / ta.sma(volume, {p.get('period', 20)})",
    "roc": lambda p: f"ta.roc(close, {p.get('period', 10)})",
}

_OP_MAP = {"<": "<", "<=": "<=", ">": ">", ">=": ">=", "==": "==", "!=": "!="}


def _render_condition(cond: dict, idx: int) -> tuple[str, str]:
    indicator = cond.get("indicator")
    if indicator not in _INDICATOR_TEMPLATES:
        raise UnsupportedStrategyError(
            f"Indicator '{indicator}' has no Pine translation yet in generator V1. "
            f"Supported: {list(_INDICATOR_TEMPLATES)}"
        )
    op = cond.get("op")
    if op not in _OP_MAP:
        raise UnsupportedStrategyError(f"Operator '{op}' not supported.")
    var_name = f"cond{idx}_val"
    expr = _INDICATOR_TEMPLATES[indicator](cond.get("params", {}))
    line = f"{var_name} = {expr}"
    check = f"{var_name} {_OP_MAP[op]} {cond['value']}"
    return line, check


def generate_pine_script(strategy_name: str, strategy_version_id: int,
                          definition: dict) -> str:
    entry = definition.get("entry")
    exit_def = definition.get("exit", {})
    sizing = definition.get("position_sizing", {"pct_of_equity": 100})

    if not entry or "conditions" not in entry:
        raise UnsupportedStrategyError("definition.entry.conditions is required.")

    direction = entry.get("direction", "long")
    if direction not in ("long", "short"):
        raise UnsupportedStrategyError("entry.direction must be 'long' or 'short'.")

    logic = entry.get("logic", "and")
    if logic not in ("and", "or"):
        raise UnsupportedStrategyError("entry.logic must be 'and' or 'or'.")
    pine_logic_op = " and " if logic == "and" else " or "

    lines, checks = [], []
    for i, cond in enumerate(entry["conditions"]):
        line, check = _render_condition(cond, i)
        lines.append(line)
        checks.append(check)

    condition_lines = "\n".join(lines)
    combined_check = pine_logic_op.join(checks)

    stop_pct = exit_def.get("stop_loss_pct")
    target_pct = exit_def.get("take_profit_pct")
    max_hold = exit_def.get("max_holding_days")
    pct_equity = sizing.get("pct_of_equity", 100)

    entry_call = "strategy.entry" 
    dir_const = "strategy.long" if direction == "long" else "strategy.short"

    exit_lines = []
    if stop_pct:
        stop_expr = (f"strategy.position_avg_price * (1 - {stop_pct})" if direction == "long"
                     else f"strategy.position_avg_price * (1 + {stop_pct})")
        exit_lines.append(f'    stop_price = {stop_expr}')
    if target_pct:
        target_expr = (f"strategy.position_avg_price * (1 + {target_pct})" if direction == "long"
                       else f"strategy.position_avg_price * (1 - {target_pct})")
        exit_lines.append(f'    target_price = {target_expr}')

    exit_call_args = ["\"" + "entry" + "\""]
    exit_kw = []
    if stop_pct:
        exit_kw.append("stop=stop_price")
    if target_pct:
        exit_kw.append("limit=target_price")
    exit_args_str = ", ".join(exit_call_args + exit_kw) if exit_kw else None

    max_hold_block = ""
    if max_hold:
        max_hold_block = f"""
// --- Time-based exit ---
var int entryBarIndex = na
if strategy.position_size != 0 and strategy.position_size[1] == 0
    entryBarIndex := bar_index
if strategy.position_size != 0 and not na(entryBarIndex) and (bar_index - entryBarIndex) >= {max_hold}
    strategy.close("entry", comment="time_exit")
"""

    script = f'''// EdgeForge — generated from Strategy Version {strategy_version_id}
// Strategy: {strategy_name}
// Generator: pinescript/generator.py V1 (mechanical translation — see
// definition JSON stored alongside this script for the source of truth).
// DO NOT hand-edit rules here without also updating the strategy version
// in EdgeForge, or this script will drift from what was actually validated.
//@version=5
strategy("{strategy_name}", overlay=true,
     default_qty_type=strategy.percent_of_equity, default_qty_value={pct_equity},
     process_orders_on_close=true)

// --- Entry conditions (translated 1:1 from validated strategy definition) ---
{condition_lines}

entryCondition = {combined_check}

if entryCondition and strategy.position_size == 0
    strategy.entry("entry", {dir_const})

// --- Exit logic ---
{chr(10).join(exit_lines) if exit_lines else "// no stop/target defined in strategy definition"}
if strategy.position_size != 0 and ({"not na(stop_price)" if stop_pct else "false"} or {"not na(target_price)" if target_pct else "false"})
    strategy.exit({exit_args_str if exit_args_str else '"entry"'})
{max_hold_block}
// --- Plots & alerts ---
plotshape(entryCondition and strategy.position_size == 0, title="Entry Signal",
     style=shape.triangleup, location=location.belowbar, color=color.green, size=size.tiny)
alertcondition(entryCondition, title="EdgeForge Entry Signal",
     message="EdgeForge: entry condition met for {strategy_name}")
'''
    return script


def review_notes_template() -> str:
    return (
        "Automated mechanical checks only (V1): condition indicators/operators "
        "were validated against the supported template list at generation time. "
        "Full semantic review (repainting, look-ahead, order-timing correctness) "
        "should be run via ai.agents.reviewer_check_code before treating this "
        "script as trade-ready."
    )
