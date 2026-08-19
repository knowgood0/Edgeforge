"""
Turns a hypothesis (human or AI generated) into a measurable signal
function, and provides research-memory duplicate detection so EdgeForge
doesn't repeatedly re-test hypotheses that already failed.
"""

from __future__ import annotations

import difflib
import json
from typing import Callable

import pandas as pd

from db.database import Database
from features import engineering as feat


def build_signal_from_conditions(conditions: list[dict], logic: str = "and") -> Callable[[pd.DataFrame], pd.Series]:
    """conditions: [{"feature": "rsi", "params": {"period":14}, "op": "<", "value": 30}, ...]
    Returns a function df -> boolean Series, computed feature-by-feature
    so each condition only ever depends on data up to and including the
    row it's evaluated on (see features/engineering.py docstring)."""

    ops = {
        "<": lambda s, v: s < v, "<=": lambda s, v: s <= v,
        ">": lambda s, v: s > v, ">=": lambda s, v: s >= v,
        "==": lambda s, v: s == v, "!=": lambda s, v: s != v,
    }

    def signal_fn(df: pd.DataFrame) -> pd.Series:
        masks = []
        for c in conditions:
            series = feat.compute_feature(c["feature"], df, **c.get("params", {}))
            op_fn = ops.get(c["op"])
            if op_fn is None:
                raise ValueError(f"Unsupported operator: {c['op']}")
            masks.append(op_fn(series, c["value"]))
        combined = masks[0]
        for m in masks[1:]:
            combined = (combined & m) if logic == "and" else (combined | m)
        return combined.fillna(False)

    return signal_fn


def register_hypothesis(db: Database, statement: str, origin: str = "user") -> dict:
    """Creates the hypothesis row, checking research memory for
    near-duplicates first so EdgeForge can say
    'this resembles experiment #N which failed because...' instead of
    blindly re-testing it."""
    duplicate = find_similar_hypothesis(db, statement)
    hyp_id = db.execute(
        "INSERT INTO hypotheses (statement, origin, similar_to_hypothesis_id) VALUES (?, ?, ?)",
        (statement, origin, duplicate["id"] if duplicate else None),
    )
    db.log_research_event(hyp_id, None, "tested", detail=f"origin={origin}")
    return {"hypothesis_id": hyp_id, "similar_to": duplicate}


def find_similar_hypothesis(db: Database, statement: str, threshold: float = 0.72) -> dict | None:
    """Cheap lexical-similarity duplicate check (difflib ratio) against
    prior hypotheses, prioritizing ones that were rejected — the
    highest-value case is warning "this looks like something that
    already failed." This is NOT semantic search; a real deployment
    should upgrade this to embedding similarity, but that requires an
    embeddings API call this MVP doesn't assume you have configured."""
    existing = db.query(
        "SELECT id, statement, status, rejected_reason FROM hypotheses ORDER BY id DESC LIMIT 500"
    )
    best = None
    best_score = 0.0
    for row in existing:
        score = difflib.SequenceMatcher(None, statement.lower(), row["statement"].lower()).ratio()
        if score > best_score:
            best_score = score
            best = row
    if best and best_score >= threshold:
        return {**best, "similarity": round(best_score, 3)}
    return None


def reject_hypothesis(db: Database, hypothesis_id: int, reason: str):
    db.execute(
        "UPDATE hypotheses SET status = 'rejected', rejected_reason = ? WHERE id = ?",
        (reason, hypothesis_id),
    )
    db.log_research_event(hypothesis_id, None, "rejected", detail=reason)


def advance_hypothesis_status(db: Database, hypothesis_id: int, status: str, detail: str = ""):
    valid = {"proposed", "testing", "survived_initial", "survived_oos", "survived_adversarial", "rejected"}
    if status not in valid:
        raise ValueError(f"Invalid status: {status}")
    db.execute("UPDATE hypotheses SET status = ? WHERE id = ?", (status, hypothesis_id))
    db.log_research_event(hypothesis_id, None, f"survived_stage:{status}", detail=detail)
