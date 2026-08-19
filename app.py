"""
EdgeForge Flask application.

Serves the PWA frontend (static/) and a JSON API under /api/*.
Run: python app.py  (dev)  or  gunicorn app:app  (prod, e.g. Render)
"""

from __future__ import annotations

import json
import os

from flask import Flask, jsonify, request, send_from_directory

from db.database import get_db
from research.hypothesis import register_hypothesis, find_similar_hypothesis
from research.scoring import status_label
from ai.provider import AIProviderError

app = Flask(__name__, static_folder="static", static_url_path="")
db = get_db()


# ------------------------------------------------------------------
# Static / PWA
# ------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/manifest.json")
def manifest():
    return send_from_directory(app.static_folder, "manifest.json")


@app.route("/sw.js")
def service_worker():
    return send_from_directory(app.static_folder, "sw.js")


# ------------------------------------------------------------------
# Dashboard
# ------------------------------------------------------------------

@app.route("/api/dashboard")
def api_dashboard():
    stats = db.research_stats()
    top_candidates = db.query(
        """SELECT h.id, h.statement, h.status, e.score, e.components_json
           FROM hypotheses h
           JOIN edge_scores e ON e.hypothesis_id = h.id
           WHERE e.id IN (SELECT MAX(id) FROM edge_scores GROUP BY hypothesis_id)
           ORDER BY e.score DESC LIMIT 10"""
    )
    for c in top_candidates:
        c["status_label"] = status_label(c["score"])
        c["components"] = json.loads(c["components_json"]) if c["components_json"] else {}
        del c["components_json"]

    recent = db.query(
        "SELECT id, statement, status, origin, created_at FROM hypotheses ORDER BY id DESC LIMIT 10"
    )
    running = db.query(
        "SELECT id, experiment_type, status, started_at FROM experiments WHERE status = 'running'"
    )
    failed = db.query(
        "SELECT id, statement, rejected_reason, created_at FROM hypotheses WHERE status = 'rejected' "
        "ORDER BY id DESC LIMIT 10"
    )

    return jsonify({
        "research_statistics": stats,
        "top_candidates": top_candidates,
        "new_discoveries": recent,
        "research_in_progress": running,
        "failed_edges": failed,
    })


# ------------------------------------------------------------------
# Hypotheses
# ------------------------------------------------------------------

@app.route("/api/hypotheses", methods=["GET"])
def list_hypotheses():
    rows = db.query("SELECT * FROM hypotheses ORDER BY id DESC LIMIT 200")
    return jsonify(rows)


@app.route("/api/hypotheses", methods=["POST"])
def create_hypothesis():
    body = request.get_json(force=True)
    statement = body.get("statement")
    if not statement:
        return jsonify({"error": "statement is required"}), 400
    result = register_hypothesis(db, statement, origin=body.get("origin", "user"))
    return jsonify(result), 201


@app.route("/api/hypotheses/check-similar", methods=["POST"])
def check_similar():
    body = request.get_json(force=True)
    statement = body.get("statement", "")
    match = find_similar_hypothesis(db, statement)
    return jsonify({"match": match})


# ------------------------------------------------------------------
# Strategies
# ------------------------------------------------------------------

@app.route("/api/strategies", methods=["GET"])
def list_strategies():
    rows = db.query(
        """SELECT s.*, sv.version_number AS latest_version, sv.edge_score, sv.validation_status
           FROM strategies s
           LEFT JOIN strategy_versions sv ON sv.id = (
               SELECT id FROM strategy_versions WHERE strategy_id = s.id ORDER BY version_number DESC LIMIT 1
           )
           WHERE s.is_archived = 0
           ORDER BY s.id DESC"""
    )
    return jsonify(rows)


@app.route("/api/strategies/<int:strategy_id>/pine-script", methods=["POST"])
def generate_pine(strategy_id: int):
    from pinescript.generator import generate_pine_script, UnsupportedStrategyError

    version = db.query_one(
        "SELECT * FROM strategy_versions WHERE strategy_id = ? ORDER BY version_number DESC LIMIT 1",
        (strategy_id,),
    )
    strategy = db.query_one("SELECT * FROM strategies WHERE id = ?", (strategy_id,))
    if not version or not strategy:
        return jsonify({"error": "strategy or version not found"}), 404

    definition = json.loads(version["definition_json"])
    try:
        script = generate_pine_script(strategy["name"], version["id"], definition)
    except UnsupportedStrategyError as e:
        return jsonify({"error": str(e)}), 422

    pine_id = db.execute(
        "INSERT INTO pine_scripts (strategy_version_id, script_text) VALUES (?, ?)",
        (version["id"], script),
    )
    return jsonify({"pine_script_id": pine_id, "script": script})


# ------------------------------------------------------------------
# Sample research workflow (real backtest, no network in this sandbox —
# see research/sample_workflow.py docstring)
# ------------------------------------------------------------------

@app.route("/api/research/run-sample", methods=["POST"])
def run_sample_research():
    from research.sample_workflow import run as run_sample

    body = request.get_json(force=True, silent=True) or {}
    symbol = body.get("symbol", "AAPL")
    try:
        result = run_sample(symbol)
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500
    return jsonify(result)


# ------------------------------------------------------------------
# AI status (so the frontend can show "AI not configured" honestly)
# ------------------------------------------------------------------

@app.route("/api/ai/status")
def ai_status():
    configured = bool(os.environ.get("GROQ_API_KEY"))
    return jsonify({"groq_configured": configured})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
