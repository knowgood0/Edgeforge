// EdgeForge frontend — vanilla JS, no build step (single-file-friendly,
// consistent with how you like your other projects deployed). Talks only
// to the real Flask API; never fabricates numbers client-side.

const view = document.getElementById("view");
const tabs = document.querySelectorAll(".tabbar button");
let currentTab = "dashboard";

tabs.forEach((btn) => {
  btn.addEventListener("click", () => {
    tabs.forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    currentTab = btn.dataset.tab;
    render();
  });
});

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error || body.reason || `Request failed (${res.status})`);
  }
  return res.json();
}

function statusPillClass(status) {
  if (["survived_adversarial", "survived_oos"].includes(status)) return "signal";
  if (["survived_initial", "testing", "proposed"].includes(status)) return "warn";
  if (status === "rejected") return "danger";
  return "neutral";
}

function edgeGaugeHTML(score) {
  const tone = score >= 75 ? "signal" : score >= 45 ? "warn" : "danger";
  const filledCount = Math.round((score / 100) * 20);
  let ticks = "";
  for (let i = 0; i < 20; i++) {
    ticks += `<div class="tick ${i < filledCount ? "filled " + tone : ""}"></div>`;
  }
  return `
    <div class="edge-score-row">
      <div class="edge-score-number mono">${score.toFixed(1)}</div>
      <div class="edge-gauge">${ticks}</div>
    </div>`;
}

async function renderDashboard() {
  view.innerHTML = `<div class="empty-state">Loading dashboard…</div>`;
  try {
    const data = await api("/api/dashboard");
    document.getElementById("t-tested").textContent = data.research_statistics.hypotheses_tested;
    document.getElementById("t-initial").textContent = data.research_statistics.survived_initial_testing;
    document.getElementById("t-oos").textContent = data.research_statistics.survived_oos_testing;
    document.getElementById("t-adversarial").textContent = data.research_statistics.survived_adversarial_validation;

    let html = `<div class="section-label">Top candidates</div>`;
    if (!data.top_candidates.length) {
      html += `<div class="empty-state">No scored candidates yet. Run research from the Research tab to populate this.</div>`;
    }
    data.top_candidates.forEach((c) => {
      html += `
        <div class="panel">
          <div class="panel-title">${escapeHTML(c.statement)}</div>
          <span class="status-pill ${statusPillClass(c.status)}">${c.status_label}</span>
          ${edgeGaugeHTML(c.score)}
          <button class="kill" data-kill="${c.id}">Try to kill this edge</button>
        </div>`;
    });

    html += `<div class="section-label">New discoveries</div>`;
    if (!data.new_discoveries.length) {
      html += `<div class="empty-state">Nothing yet — propose or run a hypothesis to get started.</div>`;
    }
    data.new_discoveries.forEach((h) => {
      html += `
        <div class="panel">
          <div class="panel-sub mono" style="font-size:11px">#${h.id} · ${h.origin} · ${h.created_at}</div>
          <div class="panel-title" style="font-size:13.5px; font-weight:500;">${escapeHTML(h.statement)}</div>
          <span class="status-pill ${statusPillClass(h.status)}">${h.status.replace(/_/g, " ")}</span>
        </div>`;
    });

    if (data.failed_edges.length) {
      html += `<div class="section-label">Failed edges</div>`;
      data.failed_edges.forEach((h) => {
        html += `
          <div class="panel">
            <div class="panel-title" style="font-size:13.5px; font-weight:500;">${escapeHTML(h.statement)}</div>
            <div class="panel-sub">${escapeHTML(h.rejected_reason || "No reason recorded.")}</div>
          </div>`;
      });
    }

    view.innerHTML = html;
    document.querySelectorAll("[data-kill]").forEach((btn) => {
      btn.addEventListener("click", () => alert(
        "Adversarial 'Try to Kill It' suite isn't implemented yet — it's " +
        "scoped for Phase 2, once strategy versions exist to attack."
      ));
    });
  } catch (e) {
    view.innerHTML = `<div class="empty-state">Couldn't load dashboard: ${escapeHTML(e.message)}</div>`;
  }
}

async function renderResearch() {
  view.innerHTML = `
    <div class="section-label">Run the sample workflow</div>
    <div class="panel">
      <div class="panel-sub">
        Runs the full pipeline for real: fetch data → build hypothesis signal →
        in-sample/out-of-sample backtest → walk-forward → parameter sensitivity →
        transaction-cost sensitivity → Monte Carlo → Edge Score.
        Requires network access to fetch price data (not available in this dev sandbox —
        run this on your deployed instance).
      </div>
      <div style="margin-top:10px;">
        <input id="sample-symbol" value="AAPL" placeholder="Symbol, e.g. AAPL" />
      </div>
      <button class="primary" id="run-sample" style="margin-top:10px; width:100%;">Run sample research</button>
      <div id="sample-result" style="margin-top:10px;"></div>
    </div>

    <div class="section-label">Propose a hypothesis</div>
    <div class="panel">
      <textarea id="hyp-text" rows="3" placeholder="e.g. Stocks with unusually high relative volume after a large decline tend to mean-revert over 5 days."></textarea>
      <button class="primary" id="submit-hyp" style="margin-top:10px; width:100%;">Register hypothesis</button>
      <div id="hyp-result" style="margin-top:10px;"></div>
    </div>
  `;

  document.getElementById("run-sample").addEventListener("click", async () => {
    const symbol = document.getElementById("sample-symbol").value.trim() || "AAPL";
    const out = document.getElementById("sample-result");
    out.innerHTML = `<div class="dim mono" style="font-size:12px;">Running…</div>`;
    try {
      const result = await api("/api/research/run-sample", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbol }),
      });
      if (result.status === "stopped") {
        out.innerHTML = `<div class="panel" style="margin:0;"><div class="panel-sub">${escapeHTML(result.reason)}</div></div>`;
      } else {
        out.innerHTML = `<pre class="mono" style="font-size:11px; white-space:pre-wrap; overflow-x:auto;">${escapeHTML(JSON.stringify(result, null, 2))}</pre>`;
      }
    } catch (e) {
      out.innerHTML = `<div class="panel-sub" style="color:var(--danger)">${escapeHTML(e.message)}</div>`;
    }
  });

  document.getElementById("submit-hyp").addEventListener("click", async () => {
    const statement = document.getElementById("hyp-text").value.trim();
    const out = document.getElementById("hyp-result");
    if (!statement) return;
    try {
      const result = await api("/api/hypotheses", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ statement, origin: "user" }),
      });
      out.innerHTML = result.similar_to
        ? `<div class="panel-sub">Registered as #${result.hypothesis_id}. Resembles hypothesis #${result.similar_to.id} (similarity ${result.similar_to.similarity}, status: ${result.similar_to.status}).</div>`
        : `<div class="panel-sub">Registered as hypothesis #${result.hypothesis_id}.</div>`;
    } catch (e) {
      out.innerHTML = `<div class="panel-sub" style="color:var(--danger)">${escapeHTML(e.message)}</div>`;
    }
  });
}

async function renderDiscoveries() {
  view.innerHTML = `<div class="empty-state">Loading…</div>`;
  try {
    const rows = await api("/api/hypotheses");
    if (!rows.length) {
      view.innerHTML = `<div class="empty-state">No hypotheses tested yet.</div>`;
      return;
    }
    let html = `<div class="section-label">All hypotheses (${rows.length})</div>`;
    rows.forEach((h) => {
      html += `
        <div class="panel">
          <div class="panel-title" style="font-size:13.5px; font-weight:500;">${escapeHTML(h.statement)}</div>
          <span class="status-pill ${statusPillClass(h.status)}">${h.status.replace(/_/g, " ")}</span>
          <div class="panel-sub" style="margin-top:6px;">#${h.id} · origin: ${h.origin} · ${h.created_at}</div>
        </div>`;
    });
    view.innerHTML = html;
  } catch (e) {
    view.innerHTML = `<div class="empty-state">${escapeHTML(e.message)}</div>`;
  }
}

async function renderStrategies() {
  view.innerHTML = `<div class="empty-state">Loading…</div>`;
  try {
    const rows = await api("/api/strategies");
    if (!rows.length) {
      view.innerHTML = `<div class="empty-state">No strategies yet. Strategies are created once a hypothesis survives validation and is formalized (Phase 2).</div>`;
      return;
    }
    let html = `<div class="section-label">Strategy library</div>`;
    rows.forEach((s) => {
      html += `
        <div class="panel">
          <div class="panel-title">${escapeHTML(s.name)}</div>
          <div class="panel-sub">v${s.latest_version || "—"} · ${s.validation_status || "unvalidated"}</div>
        </div>`;
    });
    view.innerHTML = html;
  } catch (e) {
    view.innerHTML = `<div class="empty-state">${escapeHTML(e.message)}</div>`;
  }
}

async function renderSettings() {
  view.innerHTML = `<div class="empty-state">Loading…</div>`;
  try {
    const ai = await api("/api/ai/status");
    view.innerHTML = `
      <div class="section-label">AI provider</div>
      <div class="panel">
        <div class="panel-sub">
          Groq API key: <span class="mono" style="color:${ai.groq_configured ? "var(--signal)" : "var(--danger)"}">
          ${ai.groq_configured ? "configured" : "not configured"}</span>
        </div>
        ${ai.groq_configured ? "" : `<div class="panel-sub" style="margin-top:6px;">Set GROQ_API_KEY in your environment (see .env.example) to enable the Researcher, Skeptic, Strategist, and Reviewer agents.</div>`}
      </div>
      <div class="section-label">About</div>
      <div class="panel">
        <div class="panel-sub">EdgeForge Phase 1 (foundation). Data, feature engineering, backtesting,
        validation, Edge Score, and AI research roles are implemented. Autonomous research loop,
        adversarial "Try to Kill It" suite, and paper-trading execution wiring are scoped for
        Phase 2.</div>
      </div>
    `;
  } catch (e) {
    view.innerHTML = `<div class="empty-state">${escapeHTML(e.message)}</div>`;
  }
}

function escapeHTML(str) {
  const div = document.createElement("div");
  div.textContent = str == null ? "" : String(str);
  return div.innerHTML;
}

function render() {
  if (currentTab === "dashboard") renderDashboard();
  else if (currentTab === "research") renderResearch();
  else if (currentTab === "discoveries") renderDiscoveries();
  else if (currentTab === "strategies") renderStrategies();
  else if (currentTab === "settings") renderSettings();
}

render();

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/sw.js").catch(() => {});
  });
}
