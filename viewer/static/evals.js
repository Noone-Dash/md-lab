"use strict";
const $ = s => document.querySelector(s);

function card(b, r) {
  const st = r ? r.status : "not run";
  const cls = { PASS: "e-pass", FAIL: "e-fail", ERROR: "e-fail", "NO DATA": "e-warn" }[st] || "e-idle";
  return `
  <div class="eval-card ${cls}">
    <div class="e-head">
      <span class="e-badge">${st}</span>
      <b>${b.title}</b>
      ${r ? `<span class="muted e-time">${r.seconds}s</span>` : ""}
    </div>
    <div class="e-why">${b.why}</div>
    <div class="e-nums">
      <div><span>measured</span><b>${r && r.measured !== null && r.measured !== undefined
        ? r.measured + " " + b.unit : "—"}</b></div>
      <div><span>must be in</span><b>${b.expect.min} … ${b.expect.max} ${b.unit}</b></div>
      <div><span>reference</span><b>${b.reference}</b></div>
    </div>
    ${r && r.run_id ? `<a class="jbtn" href="/track/classical?run=${r.run_id}">see the run →</a>` : ""}
    ${r && r.detail ? `<div class="e-detail">${r.detail}</div>` : ""}
  </div>`;
}

async function tick() {
  const d = await fetch("/api/evals").then(r => r.json());
  const last = d.last;
  const byId = {};
  (last ? last.results : []).forEach(r => byId[r.id] = r);

  if (d.running) {
    const p = d.progress;
    $("#summary").innerHTML = `<div class="eval-run">⏳ running benchmark
      ${p ? `${p.i}/${p.n} — <b>${p.id}</b>` : "…"} <span class="muted">(these are real simulations)</span></div>`;
    $("#run-all").disabled = true;
  } else if (last) {
    const all = last.passed === last.total;
    $("#summary").innerHTML = `<div class="eval-score ${all ? "e-pass" : "e-fail"}">
      <span class="es-big">${last.passed}/${last.total}</span>
      <span>${all ? "all physics benchmarks pass" : `${last.failed} failed, ${last.errored} errored`}</span></div>`;
    $("#run-all").disabled = false;
  } else {
    $("#summary").innerHTML = `<div class="muted">Never run. Hit “Run all benchmarks”.</div>`;
    $("#run-all").disabled = false;
  }
  $("#list").innerHTML = d.benchmarks.map(b => card(b, byId[b.id])).join("");
}

$("#run-all").onclick = async () => {
  $("#run-all").disabled = true;
  await fetch("/api/evals/run", { method: "POST", headers: { "Content-Type": "application/json" },
                                  body: JSON.stringify({}) });
  tick();
};
tick();
setInterval(tick, 2000);
