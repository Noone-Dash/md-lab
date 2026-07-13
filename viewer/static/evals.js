"use strict";
const $ = s => document.querySelector(s);

// A verdict is not just pass/fail. PASS* means "the point estimate is inside the window,
// but the run was too short to put a defensible error bar on it" — an unquantified pass.
// INCONCLUSIVE means the confidence interval STRADDLES the acceptance boundary: the run
// genuinely cannot decide the question, which is different from failing it.
const CLS = {
  PASS: "e-pass", "PASS*": "e-warn", FAIL: "e-fail", "FAIL*": "e-fail",
  ERROR: "e-fail", "NO DATA": "e-warn", INCONCLUSIVE: "e-warn", interrupted: "e-warn",
};
const NOTE = {
  "PASS*": "inside the window, but the run is too short for a defensible error bar — " +
           "this is a point estimate, not a verified one",
  "FAIL*": "outside the window, but the run is too short for a defensible error bar",
  INCONCLUSIVE: "the 95% interval straddles the acceptance boundary — this run cannot " +
                "decide the question either way. Run it longer.",
};

const num = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v))
  ? "—" : (+v).toFixed(d);

function measured(b, r) {
  if (!r || r.measured === null || r.measured === undefined) return "—";
  const u = r.uncertainty || {};
  if (r.sem !== null && r.sem !== undefined) {
    return `${r.measured} <span class="pm">±&nbsp;${r.sem}</span> ${b.unit}`;
  }
  // Say WHY there is no bar. An absent error bar on an RDF peak is correct (it is not a
  // time average); an absent one on a mean means we refused to fake it.
  const why = u.note ? "no error bar — run too short" : "not a time average";
  return `${r.measured} ${b.unit} <span class="nobar">(${why})</span>`;
}

function statsRow(r) {
  const u = (r && r.uncertainty) || {};
  if (!u.n) return "";
  const bits = [];
  if (r.ci95) bits.push(`<i>95% CI</i> [${num(r.ci95[0])}, ${num(r.ci95[1])}]`);
  if (u.tau_int_ps) bits.push(`<i>τ<sub>int</sub></i> ${num(u.tau_int_ps, 1)} ps`);
  if (u.n_eff) bits.push(`<i>N<sub>eff</sub></i> ${num(u.n_eff, 0)} <span class="muted">of ${u.n}</span>`);
  if (u.inflation > 1.05) {
    bits.push(`<i>naive bar too small by</i> ${num(u.inflation, 1)}×`);
  }
  if (u.equilibration_frac > 0) {
    bits.push(`<i>discarded</i> ${Math.round(u.equilibration_frac * 100)}% ` +
              `<span class="muted">as transient (detected)</span>`);
  }
  return `<div class="e-stats">${bits.join("<span class='sep'>·</span>")}</div>`;
}

function card(b, r) {
  const st = r ? r.status : "not run";
  const cls = CLS[st] || "e-idle";
  return `
  <div class="eval-card ${cls}">
    <div class="e-head">
      <span class="e-badge">${st}</span>
      <b>${b.title}</b>
      ${r ? `<span class="muted e-time">${r.seconds}s</span>` : ""}
    </div>
    <div class="e-why">${b.why}</div>
    <div class="e-nums">
      <div><span>measured</span><b>${measured(b, r)}</b></div>
      <div><span>must be in</span><b>${b.expect.min} … ${b.expect.max} ${b.unit}</b></div>
      <div><span>reference</span><b>${b.reference}</b></div>
    </div>
    ${statsRow(r)}
    ${NOTE[st] ? `<div class="e-caveat">${NOTE[st]}</div>` : ""}
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
    const caveats = [];
    if (last.unquantified) caveats.push(`${last.unquantified} unquantified (PASS*)`);
    if (last.inconclusive) caveats.push(`${last.inconclusive} inconclusive`);
    $("#summary").innerHTML = `<div class="eval-score ${all ? "e-pass" : "e-fail"}">
      <span class="es-big">${last.passed}/${last.total}</span>
      <span>${all ? "all physics benchmarks pass" : `${last.failed} failed, ${last.errored} errored`}
      ${caveats.length ? `<span class="muted"> — ${caveats.join(", ")}</span>` : ""}</span></div>`;
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
